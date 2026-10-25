"""Official annotation export (case format, contracts/export-format.md): one record per official moment with its
current verified state, as JSON and CSV. Also the Devpost sample, built from mock data only."""
import csv
import io
import json
import os
import secrets
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tpa_index import review
from tpa_index.review import INDICATOR, OBSERVATION, PRIMARY, SECONDARY

FIELDS = ("video_id", "start_s", "end_s", "primary_label", "secondary_labels", "indicators", "contextual_observation",
          "confidence", "source", "review_decision", "guideline_version", "reviewer_note", "review_date")
DECISIONS = {"accept": "accepted", "edit": "edited", "reject": "rejected", "defer": "deferred"}
LIST_SEP = ";"
SAMPLE_SIZE = 20


def _utc_date(ts) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%d")


def _order(r) -> tuple:
    return r["created_at"], r["review_id"]


def _labels_for(role: str, ai: list[str], latest: dict) -> list[str]:
    """AI labels of one role minus rejected, edits replaced, then surviving staff adds; de-duplicated in order."""
    out = []
    for lab in ai or []:
        r = latest.get(lab)
        if r and r["action"] == "reject":
            continue
        out.append(r["new_label"] if r and r["action"] == "edit" else lab)
    out += [r["label"] for r in sorted(latest.values(), key=_order) if r["action"] == "add" and r["label_role"] == role]
    return list(dict.fromkeys(out))


def build_records(moments, rows, labels: dict, verified_only: bool = False) -> list[dict]:
    """Pure: moments + review log rows -> export records (spec FR-002 to FR-010). Only mode=review rows on
    moments count; the latest row per (moment, label) is that label's state."""
    rows = rows.to_pylist() if hasattr(rows, "to_pylist") else list(rows)
    rows = [r for r in rows if r["mode"] == "review" and r["target_type"] == "moment"]
    state = review.latest_state(rows, "review")
    latest_by: dict[str, dict] = {}
    for (mid, lab), r in state.items():
        latest_by.setdefault(mid, {})[lab] = r
    obs_by: dict[str, list[dict]] = {}
    for r in sorted(rows, key=_order):
        if r["label_role"] == OBSERVATION:
            obs_by.setdefault(r["target_id"], []).append(r)  # every observation, even a repeated type, as stored

    moments = moments.to_pylist() if hasattr(moments, "to_pylist") else list(moments)
    out = []
    for m in sorted(moments, key=lambda m: (m["video_id"], m["start_s"], m["moment_id"])):
        latest = {lab: r for lab, r in latest_by.get(m["moment_id"], {}).items() if r["label_role"] != OBSERVATION}
        p = latest.get(m["primary_label"])
        decision = DECISIONS[p["action"]] if p and p["action"] in DECISIONS else "pending"
        if verified_only and decision == "rejected":
            continue
        observations = obs_by.get(m["moment_id"], [])
        staff_added = any(r["action"] == "add" and r["label_role"] in (SECONDARY, INDICATOR) for r in latest.values())
        # a rejected secondary/indicator is not the record's story; a rejected primary is its decision
        noted = [r for r in [*latest.values(), *observations] if r["action"] != "reject" or r is p]
        last = max(noted, key=_order) if noted else None
        out.append({
            "video_id": m["video_id"],
            "start_s": round(float(m["start_s"]), 2),
            "end_s": round(float(m["end_s"]), 2),
            "primary_label": p["new_label"] if decision == "edited" else m["primary_label"],
            "secondary_labels": _labels_for(SECONDARY, m["secondary_labels"], latest),
            "indicators": _labels_for(INDICATOR, m["indicators"], latest),
            "contextual_observation": "; ".join(f"{r['label']}: {r['note']}" if r["note"] else r["label"]
                                                for r in observations),
            "confidence": round(float(m["confidence"]), 3),
            "source": "staff" if decision == "edited" or staff_added else "ai",
            "review_decision": decision,
            "guideline_version": p["guideline_version"] if decision != "pending" else labels["guideline_version"],
            "reviewer_note": (last["note"] or "") if last else "",
            "review_date": _utc_date(last["created_at"]) if last else "",
        })
    return out


def _json_text(records) -> str:
    return json.dumps([{k: r[k] for k in FIELDS} for r in records], ensure_ascii=False, indent=2) + "\n"


def _csv_text(records) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(FIELDS)
    for r in records:
        w.writerow([LIST_SEP.join(r[k]) if isinstance(r[k], list) else r[k] for k in FIELDS])
    return buf.getvalue()


def _replace(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    os.replace(tmp, path)


def write_export(records, out_dir, guideline: str) -> tuple[Path, Path]:
    """Timestamped JSON/CSV pair (never overwritten) plus annotations-latest.* (replaced atomically)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_text, csv_text = _json_text(records), _csv_text(records)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    while True:
        base = f"annotations-{stamp}-{secrets.token_hex(2)}-g{guideline}"
        json_path, csv_path = out_dir / f"{base}.json", out_dir / f"{base}.csv"
        try:
            with open(json_path, "x", encoding="utf-8", newline="") as f:
                f.write(json_text)
            break
        except FileExistsError:
            continue  # same second and same hex: pick another
    with open(csv_path, "x", encoding="utf-8", newline="") as f:
        f.write(csv_text)
    _replace(out_dir / "annotations-latest.json", json_text)
    _replace(out_dir / "annotations-latest.csv", csv_text)
    return json_path, csv_path


def export_official(root=None, verified_only: bool = False) -> tuple[dict[str, int], tuple[Path, Path]]:
    """Official moments + the local review log -> {OUTPUT_DIR}/exports/. Returns per-decision counts and paths."""
    from collections import Counter

    from tpa_index.config import ConfigError, load_labels
    from tpa_index.paths import output_dir
    from tpa_index.schema import read_official, read_reviews

    moments = read_official("moments", root=root)
    if not moments.num_rows:
        raise ConfigError("no official moments: list a moments or predict run in config/official_runs.yaml")
    labels = load_labels()
    records = build_records(moments, read_reviews(root), labels, verified_only)
    paths = write_export(records, Path(root or output_dir()) / "exports", labels["guideline_version"])
    return dict(Counter(r["review_decision"] for r in records)), paths


# ---- Devpost sample (mock data only) ------------------------------------------------------------------------------
def _other_in_group(label: str, labels: dict) -> str:
    group = next(lab["group"] for lab in labels["labels"] if lab["name"] == label)
    return next(lab["name"] for lab in labels["labels"]
                if lab["group"] == group and lab["name"] != label and lab.get("source", "model") == "model")


def sample_reviews(moments: list[dict], labels: dict) -> list[dict]:
    """Scripted, deterministic review rows over the sample moments: every decision value, a rejected, an edited and a
    deferred secondary, one staff-added label, one observation, one label reviewed twice."""
    from tpa_index.config import observation_types, load_pipeline
    from tpa_index.mock import BASE_TIME

    pipeline = load_pipeline(labels=labels)
    staff_label = next(lab["name"] for lab in labels["labels"]
                       if lab["group"] in pipeline["app"]["indicator_groups"])
    obs_type = observation_types(labels)[0]
    out: list[dict] = []

    def add(m, make, note=None):
        r = make(m)
        n = len(out)
        r.update(review_id=f"sample-{n:03d}", created_at=BASE_TIME + timedelta(minutes=n),
                 reviewer="mock-staff-a" if n % 2 == 0 else "mock-staff-b")
        if note:
            r["note"] = note
        out.append(r)

    def primary(action, new_label=None):
        return lambda m: review.review_row(m, m["primary_label"], PRIMARY, action, "x", labels, new_label=new_label)

    def secondary(action, new_label=None):
        return lambda m: review.review_row(m, m["secondary_labels"][0], SECONDARY, action, "x", labels,
                                           new_label=new_label)

    script = {
        1: [("accept", None)], 2: [("edit", None)], 3: [("reject", "note:not this activity")],
        4: [("defer", "note:unclear from this angle")], 5: [("accept", None), ("reject", None)],
        6: [("accept", None), ("sec_reject", None)], 7: [("accept", None), ("sec_edit", None)],
        8: [("defer", None), ("sec_defer", None)], 9: [("accept", None), ("staff_add", None)],
        10: [("accept", None), ("observation", None)], 12: [("accept", None)], 13: [("edit", None)],
        14: [("reject", None)], 15: [("defer", None)], 17: [("accept", "note:clear example")],
    }
    for i, m in enumerate(moments):
        for step, extra in script.get(i, []):
            note = extra[5:] if extra and extra.startswith("note:") else None
            secs = m["secondary_labels"] or []
            if step in ("accept", "reject", "defer"):
                add(m, primary(step), note)
            elif step == "edit":
                add(m, primary("edit", _other_in_group(m["primary_label"], labels)), "relabelled by staff")
            elif step == "sec_reject" and secs:
                add(m, secondary("reject"))
            elif step == "sec_edit" and secs:
                add(m, secondary("edit", _other_in_group(secs[0], labels)))
            elif step == "sec_defer" and secs:
                add(m, secondary("defer"))
            elif step == "staff_add":
                add(m, lambda m: review.review_row(m, staff_label, INDICATOR, "add", "x", labels), "added by staff")
            elif step == "observation":
                add(m, lambda m: review.observation_row(m, obs_type, "x", labels), "framing cuts off the left side")
    return out


def write_sample(dest=None) -> tuple[Path, Path]:
    """docs/samples/annotation_export_sample.{json,csv} from a temporary mock dataset. Never reads OUTPUT_DIR or the
    review log."""
    from tpa_index.config import load_labels
    from tpa_index.mock import generate
    from tpa_index.paths import repo_root
    from tpa_index.schema import read_table

    labels = load_labels()
    with tempfile.TemporaryDirectory() as tmp:
        generate(out_root=tmp, seed=0)
        moments = sorted(read_table("moments", tmp).to_pylist(), key=lambda m: (m["video_id"], m["start_s"]))
    if len(moments) < SAMPLE_SIZE:
        raise ValueError(f"mock data has only {len(moments)} moments; the sample needs {SAMPLE_SIZE}")
    moments = moments[:SAMPLE_SIZE]
    records = build_records(moments, sample_reviews(moments, labels), labels)
    dest = Path(dest or repo_root() / "docs/samples")
    dest.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = dest / "annotation_export_sample.json", dest / "annotation_export_sample.csv"
    _replace(json_path, _json_text(records))
    _replace(csv_path, _csv_text(records))
    return json_path, csv_path
