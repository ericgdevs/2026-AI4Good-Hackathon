"""005 result folders (contracts/results.md): OUTPUT_DIR/results/{kind}-{ts}-{hex}/, files written first,
manifest.json last (its presence marks the folder complete). Aggregates and opaque IDs only."""
import csv
import hashlib
import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
COLUMNS = {
    "metrics.csv": ["system", "scope", "group", "label", "metric", "value", "ci_low", "ci_high", "support",
                    "n_segments", "status", "tag"],
    "confusion.csv": ["system", "human_primary", "model_primary", "count"],
    "signal_indicators.csv": ["indicator", "support", "tp", "fp", "fn", "precision", "recall", "f1", "ci_low",
                              "ci_high", "not_available", "status", "tag", "system"],
    "ablation.csv": ["system", "n_segments", "primary_accuracy", "primary_accuracy_ci_low", "primary_accuracy_ci_high",
                     "macro_f1_activity", "macro_f1_activity_ci_low", "macro_f1_activity_ci_high",
                     "macro_f1_category", "macro_f1_category_ci_low", "macro_f1_category_ci_high",
                     "macro_f1_actor", "macro_f1_actor_ci_low", "macro_f1_actor_ci_high", "kappa_primary",
                     "kappa_category", "rules_fired", "rules_not_applicable"],
    "topup.csv": ["label", "predicted_positive", "tp", "precision"],
    "failure_gallery.csv": ["rank", "segment_id", "video_id", "model_primary", "human_primary", "model_strength"],
    "learned_predictions.csv": ["segment_id", "video_id", "primary", "primary_prob", "activities", "categories",
                                "actors", "tag"],
}
TABLES = {"metrics.csv": "metrics", "confusion.csv": "confusion", "signal_indicators.csv": "signal_indicators",
          "ablation.csv": "ablation", "topup.csv": "topup", "failure_gallery.csv": "failure_gallery",
          "learned_predictions.csv": "learned_predictions"}


def results_root(root=None) -> Path:
    from tpa_index.paths import output_dir

    return Path(root or output_dir()) / "results"


def new_result_dir(kind: str, root=None) -> Path:
    path = results_root(root) / f"{kind}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{secrets.token_hex(2)}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def latest_result(kind: str, root=None) -> Path | None:
    """Newest complete folder of this kind (has manifest.json); incomplete folders are ignored."""
    base = results_root(root)
    done = sorted(p for p in base.glob(f"{kind}-*") if (p / "manifest.json").exists()) if base.exists() else []
    return done[-1] if done else None


def _fmt(v):
    if v is None or (isinstance(v, float) and v != v):
        return ""
    return round(v, 6) if isinstance(v, float) else v


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: _fmt(r.get(c)) for c in columns})


def _json(obj) -> str:
    def default(o):
        if isinstance(o, (set, frozenset)):
            return sorted(o)
        if hasattr(o, "item"):
            return o.item()
        return str(o)
    return json.dumps(obj, indent=2, default=default, sort_keys=False)


def _ci(value, ci, pct=True) -> str:
    if value is None:
        return "undefined"
    f = (lambda x: f"{100 * x:.0f}%") if pct else (lambda x: f"{x:.2f}")
    lo, hi = ci or (None, None)
    return f"{f(value)} (95% CI {f(lo)}–{f(hi)}" if lo is not None else f"{f(value)} (95% CI undefined"


def evaluate_summary(p: dict) -> str:
    es, lines = p["eval_set"], ["# Evaluation: model vs blind labels (eval videos only)", ""]
    per_video = ", ".join(f"{v}: {n}" for v, n in sorted(es["per_video"].items()))
    lines += [f"Labeled eval segments: {es['labeled_eval_segments']} ({per_video}); skipped {es['skipped']}, "
              f"not scored {es['not_scored']}, dev labels ignored {es['dev_labels_ignored']}.", ""]
    for h in p["headline"]:
        n = h["n_segments"]
        lines += [f"## {h['system']}", "",
                  f"- Primary activity accuracy: {_ci(h['primary_accuracy'], h['primary_accuracy_ci'])}, n={n})",
                  *[f"- Macro-F1 {g}: {_ci(h[f'macro_f1_{g}'], h[f'macro_f1_{g}_ci'])}, n={n})"
                    for g in ("activity", "category", "actor")],
                  f"- Kappa, primary activity: {_ci(h['kappa_primary'], h['kappa_primary_ci'], pct=False)}, n={n})."
                  f" {h['kappa_primary_sentence']}",
                  f"- Kappa, category: {_ci(h['kappa_category'], h['kappa_category_ci'], pct=False)}, "
                  f"n={n - h['category_kappa_excluded']}). {h['kappa_category_sentence']}", ""]
    lines += ["## Signal-based indicators", "",
              "| system | indicator | support | precision | recall | F1 (95% CI) | not available |",
              "|---|---|---|---|---|---|---|"]
    for r in p["signal_indicators"]:
        f1 = "too few to score" if r["status"] != "ok" else _ci(r["f1"], (r["ci_low"], r["ci_high"])) + ")"
        prec = "" if r["precision"] is None else f"{100 * r['precision']:.0f}%"
        rec = "" if r["recall"] is None else f"{100 * r['recall']:.0f}%"
        lines.append(f"| {r['system']} | {r['indicator']} | {r['support']} | {prec} | {rec} | {f1} | "
                     f"{r['not_available']} |")
    lines += ["", "## Limitations", "", *[f"- {x}" for x in p["limitations"]], ""]
    return "\n".join(lines)


def write_result(kind: str, payload: dict, root=None) -> Path:
    """Write every file of the result folder, then manifest.json last (atomic)."""
    from tpa_index.runs import git_commit

    out = new_result_dir(kind, root)
    files = []
    if kind == "evaluate":
        for name, key in TABLES.items():
            _write_csv(out / name, COLUMNS[name], payload[key])
            files.append(name)
        (out / "metrics.json").write_text(_json({k: payload[k] for k in (
            "metrics", "headline", "eval_set", "limitations", "guideline_versions")}))
        (out / "summary.md").write_text(evaluate_summary(payload))
        files += ["metrics.json", "summary.md"]
    else:
        raise ValueError(f"unknown result kind {kind!r}")
    for chart in payload.get("charts", {}):
        files.append(chart)
    manifest = {"schema_version": SCHEMA_VERSION, "kind": kind,
                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "git_commit": git_commit(),
                "config_hash": payload["config_hash"], "guideline_versions": payload.get("guideline_versions", []),
                "official_runs": payload["official_runs"],
                "files": {f: hashlib.sha256((out / f).read_bytes()).hexdigest() for f in files},
                "limitations": payload["limitations"]}
    tmp = out / "manifest.json.tmp"
    tmp.write_text(_json(manifest))
    os.replace(tmp, out / "manifest.json")
    return out
