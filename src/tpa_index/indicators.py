"""005 Part 2: youth-only, program-level indicators per video and phase, every number tagged.
Verified labels first (blind, then review decisions on the containing moment); otherwise the model fallback chosen by
insights.labels_source (the learned final model by default, or zero-shot). Time uses non-overlapping slices; idle and
staff-led time are kept out of youth denominators and reported separately. Never per person."""
import math

import numpy as np
import pandas as pd

from tpa_index.view import UNCERTAIN

STAFF_VERIFIED = "staff-verified"
UNVALIDATED = "model estimate (unvalidated)"
LEARNED = "model estimate (learned)"
HEADLINE = ("performance_share", "solo_moments_minutes", "youth_voice_minutes", "positive_expression_rate")
PHASES = ("early", "mid", "showcase")
LIMITATIONS = [
    "single labeler: verified labels come from one staff member",
    "model actor and category scores are weak, so staff exclusion is mostly reliable only on reviewed clips",
    "review-mode decisions are per moment: one decision verifies every segment in that moment",
    "audio conditions use a fixed 0.5 AST probability, not a dev-tuned threshold",
    "expressions are observable, unvalidated model estimates, never feelings; segments with faces only",
    "person_count includes any visible audience",
    "learned labels come from a model trained on ~206 labels from two videos; on labeled segments they are in-sample",
]


def tag(verified_share: float | None, model_tag: str = UNVALIDATED) -> str:
    """Exact tag strings from contracts/results.md."""
    if verified_share is None:
        return model_tag
    pct = round(100 * verified_share)
    if verified_share >= 1.0 or pct == 100:
        return STAFF_VERIFIED
    if verified_share <= 0.0 or pct == 0:
        return model_tag
    return f"mixed ({pct}% staff-verified)"


def segment_table(view: pd.DataFrame, verified: dict, estimates: pd.DataFrame, labels: dict) -> pd.DataFrame:
    """One row per segment with the label sets the indicators use (verified first) and their time class."""
    fired = estimates[estimates["fired"] == True]  # noqa: E712  (None means signal not available)
    est = fired.groupby("segment_id")["indicator"].apply(set).to_dict()
    rows = []
    for s in view.to_dict("records"):
        v = verified.get(s["segment_id"], {})
        sets = {"activity": v.get("activity", s["model_cleared"]["activity"]),
                "category": v.get("category", s["model_cleared"]["category"]),
                "actor": v.get("actor", s["model_cleared"]["actor"]),
                "indicator": v.get("indicator", est.get(s["segment_id"], set()))}
        flags = {g: g in v for g in sets}
        staff_v = (flags["actor"] or flags["category"]) and "actor_participant" not in sets["actor"] and (
            "actor_staff_facilitation" in sets["actor"] or "instruction" in sets["category"])
        staff_m = not (flags["actor"] or flags["category"]) and "actor_participant" not in sets["actor"] and (
            "actor_staff_facilitation" in sets["actor"] or "instruction" in sets["category"])
        cls = "idle" if s["is_idle"] else "staff_verified" if staff_v else "staff_model" if staff_m else "youth"
        audio = s["audio_ok"]
        rows.append({"segment_id": s["segment_id"], "video_id": s["video_id"], "phase": s["phase"],
                     "start_s": s["start_s"], "slice_s": s["slice_s"], "cls": cls, **sets,
                     **{f"v_{g}": flags[g] for g in sets}, "verified_any": bool(v),
                     "audio_ok": audio, "audio_available": any(x is not None for x in audio.values()),
                     "has_speech": s["has_speech"], "faces": s["faces_detected"], "expr_top": s["expr_top"]})
    return pd.DataFrame(rows)


def _union(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _overlap(intervals, union) -> float:
    total = 0.0
    for a, b in intervals:
        for c, d in union:
            total += max(0.0, min(b, d) - max(a, c))
    return total


def _is_true(v) -> bool:
    return v is True or (isinstance(v, (bool, np.bool_)) and bool(v))


def _rows_for_scope(seg: pd.DataFrame, scope_type: str, scope: str, pipeline: dict, labels: dict,
                    model_tag: str) -> list[dict]:
    from tpa_index.config import model_labels

    out = []
    youth = seg[seg["cls"] == "youth"]
    t_youth = float(youth["slice_s"].sum())

    def add(indicator, value, unit, num, den, base, vshare, tag_value=None, status="ok"):
        if status == "ok" and value is None:
            status = "undefined"
        out.append({"indicator": indicator, "scope_type": scope_type, "scope": scope,
                    "value": None if status != "ok" else value, "unit": unit, "numerator": num, "denominator": den,
                    "base": base, "tag": tag_value or tag(vshare, model_tag),
                    "verified_share": vshare if vshare is not None else 0.0, "status": status})

    def share(mask, groups, base="youth non-idle time", within=None):
        den_rows = youth if within is None else within
        num = float(den_rows.loc[mask(den_rows), "slice_s"].sum()) if len(den_rows) else 0.0
        den = float(den_rows["slice_s"].sum())
        ver = den_rows[[all(r) for r in zip(*(den_rows[f"v_{g}"] for g in groups))]] if len(den_rows) else den_rows
        vshare = float(ver["slice_s"].sum()) / den if den else None
        return (num / den if den else None), num, den, vshare

    perf = lambda d: d["category"].apply(lambda c: "performance" in c)  # noqa: E731
    v, n, d, vs = share(perf, ("category",))
    add("performance_share", v, "share", n, d, "youth non-idle time", vs)
    active = lambda d: [bool(a - {"audience_watching"}) and "instruction" not in c  # noqa: E731
                        for a, c in zip(d["activity"], d["category"])]
    v, n, d, vs = share(active, ("activity", "category"))
    add("active_participation_share", v, "share", n, d, "youth non-idle time", vs)

    # solo moments: contiguous runs of youth segments with solo_on_stage (staff tag, else the rule estimate)
    runs, solo_s, solo_ver = 0, 0.0, 0.0
    for _, g in seg.sort_values(["video_id", "start_s"]).groupby("video_id"):
        prev = False
        for r in g.to_dict("records"):
            on = r["cls"] == "youth" and "solo_on_stage" in r["indicator"]
            runs += on and not prev
            solo_s += r["slice_s"] if on else 0.0
            solo_ver += r["slice_s"] if on and r["v_indicator"] else 0.0
            prev = on
    svs = solo_ver / solo_s if solo_s else (float(youth["v_indicator"].mean()) if len(youth) else None)
    add("solo_moments_count", float(runs), "count", float(runs), None, "contiguous youth runs", svs)
    add("solo_moments_minutes", solo_s / 60, "minutes", solo_s, t_youth, "youth non-idle time", svs)

    # youth voice: performance or spoken word with singing_voice/speech audio, or transcribed words
    avail = youth[youth["audio_available"] | youth["has_speech"].notna()]
    voiced = [("performance" in c or "lyricism_spoken_word" in a) and (
        _is_true(au.get("singing_voice")) or _is_true(au.get("speech")) or _is_true(hs))
        for c, a, au, hs in zip(avail["category"], avail["activity"], avail["audio_ok"], avail["has_speech"])]
    if len(avail):
        vs_voice = float(avail.loc[avail["v_category"] & avail["v_activity"], "slice_s"].sum()) / float(
            avail["slice_s"].sum())
        sec = float(avail.loc[voiced, "slice_s"].sum())
        add("youth_voice_minutes", sec / 60, "minutes", sec, float(avail["slice_s"].sum()),
            "youth non-idle time with audio or transcript", vs_voice)
    else:
        add("youth_voice_minutes", None, "minutes", None, None, "youth non-idle time", None,
            status="signal not available")

    # activity mix and diversity
    acts = [lab["name"] for lab in model_labels(labels, {"activity"})]
    secs = {a: float(youth.loc[youth["activity"].apply(lambda s, a=a: a in s), "slice_s"].sum()) for a in acts}
    vs_act = (float(youth.loc[youth["v_activity"], "slice_s"].sum()) / t_youth) if t_youth else None
    for a in acts:
        add(f"activity_share:{a}", secs[a] / t_youth if t_youth else None, "share", secs[a], t_youth,
            "youth non-idle time", vs_act)
    total = sum(secs.values())
    probs = [s / total for s in secs.values() if s > 0] if total else []
    entropy = (-sum(p * math.log(p) for p in probs) / math.log(len(acts))) if probs and len(acts) > 1 else None
    add("activity_diversity_count", float(sum(s >= 60 for s in secs.values())), "count", None, None,
        "activities with at least 1 minute of youth time", vs_act)
    add("activity_diversity_entropy", entropy, "share", None, None, "normalized entropy of youth activity time",
        vs_act)

    # audience response: applause during performance or within applause_after_s after it, each second once
    after = pipeline["insights"]["applause_after_s"]
    applause_s, perf_s, has_audio = 0.0, 0.0, False
    for _, g in seg.groupby("video_id"):
        if not g["audio_available"].any():
            continue
        has_audio = True
        pg = g[(g["cls"] == "youth") & perf(g)]
        perf_s += float(pg["slice_s"].sum())
        window = _union([(r["start_s"], r["start_s"] + r["slice_s"] + after) for r in pg.to_dict("records")])
        clap = [(r["start_s"], r["start_s"] + r["slice_s"]) for r in g.to_dict("records")
                if r["cls"] != "idle" and _is_true(r["audio_ok"].get("applause"))]
        applause_s += _overlap(clap, window)
    vs_perf = share(perf, ("category",))[3]
    if has_audio:
        add("audience_response_per_perf_min", (applause_s / 60) / (perf_s / 60) if perf_s else None, "per_minute",
            applause_s, perf_s, "applause minutes per youth performance minute", vs_perf)
    else:
        add("audience_response_per_perf_min", None, "per_minute", None, None, "", None, status="signal not available")

    # positive expression: segments with faces only, argmax label; always an unvalidated model estimate
    faces = youth[youth["faces"].notna() & (youth["faces"] >= 1) & youth["expr_top"].notna()]
    if youth["faces"].notna().any():
        pos = float(faces.loc[faces["expr_top"] == "positive", "slice_s"].sum())
        den = float(faces["slice_s"].sum())
        add("positive_expression_rate", pos / den if den else None, "share", pos, den,
            "youth time in segments with faces", 0.0, tag_value=UNVALIDATED)
    else:
        add("positive_expression_rate", None, "share", None, None, "youth time in segments with faces", 0.0,
            tag_value=UNVALIDATED, status="signal not available")

    # original work: staff-tagged only
    ow = youth[youth["v_indicator"] & youth["indicator"].apply(lambda s: "original_work" in s)]
    if len(ow):
        sec = float(ow["slice_s"].sum())
        add("original_work_minutes", sec / 60, "minutes", sec, t_youth, "staff-tagged youth time", 1.0)
    else:
        add("original_work_minutes", None, "minutes", None, None, "staff-tagged youth time", 1.0,
            tag_value=STAFF_VERIFIED, status="not tagged yet")

    # instruction vs participation (instruction time is staff-led context)
    nonidle = seg[seg["cls"] != "idle"]
    inst = float(nonidle.loc[nonidle["category"].apply(lambda c: "instruction" in c), "slice_s"].sum())
    part = float(youth.loc[youth["category"].apply(lambda c: "participation" in c), "slice_s"].sum())
    vs_cat = float(nonidle.loc[nonidle["v_category"], "slice_s"].sum()) / float(nonidle["slice_s"].sum()) \
        if len(nonidle) else None
    add("instruction_participation_ratio", inst / part if part else None, "ratio", inst, part,
        "instruction time / youth participation time", vs_cat)

    # coverage
    if youth["faces"].notna().any():
        face_s = float(youth.loc[youth["faces"].fillna(0) >= 1, "slice_s"].sum())
        add("coverage_faces", face_s / t_youth if t_youth else None, "share", face_s, t_youth,
            "youth non-idle time", 0.0, tag_value=UNVALIDATED)
    else:
        add("coverage_faces", None, "share", None, None, "youth non-idle time", 0.0, tag_value=UNVALIDATED,
            status="signal not available")
    add("coverage_reviewed", float(seg["verified_any"].mean()) if len(seg) else None, "share",
        float(seg["verified_any"].sum()), float(len(seg)), "segments with a staff-verified label", 1.0,
        tag_value=STAFF_VERIFIED)
    return out


def time_breakdown(seg: pd.DataFrame, scopes: list[tuple[str, str, pd.Series]]) -> list[dict]:
    out = []
    for scope_type, scope, mask in scopes:
        g = seg[mask]
        by = g.groupby("cls")["slice_s"].sum()
        out.append({"scope_type": scope_type, "scope": scope, "youth_s": float(by.get("youth", 0.0)),
                    "staff_led_verified_s": float(by.get("staff_verified", 0.0)),
                    "staff_led_model_s": float(by.get("staff_model", 0.0)), "idle_s": float(by.get("idle", 0.0)),
                    "total_s": float(g["slice_s"].sum())})
    return out


def headline(rows: list[dict], phases_present: list[str]) -> list[dict]:
    """Early vs showcase (plus mid if any video has it) for the four headline indicators."""
    phases = ["early"] + (["mid"] if "mid" in phases_present else []) + ["showcase"]
    by = {(r["indicator"], r["scope"]): r for r in rows if r["scope_type"] == "phase"}
    out = []
    for ind in HEADLINE:
        early = by.get((ind, "early"))
        for ph in phases:
            r = by.get((ind, ph))
            if r is None:
                out.append({"indicator": ind, "phase": ph, "value": None, "unit": "", "tag": "", "verified_share": 0.0,
                            "change_vs_early": None, "status": "no videos in this phase"})
                continue
            change = (r["value"] - early["value"]) if ph != "early" and early and early["value"] is not None \
                and r["value"] is not None else None
            out.append({"indicator": ind, "phase": ph, "value": r["value"], "unit": r["unit"], "tag": r["tag"],
                        "verified_share": r["verified_share"], "change_vs_early": change, "status": r["status"]})
    return out


def run_insights(pipeline: dict, labels: dict, root=None) -> dict:
    """Everything for the insights result folder. Raises ConfigError first if any input is missing."""
    from tpa_index.config import ConfigError, config_hash
    from tpa_index.evaluate import eval_frame
    from tpa_index.learned import fit_final, learned_view
    from tpa_index.rules import context, indicator_estimates
    from tpa_index.schema import load_official
    from tpa_index.view import check_inputs, human_labels, segment_view, verified_labels

    check_inputs("insights", pipeline, labels, root)
    ins = pipeline["insights"]
    view, human = segment_view(pipeline, labels, root), human_labels(labels, pipeline, root)
    source = ins["labels_source"]
    if source == "learned":
        labeled, _ = eval_frame(view, human)
        if labeled.empty:
            raise ConfigError("insights.labels_source is learned but there are no blind labels on eval videos; "
                              "label some, or set labels_source: zero-shot")
        model_view, model_tag = learned_view(view, fit_final(labeled, view, labels, pipeline)), LEARNED
    else:
        model_view, model_tag = view, UNVALIDATED
    verified = verified_labels(model_view, labels, pipeline, root)
    estimates = indicator_estimates(model_view, ins["indicator_rules"], context(pipeline))
    seg = segment_table(model_view, verified, estimates, labels)
    sources = set(model_view["phase_source"])
    phase_source = sources.pop() if len(sources) == 1 else "local config (stored runs for unlisted videos)"
    phases_present = sorted(set(seg["phase"]))
    scopes = [("all", "all", pd.Series(True, index=seg.index))]
    scopes += [("phase", ph, seg["phase"] == ph) for ph in PHASES if ph in phases_present]
    scopes += [("video", vid, seg["video_id"] == vid) for vid in sorted(seg["video_id"].unique())]
    rows = []
    for scope_type, scope, mask in scopes:
        rows += _rows_for_scope(seg[mask], scope_type, scope, pipeline, labels, model_tag)
    excluded = sorted(seg.loc[~seg["phase"].isin(PHASES), "video_id"].unique())
    breakdown = time_breakdown(seg, scopes)
    total = sum(b["total_s"] for b in breakdown if b["scope_type"] == "all")
    model_staff = sum(b["staff_led_model_s"] for b in breakdown if b["scope_type"] == "all")
    return {"kind": "insights", "indicators": rows, "headline": headline(rows, phases_present),
            "time_breakdown": breakdown, "excluded_videos": excluded, "labels_source": source,
            "model_tag": model_tag, "phase_source": phase_source, "model_staff_share": model_staff / total if total else 0.0,
            "limitations": LIMITATIONS, "official_runs": {k: v for k, v in load_official().items() if v},
            "config_hash": config_hash(labels, pipeline)}
