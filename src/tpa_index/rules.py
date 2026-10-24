"""005 rule grammar (contracts/config.md): fusion rules and signal-based indicator estimates.
A condition is True, False or None (its signal is missing for the segment: no official run, or a null column)."""
import pandas as pd

UNCERTAIN = "uncertain_mixed"


def _missing(v) -> bool:
    return v is None or (not isinstance(v, (set, dict, list, str)) and pd.isna(v))


def context(pipeline: dict) -> dict:
    return {"primary_like": list(pipeline["insights"]["primary_like"])}


def evaluate_condition(cond: dict, seg: dict, ctx: dict, fired: dict | None = None) -> bool | None:
    (key, value), = cond.items()
    if key in ("all", "any"):
        results = [evaluate_condition(c, seg, ctx, fired) for c in value]
        if key == "all":
            return False if False in results else (None if None in results else True)
        return True if True in results else (None if None in results else False)
    if key == "audio":
        ok = (seg.get("audio_ok") or {}).get(value)
        return None if _missing(ok) else bool(ok)
    if key == "has_speech":
        v = seg.get("has_speech")
        return None if _missing(v) else bool(v) == value
    if key == "clears":
        return any(value in labs for labs in seg["model_cleared"].values())
    if key == "primary_in":
        allowed = ctx["primary_like"] if value == "primary_like" else value
        return seg["model_primary"] in allowed
    if key == "person_count":
        n = seg.get("person_count")
        if _missing(n):
            return None
        (op, ref), = value.items()
        return {"eq": n == ref, "gte": n >= ref, "lte": n <= ref}[op]
    if key == "indicator":
        return (fired or {}).get(value)
    raise ValueError(f"unknown condition {key!r}")  # config validation makes this unreachable


def indicator_estimates(view: pd.DataFrame, rules: dict, ctx: dict) -> pd.DataFrame:
    """segment_id, indicator, fired (True/False/None), rules evaluated in file order."""
    rows = []
    for seg in view.to_dict("records"):
        fired: dict[str, bool | None] = {}
        for name, cond in rules.items():
            fired[name] = evaluate_condition(cond, seg, ctx, fired)
            rows.append({"segment_id": seg["segment_id"], "indicator": name, "fired": fired[name]})
    return pd.DataFrame(rows, columns=["segment_id", "indicator", "fired"])


def apply_fusion(view: pd.DataFrame, fusion: list[dict], ctx: dict) -> tuple[pd.DataFrame, dict]:
    """Drop a cleared visual activity when its require_any is False, then re-pick the primary from what is left
    (uncertain_mixed if nothing). A None condition leaves the call as it was ("not applicable")."""
    counts = {r["label"]: {"fired": 0, "not_applicable": 0} for r in fusion}
    rows = []
    for seg in view.to_dict("records"):
        st = {g: dict(v) for g, v in seg["strengths"].items()}
        for rule in fusion:
            if rule["label"] not in st["activity"]:
                continue
            ok = evaluate_condition({"any": rule["require_any"]}, seg, ctx)
            if ok is None:
                counts[rule["label"]]["not_applicable"] += 1
            elif not ok:
                del st["activity"][rule["label"]]
                counts[rule["label"]]["fired"] += 1
        seg = {**seg, "strengths": st, "model_cleared": {g: set(v) for g, v in st.items()}}
        if not seg["no_call"]:
            seg["model_primary"] = max(st["activity"], key=st["activity"].get) if st["activity"] else UNCERTAIN
            seg["model_strength"] = st["activity"].get(seg["model_primary"], 0.0)
        rows.append(seg)
    return pd.DataFrame(rows, columns=view.columns), counts
