"""005 Part 1: model vs blind labels on eval videos only. Per-label P/R/F1 with support, macro-F1, primary accuracy,
confusion, Cohen's kappa, accuracy vs coverage, block-bootstrap 95% CIs, per-video rows, the zero-shot vs
signal-rules ablation, the signal-based indicators table, top-up precision and the failure gallery."""
import warnings

import numpy as np
import pandas as pd

from tpa_index.view import GROUPS, UNCERTAIN

SYSTEMS = ("zero-shot", "zero-shot + signal rules")
LEARNED = "learned (cross-validated on our labels)"
LIMITATIONS = [
    "single labeler: no human-vs-human agreement",
    "small eval set from two videos, most segments from v04; read every number with its 95% CI",
    "intervals are wide: block bootstrap over {blocks} blocks of {block} consecutive labeled segments",
    "model actor and category scores are weak, so staff exclusion is mostly reliable only on reviewed clips",
    "audio conditions use a fixed 0.5 AST probability, not a dev-tuned threshold",
    "review-mode decisions are per moment: one decision verifies every segment in that moment",
    "the learned system is trained on our {n} labels from {videos} videos: its cross-validated row is the fair number; "
    "the final model's predictions on labeled segments are in-sample",
]


# ---- the evaluation set ------------------------------------------------------------------------------------------
def eval_frame(view: pd.DataFrame, human: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Main-sample, non-skipped blind labels on eval-split segments, joined to the view (positional index)."""
    main = human[human["queue"] == "main"]
    labeled = main[~main["skipped"]]
    in_view = labeled["segment_id"].isin(view["segment_id"])
    eval_ids = set(view.loc[view["split"] == "eval", "segment_id"])
    keep = labeled[labeled["segment_id"].isin(eval_ids)].drop(columns=["video_id"])
    df = keep.merge(view, on="segment_id").sort_values(["video_id", "start_s"]).reset_index(drop=True)
    counts = {"labeled_eval_segments": len(df), "skipped": int(main["skipped"].sum()),
              "not_scored": int((~in_view).sum()),
              "dev_labels_ignored": int((in_view & ~labeled["segment_id"].isin(eval_ids)).sum()),
              "per_video": df.groupby("video_id").size().to_dict()}
    return df, counts


def _sets_matrix(sets, names) -> np.ndarray:
    return np.array([[n in s for n in names] for s in sets], dtype=bool).reshape(len(sets), len(names))


# ---- metrics -----------------------------------------------------------------------------------------------------
def _f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.maximum(d, 1), np.nan)


def label_metrics(H: np.ndarray, M: np.ndarray, names: list[str], min_support: int) -> tuple[list[dict], float | None,
                                                                                           list[str]]:
    """Per label counts and P/R/F1 (None when support < min_support), macro-F1 over scorable labels, excluded list."""
    tp, fp, fn = (H & M).sum(0), (~H & M).sum(0), (H & ~M).sum(0)
    rows, excluded, f1s = [], [], []
    for i, name in enumerate(names):
        support = int(tp[i] + fn[i])
        ok = support >= min_support
        p = tp[i] / (tp[i] + fp[i]) if ok and tp[i] + fp[i] else None  # never predicted: undefined, not 0%
        r = tp[i] / support if ok else None
        f = float(_f1(tp[i], fp[i], fn[i])) if ok else None
        rows.append({"label": name, "support": support, "predicted": int(tp[i] + fp[i]), "tp": int(tp[i]),
                     "fp": int(fp[i]), "fn": int(fn[i]), "precision": p, "recall": r, "f1": f,
                     "status": "ok" if ok else "too few to score"})
        (f1s.append(f) if ok else excluded.append(name))
    return rows, (float(np.mean(f1s)) if f1s else None), excluded


def _macro(H, M, scorable_idx):
    def stat(idx):
        if not scorable_idx:
            return None
        h, m = H[idx][:, scorable_idx], M[idx][:, scorable_idx]
        f = _f1((h & m).sum(0), (~h & m).sum(0), (h & ~m).sum(0))
        return float(np.nanmean(f)) if not np.all(np.isnan(f)) else None
    return stat


def kappa(a, b) -> float | None:
    from sklearn.metrics import cohen_kappa_score

    if len(a) < 2 or len(set(a) | set(b)) < 2:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k = cohen_kappa_score(a, b)
    return None if np.isnan(k) else float(k)


def kappa_sentence(k: float | None) -> str:
    if k is None:
        return "kappa is undefined here (only one class present)"
    bands = [(0.0, "worse than chance"), (0.2, "slight agreement beyond chance"), (0.4, "fair agreement beyond chance"),
             (0.6, "moderate agreement beyond chance"), (0.8, "substantial agreement beyond chance")]
    text = next((t for limit, t in bands if k < limit), "near-complete agreement")
    return (f"kappa {k:.2f}: {text}. 0 means no better than guessing from how often each label occurs; "
            f"1 means the model always agrees with the labeler.")


def primary_metrics(hp: np.ndarray, mp: np.ndarray) -> dict:
    real = hp != UNCERTAIN
    confusion = pd.crosstab(pd.Series(hp, name="human"), pd.Series(mp, name="model"))
    return {"accuracy": float((hp == mp).mean()) if len(hp) else None,
            "accuracy_real_activity": float((hp[real] == mp[real]).mean()) if real.any() else None,
            "n_real_activity": int(real.sum()), "kappa": kappa(list(hp), list(mp)),
            "confusion": [{"human_primary": h, "model_primary": m, "count": int(c)}
                          for (h, m), c in confusion.stack().items() if c]}


def category_sides(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    """One category per side: the human's single ticked category ('none' if none; 2+ excluded and counted), the
    model's top cleared category by strength ('none'). Returns (human, model, n_excluded) with excluded rows as ''."""
    human, model, excluded = [], [], 0
    for cats, st in zip(df["categories"], df["strengths"]):
        if len(cats) >= 2:
            excluded += 1
            human.append("")
        else:
            human.append(next(iter(cats)) if cats else "none")
        model.append(max(st["category"], key=st["category"].get) if st["category"] else "none")
    return np.array(human, dtype=object), np.array(model, dtype=object), excluded


def uncertainty_metrics(df: pd.DataFrame) -> dict:
    flagged = (df["uncertain_flag"] | df["needs_review"] | df["no_call"]).to_numpy()
    right = (df["primary"] == df["model_primary"]).to_numpy()
    n = len(df)
    return {"rate_uncertain_mixed": float(df["uncertain_flag"].mean()) if n else None,
            "rate_needs_review": float(df["needs_review"].mean()) if n else None,
            "rate_no_call": float(df["no_call"].mean()) if n else None,
            "coverage": float((~flagged).mean()) if n else None,
            "accuracy_confident": float(right[~flagged].mean()) if (~flagged).any() else None,
            "accuracy_flagged": float(right[flagged].mean()) if flagged.any() else None,
            "n_confident": int((~flagged).sum()), "n_flagged": int(flagged.sum())}


def block_ids(df: pd.DataFrame, block: int) -> np.ndarray:
    """Block number per row (positional): each video's rows in time order, cut into runs of `block`. Shared by the
    bootstrap and the learned system's cross-validation folds."""
    ids = np.zeros(len(df), dtype=int)
    next_id = 0
    for _, g in df.groupby("video_id", sort=True):
        idx = g.sort_values("start_s").index.to_numpy()
        for i in range(0, len(idx), block):
            ids[idx[i:i + block]] = next_id
            next_id += 1
    return ids


def block_bootstrap(df: pd.DataFrame, stat_fn, block: int, resamples: int, seed: int):
    """95% percentile CI. Each video's rows (time order) are cut into blocks of `block`; blocks are drawn with
    replacement until the resample is as large as the original, then trimmed. stat_fn takes positional indices.
    Returns (low, high, n_valid); resamples where the statistic is undefined are skipped."""
    ids = block_ids(df, block)
    order = df.sort_values(["video_id", "start_s"]).index.to_numpy()
    blocks = [order[ids[order] == b] for b in dict.fromkeys(ids[order])]
    n, rng, values = len(df), np.random.default_rng(seed), []
    if not blocks:
        return None, None, 0
    for _ in range(resamples):
        parts, total = [], 0
        while total < n:
            b = blocks[rng.integers(len(blocks))]
            parts.append(b)
            total += len(b)
        v = stat_fn(np.concatenate(parts)[:n])
        if v is not None and not np.isnan(v):
            values.append(v)
    if not values:
        return None, None, 0
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)), len(values)


# ---- systems -----------------------------------------------------------------------------------------------------
def _row(system, scope, group, label, metric, value, ci=(None, None), support=None, n=None, status="ok"):
    ok = status == "ok" and value is not None
    return {"system": system, "scope": scope, "group": group, "label": label, "metric": metric,
            "value": value if ok else None, "ci_low": ci[0] if ok else None, "ci_high": ci[1] if ok else None,
            "support": support, "n_segments": n, "status": status if value is not None or status != "ok" else
            "undefined", "tag": "metric"}


def system_metrics(system: str, df: pd.DataFrame, labels: dict, pipeline: dict) -> dict:
    """Every Part 1 number for one system on the evaluation set (rows for metrics.csv plus the headline)."""
    from tpa_index.config import model_labels

    ins = pipeline["insights"]
    bs = lambda fn: block_bootstrap(df, fn, ins["bootstrap"]["block"], ins["bootstrap"]["resamples"],  # noqa: E731
                                    ins["bootstrap"]["seed"])
    n, rows, head = len(df), [], {"system": system, "n_segments": len(df)}
    human_cols = {"activity": "activities", "category": "categories", "actor": "actors"}
    for g in GROUPS:
        names = [lab["name"] for lab in model_labels(labels, {g})]
        H = _sets_matrix(list(df[human_cols[g]]), names)
        M = _sets_matrix([c[g] for c in df["model_cleared"]], names)
        per, macro, excluded = label_metrics(H, M, names, ins["min_support"])
        for r in per:
            for metric in ("precision", "recall", "f1"):
                rows.append(_row(system, "all", g, r["label"], metric, r[metric], support=r["support"], n=n,
                                 status=r["status"]))
        ci = bs(_macro(H, M, [i for i, r in enumerate(per) if r["status"] == "ok"]))
        rows.append(_row(system, "all", g, "", "macro_f1", macro, ci[:2], n=n))
        head[f"macro_f1_{g}"], head[f"macro_f1_{g}_ci"], head[f"excluded_{g}"] = macro, ci[:2], excluded
        if g == "activity":
            for vid, gv in df.groupby("video_id"):
                idx = gv.index.to_numpy()
                _, vmacro, _ = label_metrics(H[idx], M[idx], names, ins["min_support"])
                rows.append(_row(system, vid, g, "", "macro_f1", vmacro, n=len(idx)))
    hp, mp = df["primary"].to_numpy(dtype=object), df["model_primary"].to_numpy(dtype=object)
    pm = primary_metrics(hp, mp)
    acc_ci = bs(lambda idx: float((hp[idx] == mp[idx]).mean()))
    k_ci = bs(lambda idx: kappa(list(hp[idx]), list(mp[idx])))
    rows += [_row(system, "all", "primary", "", "accuracy", pm["accuracy"], acc_ci[:2], n=n),
             _row(system, "all", "primary", "", "accuracy_real_activity", pm["accuracy_real_activity"],
                  n=pm["n_real_activity"]),
             _row(system, "all", "primary", "", "kappa", pm["kappa"], k_ci[:2], n=n)]
    for vid, gv in df.groupby("video_id"):
        idx = gv.index.to_numpy()
        rows.append(_row(system, vid, "primary", "", "accuracy", float((hp[idx] == mp[idx]).mean()), n=len(idx)))
    hc, mc, excluded = category_sides(df)
    use = np.flatnonzero(hc != "")
    ck = kappa(list(hc[use]), list(mc[use]))
    ck_ci = bs(lambda idx: kappa(list(hc[idx][hc[idx] != ""]), list(mc[idx][hc[idx] != ""])))
    rows.append(_row(system, "all", "category", "", "kappa", ck, ck_ci[:2], n=len(use)))
    unc = uncertainty_metrics(df)
    for key in ("rate_uncertain_mixed", "rate_needs_review", "rate_no_call", "coverage", "accuracy_confident",
                "accuracy_flagged"):
        metric = "rate" if key.startswith("rate_") else key
        label = key.removeprefix("rate_") if key.startswith("rate_") else ""
        rows.append(_row(system, "all", "uncertainty", label, metric, unc[key],
                         n=unc["n_flagged"] if key == "accuracy_flagged" else unc["n_confident"]
                         if key == "accuracy_confident" else n))
    head.update(primary_accuracy=pm["accuracy"], primary_accuracy_ci=acc_ci[:2],
                accuracy_real_activity=pm["accuracy_real_activity"], kappa_primary=pm["kappa"],
                kappa_primary_ci=k_ci[:2], kappa_primary_sentence=kappa_sentence(pm["kappa"]),
                kappa_category=ck, kappa_category_ci=ck_ci[:2], kappa_category_sentence=kappa_sentence(ck),
                category_kappa_excluded=excluded, uncertainty=unc)
    confusion = [{"system": system, **c} for c in pm["confusion"]]
    return {"rows": rows, "headline": head, "confusion": confusion}


def signal_indicator_table(df: pd.DataFrame, estimates: pd.DataFrame, rules: dict, pipeline: dict,
                           system: str = SYSTEMS[0]) -> list[dict]:
    """Signal-based indicator estimates vs the human indicator labels on the evaluation set."""
    ins = pipeline["insights"]
    out = []
    for name in rules:
        fired = estimates[estimates["indicator"] == name].set_index("segment_id")["fired"]
        f = [fired.get(sid) for sid in df["segment_id"]]
        avail = np.array([v is not None and not (isinstance(v, float) and np.isnan(v)) for v in f])
        M = np.array([bool(v) if a else False for v, a in zip(f, avail)])
        H = np.array([name in s for s in df["indicators"]])

        def stat(idx, H=H, M=M, avail=avail):
            idx = idx[avail[idx]]
            tp, fp, fn = (H[idx] & M[idx]).sum(), (~H[idx] & M[idx]).sum(), (H[idx] & ~M[idx]).sum()
            v = _f1(tp, fp, fn)
            return None if np.isnan(v) else float(v)

        a = np.flatnonzero(avail)
        tp, fp, fn = int((H[a] & M[a]).sum()), int((~H[a] & M[a]).sum()), int((H[a] & ~M[a]).sum())
        support, ok = tp + fn, tp + fn >= ins["min_support"]
        lo, hi, _ = block_bootstrap(df, stat, ins["bootstrap"]["block"], ins["bootstrap"]["resamples"],
                                    ins["bootstrap"]["seed"]) if ok else (None, None, 0)
        out.append({"indicator": name, "support": support, "tp": tp, "fp": fp, "fn": fn,
                    "precision": (tp / (tp + fp) if tp + fp else 0.0) if ok else None,
                    "recall": tp / support if ok else None, "f1": stat(np.arange(len(df))) if ok else None,
                    "ci_low": lo, "ci_high": hi, "not_available": int((~avail).sum()),
                    "status": "ok" if ok else "too few to score",
                    "tag": "model estimate (learned)" if system == LEARNED else "model estimate (unvalidated)",
                    "system": system})
    return out


def topup_precision(view: pd.DataFrame, human: pd.DataFrame, labels: dict) -> list[dict]:
    """Top-up queue only, precision only (never pooled with the main sample)."""
    from tpa_index.config import model_labels

    top = human[(human["queue"] == "topup") & ~human["skipped"]].drop(columns=["video_id"]).merge(
        view, on="segment_id")
    out = []
    for name in [lab["name"] for lab in model_labels(labels, {"activity"})]:
        pred = top[[name in c["activity"] for c in top["model_cleared"]]]
        if len(pred):
            tp = int(sum(name in s for s in pred["activities"]))
            out.append({"label": name, "predicted_positive": len(pred), "tp": tp, "precision": tp / len(pred)})
    return out


def failure_gallery(df: pd.DataFrame, n: int) -> list[dict]:
    """The n wrong primary calls the model was most confident about. IDs and labels only."""
    wrong = df[(df["primary"] != df["model_primary"]) & (df["model_primary"] != UNCERTAIN)]
    wrong = wrong.sort_values(["model_strength", "segment_id"], ascending=[False, True]).head(n)
    return [{"rank": i + 1, "segment_id": r["segment_id"], "video_id": r["video_id"],
             "model_primary": r["model_primary"], "human_primary": r["primary"],
             "model_strength": round(float(r["model_strength"]), 4)} for i, r in enumerate(wrong.to_dict("records"))]


def run_evaluate(pipeline: dict, labels: dict, root=None) -> dict:
    """Everything for the evaluate result folder. Raises ConfigError first if any input is missing."""
    from tpa_index.config import config_hash
    from tpa_index.rules import apply_fusion, context, indicator_estimates
    from tpa_index.schema import load_official, read_reviews
    from tpa_index.view import check_inputs, human_labels, segment_view

    check_inputs("evaluate", pipeline, labels, root)
    ins, ctx = pipeline["insights"], context(pipeline)
    view, human = segment_view(pipeline, labels, root), human_labels(labels, pipeline, root)
    fused, fire = apply_fusion(view, ins["fusion"], ctx)
    rows, confusion, heads, eval_set = [], [], [], None
    for system, v in zip(SYSTEMS, (view, fused)):
        df, counts = eval_frame(v, human)
        eval_set = eval_set or counts
        res = system_metrics(system, df, labels, pipeline)
        rows += res["rows"]
        confusion += res["confusion"]
        heads.append(res["headline"])
        if system == SYSTEMS[0]:
            base_df = df
    from tpa_index.learned import cross_validated, fit_final

    learned_df, _ = cross_validated(base_df, labels, pipeline)
    res = system_metrics(LEARNED, learned_df, labels, pipeline)
    rows += res["rows"]
    confusion += res["confusion"]
    heads.append(res["headline"])
    estimates = indicator_estimates(view, ins["indicator_rules"], ctx)
    signal = signal_indicator_table(base_df, estimates, ins["indicator_rules"], pipeline)
    signal += signal_indicator_table(learned_df, indicator_estimates(learned_df, ins["indicator_rules"], ctx),
                                     ins["indicator_rules"], pipeline, system=LEARNED)
    final = fit_final(base_df, view, labels, pipeline)
    ablation = [{"system": h["system"], "n_segments": h["n_segments"],
                 "primary_accuracy": h["primary_accuracy"], "primary_accuracy_ci_low": h["primary_accuracy_ci"][0],
                 "primary_accuracy_ci_high": h["primary_accuracy_ci"][1],
                 **{f"macro_f1_{g}{s}": (h[f"macro_f1_{g}"] if s == "" else h[f"macro_f1_{g}_ci"][s == "_ci_high"])
                    for g in GROUPS for s in ("", "_ci_low", "_ci_high")},
                 "kappa_primary": h["kappa_primary"], "kappa_category": h["kappa_category"],
                 "rules_fired": "" if h["system"] != SYSTEMS[1] else ";".join(
                     f"{k}:{v['fired']}" for k, v in fire.items()),
                 "rules_not_applicable": "" if h["system"] != SYSTEMS[1] else ";".join(
                     f"{k}:{v['not_applicable']}" for k, v in fire.items())} for h in heads]
    eval_ids = set(base_df["segment_id"])
    versions = sorted({r["guideline_version"] for r in read_reviews(root).to_pylist()
                       if r["mode"] == "blind" and r["target_id"] in eval_ids})
    official = load_official()
    return {"kind": "evaluate", "metrics": rows, "confusion": confusion, "signal_indicators": signal,
            "ablation": ablation, "topup": topup_precision(view, human, labels),
            "failure_gallery": failure_gallery(base_df, ins["failure_gallery"]), "headline": heads,
            "learned_predictions": [{k: (";".join(v) if isinstance(v, list) else v) for k, v in r.items()
                                     if k != "_model"} for r in final.to_dict("records")],
            "eval_set": {**eval_set, "category_kappa_excluded": heads[0]["category_kappa_excluded"]},
            "limitations": [line.format(n=len(base_df), videos=base_df["video_id"].nunique(),
                                        blocks=len(set(block_ids(base_df, ins["bootstrap"]["block"]))),
                                        block=ins["bootstrap"]["block"]) for line in LIMITATIONS],
            "guideline_versions": versions,
            "official_runs": {k: v for k, v in official.items() if v}, "config_hash": config_hash(labels, pipeline)}
