"""005 learned system: logistic regressions trained on our blind labels, using only official per-segment signals
(SigLIP label scores, AST probabilities, transcript/people/expression columns). No embeddings, no frames.
Evaluated by grouped cross-validation over the bootstrap blocks, so no prediction comes from a model that saw its
block; a final model fit on all labels predicts every segment for insights."""
import warnings

import numpy as np
import pandas as pd

from tpa_index.view import GROUPS, UNCERTAIN

TAG = "model estimate (learned)"
HUMAN_COLS = {"activity": "activities", "category": "categories", "actor": "actors"}


def _names(labels, pipeline):
    from tpa_index.config import model_labels

    visual = [lab["name"] for lab in model_labels(labels, set(pipeline["visual_groups"]) |
                                                  {pipeline["calibrate"]["null_group"]})]
    audio = [lab["name"] for lab in model_labels(labels, {"audio"})]
    return visual, audio, list(pipeline["signals"]["expression"]["mapping"]), \
        {g: [lab["name"] for lab in model_labels(labels, {g})] for g in GROUPS}


def _num(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def features(view: pd.DataFrame, labels: dict, pipeline: dict) -> tuple[np.ndarray, list[str]]:
    """One row per view row. A missing signal is 0 with a `<source>_missing` flag set to 1."""
    visual, audio, expr, _ = _names(labels, pipeline)
    cols = ([f"siglip:{n}" for n in visual] + [f"ast:{n}" for n in audio] + [f"expr:{n}" for n in expr]
            + ["has_speech", "transcript_chars", "person_count", "faces_detected",
               "audio_missing", "speech_missing", "people_missing", "faces_missing", "expr_missing"])
    rows = []
    for r in view.to_dict("records"):
        vs, a, e = r["visual_scores"] or {}, r["audio_scores"], r["expr_scores"]
        speech, chars, persons, faces = (_num(r["has_speech"]), _num(r["transcript_chars"]), _num(r["person_count"]),
                                         _num(r["faces_detected"]))
        rows.append([vs.get(n, 0.0) for n in visual] + [(a or {}).get(n, 0.0) for n in audio]
                    + [(e or {}).get(n, 0.0) for n in expr]
                    + [speech or 0.0, chars or 0.0, persons or 0.0, faces or 0.0,
                       float(a is None), float(speech is None), float(persons is None), float(faces is None),
                       float(e is None)])
    return np.asarray(rows, dtype=float).reshape(len(rows), len(cols)), cols


def _model(cfg, balanced=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(
        C=cfg["C"], max_iter=cfg["max_iter"], random_state=cfg["seed"],
        class_weight="balanced" if balanced else None))


def _fit_predict(Xtr, ytr: pd.DataFrame, Xte, names: dict, cfg) -> list[dict]:
    """Model columns (view-like) for the rows of Xte, from models fit on (Xtr, ytr) only."""
    from sklearn.exceptions import ConvergenceWarning

    n = len(Xte)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        y = ytr["primary"].to_numpy(dtype=object)
        if len(set(y)) == 1:
            primary, prob = np.array([y[0]] * n, dtype=object), np.ones(n)
        else:
            m = _model(cfg).fit(Xtr, y)
            p = m.predict_proba(Xte)
            primary, prob = m.classes_[p.argmax(1)], p.max(1)
        probs = {}
        for g in GROUPS:
            probs[g] = {}
            for label in names[g]:
                yl = np.array([label in s for s in ytr[HUMAN_COLS[g]]])
                if yl.all() or not yl.any():
                    probs[g][label] = np.full(n, float(yl.mean()) if len(yl) else 0.0)
                else:
                    probs[g][label] = _model(cfg, balanced=True).fit(Xtr, yl).predict_proba(Xte)[:, 1]
    out = []
    for i in range(n):
        st = {g: {lab: float(probs[g][lab][i]) for lab in names[g] if probs[g][lab][i] >= 0.5} for g in GROUPS}
        out.append({"model_primary": str(primary[i]), "model_strength": float(prob[i]), "strengths": st,
                    "model_cleared": {g: set(st[g]) for g in GROUPS}, "uncertain_flag": primary[i] == UNCERTAIN,
                    "needs_review": False, "no_call": False})
    return out


def cross_validated(df: pd.DataFrame, labels: dict, pipeline: dict) -> tuple[pd.DataFrame, list[dict]]:
    """The evaluation frame with model columns replaced by grouped-CV predictions. Groups are the bootstrap blocks;
    folds = min(learned.folds, number of blocks). Returns (frame, folds) with each fold's train/test block ids."""
    from sklearn.model_selection import GroupKFold

    from tpa_index.evaluate import block_ids

    ins = pipeline["insights"]
    cfg, names = ins["learned"], _names(labels, pipeline)[3]
    X, _ = features(df, labels, pipeline)
    groups = block_ids(df, ins["bootstrap"]["block"])
    k = min(cfg["folds"], len(set(groups)))
    preds: list[dict | None] = [None] * len(df)
    folds = []
    if k < 2:
        raise ValueError("too few label blocks for cross-validation")
    for tr, te in GroupKFold(n_splits=k).split(X, groups=groups):
        folds.append({"train_blocks": set(groups[tr]), "test_blocks": set(groups[te]), "test_rows": te})
        for i, p in zip(te, _fit_predict(X[tr], df.iloc[tr], X[te], names, cfg)):
            preds[i] = p
    out = df.copy()
    for col in ("model_primary", "model_strength", "strengths", "model_cleared", "uncertain_flag", "needs_review",
                "no_call"):
        out[col] = [p[col] for p in preds]
    return out, folds


def fit_final(labeled: pd.DataFrame, view: pd.DataFrame, labels: dict, pipeline: dict) -> pd.DataFrame:
    """Fit on every labeled row; predict every official segment (in-sample on the labeled ones)."""
    cfg, names = pipeline["insights"]["learned"], _names(labels, pipeline)[3]
    Xtr, _ = features(labeled, labels, pipeline)
    Xall, _ = features(view, labels, pipeline)
    preds = _fit_predict(Xtr, labeled, Xall, names, cfg)
    return pd.DataFrame([{"segment_id": sid, "video_id": vid, "primary": p["model_primary"],
                          "primary_prob": round(p["model_strength"], 4),
                          "activities": sorted(p["model_cleared"]["activity"]),
                          "categories": sorted(p["model_cleared"]["category"]),
                          "actors": sorted(p["model_cleared"]["actor"]), "tag": TAG, "_model": p}
                         for sid, vid, p in zip(view["segment_id"], view["video_id"], preds)])


def learned_view(view: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    """The view with every model column replaced by the final learned model's (for insights)."""
    by = dict(zip(final["segment_id"], final["_model"]))
    out = view.copy()
    for col in ("model_primary", "model_strength", "strengths", "model_cleared", "uncertain_flag", "needs_review",
                "no_call"):
        out[col] = [by[sid][col] for sid in view["segment_id"]]
    return out
