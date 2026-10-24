"""Score segments against labels and derive per-segment results (primary, uncertain_mixed, idle)."""
import numpy as np


def segment_labels(scores: dict[str, float], thresholds: dict[str, float],
                   activity: list[str]) -> tuple[str | None, list[str]]:
    """Primary = best activity at or above its own threshold; secondary = all other labels at or above theirs."""
    cleared = [lab for lab, s in scores.items() if s >= thresholds[lab]]
    primary = max((lab for lab in cleared if lab in activity), key=lambda lab: scores[lab], default=None)
    secondary = sorted((lab for lab in cleared if lab != primary), key=lambda lab: -scores[lab])
    return primary, secondary


def uncertain_mixed(activity_scores: dict[str, float], thresholds: dict[str, float], margin: float) -> bool:
    """No activity clears its threshold, or the best one is within `margin` of the next-highest activity."""
    cleared = [lab for lab, s in activity_scores.items() if s >= thresholds[lab]]
    if not cleared:
        return True
    top = max(cleared, key=lambda lab: activity_scores[lab])
    runner_up = max((s for lab, s in activity_scores.items() if lab != top), default=0.0)
    return activity_scores[top] - runner_up < margin


def visual_change(frames: np.ndarray) -> float | None:
    """Mean |difference| between consecutive frames (grayscale, 64x64 block means, 0-1); None if < 2 frames."""
    if len(frames) < 2:
        return None
    gray = frames.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32) / 255.0
    k = gray.shape[1] // 64
    if k >= 1:
        gray = gray[:, :k * 64, :k * 64].reshape(len(gray), 64, k, 64, k).mean(axis=(2, 4))
    return float(np.abs(np.diff(gray, axis=0)).mean())


def is_idle(idle_score: float, change: float | None, idle_cfg: dict) -> bool:
    """Idle only when the idle prompts match AND the picture barely changes (both required)."""
    return change is not None and idle_score >= idle_cfg["score_threshold"] and change < idle_cfg["change_threshold"]


def rules_rows(segments: list[dict], predictions: list[dict], pipeline: dict, labels: dict, model: str,
               run_id: str) -> list[dict]:
    """One uncertain_mixed row (model "rules") per segment, from `model`'s activity scores."""
    from tpa_index.config import model_labels, threshold

    activity = {lab["name"] for lab in model_labels(labels, {"activity"})}
    scores: dict[str, dict[str, float]] = {}
    for p in predictions:
        if p["model"] == model and p["label"] in activity:
            scores.setdefault(p["segment_id"], {})[p["label"]] = p["score"]
    out = []
    for seg in segments:
        act = scores.get(seg["segment_id"], {})
        thresholds = {lab: threshold(pipeline, model, lab) for lab in act}
        flag = uncertain_mixed(act, thresholds, pipeline["uncertain_margin"])
        out.append({"segment_id": seg["segment_id"], "label": "uncertain_mixed", "label_group": "status",
                    "score": 1.0 if flag else 0.0, "model": "rules", "run_id": run_id})
    return out


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)


def prompt_embedding(embedder, prompts: list[str]) -> np.ndarray:
    """Ensemble: mean of the normalized prompt embeddings, renormalized."""
    return _unit(embedder.embed_texts([p.lower() for p in prompts]).mean(axis=0))


def label_embeddings(embedder, labels: list[dict]) -> dict[str, np.ndarray]:
    """One ensembled vector per model-scored label, from a single text batch."""
    labels = [lab for lab in labels if lab.get("source", "model") == "model"]
    prompts = [p.lower() for lab in labels for p in lab["prompts"]]
    embeds = embedder.embed_texts(prompts)
    out, i = {}, 0
    for lab in labels:
        n = len(lab["prompts"])
        out[lab["name"]] = _unit(embeds[i:i + n].mean(axis=0))
        i += n
    return out


def window_embedding(frame_embeds: np.ndarray) -> np.ndarray:
    return _unit(frame_embeds.mean(axis=0))


def sigmoid_scores(window_embeds: np.ndarray, label_matrix: np.ndarray, scale: float, bias: float) -> np.ndarray:
    """SigLIP-style independent 0-1 scores: sigmoid(cos * scale + bias), in float32."""
    logits = window_embeds.astype(np.float32) @ label_matrix.astype(np.float32).T * np.float32(scale) + np.float32(bias)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def predict_video(video: dict, pipeline: dict, labels: dict, embedder, run_id: str, device: str = "cpu",
                  motion=None) -> dict[str, list[dict]]:
    """Score one video. Returns {"videos", "segments", "predictions", "moments"} rows; writes nothing."""
    from tpa_index.config import effective_settings, idle_threshold, model_labels
    from tpa_index.moments import merge_moments
    from tpa_index.paths import data_dir
    from tpa_index.segment import probe, segment_video

    info = probe(data_dir() / video["filename"])
    video_row = {"video_id": video["video_id"], "filename": video["filename"], "duration_s": info["duration_s"],
                 "fps": info["fps"], "orientation": info["orientation"], "kind": video["kind"],
                 "phase": video["phase"], "cohort": video.get("cohort"), "split": video["split"]}
    settings = effective_settings(pipeline, video)
    scored = model_labels(labels, set(pipeline["visual_groups"]))
    names = [lab["name"] for lab in scored]
    group_of = {lab["name"]: lab["group"] for lab in scored}
    emb = label_embeddings(embedder, scored)
    label_matrix = np.stack([emb[n] for n in names])
    idle_cfg = {**pipeline["idle"], "score_threshold": idle_threshold(pipeline, embedder.name)}
    idle_vec = prompt_embedding(embedder, idle_cfg["prompts"])[None, :]  # scored for is_idle only; no rows
    batch = pipeline["models"]["batch_windows"][device]

    segments, predictions, buffer = [], [], []

    def flush():
        frames = np.concatenate([w.frames for w in buffer])
        per_window = buffer[0].frames.shape[0]
        embeds = embedder.embed_images(frames).reshape(len(buffer), per_window, -1)
        windows = np.stack([window_embedding(e) for e in embeds])
        scores = sigmoid_scores(windows, label_matrix, embedder.logit_scale, embedder.logit_bias)
        idle_scores = sigmoid_scores(windows, idle_vec, embedder.logit_scale, embedder.logit_bias)[:, 0]
        for w, row_scores, idle_score in zip(buffer, scores, idle_scores):
            seg = dict(w.row)
            change = visual_change(w.frames)
            seg["idle_score"], seg["visual_change"] = float(idle_score), change
            seg["is_idle"] = is_idle(float(idle_score), change, idle_cfg)
            segments.append(seg)
            predictions.extend({"segment_id": seg["segment_id"], "label": n, "label_group": group_of[n],
                                "score": float(s), "model": embedder.name, "run_id": run_id}
                               for n, s in zip(names, row_scores))
        buffer.clear()

    for window in segment_video(video, settings, with_audio=False):
        buffer.append(window)
        if len(buffer) >= batch:
            flush()
    if buffer:
        flush()
    predictions += rules_rows(segments, predictions, pipeline, labels, embedder.name, run_id)
    if motion is not None:  # second pass: the motion model needs its own frame count
        predictions += _motion_rows(video, settings, motion, batch, run_id)
    moments = merge_moments(segments, predictions, pipeline, run_id, window_s=settings["window_s"],
                            model=embedder.name)
    return {"videos": [video_row], "segments": segments, "predictions": predictions, "moments": moments}


def _motion_rows(video: dict, settings: dict, motion, batch: int, run_id: str) -> list[dict]:
    from tpa_index.segment import segment_video

    rows, buffer = [], []

    def flush():
        for w, scores in zip(buffer, motion.score([w.frames for w in buffer])):
            rows.extend({"segment_id": w.row["segment_id"], "label": n, "label_group": "activity",
                         "score": float(sc), "model": motion.name, "run_id": run_id}
                        for n, sc in zip(motion.labels, scores))
        buffer.clear()

    for window in segment_video(video, {**settings, "frames_per_window": motion.frames_per_window}, with_audio=False):
        buffer.append(window)
        if len(buffer) >= batch:
            flush()
    if buffer:
        flush()
    return rows
