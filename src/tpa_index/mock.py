"""Seeded fake rows for all six tables, so downstream features run without footage."""
from datetime import datetime, timedelta, timezone

import numpy as np

from tpa_index.config import (EXAMPLE_VIDEOS, config_hash, effective_settings, load_labels, load_pipeline,
                              load_videos, model_labels)
from tpa_index.moments import merge_moments
from tpa_index.schema import append_reviews, write_table
from tpa_index.score import rules_rows
from tpa_index.segment import windows

BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)
ACTIONS = ["accept", "edit", "reject", "defer", "add"]


def _maybe_null(rng, value, p=0.05):
    return None if rng.random() < p else value


def _idle_signals(rng, idle: bool) -> dict:
    """Idle segments: high idle-prompt score, almost no visual change; others the opposite (with overlap)."""
    score = float(np.float32(rng.beta(6, 3) if idle else rng.beta(1.2, 12)))
    change = float(np.float32(rng.uniform(0.0, 0.015) if idle else rng.uniform(0.005, 0.2)))
    return {"is_idle": idle, "idle_score": score, "visual_change": change}


def _videos_and_segments(rng, videos_cfg, pipeline):
    videos, segments = [], []
    for v in videos_cfg:
        duration = float(rng.uniform(300, 1500))
        videos.append({**{k: v.get(k) for k in ("video_id", "filename", "orientation", "kind", "phase", "cohort",
                                                  "split")}, "duration_s": duration, "fps": 30.0})
        settings = effective_settings(pipeline, v)
        for start, end in windows(duration, settings["window_s"], settings["overlap_s"]):
            people = int(rng.integers(0, 16))
            speech = bool(rng.random() < 0.5)
            segments.append({
                "segment_id": f"{v['video_id']}_{round(start * 1000)}", "video_id": v["video_id"],
                "start_s": start, "end_s": end,
                "person_count": _maybe_null(rng, people),
                "faces_detected": _maybe_null(rng, int(rng.integers(0, people + 1))),
                "has_speech": _maybe_null(rng, speech),
                "transcript_chars": _maybe_null(rng, int(rng.integers(20, 400)) if speech else 0),
                **_idle_signals(rng, people == 0 or bool(rng.random() < 0.05)),
            })
    return videos, segments


def _predictions(rng, segments, labels, run_id, null_group=None):
    by_group: dict[str, list[str]] = {}
    for lab in model_labels(labels):  # rules labels are never model-scored
        by_group.setdefault(lab["group"], []).append(lab["name"])
    rows = []
    latent: dict[str, tuple[str, int]] = {}  # group -> (true label, segments left)
    current_video = None
    for seg in segments:
        if seg["video_id"] != current_video:
            current_video, latent = seg["video_id"], {}
        for group in sorted(by_group):
            names = by_group[group]
            true, left = latent.get(group, (None, 0))
            if left <= 0:
                true, left = names[int(rng.integers(len(names)))], int(rng.integers(2, 8))
            latent[group] = (true, left - 1)
            for name in names:
                if group == null_group:  # unrelated prompts: pure low noise, like a real model
                    score = rng.beta(1, 30)
                else:
                    score = rng.beta(5, 2) if name == true else rng.beta(1.5, 6)
                rows.append({"segment_id": seg["segment_id"], "label": name, "label_group": group,
                             "score": float(np.float32(score)), "model": "mock-model", "run_id": run_id})
    return rows


def _reviews(rng, moments, segments, labels):
    names = [lab["name"] for lab in model_labels(labels)]
    group_of = {lab["name"]: lab["group"] for lab in labels["labels"]}
    targets = [("moment", m["moment_id"], m["primary_label"], "primary") for m in moments if rng.random() < 0.2]
    for s in segments:
        if rng.random() < 0.05:
            label = names[int(rng.integers(len(names)))]
            targets.append(("segment", s["segment_id"], label,
                            "indicator" if group_of[label] == "indicator" else "secondary"))
    rows = []
    for i, (target_type, target_id, label, role) in enumerate(targets):
        action = ACTIONS[i] if i < len(ACTIONS) else ACTIONS[int(rng.integers(len(ACTIONS)))]
        new_label = None
        if action in ("edit", "add"):
            same_group = [n for n in names if group_of[n] == group_of[label] and n != label]
            new_label = same_group[int(rng.integers(len(same_group)))]
            if action == "add":
                label = new_label
        rows.append({
            "review_id": rng.bytes(16).hex(), "target_type": target_type, "target_id": target_id,
            "label": label, "label_role": role, "action": action, "new_label": new_label,
            "source": "staff" if action == "add" or rng.random() < 0.1 else "ai",
            "mode": "review",  # never blind: blind rows are real staff ground truth, and mock ones would skew progress
            "guideline_version": labels["guideline_version"],
            "reviewer": "mock-staff-a" if rng.random() < 0.5 else "mock-staff-b",
            "note": "mock note" if rng.random() < 0.2 else None,
            "created_at": BASE_TIME + timedelta(minutes=i),
        })
    return rows


def write_clips(videos) -> int:
    """Synthetic videos (no real footage) for the mock videos, so the review app can play them."""
    import math

    from tpa_index.paths import data_dir
    from tpa_index.synthetic import make_clip

    rows = videos.to_pylist() if hasattr(videos, "to_pylist") else list(videos)
    for v in rows:
        path = data_dir() / v["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        make_clip(path, 160, 90, seconds=math.ceil(v["duration_s"]), fps=4)
    return len(rows)


def generate(out_root=None, seed: int = 0, n_videos: int = 4) -> str:
    """Write a full mock dataset under {out_root or OUTPUT_DIR}; returns its run_id."""
    rng = np.random.default_rng(seed)
    labels, pipeline = load_labels(), load_pipeline()
    run_id = f"mock-seed{seed}"
    predict_run = f"{run_id}-predict"
    videos, segments = _videos_and_segments(rng, load_videos(EXAMPLE_VIDEOS)[:n_videos], pipeline)
    predictions = _predictions(rng, segments, labels, predict_run, pipeline["calibrate"]["null_group"])
    predictions += rules_rows(segments, predictions, pipeline, labels, "mock-model", predict_run)
    moments = []
    for v in videos:
        v_segments = [s for s in segments if s["video_id"] == v["video_id"]]
        moments += merge_moments(v_segments, predictions, pipeline, predict_run,
                                 window_s=effective_settings(pipeline, v)["window_s"], model="mock-model")
    cfg = config_hash(labels, pipeline)
    runs = [
        {"run_id": rid, "stage": stage, "model": model, "config_hash": cfg, "hardware": "mock-hardware",
         "n_segments": len(segments), "runtime_s": float(rng.uniform(5, 600)),
         "peak_mem_gb": float(rng.uniform(0.5, 12)), "started_at": BASE_TIME, "git_commit": None}
        for rid, stage, model in ((run_id, "mock-segment", None), (predict_run, "mock-predict", "mock-model"))
    ]
    for table, rows in (("videos", videos), ("segments", segments), ("predictions", predictions),
                        ("moments", moments), ("runs", runs)):  # runs last: marks the run complete
        write_table(table, rows, run_id, out_root)
    append_reviews(_reviews(rng, moments, segments, labels), out_root)
    return run_id
