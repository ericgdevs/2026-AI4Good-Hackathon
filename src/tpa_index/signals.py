"""004 signal stages: transcript, audio, people, expression. One immutable run per (video, stage), written to the
existing contract tables. Per-segment aggregates only: no identity, no crops or frames on disk, and no transcript
text outside the run's transcript.json."""
import logging
import os
import re
import tempfile
from pathlib import Path

from tpa_index.config import ConfigError
from tpa_index.paths import GuardError, repo_root

log = logging.getLogger(__name__)
logging.getLogger("faster_whisper").setLevel(logging.WARNING)  # its INFO lines are noise; text never logged anyway

SIGNAL_COLUMNS = ("person_count", "faces_detected", "has_speech", "transcript_chars", "is_idle", "idle_score",
                  "visual_change")
_WAV = re.compile(r"^tpa-audio-(\d+)-.*\.wav$")


class TranscriptError(Exception):
    """Raised in place of any transcript-stage error; the message is the original type name only (no text)."""


def scratch_dir() -> Path:
    """Folder for the temporary WAV: TPA_SCRATCH_DIR, else $TMPDIR (per Slurm job), else the system temp."""
    path = Path(os.environ.get("TPA_SCRATCH_DIR") or os.environ.get("TMPDIR") or tempfile.gettempdir())
    path = path.expanduser().resolve()
    if path.is_relative_to(repo_root()):
        raise GuardError("TPA_SCRATCH_DIR must point outside the repo")
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_stale_scratch(scratch: Path) -> int:
    """Remove WAVs left by processes that no longer run (killed jobs). Returns how many were removed."""
    import psutil

    removed = 0
    for p in scratch.glob("tpa-audio-*.wav"):
        m = _WAV.match(p.name)
        if m and not psutil.pid_exists(int(m.group(1))):
            p.unlink(missing_ok=True)
            removed += 1
    return removed


def _official_rows(table: str, video_id: str, root=None) -> list[dict]:
    """Rows of one video from the official runs (moments/predict, else segment)."""
    from tpa_index.schema import read_official

    rows = [r for r in read_official(table, root=root).to_pylist() if r["video_id"] == video_id]
    if not rows:
        rows = [r for r in read_official(table, stage="segment", root=root).to_pylist() if r["video_id"] == video_id]
    return rows


def official_segments(video_id: str, root=None) -> list[dict]:
    """The video's official segments, sorted by start (the segment IDs every stage must reproduce)."""
    rows = _official_rows("segments", video_id, root)
    if not rows:
        raise ConfigError(f"no official segments for {video_id}")
    return sorted(rows, key=lambda r: r["start_s"])


def official_video(video_id: str, root=None) -> dict:
    rows = _official_rows("videos", video_id, root)
    if not rows:
        raise ConfigError(f"no official video row for {video_id}")
    return rows[0]


def check_windows(video_row: dict, settings: dict, official: list[dict]) -> None:
    """FR-004: the windows this stage would cut must be exactly the official segments."""
    from tpa_index.segment import windows

    ours = [f"{video_row['video_id']}_{round(s * 1000)}"
            for s, _ in windows(video_row["duration_s"], settings["window_s"], settings["overlap_s"])]
    if ours != [r["segment_id"] for r in official]:
        raise ValueError("windows do not match official segments")


def base_rows(official: list[dict]) -> list[dict]:
    """Copies of the official segment rows with every optional column empty; a stage fills only its own."""
    return [{**{k: r[k] for k in ("segment_id", "video_id", "start_s", "end_s")}, **dict.fromkeys(SIGNAL_COLUMNS)}
            for r in official]


# ---- transcript ----------------------------------------------------------------------------------------------------
def assign_words(words: list[dict], segments: list[dict]) -> dict[str, tuple[bool, int]]:
    """segment_id -> (has_speech, transcript_chars). A word belongs to every segment containing its midpoint
    (so with overlapping windows it can count twice); the video's last segment also takes a midpoint at its end."""
    out = {s["segment_id"]: [False, 0] for s in segments}
    last = max((s["end_s"] for s in segments), default=0.0)
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        for s in segments:
            if s["start_s"] <= mid < s["end_s"] or (mid == s["end_s"] == last):
                out[s["segment_id"]][0] = True
                out[s["segment_id"]][1] += len(w["word"].strip())
    return {k: (v[0], v[1]) for k, v in out.items()}


def transcript_rows(official: list[dict], transcript_segments: list[dict]) -> list[dict]:
    words = [w for seg in transcript_segments for w in seg.get("words") or []]
    counts = assign_words(words, official)
    rows = base_rows(official)
    for r in rows:
        r["has_speech"], r["transcript_chars"] = counts[r["segment_id"]]
    return rows


# ---- audio ---------------------------------------------------------------------------------------------------------
def extract_audio(video: dict, duration_s: float, path: Path, rate: int = 16000) -> Path:
    """The video's audio as one 16 kHz mono PCM WAV at `path` (silence if the video has no audio track)."""
    import numpy as np
    import soundfile as sf

    from tpa_index.paths import data_dir
    from tpa_index.segment import _load_audio

    # shortcut: decodes the whole track to memory first (~230 MB per hour); stream to the file if videos get longer
    audio = _load_audio(data_dir() / video["filename"], rate)
    if audio is None:
        log.info("video %s has no audio track; using silence", video["video_id"])
        audio = np.zeros(int(round(duration_s * rate)), np.float32)
    sf.write(str(path), audio, rate, subtype="PCM_16")
    return path


def read_clip(wav_path: Path, start_s: float, end_s: float, n_samples: int, rate: int = 16000):
    """One window's samples, zero-padded (silence) or cut to the model's input length."""
    import numpy as np
    import soundfile as sf

    data, _ = sf.read(str(wav_path), start=int(round(start_s * rate)), frames=int(round((end_s - start_s) * rate)),
                      dtype="float32", always_2d=False)
    out = np.zeros(n_samples, np.float32)
    n = min(n_samples, len(data))
    out[:n] = data[:n]
    return out


def map_scores(probs, class_index: dict[str, list[int]]):
    """(n, n_classes) model probabilities -> (n, n_labels): each label takes its highest mapped class."""
    import numpy as np

    return np.stack([probs[:, idx].max(axis=1) for idx in class_index.values()], axis=1).astype(np.float32)


# ---- people and expression ----------------------------------------------------------------------------------------
def median_count(counts: list[int]) -> int:
    """Per-frame counts -> one count for the segment: the median, halves rounded down."""
    import math
    import statistics

    return int(math.floor(statistics.median(counts)))


def aggregate_expression(per_frame_scores: list, labels: list[str]) -> tuple[int, dict[str, float] | None]:
    """per_frame_scores: one (n_faces, n_labels) array per sampled frame. Returns (faces_detected, scores):
    faces_detected = most usable faces in any one frame; scores = mean over every face in every frame,
    or None when there is no face at all (then no expression rows are written)."""
    import numpy as np

    faces = max((len(a) for a in per_frame_scores), default=0)
    if not faces:
        return 0, None
    mean = np.concatenate([a for a in per_frame_scores if len(a)]).mean(axis=0)
    return faces, {label: float(v) for label, v in zip(labels, mean)}


# ---- orchestration -------------------------------------------------------------------------------------------------
AUDIO_STAGES, FRAME_STAGES = ("transcript", "audio"), ("people", "expression")


def _write_run_tables(run_id: str, video_row: dict, rows: list[dict], predictions=None, root=None) -> None:
    from tpa_index.schema import write_table

    write_table("videos", [video_row], run_id, root)
    write_table("segments", rows, run_id, root)
    if predictions is not None:
        write_table("predictions", predictions, run_id, root)


def _transcript_stage(model, wav: Path, video_row: dict, official: list[dict], run_id: str, root) -> int:
    """Whole-file transcription. Any error leaves as TranscriptError(<type name>): text can never ride along in an
    exception message, its context or a log line."""
    import json

    from tpa_index.schema import _runs_root

    failed = None
    try:
        segments = model.transcribe(wav)
        _write_run_tables(run_id, video_row, transcript_rows(official, segments), root=root)
        with open(_runs_root(root) / run_id / "transcript.json", "w") as f:
            json.dump({"video_id": video_row["video_id"], "model": model.name,
                       "language": getattr(model, "language", None), "segments": segments}, f, ensure_ascii=False)
        del segments
    except Exception as e:  # noqa: BLE001 (re-raised below, outside the handler, so no __context__ is kept)
        failed = type(e).__name__
    if failed:
        raise TranscriptError(failed)
    return len(official)


def _audio_stage(model, wav: Path, video_row: dict, official: list[dict], run_id: str, batch: int, rate: int,
                 root) -> int:
    import numpy as np

    preds = []
    for i in range(0, len(official), batch):
        chunk = official[i:i + batch]
        scores = model.score([read_clip(wav, s["start_s"], s["end_s"], model.input_samples, rate) for s in chunk])
        preds += [{"segment_id": s["segment_id"], "label": label, "label_group": "audio",
                   "score": float(np.clip(v, 0.0, 1.0)), "model": model.name, "run_id": run_id}
                  for s, row in zip(chunk, scores) for label, v in zip(model.labels, row)]
    _write_run_tables(run_id, video_row, base_rows(official), preds, root)
    return len(official)


def _fail(results: dict, key: tuple[str, str], e: BaseException) -> None:
    name = type(e).__name__ if not isinstance(e, TranscriptError) else f"TranscriptError({e})"
    results[key] = f"failed:{name}"
    log.error("%s failed for %s: %s", key[1], key[0], name)


def run_signals(videos: list[dict], stages, models: dict, pipeline: dict, labels: dict, force: bool = False,
                root=None, device: str = "cpu") -> dict[tuple[str, str], str]:
    """Run the selected stages per video; one run per (video, stage). Returns {(video_id, stage): status} with
    status "done:<run_id>", "skipped:<run_id>" or "failed:<ErrorType>". One pair failing never stops the others."""
    from tpa_index.config import effective_settings, stage_config_hash
    from tpa_index.runs import track_run
    from tpa_index.schema import find_complete_run

    stages = [s for s in (*AUDIO_STAGES, *FRAME_STAGES) if s in set(stages)]
    scratch = scratch_dir()
    clean_stale_scratch(scratch)
    rate, batch = pipeline["audio_sample_rate"], pipeline["models"]["batch_windows"][device]
    results: dict[tuple[str, str], str] = {}
    for video in videos:
        vid = video["video_id"]
        try:
            official, video_row = official_segments(vid, root), official_video(vid, root)
            check_windows(video_row, effective_settings(pipeline, video), official)
        except Exception as e:  # noqa: BLE001
            for stage in stages:
                _fail(results, (vid, stage), e)
            continue
        todo = []
        for stage in stages:
            cfg = stage_config_hash(stage, labels, pipeline, video)
            done = None if force else find_complete_run(stage, vid, models[stage].name, cfg, root)
            if done:
                results[(vid, stage)] = f"skipped:{done}"
            else:
                todo.append((stage, cfg))

        audio_todo = [(s, c) for s, c in todo if s in AUDIO_STAGES]
        wav = scratch / f"tpa-audio-{os.getpid()}-{vid}.wav"
        try:
            for stage, cfg in audio_todo:
                try:
                    with track_run(stage, model=models[stage].name, cfg_hash=cfg, root=root) as ctx:
                        if not wav.exists():  # extracted once, inside the first audio stage's run
                            extract_audio(video, video_row["duration_s"], wav, rate)
                        if stage == "transcript":
                            ctx.n_segments = _transcript_stage(models[stage], wav, video_row, official, ctx.run_id,
                                                               root)
                        else:
                            ctx.n_segments = _audio_stage(models[stage], wav, video_row, official, ctx.run_id,
                                                          batch, rate, root)
                    results[(vid, stage)] = f"done:{ctx.run_id}"
                except Exception as e:  # noqa: BLE001
                    _fail(results, (vid, stage), e)
        finally:
            wav.unlink(missing_ok=True)

        frame_todo = [(s, c) for s, c in todo if s in FRAME_STAGES]
        if frame_todo:
            results.update(_frame_stages(video, video_row, official, frame_todo, models, pipeline, root, device))
    return results


def _frame_stages(video, video_row, official, todo, models, pipeline, root, device) -> dict:
    """People and expression share one streaming pass over 002's frame sampling, one official segment at a time.
    Each stage fails on its own; frames and face crops stay in memory and are dropped as soon as they are scored.
    The pass is timed once, and both runs record that runtime and peak memory (overlapping: do not sum)."""
    import time

    import av
    import numpy as np

    from tpa_index.config import effective_settings
    from tpa_index.paths import data_dir
    from tpa_index.runs import _peak_mem_gb, track_run
    from tpa_index.segment import _sample_frames

    vid, results = video["video_id"], {}
    active = [stage for stage, _ in todo]
    out: dict[str, dict] = {stage: {} for stage in active}
    n_frames = effective_settings(pipeline, video)["frames_per_window"]
    if device == "cuda":
        import torch

        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        with av.open(str(data_dir() / video["filename"])) as container:
            stream = container.streams.video[0]
            for seg in official:
                if not active:
                    break
                frames = _sample_frames(container, stream, seg["start_s"], seg["end_s"], n_frames)
                for stage in list(active):
                    try:
                        if stage == "people":
                            out[stage][seg["segment_id"]] = median_count(models["people"].count(frames))
                        else:
                            scorer, per_frame = models["expression"], []
                            for frame in frames:
                                crops = scorer.faces(frame)
                                per_frame.append(scorer.score(crops) if crops
                                                 else np.zeros((0, len(scorer.labels)), np.float32))
                                del crops
                            out[stage][seg["segment_id"]] = aggregate_expression(per_frame, scorer.labels)
                    except Exception as e:  # noqa: BLE001
                        _fail(results, (vid, stage), e)
                        active.remove(stage)
                del frames
    except Exception as e:  # noqa: BLE001 (the video could not be decoded: every remaining stage fails)
        for stage in active:
            _fail(results, (vid, stage), e)
        return results
    runtime, peak = max(time.perf_counter() - t0, 1e-6), _peak_mem_gb(device)

    for stage, cfg in todo:
        if stage not in active:
            continue
        try:
            with track_run(stage, model=models[stage].name, cfg_hash=cfg, root=root) as ctx:
                rows, preds = base_rows(official), None
                if stage == "people":
                    for r in rows:
                        r["person_count"] = out[stage][r["segment_id"]]
                else:
                    preds = []
                    for r in rows:
                        faces, scores = out[stage][r["segment_id"]]
                        r["faces_detected"] = faces
                        preds += [{"segment_id": r["segment_id"], "label": label, "label_group": "expression",
                                   "score": float(np.clip(v, 0.0, 1.0)), "model": models[stage].name,
                                   "run_id": ctx.run_id} for label, v in (scores or {}).items()]
                _write_run_tables(ctx.run_id, video_row, rows, preds, root)
                ctx.n_segments, ctx.runtime_s, ctx.peak_mem_gb = len(rows), runtime, peak
            results[(vid, stage)] = f"done:{ctx.run_id}"
        except Exception as e:  # noqa: BLE001
            _fail(results, (vid, stage), e)
    return results
