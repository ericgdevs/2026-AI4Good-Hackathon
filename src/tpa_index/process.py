"""Process tab: run the laptop pipeline on one configured video, in a background thread, one video at a time.
Reuses the predict and signal stage code (cli._predict_run, signals.run_signals). The `moments` CLI step is not run:
it only rebuilds from official HPC runs, and predict already writes this run's moments. Transcript text is never
written. Finished runs are registered in OUTPUT_DIR/local_runs.yaml, never in official_runs.yaml, so no result
(evaluate, insights, Scorecard, export) changes."""
import os
import threading
import time
from statistics import median

from tpa_index.config import ConfigError, load_labels, load_pipeline
from tpa_index.paths import GuardError, redact

STAGES = (("predict", "Activity", "SigLIP2"), ("transcript", "Speech present (yes/no only)", "Whisper"),
          ("audio", "Sounds", "AST"), ("people", "People counts", "YOLO11"),
          ("expression", "Expressions (unvalidated)", "YuNet + EmotiEffLib"))
TITLES = {s: t for s, t, _ in STAGES}
SIGNAL_ORDER = ("transcript", "audio", "people", "expression")  # run_signals' order: audio pass, then frame pass
FRAME_STAGES = ("people", "expression")
BUSY = "Another video is being processed. Wait for it to finish, then try again."
_LOCK = threading.Lock()


def plain(stage: str, error) -> str:
    """A plain-English line for a failed step. `error` is an exception or an error type name (run_signals reports
    the type only). Raw messages are shown only for our own config errors, redacted."""
    title = TITLES.get(stage, stage)
    name = error if isinstance(error, str) else type(error).__name__
    text = "" if isinstance(error, str) else str(error)
    if isinstance(error, GuardError) or name == "GuardError":
        return f"{title}: the data folders are not set up on this laptop (DATA_DIR and OUTPUT_DIR in .env)."
    if isinstance(error, ConfigError) and ("not found" in text or "weights" in text):
        return (f"{title}: the model files for this step are not on this laptop yet. Ask whoever set up the laptop "
                "to download the laptop models (see docs/weights.md).")
    if isinstance(error, ConfigError):
        return f"{title}: a settings problem stopped this step ({redact(text)})."
    if name in ("FileNotFoundError", "InvalidDataError", "FFmpegError", "IsADirectoryError", "PermissionError"):
        return f"{title}: the video file could not be opened or read."
    if name == "TranscriptError":
        return f"{title}: speech detection could not finish on this video."
    return f"{title}: stopped unexpectedly ({name})."


def official_videos(root=None) -> set[str]:
    from tpa_index.schema import _official_by_video, load_official

    official = load_official()
    return set(_official_by_video(official, "predict", root)) | set(_official_by_video(official, "moments", root))


def video_choices(root=None) -> tuple[list[tuple[str, str]], str]:
    """(choices, note) from the video config: video IDs only, never filenames."""
    from tpa_index.config import load_videos

    try:
        videos = load_videos()
        official = official_videos(root)
    except (ConfigError, GuardError) as e:
        return [], redact(f"No videos to process: {e}")
    return [(f"{v['video_id']} · {'has official results' if v['video_id'] in official else 'new'}"
             + (" · eval video" if v["split"] == "eval" else ""), v["video_id"])
            for v in sorted(videos, key=lambda v: v["video_id"])], ""


def estimate(duration_s: float, root=None) -> float | None:
    """Seconds of processing for a video this long, from earlier laptop runs (median seconds per video second per
    stage; people and expression share one pass, so the slower counts). None before the first laptop run."""
    from tpa_index.schema import load_official, local_runs_path, read_table

    rates: dict[str, list[float]] = {}
    for stage, ids in load_official(local_runs_path(root)).items():
        for rid in ids:
            run = read_table("runs", root, run_ids=[rid]).to_pylist()
            vids = read_table("videos", root, run_ids=[rid]).to_pylist()
            secs = sum(v["duration_s"] for v in vids)
            if run and secs > 0:
                rates.setdefault(stage, []).append(run[0]["runtime_s"] / secs)
    rate = {s: median(v) for s, v in rates.items()}
    if "predict" not in rate:
        return None
    frames = max((rate[s] for s in FRAME_STAGES if s in rate), default=0.0)
    return duration_s * (rate["predict"] + rate.get("transcript", 0.0) + rate.get("audio", 0.0) + frames)


class Job:
    """One Process run: stage states, counts and the final message. Read by the app while the worker writes."""

    def __init__(self, video: dict, live, clock=time.monotonic):
        self.video, self.live, self.clock = video, live, clock
        self.started, self.finished = clock(), None
        self.status = {s: "waiting" for s, *_ in STAGES}
        self.errors: dict[str, str] = {}
        self.run_ids: dict[str, list[str]] = {}
        self.moments: list[dict] = []
        self.n_segments = self.duration_s = self.estimate_s = None
        self.message = ""
        self.done = threading.Event()
        self._lock = threading.Lock()

    def set(self, stage: str, status: str, error: str | None = None) -> None:
        with self._lock:
            self.status[stage] = status
            if error:
                self.errors[stage] = error

    def _advance(self) -> None:
        nxt = next((s for s in SIGNAL_ORDER if self.status[s] == "waiting"), None)
        if nxt is not None:
            for s in FRAME_STAGES if nxt in FRAME_STAGES else (nxt,):
                if self.status[s] == "waiting":
                    self.status[s] = "running"

    def report(self, key, status: str, n_segments=None, secs=None) -> None:
        """run_signals callback: one (video, stage) pair finished."""
        _, stage = key
        kind, _, detail = status.partition(":")
        with self._lock:
            if kind == "done":
                self.status[stage] = "done"
                self.run_ids[stage] = [detail]
            else:
                self.status[stage] = "failed"
                self.errors[stage] = plain(stage, detail)
            self._advance()

    def snapshot(self) -> dict:
        with self._lock:
            end = self.finished if self.finished is not None else self.clock()
            counts = self.live.view()["counts"] if self.live is not None else {"predict": 0, "frames": 0}
            return {"video_id": self.video["video_id"], "status": dict(self.status), "errors": dict(self.errors),
                    "elapsed_s": end - self.started, "estimate_s": self.estimate_s, "n_segments": self.n_segments,
                    "counts": counts, "done": self.done.is_set(), "message": self.message}


def start(video_id: str, root=None, make_embedder=None, make_models=None, make_blur=None) -> "Job | str":
    """Start processing in a background thread and return its Job, or a plain-English reason it cannot start."""
    from tpa_index.config import load_videos
    from tpa_index.preview import FaceBlur, Live

    if not _LOCK.acquire(blocking=False):
        return BUSY
    try:
        video = next((v for v in load_videos() if v["video_id"] == video_id), None)
        if video is None:
            _LOCK.release()
            return "Pick a video from the list first."
        pipeline = load_pipeline()
        try:
            blur = (make_blur or FaceBlur.load)(pipeline)
        except Exception:  # noqa: BLE001 - fail closed: the preview shows no images, processing still runs
            blur = None
        from tpa_index.config import load_theme

        job = Job(video, Live(blur, box_colour=load_theme()["accent"]))
        threading.Thread(target=_worker, args=(job, root, make_embedder, make_models), daemon=True,
                         name="tpa-process").start()
        return job
    except BaseException:
        _LOCK.release()
        raise


def _worker(job: Job, root, make_embedder, make_models) -> None:
    try:
        run(job, root, make_embedder, make_models)
    except Exception as e:  # noqa: BLE001 - the app must always get a message
        job.message = plain("predict", e)
    finally:
        job.finished = job.clock()
        job.done.set()
        _LOCK.release()


def run(job: Job, root=None, make_embedder=None, make_models=None) -> None:
    """Predict, then every signal stage (each model loaded on its own, so one missing model fails only its step),
    then register what finished. Steps run one after another; a failed signal step never stops the others."""
    from tpa_index import cli
    from tpa_index.config import effective_settings
    from tpa_index.paths import data_dir
    from tpa_index.runs import device, track_run
    from tpa_index.schema import register_local
    from tpa_index.segment import probe, windows
    from tpa_index.signals import run_signals

    make_embedder = make_embedder or cli._make_embedder
    make_models = make_models or cli._make_signal_models
    pipeline, labels = load_pipeline(), load_labels()
    dev = device()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # weights come from the local cache only
    os.environ["YOLO_OFFLINE"] = "1"
    v = job.video
    job.set("predict", "running")
    try:
        job.duration_s = probe(data_dir() / v["filename"])["duration_s"]
        s = effective_settings(pipeline, v)
        job.n_segments = len(windows(job.duration_s, s["window_s"], s["overlap_s"]))
        job.estimate_s = estimate(job.duration_s, root)
        embedder = make_embedder(pipeline["models"]["main"]["hpc" if dev == "cuda" else "mac"], dev)
        motion_cfg = pipeline["models"].get("motion") or {}
        motion = cli._make_motion(motion_cfg, labels, dev) if motion_cfg.get("enabled") else None
        with track_run("predict", model=embedder.name, root=root) as ctx:
            out = cli._predict_run(v, ctx.run_id, pipeline, labels, embedder, dev, motion, root, job.live)
            ctx.n_segments = len(out["segments"])
        del embedder, motion  # free the activity model before the signal models load
    except Exception as e:  # noqa: BLE001
        job.set("predict", "failed", plain("predict", e))
        for stage in SIGNAL_ORDER:
            job.set(stage, "skipped")
        job.message = "Processing stopped: " + job.errors["predict"]
        return
    job.set("predict", "done")
    job.run_ids["predict"] = [ctx.run_id]
    job.moments = out["moments"]

    models = {}
    for stage in SIGNAL_ORDER:
        try:
            models.update(make_models([stage], pipeline, dev))
        except Exception as e:  # noqa: BLE001
            job.set(stage, "failed", plain(stage, e))
    with job._lock:
        job._advance()
    if models:
        run_signals([v], list(models), models, pipeline, labels, force=True, root=root, device=dev,
                    report=job.report, keep_transcript=False, segments={v["video_id"]: (out["segments"], out["videos"][0])},
                    preview=job.live)
    register_local(job.run_ids, root)
    job.message = summary(job, root)


def summary(job: Job, root=None) -> str:
    """The plain-English result: moments found (with the uncalibrated caveat), where the run shows, what failed."""
    from tpa_index.app import LOCAL_BADGE

    vid, n = job.video["video_id"], len(job.moments)
    lines = []
    if n == 0:
        lines.append("No clear moments were found in this video. The laptop's activity model has no calibrated "
                     "cut-offs yet, so it rarely marks a moment. The clip-by-clip suggestions in the preview are AI "
                     "suggestions only.")
    else:
        lines.append(f"{n} moment{'s' if n != 1 else ''} found. The laptop's activity model is uncalibrated, so "
                     "expect fewer and less reliable moments than in the official results.")
    if vid in official_videos(root):
        lines.append("This video already has official results. They are unchanged; this laptop run is shown only "
                     "here.")
    else:
        lines.append(f"It now appears in Search moments and Review queue, marked “{LOCAL_BADGE}”. Reviews can't be "
                     "saved on it.")
        if job.video["split"] == "eval":
            lines.append("As an evaluation video, it stays hidden in Review queue until blind labeling is done.")
    failed = [job.errors[s] for s, *_ in STAGES if s in job.errors]
    if failed:
        lines.append("Some steps did not finish: " + " ".join(failed))
    lines.append("Laptop runs are never counted in the Scorecard or the export.")
    return "\n\n".join(lines)
