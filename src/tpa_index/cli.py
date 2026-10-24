import argparse
import sys

from tpa_index import config
from tpa_index.config import ConfigError, load_labels, load_pipeline, load_videos
from tpa_index.paths import GuardError, check_guard, data_dir, install_redaction, redact


def _check(args) -> int:
    labels = load_labels()
    load_pipeline()
    videos = load_videos()
    print("guard: ok")
    print(f"configs: ok ({len(labels['labels'])} labels, guideline {labels['guideline_version']})")
    from tpa_index.segment import discover_videos

    mapped = {(data_dir() / v["filename"]).resolve() for v in videos}
    print(f"unmapped videos: {sum(p.resolve() not in mapped for p in discover_videos())}")  # count only, never names
    import torch

    from tpa_index.runs import device, hardware_name

    dev = device()
    print(f"device: {dev}")
    print(f"hardware: {hardware_name(dev)}")
    print(f"torch: {torch.__version__}")
    return 0


def _mock(args) -> int:
    from tpa_index.mock import generate
    from tpa_index.schema import read_reviews, read_table, validate_dataset

    run_id = generate(seed=args.seed, n_videos=args.n_videos)
    validate_dataset()
    print(f"mock run: {run_id}")
    for table in ("videos", "segments", "predictions", "moments", "runs"):
        print(f"  {table}: {read_table(table).num_rows}")
    print(f"  reviews: {read_reviews().num_rows}")
    return 0


def _select_videos(args) -> list[dict]:
    if args.videos is None and not config.LOCAL_VIDEOS.exists():
        raise ConfigError("create config/videos.local.yaml or pass --videos")
    videos = load_videos(args.videos)
    if args.video_id:
        unknown = set(args.video_id) - {v["video_id"] for v in videos}
        if unknown:
            raise ConfigError(f"unknown video_id(s): {', '.join(sorted(unknown))}")
        videos = [v for v in videos if v["video_id"] in args.video_id]
    if getattr(args, "kind", None):
        videos = [v for v in videos if v["kind"] in args.kind]
        if not videos:
            raise ConfigError(f"no mapped videos of kind {', '.join(args.kind)}")
    return videos


def _per_video(stage: str, videos: list[dict], model, work) -> int:
    """One tracked run per video: a bad file never discards the others' work."""
    from tpa_index.runs import track_run

    failed = []
    for v in videos:
        try:
            with track_run(stage, model=model) as ctx:
                ctx.n_segments = work(v, ctx.run_id)
        except (GuardError, ConfigError):
            raise
        except Exception as e:
            failed.append(v["video_id"])
            print(redact(f"{stage} failed for {v['video_id']}: {type(e).__name__}: {e}"), file=sys.stderr)
            continue
        print(f"{stage} run: {ctx.run_id} ({v['video_id']}, {ctx.n_segments} segments)")
    if failed:
        print(f"failed videos: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


def _video_row(v: dict, info: dict) -> dict:
    return {"video_id": v["video_id"], "filename": v["filename"], "duration_s": info["duration_s"],
            "fps": info["fps"], "orientation": info["orientation"], "kind": v["kind"], "phase": v["phase"],
            "cohort": v.get("cohort"), "split": v["split"]}


def _segment(args) -> int:
    from tpa_index.config import effective_settings
    from tpa_index.schema import write_table
    from tpa_index.segment import probe, segment_video

    videos = _select_videos(args)
    pipeline = load_pipeline()

    def work(v, run_id):
        write_table("videos", [_video_row(v, probe(data_dir() / v["filename"]))], run_id)
        rows = [w.row for w in segment_video(v, effective_settings(pipeline, v), pipeline["audio_sample_rate"])]
        write_table("segments", rows, run_id)
        return len(rows)

    return _per_video("segment", videos, None, work)


def _make_embedder(model_id: str, dev: str):
    from tpa_index.embed import SiglipEmbedder

    return SiglipEmbedder(model_id, dev)


def _make_motion(motion_cfg: dict, labels: dict, dev: str):
    from tpa_index.xclip import XClipScorer

    by_name = {lab["name"]: lab for lab in labels["labels"]}
    return XClipScorer(motion_cfg["id"], dev, [by_name[n] for n in motion_cfg["labels"]],
                       motion_cfg["other_prompt"], motion_cfg["frames_per_window"])


def _predict(args) -> int:
    import os

    from tpa_index.runs import device
    from tpa_index.schema import write_table
    from tpa_index.score import predict_video

    videos = _select_videos(args)
    pipeline, labels = load_pipeline(), load_labels()
    dev = device()
    model_id = args.model or pipeline["models"]["main"]["hpc" if dev == "cuda" else "mac"]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # weights come from the local cache only
    embedder = _make_embedder(model_id, dev)  # once, before any video; missing model -> ConfigError (exit 3)
    motion_cfg = pipeline["models"].get("motion") or {}
    motion = _make_motion(motion_cfg, labels, dev) if motion_cfg.get("enabled") else None

    def work(v, run_id):
        out = predict_video(v, pipeline, labels, embedder, run_id, device=dev, motion=motion)
        for table in ("videos", "segments", "predictions", "moments"):
            write_table(table, out[table], run_id)
        return len(out["segments"])

    return _per_video("predict", videos, embedder.name, work)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tpa-index")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check").set_defaults(func=_check)
    mock = sub.add_parser("mock")
    mock.add_argument("--seed", type=int, default=0)
    mock.add_argument("--n-videos", type=int, default=4)
    mock.set_defaults(func=_mock)
    for name, func in (("segment", _segment), ("predict", _predict)):
        cmd = sub.add_parser(name)
        cmd.add_argument("--videos", help="videos YAML (default: config/videos.local.yaml)")
        cmd.add_argument("--video-id", action="append", help="only these video_ids (repeatable)")
        cmd.set_defaults(func=func)
    pred = sub.choices["predict"]
    pred.add_argument("--kind", action="append", help="only videos of this kind (repeatable), e.g. promo")
    pred.add_argument("--model", help="model id (default: models.main.hpc on cuda, models.main.mac otherwise)")
    args = parser.parse_args(argv)

    install_redaction()
    try:
        check_guard()
        return args.func(args)
    except GuardError as e:
        print(redact(str(e)), file=sys.stderr)
        return 2
    except ConfigError as e:
        print(redact(str(e)), file=sys.stderr)
        return 3
