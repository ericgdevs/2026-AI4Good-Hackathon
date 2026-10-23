import argparse
import sys

from tpa_index.config import ConfigError, load_labels, load_pipeline, load_videos
from tpa_index.paths import GuardError, check_guard, data_dir, install_redaction, redact


def _check(args) -> int:
    labels = load_labels()
    load_pipeline()
    load_videos()
    print("guard: ok")
    print(f"configs: ok ({len(labels['labels'])} labels, guideline {labels['guideline_version']})")
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


def _segment(args) -> int:
    from tpa_index.config import effective_settings
    from tpa_index.runs import track_run
    from tpa_index.schema import write_table
    from tpa_index.segment import probe, segment_video

    pipeline = load_pipeline()
    videos = load_videos(args.videos)
    if args.video_id:
        unknown = set(args.video_id) - {v["video_id"] for v in videos}
        if unknown:
            raise ConfigError(f"unknown video_id(s): {', '.join(sorted(unknown))}")
        videos = [v for v in videos if v["video_id"] in args.video_id]
    try:
        with track_run("segment") as ctx:
            video_rows, segment_rows = [], []
            for v in videos:
                info = probe(data_dir() / v["filename"])
                video_rows.append({"video_id": v["video_id"], "filename": v["filename"],
                                   "duration_s": info["duration_s"], "fps": info["fps"],
                                   "orientation": info["orientation"], "kind": v["kind"], "phase": v["phase"],
                                   "cohort": v.get("cohort"), "split": v["split"]})
                settings = effective_settings(pipeline, v)
                segment_rows += [w.row for w in segment_video(v, settings, pipeline["audio_sample_rate"])]
            write_table("videos", video_rows, ctx.run_id)
            write_table("segments", segment_rows, ctx.run_id)
            ctx.n_segments = len(segment_rows)
    except (GuardError, ConfigError):
        raise
    except Exception as e:
        print(redact(f"segment failed: {type(e).__name__}: {e}"), file=sys.stderr)
        return 1
    print(f"segment run: {ctx.run_id} ({len(video_rows)} videos, {ctx.n_segments} segments)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tpa-index")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check").set_defaults(func=_check)
    mock = sub.add_parser("mock")
    mock.add_argument("--seed", type=int, default=0)
    mock.add_argument("--n-videos", type=int, default=4)
    mock.set_defaults(func=_mock)
    seg = sub.add_parser("segment")
    seg.add_argument("--videos", help="videos YAML (default: config/videos.local.yaml, else the example)")
    seg.add_argument("--video-id", action="append", help="only these video_ids (repeatable)")
    seg.set_defaults(func=_segment)
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
