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


def _main_model(predictions) -> str | None:
    """Most common non-rules model in a predictions table."""
    from collections import Counter

    counts = Counter(m for m in predictions["model"].to_pylist() if m != "rules")
    return counts.most_common(1)[0][0] if counts else None


def _calibrate(args) -> int:
    import numpy as np

    from tpa_index.calibrate import (dev_scores, idle_candidates, propose_idle_threshold, propose_thresholds,
                                     write_local_thresholds)
    from tpa_index.schema import read_official

    if args.split != "dev":
        raise ConfigError("calibration uses the dev split only")
    pipeline = load_pipeline()
    cal, idle = pipeline["calibrate"], pipeline["idle"]
    top_pct = args.top_pct if args.top_pct is not None else cal["top_pct"]
    idle_top_pct = args.idle_top_pct if args.idle_top_pct is not None else idle["calibrate_top_pct"]
    preds, segs, videos = (read_official(t, args.stage) for t in ("predictions", "segments", "videos"))
    if not preds.num_rows or not segs.num_rows or not videos.num_rows \
            or "dev" not in set(videos["split"].to_pylist()):
        raise ConfigError(f"no official dev runs for stage {args.stage}")
    model = _main_model(preds)
    proposals = propose_thresholds(preds, videos, model, top_pct, cal["min_segments"], cal["min_threshold"])
    scores = dev_scores(preds, videos, model)
    print(f"model: {model}  label top_pct: {top_pct}  floor: {cal['min_threshold']}")
    print(f"{'label':<20} {'threshold':>9} {'n_dev':>6} {'clearing':>8}  floored")
    for label, (value, floored) in proposals.items():
        n = len(scores[label])
        clearing = "-" if value is None else int((scores[label] >= value).sum())
        shown = "too few" if value is None else f"{value:.4f}"
        print(f"{label:<20} {shown:>9} {n:>6} {clearing:>8}  {'yes' if floored else ''}")
    at_floor = [label for label, (_, floored) in proposals.items() if floored]
    print(f"labels at the floor (min_threshold): {', '.join(at_floor) if at_floor else 'none'}")
    idle_value = propose_idle_threshold(segs, videos, idle_top_pct, idle["change_threshold"], cal["min_segments"])
    n_idle = len(idle_candidates(segs, videos, idle["change_threshold"]))
    idle_shown = "too few, keeps fallback" if idle_value is None else f"{idle_value:.4f}"
    print(f"idle score_threshold: {idle_shown} ({n_idle} dev low-change segments, top_pct {idle_top_pct})")
    if args.dry_run:
        print("dry run: nothing written")
        return 0
    write_local_thresholds(model, {k: v for k, (v, _) in proposals.items()}, idle_value)
    print(f"written to {config.THRESHOLDS_LOCAL.name} (gitignored, data-derived)")
    return 0


def _moments(args) -> int:
    from tpa_index.config import effective_settings, idle_threshold
    from tpa_index.moments import merge_moments
    from tpa_index.runs import track_run
    from tpa_index.schema import load_official, read_table, write_table
    from tpa_index.score import is_idle, rules_rows

    pipeline, labels = load_pipeline(), load_labels()
    sources = load_official().get("predict", [])
    if not sources:
        raise ConfigError("no official predict runs to rebuild")
    failed, wanted = [], set(args.video_id or [])
    for source in sources:
        videos = read_table("videos", run_ids=[source]).to_pylist()
        if not videos:
            raise ConfigError(f"official run {source} not found or incomplete")
        if wanted and not wanted & {v["video_id"] for v in videos}:
            continue
        try:
            preds = read_table("predictions", run_ids=[source])
            model = _main_model(preds)
            with track_run("moments", model=model) as ctx:
                model_rows = [p for p in preds.to_pylist() if p["model"] != "rules"]
                idle_cfg = {**pipeline["idle"], "score_threshold": idle_threshold(pipeline, model)}
                segments = []
                for s in read_table("segments", run_ids=[source]).to_pylist():
                    has_inputs = s["idle_score"] is not None and s["visual_change"] is not None
                    s["is_idle"] = has_inputs and is_idle(s["idle_score"], s["visual_change"], idle_cfg)
                    segments.append(s)
                rules = rules_rows(segments, model_rows, pipeline, labels, model, ctx.run_id)
                moments = []
                for v in videos:
                    v_segs = [s for s in segments if s["video_id"] == v["video_id"]]
                    moments += merge_moments(v_segs, model_rows + rules, pipeline, ctx.run_id,
                                             window_s=effective_settings(pipeline, v)["window_s"], model=model)
                write_table("videos", videos, ctx.run_id)
                write_table("segments", segments, ctx.run_id)
                write_table("predictions", rules, ctx.run_id)
                write_table("moments", moments, ctx.run_id)
                ctx.n_segments = len(segments)
        except (GuardError, ConfigError):
            raise
        except Exception as e:
            failed.append(source)
            print(redact(f"moments failed for {source}: {type(e).__name__}: {e}"), file=sys.stderr)
            continue
        print(f"moments run: {ctx.run_id} (from {source}, {len(moments)} moments)")
    if failed:
        print(f"failed runs: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


def _runs(args) -> int:
    from tpa_index.schema import list_runs

    cols = ["run_id", "stage", "video_ids", "hardware", "n_segments", "n_predictions", "runtime_s", "peak_mem_gb"]
    rows = [{**r, "runtime_s": f"{r['runtime_s']:.1f}",
             "peak_mem_gb": "-" if r["peak_mem_gb"] is None else f"{r['peak_mem_gb']:.2f}"} for r in list_runs()]
    widths = {c: max([len(c)] + [len(str(r[c])) for r in rows]) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tpa-index")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check").set_defaults(func=_check)
    sub.add_parser("runs", help="list complete runs (no filenames or paths)").set_defaults(func=_runs)
    cal = sub.add_parser("calibrate", help="propose thresholds from official dev-split data")
    cal.add_argument("--stage", default="predict")
    cal.add_argument("--split", default="dev")
    cal.add_argument("--top-pct", type=float, help="label percentile (default: calibrate.top_pct)")
    cal.add_argument("--idle-top-pct", type=float, help="idle percentile (default: idle.calibrate_top_pct)")
    cal.add_argument("--dry-run", action="store_true")
    cal.set_defaults(func=_calibrate)
    mom = sub.add_parser("moments", help="rebuild idle flags and moments from official predict runs")
    mom.add_argument("--video-id", action="append")
    mom.set_defaults(func=_moments)
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
