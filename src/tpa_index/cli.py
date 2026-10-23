import argparse
import sys

from tpa_index.config import ConfigError, load_labels, load_pipeline, load_videos
from tpa_index.paths import GuardError, check_guard, install_redaction, redact


def _check(args) -> int:
    labels = load_labels()
    load_pipeline()
    load_videos()
    print("guard: ok")
    print(f"configs: ok ({len(labels['labels'])} labels, guideline {labels['guideline_version']})")
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
    print("segment: not implemented")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tpa-index")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check").set_defaults(func=_check)
    mock = sub.add_parser("mock")
    mock.add_argument("--seed", type=int, default=0)
    mock.add_argument("--n-videos", type=int, default=4)
    mock.set_defaults(func=_mock)
    sub.add_parser("segment").set_defaults(func=_segment)
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
