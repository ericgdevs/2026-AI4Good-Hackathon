"""Data locations from .env, the inside-repo guard, and redaction of data paths/filenames."""
import logging
import os
import re
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


load_dotenv(repo_root() / ".env", override=False)


class GuardError(Exception):
    pass


def _setting(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise GuardError(f"{name} is not set (add it to .env)")
    path = Path(value).expanduser().resolve()
    if path.is_relative_to(repo_root()):
        raise GuardError(f"{name} must point outside the repo")
    return path


def data_dir() -> Path:
    return _setting("DATA_DIR")


def output_dir() -> Path:
    return _setting("OUTPUT_DIR")


def check_guard() -> None:
    data_dir()
    output_dir()


_replacements: list[tuple[str, str]] | None = None


def _reset_redaction_cache() -> None:
    global _replacements
    _replacements = None


def _build_replacements() -> list[tuple[str, str]]:
    from tpa_index.config import load_videos

    pairs = []
    for name, tag in (("DATA_DIR", "<DATA_DIR>"), ("OUTPUT_DIR", "<OUTPUT_DIR>")):
        raw = os.environ.get(name)
        if raw:
            pairs += [(raw.rstrip("/"), tag), (str(Path(raw).expanduser().resolve()), tag)]
    try:
        videos = load_videos()
    except Exception:  # redaction must never fail because a config is broken
        videos = []
    for v in videos:
        rel = Path(str(v.get("filename", "")))
        tag = f"<video:{v.get('video_id', '?')}>"
        pairs += [(str(rel), tag), (rel.name, tag)]
        if len(rel.stem) >= 4:  # shortcut: short stems would mangle unrelated text; revisit if real names are short
            pairs.append((rel.stem, tag))
    pairs = [(s, t) for s, t in pairs if s and s != "."]
    return sorted(set(pairs), key=lambda p: -len(p[0]))


def redact(text: str) -> str:
    global _replacements
    if _replacements is None:
        _replacements = _build_replacements()
    for secret, tag in _replacements:
        text = text.replace(secret, tag)
    return re.sub(r"<DATA_DIR>/[^'\"\n]*", "<DATA_DIR>/<file>", text)  # to quote/EOL: names may contain spaces


class _RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = ()
        return True


def _excepthook(exc_type, exc, tb) -> None:
    sys.stderr.write(redact("".join(traceback.format_exception(exc_type, exc, tb))))


def install_redaction() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO)
    for handler in root.handlers:
        if not any(isinstance(f, _RedactFilter) for f in handler.filters):
            handler.addFilter(_RedactFilter())
    sys.excepthook = _excepthook
