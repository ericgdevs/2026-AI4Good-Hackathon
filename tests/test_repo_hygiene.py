import json
import subprocess

import yaml

from tpa_index.paths import repo_root

MEDIA = (".mp4", ".mov", ".mkv", ".avi", ".wav", ".mp3", ".m4a", ".parquet", ".duckdb")


def git(*args):
    return subprocess.run(["git", *args], cwd=repo_root(), capture_output=True, text=True)


def test_local_files_are_ignored():
    assert git("check-ignore", "-q", "config/videos.local.yaml").returncode == 0
    assert git("check-ignore", "-q", ".env").returncode == 0


def test_assistant_read_deny_rules():
    deny = json.loads((repo_root() / ".claude/settings.json").read_text())["permissions"]["deny"]
    assert "Read(./config/videos.local.yaml)" in deny


def test_no_media_or_outputs_tracked():
    tracked = git("ls-files").stdout.splitlines()
    assert not [f for f in tracked if f.lower().endswith(MEDIA)]


def test_no_tracked_file_mentions_local_filenames():
    local = repo_root() / "config/videos.local.yaml"
    if not local.exists():
        return
    names = set()
    for v in (yaml.safe_load(local.read_text()) or {}).get("videos") or []:
        name = str(v.get("filename", "")).rsplit("/", 1)[-1]
        names |= {name, name.rsplit(".", 1)[0]} - {""}
    hits = 0
    for f in git("ls-files").stdout.splitlines():
        try:
            text = (repo_root() / f).read_text(errors="ignore")
        except (OSError, IsADirectoryError):
            continue
        hits += any(n in text for n in names if len(n) >= 4)
    assert hits == 0, f"{hits} tracked file(s) mention a local video filename"  # count only, never names
