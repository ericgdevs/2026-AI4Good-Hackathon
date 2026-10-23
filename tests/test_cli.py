import os
import subprocess
import sys

from tpa_index.paths import repo_root

RUN = "import sys; from tpa_index.cli import main; sys.exit(main(sys.argv[1:]))"


def run_cli(*args, **env):
    return subprocess.run([sys.executable, "-c", RUN, *args], capture_output=True, text=True,
                          env={**os.environ, **env})


def test_guard_failure_exits_2_without_path():
    inside = str(repo_root() / "outputs")
    result = run_cli("mock", OUTPUT_DIR=inside)
    assert result.returncode == 2
    assert "OUTPUT_DIR" in result.stderr
    assert inside not in result.stdout + result.stderr


def test_mock_command_writes_dataset():
    result = run_cli("mock", "--seed", "0", "--n-videos", "3")
    assert result.returncode == 0, result.stderr
    assert "videos: 3" in result.stdout
    assert os.environ["OUTPUT_DIR"] not in result.stdout + result.stderr
