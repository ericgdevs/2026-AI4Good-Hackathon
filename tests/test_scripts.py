import subprocess

from tpa_index.paths import repo_root

PREDICT = repo_root() / "scripts/slurm/predict.sbatch"


def test_predict_sbatch_syntax_and_directives():
    assert subprocess.run(["bash", "-n", str(PREDICT)]).returncode == 0
    text = PREDICT.read_text()
    for directive in ("--partition=gpu", "--gres=gpu:6KSE:1", "--cpus-per-task=8", "--mem=64G",
                      "--time=02:00:00", "--output=slurm-%x-%j.out"):
        assert f"#SBATCH {directive}" in text, directive
    assert 'export HF_HOME="$HOME/hf-cache"' in text and "export HF_HUB_OFFLINE=1" in text
    assert "uv sync --frozen" in text and 'uv run tpa-index predict "$@"' in text
