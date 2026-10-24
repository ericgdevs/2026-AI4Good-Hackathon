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


# ---- sync scripts: never contact a host; fake rsync or real rsync into local temp folders only ----------------
import os  # noqa: E402
import shutil  # noqa: E402

import pytest  # noqa: E402
import yaml  # noqa: E402

CODE_SYNC = repo_root() / "scripts/sync_code_to_hpc.sh"
OUT_SYNC = repo_root() / "scripts/sync_outputs_from_hpc.sh"


@pytest.fixture
def env_file(tmp_path):
    path = tmp_path / "test.env"
    path.write_text("HPC_USER=mockuser\nHPC_HOST=hpc.invalid\nHPC_OUTPUT_DIR=/hpc/outputs\n")
    return path


@pytest.fixture
def fake_rsync(tmp_path):
    """rsync stand-in: logs its arguments; for the thresholds fetch, writes a fixture file."""
    bin_dir, log, fixture = tmp_path / "bin", tmp_path / "rsync.log", tmp_path / "hpc-thresholds.yaml"
    bin_dir.mkdir()
    fixture.write_text(yaml.safe_dump({"label_thresholds": {"hpc-model": {"singing": 0.12}},
                                       "idle_thresholds": {"hpc-model": 0.03}}))
    (bin_dir / "rsync").write_text(f"""#!/bin/bash
echo "$*" >> "{log}"
for last; do :; done
case "$*" in *thresholds.local.yaml*) cp "{fixture}" "$last" ;; esac
""")
    (bin_dir / "rsync").chmod(0o755)
    return {"PATH": f"{bin_dir}:{os.environ['PATH']}", "log": log}


def _run(script, env, *args):
    return subprocess.run(["bash", str(script), *args], capture_output=True, text=True, env=env, timeout=120)


def test_exclude_list_covers_secrets_data_and_weights():
    rules = [line.strip() for line in (repo_root() / "scripts/sync.exclude").read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    for pattern in (".env", ".env.*", ".venv/", "models/", "__pycache__/", "config/*.local.yaml", "*.mp4",
                    "outputs/", "*.parquet", "*.duckdb"):
        assert pattern in rules, pattern
    assert not any("official_runs" in r for r in rules)
    assert not any(r.startswith("!") for r in rules)  # "!" clears rsync's exclude list


def test_code_sync_command(env_file, fake_rsync):
    env = {**os.environ, "PATH": fake_rsync["PATH"], "TPA_ENV_FILE": str(env_file)}
    result = _run(CODE_SYNC, env)
    assert result.returncode == 0, result.stderr
    (call,) = fake_rsync["log"].read_text().splitlines()
    assert "--exclude-from=" in call and "--delete" not in call and " -n" not in call
    assert call.endswith("mockuser@hpc.invalid:~/tpa-work/repo/")
    _run(CODE_SYNC, env, "--dry-run")
    assert "-n -v" in fake_rsync["log"].read_text().splitlines()[-1]


def test_missing_key_stops_before_rsync(tmp_path, fake_rsync):
    env_path = tmp_path / "partial.env"
    env_path.write_text("HPC_USER=mockuser\n")
    env = {**os.environ, "PATH": fake_rsync["PATH"], "TPA_ENV_FILE": str(env_path)}
    for script in (CODE_SYNC, OUT_SYNC):
        result = _run(script, env)
        assert result.returncode == 2 and "missing HPC_HOST" in result.stderr
    assert not fake_rsync["log"].exists()


def test_output_sync_command_and_guard(env_file, fake_rsync, tmp_path):
    target = tmp_path / "thresholds.local.yaml"
    env = {**os.environ, "PATH": fake_rsync["PATH"], "TPA_ENV_FILE": str(env_file),
           "TPA_THRESHOLDS_FILE": str(target)}
    inside = _run(OUT_SYNC, {**env, "OUTPUT_DIR": str(repo_root() / "outputs")})
    assert inside.returncode == 2 and "data guard" in inside.stderr and not fake_rsync["log"].exists()
    result = _run(OUT_SYNC, env)
    assert result.returncode == 0, result.stderr
    runs_call = fake_rsync["log"].read_text().splitlines()[0]
    assert "--ignore-existing" in runs_call and "--exclude=reviews.duckdb" in runs_call
    assert "mockuser@hpc.invalid:/hpc/outputs/runs/" in runs_call
    fetch = fake_rsync["log"].read_text().splitlines()[1]
    assert "mockuser@hpc.invalid:~/tpa-work/repo/config/thresholds.local.yaml" in fetch  # ~ left for HPC
    assert yaml.safe_load(target.read_text())["idle_thresholds"] == {"hpc-model": 0.03}


def test_output_sync_dry_run_writes_nothing(env_file, fake_rsync, tmp_path):
    target = tmp_path / "thresholds.local.yaml"
    env = {**os.environ, "PATH": fake_rsync["PATH"], "TPA_ENV_FILE": str(env_file),
           "TPA_THRESHOLDS_FILE": str(target)}
    result = _run(OUT_SYNC, env, "--dry-run")
    assert result.returncode == 0 and "would merge thresholds for models: hpc-model" in result.stdout
    assert "-n -v" in fake_rsync["log"].read_text().splitlines()[0] and not target.exists()


def test_code_sync_real_rsync_into_local_folder(tmp_path, env_file):
    repo, dest = tmp_path / "fakerepo", tmp_path / "dest"
    (repo / "scripts").mkdir(parents=True)
    for name in ("sync_code_to_hpc.sh", "sync.exclude"):
        shutil.copy(repo_root() / "scripts" / name, repo / "scripts" / name)
    files = {"src/tpa_index/a.py": "x", "config/labels.yaml": "x", "config/official_runs.yaml": "predict: []",
             ".env.example": "x", ".env": "SECRET", ".env.local": "SECRET", ".venv/lib/x.py": "x",
             "models/m.pt": "x", "src/tpa_index/__pycache__/a.pyc": "x", "config/videos.local.yaml": "x",
             "config/thresholds.local.yaml": "x", "data/v.mp4": "x", "outputs/runs/r/s.parquet": "x",
             "clip.mov": "x", "slurm-tpa-1.out": "x"}
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    env = {**os.environ, "TPA_ENV_FILE": str(env_file), "TPA_SYNC_DEST": f"{dest}/"}
    result = _run(repo / "scripts/sync_code_to_hpc.sh", env)
    assert result.returncode == 0, result.stderr
    sent = {str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()}
    assert {"src/tpa_index/a.py", "config/labels.yaml", "config/official_runs.yaml", ".env.example"} <= sent
    leaked = sent & {".env", ".env.local", ".venv/lib/x.py", "models/m.pt", "src/tpa_index/__pycache__/a.pyc",
                     "config/videos.local.yaml", "config/thresholds.local.yaml", "data/v.mp4",
                     "outputs/runs/r/s.parquet", "clip.mov", "slurm-tpa-1.out"}
    assert not leaked, leaked


def test_output_sync_real_rsync_from_local_folder(tmp_path):
    hpc_out, hpc_repo, local = tmp_path / "hpc-out", tmp_path / "hpc-repo", tmp_path / "local-out"
    for rel, text in {"runs/r1/segments.parquet": "hpc", "runs/r2/segments.parquet": "hpc",
                      "runs/r2/clip.mp4": "x", "runs/r2/reviews.duckdb": "x"}.items():
        (hpc_out / rel).parent.mkdir(parents=True, exist_ok=True)
        (hpc_out / rel).write_text(text)
    (hpc_out / "reviews.duckdb").write_text("hpc reviews")
    (local / "runs/r1").mkdir(parents=True)
    (local / "runs/r1/segments.parquet").write_text("local")
    (local / "reviews.duckdb").write_text("local reviews")
    (hpc_repo / "config").mkdir(parents=True)
    (hpc_repo / "config/thresholds.local.yaml").write_text(
        yaml.safe_dump({"label_thresholds": {"hpc-model": {"singing": 0.12}}}))
    target = tmp_path / "thresholds.local.yaml"
    target.write_text(yaml.safe_dump({"label_thresholds": {"mac-model": {"singing": 0.2}},
                                      "idle_thresholds": {"mac-model": 0.3}}))
    env_path = tmp_path / "local.env"
    env_path.write_text(f"HPC_USER=mockuser\nHPC_HOST=hpc.invalid\nHPC_OUTPUT_DIR={hpc_out}\nHPC_REPO_DIR={hpc_repo}\n")
    env = {**os.environ, "TPA_ENV_FILE": str(env_path), "TPA_SYNC_REMOTE": "", "OUTPUT_DIR": str(local),
           "TPA_THRESHOLDS_FILE": str(target)}
    result = _run(OUT_SYNC, env)
    assert result.returncode == 0, result.stderr
    assert (local / "runs/r1/segments.parquet").read_text() == "local"  # existing run never overwritten
    assert (local / "runs/r2/segments.parquet").read_text() == "hpc"  # new run copied
    assert not (local / "runs/r2/clip.mp4").exists() and not (local / "runs/r2/reviews.duckdb").exists()
    assert (local / "reviews.duckdb").read_text() == "local reviews"  # review log never touched
    data = yaml.safe_load(target.read_text())
    assert data["label_thresholds"] == {"mac-model": {"singing": 0.2}, "hpc-model": {"singing": 0.12}}
    assert data["idle_thresholds"] == {"mac-model": 0.3}
