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
    for pattern in (".env", ".env.*", ".venv/", "models/", "__pycache__/", "*.local.yaml", "*.mp4", "*.MOV",
                    "outputs/", "*.parquet", "*.duckdb"):
        assert pattern in rules, pattern
    assert not any("official_runs" in r for r in rules)
    assert not any(r.startswith("!") for r in rules)  # "!" clears rsync's exclude list


def test_code_sync_command(env_file, fake_rsync):
    env = {**os.environ, "PATH": fake_rsync["PATH"], "TPA_ENV_FILE": str(env_file)}
    result = _run(CODE_SYNC, env)
    assert result.returncode == 0, result.stderr
    (call,) = fake_rsync["log"].read_text().splitlines()
    assert call.count("--exclude-from=") == 2 and "--delete" not in call and " -n" not in call
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


@pytest.mark.parametrize("with_git", [True, False])
def test_code_sync_real_rsync_into_local_folder(tmp_path, env_file, with_git):
    repo, dest = tmp_path / "fakerepo", tmp_path / "dest"
    (repo / "scripts").mkdir(parents=True)
    for name in ("sync_code_to_hpc.sh", "sync.exclude"):
        shutil.copy(repo_root() / "scripts" / name, repo / "scripts" / name)
    if with_git:  # same .gitignore as the real repo; without git, sync.exclude alone must hold
        shutil.copy(repo_root() / ".gitignore", repo / ".gitignore")
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
    files = {"src/tpa_index/a.py": "x", "config/labels.yaml": "x", "config/official_runs.yaml": "predict: []",
             ".env.example": "x", ".env": "SECRET", ".env.local": "SECRET", ".venv/lib/x.py": "x",
             "models/m.pt": "x", "src/tpa_index/__pycache__/a.pyc": "x", "config/videos.local.yaml": "x",
             "config/thresholds.local.yaml": "x", "data/v.mp4": "x", "outputs/runs/r/s.parquet": "x",
             "clip.mov": "x", "slurm-tpa-1.out": "x", "TEST.MOV": "x", "clip.MP4": "x",
             "notes/thresholds.local.yaml": "x"}
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
                     "outputs/runs/r/s.parquet", "clip.mov", "slurm-tpa-1.out", "TEST.MOV", "clip.MP4",
                     "notes/thresholds.local.yaml"}
    assert not leaked, leaked
    if with_git:  # only git ignores this one (*token*): proves the git-ignored list is applied
        (repo / "api_token.txt").write_text("x")
        assert _run(repo / "scripts/sync_code_to_hpc.sh", env).returncode == 0
        assert not (dest / "api_token.txt").exists()


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


def test_output_sync_skips_transcript_json(tmp_path):
    """SC-011: transcripts stay on HPC; the output sync copies Parquet only (local folders, no real host)."""
    hpc_out, hpc_repo, local = tmp_path / "hpc-out", tmp_path / "hpc-repo", tmp_path / "local-out"
    run = hpc_out / "runs/transcript-20261024T000000-0001"
    run.mkdir(parents=True)
    (run / "segments.parquet").write_text("hpc")
    (run / "runs.parquet").write_text("hpc")
    (run / "transcript.json").write_text('{"segments": []}')
    local.mkdir()
    (hpc_repo / "config").mkdir(parents=True)
    env_path = tmp_path / "local.env"
    env_path.write_text(f"HPC_USER=mockuser\nHPC_HOST=hpc.invalid\nHPC_OUTPUT_DIR={hpc_out}\nHPC_REPO_DIR={hpc_repo}\n")
    env = {**os.environ, "TPA_ENV_FILE": str(env_path), "TPA_SYNC_REMOTE": "", "OUTPUT_DIR": str(local),
           "TPA_THRESHOLDS_FILE": str(tmp_path / "thresholds.local.yaml")}
    result = _run(OUT_SYNC, env)
    assert result.returncode == 0, result.stderr
    copied = local / "runs/transcript-20261024T000000-0001"
    assert (copied / "segments.parquet").exists() and (copied / "runs.parquet").exists()
    assert not list(local.rglob("transcript*.json"))


# ---- 004: signals job and weights download (stubbed tools; nothing downloaded, no GPU) ----------------------------
SIGNALS = repo_root() / "scripts/slurm/signals.sbatch"
WEIGHTS = repo_root() / "scripts/download_weights.sh"


def _stub_bin(tmp_path, **scripts):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in scripts.items():
        (bin_dir / name).write_text(f"#!/bin/bash\n{body}\n")
        (bin_dir / name).chmod(0o755)
    return bin_dir


def test_signals_sbatch_syntax_and_directives():
    assert subprocess.run(["bash", "-n", str(SIGNALS)]).returncode == 0
    text = SIGNALS.read_text()
    for directive in ("--partition=gpu", "--gres=gpu:6KSE:1", "--cpus-per-task=8", "--mem=64G",
                      "--time=03:00:00", "--output=slurm-%x-%j.out"):
        assert f"#SBATCH {directive}" in text, directive
    for line in ('export HF_HOME="$HOME/hf-cache"', "export HF_HUB_OFFLINE=1", "export YOLO_OFFLINE=1",
                 'export TPA_WEIGHTS_DIR="$HOME/hf-cache/tpa-weights"', "uv sync --frozen",
                 'uv run --frozen tpa-index signals "$@"'):
        assert line in text, line
    assert "--stages transcript,audio" in text and "--stages people,expression" in text


def test_signals_sbatch_prepends_cuda12_libs_and_passes_args(tmp_path):
    bin_dir = _stub_bin(tmp_path, **{
        "nvidia-smi": "echo GPU 0",
        "uv": 'case "$*" in *python*) echo "/w/nvidia/cublas/lib:/w/nvidia/cudnn/lib";; '
              '*tpa-index*) echo "LD=$LD_LIBRARY_PATH YOLO=$YOLO_OFFLINE W=$TPA_WEIGHTS_DIR ARGS=$*";; esac'})
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path), "SLURM_SUBMIT_DIR": str(tmp_path),
           "LD_LIBRARY_PATH": "/old"}
    result = subprocess.run(["bash", str(SIGNALS), "--stages", "people,expression", "--video-id", "v03"],
                            capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert "LD=/w/nvidia/cublas/lib:/w/nvidia/cudnn/lib:/old" in result.stdout
    assert f"YOLO=1 W={tmp_path}/hf-cache/tpa-weights" in result.stdout
    assert result.stdout.rstrip().endswith("signals --stages people,expression --video-id v03")


def test_download_weights_script_refuses_in_job(tmp_path):
    assert subprocess.run(["bash", "-n", str(WEIGHTS)]).returncode == 0
    marker = tmp_path / "called"
    bin_dir = _stub_bin(tmp_path, uv=f"touch {marker}", curl=f"touch {marker}")
    result = subprocess.run(["bash", str(WEIGHTS)], capture_output=True, text=True,
                            env={"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path), "SLURM_JOB_ID": "1"})
    assert result.returncode == 2 and "login node" in result.stderr
    assert not marker.exists()


def _weights_stubs(tmp_path, yunet_stdout):
    """uv/curl stand-ins: `hf download` prints coloured status like the real CLI; curl writes its -o file."""
    yunet = tmp_path / "hf/yunet.onnx"
    yunet.parent.mkdir(parents=True)
    yunet.write_bytes(b"yunet")
    hf_noise = r"printf '\033[32m✓ Downloaded\033[0m path: /somewhere\n'"
    return _stub_bin(tmp_path, uv=f'case "$*" in *python*) {yunet_stdout.format(yunet=yunet)};; *) {hf_noise};; esac',
                     curl='while [ $# -gt 0 ]; do [ "$1" = -o ] && { echo weights > "$2"; }; shift; done'), yunet


def test_download_weights_checksums_real_files(tmp_path):
    import hashlib

    bin_dir, yunet = _weights_stubs(tmp_path, "echo 'Fetching 1 file'; echo {yunet}")
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path), "TPA_WEIGHTS_DIR": str(tmp_path / "w")}
    result = subprocess.run(["bash", str(WEIGHTS)], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    weights = hashlib.sha256(b"weights\n").hexdigest()
    assert f"yolo11n.pt  {weights}" in result.stdout and f"enet_b0_8_va_mtl.onnx  {weights}" in result.stdout
    assert f"face_detection_yunet_2023mar.onnx  {hashlib.sha256(b'yunet').hexdigest()}" in result.stdout


def test_download_weights_stops_when_yunet_path_is_not_a_file(tmp_path):
    bin_dir, _ = _weights_stubs(tmp_path, r"printf '\033[32m✓ Downloaded\033[0m path: nope\n'")
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path), "TPA_WEIGHTS_DIR": str(tmp_path / "w")}
    result = subprocess.run(["bash", str(WEIGHTS)], capture_output=True, text=True, env=env)
    assert result.returncode == 1 and "did not return a file path" in result.stderr
    assert "sha256 (file" not in result.stdout
