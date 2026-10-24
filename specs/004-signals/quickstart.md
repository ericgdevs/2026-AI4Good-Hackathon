# Quickstart: Extra Per-Segment Signals (004)

Only steps 2–3 belong to this feature's development work. Steps 1 and 4–6 are for the team on HPC after merge. This feature's own work never runs a stage on real data and never connects to HPC.

## 1. Pre-download and pin weights (HPC login node, once; Mac, once)

```bash
export HF_HOME=$HOME/hf-cache
uv run hf download Systran/faster-whisper-large-v3        # Mac: Systran/faster-whisper-small (already cached)
uv run hf download MIT/ast-finetuned-audioset-10-10-0.4593
uv run hf download opencv/face_detection_yunet face_detection_yunet_2023mar.onnx
mkdir -p $HF_HOME/tpa-weights && cd $HF_HOME/tpa-weights
# official GitHub releases (version-pinned): ultralytics/assets yolo11m.pt (Mac: yolo11n.pt),
# sb-ai-lab/EmotiEffLib models/affectnet_emotions/onnx/enet_b0_8_va_mtl.onnx
sha256sum yolo11m.pt enet_b0_8_va_mtl.onnx                # copy into models.people / models.expression sha256
```

Record each source, version and licence in `docs/weights.md`.

## 2. Develop and test on the Mac (mock data, fakes)

```bash
uv run pytest                                  # default: no model loaded, synthetic clips only
uv run pytest -m slow tests/test_signals_slow.py -k yunet   # one real model at a time, smallest sizes
```

## 3. Mock end-to-end on the Mac (scratch only)

```bash
export DATA_DIR=$PWD/../scratch/data OUTPUT_DIR=$PWD/../scratch/out TPA_SCRATCH_DIR=$PWD/../scratch/tmp
uv run tpa-index mock --clips
uv run tpa-index signals --stages people,expression --kind clip    # uses Mac models; synthetic clips have no faces
uv run tpa-index runs
```

## 4. HPC smoke test (one promo video, not listed as official)

```bash
sbatch scripts/slurm/signals.sbatch --stages transcript,audio --kind promo
```

If transcript fails with "could not load on cuda", set `models.transcript.hpc_device: cpu` (research R3).

### CUDA 12 libraries for faster-whisper (HPC)

The nodes have CUDA 13, but faster-whisper's CTranslate2 backend needs CUDA 12 cuBLAS and cuDNN 9. These come from
the Linux-only dependencies `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` (in `uv.lock`; nothing changes on the Mac).
`scripts/slurm/signals.sbatch` finds their library folders from the installed `nvidia.cublas.lib` and
`nvidia.cudnn.lib` packages and prepends them to `LD_LIBRARY_PATH` before running, so a manual export is not needed:

```bash
export LD_LIBRARY_PATH="$(uv run python -c 'import nvidia.cublas.lib as b, nvidia.cudnn.lib as d; print(b.__path__[0] + ":" + d.__path__[0])'):${LD_LIBRARY_PATH:-}"
```

If transcript still stops with "could not load on cuda: missing lib…", the error names the missing library.

## 5. HPC full run: two parallel jobs

```bash
sbatch scripts/slurm/signals.sbatch --stages transcript,audio
sbatch scripts/slurm/signals.sbatch --stages people,expression
# timed out? resubmit the same line: finished (video, stage) pairs are skipped
```

## 6. Make official, sync, calibrate

1. Add the run IDs to the new sections of `config/official_runs.yaml`, identically on HPC and in the repo.
2. Run `scripts/sync_outputs_from_hpc.sh`. Parquet comes down. `transcript.json` stays on HPC.
3. Run `uv run tpa-index calibrate --stage audio --split dev`, then the same with `--stage expression`.
