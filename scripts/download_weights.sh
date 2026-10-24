#!/bin/bash
# Download every 004 signal weight into the local weights cache and print sha256 values.
# Run on the UNF HPC login node (and once on the Mac). Never inside a Slurm job: jobs run offline.
# Usage: bash scripts/download_weights.sh
# Paste the printed sha256 values into config/pipeline.yaml (models.people.*.sha256,
# models.expression.model.sha256) and docs/weights.md. Checksums of public weights may be committed.
set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "refusing to download inside a Slurm job; run on the login node" >&2
    exit 2
fi

export HF_HOME="${HF_HOME:-$HOME/hf-cache}"
TPA_WEIGHTS_DIR="${TPA_WEIGHTS_DIR:-$HF_HOME/tpa-weights}"
YOLO_RELEASE="v8.3.0"                 # ultralytics/assets release tag
EMOTIEFFLIB_REF="main"                # pin to a commit hash for reproducible downloads; sha256 is the real pin

mkdir -p "$TPA_WEIGHTS_DIR"
sha() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

# Hugging Face (offline at run time via HF_HUB_OFFLINE=1)
for repo in Systran/faster-whisper-large-v3 Systran/faster-whisper-small \
            MIT/ast-finetuned-audioset-10-10-0.4593; do
    uv run hf download "$repo" >/dev/null
    echo "hf: $repo"
done
yunet=$(uv run hf download opencv/face_detection_yunet face_detection_yunet_2023mar.onnx)

# Official GitHub releases (not on Hugging Face)
for f in yolo11n.pt yolo11m.pt; do
    curl -fL --retry 3 -o "$TPA_WEIGHTS_DIR/$f" \
        "https://github.com/ultralytics/assets/releases/download/$YOLO_RELEASE/$f"
done
curl -fL --retry 3 -o "$TPA_WEIGHTS_DIR/enet_b0_8_va_mtl.onnx" \
    "https://github.com/sb-ai-lab/EmotiEffLib/raw/$EMOTIEFFLIB_REF/models/affectnet_emotions/onnx/enet_b0_8_va_mtl.onnx"

echo "sha256 (file  sha256):"
for f in yolo11n.pt yolo11m.pt enet_b0_8_va_mtl.onnx; do
    printf '%s  %s\n' "$f" "$(sha "$TPA_WEIGHTS_DIR/$f")"
done
printf '%s  %s\n' "face_detection_yunet_2023mar.onnx" "$(sha "$yunet")"
