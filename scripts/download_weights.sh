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
YUNET_REVISION="3cc26e7f1014a5ee5d74a42acee58bafc9d0a310"   # HF snapshot, as in config/pipeline.yaml

mkdir -p "$TPA_WEIGHTS_DIR"
sha() {
    [[ -f "$1" ]] || { echo "not a file, cannot checksum: $1" >&2; exit 1; }
    if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1
}

# Hugging Face (offline at run time via HF_HUB_OFFLINE=1)
for repo in Systran/faster-whisper-large-v3 Systran/faster-whisper-small \
            MIT/ast-finetuned-audioset-10-10-0.4593; do
    uv run hf download "$repo" >/dev/null
    echo "hf: $repo"
done
# faster-whisper needs CUDA 12 cuBLAS and cuDNN next to torch's CUDA 13 ones. Both cuDNN wheels write
# nvidia/cudnn/lib/libcudnn.so.9, so install the cu12 pair last: the venv state that was verified on the node.
if [[ "$(uname)" == "Linux" ]]; then
    uv sync --frozen --reinstall-package nvidia-cublas-cu12 --reinstall-package nvidia-cudnn-cu12
fi

# the path only: `hf download` prints coloured status text, so ask huggingface_hub for the file path directly
yunet=$(uv run python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('opencv/face_detection_yunet', 'face_detection_yunet_2023mar.onnx', revision='$YUNET_REVISION'))" | tail -n 1)
[[ -f "$yunet" ]] || { echo "YuNet download did not return a file path" >&2; exit 1; }

# Official GitHub releases (not on Hugging Face)
for f in yolo11n.pt yolo11m.pt; do
    curl -fL --retry 3 -o "$TPA_WEIGHTS_DIR/$f" \
        "https://github.com/ultralytics/assets/releases/download/$YOLO_RELEASE/$f"
done
curl -fL --retry 3 -o "$TPA_WEIGHTS_DIR/enet_b0_8_va_mtl.onnx" \
    "https://github.com/sb-ai-lab/EmotiEffLib/raw/$EMOTIEFFLIB_REF/models/affectnet_emotions/onnx/enet_b0_8_va_mtl.onnx"

echo "sha256 (file  sha256):"
for f in yolo11n.pt yolo11m.pt enet_b0_8_va_mtl.onnx; do
    h=$(sha "$TPA_WEIGHTS_DIR/$f")   # an assignment, so set -e stops on a failed checksum
    printf '%s  %s\n' "$f" "$h"
done
h=$(sha "$yunet")
printf '%s  %s  (revision %s)\n' "face_detection_yunet_2023mar.onnx" "$h" "$YUNET_REVISION"
