#!/bin/bash
# Pull finished run folders from UNF HPC into the local OUTPUT_DIR, then merge HPC's calibrated thresholds.
# Usage: scripts/sync_outputs_from_hpc.sh [--dry-run]
# Copies only run folders' Parquet files; never overwrites local runs, never copies reviews.duckdb or media.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
ENV_FILE=${TPA_ENV_FILE:-$REPO/.env}

env_value() {  # parse KEY=value without executing the file
    [ -f "$ENV_FILE" ] || return 0
    { grep -E "^$1=" "$ENV_FILE" || true; } | tail -n 1 | cut -d= -f2- | sed -e 's/^["'\'']//' -e 's/["'\'']$//'
}

HPC_USER=$(env_value HPC_USER)
HPC_HOST=$(env_value HPC_HOST)
HPC_OUTPUT_DIR=$(env_value HPC_OUTPUT_DIR)
HPC_REPO_DIR=$(env_value HPC_REPO_DIR)
HPC_REPO_DIR=${HPC_REPO_DIR:-'~/tpa-work/repo'}  # quoted: ~ must expand on HPC, not here
for key in HPC_USER HPC_HOST HPC_OUTPUT_DIR; do
    if [ -z "${!key}" ]; then echo "missing $key in .env" >&2; exit 2; fi
done

# local OUTPUT_DIR through the data guard (must be outside the repo)
if ! LOCAL=$(uv run --project "$REPO" python -c "from tpa_index.paths import output_dir; print(output_dir())" 2>/dev/null); then
    echo "local OUTPUT_DIR fails the data guard (unset or inside the repo)" >&2
    exit 2
fi

DRY=()
[ "${1:-}" = "--dry-run" ] && DRY=(-n -v)
REMOTE=${TPA_SYNC_REMOTE-$HPC_USER@$HPC_HOST:}  # TPA_SYNC_REMOTE="" means a local source, for tests only

mkdir -p "$LOCAL/runs"
rsync -az ${DRY[@]+"${DRY[@]}"} --ignore-existing --exclude=reviews.duckdb \
    --include='*/' --include='*.parquet' --exclude='*' \
    "$REMOTE$HPC_OUTPUT_DIR/runs/" "$LOCAL/runs/"

# thresholds are keyed by model: HPC's models replace their own blocks, the Mac's other models stay
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT
if rsync -az "$REMOTE$HPC_REPO_DIR/config/thresholds.local.yaml" "$TMP" 2>/dev/null && [ -s "$TMP" ]; then
    TARGET=${TPA_THRESHOLDS_FILE:-$REPO/config/thresholds.local.yaml}  # TPA_THRESHOLDS_FILE: for tests only
    uv run --project "$REPO" python - "$TMP" "$TARGET" "${1:-}" <<'PY'
import sys
from pathlib import Path

import yaml

src, target, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
incoming = yaml.safe_load(src.read_text()) or {}
merged = (yaml.safe_load(target.read_text()) or {}) if target.exists() else {}
models = sorted(set(incoming.get("label_thresholds", {})) | set(incoming.get("idle_thresholds", {})))
if mode == "--dry-run":
    print("would merge thresholds for models:", ", ".join(models) or "none")
    sys.exit(0)
for key in ("label_thresholds", "idle_thresholds"):
    merged.setdefault(key, {}).update(incoming.get(key) or {})
target.write_text("# written by tpa-index calibrate; gitignored, data-derived\n" + yaml.safe_dump(merged, sort_keys=True))
print("merged thresholds for models:", ", ".join(models) or "none")
PY
else
    echo "no thresholds file on HPC; local thresholds unchanged"
fi
