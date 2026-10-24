#!/bin/bash
# Push the repo (code and committed config only) to ~/tpa-work/repo on UNF HPC.
# Usage: scripts/sync_code_to_hpc.sh [--dry-run]
# Never sends data, .env, config/*.local.yaml, model weights or outputs (see scripts/sync.exclude). No --delete.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
ENV_FILE=${TPA_ENV_FILE:-$REPO/.env}

env_value() {  # parse KEY=value without executing the file
    [ -f "$ENV_FILE" ] || return 0
    { grep -E "^$1=" "$ENV_FILE" || true; } | tail -n 1 | cut -d= -f2- | sed -e 's/^["'\'']//' -e 's/["'\'']$//'
}

HPC_USER=$(env_value HPC_USER)
HPC_HOST=$(env_value HPC_HOST)
for key in HPC_USER HPC_HOST; do
    if [ -z "${!key}" ]; then echo "missing $key in .env" >&2; exit 2; fi
done

FLAGS=(-az)
[ "${1:-}" = "--dry-run" ] && FLAGS+=(-n -v)
DEST=${TPA_SYNC_DEST:-"$HPC_USER@$HPC_HOST:~/tpa-work/repo/"}  # TPA_SYNC_DEST: local folder, for tests only

rsync "${FLAGS[@]}" --exclude-from="$REPO/scripts/sync.exclude" "$REPO/" "$DEST"
