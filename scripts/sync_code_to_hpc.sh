#!/bin/bash
# Push the repo (code and committed config only) to ~/tpa-work/repo on UNF HPC.
# Usage: scripts/sync_code_to_hpc.sh [--dry-run]
# Never sends data, .env, *.local.yaml, model weights, outputs or anything git ignores
# (scripts/sync.exclude plus `git ls-files -oi`). No --delete.
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

# everything git ignores stays behind too (git matches case-insensitively here, rsync does not)
GITIGNORED=$(mktemp)
trap 'rm -f "$GITIGNORED"' EXIT
{ git -C "$REPO" -c core.quotePath=false ls-files -oi --exclude-standard --directory 2>/dev/null || true; } \
    | sed 's|^|/|' > "$GITIGNORED"

rsync "${FLAGS[@]}" --exclude-from="$REPO/scripts/sync.exclude" --exclude-from="$GITIGNORED" "$REPO/" "$DEST"
