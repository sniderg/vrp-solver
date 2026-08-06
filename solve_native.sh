#!/usr/bin/env bash
set -euo pipefail

# Usage: ./solve_native.sh <instance_xml_path> <output_solution_xml_path> [timeout_seconds]
# The output is promoted only after the released ROADEF V2 checker accepts it.

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <instance_xml_path> <output_solution_xml_path> [timeout_seconds]"
    exit 1
fi

INST_PATH="$1"
OUT_PATH="$2"
TIME_LIMIT="${3:-300}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Some managed environments make uv's default home cache read-only.  Keep the
# entry point self-contained and permit callers to override the location.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRIPT_DIR/.uv-cache}"
mkdir -p "$UV_CACHE_DIR"

cd "$SCRIPT_DIR"
OUT_DIR="$(dirname "$OUT_PATH")"
mkdir -p "$OUT_DIR"
PENDING_PATH="$(mktemp "${OUT_PATH}.pending.XXXXXX")"
trap 'rm -f "$PENDING_PATH"' EXIT

# Cold starts use only the instance plus feature-derived policy.  Checkpoint
# restarts and their time slices are owned by native-solve, so no interactive
# or LLM-assisted continuation is needed.
echo "Native chain-first cold-start solver: $INST_PATH; target limit ${TIME_LIMIT}s"
uv run vrp-solver native-solve "$INST_PATH" "$PENDING_PATH" \
    --seed "${NATIVE_SEED:-0}" \
    --time-limit "$TIME_LIMIT" \
    --restart-rounds "${NATIVE_RESTART_ROUNDS:-2}" \
    --iterations "${NATIVE_ITERATIONS:-64}" \
    --candidates-per-move "${NATIVE_CANDIDATES_PER_MOVE:-120}" \
    --workers "${NATIVE_WORKERS:-2}"

uv run vrp-solver verify-official "$INST_PATH" "$PENDING_PATH"
mv "$PENDING_PATH" "$OUT_PATH"
echo "Accepted native solution: $OUT_PATH"
