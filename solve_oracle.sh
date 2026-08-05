#!/usr/bin/env bash
set -euo pipefail

# Usage: ./solve_oracle.sh <instance_xml_path> <output_solution_xml_path> [timeout_seconds] [seed] [iterations] [workers]
# Runs the supplied legacy Solver.exe only as a behavioral oracle.  Its output
# is checked by the released ROADEF V2 checker before it replaces OUT_PATH.

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <instance_xml_path> <output_solution_xml_path> [timeout_seconds] [seed] [iterations] [workers]"
    exit 1
fi

INST_PATH="$1"
OUT_PATH="$2"
TIME_LIMIT="${3:-300}"
SEED="${4:-1}"
ITERATIONS="${5:-0}"
WORKERS="${6:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORACLE_WINE_PREFIX="${WINEPREFIX:-/private/tmp/graydon-wine-prefix}"
# Keep uv independent of a possibly read-only home-directory cache.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRIPT_DIR/.uv-cache}"
mkdir -p "$UV_CACHE_DIR"

cd "$SCRIPT_DIR"
PENDING_PATH="$(mktemp "${OUT_PATH}.pending.XXXXXX")"
LOG_PATH="${PENDING_PATH}.oracle.log"

WINEPREFIX="$ORACLE_WINE_PREFIX" uv run --extra gurobi python scripts/run_solver_oracle.py \
    --instance "$INST_PATH" \
    --output "$PENDING_PATH" \
    --seed "$SEED" \
    --time-limit "$TIME_LIMIT" \
    --iterations "$ITERATIONS" \
    --workers "$WORKERS" \
    --id "$(basename "$OUT_PATH")" \
    --log "$LOG_PATH"

uv run --extra gurobi vrp-solver verify-official "$INST_PATH" "$PENDING_PATH"
mv "$PENDING_PATH" "$OUT_PATH"
mv "$LOG_PATH" "${OUT_PATH}.oracle.log"
echo "Accepted oracle solution: $OUT_PATH"
