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
PENDING_PATH="$(mktemp "${OUT_PATH}.pending.XXXXXX")"

# This deliberately avoids the old Gurobi/WLS path: cold starts must work
# without network access or a commercial-token service.  The constructor is
# stateful (source/customer/reload chains), unlike the obsolete one-load MIP.
# TIME_LIMIT is retained in the public interface for compatibility; retry
# count is bounded explicitly until a deadline-aware portfolio is added.
NATIVE_RETRIES="${NATIVE_RETRIES:-1}"
echo "Native stateful cold-start solver: $INST_PATH; target limit ${TIME_LIMIT}s; retries $NATIVE_RETRIES"
uv run vrp-solver paper-construct-solution "$INST_PATH" "$PENDING_PATH" \
    --seed "${NATIVE_SEED:-1}" \
    --retries "$NATIVE_RETRIES" \
    --selection-range "${NATIVE_SELECTION_RANGE:-1.5}" \
    --refill-coefficient "${NATIVE_REFILL_COEFFICIENT:-2.0}" \
    --candidate-pool-size "${NATIVE_CANDIDATE_POOL_SIZE:-64}" \
    --economic-urgency-minutes "${NATIVE_ECONOMIC_URGENCY_MINUTES:-0}"

uv run --extra gurobi vrp-solver verify-official "$INST_PATH" "$PENDING_PATH"
mv "$PENDING_PATH" "$OUT_PATH"
echo "Accepted native solution: $OUT_PATH"
