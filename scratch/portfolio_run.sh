#!/usr/bin/env bash
# Seed-portfolio + fast-engine experiment, 2026-08-20.
# Phase A: V2.26 seeds 2-9 (seed 1 reached 4 errors in the 08-19 sweep).
# Phase B: V2.12 seeds 2-9 (seed 1 plateaus at 119 errors).
# Phase C: fast-engine batch on the four open instances.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
INST_DIR=roadef_2016_data/set_B/Instances_B_V25-11042016
OUT=out/portfolio
mkdir -p "$OUT"

run_portfolio () {
  local inst=$1
  echo "=== phase $inst start $(date +%H:%M:%S) ==="
  for s in 2 3 4 5 6 7 8 9; do
    $PY -u -m vrp_solver.cli native-solve \
      "$INST_DIR/$inst.xml" "$OUT/${inst}_s${s}.xml" \
      --seed "$s" --time-limit 1800 --restart-rounds 8 \
      --no-improvement-limit 10000 \
      > "$OUT/${inst}_s${s}.log" 2>&1 &
  done
  wait
  for s in 2 3 4 5 6 7 8 9; do
    local errs verdict
    errs=$(grep "^local_errors," "$OUT/${inst}_s${s}.log" | tail -1 | cut -d, -f2)
    verdict=$($PY -m vrp_solver.cli verify-official \
      "$INST_DIR/$inst.xml" "$OUT/${inst}_s${s}.xml" 2>/dev/null \
      | grep -E "^official_valid,|^official_logistic_ratio," | tr '\n' ' ')
    echo "portfolio,$inst,seed,$s,local_errors,$errs,$verdict"
  done
}

run_portfolio V2.26
run_portfolio V2.12

echo "=== phase fast-batch start $(date +%H:%M:%S) ==="
$PY -u -m vrp_solver.cli native-solve-batch \
  "$INST_DIR" out/fast_open4 \
  --only V2.17 V2.18 V2.22 V2.23 \
  --engine fast --seed 1 --time-limit 1800 --concurrency 4 \
  --summary-csv out/fast_open4/summary.csv
echo "=== all phases done $(date +%H:%M:%S) ==="
