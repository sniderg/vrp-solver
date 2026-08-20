#!/usr/bin/env bash
# LR polish sweep, 2026-08-20: fast-search resume from each best valid
# artifact. The publication incumbent is ranked (errors, LR), so a valid
# seed can only be replaced by a valid-or-better state; the released
# checker still decides each output.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
INST_DIR=roadef_2016_data/set_B/Instances_B_V25-11042016
OUT=out/lr_polish
mkdir -p "$OUT"

# instance -> current best valid artifact (baseline LR in comment)
declare -A BEST=(
  [V2.12]=artifacts/valid/V2.12_native_repair.xml      # 0.027209
  [V2.13]=artifacts/valid/V2.13_deep_polished.xml      # 0.042045
  [V2.14]=artifacts/valid/V2.14_deep_polished.xml      # 0.075429
  [V2.15]=artifacts/valid/V2.15_cold_fast.xml          # 0.039686
  [V2.16.2]=artifacts/valid/V2.16.2_deep_polished.xml  # 0.025961
  [V2.19]=artifacts/valid/V2.19_deep_polished.xml      # 0.080677
  [V2.20.2]=artifacts/valid/V2.20.2_deep_polished.xml  # 0.031151
  [V2.21.2]=artifacts/valid/V2.21.2_cold.xml           # 0.032982
  [V2.24]=artifacts/valid/V2.24_deep_polished.xml      # 0.020387
  [V2.25]=artifacts/valid/V2.25_deep_polished.xml      # 0.027588
  [V2.26]=artifacts/valid/V2.26_deep_polished.xml      # 0.030957
)

run_wave () {
  for inst in "$@"; do
    $PY -u -m vrp_solver.cli native-solve \
      "$INST_DIR/$inst.xml" "$OUT/${inst}_lrpolish.xml" \
      --resume-from "${BEST[$inst]}" --engine fast \
      --seed 42 --time-limit 900 --restart-rounds 1 \
      --no-improvement-limit 10000 \
      > "$OUT/${inst}_lrpolish.log" 2>&1 &
  done
  wait
}

echo "=== wave 1 $(date +%H:%M:%S) ==="
run_wave V2.12 V2.13 V2.14 V2.15 V2.16.2 V2.19
echo "=== wave 2 $(date +%H:%M:%S) ==="
run_wave V2.20.2 V2.21.2 V2.24 V2.25 V2.26

echo "=== verify $(date +%H:%M:%S) ==="
for inst in "${!BEST[@]}"; do
  verdict=$($PY -m vrp_solver.cli verify-official \
    "$INST_DIR/$inst.xml" "$OUT/${inst}_lrpolish.xml" 2>/dev/null \
    | grep -E "^official_valid,|^official_logistic_ratio," | tr '\n' ' ')
  echo "lrpolish,$inst,$verdict"
done
echo "=== done $(date +%H:%M:%S) ==="
