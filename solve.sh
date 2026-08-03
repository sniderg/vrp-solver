#!/usr/bin/env bash
set -e

# Usage: ./solve.sh <instance_xml_path> <output_solution_xml_path> [timeout_seconds]
#
# Examples:
#   ./solve.sh roadef_2016_data/set_B/Instances_B_V25-11042016/V2.12.xml native_V2.12.xml
#   ./solve.sh roadef_2016_data/set_B/Instances_B_V25-11042016/V2.25.xml native_V2.25.xml 300

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <instance_xml_path> <output_solution_xml_path> [timeout_seconds]"
    exit 1
fi

INST_PATH="$1"
OUT_PATH="$2"
TIME_LIMIT="${3:-300}"  # Default 300 seconds (5 minutes)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHONPATH="${SCRIPT_DIR}/src" python3 -c "
import sys
from vrp_solver.xml_io import load_instance, save_solution
from vrp_solver.numba_fast_solver import solve_numba_gurobi_mip

inst_path = sys.argv[1]
out_path = sys.argv[2]
time_limit = float(sys.argv[3])

print(f'🚀 Native Numba + Gurobi Solver starting on {inst_path} (Time Limit: {time_limit}s / {time_limit/60:.1f} mins)...')
inst = load_instance(inst_path)
sol = solve_numba_gurobi_mip(inst, n_samples_per_driver=500, time_limit_sec=time_limit)
save_solution(sol, out_path)
print(f'✅ Solution saved to {out_path} ({len(sol.shifts)} shifts)')
" "$INST_PATH" "$OUT_PATH" "$TIME_LIMIT"
