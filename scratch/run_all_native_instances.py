"""Run Native Numba + Gurobi Solver across all Set B and Set X instances.

This script executes solve_numba_gurobi_mip on all 15 Set B instances (V2.12.xml to V2.26.xml)
and 5 Set X instances (X1.xml to X5.xml), saving 100% native XML solutions to scratch/native_solutions/
and running the C++ checker binary (check_solution) on each.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

from vrp_solver.xml_io import load_instance, save_solution
from vrp_solver.numba_fast_solver import solve_numba_gurobi_mip

DATA_DIR = Path("roadef_2016_data")
OUT_DIR = Path("scratch/native_solutions")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Discover all Set B and Set X instance files (excluding macOS resource fork ._* files)
all_xmls = sorted(list(DATA_DIR.glob("**/*.xml")))
instances = [p for p in all_xmls if not p.name.startswith("._") and ("set_B" in str(p) or "set_X" in str(p))]

print(f"Found {len(instances)} valid Set B and Set X instance files:")
for inst_p in instances:
    print(f"  - {inst_p.name} ({inst_p})")

results = []

for inst_path in instances:
    inst_name = inst_path.stem
    out_xml_path = OUT_DIR / f"{inst_name}_native.xml"
    print(f"\n==========================================")
    print(f"🚀 Running Native Solver on {inst_name} ({inst_path})")
    print(f"==========================================")
    
    t0 = time.time()
    try:
        inst = load_instance(str(inst_path))
        sol = solve_numba_gurobi_mip(inst, n_samples_per_driver=500, time_limit_sec=120.0)
        elapsed = time.time() - t0
        
        save_solution(sol, str(out_xml_path))
        print(f"✅ Saved native solution to {out_xml_path} ({len(sol.shifts)} shifts, elapsed {elapsed:.2f}s)")
        
        # Verify with C++ checker
        checker_cmd = ["python3", "-m", "vrp_solver.cli", "check", str(inst_path), str(out_xml_path)]
        res = subprocess.run(checker_cmd, capture_output=True, text=True)
        
        checker_passed = ("SOLUTION OK" in res.stdout or "CHECKING SUCCESSFUL" in res.stdout) and "CHECKING FAILED" not in res.stdout
        status_str = "PASSED" if checker_passed else "CHECKER_ISSUES"
        print(f"Checker Status: {status_str}")
        
        results.append((inst_name, len(sol.shifts), elapsed, status_str))
    except Exception as e:
        print(f"❌ Failed on {inst_name}: {e}")
        results.append((inst_name, 0, time.time() - t0, f"FAILED: {e}"))

print("\n\n==========================================")
print("📊 BATCH NATIVE SOLVER SUMMARY REPORT")
print("==========================================")
for name, shifts, elapsed, status in results:
    print(f"  {name:12s} | Shifts: {shifts:3d} | Time: {elapsed:6.1f}s | Status: {status}")
