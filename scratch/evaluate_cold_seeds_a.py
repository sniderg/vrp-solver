"""Quickly construct and evaluate cold seed solutions for all 11 Set A instances."""
from pathlib import Path
from vrp_solver.xml_io import load_instance, save_solution
from vrp_solver.solver.greedy import construct_solution
from vrp_solver.contest import score_prefix_with_feasibility_tail
from vrp_solver.evaluate import run_checker

SET_A_DIR = Path("roadef_2016_data/set_A_v1_1/Instances V1.1")
CHECKER_EXE = Path("roadef_2016_data/checker_v1_1/Checker V1 v1.1.0.0/Challenge_Roadef_EURO_Checker_V1/bin/Release/IRP_Roadef_Challenge_Checker.exe")
OUT_DIR = Path("scratch/set_a_seeds")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEXALY = {
    "V_1.1": 0.027466, "V_1.2": 0.027304, "V_1.3": 0.013279,
    "V_1.4": 0.015495, "V_1.5": 0.011877, "V_1.6": 0.012812,
    "V_1.7": 0.012890, "V_1.8": 0.007756, "V_1.9": 0.015279,
    "V_1.10": 0.018941, "V_1.11": 0.028666,
}

HORIZONS = {
    "V_1.1": 30, "V_1.2": 30, "V_1.3": 10, "V_1.4": 10,
    "V_1.5": 10, "V_1.6": 35, "V_1.7": 10, "V_1.8": 3,
    "V_1.9": 35, "V_1.10": 10, "V_1.11": 35,
}

print(f"{'Instance':<8} | {'Feasible':<10} | {'Errors':<8} | {'Hard':<6} | {'Cost':<10} | {'Delivered Qty':<14} | {'Seed LR':<12} | {'Official Valid':<16} | {'Best Known':<10}")
print("-" * 110)

for key, horizon in HORIZONS.items():
    xml_path = SET_A_DIR / f"Instance_{key}.xml"
    if not xml_path.exists():
        continue
    inst = load_instance(xml_path)
    sol, report = construct_solution(inst, safety_buffer=0.20)
    sc = score_prefix_with_feasibility_tail(inst, sol, score_days=horizon, feasibility_days=horizon, ignore_tail_call_ins=True)
    lr = sc.scored_estimated_cost / max(1.0, sc.scored_delivered_quantity)
    out_xml = OUT_DIR / f"{key}_seed.xml"
    save_solution(sol, out_xml)
    
    valid, off_lr, first_err = run_checker(xml_path, out_xml, checker_exe=CHECKER_EXE)
    off_str = f"VALID({off_lr:.6f})" if valid else f"INVALID({first_err})"
    
    print(f"{key:<8} | {str(sc.feasible):<10} | {sc.feasibility_errors:<8} | {sc.hard_violations:<6} | {sc.scored_estimated_cost:<10.2f} | {sc.scored_delivered_quantity:<14.2f} | {lr:<12.6f} | {off_str:<16} | {HEXALY[key]:<10.6f}")
