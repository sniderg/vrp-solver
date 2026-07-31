"""Push extended surgical search and polishing on Set A instances to test minimal gap to Hexaly best-known scores."""
import time
from pathlib import Path

from vrp_solver.contest import score_prefix_with_feasibility_tail
from vrp_solver.evaluate import run_checker
from vrp_solver.solver.surgical_search import SurgicalSearchConfig, surgical_search
from vrp_solver.xml_io import load_instance, load_solution, save_solution

SET_A_DIR = Path("roadef_2016_data/set_A_v1_1/Instances V1.1")
HUST_DIR = Path("roadef_2016_data/hust_smart_results")
CHECKER_EXE = Path("roadef_2016_data/checker_v1_1/Checker V1 v1.1.0.0/Challenge_Roadef_EURO_Checker_V1/bin/Release/IRP_Roadef_Challenge_Checker.exe")
OUT_DIR = Path("scratch/push_hexaly_gaps")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    ("V_1.1", 30, 0.027466, "v1_1.1_cached_expand3_pruned_maxfill.xml"),
    ("V_1.2", 30, 0.027304, "v1_1.2_improved.xml"),
    ("V_1.5", 10, 0.011877, "v1_1.5_improved_squeezed.xml"),
    ("V_1.6", 35, 0.012812, "v1_1.6_improved_squeezed.xml"),
    ("V_1.9", 35, 0.015279, "v1_1.9_rescued_feasible.xml"),
]

print(f"{'Instance':<8} | {'Hexaly Best':<12} | {'Starting LR':<12} | {'Pushed LR':<12} | {'Gap %':<10} | {'Official Valid':<16} | {'Time (s)':<8}", flush=True)
print("-" * 90, flush=True)

for inst_key, days, best_known, seed_file in TARGETS:
    t0 = time.time()
    xml_path = SET_A_DIR / f"Instance_{inst_key}.xml"
    seed_path = HUST_DIR / seed_file
    
    inst = load_instance(xml_path)
    seed_sol = load_solution(seed_path)
    
    sc_start = score_prefix_with_feasibility_tail(inst, seed_sol, score_days=days, feasibility_days=days, ignore_tail_call_ins=True)
    lr_start = sc_start.scored_estimated_cost / max(1.0, sc_start.scored_delivered_quantity)
    
    # Run extended 500-iteration surgical search with 8 candidates per move
    final_sol, steps = surgical_search(
        inst,
        seed_sol,
        config=SurgicalSearchConfig(
            end_day=days,
            iterations=500,
            candidates_per_move=8,
            no_improvement_limit=500,
            workers=4,
        ),
    )
    
    sc_final = score_prefix_with_feasibility_tail(inst, final_sol, score_days=days, feasibility_days=days, ignore_tail_call_ins=True)
    lr_final = sc_final.scored_estimated_cost / max(1.0, sc_final.scored_delivered_quantity)
    
    out_xml = OUT_DIR / f"{inst_key}_pushed.xml"
    save_solution(final_sol, out_xml)
    
    valid, off_lr, first_err = run_checker(xml_path, out_xml, checker_exe=CHECKER_EXE)
    valid_str = f"VALID({off_lr:.6f})" if valid else f"INVALID({first_err})"
    
    gap = ((lr_final - best_known) / best_known) * 100.0
    dt = time.time() - t0
    
    print(f"{inst_key:<8} | {best_known:<12.6f} | {lr_start:<12.6f} | {lr_final:<12.6f} | {gap:<+9.2f}% | {valid_str:<16} | {dt:<8.2f}", flush=True)
