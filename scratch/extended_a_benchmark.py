"""Extended benchmark script: Runs multi-iteration surgical search across all 11 Set A V1 instances to push scores toward Hexaly best-known records."""
import sys
import time
from pathlib import Path

from vrp_solver.contest import score_prefix_with_feasibility_tail
from vrp_solver.evaluate import run_checker
from vrp_solver.solver.greedy import construct_solution
from vrp_solver.solver.surgical_search import SurgicalSearchConfig, surgical_search
from vrp_solver.xml_io import load_instance, load_solution, save_solution

SET_A_DIR = Path("roadef_2016_data/set_A_v1_1/Instances V1.1")
HUST_DIR = Path("roadef_2016_data/hust_smart_results")
CHECKER_EXE = Path("roadef_2016_data/checker_v1_1/Checker V1 v1.1.0.0/Challenge_Roadef_EURO_Checker_V1/bin/Release/IRP_Roadef_Challenge_Checker.exe")
OUT_DIR = Path("scratch/extended_a_run")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARKS = {
    "V_1.1":  {"horizon": 30, "best_known": 0.027466, "seed": "v1_1.1_cached_expand3_pruned_maxfill.xml"},
    "V_1.2":  {"horizon": 30, "best_known": 0.027304, "seed": "v1_1.2_improved.xml"},
    "V_1.3":  {"horizon": 10, "best_known": 0.013279, "seed": "v1_1.3_improved_squeezed.xml"},
    "V_1.4":  {"horizon": 10, "best_known": 0.015495, "seed": "v1_1.4_official_greedy.xml"},
    "V_1.5":  {"horizon": 10, "best_known": 0.011877, "seed": "v1_1.5_improved_squeezed.xml"},
    "V_1.6":  {"horizon": 35, "best_known": 0.012812, "seed": "v1_1.6_improved_squeezed.xml"},
    "V_1.7":  {"horizon": 10, "best_known": 0.012890, "seed": "v1_1.7_improved_squeezed.xml"},
    "V_1.8":  {"horizon": 3,  "best_known": 0.007756, "seed": "v1_1.8_improved_squeezed.xml"},
    "V_1.9":  {"horizon": 35, "best_known": 0.015279, "seed": "v1_1.9_rescued_feasible.xml"},
    "V_1.10": {"horizon": 10, "best_known": 0.018941, "seed": "v1_1.10_official_greedy.xml"},
    "V_1.11": {"horizon": 35, "best_known": 0.028666, "seed": "v1_1.11_rescued.xml"},
}

print(f"{'Instance':<8} | {'Starting LR':<12} | {'Improved LR':<12} | {'Official Checker':<16} | {'Best Known':<10} | {'Gap %':<8} | {'Time (s)':<8}", flush=True)
print("-" * 95, flush=True)

results = []
for inst_key, meta in BENCHMARKS.items():
    t0 = time.time()
    xml_path = SET_A_DIR / f"Instance_{inst_key}.xml"
    if not xml_path.exists():
        continue
    inst = load_instance(xml_path)
    score_days = meta["horizon"]
    
    # Load starting seed solution
    seed_xml = HUST_DIR / meta["seed"]
    if seed_xml.exists():
        seed_sol = load_solution(seed_xml)
    else:
        seed_sol, _ = construct_solution(inst, safety_buffer=0.20)
        
    sc_start = score_prefix_with_feasibility_tail(inst, seed_sol, score_days=score_days, feasibility_days=score_days, ignore_tail_call_ins=True)
    lr_start = sc_start.scored_estimated_cost / max(1.0, sc_start.scored_delivered_quantity)
    
    # Run extended surgical search (100 iterations per instance)
    improved_sol, steps = surgical_search(
        inst,
        seed_sol,
        config=SurgicalSearchConfig(
            end_day=score_days,
            iterations=100,
            candidates_per_move=8,
            no_improvement_limit=100,
            workers=4,
        ),
    )
    sc_final = score_prefix_with_feasibility_tail(inst, improved_sol, score_days=score_days, feasibility_days=score_days, ignore_tail_call_ins=True)
    lr_final = sc_final.scored_estimated_cost / max(1.0, sc_final.scored_delivered_quantity)
    
    out_xml = OUT_DIR / f"{inst_key}_extended.xml"
    save_solution(improved_sol, out_xml)
    
    valid, off_lr, first_err = run_checker(xml_path, out_xml, checker_exe=CHECKER_EXE)
    check_str = f"VALID({off_lr:.6f})" if valid else f"INVALID({first_err})"
    
    gap = ((lr_final - meta["best_known"]) / meta["best_known"]) * 100.0
    dt = time.time() - t0
    
    print(f"{inst_key:<8} | {lr_start:<12.6f} | {lr_final:<12.6f} | {check_str:<16} | {meta['best_known']:<10.6f} | {gap:<+7.2f}% | {dt:<8.2f}", flush=True)
    results.append({
        "instance": inst_key,
        "lr_start": lr_start,
        "lr_final": lr_final,
        "valid": valid,
        "best_known": meta["best_known"],
        "gap_percent": gap,
        "time": dt,
    })

print("-" * 95, flush=True)
valid_cnt = sum(1 for r in results if r["valid"])
avg_gap = sum(r["gap_percent"] for r in results if r["valid"]) / max(1, valid_cnt)
print(f"Completed {len(results)} instances | Valid by Official Checker: {valid_cnt}/{len(results)} | Average Gap to Best Known: {avg_gap:+.2f}%", flush=True)
