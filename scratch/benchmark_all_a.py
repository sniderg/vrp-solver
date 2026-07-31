"""Benchmark Python vrp_solver (construct_solution + surgical_search) across all Set A V1 instances."""
import time
from pathlib import Path

from vrp_solver.contest import score_prefix_with_feasibility_tail
from vrp_solver.evaluate import run_checker
from vrp_solver.solver.greedy import construct_solution
from vrp_solver.solver.surgical_search import SurgicalSearchConfig, surgical_search
from vrp_solver.xml_io import load_instance, save_solution

SET_A_DIR = Path("roadef_2016_data/set_A_v1_1/Instances V1.1")
CHECKER_EXE = Path("roadef_2016_data/checker_v1_1/Checker V1 v1.1.0.0/Challenge_Roadef_EURO_Checker_V1/bin/Release/IRP_Roadef_Challenge_Checker.exe")
OUT_DIR = Path("scratch/set_a_benchmark")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEXALY_BENCHMARKS = {
    "V_1.1": {"customers": 12, "horizon": 30, "best_known": 0.027466, "hexaly": 0.027485},
    "V_1.2": {"customers": 12, "horizon": 30, "best_known": 0.027304, "hexaly": 0.027477},
    "V_1.3": {"customers": 53, "horizon": 10, "best_known": 0.013279, "hexaly": 0.013505},
    "V_1.4": {"customers": 64, "horizon": 10, "best_known": 0.015495, "hexaly": 0.015464},
    "V_1.5": {"customers": 54, "horizon": 10, "best_known": 0.011877, "hexaly": 0.011841},
    "V_1.6": {"customers": 54, "horizon": 35, "best_known": 0.012812, "hexaly": 0.012880},
    "V_1.7": {"customers": 99, "horizon": 10, "best_known": 0.012890, "hexaly": 0.012621},
    "V_1.8": {"customers": 99, "horizon": 3,  "best_known": 0.007756, "hexaly": 0.007756},
    "V_1.9": {"customers": 99, "horizon": 35, "best_known": 0.015279, "hexaly": 0.015815},
    "V_1.10": {"customers": 89, "horizon": 10, "best_known": 0.018941, "hexaly": 0.018371},
    "V_1.11": {"customers": 89, "horizon": 35, "best_known": 0.028666, "hexaly": 0.028957},
}

print(f"{'Instance':<8} | {'Cust':<4} | {'Days':<4} | {'Cold Seed LR':<12} | {'Surgical LR':<12} | {'Official Valid':<14} | {'Best Known':<10} | {'Time (s)':<8}")
print("-" * 95)

results = []
for inst_key, meta in HEXALY_BENCHMARKS.items():
    t0 = time.time()
    file_name = f"Instance_{inst_key}.xml"
    xml_path = SET_A_DIR / file_name
    if not xml_path.exists():
        print(f"Skipping {inst_key}: {xml_path} not found")
        continue

    inst = load_instance(xml_path)
    score_days = meta["horizon"]
    
    # 1. Cold Seed Constructor
    seed_sol, report = construct_solution(inst, safety_buffer=0.20)
    sc_seed = score_prefix_with_feasibility_tail(inst, seed_sol, score_days=score_days, feasibility_days=score_days, ignore_tail_call_ins=True)
    lr_seed = sc_seed.scored_estimated_cost / max(1.0, sc_seed.scored_delivered_quantity)

    # 2. Surgical Local Search (30 iterations)
    final_sol, steps = surgical_search(
        inst,
        seed_sol,
        config=SurgicalSearchConfig(
            end_day=score_days,
            iterations=30,
            candidates_per_move=8,
            no_improvement_limit=30,
            workers=4,
        ),
    )
    sc_final = score_prefix_with_feasibility_tail(inst, final_sol, score_days=score_days, feasibility_days=score_days, ignore_tail_call_ins=True)
    lr_final = sc_final.scored_estimated_cost / max(1.0, sc_final.scored_delivered_quantity)
    
    out_xml = OUT_DIR / f"{inst_key}_out.xml"
    save_solution(final_sol, out_xml)

    # 3. Official Mono Checker Validation
    official_valid, official_ratio, official_first_rule = run_checker(xml_path, out_xml, checker_exe=CHECKER_EXE)
    valid_str = f"VALID({official_ratio:.6f})" if official_valid else f"INVALID({official_first_rule or 'ERR'})"

    dt = time.time() - t0
    print(f"{inst_key:<8} | {meta['customers']:<4} | {score_days:<4} | {lr_seed:<12.6f} | {lr_final:<12.6f} | {valid_str:<14} | {meta['best_known']:<10.6f} | {dt:<8.2f}")
    results.append({
        "instance": inst_key,
        "seed_lr": lr_seed,
        "surgical_lr": lr_final,
        "official_valid": official_valid,
        "official_ratio": official_ratio,
        "best_known": meta["best_known"],
        "hexaly": meta["hexaly"],
        "time_s": dt,
    })

print("-" * 95)
valid_count = sum(1 for r in results if r["official_valid"])
print(f"Total benchmarked: {len(results)} | Valid by Official Mono Checker: {valid_count}/{len(results)}")
