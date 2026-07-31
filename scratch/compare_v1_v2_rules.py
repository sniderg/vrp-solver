"""Compare Set A instance evaluations under V1 rules vs V2 rules against Hexaly benchmarks."""
from pathlib import Path
from vrp_solver.xml_io import load_instance
from vrp_solver.solver.greedy import construct_solution
from vrp_solver.contest import score_prefix_with_feasibility_tail

V1_DIR = Path("roadef_2016_data/set_A_v1_1/Instances V1.1")
V2_DIR = Path("roadef_2016_data/set_A")

HEXALY_BENCHMARKS = {
    "V_1.1": {"horizon": 30, "hexaly_v1": 0.027485},
    "V_1.2": {"horizon": 30, "hexaly_v1": 0.027477},
    "V_1.3": {"horizon": 10, "hexaly_v1": 0.013505},
    "V_1.4": {"horizon": 10, "hexaly_v1": 0.015464},
    "V_1.5": {"horizon": 10, "hexaly_v1": 0.011841},
    "V_1.6": {"horizon": 35, "hexaly_v1": 0.012880},
    "V_1.7": {"horizon": 10, "hexaly_v1": 0.012621},
    "V_1.8": {"horizon": 3,  "hexaly_v1": 0.007756},
    "V_1.9": {"horizon": 35, "hexaly_v1": 0.015815},
    "V_1.10": {"horizon": 10, "hexaly_v1": 0.018371},
    "V_1.11": {"horizon": 35, "hexaly_v1": 0.028957},
}

print(f"{'Instance':<8} | {'Hexaly (V1)':<12} | {'V1 Raw Seed LR':<14} | {'V2 Seed LR':<14} | {'V2 / V1 Ratio':<14} | {'V2 / 2.0 (Norm)':<16}")
print("-" * 88)

for key, meta in HEXALY_BENCHMARKS.items():
    v1_path = V1_DIR / f"Instance_{key}.xml"
    v2_path = V2_DIR / f"Instance_{key}_ConvertedTo_V2.xml"
    
    if not v1_path.exists() or not v2_path.exists():
        continue
        
    inst_v1 = load_instance(v1_path)
    inst_v2 = load_instance(v2_path)
    
    # Build seed solution on V1 instance
    sol, _ = construct_solution(inst_v1, safety_buffer=0.20)
    
    # Score on V1 rules
    sc_v1 = score_prefix_with_feasibility_tail(inst_v1, sol, score_days=meta["horizon"], feasibility_days=meta["horizon"], ignore_tail_call_ins=True)
    lr_v1 = sc_v1.scored_estimated_cost / max(1.0, sc_v1.scored_delivered_quantity)
    
    # Score on V2 rules
    sc_v2 = score_prefix_with_feasibility_tail(inst_v2, sol, score_days=meta["horizon"], feasibility_days=meta["horizon"], ignore_tail_call_ins=True)
    lr_v2 = sc_v2.scored_estimated_cost / max(1.0, sc_v2.scored_delivered_quantity)
    
    ratio = lr_v2 / max(1e-9, lr_v1)
    v2_norm = lr_v2 / 2.0
    
    print(f"{key:<8} | {meta['hexaly_v1']:<12.6f} | {lr_v1:<14.6f} | {lr_v2:<14.6f} | {ratio:<14.2f}x | {v2_norm:<16.6f}")
