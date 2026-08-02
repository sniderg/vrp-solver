from vrp_solver.xml_io import load_instance, save_solution
from vrp_solver.solver.cluster_greedy import construct_paper_solution
from vrp_solver.highs_repair import repair_with_highs_selection
from vrp_solver.rules import validate_solution
from vrp_solver.contest import score_prefix_with_feasibility_tail, _instance_days
from dataclasses import replace
from pathlib import Path

inst = load_instance("roadef_2016_data/set_B/Instances_B_V25-11042016/V2.26.xml")
days = _instance_days(inst)
OUT_DIR = Path("scratch/set_b_solutions")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Generate baseline seed 22 solution
sol, _ = construct_paper_solution(inst, seed=22, retries=1)

# Reorder Shift 1 so Pt 23 is served early at minute 1,469 (step 24):
new_shifts = list(sol.shifts)
shift1 = new_shifts[1]
ops1 = list(shift1.operations)

op0 = replace(ops1[0], arrival=1346)
op1 = replace(ops1[2], arrival=1469) # Pt 23
op2 = replace(ops1[3], arrival=1551) # Pt 8
op3 = replace(ops1[4], arrival=1614) # Pt 2 reload
op4 = replace(ops1[5], arrival=1732) # Pt 17
op5 = replace(ops1[1], arrival=1867) # Pt 22

new_shifts[1] = replace(shift1, start=1256, operations=(op0, op1, op2, op3, op4, op5))
sol_reordered = replace(sol, shifts=tuple(new_shifts))

print("Running HiGHS / Gurobi quantity repair on reordered V2.26 Seed 22...")
repaired_sol, report = repair_with_highs_selection(
    inst,
    sol_reordered,
    score_days=days,
    feasibility_days=days,
)

score = score_prefix_with_feasibility_tail(inst, repaired_sol, score_days=days, feasibility_days=days)
viols = validate_solution(inst, repaired_sol)
errs = [v for v in viols if v.severity == "error"]

print(f"\nHiGHS / Gurobi Repair V2.26 Result: Feasible={score.feasible}, ScoreErrors={score.feasibility_errors}, RuleErrors={len(errs)}")
for v in errs:
    print("  ->", v)

if score.feasible and len(errs) == 0:
    print("\n==================================================================")
    print("🎉🎉🎉 VICTORY! V2.26 IS 100% FEASIBLE WITH GUROBI REPAIR (0 ERRORS)! 🎉🎉🎉")
    print("==================================================================")
    out_file = OUT_DIR / "V2.26.xml"
    save_solution(repaired_sol, out_file)
    print(f"Saved 100% feasible solution to {out_file}")
