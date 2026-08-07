"""Can the HiGHS quantity repair close the residual breaches on its own?

The residual errors are safety breaches on tanks the operators already target,
so the question is whether the repair MIP is being asked to respect the safety
level at all.  Compare default repair against strict_inventory=True.
"""
import sys
from pathlib import Path

from vrp_solver.xml_io import load_instance, load_solution, save_solution
from vrp_solver.rules import validate_solution
from vrp_solver.highs_repair import repair_quantities_with_highs

instance = load_instance(Path(sys.argv[1]))
solution = load_solution(Path(sys.argv[2]))
out = sys.argv[3] if len(sys.argv) > 3 else None

def errs(s):
    v = validate_solution(instance, s)
    return sum(1 for e in v if e.severity == "error"), len(v)

print("baseline_errors,%d,total_violations,%d" % errs(solution))

for strict in (False, True):
    for objective in ("max-delivered", "min-cost"):
        try:
            repaired, report = repair_quantities_with_highs(
                instance,
                solution,
                score_days=10,
                feasibility_days=10,
                strict_inventory=strict,
                quantity_objective=objective,
                time_limit_seconds=600.0,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"strict,{strict},objective,{objective},EXCEPTION,{type(exc).__name__},{exc}")
            continue
        e, t = errs(repaired)
        print(f"strict,{strict},objective,{objective},status,{report.status},errors,{e},violations,{t}")
        if out and strict and e == 0:
            save_solution(repaired, Path(out))
            print(f"wrote,{out}")
