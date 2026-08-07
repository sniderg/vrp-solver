"""Can the timing MIP retime each EXISTING shift, unmodified?

If it cannot reproduce a route the constructor already built and the checker
accepts, then every mutation hosted by that route is rejected for a modelling
reason rather than a real infeasibility -- the operator is structurally blind to
those routes.
"""
import sys
from pathlib import Path

from vrp_solver.xml_io import load_instance, load_solution
from vrp_solver.rules import derive_solution
from vrp_solver.highs_time_opt import try_optimize_shift_times
from vrp_solver.solver.surgical_search import _resource_slot_end

instance = load_instance(Path(sys.argv[1]))
solution = load_solution(Path(sys.argv[2]))
derived = derive_solution(instance, solution)

ok = fail = ok_unbounded = 0
failed_idx = []
for pos, shift in enumerate(solution.shifts):
    slot = _resource_slot_end(instance, solution, derived, pos)
    if try_optimize_shift_times(instance, shift, latest_end=slot) is not None:
        ok += 1
        continue
    fail += 1
    failed_idx.append(pos)
    if try_optimize_shift_times(instance, shift, latest_end=None) is not None:
        ok_unbounded += 1

print(f"shifts,{len(solution.shifts)}")
print(f"retimed_successfully,{ok}")
print(f"MIP_CANNOT_REPRODUCE_OWN_ROUTE,{fail}")
print(f"  of_those_ok_without_resource_slot_bound,{ok_unbounded}")
print(f"failing_shift_indices,{failed_idx}")
for pos in failed_idx[:5]:
    s = solution.shifts[pos]
    print(f"  shift,{pos},driver,{s.driver},trailer,{s.trailer},start,{s.start},"
          f"end,{derived[pos].end},ops,{len(s.operations)},layovers,{derived[pos].layovers},"
          f"slot_end,{_resource_slot_end(instance, solution, derived, pos)}")
