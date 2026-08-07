"""Separate the two ways try_optimize_shift_times can reject an insertion:

(a) _feasible_operation_windows returns empty for some operation -> early None,
    a structural/window impossibility;
(b) the MIP itself is infeasible -> the chain cannot be timed.

Also re-tests each failure with latest_end=None to see whether the resource-slot
bound is what binds.
"""
import sys
from dataclasses import replace
from pathlib import Path

from vrp_solver.xml_io import load_instance, load_solution
from vrp_solver.model import Operation
from vrp_solver.rules import derive_solution
from vrp_solver.highs_time_opt import (
    try_optimize_shift_times, _feasible_operation_windows,
)
from vrp_solver.solver.pressure import pressure_points
from vrp_solver.solver.surgical_search import (
    SurgicalSearchConfig, _resource_slot_end,
)

instance = load_instance(Path(sys.argv[1]))
solution = load_solution(Path(sys.argv[2]))
config = SurgicalSearchConfig(end_day=10, candidates_per_move=120)
pressure = pressure_points(instance, solution, end_day=config.end_day)
derived = derive_solution(instance, solution)

no_window = mip_infeasible = ok = ok_without_slot = 0
for point in pressure[: config.pressure_customers]:
    customer = instance.customer_by_point[point.customer]
    for shift_pos, shift in enumerate(solution.shifts):
        if shift.start >= point.first_minute:
            continue
        if shift.trailer not in customer.allowed_trailers:
            continue
        for op_pos in range(1, len(shift.operations) + 1):
            available = derived[shift_pos].operations[op_pos - 1].trailer_quantity
            quantity = min(available, customer.capacity * 0.5, 20_000.0)
            if quantity < customer.min_operation_quantity:
                continue
            operations = list(shift.operations)
            anchor = operations[op_pos - 1]
            operations.insert(op_pos, Operation(customer.index, anchor.arrival, quantity))
            trial = replace(shift, operations=tuple(operations))
            slot = _resource_slot_end(instance, solution, derived, shift_pos)

            if any(not _feasible_operation_windows(instance, o) for o in trial.operations):
                no_window += 1
                continue
            if try_optimize_shift_times(instance, trial, latest_end=slot) is not None:
                ok += 1
                continue
            mip_infeasible += 1
            if try_optimize_shift_times(instance, trial, latest_end=None) is not None:
                ok_without_slot += 1

print(f"rejected_no_feasible_window,{no_window}")
print(f"rejected_mip_infeasible,{mip_infeasible}")
print(f"  of_those_feasible_if_slot_bound_removed,{ok_without_slot}")
print(f"accepted,{ok}")
