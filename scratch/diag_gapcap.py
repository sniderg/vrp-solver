"""Of the insertions that fail even with no resource-slot bound, how many are
blocked by the anti-second-layover gap cap?

The timing MIP caps every consecutive gap at layover_duration + travel + setup - 1
unless that index is the single represented layover split, and a split is only
enumerated when the PREVIOUS stop is a layover_customer.  A route whose stops are
all layover-ineligible therefore cannot absorb any long wait, so an inserted stop
that needs one is infeasible no matter how much slack the day has.
"""
import sys
from dataclasses import replace
from pathlib import Path

from vrp_solver.xml_io import load_instance, load_solution
from vrp_solver.model import Operation
from vrp_solver.rules import derive_solution
from vrp_solver.highs_time_opt import try_optimize_shift_times
from vrp_solver.solver.pressure import pressure_points
from vrp_solver.solver.surgical_search import (
    SurgicalSearchConfig, _resource_slot_end,
)

instance = load_instance(Path(sys.argv[1]))
solution = load_solution(Path(sys.argv[2]))
config = SurgicalSearchConfig(end_day=10, candidates_per_move=120)
pressure = pressure_points(instance, solution, end_day=config.end_day)
derived = derive_solution(instance, solution)

hard_fail = 0
splits_available = 0
needs_long_gap = 0
for point in pressure[: config.pressure_customers]:
    customer = instance.customer_by_point[point.customer]
    for shift_pos, shift in enumerate(solution.shifts):
        if shift.start >= point.first_minute or shift.trailer not in customer.allowed_trailers:
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
            if try_optimize_shift_times(instance, trial, latest_end=None) is not None:
                continue
            hard_fail += 1
            eligible = sum(
                1
                for i in range(1, len(trial.operations))
                if (c := instance.customer_by_point.get(trial.operations[i - 1].point))
                is not None and c.layover_customer
            )
            if eligible:
                splits_available += 1
            # Does the ORIGINAL route already contain a gap longer than the cap?
            driver = instance.drivers[shift.driver]
            d = derived[shift_pos]
            long_gap = False
            prev = None
            for i, o in enumerate(shift.operations):
                if prev is not None:
                    travel = instance.time_matrix[prev.point][o.point]
                    setup = instance.setup_time_for_point(prev.point)
                    if o.arrival - prev.arrival > driver.layover_duration + travel + setup - 1:
                        long_gap = True
                prev = o
            if long_gap:
                needs_long_gap += 1

print(f"insertions_infeasible_even_unbounded,{hard_fail}")
print(f"  had_at_least_one_legal_layover_split,{splits_available}")
print(f"  host_route_already_has_an_over_cap_gap,{needs_long_gap}")
