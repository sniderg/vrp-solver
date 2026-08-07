"""Which stage discards every insert_operation mutation?

Replays the generator loop body, counting outcomes at each stage: quantity
floor, shift-time optimisation, and the resource-conflict repair.
"""
import sys
from dataclasses import replace
from pathlib import Path

from vrp_solver.xml_io import load_instance, load_solution
from vrp_solver.model import Operation, Solution
from vrp_solver.rules import derive_solution
from vrp_solver.highs_time_opt import try_optimize_shift_times
from vrp_solver.solver.pressure import pressure_points
from vrp_solver.solver.surgical_search import (
    SurgicalSearchConfig,
    _resource_slot_end,
    _repair_mutation_resource_conflicts,
    _minimum_quantity_candidates,
)

instance = load_instance(Path(sys.argv[1]))
solution = load_solution(Path(sys.argv[2]))
config = SurgicalSearchConfig(end_day=10, candidates_per_move=120)
pressure = pressure_points(instance, solution, end_day=config.end_day)
derived = derive_solution(instance, solution)

print("min_quantity_candidates,%d" % len(_minimum_quantity_candidates(instance, solution)))

pairs = qty_ok = timing_ok = 0
timing_fail_by_customer = {}
for point in pressure[: config.pressure_customers]:
    customer = instance.customer_by_point[point.customer]
    tf = 0
    for shift_pos, shift in enumerate(solution.shifts):
        if shift.start >= point.first_minute:
            continue
        if shift.trailer not in customer.allowed_trailers:
            continue
        for op_pos in range(1, len(shift.operations) + 1):
            pairs += 1
            available = derived[shift_pos].operations[op_pos - 1].trailer_quantity
            quantity = min(available, customer.capacity * 0.5, 20_000.0)
            if quantity < customer.min_operation_quantity:
                continue
            qty_ok += 1
            operations = list(shift.operations)
            anchor = operations[op_pos - 1]
            operations.insert(op_pos, Operation(customer.index, anchor.arrival, quantity))
            mutated = try_optimize_shift_times(
                instance,
                replace(shift, operations=tuple(operations)),
                latest_end=_resource_slot_end(instance, solution, derived, shift_pos),
            )
            if mutated is None:
                tf += 1
                continue
            timing_ok += 1
    timing_fail_by_customer[point.customer] = tf

print(f"candidate_pairs_considered,{pairs}")
print(f"passed_quantity_floor,{qty_ok}")
print(f"passed_shift_time_optimisation,{timing_ok}")
for c, tf in timing_fail_by_customer.items():
    print(f"timing_failures_customer_{c},{tf}")
