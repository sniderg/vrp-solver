"""Do the failing host routes contain layovers?

The source comment states the generic timing MIP "deliberately models a
no-layover chain".  If nearly every candidate host route contains a layover,
try_optimize_shift_times must reject the insertion regardless of slack, which
would explain 187/191 insertions dying at the retiming stage.
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

tried = {"layover": [0, 0], "no_layover": [0, 0]}  # [attempts, successes]
for point in pressure[: config.pressure_customers]:
    customer = instance.customer_by_point[point.customer]
    for shift_pos, shift in enumerate(solution.shifts):
        if shift.start >= point.first_minute:
            continue
        if shift.trailer not in customer.allowed_trailers:
            continue
        key = "layover" if derived[shift_pos].layovers else "no_layover"
        for op_pos in range(1, len(shift.operations) + 1):
            available = derived[shift_pos].operations[op_pos - 1].trailer_quantity
            quantity = min(available, customer.capacity * 0.5, 20_000.0)
            if quantity < customer.min_operation_quantity:
                continue
            operations = list(shift.operations)
            anchor = operations[op_pos - 1]
            operations.insert(op_pos, Operation(customer.index, anchor.arrival, quantity))
            tried[key][0] += 1
            mutated = try_optimize_shift_times(
                instance,
                replace(shift, operations=tuple(operations)),
                latest_end=_resource_slot_end(instance, solution, derived, shift_pos),
            )
            if mutated is not None:
                tried[key][1] += 1

for key, (attempts, ok) in tried.items():
    rate = (100.0 * ok / attempts) if attempts else 0.0
    print(f"host_type,{key},insertions_attempted,{attempts},retiming_succeeded,{ok},rate_pct,{rate:.1f}")

n_lay = sum(1 for d in derived if d.layovers)
print(f"shifts_total,{len(derived)},shifts_with_layover,{n_lay}")
