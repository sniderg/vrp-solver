"""Why do insert_operation / create_shift / multiroute generate zero candidates?

Counts, per pressure customer, how many (shift, position) pairs survive each
successive filter in _insert_operation_candidates.  Diagnostic only.
"""
import sys
from pathlib import Path

from vrp_solver.xml_io import load_instance, load_solution
from vrp_solver.solver.surgical_search import (
    SurgicalSearchConfig,
    pressure_points,
    _candidates,
)
from vrp_solver.rules import derive_solution

inst_p, sol_p = sys.argv[1], sys.argv[2]
instance = load_instance(Path(inst_p))
solution = load_solution(Path(sol_p))
config = SurgicalSearchConfig(end_day=instance.horizon, candidates_per_move=120)

pressure = pressure_points(instance, solution, end_day=config.end_day)
derived = derive_solution(instance, solution)
print(f"shifts,{len(solution.shifts)}")
print(f"pressure_points,{len(pressure)},pressure_customers_cap,{config.pressure_customers}")

n_no_shift_before = 0
n_trailer_blocked = 0
n_qty_too_small = 0
n_timing_failed = 0
n_ok = 0
for point in pressure[: config.pressure_customers]:
    customer = instance.customer_by_point[point.customer]
    before = [s for s in solution.shifts if s.start < point.first_minute]
    if not before:
        n_no_shift_before += 1
        continue
    allowed = [s for s in before if s.trailer in customer.allowed_trailers]
    if not allowed:
        n_trailer_blocked += 1
        continue
    ok_qty = False
    for shift_pos, shift in enumerate(solution.shifts):
        if shift.start >= point.first_minute:
            continue
        if shift.trailer not in customer.allowed_trailers:
            continue
        for op_pos in range(1, len(shift.operations) + 1):
            available = derived[shift_pos].operations[op_pos - 1].trailer_quantity
            quantity = min(available, customer.capacity * 0.5, 20_000.0)
            if quantity >= customer.min_operation_quantity:
                ok_qty = True
                break
        if ok_qty:
            break
    if not ok_qty:
        n_qty_too_small += 1
        continue
    n_ok += 1

print(f"customers_no_shift_starts_before_breach,{n_no_shift_before}")
print(f"customers_trailer_incompatible,{n_trailer_blocked}")
print(f"customers_quantity_below_min,{n_qty_too_small}")
print(f"customers_reaching_timing_stage,{n_ok}")

import random
for op in ("insert_operation", "create_shift", "multiroute_pressure_block"):
    cands = _candidates(instance, solution, op, config, random.Random(0))
    print(f"generated,{op},{len(cands)}")
