"""Where does create_shift lose all its candidates?

Counts raw route candidates from each generator, then how many survive the
resource-placement gate.
"""
import sys
from pathlib import Path

from vrp_solver.xml_io import load_instance, load_solution
from vrp_solver.solver.pressure import pressure_points
from vrp_solver.solver.surgical_search import (
    SurgicalSearchConfig,
    _narrow_window_pressures,
    _resource_safe_created_candidates,
    _call_in_shift_candidates,
    _reindex,
)
from vrp_solver.solver.targeted_rescue import (
    RescueConfig,
    generate_rescue_candidates,
    generate_carryover_rescue_candidates,
    generate_chain_rescue_candidates,
)
from vrp_solver.model import Solution
from dataclasses import replace

instance = load_instance(Path(sys.argv[1]))
solution = load_solution(Path(sys.argv[2]))
config = SurgicalSearchConfig(end_day=10, candidates_per_move=120)

pressure = [
    item.customer for item in _narrow_window_pressures(
        instance, pressure_points(instance, solution, end_day=config.end_day)
    )
][: config.pressure_customers]
print(f"pressure_customers_passed_in,{len(pressure)},{pressure}")

rescue = RescueConfig(
    start_day=0, end_day=config.end_day, replace_from_day=0,
    max_customers=config.pressure_customers,
    samples_per_customer=config.samples_per_customer,
    max_chain_length=3, nearest_chain_neighbors=5, target_fill_ratio=0.98,
)
a = generate_rescue_candidates(instance, solution, pressure, config=rescue)
b = generate_carryover_rescue_candidates(instance, solution, pressure, config=rescue)
c = generate_chain_rescue_candidates(instance, solution, pressure, config=rescue)
print(f"generate_rescue_candidates,{len(a)}")
print(f"generate_carryover_rescue_candidates,{len(b)}")
print(f"generate_chain_rescue_candidates,{len(c)}")

shifts = a + b + c
ordinary = [
    _reindex(Solution((*solution.shifts, replace(s, index=len(solution.shifts)))))
    for s in shifts
]
call_in = _call_in_shift_candidates(instance, solution, config)
print(f"call_in_shift_candidates,{len(call_in)}")
print(f"raw_total_before_resource_gate,{len(call_in) + len(ordinary)}")
final = _resource_safe_created_candidates(instance, solution, call_in + ordinary, config)
print(f"after_resource_placement_gate,{len(final)}")
