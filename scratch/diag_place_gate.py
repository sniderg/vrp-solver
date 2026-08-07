"""Which condition inside the resource-placement gate rejects every route?

Stages: trailer compatibility, then _place_created_shift_in_resource_gap, which
can fail on (a) no containing customer window, (b) empty translation interval,
(c) no idle driver/trailer gap wide enough.
"""
import sys
from dataclasses import replace
from pathlib import Path

from vrp_solver.xml_io import load_instance, load_solution
from vrp_solver.model import Solution
from vrp_solver.rules import derive_solution
from vrp_solver.solver.pressure import pressure_points
from vrp_solver.solver.surgical_search import (
    SurgicalSearchConfig, _narrow_window_pressures, _reindex,
    _route_allows_trailer, _place_created_shift_in_resource_gap,
)
from vrp_solver.solver.targeted_rescue import (
    RescueConfig, generate_carryover_rescue_candidates,
    generate_chain_rescue_candidates,
)

instance = load_instance(Path(sys.argv[1]))
solution = load_solution(Path(sys.argv[2]))
config = SurgicalSearchConfig(end_day=10, candidates_per_move=120)
current_derived = derive_solution(instance, solution)

pressure = [i.customer for i in _narrow_window_pressures(
    instance, pressure_points(instance, solution, end_day=config.end_day))][: config.pressure_customers]
rescue = RescueConfig(start_day=0, end_day=config.end_day, replace_from_day=0,
    max_customers=config.pressure_customers, samples_per_customer=config.samples_per_customer,
    max_chain_length=3, nearest_chain_neighbors=5, target_fill_ratio=0.98)
routes = (generate_carryover_rescue_candidates(instance, solution, pressure, config=rescue)
          + generate_chain_rescue_candidates(instance, solution, pressure, config=rescue))
print(f"routes,{len(routes)}")

pairs = tuple((d.index, t) for d in instance.drivers for t in d.trailer_ids)
print(f"resource_pairs,{len(pairs)}")

trailer_ok = placed_ok = 0
routes_with_any_trailer = 0
for created in routes:
    any_trailer = False
    any_placed = False
    for driver_id, trailer_id in pairs:
        if not _route_allows_trailer(instance, created, trailer_id):
            continue
        any_trailer = True
        trailer_ok += 1
        if _place_created_shift_in_resource_gap(
            instance, solution, current_derived,
            replace(created, driver=driver_id, trailer=trailer_id),
        ) is not None:
            any_placed = True
            placed_ok += 1
            break
    routes_with_any_trailer += int(any_trailer)
    if any_placed:
        pass

print(f"routes_with_a_compatible_trailer,{routes_with_any_trailer}")
print(f"(route,pair)_trailer_compatible,{trailer_ok}")
print(f"(route,pair)_successfully_placed,{placed_ok}")
