"""For one representative route, print exactly why every start overlaps.

Shows the translation interval, the driver's windows, the existing commitments on
that driver and trailer, and the route duration.  With min_inter_shift_duration
720 on every driver, a new shift needs a 720+duration+720 clear span.
"""
import sys
from dataclasses import replace
from pathlib import Path

from vrp_solver.xml_io import load_instance, load_solution
from vrp_solver.model import Solution
from vrp_solver.rules import derive_solution
from vrp_solver.solver.pressure import pressure_points
from vrp_solver.solver.surgical_search import (
    SurgicalSearchConfig, _narrow_window_pressures, _route_allows_trailer,
)
from vrp_solver.solver.targeted_rescue import (
    RescueConfig, generate_carryover_rescue_candidates,
)

instance = load_instance(Path(sys.argv[1]))
solution = load_solution(Path(sys.argv[2]))
config = SurgicalSearchConfig(end_day=10, candidates_per_move=120)
derived = derive_solution(instance, solution)
pressure = [i.customer for i in _narrow_window_pressures(
    instance, pressure_points(instance, solution, end_day=config.end_day))][: config.pressure_customers]
rescue = RescueConfig(start_day=0, end_day=config.end_day, replace_from_day=0,
    max_customers=config.pressure_customers, samples_per_customer=config.samples_per_customer,
    max_chain_length=3, nearest_chain_neighbors=5, target_fill_ratio=0.98)
routes = generate_carryover_rescue_candidates(instance, solution, pressure, config=rescue)

created = routes[0]
print(f"route_points,{[o.point for o in created.operations]}")
trial = derive_solution(instance, Solution((replace(created, index=0),)))[0]
duration = trial.end - created.start
print(f"route_start,{created.start},route_end,{trial.end},duration,{duration}")

for driver in instance.drivers:
    for trailer_id in driver.trailer_ids:
        if not _route_allows_trailer(instance, created, trailer_id):
            continue
        busy_d = sorted((s.start, d.end) for s, d in zip(solution.shifts, derived)
                        if s.driver == driver.index)
        busy_t = sorted((s.start, d.end) for s, d in zip(solution.shifts, derived)
                        if s.trailer == trailer_id)
        print(f"\ndriver,{driver.index},trailer,{trailer_id},"
              f"min_inter_shift,{driver.min_inter_shift_duration}")
        print(f"  windows,{[(w.start, w.end) for w in driver.time_windows]}")
        print(f"  driver_busy,{busy_d}")
        print(f"  trailer_busy,{busy_t}")
        # Required clear span for a legal insertion between two driver shifts.
        need = duration + 2 * driver.min_inter_shift_duration
        print(f"  required_clear_span_between_driver_shifts,{need}")
        gaps = []
        for i in range(len(busy_d) - 1):
            gaps.append(busy_d[i + 1][0] - busy_d[i][1])
        print(f"  actual_driver_inter_shift_gaps,{sorted(gaps)}")
