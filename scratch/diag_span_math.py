"""Confirm the binding constraint: required clear span vs available gaps.

A new shift on a driver needs min_inter_shift_duration of separation on BOTH
sides.  Trailer continuity needs no separation but the trailer must be free.
Count, over all (route, pair) attempts, how many have ANY gap wide enough --
first on the driver, then on driver-and-trailer jointly.
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
    generate_chain_rescue_candidates,
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
routes = (generate_carryover_rescue_candidates(instance, solution, pressure, config=rescue)
          + generate_chain_rescue_candidates(instance, solution, pressure, config=rescue))
pairs = tuple((d.index, t) for d in instance.drivers for t in d.trailer_ids)

durations = []
fits_driver = fits_both = attempts = 0
for created in routes:
    trial = derive_solution(instance, Solution((replace(created, index=0),)))[0]
    duration = trial.end - created.start
    durations.append(duration)
    for driver_id, trailer_id in pairs:
        if not _route_allows_trailer(instance, created, trailer_id):
            continue
        attempts += 1
        driver = instance.drivers[driver_id]
        sep = driver.min_inter_shift_duration
        busy_d = sorted((s.start, d.end) for s, d in zip(solution.shifts, derived)
                        if s.driver == driver_id)
        busy_t = sorted((s.start, d.end) for s, d in zip(solution.shifts, derived)
                        if s.trailer == trailer_id)
        ok_d = ok_both = False
        for w in driver.time_windows:
            cursor = w.start
            for bs, be in busy_d:
                if be <= w.start or bs >= w.end:
                    continue
                if bs - cursor >= duration + (sep if cursor != w.start else 0):
                    ok_d = True
                    lo, hi = cursor, bs
                    if not any(ts < hi and lo < te for ts, te in busy_t):
                        ok_both = True
                cursor = max(cursor, be + sep)
            if w.end - cursor >= duration:
                ok_d = True
                if not any(ts < w.end and cursor < te for ts, te in busy_t):
                    ok_both = True
        fits_driver += int(ok_d)
        fits_both += int(ok_both)

durations.sort()
print(f"route_duration_min,{durations[0]},median,{durations[len(durations)//2]},max,{durations[-1]}")
print(f"(route,pair)_attempts,{attempts}")
print(f"fits_a_driver_gap,{fits_driver}")
print(f"fits_driver_AND_trailer_free,{fits_both}")
