"""Prototype: does adding idle-gap starts unblock the placement gate?

The gate only tries start = int(low) plus the ends of shifts already on the same
driver/trailer.  If `low` lands inside a busy interval and no shift end falls in
[low, high], exactly one start is tried, it overlaps, and the route is discarded
even though a wide idle gap exists later.  Add each resource's idle-gap start as
a candidate and compare placement rates.
"""
import sys
from dataclasses import replace
from pathlib import Path

from vrp_solver.xml_io import load_instance, load_solution
from vrp_solver.model import Solution
from vrp_solver.rules import derive_solution
from vrp_solver.solver.pressure import pressure_points
from vrp_solver.solver.surgical_search import (
    SurgicalSearchConfig, _narrow_window_pressures,
    _route_allows_trailer, _resource_overlap,
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


def bounds(shift):
    trial = derive_solution(instance, Solution((replace(shift, index=0),)))[0]
    duration = trial.end - shift.start
    dl, dh = -shift.start, instance.latest_time - trial.end
    for op, dop in zip(shift.operations, trial.operations):
        cust = instance.customer_by_point.get(op.point)
        if cust is None:
            continue
        containing = [w for w in cust.time_windows
                      if w.start <= op.arrival and dop.departure <= w.end]
        if not containing:
            return None
        dl = max(dl, min(w.start - op.arrival for w in containing))
        dh = min(dh, max(w.end - dop.departure for w in containing))
    if dl > dh:
        return None
    return dl, dh, duration


def try_place(shift, extra_gap_starts):
    b = bounds(shift)
    if b is None:
        return False
    dl, dh, duration = b
    driver = instance.drivers[shift.driver]
    starts = set()
    for w in driver.time_windows:
        low = max(shift.start + dl, w.start)
        high = min(shift.start + dh, w.end - duration)
        if low > high:
            continue
        starts.add(int(low))
        for other, od in zip(solution.shifts, derived):
            if other.driver == shift.driver:
                cs = od.end + driver.min_inter_shift_duration
                if low <= cs <= high:
                    starts.add(cs)
            if other.trailer == shift.trailer and low <= od.end <= high:
                starts.add(od.end)
        if extra_gap_starts:
            # Every busy interval's end on EITHER resource, clamped into range.
            for other, od in zip(solution.shifts, derived):
                if other.driver != shift.driver and other.trailer != shift.trailer:
                    continue
                for cand in (od.end, od.end + driver.min_inter_shift_duration):
                    if cand < low:
                        continue
                    if cand <= high:
                        starts.add(cand)
            # And the window start itself, plus a scan of gap boundaries.
            busy = sorted(
                (s.start, d.end) for s, d in zip(solution.shifts, derived)
                if s.driver == shift.driver or s.trailer == shift.trailer
            )
            cursor = low
            for bs, be in busy:
                if be <= low or bs >= high:
                    continue
                if cursor <= high:
                    starts.add(int(cursor))
                cursor = max(cursor, be + driver.min_inter_shift_duration)
            if cursor <= high:
                starts.add(int(cursor))
    for start in sorted(starts):
        delta = start - shift.start
        placed = replace(shift, start=start, operations=tuple(
            replace(o, arrival=o.arrival + delta) for o in shift.operations))
        if not _resource_overlap(instance, solution, derived, -1, placed, start + duration):
            return True
    return False


for label, extra in (("current", False), ("with_gap_starts", True)):
    ok = 0
    routes_ok = 0
    for created in routes:
        hit = False
        for driver_id, trailer_id in pairs:
            if not _route_allows_trailer(instance, created, trailer_id):
                continue
            if try_place(replace(created, driver=driver_id, trailer=trailer_id), extra):
                ok += 1
                hit = True
                break
        routes_ok += int(hit)
    print(f"variant,{label},routes_placed,{routes_ok},of,{len(routes)}")
