"""Split placement failures into window-interval vs resource-overlap causes."""
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

no_window = empty_interval = no_start = overlap = 0
total = 0
for created in routes:
    for driver_id, trailer_id in pairs:
        if not _route_allows_trailer(instance, created, trailer_id):
            continue
        total += 1
        shift = replace(created, driver=driver_id, trailer=trailer_id)
        trial = derive_solution(instance, Solution((replace(shift, index=0),)))[0]
        duration = trial.end - shift.start
        dl, dh = -shift.start, instance.latest_time - trial.end
        bad = False
        for op, dop in zip(shift.operations, trial.operations):
            cust = instance.customer_by_point.get(op.point)
            if cust is None:
                continue
            containing = [w for w in cust.time_windows
                          if w.start <= op.arrival and dop.departure <= w.end]
            if not containing:
                bad = True
                break
            dl = max(dl, min(w.start - op.arrival for w in containing))
            dh = min(dh, max(w.end - dop.departure for w in containing))
        if bad:
            no_window += 1
            continue
        if dl > dh:
            empty_interval += 1
            continue
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
                if other.trailer == shift.trailer:
                    if low <= od.end <= high:
                        starts.add(od.end)
        if not starts:
            no_start += 1
            continue
        placed_any = False
        for start in sorted(starts):
            delta = start - shift.start
            placed = replace(shift, start=start, operations=tuple(
                replace(o, arrival=o.arrival + delta) for o in shift.operations))
            if not _resource_overlap(instance, solution, derived, -1, placed, start + duration):
                placed_any = True
                break
        if not placed_any:
            overlap += 1

print(f"(route,pair)_attempts,{total}")
print(f"rejected_no_containing_customer_window,{no_window}")
print(f"rejected_empty_translation_interval,{empty_interval}")
print(f"rejected_no_legal_driver_start,{no_start}")
print(f"rejected_resource_overlap_at_every_start,{overlap}")
