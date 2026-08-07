"""Is the fleet actually saturated, or just the placement search?

Reports, per driver, total committed minutes against total time-window minutes,
and the largest idle gap that a new route could occupy.
"""
import sys
from pathlib import Path

from vrp_solver.xml_io import load_instance, load_solution
from vrp_solver.rules import derive_solution

instance = load_instance(Path(sys.argv[1]))
solution = load_solution(Path(sys.argv[2]))
derived = derive_solution(instance, solution)

print(f"drivers,{len(instance.drivers)},trailers_per_driver,"
      f"{[len(d.trailer_ids) for d in instance.drivers]}")
print(f"shifts,{len(solution.shifts)}")

for driver in instance.drivers:
    windows = sorted((w.start, w.end) for w in driver.time_windows)
    avail = sum(e - s for s, e in windows)
    busy = [(s.start, d.end) for s, d in zip(solution.shifts, derived)
            if s.driver == driver.index]
    busy.sort()
    used = sum(e - s for s, e in busy)
    # Largest idle gap inside any driver window.
    biggest = 0
    for ws, we in windows:
        cursor = ws
        for bs, be in busy:
            if be <= ws or bs >= we:
                continue
            biggest = max(biggest, bs - cursor)
            cursor = max(cursor, be + driver.min_inter_shift_duration)
        biggest = max(biggest, we - cursor)
    print(f"driver,{driver.index},shifts,{len(busy)},"
          f"window_minutes,{avail},committed_minutes,{used},"
          f"utilisation_pct,{100.0*used/avail if avail else 0:.1f},"
          f"largest_idle_gap_minutes,{biggest},"
          f"min_inter_shift,{driver.min_inter_shift_duration}")

durations = [d.end - s.start for s, d in zip(solution.shifts, derived)]
durations.sort()
print(f"existing_shift_duration_min,{durations[0]},median,{durations[len(durations)//2]},max,{durations[-1]}")
