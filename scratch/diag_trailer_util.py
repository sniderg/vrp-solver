"""Trailer utilisation: is the trailer the scarce resource, not the driver?"""
import sys
from pathlib import Path

from vrp_solver.xml_io import load_instance, load_solution
from vrp_solver.rules import derive_solution

instance = load_instance(Path(sys.argv[1]))
solution = load_solution(Path(sys.argv[2]))
derived = derive_solution(instance, solution)

trailers = sorted({t for d in instance.drivers for t in d.trailer_ids})
print(f"trailers,{len(trailers)},drivers,{len(instance.drivers)}")
horizon = instance.latest_time
for t in trailers:
    busy = sorted((s.start, d.end) for s, d in zip(solution.shifts, derived)
                  if s.trailer == t)
    used = sum(e - s for s, e in busy)
    merged = []
    for s, e in busy:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    gaps = []
    cursor = 0
    for s, e in merged:
        gaps.append(s - cursor)
        cursor = e
    gaps.append(horizon - cursor)
    gaps = [g for g in gaps if g > 0]
    drivers_for = [d.index for d in instance.drivers if t in d.trailer_ids]
    print(f"trailer,{t},shifts,{len(busy)},committed_minutes,{used},"
          f"horizon,{horizon},utilisation_pct,{100.0*used/horizon:.1f},"
          f"largest_free_gap,{max(gaps) if gaps else 0},"
          f"free_gaps_over_800,{sum(1 for g in gaps if g > 800)},"
          f"usable_by_drivers,{drivers_for}")
