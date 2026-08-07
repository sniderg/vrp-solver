"""How large are the remaining safety breaches, and are their tanks targeted?

A breach of a few litres is a quantity-repair problem; a breach of thousands is
a topology problem.  pressure_points is what every insertion operator uses to
choose targets, so a breaching tank absent from it can never be fixed.
"""
import re
import sys
from pathlib import Path

from vrp_solver.xml_io import load_instance, load_solution
from vrp_solver.rules import validate_solution
from vrp_solver.solver.pressure import pressure_points

instance = load_instance(Path(sys.argv[1]))
solution = load_solution(Path(sys.argv[2]))
errors = [e for e in validate_solution(instance, solution) if e.code == "QS02"]
targets = {p.customer for p in pressure_points(instance, solution, end_day=instance.horizon)}

rows = []
for e in errors:
    m = re.search(r"inventory ([0-9.]+) is below safety level ([0-9.]+)", e.message)
    if not m:
        continue
    inv, safe = float(m.group(1)), float(m.group(2))
    rows.append((safe - inv, e.point))

rows.sort(reverse=True)
print(f"breaches,{len(rows)}")
buckets = {"<1L": 0, "1-10L": 0, "10-100L": 0, "100-1000L": 0, ">1000L": 0}
for gap, _ in rows:
    if gap < 1: buckets["<1L"] += 1
    elif gap < 10: buckets["1-10L"] += 1
    elif gap < 100: buckets["10-100L"] += 1
    elif gap < 1000: buckets["100-1000L"] += 1
    else: buckets[">1000L"] += 1
for k, v in buckets.items():
    print(f"deficit_{k},{v}")

breach_pts = {pt for _, pt in rows}
print(f"distinct_breaching_tanks,{len(breach_pts)}")
print(f"pressure_targets,{len(targets)}")
print(f"breaching_tanks_that_are_targeted,{len(breach_pts & targets)}")
print(f"breaching_tanks_never_targeted,{len(breach_pts - targets)}")
print("largest_breaches (deficit_litres,point,targeted):")
for gap, pt in rows[:10]:
    print(f"  {gap:.3f},{pt},{pt in targets}")
