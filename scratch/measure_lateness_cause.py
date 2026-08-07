"""Attribute late first-visits to resource starvation vs route stretching.

For every VMI tank served after its natural breach, compare the breach instant
against the *start* of the shift that served it:

  shift_start > breach   -> resource starvation (no shift was even under way)
  shift_start <= breach  -> route stretching (shift was out, tank served late
                            in a long chain)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vrp_solver.cli import load_instance
from vrp_solver.inventory import project_customer_inventory
from vrp_solver.solver import cluster_greedy as cg

INSTANCE = sys.argv[1] if len(sys.argv) > 1 else (
    "roadef_2016_data/set_B/Instances_B_V25-11042016/V2.12.xml"
)

instance = load_instance(INSTANCE)
policy = cg.derive_cluster_construction_policy(instance)
end_day = max(1, (instance.horizon * instance.unit + 1439) // 1440)
solution, report = cg.construct_cluster_solution(
    instance,
    safety_buffer=0.20,
    neighborhood_size=policy.neighborhood_size,
    score_cutoff_minute=end_day * 1440,
    global_pressure_fill=policy.global_pressure_fill,
    tie_break_seed=1,
)

# first visit -> (arrival, shift_start, stop_position, shift_stops)
first: dict[int, tuple[int, int, int, int]] = {}
for shift in solution.shifts:
    stops = [op for op in shift.operations if op.quantity > 0]
    for position, op in enumerate(stops):
        if op.point not in instance.customer_by_point:
            continue
        prev = first.get(op.point)
        if prev is None or op.arrival < prev[0]:
            first[op.point] = (op.arrival, shift.start, position, len(stops))

starvation = stretching = 0
starve_days: list[float] = []
stretch_days: list[float] = []
stretch_positions: list[int] = []
for customer in instance.customers:
    if customer.call_in:
        continue
    events = project_customer_inventory(instance, customer, {})
    breach = next((e.step for e in events if e.safety_breach), None)
    if breach is None:
        continue
    entry = first.get(customer.index)
    if entry is None:
        continue
    arrival, shift_start, position, stops = entry
    breach_minute = breach * instance.unit
    if arrival <= breach_minute:
        continue
    late_days = (arrival - breach_minute) / 1440.0
    if shift_start > breach_minute:
        starvation += 1
        starve_days.append(late_days)
    else:
        stretching += 1
        stretch_days.append(late_days)
        stretch_positions.append(position)

print(f"instance,{instance.name}")
print(f"late_total,{starvation + stretching}")
print(f"starvation,{starvation}," + (
    f"mean_days,{sum(starve_days)/len(starve_days):.2f}" if starve_days else "mean_days,-"))
print(f"stretching,{stretching}," + (
    f"mean_days,{sum(stretch_days)/len(stretch_days):.2f}," if stretch_days else "mean_days,-,")
    + (f"mean_stop_position,{sum(stretch_positions)/len(stretch_positions):.1f}"
       if stretch_positions else "mean_stop_position,-"))

# How long do shifts hold a resource, and how much of that is productive?
holds = []
for shift in solution.shifts:
    if not shift.operations:
        continue
    end = shift.operations[-1].arrival
    holds.append((end - shift.start, len([o for o in shift.operations if o.quantity > 0])))
holds.sort(reverse=True)
print("longest_holds_hours_stops," + " ".join(
    f"{h/60:.0f}h/{n}" for h, n in holds[:10]))
print(f"total_stops,{sum(n for _, n in holds)},shifts,{len(holds)}")
