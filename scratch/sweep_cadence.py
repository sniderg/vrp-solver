"""Sweep the cadence controls (idle cap, idle penalty) on a Set B instance.

Reports, per setting: seed error count, safety-deficit quantity-minutes, late
first-visit counts split by cause, shift count, and realised idle minutes.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vrp_solver.cli import load_instance
from vrp_solver.inventory import project_customer_inventory, tank_aggregates
from vrp_solver.rules import validate_solution
from vrp_solver.solver import cluster_greedy as cg

INSTANCE = sys.argv[1] if len(sys.argv) > 1 else (
    "roadef_2016_data/set_B/Instances_B_V25-11042016/V2.12.xml"
)

SETTINGS = [
    (None, 0.0),      # baseline
    (None, 40.0),
    (None, 120.0),
    (720, 0.0),
    (720, 120.0),
    (360, 120.0),
    (180, 120.0),
    (180, 300.0),
]

instance = load_instance(INSTANCE)
policy = cg.derive_cluster_construction_policy(instance)
end_day = max(1, (instance.horizon * instance.unit + 1439) // 1440)

natural_breach = {}
for customer in instance.customers:
    if customer.call_in:
        continue
    events = project_customer_inventory(instance, customer, {})
    step = next((e.step for e in events if e.safety_breach), None)
    if step is not None:
        natural_breach[customer.index] = step * instance.unit

print(f"instance,{instance.name},breaching_tanks,{len(natural_breach)}")
print("cap,penalty,errors,safety_qm,late,starvation,stretching,unserved,"
      "shifts,ops,idle_min,travel_min,seconds")

for cap, penalty in SETTINGS:
    started = time.monotonic()
    solution, report = cg.construct_cluster_solution(
        instance,
        safety_buffer=0.20,
        neighborhood_size=policy.neighborhood_size,
        score_cutoff_minute=end_day * 1440,
        global_pressure_fill=policy.global_pressure_fill,
        tie_break_seed=1,
        max_idle_wait_minutes=cap,
        idle_wait_penalty_per_hour=penalty,
    )
    elapsed = time.monotonic() - started
    errors = sum(v.severity == "error" for v in validate_solution(instance, solution))
    _, _, _, safety_qm = tank_aggregates(instance, solution)

    first = {}
    for shift in solution.shifts:
        for op in shift.operations:
            if op.quantity > 0 and op.point in instance.customer_by_point:
                prev = first.get(op.point)
                if prev is None or op.arrival < prev[0]:
                    first[op.point] = (op.arrival, shift.start)
    starvation = stretching = unserved = 0
    for point, breach_minute in natural_breach.items():
        entry = first.get(point)
        if entry is None:
            unserved += 1
            continue
        arrival, shift_start = entry
        if arrival <= breach_minute:
            continue
        if shift_start > breach_minute:
            starvation += 1
        else:
            stretching += 1

    idle = travel = 0
    for shift in solution.shifts:
        prev_point, prev_time = instance.base_index, shift.start
        for op in shift.operations:
            leg = instance.time_matrix[prev_point][op.point]
            travel += leg
            idle += max(0, op.arrival - prev_time - leg)
            obj = instance.customer_by_point.get(op.point)
            setup = obj.setup_time if obj else instance.source_by_point[op.point].setup_time
            prev_time, prev_point = op.arrival + setup, op.point

    print(f"{cap},{penalty:.0f},{errors},{safety_qm:.0f},"
          f"{starvation + stretching},{starvation},{stretching},{unserved},"
          f"{report.shifts},{report.operations},{idle},{travel},{elapsed:.0f}")
