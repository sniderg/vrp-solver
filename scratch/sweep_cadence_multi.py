"""Cadence sweep across several Set B instances (one instance per process).

Usage: sweep_cadence_multi.py <instance.xml>
Prints one CSV row per setting so results can be pooled across parallel runs.
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

INSTANCE = sys.argv[1]
SETTINGS = [(None, 0.0), (180, 0.0), (180, 120.0), (300, 120.0)]

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
                if prev is None or op.arrival < prev:
                    first[op.point] = op.arrival
    late = sum(
        1 for point, breach in natural_breach.items()
        if point in first and first[point] > breach
    )
    unserved = sum(1 for point in natural_breach if point not in first)

    idle = 0
    for shift in solution.shifts:
        prev_point, prev_time = instance.base_index, shift.start
        for op in shift.operations:
            leg = instance.time_matrix[prev_point][op.point]
            idle += max(0, op.arrival - prev_time - leg)
            obj = instance.customer_by_point.get(op.point)
            setup = obj.setup_time if obj else instance.source_by_point[op.point].setup_time
            prev_time, prev_point = op.arrival + setup, op.point

    print(
        f"{instance.name},{cap},{penalty:.0f},{errors},{safety_qm:.0f},{late},"
        f"{unserved},{report.shifts},{report.operations},{idle},{elapsed:.0f}",
        flush=True,
    )
