"""Measure why V2.12 cold-start first visits land after the natural breach.

Instruments the cluster constructor to record, for every accepted candidate,
whether the economic-step deferral moved the arrival, and compares first-visit
timing against each tank's natural (no-delivery) breach step.
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

stats = {"calls": 0, "deferred": 0, "deferred_minutes": 0, "accepted": 0,
         "accepted_deferred": 0}

_orig = cg._candidate_for_customer
_orig_econ = cg._first_economic_service_step


def traced_econ(events, customer, start_step):
    result = _orig_econ(events, customer, start_step)
    if result is not None and result > start_step:
        stats["deferred"] += 1
        stats["deferred_minutes"] += result - start_step
    stats["calls"] += 1
    return result


cg._first_economic_service_step = traced_econ

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

print(f"instance,{instance.name}")
print(f"shifts,{report.shifts},operations,{report.operations}")
print(f"econ_calls,{stats['calls']},econ_deferred,{stats['deferred']},"
      f"deferred_steps_total,{stats['deferred_minutes']}")

# First visit per customer vs natural breach step.
first_visit: dict[int, int] = {}
for shift in solution.shifts:
    for op in shift.operations:
        if op.quantity > 0 and op.point in instance.customer_by_point:
            prev = first_visit.get(op.point)
            if prev is None or op.arrival < prev:
                first_visit[op.point] = op.arrival

late = early = never = 0
late_days: list[float] = []
early_days: list[float] = []
for customer in instance.customers:
    if customer.call_in:
        continue
    events = project_customer_inventory(instance, customer, {})
    breach = next((e.step for e in events if e.safety_breach), None)
    if breach is None:
        continue
    visit = first_visit.get(customer.index)
    if visit is None:
        never += 1
        continue
    delta_days = (visit - breach * instance.unit) / 1440.0
    if delta_days > 0:
        late += 1
        late_days.append(delta_days)
    else:
        early += 1
        early_days.append(-delta_days)

print(f"breaching_tanks,{late + early + never}")
print(f"served_late,{late},served_early,{early},never_served,{never}")
if late_days:
    print(f"late_days_mean,{sum(late_days)/len(late_days):.2f},"
          f"late_days_max,{max(late_days):.2f}")
if early_days:
    print(f"early_days_mean,{sum(early_days)/len(early_days):.2f}")

# Shift-length / cadence profile.
lengths = []
for shift in solution.shifts:
    if not shift.operations:
        continue
    lengths.append((shift.operations[-1].arrival - shift.start) / 60.0)
if lengths:
    lengths.sort()
    print(f"shifts,{len(lengths)},shift_hours_mean,{sum(lengths)/len(lengths):.2f},"
          f"median,{lengths[len(lengths)//2]:.2f},max,{max(lengths):.2f}")

# Driver window utilisation: how much of each driver's windows were used.
used = {}
for shift in solution.shifts:
    span = (shift.operations[-1].arrival - shift.start) if shift.operations else 0
    used[shift.driver] = used.get(shift.driver, 0) + span
total_window = sum(
    sum(w.end - w.start for w in d.time_windows) for d in instance.drivers
)
print(f"driver_window_minutes,{total_window},used_minutes,{sum(used.values())},"
      f"utilisation,{sum(used.values())/max(1,total_window):.3f}")
print(f"drivers_total,{len(instance.drivers)},drivers_used,{len(used)}")
