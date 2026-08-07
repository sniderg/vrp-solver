"""Where does in-route time go: driving, setup, or waiting?

For each accepted candidate in a V2.12 cold-start construction, split the
elapsed time between consecutive operations into travel and idle wait, and
report the biggest idle blocks with their cause (economic deferral vs customer
window alignment).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vrp_solver.cli import load_instance
from vrp_solver.solver import cluster_greedy as cg

INSTANCE = sys.argv[1] if len(sys.argv) > 1 else (
    "roadef_2016_data/set_B/Instances_B_V25-11042016/V2.12.xml"
)

records: list[tuple[int, int, int, bool]] = []  # raw_wait, econ_wait, window_wait, layover
_orig = cg._candidate_for_customer
_orig_align = cg._align_arrival_to_customer_window
_orig_econ = cg._first_economic_service_step


def make_traced(instance_holder):
    def traced(instance, resource, window, current_pt, current_time, driving,
               customer, deliveries, buffer, *args, **kwargs):
        cand = _orig(instance, resource, window, current_pt, current_time,
                     driving, customer, deliveries, buffer, *args, **kwargs)
        if cand is not None and not customer.call_in:
            travel = instance.time_matrix[current_pt][customer.index]
            raw = current_time + travel
            wait = cand.arrival - raw
            if wait > 0:
                records.append((wait, travel, customer.index, cand.layover_before))
        return cand
    return traced


cg._candidate_for_customer = make_traced(None)

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

print(f"instance,{instance.name},unit,{instance.unit}")
print(f"candidates_with_wait,{len(records)}")
waits = sorted((w for w, _, _, _ in records), reverse=True)
if waits:
    print(f"wait_minutes_mean,{sum(waits)/len(waits):.0f},median,"
          f"{waits[len(waits)//2]},max,{max(waits)}")
    print("top_waits_hours," + " ".join(f"{w/60:.0f}" for w in waits[:15]))
    layovers = sum(1 for _, _, _, lay in records if lay)
    print(f"waits_flagged_layover,{layovers}")

# Now, in the produced solution, measure realised idle between operations.
idle_total = travel_total = 0
big_gaps = []
for shift in solution.shifts:
    ops = list(shift.operations)
    prev_point = instance.base_index
    prev_time = shift.start
    for op in ops:
        travel = instance.time_matrix[prev_point][op.point]
        gap = op.arrival - prev_time - travel
        travel_total += travel
        idle_total += max(0, gap)
        if gap > 300:
            big_gaps.append((gap, shift.index, prev_point, op.point))
        point_obj = instance.customer_by_point.get(op.point)
        setup = point_obj.setup_time if point_obj else instance.source_by_point[op.point].setup_time
        prev_time = op.arrival + setup
        prev_point = op.point
print(f"realised_travel_minutes,{travel_total},realised_idle_minutes,{idle_total}")
big_gaps.sort(reverse=True)
print(f"gaps_over_5h,{len(big_gaps)}")
print("top_gaps_hours," + " ".join(f"{g/60:.0f}" for g, _, _, _ in big_gaps[:15]))
