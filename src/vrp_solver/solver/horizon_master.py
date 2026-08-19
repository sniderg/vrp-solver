from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np

from ..model import Instance, Shift, Operation, Solution
from ..fast.state import FastInstance, instance_days
from ..fast.retime import earliest_arrivals


def construct_horizon_master_solution(
    instance: Instance,
    safety_buffer: float = 0.05,
) -> Solution:
    days = instance_days(instance)
    fi = FastInstance(instance, score_days=days)
    plant_pt = instance.sources[0].index
    base_pt = 0

    clusters = []
    visited = set()
    for c in instance.customers:
        if c.index in visited:
            continue
        cluster = [c.index]
        visited.add(c.index)
        for other in instance.customers:
            if other.index not in visited:
                if instance.time_matrix[c.index][other.index] <= 120:
                    cluster.append(other.index)
                    visited.add(other.index)
        clusters.append(cluster)

    shifts: List[Shift] = []
    driver_avail = {d.index: 0 for d in instance.drivers}
    trailer_avail = {t.index: 0 for t in instance.trailers}
    trailers_q = {t.index: t.initial_quantity for t in instance.trailers}
    tank_stock = {c.index: c.initial_tank_quantity for c in instance.customers}

    for day in range(1, days + 1):
        day_start = day * 1440
        for c in instance.customers:
            day_i = min(len(c.forecast) - 1, day - 1)
            cons = c.forecast[day_i] if c.forecast else 0.0
            tank_stock[c.index] = max(0.0, tank_stock[c.index] - cons)

        for dr in instance.drivers:
            for slot_offset in [360, 840]:
                target_start = day_start + slot_offset
                if target_start < driver_avail[dr.index]:
                    continue

                matched_window = None
                for w in dr.time_windows:
                    if w.start <= target_start and target_start + 480 <= w.end:
                        matched_window = w
                        break
                if matched_window is None:
                    continue

                avail_trailers = [
                    t.index for t in instance.trailers
                    if trailer_avail[t.index] <= target_start and t.index in dr.trailer_ids
                ]
                if not avail_trailers:
                    continue

                tr_idx = avail_trailers[0]
                tr = instance.trailers[tr_idx]
                cap = tr.capacity
                curr_q = trailers_q[tr_idx]

                cluster_urgencies = []
                for cl_idx, cl in enumerate(clusters):
                    urg = sum(instance.customer_by_point[pt].safety_level - tank_stock[pt] for pt in cl)
                    cluster_urgencies.append((urg, cl))
                cluster_urgencies.sort(key=lambda x: x[0], reverse=True)

                chosen_pts = None
                for _, cl in cluster_urgencies:
                    valid_pts = []
                    tot_driv = instance.time_matrix[base_pt][plant_pt]
                    prev = plant_pt
                    for pt in cl:
                        c = instance.customer_by_point[pt]
                        if tr_idx not in c.allowed_trailers:
                            continue
                        d_leg = instance.time_matrix[prev][pt]
                        d_ret = instance.time_matrix[pt][base_pt]
                        if tot_driv + d_leg + d_ret <= dr.max_driving_duration:
                            tot_driv += d_leg
                            prev = pt
                            valid_pts.append(pt)
                            if len(valid_pts) >= 4:
                                break
                    if valid_pts:
                        chosen_pts = valid_pts
                        break

                if not chosen_pts:
                    continue

                ops: List[Tuple[int, float]] = []
                needed = cap - curr_q
                if needed > 1e-3:
                    ops.append((plant_pt, -needed))
                    curr_q = cap

                # Distribute capacity equally across chosen customers
                per_cust_share = cap / max(1, len(chosen_pts))
                for pt in chosen_pts:
                    c = instance.customer_by_point[pt]
                    ullage = max(0.0, c.capacity - tank_stock[pt])
                    drop = min(curr_q, max(c.min_operation_quantity, min(ullage, per_cust_share)))
                    if drop >= c.min_operation_quantity and curr_q >= drop:
                        ops.append((pt, drop))
                        curr_q -= drop
                        tank_stock[pt] += drop

                if len(ops) <= 1:
                    continue

                pts = [pt for pt, _ in ops]
                arrivals = earliest_arrivals(fi, target_start, pts, dr.index)
                
                has_gap = any(arrivals[k+1] - arrivals[k] > 720 for k in range(len(arrivals) - 1))
                if has_gap:
                    continue

                in_window = True
                for pt, arr in zip(pts, arrivals):
                    if pt in instance.customer_by_point:
                        c = instance.customer_by_point[pt]
                        if not any(w.start <= arr <= w.end for w in c.time_windows):
                            in_window = False
                            break
                if not in_window:
                    continue

                shift_end = arrivals[-1] + instance.time_matrix[pts[-1]][base_pt] + 50
                if shift_end > matched_window.end:
                    continue

                trailers_q[tr_idx] = curr_q
                shift_ops = tuple(
                    Operation(point=pt, arrival=arr, quantity=qty)
                    for (pt, qty), arr in zip(ops, arrivals)
                )

                driver_avail[dr.index] = shift_end + dr.min_inter_shift_duration
                trailer_avail[tr_idx] = shift_end

                shifts.append(
                    Shift(
                        index=len(shifts),
                        driver=dr.index,
                        trailer=tr_idx,
                        start=target_start,
                        operations=shift_ops,
                    )
                )

    return Solution(shifts=tuple(shifts))
