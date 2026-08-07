"""Do joint driver+trailer idle windows exist, and do breaching tanks fit them?

The placement gate translates a pre-built route rigidly and demands the driver
AND trailer be simultaneously free, with min_inter_shift_duration separation on
the driver side.  The rescue generators choose service times from customer
windows alone, with no knowledge of resource availability -- so they can propose
routes that no joint gap can host.  Enumerate the joint gaps directly.
"""
import sys
from pathlib import Path

from vrp_solver.xml_io import load_instance, load_solution
from vrp_solver.rules import derive_solution
from vrp_solver.solver.pressure import pressure_points

instance = load_instance(Path(sys.argv[1]))
solution = load_solution(Path(sys.argv[2]))
derived = derive_solution(instance, solution)

pressure = pressure_points(instance, solution, end_day=10)
breaching = [instance.customer_by_point[p.customer] for p in pressure]

joint = []
for driver in instance.drivers:
    sep = driver.min_inter_shift_duration
    busy_d = sorted((s.start, d.end) for s, d in zip(solution.shifts, derived)
                    if s.driver == driver.index)
    for trailer_id in driver.trailer_ids:
        busy_t = sorted((s.start, d.end) for s, d in zip(solution.shifts, derived)
                        if s.trailer == trailer_id)
        for w in driver.time_windows:
            cursor = w.start
            edges = [(bs, be) for bs, be in busy_d if be > w.start and bs < w.end]
            for bs, be in edges:
                lo = cursor + (sep if cursor != w.start else 0)
                if bs - lo > 0:
                    joint.append((driver.index, trailer_id, lo, bs, busy_t))
                cursor = max(cursor, be)
            lo = cursor + (sep if cursor != w.start else 0)
            if w.end - lo > 0:
                joint.append((driver.index, trailer_id, lo, w.end, busy_t))

usable = []
for drv, trl, lo, hi, busy_t in joint:
    # Subtract trailer commitments to get truly joint-free subintervals.
    free = [(lo, hi)]
    for ts, te in busy_t:
        nxt = []
        for fs, fe in free:
            if te <= fs or ts >= fe:
                nxt.append((fs, fe))
                continue
            if ts > fs:
                nxt.append((fs, ts))
            if te < fe:
                nxt.append((te, fe))
        free = nxt
    for fs, fe in free:
        if fe - fs >= 300:
            usable.append((drv, trl, fs, fe))

usable.sort(key=lambda x: x[3] - x[2], reverse=True)
print(f"driver_side_gaps,{len(joint)}")
print(f"joint_free_intervals_at_least_300min,{len(usable)}")
for drv, trl, fs, fe in usable[:10]:
    print(f"  driver,{drv},trailer,{trl},from,{fs},to,{fe},minutes,{fe - fs}")

print(f"breaching_customers,{len(breaching)}")
for cust, p in zip(breaching, pressure):
    hosts = 0
    for drv, trl, fs, fe in usable:
        if trl not in cust.allowed_trailers:
            continue
        if any(w.start < fe and fs < w.end for w in cust.time_windows):
            hosts += 1
    print(f"  customer,{cust.index},breach_minute,{p.first_minute},"
          f"joint_windows_that_could_serve_it,{hosts}")
