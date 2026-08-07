"""Is the retiming failure a lack of slack, or a hard driver-duration limit?

For each pressure customer's candidate host shifts, report the slack between the
route's current end and the latest legal return, plus the driver's remaining
driving-duration headroom.  If slack is ample but insertions still fail, the
binding constraint is the driver duration cap, not the resource slot.
"""
import sys
from pathlib import Path

from vrp_solver.xml_io import load_instance, load_solution
from vrp_solver.rules import derive_solution
from vrp_solver.solver.pressure import pressure_points
from vrp_solver.solver.surgical_search import (
    SurgicalSearchConfig, _resource_slot_end,
)

instance = load_instance(Path(sys.argv[1]))
solution = load_solution(Path(sys.argv[2]))
config = SurgicalSearchConfig(end_day=10, candidates_per_move=120)
pressure = pressure_points(instance, solution, end_day=config.end_day)
derived = derive_solution(instance, solution)

for point in pressure[: config.pressure_customers]:
    customer = instance.customer_by_point[point.customer]
    hosts = 0
    slacks = []
    drive_head = []
    for shift_pos, shift in enumerate(solution.shifts):
        if shift.start >= point.first_minute:
            continue
        if shift.trailer not in customer.allowed_trailers:
            continue
        hosts += 1
        end = derived[shift_pos].end
        slot = _resource_slot_end(instance, solution, derived, shift_pos)
        slacks.append(slot - end)
        driver = instance.drivers[shift.driver]
        # Driving duration excludes setup time at each stop.
        travel = sum(
            instance.time_matrix[a][b]
            for a, b in zip(
                (shift.operations[0].point, *[o.point for o in shift.operations]),
                [o.point for o in shift.operations],
            )
        )
        drive_head.append(driver.max_driving_duration - travel)
    if not hosts:
        print(f"customer,{point.customer},no_hosts")
        continue
    print(
        f"customer,{point.customer},hosts,{hosts},"
        f"slot_slack_min,{min(slacks)},slot_slack_max,{max(slacks)},"
        f"drive_headroom_min,{min(drive_head)},drive_headroom_max,{max(drive_head)},"
        f"setup_time,{customer.setup_time}"
    )
