"""Do the host routes even overlap the pressure customer's delivery windows?

_insert_operation_candidates only requires shift.start < breach_minute.  It never
checks that the customer can legally be served during that shift, so a route
running entirely outside the customer's windows generates 0 candidates while
still consuming a search step.
"""
import sys
from pathlib import Path

from vrp_solver.xml_io import load_instance, load_solution
from vrp_solver.rules import derive_solution
from vrp_solver.solver.pressure import pressure_points
from vrp_solver.solver.surgical_search import SurgicalSearchConfig

instance = load_instance(Path(sys.argv[1]))
solution = load_solution(Path(sys.argv[2]))
config = SurgicalSearchConfig(end_day=10, candidates_per_move=120)
pressure = pressure_points(instance, solution, end_day=config.end_day)
derived = derive_solution(instance, solution)

for point in pressure[: config.pressure_customers]:
    customer = instance.customer_by_point[point.customer]
    wins = getattr(customer, "time_windows", ())
    hosts = overlapping = 0
    for shift_pos, shift in enumerate(solution.shifts):
        if shift.start >= point.first_minute:
            continue
        if shift.trailer not in customer.allowed_trailers:
            continue
        hosts += 1
        s, e = shift.start, derived[shift_pos].end
        if any(w.start < e and s < w.end for w in wins) or not wins:
            overlapping += 1
    print(
        f"customer,{point.customer},windows,{len(wins)},"
        f"hosts_tried,{hosts},hosts_overlapping_a_window,{overlapping},"
        f"breach_minute,{point.first_minute}"
    )
    for w in list(wins)[:4]:
        print(f"  window,{w.start},{w.end}")
