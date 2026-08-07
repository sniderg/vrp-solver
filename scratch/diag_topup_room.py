"""Is there room to deliver a legal minimum drop into the breaching tanks?

A breach of tens of litres needs a small top-up, but every operation must be at
least min_operation_quantity.  If the tank's free ullage at the insertion time is
below that floor, no legal insertion exists and no retiming can create one.
"""
import sys
from pathlib import Path

from vrp_solver.xml_io import load_instance, load_solution
from vrp_solver.inventory import tank_events
from vrp_solver.solver.pressure import pressure_points
from vrp_solver.solver.surgical_search import SurgicalSearchConfig

instance = load_instance(Path(sys.argv[1]))
solution = load_solution(Path(sys.argv[2]))
config = SurgicalSearchConfig(end_day=10, candidates_per_move=120)
pressure = pressure_points(instance, solution, end_day=config.end_day)

for point in pressure[: config.pressure_customers]:
    customer = instance.customer_by_point[point.customer]
    cap = customer.capacity
    floor = customer.min_operation_quantity
    try:
        events = tank_events(instance, solution, customer.index)
        peak = max(getattr(e, "quantity", getattr(e, "level", 0.0)) for e in events)
    except Exception as exc:  # noqa: BLE001
        peak = float("nan")
        print(f"customer,{point.customer},tank_events_error,{type(exc).__name__}")
    delivered = sum(
        o.quantity
        for s in solution.shifts
        for o in s.operations
        if o.point == customer.index
    )
    print(
        f"customer,{point.customer},capacity,{cap:.1f},"
        f"min_operation_quantity,{floor:.1f},"
        f"min_qty_as_pct_of_capacity,{100.0 * floor / cap:.1f},"
        f"peak_level,{peak:.1f},ullage_at_peak,{cap - peak:.1f},"
        f"total_delivered,{delivered:.1f},safety_level,{customer.safety_level:.1f}"
    )
