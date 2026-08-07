"""Is V2.15 genuinely resource-bound, or is the incumbent wasting its fleet?

Compares total driver time actually needed to serve the breaching tanks against
the joint driver+trailer idle capacity remaining.  Also reports how much of the
committed time is idle waiting, because idle time inside a shift blocks a
resource just as effectively as travel and is the one thing the planner can
reclaim without new capacity.
"""
import sys
from pathlib import Path

from vrp_solver.xml_io import load_instance, load_solution
from vrp_solver.rules import derive_solution
from vrp_solver.solver.pressure import pressure_points

instance = load_instance(Path(sys.argv[1]))
solution = load_solution(Path(sys.argv[2]))
derived = derive_solution(instance, solution)

total_committed = 0
total_travel = 0
total_setup = 0
for shift, d in zip(solution.shifts, derived):
    total_committed += d.end - shift.start
    last = instance.base_index
    for op in shift.operations:
        total_travel += instance.time_matrix[last][op.point]
        total_setup += instance.setup_time_for_point(op.point)
        last = op.point
    total_travel += instance.time_matrix[last][instance.base_index]

idle = total_committed - total_travel - total_setup
print(f"shifts,{len(solution.shifts)}")
print(f"committed_minutes,{total_committed}")
print(f"travel_minutes,{total_travel}")
print(f"setup_minutes,{total_setup}")
print(f"idle_minutes,{idle},idle_pct_of_committed,{100.0*idle/total_committed:.1f}")

pressure = pressure_points(instance, solution, end_day=10)
print(f"breaching_tanks,{len(pressure)}")
need = 0
for p in pressure:
    cust = instance.customer_by_point[p.customer]
    # Round trip from base plus setup is the minimum marginal cost of serving it.
    rt = (instance.time_matrix[instance.base_index][cust.index]
          + instance.time_matrix[cust.index][instance.base_index])
    need += rt + cust.setup_time
    print(f"  customer,{cust.index},round_trip_plus_setup,{rt + cust.setup_time}")
print(f"minimum_extra_resource_minutes_needed,{need}")
