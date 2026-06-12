from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from vrp_solver.contest import score_prefix_with_feasibility_tail
from vrp_solver.inventory import delivery_by_customer_step, project_customer_inventory
from vrp_solver.model import Operation, Shift, Solution
from vrp_solver.rules import derive_solution, is_time_window_valid, validate_solution
from vrp_solver.solver.targeted_rescue import normalize_source_loads
from vrp_solver.xml_io import load_instance, load_solution, save_solution


ROOT = Path(__file__).resolve().parents[1]
INSTANCE = ROOT / "roadef_2016_data/set_B/Instances_B_V25-11042016/V2.12.xml"
START = ROOT / "scratch/b_v2_replace_neighborhood/V2.12/best.xml"
OUT = ROOT / "scratch/b_v2_insert_109/V2.12"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    instance = load_instance(INSTANCE)
    incumbent = load_solution(START)
    best = incumbent
    best_score = score(instance, incumbent)
    print("start", best_score.feasibility_errors, best_score.hard_violations)
    attempts = 0
    for shift_index, shift in enumerate(incumbent.shifts):
        # Only shifts plausibly overlapping 109's pre-breach service windows.
        if not (2000 <= shift.start <= 4300):
            continue
        for insert_pos in range(len(shift.operations) + 1):
            for target_window_start in (2250, 3690):
                trial_shift = insert_customer(instance, incumbent, shift, insert_pos, target_window_start)
                if trial_shift is None:
                    continue
                attempts += 1
                trial_shifts = list(incumbent.shifts)
                trial_shifts[shift_index] = trial_shift
                trial = normalize_source_loads(
                    instance,
                    Solution(shifts=tuple(replace(s, index=i) for i, s in enumerate(trial_shifts))),
                )
                trial_score = score(instance, trial)
                if trial_score.hard_violations == 0 and trial_score.feasibility_errors < best_score.feasibility_errors:
                    best = trial
                    best_score = trial_score
                    out = OUT / f"best_{best_score.feasibility_errors}.xml"
                    save_solution(best, out)
                    print(
                        "accept",
                        best_score.feasibility_errors,
                        "shift",
                        shift.index,
                        "pos",
                        insert_pos,
                        "window",
                        target_window_start,
                        out,
                    )
    final = OUT / "best.xml"
    save_solution(best, final)
    print("attempts", attempts)
    print("final", best_score.feasibility_errors, best_score.hard_violations, final)


def insert_customer(
    instance,
    solution: Solution,
    shift: Shift,
    insert_pos: int,
    target_window_start: int,
) -> Shift | None:
    customer = instance.customer_by_point[109]
    deliveries = delivery_by_customer_step(solution).get(109, {})
    # Remove current late 109 delivery when estimating room for a potential earlier duplicate.
    deliveries = {arrival: qty for arrival, qty in deliveries.items() if arrival < shift.start or arrival > shift.start + 3000}
    operations = list(shift.operations)
    previous_point = instance.base_index if insert_pos == 0 else operations[insert_pos - 1].point
    previous_arrival = shift.start if insert_pos == 0 else operations[insert_pos - 1].arrival
    previous_departure = previous_arrival + instance.setup_time_for_point(previous_point)
    raw_arrival = previous_departure + instance.time_matrix[previous_point][109]
    arrival = max(raw_arrival, target_window_start)
    if not is_time_window_valid(arrival, arrival + customer.setup_time, customer.time_windows):
        return None
    if arrival >= 10 * 1440:
        return None
    events = project_customer_inventory(instance, customer, deliveries)
    step = min(max(arrival // instance.unit, 0), instance.horizon - 1)
    inv = events[step].after_consumption
    room = customer.capacity - inv
    if room < customer.min_operation_quantity:
        return None
    needed = max(customer.min_operation_quantity, customer.safety_level - inv + 900.0)
    quantity = min(room, needed)
    if quantity < customer.min_operation_quantity:
        return None
    candidate_ops = operations[:insert_pos] + [Operation(point=109, arrival=arrival, quantity=quantity)] + operations[insert_pos:]
    rescheduled = reschedule(instance, shift, candidate_ops)
    if rescheduled is None:
        return None
    return replace(shift, operations=tuple(rescheduled))


def reschedule(instance, shift: Shift, operations: list[Operation]) -> list[Operation] | None:
    current_time = shift.start
    current_point = instance.base_index
    out: list[Operation] = []
    for operation in operations:
        arrival = max(operation.arrival, current_time + instance.time_matrix[current_point][operation.point])
        if operation.point in instance.customer_by_point:
            customer = instance.customer_by_point[operation.point]
            if not is_time_window_valid(arrival, arrival + customer.setup_time, customer.time_windows):
                return None
        out.append(replace(operation, arrival=arrival))
        current_time = arrival + instance.setup_time_for_point(operation.point)
        current_point = operation.point
    return out


def score(instance, solution: Solution):
    return score_prefix_with_feasibility_tail(
        instance,
        solution,
        score_days=10,
        feasibility_days=10,
        ignore_tail_call_ins=True,
    )


if __name__ == "__main__":
    main()
