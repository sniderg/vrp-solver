from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

from vrp_solver.contest import score_prefix_with_feasibility_tail
from vrp_solver.model import Operation, Shift, Solution
from vrp_solver.rules import derive_solution, validate_solution
from vrp_solver.solver.targeted_rescue import normalize_source_loads
from vrp_solver.xml_io import load_instance, load_solution, save_solution


ROOT = Path(__file__).resolve().parents[1]
INSTANCE = ROOT / "roadef_2016_data/set_B/Instances_B_V25-11042016/V2.12.xml"
START = ROOT / "scratch/b_v2_replace_neighborhood/V2.12/best.xml"
OUT = ROOT / "scratch/b_v2_insert_callins/V2.12"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    instance = load_instance(INSTANCE)
    best = load_solution(START)
    best_score = score(instance, best)
    print("start", best_score.feasibility_errors, best_score.hard_violations, callin_errors(instance, best), flush=True)
    improved = True
    round_index = 0
    while improved and round_index < 4:
        improved = False
        round_index += 1
        for point in callin_errors(instance, best):
            candidate = best_insertion_for_point(instance, best, point, best_score.feasibility_errors)
            if candidate is None:
                continue
            best, best_score = candidate
            improved = True
            out = OUT / f"round{round_index}_{point}_{best_score.feasibility_errors}.xml"
            save_solution(best, out)
            print("accept", point, best_score.feasibility_errors, out, flush=True)
            break
    final = OUT / "best.xml"
    save_solution(best, final)
    print("final", best_score.feasibility_errors, best_score.hard_violations, final, flush=True)


def best_insertion_for_point(instance, solution: Solution, point: int, current_errors: int):
    customer = instance.customer_by_point[point]
    order = customer.orders[0]
    quantity = order.min_quantity_to_satisfy
    best = None
    best_score = None
    tried = 0
    for shift_index, shift in enumerate(solution.shifts):
        if shift.trailer not in customer.allowed_trailers:
            continue
        if not any(operation.point in instance.source_by_point for operation in shift.operations):
            continue
        for insert_pos in range(len(shift.operations) + 1):
            trial_shift = insert_operation(instance, shift, insert_pos, point, quantity)
            if trial_shift is None:
                continue
            tried += 1
            trial_shifts = list(solution.shifts)
            trial_shifts[shift_index] = trial_shift
            trial = normalize_source_loads(
                instance,
                Solution(tuple(replace(s, index=i) for i, s in enumerate(trial_shifts))),
            )
            trial_score = score(instance, trial)
            if trial_score.hard_violations != 0:
                continue
            if structural_errors(instance, trial):
                continue
            if trial_score.feasibility_errors >= current_errors:
                continue
            if best_score is None or trial_score.feasibility_errors < best_score.feasibility_errors:
                best = trial
                best_score = trial_score
    if tried:
        print("point", point, "tried", tried, "best", None if best_score is None else best_score.feasibility_errors, flush=True)
    if best is None:
        return None
    return best, best_score


def insert_operation(instance, shift: Shift, insert_pos: int, point: int, quantity: float) -> Shift | None:
    ops = list(shift.operations)
    order = instance.customer_by_point[point].orders[0]
    desired = order.earliest_time
    if insert_pos > 0:
        desired = max(desired, ops[insert_pos - 1].arrival)
    if insert_pos < len(ops):
        desired = min(max(desired, order.earliest_time), ops[insert_pos].arrival)
    ops.insert(insert_pos, Operation(point, desired, quantity))
    rescheduled = reschedule(instance, shift, ops)
    if rescheduled is None:
        return None
    new_shift = replace(shift, operations=tuple(rescheduled))
    derived = derive_solution(instance, Solution((replace(new_shift, index=0),)))[0]
    driver = instance.drivers[shift.driver]
    if derived.layovers:
        return None
    if any(op.driving_since_layover > driver.max_driving_duration for op in derived.operations):
        return None
    return new_shift


def reschedule(instance, shift: Shift, operations: list[Operation]) -> list[Operation] | None:
    current_time = shift.start
    current_point = instance.base_index
    out: list[Operation] = []
    for operation in operations:
        arrival = max(operation.arrival, current_time + instance.time_matrix[current_point][operation.point])
        customer = instance.customer_by_point.get(operation.point)
        if customer is not None:
            latest = customer.orders[0].latest_time if customer.call_in and customer.orders else None
            earliest = customer.orders[0].earliest_time if customer.call_in and customer.orders else None
            placed = None
            for window in customer.time_windows:
                candidate = max(arrival, window.start)
                if earliest is not None:
                    candidate = max(candidate, earliest)
                if candidate + customer.setup_time > window.end:
                    continue
                if latest is not None and candidate + customer.setup_time > latest:
                    continue
                placed = candidate
                break
            if placed is None:
                return None
            arrival = placed
        out.append(replace(operation, arrival=arrival))
        current_time = arrival + instance.setup_time_for_point(operation.point)
        current_point = operation.point
    return out


def callin_errors(instance, solution):
    return [
        violation.point
        for violation in validate_solution(instance, solution)
        if violation.severity == "error" and violation.code == "QS01" and violation.point is not None
    ]


def structural_errors(instance, solution):
    return [
        violation
        for violation in validate_solution(instance, solution)
        if violation.severity == "error" and violation.code not in {"QS01", "QS02"}
    ]


def score(instance, solution):
    return score_prefix_with_feasibility_tail(
        instance,
        solution,
        score_days=10,
        feasibility_days=10,
        ignore_tail_call_ins=True,
    )


if __name__ == "__main__":
    main()
