from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from vrp_solver.contest import score_prefix_with_feasibility_tail
from vrp_solver.inventory import delivery_by_customer_step, project_customer_inventory, tank_violations
from vrp_solver.model import Operation, Shift, Solution
from vrp_solver.rules import is_time_window_valid, validate_solution
from vrp_solver.solver.targeted_rescue import normalize_source_loads
from vrp_solver.xml_io import load_instance, load_solution, save_solution


ROOT = Path(__file__).resolve().parents[1]
INSTANCE = ROOT / "roadef_2016_data/set_B/Instances_B_V25-11042016/V2.12.xml"
START = ROOT / "scratch/b_v2_replace_neighborhood/V2.12/best.xml"
OUT = ROOT / "scratch/b_v2_split_residual_tanks/V2.12"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    instance = load_instance(INSTANCE)
    best = load_solution(START)
    best_score = score(instance, best)
    print("start", best_score.feasibility_errors, best_score.hard_violations)
    improved = True
    round_index = 0
    while improved and round_index < 4:
        improved = False
        round_index += 1
        points = [v.point for v in tank_violations(instance, best)]
        for point in sorted(set(points), key=lambda p: points.count(p), reverse=True):
            candidate = best_point_split(instance, best, point, best_score.feasibility_errors)
            if candidate is None:
                continue
            best, best_score = candidate
            improved = True
            out = OUT / f"round{round_index}_{point}_{best_score.feasibility_errors}.xml"
            save_solution(best, out)
            print("accept", "round", round_index, "point", point, "errors", best_score.feasibility_errors, out)
            break
    final = OUT / "best.xml"
    save_solution(best, final)
    print("final", best_score.feasibility_errors, best_score.hard_violations, final)


def best_point_split(instance, solution: Solution, point: int, current_errors: int):
    customer = instance.customer_by_point[point]
    deliveries = delivery_by_customer_step(solution).get(point, {})
    events = project_customer_inventory(instance, customer, deliveries)
    breach = next((event for event in events if event.safety_breach), None)
    if breach is None:
        return None
    best = None
    best_score = None
    for shift_index, shift in enumerate(solution.shifts):
        for op_index, operation in enumerate(shift.operations):
            if operation.point != point or operation.quantity <= 0 or operation.arrival <= breach.time_start:
                continue
            windows = [
                window
                for window in customer.time_windows
                if shift.start <= window.start <= operation.arrival
                or shift.start <= window.end <= operation.arrival
            ]
            for window in windows:
                for arrival in {window.start, max(window.start, breach.time_start), max(window.start, operation.arrival - 1440), max(window.start, operation.arrival - 720)}:
                    if arrival >= operation.arrival:
                        continue
                    if not is_time_window_valid(arrival, arrival + customer.setup_time, customer.time_windows):
                        continue
                    for qty in quantities(customer, operation.quantity):
                        trial_shift = insert_and_reduce(instance, shift, op_index, arrival, qty)
                        if trial_shift is None:
                            continue
                        trial_shifts = list(solution.shifts)
                        trial_shifts[shift_index] = trial_shift
                        trial = normalize_source_loads(
                            instance,
                            Solution(shifts=tuple(replace(s, index=i) for i, s in enumerate(trial_shifts))),
                        )
                        trial_score = score(instance, trial)
                        if trial_score.hard_violations != 0:
                            continue
                        if has_structural_errors(instance, trial):
                            continue
                        if trial_score.feasibility_errors >= current_errors:
                            continue
                        if best_score is None or trial_score.feasibility_errors < best_score.feasibility_errors:
                            best = trial
                            best_score = trial_score
    if best is None:
        return None
    return best, best_score


def quantities(customer, available: float) -> list[float]:
    raw = [
        customer.min_operation_quantity,
        min(available - 1e-6, customer.min_operation_quantity * 1.25),
        min(available - 1e-6, customer.min_operation_quantity * 1.5),
        min(available - 1e-6, customer.min_operation_quantity * 2.0),
        min(available - 1e-6, available * 0.25),
        min(available - 1e-6, available * 0.5),
    ]
    return sorted({round(q, 6) for q in raw if q > 1e-6 and q < available - 1e-6})


def insert_and_reduce(instance, shift: Shift, op_index: int, arrival: int, qty: float) -> Shift | None:
    original = shift.operations[op_index]
    if qty >= original.quantity:
        return None
    ops = list(shift.operations)
    ops[op_index] = replace(original, quantity=original.quantity - qty)
    insert_pos = op_index
    while insert_pos > 0 and ops[insert_pos - 1].arrival > arrival:
        insert_pos -= 1
    ops.insert(insert_pos, Operation(point=original.point, arrival=arrival, quantity=qty))
    rescheduled = reschedule(instance, shift, ops)
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


def has_structural_errors(instance, solution: Solution) -> bool:
    return any(
        violation.severity == "error" and violation.code not in {"QS01", "QS02"}
        for violation in validate_solution(instance, solution)
    )


if __name__ == "__main__":
    main()
