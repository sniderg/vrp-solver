from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from vrp_solver.contest import score_prefix_with_feasibility_tail
from vrp_solver.model import Operation, Shift, Solution
from vrp_solver.rules import is_time_window_valid
from vrp_solver.solver.targeted_rescue import normalize_source_loads
from vrp_solver.xml_io import load_instance, load_solution, save_solution


ROOT = Path(__file__).resolve().parents[1]
INSTANCE = ROOT / "roadef_2016_data/set_B/Instances_B_V25-11042016/V2.12.xml"
START = ROOT / "scratch/b_v2_replace_neighborhood/V2.12/best.xml"
OUT = ROOT / "scratch/b_v2_insert_split_109/V2.12"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    instance = load_instance(INSTANCE)
    incumbent = load_solution(START)
    best = incumbent
    best_score = score(instance, incumbent)
    print("start", best_score.feasibility_errors, best_score.hard_violations)
    attempts = 0
    for shift_index, shift in enumerate(incumbent.shifts):
        if not (2500 <= shift.start <= 4300):
            continue
        for insert_pos in range(len(shift.operations) + 1):
            for qty in (426.33864, 500.0, 600.0, 700.0, 800.0, 900.0):
                candidate = insert_into_shift(instance, shift, insert_pos, qty)
                if candidate is None:
                    continue
                trial_shifts = list(incumbent.shifts)
                trial_shifts[shift_index] = candidate
                trial_shifts = reduce_late_109(trial_shifts, qty)
                if trial_shifts is None:
                    continue
                attempts += 1
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
                    print("accept", best_score.feasibility_errors, "shift", shift.index, "pos", insert_pos, "qty", qty, out)
    final = OUT / "best.xml"
    save_solution(best, final)
    print("attempts", attempts)
    print("final", best_score.feasibility_errors, best_score.hard_violations, final)


def insert_into_shift(instance, shift: Shift, insert_pos: int, qty: float) -> Shift | None:
    customer = instance.customer_by_point[109]
    operations = list(shift.operations)
    previous_point = instance.base_index if insert_pos == 0 else operations[insert_pos - 1].point
    previous_arrival = shift.start if insert_pos == 0 else operations[insert_pos - 1].arrival
    previous_departure = previous_arrival + instance.setup_time_for_point(previous_point)
    raw_arrival = previous_departure + instance.time_matrix[previous_point][109]
    # Favor the 3690-4020 window because it directly precedes the first breach.
    arrival = max(raw_arrival, 3690)
    if not is_time_window_valid(arrival, arrival + customer.setup_time, customer.time_windows):
        return None
    candidate_ops = operations[:insert_pos] + [Operation(point=109, arrival=arrival, quantity=qty)] + operations[insert_pos:]
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


def reduce_late_109(shifts: list[Shift], qty: float) -> list[Shift] | None:
    remaining = qty
    updated = list(shifts)
    for shift_index in range(len(updated) - 1, -1, -1):
        shift = updated[shift_index]
        ops = list(shift.operations)
        for op_index in range(len(ops) - 1, -1, -1):
            operation = ops[op_index]
            if operation.point != 109 or operation.arrival < 8000 or operation.quantity <= 0:
                continue
            reduction = min(remaining, operation.quantity)
            new_qty = operation.quantity - reduction
            if new_qty <= 1e-6:
                ops.pop(op_index)
            else:
                ops[op_index] = replace(operation, quantity=new_qty)
            remaining -= reduction
            updated[shift_index] = replace(shift, operations=tuple(ops))
            if remaining <= 1e-6:
                return updated
    return None


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
