from __future__ import annotations

from dataclasses import dataclass, replace

from .analysis import summarize_solution
from .contest import score_prefix_with_feasibility_tail
from .model import Instance, Solution
from .rules import derive_solution, is_driving_duration_valid, is_time_window_valid, is_trailer_allowed


@dataclass(frozen=True)
class PruneReport:
    initial_shifts: int
    final_shifts: int
    removed_shifts: tuple[int, ...]
    initial_cost: float
    final_cost: float
    feasible: bool

    def flat(self) -> dict[str, object]:
        row = self.__dict__.copy()
        row["removed_shifts"] = " ".join(str(index) for index in self.removed_shifts)
        return row


def prune_redundant_shifts(
    instance: Instance,
    solution: Solution,
    *,
    score_days: int,
    feasibility_days: int | None = None,
    ignore_tail_call_ins: bool = False,
    max_passes: int = 3,
) -> tuple[Solution, PruneReport]:
    """Remove whole shifts when contest feasibility survives.

    This is intentionally conservative. It does not reinsert operations or alter
    timing; it only deletes shifts that are redundant under the contest metric.
    """

    current = solution
    initial = score_prefix_with_feasibility_tail(
        instance,
        current,
        score_days=score_days,
        feasibility_days=feasibility_days,
        ignore_tail_call_ins=ignore_tail_call_ins,
    )
    removed: list[int] = []

    for _ in range(max_passes):
        changed = False
        for shift_index in _removal_order(instance, current):
            candidate = Solution(
                shifts=tuple(
                    shift
                    for shift in current.shifts
                    if shift.index != shift_index
                )
            )
            score = score_prefix_with_feasibility_tail(
                instance,
                candidate,
                score_days=score_days,
                feasibility_days=feasibility_days,
                ignore_tail_call_ins=ignore_tail_call_ins,
            )
            if not score.feasible:
                continue
            current = candidate
            removed.append(shift_index)
            changed = True
            break
        if not changed:
            break

    final = score_prefix_with_feasibility_tail(
        instance,
        current,
        score_days=score_days,
        feasibility_days=feasibility_days,
        ignore_tail_call_ins=ignore_tail_call_ins,
    )
    return current, PruneReport(
        initial_shifts=initial.scored_shifts,
        final_shifts=final.scored_shifts,
        removed_shifts=tuple(removed),
        initial_cost=initial.scored_estimated_cost,
        final_cost=final.scored_estimated_cost,
        feasible=final.feasible,
    )


def trim_redundant_deliveries(
    instance: Instance,
    solution: Solution,
    *,
    score_days: int,
    feasibility_days: int | None = None,
    ignore_tail_call_ins: bool = False,
    step_fraction: float = 0.25,
    max_rounds: int = 5,
) -> Solution:
    """Conservatively reduce delivery quantities while preserving feasibility."""

    current = solution
    for _ in range(max_rounds):
        changed = False
        for shift in sorted(current.shifts, key=lambda item: item.index, reverse=True):
            for operation_index in range(len(shift.operations) - 1, -1, -1):
                operation = shift.operations[operation_index]
                customer = instance.customer_by_point.get(operation.point)
                if customer is None or customer.call_in or operation.quantity <= 0:
                    continue
                reduction = max(customer.min_operation_quantity, operation.quantity * step_fraction)
                if operation.quantity - reduction <= 0:
                    continue
                candidate = _replace_operation_quantity(
                    current,
                    shift.index,
                    operation_index,
                    operation.quantity - reduction,
                )
                score = score_prefix_with_feasibility_tail(
                    instance,
                    candidate,
                    score_days=score_days,
                    feasibility_days=feasibility_days,
                    ignore_tail_call_ins=ignore_tail_call_ins,
                )
                if not score.feasible:
                    continue
                current = candidate
                changed = True
                break
            if changed:
                break
        if not changed:
            break
    return current


def remove_redundant_source_visits(
    instance: Instance,
    solution: Solution,
    *,
    score_days: int,
    feasibility_days: int | None = None,
    ignore_tail_call_ins: bool = False,
    max_rounds: int = 10,
) -> Solution:
    """Remove source loading operations when feasibility survives."""

    current = solution
    for _ in range(max_rounds):
        changed = False
        for shift in sorted(current.shifts, key=lambda item: item.index, reverse=True):
            for operation_index in range(len(shift.operations) - 1, -1, -1):
                operation = shift.operations[operation_index]
                if operation.point not in instance.source_by_point or operation.quantity >= 0:
                    continue
                candidate = _remove_operation(current, shift.index, operation_index)
                score = score_prefix_with_feasibility_tail(
                    instance,
                    candidate,
                    score_days=score_days,
                    feasibility_days=feasibility_days,
                    ignore_tail_call_ins=ignore_tail_call_ins,
                )
                if not score.feasible:
                    continue
                current = candidate
                changed = True
                break
            if changed:
                break
        if not changed:
            break
    return current


def move_single_customer_shifts(
    instance: Instance,
    solution: Solution,
    *,
    score_days: int,
    feasibility_days: int | None = None,
    ignore_tail_call_ins: bool = False,
    max_moves: int = 10,
) -> Solution:
    """Try to absorb one-customer shifts into existing shifts.

    The move is intentionally narrow: append the single delivery from a source
    shift onto the tail of another shift whose trailer already has enough final
    load. This keeps the edit small enough that full contest revalidation can
    police the edge cases.
    """

    current = solution
    moves = 0
    while moves < max_moves:
        move = _find_single_shift_move(
            instance,
            current,
            score_days=score_days,
            feasibility_days=feasibility_days,
            ignore_tail_call_ins=ignore_tail_call_ins,
        )
        if move is None:
            break
        current = move
        moves += 1
    return current


def _replace_operation_quantity(
    solution: Solution,
    shift_index: int,
    operation_index: int,
    quantity: float,
) -> Solution:
    shifts = []
    for shift in solution.shifts:
        if shift.index != shift_index:
            shifts.append(shift)
            continue
        operations = list(shift.operations)
        operations[operation_index] = replace(operations[operation_index], quantity=quantity)
        shifts.append(replace(shift, operations=tuple(operations)))
    return Solution(shifts=tuple(shifts))


def _remove_operation(
    solution: Solution,
    shift_index: int,
    operation_index: int,
) -> Solution:
    shifts = []
    for shift in solution.shifts:
        if shift.index != shift_index:
            shifts.append(shift)
            continue
        operations = list(shift.operations)
        operations.pop(operation_index)
        shifts.append(replace(shift, operations=tuple(operations)))
    return Solution(shifts=tuple(shifts))


def _find_single_shift_move(
    instance: Instance,
    solution: Solution,
    *,
    score_days: int,
    feasibility_days: int | None,
    ignore_tail_call_ins: bool,
) -> Solution | None:
    derived_by_index = {derived.shift.index: derived for derived in derive_solution(instance, solution)}
    summaries = {summary.index: summary for summary in summarize_solution(instance, solution)}
    shifts_by_index = {shift.index: shift for shift in solution.shifts}

    candidate_source_indices = [
        summary.index
        for summary in sorted(
            summaries.values(),
            key=lambda item: (-item.estimated_cost, item.delivered_quantity, item.operations),
        )
        if _is_single_customer_shift(instance, shifts_by_index[summary.index])
    ]
    for source_shift_index in candidate_source_indices:
        source_shift = shifts_by_index[source_shift_index]
        delivery_index = next(
            index
            for index, operation in enumerate(source_shift.operations)
            if operation.quantity > 0 and operation.point in instance.customer_by_point
        )
        delivery = source_shift.operations[delivery_index]
        customer = instance.customer_by_point[delivery.point]

        target_summaries = sorted(
            (summary for summary in summaries.values() if summary.index != source_shift_index),
            key=lambda item: (item.estimated_cost, -item.final_load),
        )
        for target_summary in target_summaries:
            if target_summary.final_load + 1e-6 < delivery.quantity:
                continue
            target_shift = shifts_by_index[target_summary.index]
            if not is_trailer_allowed(instance, delivery.point, target_shift.trailer):
                continue
            appended = _append_delivery_if_feasible(
                instance,
                derived_by_index[target_shift.index],
                delivery.point,
                delivery.quantity,
            )
            if appended is None:
                continue
            candidate = _move_single_delivery(solution, source_shift_index, target_shift.index, delivery_index, appended)
            score = score_prefix_with_feasibility_tail(
                instance,
                candidate,
                score_days=score_days,
                feasibility_days=feasibility_days,
                ignore_tail_call_ins=ignore_tail_call_ins,
            )
            if score.feasible:
                return candidate
    return None


def _is_single_customer_shift(instance: Instance, shift) -> bool:
    delivery_count = sum(
        1
        for operation in shift.operations
        if operation.quantity > 0 and operation.point in instance.customer_by_point
    )
    return delivery_count == 1


def _append_delivery_if_feasible(instance: Instance, derived_shift, point: int, quantity: float):
    shift = derived_shift.shift
    driver = instance.drivers[shift.driver]
    if not derived_shift.operations:
        return None

    last_derived = derived_shift.operations[-1]
    last_point = shift.operations[-1].point
    arrival = last_derived.departure + instance.time_matrix[last_point][point]
    departure = arrival + instance.setup_time_for_point(point)
    return_time = instance.time_matrix[point][instance.base_index]
    new_end = departure + return_time

    customer = instance.customer_by_point[point]
    if not is_time_window_valid(arrival, departure, customer.time_windows):
        return None
    if not is_time_window_valid(shift.start, new_end, driver.time_windows):
        return None
    if not is_driving_duration_valid(
        driver,
        last_derived.driving_since_layover + instance.time_matrix[last_point][point] + return_time,
    ):
        return None
    return replace(
        shift.operations[-1],
        point=point,
        arrival=arrival,
        quantity=quantity,
    )


def _move_single_delivery(
    solution: Solution,
    source_shift_index: int,
    target_shift_index: int,
    delivery_index: int,
    appended_operation,
) -> Solution:
    new_shifts = []
    for shift in solution.shifts:
        if shift.index == source_shift_index:
            if delivery_index == -1:
                continue
            remaining = [
                operation
                for index, operation in enumerate(shift.operations)
                if index != delivery_index and not (operation.quantity < 0)
            ]
            if remaining:
                new_shifts.append(replace(shift, operations=tuple(remaining)))
            continue
        if shift.index == target_shift_index:
            operations = list(shift.operations)
            operations.append(appended_operation)
            new_shifts.append(replace(shift, operations=tuple(operations)))
            continue
        new_shifts.append(shift)
    return Solution(shifts=tuple(new_shifts))


def _removal_order(instance: Instance, solution: Solution) -> list[int]:
    summaries = summarize_solution(instance, solution)
    return [
        summary.index
        for summary in sorted(
            summaries,
            key=lambda item: (
                item.delivered_quantity,
                -item.estimated_cost,
                item.operations,
            ),
        )
    ]
