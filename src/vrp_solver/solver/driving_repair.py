"""Small, validator-driven repairs for return-leg driving violations.

The XML format has no explicit ``rest`` operation.  A legal layover is instead
derived from a sufficiently long gap before an operation.  This module adds a
compatible source visit before the final customer, allowing the required gap
to be represented while keeping trailer quantities unchanged.  It is useful
both for cold starts and for route-column post-processing.
"""
from __future__ import annotations

from dataclasses import replace

from ..highs_time_opt import _feasible_operation_windows
from ..inventory import delivery_by_customer_step, project_customer_inventory
from ..model import Instance, Operation, Shift, Solution
from ..rules import derive_solution, validate_solution


EPSILON = 1e-6


def repair_return_driving(instance: Instance, solution: Solution) -> Solution:
    """Repair DRI03 final-return violations when one source/rest insertion fits.

    Each prospective mutation is checked against the native rule model and is
    accepted only if it strictly reduces hard (non-QS01/QS02) violations and
    does not introduce any new violation type.  The released checker remains
    the final gate, but this makes the repair fail closed during construction.
    """
    current = solution
    current_violations = validate_solution(instance, current)
    for violation in tuple(current_violations):
        if violation.code != "DRI03" or violation.shift is None:
            continue
        shift = next((item for item in current.shifts if item.index == violation.shift), None)
        if shift is None or violation.operation != len(shift.operations) - 1:
            continue
        candidates = _insert_source_layover(instance, current, shift)
        candidates.extend(_rebuild_tail_with_layover_anchor(instance, current, shift))
        ranked = []
        for repaired in candidates:
            candidate_violations = validate_solution(instance, repaired)
            if _improves(current_violations, candidate_violations):
                ranked.append((candidate_violations, repaired))
        if ranked:
            current_violations, current = min(ranked, key=lambda item: _score(item[0]))
    return current


def _insert_source_layover(instance: Instance, solution: Solution, shift: Shift) -> list[Solution]:
    if not shift.operations:
        return []
    final = shift.operations[-1]
    customer = instance.customer_by_point.get(final.point)
    if customer is None or not customer.layover_customer:
        return []
    driver = instance.drivers[shift.driver]
    trailer = instance.trailers[shift.trailer]
    prefix = shift.operations[:-1]
    previous_point = prefix[-1].point if prefix else instance.base_index
    previous_departure = (
        prefix[-1].arrival + instance.setup_time_for_point(prefix[-1].point)
        if prefix
        else shift.start
    )
    windows = _feasible_operation_windows(instance, final)
    for source in sorted(
        (item for item in instance.sources if trailer.index in item.allowed_trailers),
        key=lambda item: instance.time_matrix[previous_point][item.index] + instance.time_matrix[item.index][final.point],
    ):
        source_arrival = previous_departure + instance.time_matrix[previous_point][source.index]
        source_departure = source_arrival + source.setup_time
        raw_final_arrival = (
            source_departure
            + instance.time_matrix[source.index][final.point]
            + driver.layover_duration
        )
        arrival = _first_window_arrival(raw_final_arrival, windows, instance.setup_time_for_point(final.point))
        if arrival is None:
            continue
        # The rest resets the final driving spell.  The final customer and
        # return leg must then fit inside the driver's continuous-drive limit.
        if (
            instance.time_matrix[source.index][final.point]
            + instance.time_matrix[final.point][instance.base_index]
            > driver.max_driving_duration
        ):
            continue
        source_op = Operation(point=source.index, arrival=source_arrival, quantity=0.0)
        candidate_shift = replace(shift, operations=prefix + (source_op, replace(final, arrival=arrival)))
        candidate_shifts = tuple(candidate_shift if item.index == shift.index else item for item in solution.shifts)
        return [Solution(shifts=candidate_shifts)]
    return []


def _rebuild_tail_with_layover_anchor(instance: Instance, solution: Solution, shift: Shift) -> list[Solution]:
    """Insert a serviceable layover anchor before an otherwise unreachable tail.

    A far customer may be impossible as ``source -> customer -> base`` even
    with a rest.  The native oracle's characteristic pattern is instead
    ``source -> layover customer -> far customer -> base``.  The layover is
    represented by the gap before the anchor; after it, the anchor-to-tail
    driving spell fits the driver's limit.
    """
    if not shift.operations:
        return []
    final = shift.operations[-1]
    final_customer = instance.customer_by_point.get(final.point)
    if final_customer is None or final_customer.call_in:
        return []
    driver = instance.drivers[shift.driver]
    trailer = instance.trailers[shift.trailer]
    prefix = shift.operations[:-1]
    previous_point = prefix[-1].point if prefix else instance.base_index
    previous_departure = prefix[-1].arrival + instance.setup_time_for_point(prefix[-1].point) if prefix else shift.start
    derived = next(item for item in derive_solution(instance, solution) if item.shift.index == shift.index)
    driving_before_source = derived.operations[len(prefix) - 1].driving_since_layover if prefix else 0
    deliveries = delivery_by_customer_step(solution)
    candidates: list[Solution] = []
    for source in instance.sources:
        if trailer.index not in source.allowed_trailers:
            continue
        to_source = instance.time_matrix[previous_point][source.index]
        if driving_before_source + to_source > driver.max_driving_duration:
            continue
        source_arrival = previous_departure + to_source
        source_departure = source_arrival + source.setup_time
        # Do not refill here.  A refill changes the persistent trailer state
        # and can invalidate every later shift using that trailer.  Instead,
        # the source is a legal waypoint before the derived rest and the
        # anchor receives payload reassigned from the final VMI delivery.
        source_load = 0.0
        for anchor in instance.customers:
            if (
                anchor.call_in
                or not anchor.layover_customer
                or trailer.index not in anchor.allowed_trailers
                or anchor.index == final.point
            ):
                continue
            # Before the rest, source-to-anchor must be drivable; after it,
            # anchor-to-tail-to-base must be drivable.
            if instance.time_matrix[source.index][anchor.index] > driver.max_driving_duration:
                continue
            if (
                instance.time_matrix[anchor.index][final.point]
                + instance.time_matrix[final.point][instance.base_index]
                > driver.max_driving_duration
            ):
                continue
            anchor_arrival = _first_window_arrival(
                source_departure + instance.time_matrix[source.index][anchor.index] + driver.layover_duration,
                anchor.time_windows,
                anchor.setup_time,
            )
            if anchor_arrival is None:
                continue
            final_arrival = _first_window_arrival(
                anchor_arrival + anchor.setup_time + instance.time_matrix[anchor.index][final.point],
                _feasible_operation_windows(instance, final),
                final_customer.setup_time,
            )
            if final_arrival is None:
                continue
            events = project_customer_inventory(instance, anchor, deliveries.get(anchor.index, {}))
            inventory = events[min(instance.horizon - 1, anchor_arrival // instance.unit)].after_consumption
            transferable = max(0.0, final.quantity - final_customer.min_operation_quantity)
            anchor_quantity = min(transferable, max(0.0, anchor.capacity - inventory))
            if anchor_quantity + EPSILON < anchor.min_operation_quantity:
                continue
            operations = prefix + (
                Operation(source.index, source_arrival, -source_load),
                Operation(anchor.index, anchor_arrival, anchor_quantity),
                replace(final, arrival=final_arrival, quantity=final.quantity - anchor_quantity),
            )
            candidate_shift = replace(shift, operations=operations)
            candidate_shifts = tuple(candidate_shift if item.index == shift.index else item for item in solution.shifts)
            candidates.append(Solution(shifts=candidate_shifts))
    return candidates


def _first_window_arrival(raw_arrival: int, windows, setup: int) -> int | None:
    for window in windows:
        arrival = max(raw_arrival, window.start)
        if arrival + setup <= window.end:
            return arrival
    return None


def _improves(before, after) -> bool:
    def hard(violations):
        return [item for item in violations if item.code not in {"QS01", "QS02"}]

    before_hard = hard(before)
    after_hard = hard(after)
    before_codes = {item.code for item in before}
    after_codes = {item.code for item in after}
    return len(after_hard) < len(before_hard) and after_codes.issubset(before_codes)


def _score(violations) -> tuple[int, int, int]:
    return (
        len([item for item in violations if item.code not in {"QS01", "QS02"}]),
        len([item for item in violations if item.code == "QS01"]),
        len([item for item in violations if item.code == "QS02"]),
    )
