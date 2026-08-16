"""Bounded joint timing repair for a connected driver/trailer block.

This deliberately solves *timing only*.  Route order, resource assignment,
and quantities remain fixed, which keeps the model small enough to place in a
native neighbourhood loop.  Unlike ``try_optimize_shift_times``, resource
successors inside the block are decision variables rather than fixed end
boundaries.
"""
from __future__ import annotations

from dataclasses import replace
import time

import numpy as np

from .highs_time_opt import _driving_valid, _feasible_operation_windows
from .model import Instance, Operation, Shift, Solution
from .rules import derive_solution


def retime_connected_resource_block(
    instance: Instance,
    solution: Solution,
    anchor_shift_index: int,
    *,
    max_shifts: int = 5,
) -> Solution | None:
    """Jointly retime a small resource-connected block around ``anchor``.

    The original order of shifts on every driver and trailer is retained.  The
    block includes nearest predecessor/successor shifts sharing either anchor
    resource, then expands through their resources until the bounded size is
    reached.  Unchanged neighbouring shifts become hard boundary constraints.
    """
    return retime_resource_blocks(
        instance, solution, (anchor_shift_index,), max_shifts=max_shifts,
    )


def retime_resource_blocks(
    instance: Instance,
    solution: Solution,
    anchor_shift_indices: tuple[int, ...],
    *,
    max_shifts: int = 8,
) -> Solution | None:
    """Jointly retime the union of several connected resource components."""
    by_index = {shift.index: shift for shift in solution.shifts}
    anchors = [index for index in anchor_shift_indices if index in by_index]
    if len(anchors) != len(anchor_shift_indices):
        return None
    derived = {item.shift.index: item for item in derive_solution(instance, solution)}
    selected = set(anchors)
    frontier = list(anchors)
    while frontier and len(selected) < max_shifts:
        current = by_index[frontier.pop(0)]
        neighbours = [
            shift for shift in solution.shifts
            if shift.index not in selected
            and (shift.driver == current.driver or shift.trailer == current.trailer)
        ]
        neighbours.sort(
            key=lambda shift: (
                abs(shift.start - current.start), shift.start, shift.index,
            ),
        )
        for neighbour in neighbours:
            if len(selected) >= max_shifts:
                break
            selected.add(neighbour.index)
            frontier.append(neighbour.index)

    selected_shifts = sorted(
        (by_index[index] for index in selected), key=lambda shift: shift.index,
    )
    return _solve_joint_timing(instance, solution, selected_shifts, derived)


def generate_pressure_block_insertions(
    instance: Instance,
    solution: Solution,
    *,
    customer_point: int,
    first_minute: int,
    radius: int = 4_320,
    max_candidates: int = 24,
    max_block_shifts: int = 5,
    deadline: float | None = None,
) -> list[Solution]:
    """Insert an early VMI duplicate into nearby compatible route chains.

    This is a topology generator, not a quantity repair.  It deliberately
    retains any existing (possibly late) visit: it may be a layover-enabling
    stop.  The caller's hard quantity model later chooses active quantities.
    Each proposed insertion is jointly retimed with its driver/trailer block.
    """
    customer = instance.customer_by_point.get(customer_point)
    if customer is None or customer.call_in:
        return []
    seed_quantity = max(customer.min_operation_quantity, 10.0e-6)
    candidates: list[Solution] = []
    seen: set[tuple[tuple[int, int, tuple[tuple[int, int], ...]], ...]] = set()
    recipients = [
        shift for shift in solution.shifts
        if shift.trailer in customer.allowed_trailers
        and shift.start <= first_minute
        and first_minute - shift.start <= radius
    ]
    recipients.sort(key=lambda shift: (abs(shift.start - first_minute), shift.start, shift.index))
    for recipient in recipients:
        if deadline is not None and time.monotonic() >= deadline:
            return candidates
        # Try internal gaps first: this is where waiting time can absorb a
        # delivery even when the route tail is saturated.
        positions = list(range(1, len(recipient.operations))) + [0, len(recipient.operations)]
        for position in positions:
            if deadline is not None and time.monotonic() >= deadline:
                return candidates
            patterns = [(Operation(customer_point, first_minute, seed_quantity),)]
            # A duplicate visit without a preceding load may be a useful
            # insertion in an already-loaded segment.  Also enumerate an
            # explicit reload before it, because a route's trailer stock is a
            # path constraint rather than a per-visit quantity bound.
            patterns.extend(
                (
                    Operation(source.index, first_minute, -seed_quantity),
                    Operation(customer_point, first_minute, seed_quantity),
                )
                for source in instance.sources
                if recipient.trailer in source.allowed_trailers
            )
            for pattern in patterns:
                operations = list(recipient.operations)
                operations[position:position] = pattern
                mutated = replace(recipient, operations=tuple(operations))
                topology = Solution(tuple(
                    mutated if shift.index == recipient.index else shift
                    for shift in solution.shifts
                ))
                timed = retime_connected_resource_block(
                    instance,
                    topology,
                    recipient.index,
                    max_shifts=max_block_shifts,
                )
                if timed is None:
                    continue
                inserted = next(
                    shift for shift in timed.shifts if shift.index == recipient.index
                )
                if not any(
                    operation.point == customer_point
                    and operation.arrival <= first_minute
                    for operation in inserted.operations
                ):
                    continue
                fingerprint = tuple(sorted(
                    (
                        shift.index,
                        shift.start,
                        tuple((operation.point, operation.arrival) for operation in shift.operations),
                    ) for shift in timed.shifts
                ))
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                candidates.append(timed)
                if len(candidates) >= max_candidates:
                    return candidates
    return candidates


def generate_pressure_block_substitutions(
    instance: Instance,
    solution: Solution,
    *,
    customer_point: int,
    first_minute: int,
    radius: int = 4_320,
    max_candidates: int = 24,
    max_block_shifts: int = 5,
) -> list[Solution]:
    """Replace an optional nearby VMI stop with an earlier pressure visit.

    This is deliberately different from exchanging the existing late visit:
    the late visit remains in place for route/layover structure while the hard
    quantity repair may deactivate or rebalance the displaced optional stop.
    """
    target = instance.customer_by_point.get(customer_point)
    if target is None or target.call_in:
        return []
    seed_quantity = max(target.min_operation_quantity, 10.0e-6)
    candidates: list[Solution] = []
    seen: set[tuple[tuple[int, int, tuple[tuple[int, int], ...]], ...]] = set()
    slots = [
        (shift, position)
        for shift in solution.shifts
        if shift.trailer in target.allowed_trailers
        for position, operation in enumerate(shift.operations)
        if (
            operation.point != customer_point
            and operation.arrival <= first_minute
            and first_minute - operation.arrival <= radius
            and (
                operation.point in instance.source_by_point
                or (
                    operation.point in instance.customer_by_point
                    and not instance.customer_by_point[operation.point].call_in
                    and not instance.customer_by_point[operation.point].layover_customer
                )
            )
        )
    ]
    slots.sort(key=lambda item: (abs(item[0].operations[item[1]].arrival - first_minute), item[0].start, item[0].index))
    for recipient, position in slots:
        operations = list(recipient.operations)
        operations[position] = Operation(customer_point, first_minute, seed_quantity)
        topology = Solution(tuple(
            replace(recipient, operations=tuple(operations)) if shift.index == recipient.index else shift
            for shift in solution.shifts
        ))
        timed = retime_connected_resource_block(
            instance, topology, recipient.index, max_shifts=max_block_shifts,
        )
        if timed is None:
            continue
        rebuilt = next(shift for shift in timed.shifts if shift.index == recipient.index)
        if not any(
            operation.point == customer_point and operation.arrival <= first_minute
            for operation in rebuilt.operations
        ):
            continue
        fingerprint = tuple(sorted(
            (shift.index, shift.start, tuple((operation.point, operation.arrival) for operation in shift.operations))
            for shift in timed.shifts
        ))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        candidates.append(timed)
        if len(candidates) >= max_candidates:
            return candidates
    return candidates


def generate_pressure_substitution_ejections(
    instance: Instance,
    solution: Solution,
    *,
    customer_point: int,
    first_minute: int,
    radius: int = 4_320,
    max_candidates: int = 24,
    deadline: float | None = None,
) -> list[Solution]:
    """Replace an optional stop, then relocate it into a second route.

    This bounded two-route ejection preserves customer coverage that makes a
    plain pressure substitution quantity-infeasible.
    """
    target = instance.customer_by_point.get(customer_point)
    if target is None or target.call_in:
        return []
    result: list[Solution] = []
    seen: set[tuple[tuple[int, tuple[int, ...]], ...]] = set()
    slots = [
        (shift, position) for shift in solution.shifts
        if shift.trailer in target.allowed_trailers
        for position, operation in enumerate(shift.operations)
        if operation.point in instance.customer_by_point and operation.point != customer_point
        and operation.arrival <= first_minute and first_minute - operation.arrival <= radius
        and not instance.customer_by_point[operation.point].call_in
        and not instance.customer_by_point[operation.point].layover_customer
    ]
    slots.sort(key=lambda item: abs(item[0].operations[item[1]].arrival - first_minute))
    for donor, donor_pos in slots[:12]:
        if deadline is not None and time.monotonic() >= deadline:
            return result
        displaced = donor.operations[donor_pos]
        displaced_customer = instance.customer_by_point[displaced.point]
        donor_ops = list(donor.operations)
        donor_ops[donor_pos] = Operation(
            customer_point, first_minute,
            max(target.min_operation_quantity, 10.0e-6),
        )
        for recipient in solution.shifts:
            if deadline is not None and time.monotonic() >= deadline:
                return result
            if recipient.index == donor.index or recipient.trailer not in displaced_customer.allowed_trailers:
                continue
            if abs(recipient.start - displaced.arrival) > radius:
                continue
            for position in range(len(recipient.operations) + 1):
                if deadline is not None and time.monotonic() >= deadline:
                    return result
                patterns = [(displaced,)]
                patterns.extend(
                    (
                        Operation(source.index, displaced.arrival, -max(
                            displaced_customer.min_operation_quantity,
                            10.0e-6,
                        )),
                        displaced,
                    )
                    for source in instance.sources
                    if recipient.trailer in source.allowed_trailers
                )
                for pattern in patterns:
                    recipient_ops = list(recipient.operations)
                    recipient_ops[position:position] = pattern
                    topology = Solution(tuple(
                        replace(donor, operations=tuple(donor_ops)) if shift.index == donor.index
                        else replace(recipient, operations=tuple(recipient_ops)) if shift.index == recipient.index
                        else shift
                        for shift in solution.shifts
                    ))
                    timed = retime_resource_blocks(
                        instance,
                        topology,
                        (donor.index, recipient.index),
                        max_shifts=8,
                    )
                    if timed is None:
                        continue
                    rebuilt = next(shift for shift in timed.shifts if shift.index == donor.index)
                    if not any(op.point == customer_point and op.arrival <= first_minute for op in rebuilt.operations):
                        continue
                    fingerprint = tuple(sorted((shift.index, tuple(op.point for op in shift.operations)) for shift in timed.shifts))
                    if fingerprint in seen:
                        continue
                    seen.add(fingerprint)
                    result.append(timed)
                    if len(result) >= max_candidates:
                        return result
    return result


def _solve_joint_timing(instance, solution, block, derived) -> Solution | None:
    try:
        import highspy
    except ModuleNotFoundError as exc:
        raise RuntimeError("highspy is not installed; run `uv sync --extra milp`") from exc

    if not block:
        return solution
    block_ids = {shift.index for shift in block}
    windows = {
        shift.index: [_feasible_operation_windows(instance, op) for op in shift.operations]
        for shift in block
    }
    if any(any(not choices for choices in per_shift) for per_shift in windows.values()):
        return None

    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    inf = highspy.kHighsInf
    # Keep every model bounded, and prefer a schedule close to the incumbent.
    starts: dict[int, int] = {}
    arrivals: dict[tuple[int, int], int] = {}
    start_deviations: dict[int, int] = {}
    operation_window_columns: dict[tuple[int, int, int], int] = {}
    driver_window_columns: dict[tuple[int, int], int] = {}

    def add_column(cost: float, lower: float = 0.0, upper: float = inf) -> int:
        column = highs.getNumCol()
        highs.addCol(cost, lower, upper, 0, np.array([], dtype=np.int32), np.array([], dtype=np.float64))
        return column

    for shift in block:
        starts[shift.index] = add_column(0.02)
        start_deviations[shift.index] = add_column(0.02)
        for position, operation in enumerate(shift.operations):
            arrivals[(shift.index, position)] = add_column(0.0)
            add_column(1.0)
            for choice in range(len(windows[shift.index][position])):
                column = add_column(0.0, 0.0, 1.0)
                highs.changeColIntegrality(column, highspy.HighsVarType.kInteger)
                operation_window_columns[(shift.index, position, choice)] = column
        for choice in range(len(instance.drivers[shift.driver].time_windows)):
            column = add_column(0.0, 0.0, 1.0)
            highs.changeColIntegrality(column, highspy.HighsVarType.kInteger)
            driver_window_columns[(shift.index, choice)] = column

    def add_row(lower, upper, columns, values) -> None:
        highs.addRow(
            lower, upper, len(columns), np.array(columns, dtype=np.int32),
            np.array(values, dtype=np.float64),
        )

    # Absolute deviations: starts and arrivals stay near the incumbent when
    # there is no resource reason to move them.
    for shift in block:
        start = starts[shift.index]
        start_dev = start_deviations[shift.index]
        add_row(-inf, shift.start, [start, start_dev], [1.0, -1.0])
        add_row(shift.start, inf, [start, start_dev], [1.0, 1.0])
        for position, operation in enumerate(shift.operations):
            arrival = arrivals[(shift.index, position)]
            # Arrival deviation columns are appended immediately after their
            # arrival.  Looking it up by position avoids exposing model state.
            deviation = arrival + 1
            add_row(-inf, operation.arrival, [arrival, deviation], [1.0, -1.0])
            add_row(operation.arrival, inf, [arrival, deviation], [1.0, 1.0])

    big_m = float(max(30_000, instance.latest_time + 30_000))
    for shift in block:
        driver = instance.drivers[shift.driver]
        start = starts[shift.index]
        # Every entire shift must fit one of the driver's legal windows.
        choices = [driver_window_columns[(shift.index, choice)] for choice in range(len(driver.time_windows))]
        if not choices:
            return None
        add_row(1.0, 1.0, choices, [1.0] * len(choices))
        last_arrival = arrivals[(shift.index, len(shift.operations) - 1)]
        last = shift.operations[-1]
        end_offset = instance.setup_time_for_point(last.point) + instance.time_matrix[last.point][instance.base_index]
        for choice, window in enumerate(driver.time_windows):
            enabled = driver_window_columns[(shift.index, choice)]
            add_row(window.start - big_m, inf, [start, enabled], [1.0, -big_m])
            add_row(-inf, window.end + big_m - end_offset, [last_arrival, enabled], [1.0, big_m])

        replay = derived[shift.index]
        has_layover_customer = any(
            operation.point in instance.customer_by_point
            and instance.customer_by_point[operation.point].layover_customer
            for operation in shift.operations
        )
        layover_before = (
            {
                position for position, operation in enumerate(replay.operations)
                if operation.layover_before
            }
            if has_layover_customer
            else set()
        )
        previous_point = instance.base_index
        for position, operation in enumerate(shift.operations):
            arrival = arrivals[(shift.index, position)]
            feasible = windows[shift.index][position]
            enabled = [operation_window_columns[(shift.index, position, choice)] for choice in range(len(feasible))]
            add_row(1.0, 1.0, enabled, [1.0] * len(enabled))
            setup = instance.setup_time_for_point(operation.point)
            for choice, window in enumerate(feasible):
                selector = operation_window_columns[(shift.index, position, choice)]
                add_row(window.start - big_m, inf, [arrival, selector], [1.0, -big_m])
                add_row(-inf, window.end + big_m - setup, [arrival, selector], [1.0, big_m])
            travel = instance.time_matrix[previous_point][operation.point]
            if position == 0:
                add_row(travel, inf, [arrival, start], [1.0, -1.0])
                add_row(-inf, driver.layover_duration + travel - 1, [arrival, start], [1.0, -1.0])
            else:
                previous = arrivals[(shift.index, position - 1)]
                previous_setup = instance.setup_time_for_point(shift.operations[position - 1].point)
                rest = driver.layover_duration if position in layover_before else 0
                add_row(previous_setup + travel + rest, inf, [arrival, previous], [1.0, -1.0])
                if position not in layover_before:
                    add_row(-inf, previous_setup + travel + driver.layover_duration - 1, [arrival, previous], [1.0, -1.0])
            previous_point = operation.point

    # Preserve original order on every shared resource.  Outside-block shifts
    # remain fixed boundaries; inside-block shifts are coupled decision vars.
    for resource in ("driver", "trailer"):
        resource_ids = {getattr(shift, resource) for shift in block}
        for resource_id in resource_ids:
            ordered = sorted(
                [shift for shift in solution.shifts if getattr(shift, resource) == resource_id],
                key=lambda shift: (shift.start, shift.index),
            )
            for previous, following in zip(ordered, ordered[1:]):
                if previous.index not in block_ids and following.index not in block_ids:
                    continue
                gap = (instance.drivers[resource_id].min_inter_shift_duration if resource == "driver" else 0)
                if previous.index in block_ids:
                    previous_arrival = arrivals[(previous.index, len(previous.operations) - 1)]
                    previous_end_offset = instance.setup_time_for_point(previous.operations[-1].point) + instance.time_matrix[previous.operations[-1].point][instance.base_index]
                    if following.index in block_ids:
                        add_row(previous_end_offset + gap, inf, [starts[following.index], previous_arrival], [1.0, -1.0])
                    else:
                        add_row(-inf, following.start - previous_end_offset - gap, [previous_arrival], [1.0])
                elif following.index in block_ids:
                    add_row(derived[previous.index].end + gap, inf, [starts[following.index]], [1.0])

    highs.run()
    if highs.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        return None
    values = highs.getSolution().col_value
    replacement = {}
    for shift in block:
        operations = tuple(
            replace(operation, arrival=int(round(values[arrivals[(shift.index, position)]])))
            for position, operation in enumerate(shift.operations)
        )
        timed = replace(shift, start=int(round(values[starts[shift.index]])), operations=operations)
        if not _driving_valid(instance, timed):
            return None
        replacement[shift.index] = timed
    return Solution(tuple(replacement.get(shift.index, shift) for shift in solution.shifts))
