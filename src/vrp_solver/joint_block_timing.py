"""Bounded joint timing repair for a connected driver/trailer block.

This deliberately solves *timing only*.  Route order, resource assignment,
and quantities remain fixed, which keeps the model small enough to place in a
native neighbourhood loop.  Unlike ``try_optimize_shift_times``, resource
successors inside the block are decision variables rather than fixed end
boundaries.
"""
from __future__ import annotations

from dataclasses import replace
from itertools import combinations

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
    compact: bool = False,
) -> Solution | None:
    """Jointly retime a small resource-connected block around ``anchor``.

    The original order of shifts on every driver and trailer is retained.  The
    block includes nearest predecessor/successor shifts sharing either anchor
    resource, then expands through their resources until the bounded size is
    reached.  Unchanged neighbouring shifts become hard boundary constraints.
    """
    return retime_resource_blocks(
        instance,
        solution,
        (anchor_shift_index,),
        max_shifts=max_shifts,
        compact=compact,
    )


def retime_resource_blocks(
    instance: Instance,
    solution: Solution,
    anchor_shift_indices: tuple[int, ...],
    *,
    max_shifts: int = 8,
    compact: bool = False,
    allow_resource_reorder: bool = False,
) -> Solution | None:
    """Jointly retime the union of several connected resource components.

    ``compact`` finds the earliest legal schedule rather than preserving the
    incumbent timestamps.  It is used only for an atomic create-and-place move:
    shortening committed shifts can expose a jointly available driver/trailer
    interval that aggregate resource utilisation hides.
    """
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
    if allow_resource_reorder and not _closed_resource_component(
        solution, selected_shifts,
    ):
        return None
    return _solve_joint_timing(
        instance,
        solution,
        selected_shifts,
        derived,
        compact=compact,
        allow_resource_reorder=allow_resource_reorder,
    )


def schedule_route_templates(
    instance: Instance,
    solution: Solution,
    *,
    preserve_inventory_steps: bool = True,
    reassign_trailers: bool = True,
    preserve_trailer_order: bool = False,
    preserve_driver_order: bool = False,
    time_limit: float = 300.0,
) -> Solution | None:
    """Jointly assign resources, sequence routes, and retime a complete plan."""
    try:
        import highspy
    except ModuleNotFoundError as exc:
        raise RuntimeError("highspy is not installed; run `uv sync --extra milp`") from exc

    shifts = list(solution.shifts)
    if not shifts:
        return solution
    replay = {
        item.shift.index: item for item in derive_solution(instance, solution)
    }
    operation_windows = {
        shift.index: [
            _feasible_operation_windows(instance, operation)
            for operation in shift.operations
        ]
        for shift in shifts
    }
    if any(
        not windows
        for per_shift in operation_windows.values()
        for windows in per_shift
    ):
        return None

    resource_options: dict[int, list[tuple[int, int]]] = {}
    for shift in shifts:
        has_layover = bool(replay[shift.index].layovers)
        carried = 0.0
        required_capacity = 0.0
        for operation in shift.operations:
            carried -= operation.quantity
            required_capacity = max(required_capacity, carried)
        options: list[tuple[int, int]] = []
        for trailer in instance.trailers:
            if not reassign_trailers and trailer.index != shift.trailer:
                continue
            if required_capacity > trailer.capacity + 1e-6:
                continue
            if not all(
                (
                    operation.point not in instance.source_by_point
                    or trailer.index
                    in instance.source_by_point[operation.point].allowed_trailers
                )
                and (
                    operation.point not in instance.customer_by_point
                    or trailer.index
                    in instance.customer_by_point[operation.point].allowed_trailers
                )
                for operation in shift.operations
            ):
                continue
            for driver in instance.drivers:
                if trailer.index not in driver.trailer_ids:
                    continue
                if (
                    has_layover
                    and driver.layover_duration
                    != instance.drivers[shift.driver].layover_duration
                ):
                    continue
                if not _driving_valid(
                    instance,
                    replace(shift, driver=driver.index, trailer=trailer.index),
                ):
                    continue
                options.append((driver.index, trailer.index))
        if not options:
            return None
        resource_options[shift.index] = options

    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.setOptionValue("time_limit", time_limit)
    inf = highspy.kHighsInf
    big_m = float(max(30_000, instance.latest_time + 30_000))

    def add_column(
        cost: float = 0.0,
        lower: float = 0.0,
        upper: float = inf,
        *,
        integer: bool = False,
    ) -> int:
        column = highs.getNumCol()
        highs.addCol(
            cost, lower, upper, 0,
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float64),
        )
        if integer:
            highs.changeColIntegrality(column, highspy.HighsVarType.kInteger)
        return column

    def add_row(lower, upper, columns, values) -> None:
        highs.addRow(
            lower,
            upper,
            len(columns),
            np.array(columns, dtype=np.int32),
            np.array(values, dtype=np.float64),
        )

    starts: dict[int, int] = {}
    start_deviations: dict[int, int] = {}
    arrivals: dict[tuple[int, int], int] = {}
    arrival_deviations: dict[tuple[int, int], int] = {}
    assignment_columns: dict[tuple[int, int, int], int] = {}
    operation_window_columns: dict[tuple[int, int, int], int] = {}
    driver_window_columns: dict[tuple[int, int, int], int] = {}

    for shift in shifts:
        starts[shift.index] = add_column(0.0001)
        start_deviations[shift.index] = add_column(0.01)
        for position, _operation in enumerate(shift.operations):
            arrivals[(shift.index, position)] = add_column()
            arrival_deviations[(shift.index, position)] = add_column(0.001)
            for choice in range(
                len(operation_windows[shift.index][position])
            ):
                operation_window_columns[
                    (shift.index, position, choice)
                ] = add_column(upper=1.0, integer=True)
        for driver, trailer in resource_options[shift.index]:
            assignment_columns[
                (shift.index, driver, trailer)
            ] = add_column(upper=1.0, integer=True)
        for driver in instance.drivers:
            if not any(
                option_driver == driver.index
                for option_driver, _trailer in resource_options[shift.index]
            ):
                continue
            for choice in range(len(driver.time_windows)):
                driver_window_columns[
                    (shift.index, driver.index, choice)
                ] = add_column(upper=1.0, integer=True)

    for shift in shifts:
        shift_id = shift.index
        start = starts[shift_id]
        start_deviation = start_deviations[shift_id]
        add_row(
            -inf, shift.start,
            [start, start_deviation], [1.0, -1.0],
        )
        add_row(
            shift.start, inf,
            [start, start_deviation], [1.0, 1.0],
        )
        assignments = [
            assignment_columns[(shift_id, driver, trailer)]
            for driver, trailer in resource_options[shift_id]
        ]
        add_row(1.0, 1.0, assignments, [1.0] * len(assignments))

        last_arrival = arrivals[(shift_id, len(shift.operations) - 1)]
        last_operation = shift.operations[-1]
        end_offset = (
            instance.setup_time_for_point(last_operation.point)
            + instance.time_matrix[last_operation.point][instance.base_index]
        )
        for driver in instance.drivers:
            driver_assignments = [
                assignment_columns[(shift_id, option_driver, trailer)]
                for option_driver, trailer in resource_options[shift_id]
                if option_driver == driver.index
            ]
            if not driver_assignments:
                continue
            windows = [
                driver_window_columns[
                    (shift_id, driver.index, choice)
                ]
                for choice in range(len(driver.time_windows))
            ]
            add_row(
                0.0, 0.0,
                [*windows, *driver_assignments],
                [*([1.0] * len(windows)), *([-1.0] * len(driver_assignments))],
            )
            for choice, window in enumerate(driver.time_windows):
                enabled = driver_window_columns[
                    (shift_id, driver.index, choice)
                ]
                add_row(
                    window.start - big_m,
                    inf,
                    [start, enabled],
                    [1.0, -big_m],
                )
                add_row(
                    -inf,
                    window.end + big_m - end_offset,
                    [last_arrival, enabled],
                    [1.0, big_m],
                )

        layover_before = {
            position
            for position, operation in enumerate(replay[shift_id].operations)
            if operation.layover_before
        }
        previous_point = instance.base_index
        for position, operation in enumerate(shift.operations):
            arrival = arrivals[(shift_id, position)]
            deviation = arrival_deviations[(shift_id, position)]
            add_row(
                -inf, operation.arrival,
                [arrival, deviation], [1.0, -1.0],
            )
            add_row(
                operation.arrival, inf,
                [arrival, deviation], [1.0, 1.0],
            )
            windows = operation_windows[shift_id][position]
            selectors = [
                operation_window_columns[(shift_id, position, choice)]
                for choice in range(len(windows))
            ]
            add_row(1.0, 1.0, selectors, [1.0] * len(selectors))
            setup = instance.setup_time_for_point(operation.point)
            for choice, window in enumerate(windows):
                enabled = operation_window_columns[
                    (shift_id, position, choice)
                ]
                add_row(
                    window.start - big_m,
                    inf,
                    [arrival, enabled],
                    [1.0, -big_m],
                )
                add_row(
                    -inf,
                    window.end + big_m - setup,
                    [arrival, enabled],
                    [1.0, big_m],
                )
            customer = instance.customer_by_point.get(operation.point)
            if (
                preserve_inventory_steps
                and customer is not None
                and not customer.call_in
                and operation.quantity > 0.0
            ):
                step = min(
                    max(operation.arrival // instance.unit, 0),
                    instance.horizon - 1,
                )
                add_row(
                    step * instance.unit,
                    (step + 1) * instance.unit - 1,
                    [arrival],
                    [1.0],
                )
            travel = instance.time_matrix[previous_point][operation.point]
            if position == 0:
                add_row(
                    travel,
                    inf,
                    [arrival, start],
                    [1.0, -1.0],
                )
                max_layover = instance.drivers[shift.driver].layover_duration
                add_row(
                    -inf,
                    travel + max_layover - 1,
                    [arrival, start],
                    [1.0, -1.0],
                )
            else:
                previous = arrivals[(shift_id, position - 1)]
                previous_setup = instance.setup_time_for_point(
                    shift.operations[position - 1].point
                )
                rest = (
                    instance.drivers[shift.driver].layover_duration
                    if position in layover_before
                    else 0
                )
                add_row(
                    previous_setup + travel + rest,
                    inf,
                    [arrival, previous],
                    [1.0, -1.0],
                )
                if position not in layover_before:
                    add_row(
                        -inf,
                        previous_setup
                        + travel
                        + instance.drivers[shift.driver].layover_duration
                        - 1,
                        [arrival, previous],
                        [1.0, -1.0],
                    )
            previous_point = operation.point

    for left, right in combinations(shifts, 2):
        left_last = arrivals[(left.index, len(left.operations) - 1)]
        right_last = arrivals[(right.index, len(right.operations) - 1)]
        left_end_offset = (
            instance.setup_time_for_point(left.operations[-1].point)
            + instance.time_matrix[left.operations[-1].point][instance.base_index]
        )
        right_end_offset = (
            instance.setup_time_for_point(right.operations[-1].point)
            + instance.time_matrix[right.operations[-1].point][instance.base_index]
        )
        for driver in instance.drivers:
            left_assignments = [
                assignment_columns[(left.index, option_driver, trailer)]
                for option_driver, trailer in resource_options[left.index]
                if option_driver == driver.index
            ]
            right_assignments = [
                assignment_columns[(right.index, option_driver, trailer)]
                for option_driver, trailer in resource_options[right.index]
                if option_driver == driver.index
            ]
            if not left_assignments or not right_assignments:
                continue
            if preserve_driver_order:
                if (left.start, left.index) <= (right.start, right.index):
                    early, late = left, right
                    early_last = left_last
                    early_end_offset = left_end_offset
                else:
                    early, late = right, left
                    early_last = right_last
                    early_end_offset = right_end_offset
                add_row(
                    early_end_offset + driver.min_inter_shift_duration
                    - 2 * big_m,
                    inf,
                    [
                        starts[late.index],
                        early_last,
                        *left_assignments,
                        *right_assignments,
                    ],
                    [
                        1.0,
                        -1.0,
                        *(
                            [-big_m]
                            * (len(left_assignments) + len(right_assignments))
                        ),
                    ],
                )
                continue
            order = add_column(upper=1.0, integer=True)
            gap = driver.min_inter_shift_duration
            add_row(
                left_end_offset + gap - 3 * big_m,
                inf,
                [
                    starts[right.index], left_last, order,
                    *left_assignments, *right_assignments,
                ],
                [
                    1.0, -1.0, -big_m,
                    *([-big_m] * (len(left_assignments) + len(right_assignments))),
                ],
            )
            add_row(
                right_end_offset + gap - 2 * big_m,
                inf,
                [
                    starts[left.index], right_last, order,
                    *left_assignments, *right_assignments,
                ],
                [
                    1.0, -1.0, big_m,
                    *([-big_m] * (len(left_assignments) + len(right_assignments))),
                ],
            )
        for trailer in instance.trailers:
            if (
                preserve_trailer_order
                and not reassign_trailers
                and left.trailer == trailer.index
                and right.trailer == trailer.index
            ):
                if (left.start, left.index) <= (right.start, right.index):
                    early, late = left, right
                    early_last = left_last
                    early_end_offset = left_end_offset
                else:
                    early, late = right, left
                    early_last = right_last
                    early_end_offset = right_end_offset
                add_row(
                    early_end_offset,
                    inf,
                    [starts[late.index], early_last],
                    [1.0, -1.0],
                )
                continue
            left_assignments = [
                assignment_columns[(left.index, driver, option_trailer)]
                for driver, option_trailer in resource_options[left.index]
                if option_trailer == trailer.index
            ]
            right_assignments = [
                assignment_columns[(right.index, driver, option_trailer)]
                for driver, option_trailer in resource_options[right.index]
                if option_trailer == trailer.index
            ]
            if not left_assignments or not right_assignments:
                continue
            order = add_column(upper=1.0, integer=True)
            add_row(
                left_end_offset - 3 * big_m,
                inf,
                [
                    starts[right.index], left_last, order,
                    *left_assignments, *right_assignments,
                ],
                [
                    1.0, -1.0, -big_m,
                    *([-big_m] * (len(left_assignments) + len(right_assignments))),
                ],
            )
            add_row(
                right_end_offset - 2 * big_m,
                inf,
                [
                    starts[left.index], right_last, order,
                    *left_assignments, *right_assignments,
                ],
                [
                    1.0, -1.0, big_m,
                    *([-big_m] * (len(left_assignments) + len(right_assignments))),
                ],
            )

    from .milp_monitor import timed_run
    timed_run(highs, "block_timing")
    status = highs.modelStatusToString(highs.getModelStatus())
    has_solution = (
        highs.getInfo().primal_solution_status == 2
        or "Optimal" in status
        or "Feasible" in status
    )
    if not has_solution:
        return None
    values = highs.getSolution().col_value
    scheduled: list[Shift] = []
    for shift in shifts:
        driver, trailer = max(
            resource_options[shift.index],
            key=lambda option: values[
                assignment_columns[(shift.index, option[0], option[1])]
            ],
        )
        operations = tuple(
            replace(
                operation,
                arrival=int(round(values[arrivals[(shift.index, position)]])),
            )
            for position, operation in enumerate(shift.operations)
        )
        scheduled.append(
            replace(
                shift,
                driver=driver,
                trailer=trailer,
                start=int(round(values[starts[shift.index]])),
                operations=operations,
            )
        )
    result = Solution(tuple(scheduled))
    if any(not _driving_valid(instance, shift) for shift in result.shifts):
        return None
    return result


def _closed_resource_component(
    solution: Solution,
    block: list[Shift],
) -> bool:
    """Ensure ordered timing cannot cross an unmodelled resource boundary."""
    block_ids = {shift.index for shift in block}
    for resource in ("driver", "trailer"):
        resource_ids = {getattr(shift, resource) for shift in block}
        for resource_id in resource_ids:
            if any(
                shift.index not in block_ids
                for shift in solution.shifts
                if getattr(shift, resource) == resource_id
            ):
                return False
    return True


def generate_pressure_block_insertions(
    instance: Instance,
    solution: Solution,
    *,
    customer_point: int,
    first_minute: int,
    radius: int = 4_320,
    max_candidates: int = 24,
    max_block_shifts: int = 5,
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
        # Try internal gaps first: this is where waiting time can absorb a
        # delivery even when the route tail is saturated.
        positions = list(range(1, len(recipient.operations))) + [0, len(recipient.operations)]
        for position in positions:
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
            operation.point in instance.customer_by_point
            and operation.point != customer_point
            and operation.arrival <= first_minute
            and first_minute - operation.arrival <= radius
            and not instance.customer_by_point[operation.point].call_in
            and not instance.customer_by_point[operation.point].layover_customer
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
        displaced = donor.operations[donor_pos]
        displaced_customer = instance.customer_by_point[displaced.point]
        donor_ops = list(donor.operations)
        donor_ops[donor_pos] = Operation(
            customer_point, first_minute,
            max(target.min_operation_quantity, 10.0e-6),
        )
        for recipient in solution.shifts:
            if recipient.index == donor.index or recipient.trailer not in displaced_customer.allowed_trailers:
                continue
            if abs(recipient.start - displaced.arrival) > radius:
                continue
            for position in range(len(recipient.operations) + 1):
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


def _solve_joint_timing(
    instance,
    solution,
    block,
    derived,
    *,
    compact: bool,
    allow_resource_reorder: bool,
) -> Solution | None:
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
        starts[shift.index] = add_column(0.0001 if compact else 0.02)
        start_deviations[shift.index] = add_column(0.0 if compact else 0.02)
        for position, operation in enumerate(shift.operations):
            arrivals[(shift.index, position)] = add_column(
                0.001 if compact else 0.0,
            )
            add_column(0.0 if compact else 1.0)
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

    if allow_resource_reorder:
        for left, right in combinations(block, 2):
            shared_driver = left.driver == right.driver
            shared_trailer = left.trailer == right.trailer
            if not (shared_driver or shared_trailer):
                continue
            gap = (
                instance.drivers[left.driver].min_inter_shift_duration
                if shared_driver else 0
            )
            order = add_column(0.0, 0.0, 1.0)
            highs.changeColIntegrality(order, highspy.HighsVarType.kInteger)
            left_arrival = arrivals[(left.index, len(left.operations) - 1)]
            right_arrival = arrivals[(right.index, len(right.operations) - 1)]
            left_end = (
                instance.setup_time_for_point(left.operations[-1].point)
                + instance.time_matrix[left.operations[-1].point][instance.base_index]
            )
            right_end = (
                instance.setup_time_for_point(right.operations[-1].point)
                + instance.time_matrix[right.operations[-1].point][instance.base_index]
            )
            # order=1 means left precedes right; otherwise right precedes left.
            add_row(
                left_end + gap - big_m,
                inf,
                [starts[right.index], left_arrival, order],
                [1.0, -1.0, -big_m],
            )
            add_row(
                right_end + gap,
                inf,
                [starts[left.index], right_arrival, order],
                [1.0, -1.0, big_m],
            )
    else:
        # Preserve original order on every shared resource. Outside-block shifts
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

    from .milp_monitor import timed_run
    timed_run(highs, "block_timing_2")
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
