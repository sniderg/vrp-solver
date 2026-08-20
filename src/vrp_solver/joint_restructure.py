"""Reload-augmented strict quantity repair: the first joint restructuring MIP.

Motivation (measured, 2026-08-20): V2.23's best search state (61 errors,
SHI06 x203) is infeasible for the strict quantity MILP over its fixed
topology, unmoved by 2.18M fast-search steps, and unmoved by the surgical
operator rotation.  The one untried freedom is *adding reload stops* so a
trailer's stock chain can be replenished mid-route.

This module extends the strict repair model with candidate source-visit
insertions that fit entirely inside existing route gaps: the detour
prev -> source -> next must return before the next stop's original arrival.
Under that restriction no existing arrival changes, so the customer
inventory rows of the strict model are untouched; each candidate adds one
binary (insert or not), one reload quantity coupled to it, and terms in the
trailer stock-chain rows.  At most one insertion per shift keeps candidates
independently timing-safe.

The output is replayed through local validation by the caller and, as
always, only `verify-official` confers validity.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .fast.state import FastInstance
from .highs_repair import EPSILON, _DeliveryVariable, _SourceLoadVariable
from .model import Instance, Operation, Solution


@dataclass(frozen=True)
class _ReloadCandidate:
    shift_index: int
    insert_before: int  # operation index in the shift's operation tuple
    source_point: int
    arrival: int
    max_quantity: float
    # v2 (remove-and-insert): this insertion is legal only if the delivery at
    # ``requires_removal`` (an operation index in the same shift) is removed,
    # freeing its time slot.  -1 means unconditional (fits an existing gap).
    requires_removal: int = -1
    # v2: insert after the last operation, extending the shift end within the
    # driver/trailer chain slack computed by the candidate generator.
    at_end: bool = False


@dataclass(frozen=True)
class RestructureReport:
    status: str
    candidates: int
    chosen: int
    variables: int
    constraints: int


def _shi06_trailers(instance: Instance, solution: Solution) -> set[int]:
    from .rules import validate_solution

    trailer_of = {shift.index: shift.trailer for shift in solution.shifts}
    trailers: set[int] = set()
    for violation in validate_solution(instance, solution):
        if violation.severity == "error" and violation.code == "SHI06":
            trailer = trailer_of.get(violation.shift)
            if trailer is not None:
                trailers.add(trailer)
    return trailers


def _gap_candidates(
    instance: Instance,
    fi: FastInstance,
    solution: Solution,
    target_trailers: set[int],
) -> list[_ReloadCandidate]:
    """Candidate reload insertions that leave every existing arrival intact.

    A candidate at position k requires the detour
    ``departure(k-1) -> source -> op[k]`` to arrive no later than op[k]'s
    original arrival, and the extra driving to stay under the driver's cap
    (a conservative check that ignores layover resets; the caller's replay
    catches anything it misses).
    """
    time = fi.time_matrix
    setup = fi.setup_time
    sources = [source.index for source in instance.sources]
    candidates: list[_ReloadCandidate] = []
    for shift in solution.shifts:
        if shift.trailer not in target_trailers or not shift.operations:
            continue
        trailer_capacity = instance.trailers[shift.trailer].capacity
        max_driving = fi.driver_max_driving[shift.driver]
        base_driving = 0
        prev = fi.base
        for op in shift.operations:
            base_driving += time[prev][op.point]
            prev = op.point
        base_driving += time[prev][fi.base]

        prev_point = fi.base
        prev_departure = shift.start
        for op_index, op in enumerate(shift.operations):
            for source_point in sources:
                if source_point == prev_point or source_point == op.point:
                    continue
                arrival_at_source = prev_departure + time[prev_point][source_point]
                back_at_next = (
                    arrival_at_source
                    + setup[source_point]
                    + time[source_point][op.point]
                )
                extra_driving = (
                    time[prev_point][source_point]
                    + time[source_point][op.point]
                    - time[prev_point][op.point]
                )
                if (
                    back_at_next <= op.arrival
                    and base_driving + extra_driving <= max_driving
                ):
                    candidates.append(
                        _ReloadCandidate(
                            shift_index=shift.index,
                            insert_before=op_index,
                            source_point=source_point,
                            arrival=arrival_at_source,
                            max_quantity=trailer_capacity,
                        )
                    )
            prev_point = op.point
            prev_departure = op.arrival + setup[op.point]
    return candidates


@dataclass(frozen=True)
class _StopCandidate:
    shift_index: int
    insert_before: int
    customer_point: int
    arrival: int
    arrival_step: int
    min_quantity: float
    max_quantity: float
    requires_removal: int = -1


def _allowed_end_map(fi: FastInstance, solution: Solution) -> dict[int, int]:
    """Latest legal shift end per shift: driver's next start minus DRI01
    separation, trailer's next start (TL01), the hosting driver window
    (DRI08), and the scoring cutoff."""
    next_by_driver: dict[tuple[int, int], int] = {}
    next_by_trailer: dict[tuple[int, int], int] = {}
    by_driver: dict[int, list] = {}
    by_trailer: dict[int, list] = {}
    for shift in solution.shifts:
        by_driver.setdefault(shift.driver, []).append(shift)
        by_trailer.setdefault(shift.trailer, []).append(shift)
    for driver, shifts in by_driver.items():
        ordered = sorted(shifts, key=lambda s: (s.start, s.index))
        for here, after in zip(ordered, ordered[1:]):
            next_by_driver[(driver, here.index)] = after.start
    for trailer, shifts in by_trailer.items():
        ordered = sorted(shifts, key=lambda s: (s.start, s.index))
        for here, after in zip(ordered, ordered[1:]):
            next_by_trailer[(trailer, here.index)] = after.start
    allowed: dict[int, int] = {}
    for shift in solution.shifts:
        window_end = fi.cutoff
        for w_start, w_end in fi.driver_windows[shift.driver]:
            if w_start <= shift.start < w_end:
                window_end = min(window_end, w_end)
                break
        allowed[shift.index] = min(
            next_by_driver.get((shift.driver, shift.index), fi.cutoff)
            - fi.driver_min_inter_shift[shift.driver],
            next_by_trailer.get((shift.trailer, shift.index), fi.cutoff),
            window_end,
        )
    return allowed


def _uncovered_breach_steps(
    instance: Instance, solution: Solution
) -> dict[int, int]:
    """Customers whose tank breaches even when every existing visit fills the
    tank to capacity (greedy max fill), with the first such step.  These make
    any fixed-stop-set model infeasible: only a new visit at or before that
    step can help.  Subsumes the breach-before-first-visit case."""
    visit_steps: dict[int, list[int]] = {}
    for shift in solution.shifts:
        for op in shift.operations:
            if op.point in instance.customer_by_point and op.quantity > EPSILON:
                visit_steps.setdefault(op.point, []).append(
                    op.arrival // instance.unit
                )
    breaches: dict[int, int] = {}
    for customer in instance.customers:
        if customer.call_in:
            continue
        steps = sorted(visit_steps.get(customer.index, []))
        level = customer.initial_tank_quantity
        cursor = 0
        for step in range(instance.horizon):
            if step < len(customer.forecast):
                level -= customer.forecast[step]
            while cursor < len(steps) and steps[cursor] <= step:
                level = customer.capacity
                cursor += 1
            if level < customer.safety_level - EPSILON:
                breaches[customer.index] = step
                break
    return breaches


def _stop_candidates(
    instance: Instance,
    fi: FastInstance,
    solution: Solution,
    needy: dict[int, int],
    removable: set[tuple[int, int]],
    allowed_end: dict[int, int],
) -> list[_StopCandidate]:
    """Delivery-stop insertions for customers breaching before their first
    visit: in existing gaps, in removal-freed slots (requires_removal), and
    at route ends within chain slack.  Each candidate lands at or before the
    customer's breach step, inside a customer time window, on a qualified
    trailer (SHI05), within the driving cap; no surviving arrival changes."""
    time = fi.time_matrix
    setup = fi.setup_time
    candidates: list[_StopCandidate] = []

    def _try(shift, op_index, prev_point, prev_departure, next_point,
             next_arrival, base_driving, max_driving, requires_removal):
        for point, breach_step in needy.items():
            allowed = fi.trailer_allowed[point]
            if allowed is not None and shift.trailer not in allowed:
                continue
            arrival = prev_departure + time[prev_point][point]
            if arrival // instance.unit > breach_step:
                continue
            windows = fi.customer_windows[point]
            if windows and not any(s <= arrival < e for s, e in windows):
                continue
            if next_point is not None:
                back = arrival + setup[point] + time[point][next_point]
                extra = (
                    time[prev_point][point]
                    + time[point][next_point]
                    - time[prev_point][next_point]
                )
                fits = back <= next_arrival and base_driving + extra <= max_driving
            else:
                new_end = arrival + setup[point] + time[point][fi.base]
                extra = (
                    time[prev_point][point]
                    + time[point][fi.base]
                    - time[prev_point][fi.base]
                )
                fits = (
                    new_end <= allowed_end.get(shift.index, fi.cutoff)
                    and base_driving + extra <= max_driving
                )
            if fits:
                customer = instance.customer_by_point[point]
                candidates.append(
                    _StopCandidate(
                        shift_index=shift.index,
                        insert_before=op_index,
                        customer_point=point,
                        arrival=arrival,
                        arrival_step=min(
                            max(arrival // instance.unit, 0), instance.horizon - 1
                        ),
                        min_quantity=max(customer.min_operation_quantity, 10 * EPSILON),
                        max_quantity=customer.capacity,
                        requires_removal=requires_removal,
                    )
                )

    for shift in solution.shifts:
        if not shift.operations:
            continue
        ops = shift.operations
        max_driving = fi.driver_max_driving[shift.driver]
        base_driving = 0
        prev = fi.base
        for op in ops:
            base_driving += time[prev][op.point]
            prev = op.point
        base_driving += time[prev][fi.base]

        prev_point = fi.base
        prev_departure = shift.start
        for op_index, op in enumerate(ops):
            # In an existing gap before op_index.
            _try(shift, op_index, prev_point, prev_departure,
                 op.point, op.arrival, base_driving, max_driving, -1)
            # In the slot freed by removing op_index.
            if (shift.index, op_index) in removable:
                if op_index + 1 < len(ops):
                    nxt, nxt_arr = ops[op_index + 1].point, ops[op_index + 1].arrival
                else:
                    nxt, nxt_arr = None, None
                _try(shift, op_index, prev_point, prev_departure,
                     nxt, nxt_arr, base_driving, max_driving, op_index)
            prev_point = op.point
            prev_departure = op.arrival + setup[op.point]
        # At the end of the route, within chain slack.
        last = ops[-1]
        _try(shift, len(ops), last.point, last.arrival + setup[last.point],
             None, None, base_driving, max_driving, -1)
    return candidates


@dataclass(frozen=True)
class _ShiftCandidate:
    """A whole new mini-shift: base -> [source] -> customer -> base."""

    driver: int
    trailer: int
    start: int
    source_point: int  # -1 for the no-reload variant
    source_arrival: int
    customer_point: int
    customer_arrival: int
    arrival_step: int
    min_quantity: float
    max_quantity: float
    trailer_capacity: float


def _busy_intervals(
    fi: FastInstance, solution: Solution
) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    """(driver -> [(start, end)], trailer -> [(start, end)]) sorted by start.
    End is the return-to-base minute of the shift's last operation."""
    time = fi.time_matrix
    setup = fi.setup_time
    by_driver: dict[int, list[tuple[int, int]]] = {}
    by_trailer: dict[int, list[tuple[int, int]]] = {}
    for shift in solution.shifts:
        if shift.operations:
            last = shift.operations[-1]
            end = last.arrival + setup[last.point] + time[last.point][fi.base]
        else:
            end = shift.start
        by_driver.setdefault(shift.driver, []).append((shift.start, end))
        by_trailer.setdefault(shift.trailer, []).append((shift.start, end))
    for intervals in by_driver.values():
        intervals.sort()
    for intervals in by_trailer.values():
        intervals.sort()
    return by_driver, by_trailer


def _free_intervals(
    busy: list[tuple[int, int]], horizon_end: int, margin_before: int, margin_after: int
) -> list[tuple[int, int]]:
    """Gaps between busy intervals, shrunk by the required margins."""
    free: list[tuple[int, int]] = []
    cursor = 0
    for start, end in busy:
        gap_end = start - margin_before
        if gap_end > cursor:
            free.append((cursor, gap_end))
        cursor = max(cursor, end + margin_after)
    if horizon_end > cursor:
        free.append((cursor, horizon_end))
    return free


def _shift_candidates(
    instance: Instance,
    fi: FastInstance,
    solution: Solution,
    needy: dict[int, int],
    max_per_customer: int = 40,
) -> list[_ShiftCandidate]:
    """v4: candidate new shifts serving one needy customer each, placed in a
    joint driver+trailer idle window with DRI01 separation on both sides,
    inside a driver time window (DRI08), within the driving cap (DRI03),
    qualified (TL03 driver-trailer, SHI05 trailer-customer), arriving inside
    a customer window at or before the breach step."""
    time = fi.time_matrix
    setup = fi.setup_time
    sources = [source.index for source in instance.sources]
    busy_driver, busy_trailer = _busy_intervals(fi, solution)

    candidates: list[_ShiftCandidate] = []
    for point, breach_step in needy.items():
        deadline = (breach_step + 1) * instance.unit
        customer = instance.customer_by_point[point]
        allowed = fi.trailer_allowed[point]
        found = 0
        for driver_id in range(len(fi.drivers)):
            if found >= max_per_customer:
                break
            separation = fi.driver_min_inter_shift[driver_id]
            driver_free = _free_intervals(
                busy_driver.get(driver_id, []), fi.cutoff, separation, separation
            )
            for trailer_id in fi.driver_trailers[driver_id]:
                if allowed is not None and trailer_id not in allowed:
                    continue
                trailer_free = _free_intervals(
                    busy_trailer.get(trailer_id, []), fi.cutoff, 0, 0
                )
                for source_point in [-1] + sources:
                    if source_point >= 0:
                        drive = (
                            time[fi.base][source_point]
                            + time[source_point][point]
                            + time[point][fi.base]
                        )
                        lead = time[fi.base][source_point] + setup[source_point] + time[source_point][point]
                    else:
                        drive = time[fi.base][point] + time[point][fi.base]
                        lead = time[fi.base][point]
                    if drive > fi.driver_max_driving[driver_id]:
                        continue
                    tail = setup[point] + time[point][fi.base]
                    for d_start, d_end in driver_free:
                        if d_start >= deadline:
                            break
                        for t_start, t_end in trailer_free:
                            lo = max(d_start, t_start)
                            hi = min(d_end, t_end)
                            if lo >= hi or lo >= deadline:
                                continue
                            # Also require a driver window to host the shift.
                            for w_start, w_end in fi.driver_windows[driver_id]:
                                start_lo = max(lo, w_start)
                                end_hi = min(hi, w_end)
                                if start_lo >= end_hi:
                                    continue
                                # Latest legal arrival: refilling just before
                                # the breach step maximizes the coverage a
                                # single new visit provides.
                                arrival_lo = start_lo + lead
                                arrival_hi = min(deadline - 1, end_hi - tail)
                                windows = fi.customer_windows[point]
                                if windows:
                                    fitted = None
                                    for c_start, c_end in sorted(
                                        windows, reverse=True
                                    ):
                                        candidate = min(arrival_hi, c_end - 1)
                                        if candidate >= max(arrival_lo, c_start):
                                            fitted = candidate
                                            break
                                    if fitted is None:
                                        continue
                                    arrival = fitted
                                else:
                                    if arrival_hi < arrival_lo:
                                        continue
                                    arrival = arrival_hi
                                start = arrival - lead
                                candidates.append(
                                    _ShiftCandidate(
                                        driver=driver_id,
                                        trailer=trailer_id,
                                        start=start,
                                        source_point=source_point,
                                        source_arrival=(
                                            start + time[fi.base][source_point]
                                            if source_point >= 0
                                            else -1
                                        ),
                                        customer_point=point,
                                        customer_arrival=arrival,
                                        arrival_step=min(
                                            max(arrival // instance.unit, 0),
                                            instance.horizon - 1,
                                        ),
                                        min_quantity=max(
                                            customer.min_operation_quantity,
                                            10 * EPSILON,
                                        ),
                                        max_quantity=customer.capacity,
                                        trailer_capacity=instance.trailers[
                                            trailer_id
                                        ].capacity,
                                    )
                                )
                                found += 1
                                break  # one candidate per joint interval
                        if found >= max_per_customer:
                            break
                    if found >= max_per_customer:
                        break
    return candidates


def _removable_ops(instance: Instance, solution: Solution) -> set[tuple[int, int]]:
    """Delivery operations whose removal cannot fire QS01 or LAY02.

    VMI customers only (no call-in orders), and never a layover-designated
    customer, whose presence may legalise a represented rest on the route.
    """
    removable: set[tuple[int, int]] = set()
    for shift in solution.shifts:
        for op_index, op in enumerate(shift.operations):
            customer = instance.customer_by_point.get(op.point)
            if (
                customer is not None
                and not customer.call_in
                and not customer.orders
                and not customer.layover_customer
                and op.quantity > EPSILON
            ):
                removable.add((shift.index, op_index))
    return removable


def _extended_candidates(
    instance: Instance,
    fi: FastInstance,
    solution: Solution,
    target_trailers: set[int],
    removable: set[tuple[int, int]],
) -> list[_ReloadCandidate]:
    """v2 candidates: removal-conditioned insertions and end-of-route ones.

    Every existing arrival that survives stays exactly as-is, so the
    customer inventory rows remain valid.  A removal-conditioned candidate
    occupies the removed stop's time slot (detour prev -> source -> next
    must return before *next*'s original arrival).  An end candidate extends
    the shift within the explicit slack to the driver's next shift (DRI01
    separation), the trailer's next shift (TL01), the driver window that
    hosts the shift (DRI08), and the scoring cutoff.
    """
    time = fi.time_matrix
    setup = fi.setup_time
    sources = [source.index for source in instance.sources]

    next_start_by_driver: dict[tuple[int, int], int] = {}
    next_start_by_trailer: dict[tuple[int, int], int] = {}
    by_driver: dict[int, list] = {}
    by_trailer: dict[int, list] = {}
    for shift in solution.shifts:
        by_driver.setdefault(shift.driver, []).append(shift)
        by_trailer.setdefault(shift.trailer, []).append(shift)
    for driver, shifts in by_driver.items():
        ordered = sorted(shifts, key=lambda s: (s.start, s.index))
        for here, after in zip(ordered, ordered[1:]):
            next_start_by_driver[(driver, here.index)] = after.start
    for trailer, shifts in by_trailer.items():
        ordered = sorted(shifts, key=lambda s: (s.start, s.index))
        for here, after in zip(ordered, ordered[1:]):
            next_start_by_trailer[(trailer, here.index)] = after.start

    candidates: list[_ReloadCandidate] = []
    for shift in solution.shifts:
        if shift.trailer not in target_trailers or not shift.operations:
            continue
        trailer_capacity = instance.trailers[shift.trailer].capacity
        ops = shift.operations

        window_end = fi.cutoff
        for w_start, w_end in fi.driver_windows[shift.driver]:
            if w_start <= shift.start < w_end:
                window_end = min(window_end, w_end)
                break
        allowed_end = min(
            next_start_by_driver.get(
                (shift.driver, shift.index), fi.cutoff
            )
            - fi.driver_min_inter_shift[shift.driver],
            next_start_by_trailer.get((shift.trailer, shift.index), fi.cutoff),
            window_end,
        )

        # Removal-conditioned: take over the removed stop's slot.
        for op_index, op in enumerate(ops):
            if (shift.index, op_index) not in removable:
                continue
            prev_point = fi.base if op_index == 0 else ops[op_index - 1].point
            prev_departure = (
                shift.start
                if op_index == 0
                else ops[op_index - 1].arrival + setup[prev_point]
            )
            for source_point in sources:
                arrival_at_source = prev_departure + time[prev_point][source_point]
                departure_from_source = arrival_at_source + setup[source_point]
                if op_index + 1 < len(ops):
                    next_op = ops[op_index + 1]
                    fits = (
                        departure_from_source + time[source_point][next_op.point]
                        <= next_op.arrival
                    )
                else:
                    fits = (
                        departure_from_source + time[source_point][fi.base]
                        <= allowed_end
                    )
                if fits:
                    candidates.append(
                        _ReloadCandidate(
                            shift_index=shift.index,
                            insert_before=op_index,
                            source_point=source_point,
                            arrival=arrival_at_source,
                            max_quantity=trailer_capacity,
                            requires_removal=op_index,
                        )
                    )

        # End-of-route: extend the shift within the chain slack.
        last = ops[-1]
        last_departure = last.arrival + setup[last.point]
        for source_point in sources:
            arrival_at_source = last_departure + time[last.point][source_point]
            new_end = (
                arrival_at_source
                + setup[source_point]
                + time[source_point][fi.base]
            )
            if new_end <= allowed_end:
                candidates.append(
                    _ReloadCandidate(
                        shift_index=shift.index,
                        insert_before=len(ops),
                        source_point=source_point,
                        arrival=arrival_at_source,
                        max_quantity=trailer_capacity,
                        at_end=True,
                    )
                )
    return candidates


def reload_augmented_repair(
    instance: Instance,
    solution: Solution,
    *,
    score_days: int,
    time_limit_seconds: float = 600.0,
    target_trailers: set[int] | None = None,
    allow_removals: bool = True,
    allow_stop_insertions: bool = True,
    allow_new_shifts: bool = True,
) -> tuple[Solution, RestructureReport]:
    """Strict quantity repair with reload insertions and paired removals."""
    import highspy

    from .contest import truncate_solution

    cutoff = score_days * 1440
    working = truncate_solution(solution, cutoff)
    fi = FastInstance(instance, score_days=score_days)
    if target_trailers is None:
        target_trailers = _shi06_trailers(instance, working)

    # --- variables over the existing topology (mirrors strict repair) -----
    variables: list[_DeliveryVariable] = []
    load_variables: list[_SourceLoadVariable] = []
    for shift in working.shifts:
        for op_index, op in enumerate(shift.operations):
            customer = instance.customer_by_point.get(op.point)
            if customer:
                if customer.call_in:
                    min_quantity = max_quantity = op.quantity
                else:
                    min_quantity = customer.min_operation_quantity
                    if customer.orders:
                        min_quantity = max(min_quantity, op.quantity)
                    max_quantity = customer.capacity
                variables.append(
                    _DeliveryVariable(
                        shift_index=shift.index,
                        operation_index=op_index,
                        point=op.point,
                        arrival=op.arrival,
                        arrival_step=min(
                            max(op.arrival // instance.unit, 0), instance.horizon - 1
                        ),
                        original_quantity=op.quantity,
                        min_quantity=min_quantity,
                        max_quantity=max_quantity,
                        is_fixed=False,
                    )
                )
            else:
                source = instance.source_by_point.get(op.point)
                if source and op.quantity < -EPSILON:
                    load_variables.append(
                        _SourceLoadVariable(
                            shift_index=shift.index,
                            operation_index=op_index,
                            point=op.point,
                            arrival=op.arrival,
                            original_quantity=-op.quantity,
                            max_quantity=instance.trailers[shift.trailer].capacity,
                            is_fixed=False,
                        )
                    )

    # Removals are allowed on every trailer: freeing a slot anywhere may be
    # the only way to host a needy customer's early visit, and the strict
    # inventory rows price the lost delivery globally.
    removable = _removable_ops(instance, working) if allow_removals else set()
    candidates = _gap_candidates(instance, fi, working, target_trailers)
    if allow_removals:
        candidates += _extended_candidates(
            instance, fi, working, target_trailers, removable
        )

    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    inf = highspy.kHighsInf
    no_idx = np.array([], dtype=np.int32)
    no_val = np.array([], dtype=np.float64)

    q_indices: list[int] = []
    for var in variables:
        is_removable = (var.shift_index, var.operation_index) in removable
        lower = 0.0 if is_removable else var.min_quantity
        highs.addCol(-1.0, lower, var.max_quantity, 0, no_idx, no_val)
        q_indices.append(highs.getNumCol() - 1)
    load_indices: list[int] = []
    for lvar in load_variables:
        highs.addCol(0.0, 0.0, lvar.max_quantity, 0, no_idx, no_val)
        load_indices.append(highs.getNumCol() - 1)

    # Removal binaries: d=1 forces q to 0; d=0 restores the min-quantity
    # floor.  A small cost discourages gratuitous removals (the lost
    # delivered quantity already costs -1 per unit in the objective).
    d_by_op: dict[tuple[int, int], int] = {}
    for i, var in enumerate(variables):
        key = (var.shift_index, var.operation_index)
        if key not in removable:
            continue
        highs.addCol(0.25, 0.0, 1.0, 0, no_idx, no_val)
        d_idx = highs.getNumCol() - 1
        highs.changeColIntegrality(d_idx, highspy.HighsVarType.kInteger)
        pair = np.array([q_indices[i], d_idx], dtype=np.int32)
        # q + minq * d >= minq
        highs.addRow(
            var.min_quantity,
            inf,
            2,
            pair,
            np.array([1.0, var.min_quantity], dtype=np.float64),
        )
        # q + maxq * d <= maxq
        highs.addRow(
            -inf,
            var.max_quantity,
            2,
            pair,
            np.array([1.0, var.max_quantity], dtype=np.float64),
        )
        d_by_op[key] = d_idx

    # Candidate columns: y (binary, small cost to prefer fewer insertions)
    # and r (reload amount, coupled r <= cap * y).
    y_indices: list[int] = []
    r_indices: list[int] = []
    for cand in candidates:
        highs.addCol(1.0, 0.0, 1.0, 0, no_idx, no_val)
        y_idx = highs.getNumCol() - 1
        highs.changeColIntegrality(y_idx, highspy.HighsVarType.kInteger)
        highs.addCol(0.0, 0.0, cand.max_quantity, 0, no_idx, no_val)
        r_idx = highs.getNumCol() - 1
        highs.addRow(
            -inf,
            0.0,
            2,
            np.array([r_idx, y_idx], dtype=np.int32),
            np.array([1.0, -cand.max_quantity], dtype=np.float64),
        )
        y_indices.append(y_idx)
        r_indices.append(r_idx)
        if cand.requires_removal >= 0:
            d_idx = d_by_op.get((cand.shift_index, cand.requires_removal))
            if d_idx is None:
                # Candidate's removal target is not modeled; forbid it.
                highs.changeColBounds(y_idx, 0.0, 0.0)
            else:
                # y <= d: the slot exists only if the stop is removed.
                highs.addRow(
                    -inf,
                    0.0,
                    2,
                    np.array([y_idx, d_idx], dtype=np.int32),
                    np.array([1.0, -1.0], dtype=np.float64),
                )
    # v3: delivery-stop insertions for customers breaching before their
    # first visit -- the class of infeasibility no reload/removal can fix.
    stop_candidates: list[_StopCandidate] = []
    shift_candidates: list[_ShiftCandidate] = []
    if allow_stop_insertions:
        needy = _uncovered_breach_steps(instance, working)
        if needy:
            stop_candidates = _stop_candidates(
                instance,
                fi,
                working,
                needy,
                removable,
                _allowed_end_map(fi, working),
            )
            if allow_new_shifts:
                shift_candidates = _shift_candidates(instance, fi, working, needy)
    w_indices: list[int] = []
    u_indices: list[int] = []
    for scand in stop_candidates:
        highs.addCol(1.0, 0.0, 1.0, 0, no_idx, no_val)
        w_idx = highs.getNumCol() - 1
        highs.changeColIntegrality(w_idx, highspy.HighsVarType.kInteger)
        highs.addCol(-1.0, 0.0, scand.max_quantity, 0, no_idx, no_val)
        u_idx = highs.getNumCol() - 1
        pair = np.array([u_idx, w_idx], dtype=np.int32)
        highs.addRow(
            -inf, 0.0, 2, pair,
            np.array([1.0, -scand.max_quantity], dtype=np.float64),
        )
        highs.addRow(
            0.0, inf, 2, pair,
            np.array([1.0, -scand.min_quantity], dtype=np.float64),
        )
        w_indices.append(w_idx)
        u_indices.append(u_idx)
        if scand.requires_removal >= 0:
            d_idx = d_by_op.get((scand.shift_index, scand.requires_removal))
            if d_idx is None:
                highs.changeColBounds(w_idx, 0.0, 0.0)
            else:
                highs.addRow(
                    -inf,
                    0.0,
                    2,
                    np.array([w_idx, d_idx], dtype=np.int32),
                    np.array([1.0, -1.0], dtype=np.float64),
                )

    # v4: whole new mini-shift columns.
    n_indices: list[int] = []
    nu_indices: list[int] = []
    nr_indices: list[int] = []
    for kand in shift_candidates:
        highs.addCol(2.0, 0.0, 1.0, 0, no_idx, no_val)
        n_idx = highs.getNumCol() - 1
        highs.changeColIntegrality(n_idx, highspy.HighsVarType.kInteger)
        highs.addCol(-1.0, 0.0, kand.max_quantity, 0, no_idx, no_val)
        nu_idx = highs.getNumCol() - 1
        pair = np.array([nu_idx, n_idx], dtype=np.int32)
        highs.addRow(
            -inf, 0.0, 2, pair,
            np.array([1.0, -kand.max_quantity], dtype=np.float64),
        )
        highs.addRow(
            0.0, inf, 2, pair,
            np.array([1.0, -kand.min_quantity], dtype=np.float64),
        )
        if kand.source_point >= 0:
            highs.addCol(0.0, 0.0, kand.trailer_capacity, 0, no_idx, no_val)
            nr_idx = highs.getNumCol() - 1
            highs.addRow(
                -inf, 0.0, 2,
                np.array([nr_idx, n_idx], dtype=np.int32),
                np.array([1.0, -kand.trailer_capacity], dtype=np.float64),
            )
        else:
            nr_idx = -1
        n_indices.append(n_idx)
        nu_indices.append(nu_idx)
        nr_indices.append(nr_idx)
    # A driver or trailer hosts at most one new shift (the joint windows are
    # verified per candidate, not pairwise between candidates).
    per_driver: dict[int, list[int]] = {}
    per_trailer: dict[int, list[int]] = {}
    for kand, n_idx in zip(shift_candidates, n_indices):
        per_driver.setdefault(kand.driver, []).append(n_idx)
        per_trailer.setdefault(kand.trailer, []).append(n_idx)
    for indices in list(per_driver.values()) + list(per_trailer.values()):
        if len(indices) > 1:
            highs.addRow(
                -inf, 1.0, len(indices),
                np.array(indices, dtype=np.int32),
                np.ones(len(indices), dtype=np.float64),
            )

    by_shift: dict[int, list[int]] = {}
    for cand, y_idx in zip(candidates, y_indices):
        by_shift.setdefault(cand.shift_index, []).append(y_idx)
    for scand, w_idx in zip(stop_candidates, w_indices):
        by_shift.setdefault(scand.shift_index, []).append(w_idx)
    for indices in by_shift.values():
        if len(indices) > 1:
            highs.addRow(
                -inf,
                1.0,
                len(indices),
                np.array(indices, dtype=np.int32),
                np.ones(len(indices), dtype=np.float64),
            )

    # --- strict customer inventory rows (arrivals unchanged by design;
    #     v3 stop candidates join their customer's rows from their step) ---
    by_point: dict[int, list[tuple[int, _DeliveryVariable]]] = {}
    for index, var in enumerate(variables):
        by_point.setdefault(var.point, []).append((index, var))
    stops_by_point: dict[int, list[tuple[int, _StopCandidate]]] = {}
    for index, scand in enumerate(stop_candidates):
        stops_by_point.setdefault(scand.customer_point, []).append((index, scand))
    newshifts_by_point: dict[int, list[tuple[int, _ShiftCandidate]]] = {}
    for index, kand in enumerate(shift_candidates):
        newshifts_by_point.setdefault(kand.customer_point, []).append((index, kand))
    for customer in instance.customers:
        if customer.call_in:
            continue
        cust_vars = by_point.get(customer.index, [])
        cust_stops = stops_by_point.get(customer.index, [])
        cumulative = 0.0
        for step in range(instance.horizon):
            if step < len(customer.forecast):
                cumulative += customer.forecast[step]
            indices = [q_indices[i] for i, v in cust_vars if v.arrival_step <= step]
            indices += [
                u_indices[i] for i, s in cust_stops if s.arrival_step <= step
            ]
            indices += [
                nu_indices[i]
                for i, k in newshifts_by_point.get(customer.index, [])
                if k.arrival_step <= step
            ]
            if not indices:
                if customer.initial_tank_quantity - cumulative < customer.safety_level - EPSILON:
                    highs.addRow(1.0, inf, 0, no_idx, no_val)
                continue
            arr = np.array(indices, dtype=np.int32)
            ones = np.ones(len(indices), dtype=np.float64)
            lower = customer.safety_level - customer.initial_tank_quantity + cumulative
            upper = customer.capacity - customer.initial_tank_quantity + cumulative
            highs.addRow(lower, inf, len(indices), arr, ones)
            highs.addRow(-inf, upper, len(indices), arr, ones)

    # --- trailer stock chains including candidate reloads ------------------
    q_by_op = {
        (v.shift_index, v.operation_index): q_indices[i]
        for i, v in enumerate(variables)
    }
    load_by_op = {
        (v.shift_index, v.operation_index): load_indices[i]
        for i, v in enumerate(load_variables)
    }
    cands_by_pos: dict[tuple[int, int], list[int]] = {}
    for cand, r_idx in zip(candidates, r_indices):
        cands_by_pos.setdefault((cand.shift_index, cand.insert_before), []).append(r_idx)
    # v3 stop candidates consume trailer stock at their position.
    stops_by_pos: dict[tuple[int, int], list[int]] = {}
    for scand, u_idx in zip(stop_candidates, u_indices):
        stops_by_pos.setdefault(
            (scand.shift_index, scand.insert_before), []
        ).append(u_idx)

    shifts_by_trailer: dict[int, list] = {}
    for shift in working.shifts:
        shifts_by_trailer.setdefault(shift.trailer, []).append(("real", shift))
    for index, kand in enumerate(shift_candidates):
        shifts_by_trailer.setdefault(kand.trailer, []).append(("cand", (index, kand)))
    for trailer in instance.trailers:
        constant = trailer.initial_quantity
        columns: list[int] = []
        coefficients: list[float] = []
        entries = sorted(
            shifts_by_trailer.get(trailer.index, []),
            key=lambda item: (
                item[1].start if item[0] == "real" else item[1][1].start,
                item[1].index if item[0] == "real" else 1 << 30,
            ),
        )
        for kind, payload in entries:
            if kind == "cand":
                index, kand = payload
                if kand.source_point >= 0 and nr_indices[index] >= 0:
                    columns.append(nr_indices[index])
                    coefficients.append(1.0)
                    highs.addRow(
                        -constant,
                        trailer.capacity - constant,
                        len(columns),
                        np.array(columns, dtype=np.int32),
                        np.array(coefficients, dtype=np.float64),
                    )
                columns.append(nu_indices[index])
                coefficients.append(-1.0)
                highs.addRow(
                    -constant,
                    trailer.capacity - constant,
                    len(columns),
                    np.array(columns, dtype=np.int32),
                    np.array(coefficients, dtype=np.float64),
                )
                continue
            shift = payload
            for op_index, op in enumerate(shift.operations):
                for r_idx in cands_by_pos.get((shift.index, op_index), []):
                    columns.append(r_idx)
                    coefficients.append(1.0)
                    # After a candidate reload the stock must fit the trailer.
                    highs.addRow(
                        -constant,
                        trailer.capacity - constant,
                        len(columns),
                        np.array(columns, dtype=np.int32),
                        np.array(coefficients, dtype=np.float64),
                    )
                for u_idx in stops_by_pos.get((shift.index, op_index), []):
                    columns.append(u_idx)
                    coefficients.append(-1.0)
                    # After a candidate delivery the stock must stay >= 0.
                    highs.addRow(
                        -constant,
                        trailer.capacity - constant,
                        len(columns),
                        np.array(columns, dtype=np.int32),
                        np.array(coefficients, dtype=np.float64),
                    )
                key = (shift.index, op_index)
                if key in q_by_op:
                    columns.append(q_by_op[key])
                    coefficients.append(-1.0)
                elif key in load_by_op:
                    columns.append(load_by_op[key])
                    coefficients.append(1.0)
                elif op.point in instance.source_by_point and op.quantity < -EPSILON:
                    constant += -op.quantity
                elif op.quantity > EPSILON:
                    constant -= op.quantity
                if columns:
                    highs.addRow(
                        -constant,
                        trailer.capacity - constant,
                        len(columns),
                        np.array(columns, dtype=np.int32),
                        np.array(coefficients, dtype=np.float64),
                    )
            # End-of-route candidates act after the shift's last operation.
            end_pos = (shift.index, len(shift.operations))
            for r_idx in cands_by_pos.get(end_pos, []):
                columns.append(r_idx)
                coefficients.append(1.0)
                highs.addRow(
                    -constant,
                    trailer.capacity - constant,
                    len(columns),
                    np.array(columns, dtype=np.int32),
                    np.array(coefficients, dtype=np.float64),
                )
            for u_idx in stops_by_pos.get(end_pos, []):
                columns.append(u_idx)
                coefficients.append(-1.0)
                highs.addRow(
                    -constant,
                    trailer.capacity - constant,
                    len(columns),
                    np.array(columns, dtype=np.int32),
                    np.array(coefficients, dtype=np.float64),
                )

    highs.setOptionValue("time_limit", max(0.01, float(time_limit_seconds)))
    from .milp_monitor import timed_run
    timed_run(highs, "reload_restructure")
    status = highs.modelStatusToString(highs.getModelStatus())
    has_solution = "Optimal" in status or "Feasible" in status
    if not has_solution:
        return working, RestructureReport(
            status=status,
            candidates=len(candidates),
            chosen=0,
            variables=highs.getNumCol(),
            constraints=highs.getNumRow(),
        )

    values = highs.getSolution().col_value
    chosen: dict[tuple[int, int], tuple[int, int, float]] = {}
    chosen_count = 0
    for cand, y_idx, r_idx in zip(candidates, y_indices, r_indices):
        if values[y_idx] > 0.5 and values[r_idx] > EPSILON:
            chosen[(cand.shift_index, cand.insert_before)] = (
                cand.source_point,
                cand.arrival,
                float(values[r_idx]),
            )
            chosen_count += 1
    removed = {key for key, d_idx in d_by_op.items() if values[d_idx] > 0.5}
    chosen_stops: dict[tuple[int, int], tuple[int, int, float]] = {}
    for scand, w_idx, u_idx in zip(stop_candidates, w_indices, u_indices):
        if values[w_idx] > 0.5 and values[u_idx] > EPSILON:
            chosen_stops[(scand.shift_index, scand.insert_before)] = (
                scand.customer_point,
                scand.arrival,
                float(values[u_idx]),
            )
            chosen_count += 1
    new_shifts: list = []
    next_index = max((s.index for s in working.shifts), default=0) + 1
    for kand, n_idx, nu_idx, nr_idx in zip(
        shift_candidates, n_indices, nu_indices, nr_indices
    ):
        if values[n_idx] > 0.5 and values[nu_idx] > EPSILON:
            operations = []
            if kand.source_point >= 0 and nr_idx >= 0 and values[nr_idx] > EPSILON:
                operations.append(
                    Operation(
                        point=kand.source_point,
                        arrival=kand.source_arrival,
                        quantity=-float(values[nr_idx]),
                    )
                )
            operations.append(
                Operation(
                    point=kand.customer_point,
                    arrival=kand.customer_arrival,
                    quantity=float(values[nu_idx]),
                )
            )
            from .model import Shift

            new_shifts.append(
                Shift(
                    index=next_index,
                    driver=kand.driver,
                    trailer=kand.trailer,
                    start=kand.start,
                    operations=tuple(operations),
                )
            )
            next_index += 1
            chosen_count += 1

    q_by_ref = {
        (v.shift_index, v.operation_index): max(0.0, float(values[q_indices[i]]))
        for i, v in enumerate(variables)
    }
    load_by_ref = {
        (v.shift_index, v.operation_index): max(0.0, float(values[load_indices[i]]))
        for i, v in enumerate(load_variables)
    }
    shifts = []
    for shift in working.shifts:
        operations: list[Operation] = []
        for op_index, op in enumerate(shift.operations):
            key = (shift.index, op_index)
            if key in chosen:
                source_point, arrival, reload = chosen[key]
                operations.append(
                    Operation(point=source_point, arrival=arrival, quantity=-reload)
                )
            if key in chosen_stops:
                point, arrival, quantity = chosen_stops[key]
                operations.append(
                    Operation(point=point, arrival=arrival, quantity=quantity)
                )
            if key in removed:
                continue
            if key in load_by_ref:
                operations.append(replace(op, quantity=-load_by_ref[key]))
            elif key in q_by_ref:
                quantity = q_by_ref[key]
                if quantity > EPSILON:
                    operations.append(replace(op, quantity=quantity))
            else:
                operations.append(op)
        end_key = (shift.index, len(shift.operations))
        if end_key in chosen:
            source_point, arrival, reload = chosen[end_key]
            operations.append(
                Operation(point=source_point, arrival=arrival, quantity=-reload)
            )
        if end_key in chosen_stops:
            point, arrival, quantity = chosen_stops[end_key]
            operations.append(
                Operation(point=point, arrival=arrival, quantity=quantity)
            )
        if operations:
            shifts.append(replace(shift, operations=tuple(operations)))
    shifts.extend(new_shifts)
    return Solution(shifts=tuple(shifts)), RestructureReport(
        status=status,
        candidates=len(candidates),
        chosen=chosen_count,
        variables=highs.getNumCol(),
        constraints=highs.getNumRow(),
    )
