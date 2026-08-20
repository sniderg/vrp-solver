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


def reload_augmented_repair(
    instance: Instance,
    solution: Solution,
    *,
    score_days: int,
    time_limit_seconds: float = 600.0,
    target_trailers: set[int] | None = None,
) -> tuple[Solution, RestructureReport]:
    """Strict quantity repair with optional in-gap reload insertions."""
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

    candidates = _gap_candidates(instance, fi, working, target_trailers)

    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    inf = highspy.kHighsInf
    no_idx = np.array([], dtype=np.int32)
    no_val = np.array([], dtype=np.float64)

    q_indices: list[int] = []
    for var in variables:
        highs.addCol(-1.0, var.min_quantity, var.max_quantity, 0, no_idx, no_val)
        q_indices.append(highs.getNumCol() - 1)
    load_indices: list[int] = []
    for lvar in load_variables:
        highs.addCol(0.0, 0.0, lvar.max_quantity, 0, no_idx, no_val)
        load_indices.append(highs.getNumCol() - 1)

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
    by_shift: dict[int, list[int]] = {}
    for cand, y_idx in zip(candidates, y_indices):
        by_shift.setdefault(cand.shift_index, []).append(y_idx)
    for indices in by_shift.values():
        if len(indices) > 1:
            highs.addRow(
                -inf,
                1.0,
                len(indices),
                np.array(indices, dtype=np.int32),
                np.ones(len(indices), dtype=np.float64),
            )

    # --- strict customer inventory rows (arrivals unchanged by design) ----
    by_point: dict[int, list[tuple[int, _DeliveryVariable]]] = {}
    for index, var in enumerate(variables):
        by_point.setdefault(var.point, []).append((index, var))
    for customer in instance.customers:
        if customer.call_in:
            continue
        cust_vars = by_point.get(customer.index, [])
        cumulative = 0.0
        for step in range(instance.horizon):
            if step < len(customer.forecast):
                cumulative += customer.forecast[step]
            indices = [q_indices[i] for i, v in cust_vars if v.arrival_step <= step]
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

    shifts_by_trailer: dict[int, list] = {}
    for shift in working.shifts:
        shifts_by_trailer.setdefault(shift.trailer, []).append(shift)
    for trailer in instance.trailers:
        constant = trailer.initial_quantity
        columns: list[int] = []
        coefficients: list[float] = []
        for shift in sorted(
            shifts_by_trailer.get(trailer.index, []),
            key=lambda item: (item.start, item.index),
        ):
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
    chosen: dict[tuple[int, int], tuple[int, float]] = {}
    chosen_count = 0
    for cand, y_idx, r_idx in zip(candidates, y_indices, r_indices):
        if values[y_idx] > 0.5 and values[r_idx] > EPSILON:
            chosen[(cand.shift_index, cand.insert_before)] = (
                cand.source_point,
                float(values[r_idx]),
            )
            chosen_count += 1
    cand_arrival = {
        (cand.shift_index, cand.insert_before): cand.arrival for cand in candidates
    }

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
                source_point, reload = chosen[key]
                operations.append(
                    Operation(
                        point=source_point,
                        arrival=cand_arrival[key],
                        quantity=-reload,
                    )
                )
            if key in load_by_ref:
                operations.append(replace(op, quantity=-load_by_ref[key]))
            elif key in q_by_ref:
                quantity = q_by_ref[key]
                if quantity > EPSILON:
                    operations.append(replace(op, quantity=quantity))
            else:
                operations.append(op)
        if operations:
            shifts.append(replace(shift, operations=tuple(operations)))
    return Solution(shifts=tuple(shifts)), RestructureReport(
        status=status,
        candidates=len(candidates),
        chosen=chosen_count,
        variables=highs.getNumCol(),
        constraints=highs.getNumRow(),
    )
