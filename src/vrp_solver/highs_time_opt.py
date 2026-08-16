from __future__ import annotations

import logging
from dataclasses import replace
import numpy as np

from .model import Instance, Solution, Shift, TimeWindow
from .rules import derive_solution

logger = logging.getLogger(__name__)

def try_optimize_shift_times(
    instance: Instance,
    shift: Shift,
    *,
    latest_end: int | None = None,
) -> Shift | None:
    """Return a retimed shift, or ``None`` when its timing MIP is infeasible.

    This mirrors Solver.exe's affected-shift callback: a structural mutation is
    committed only when the timing model can produce a schedule.
    """
    candidate = _try_optimize_shift_times(
        instance, shift, latest_end=latest_end, layover_before_index=None,
    )
    if candidate is not None:
        return candidate

    # A driver may reach a layover customer early and take the represented
    # rest while waiting for that customer's service window.  This is a
    # distinct, common route pattern: the layover precedes operation zero,
    # rather than sitting between two operations.  The deterministic replay
    # already recognises this gap, so the timing model must be able to create
    # it as well.
    first_customer = instance.customer_by_point.get(shift.operations[0].point)
    if first_customer is not None and first_customer.layover_customer:
        candidate = _try_optimize_shift_times(
            instance,
            shift,
            latest_end=latest_end,
            layover_before_index=0,
        )
        if candidate is not None:
            return candidate

    # Replay represents a layover as excess time before the current operation.
    # That gap can be spent after service at the previous layover site or
    # while waiting to begin service at the current layover site. Enumerate
    # both legal interpretations instead of adding a fragile big-M choice.
    for operation_index in range(1, len(shift.operations)):
        previous_customer = instance.customer_by_point.get(
            shift.operations[operation_index - 1].point,
        )
        current_customer = instance.customer_by_point.get(
            shift.operations[operation_index].point,
        )
        if not (
            (previous_customer is not None and previous_customer.layover_customer)
            or (current_customer is not None and current_customer.layover_customer)
        ):
            continue
        candidate = _try_optimize_shift_times(
            instance,
            shift,
            latest_end=latest_end,
            layover_before_index=operation_index,
        )
        if candidate is not None:
            return candidate
    return None


def _try_optimize_shift_times(
    instance: Instance,
    shift: Shift,
    *,
    latest_end: int | None,
    layover_before_index: int | None,
) -> Shift | None:
    try:
        import highspy
    except ModuleNotFoundError as exc:
        raise RuntimeError("highspy is not installed; run `uv sync --extra milp`") from exc

    driver = instance.drivers[shift.driver]
    n = len(shift.operations)
    if n == 0:
        return shift
    operation_windows = [
        _feasible_operation_windows(instance, operation)
        for operation in shift.operations
    ]
    if any(not windows for windows in operation_windows):
        return None
        
    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    inf = highspy.kHighsInf
    
    # Decision variables:
    # 0 to n-1: a_i (arrival times)
    # n to 2n-1: d_i (absolute deviations from original arrival times)
    # y_{i, w} (binary indicator for selecting window w of customer i)
    y_cols = {}
    col_count = 2 * n
    
    for i, windows in enumerate(operation_windows):
        for w in range(len(windows)):
            y_cols[(i, w)] = col_count
            col_count += 1
            
    # Add columns:
    # a_i: obj = 0.0, bounds = [0, inf]
    for i in range(n):
        highs.addCol(0.0, 0.0, inf, 0, np.array([], dtype=np.int32), np.array([], dtype=np.float64))
        
    # d_i: obj = 1.0 (minimize sum of deviations to preserve schedule shape), bounds = [0, inf]
    for i in range(n):
        highs.addCol(1.0, 0.0, inf, 0, np.array([], dtype=np.int32), np.array([], dtype=np.float64))
        
    # y_{i, w}: obj = 0.0, bounds = [0, 1], integer
    for i, windows in enumerate(operation_windows):
        for w in range(len(windows)):
            y_idx = y_cols[(i, w)]
            highs.addCol(0.0, 0.0, 1.0, 0, np.array([], dtype=np.int32), np.array([], dtype=np.float64))
            highs.changeColIntegrality(y_idx, highspy.HighsVarType.kInteger)
            
    # Constraints:
    # 1. d_i >= a_i - original_arrival  =>  a_i - d_i <= original_arrival
    # 2. d_i >= original_arrival - a_i  =>  a_i + d_i >= original_arrival
    for i, op in enumerate(shift.operations):
        a_idx = i
        d_idx = n + i
        highs.addRow(-inf, op.arrival, 2, np.array([a_idx, d_idx], dtype=np.int32), np.array([1.0, -1.0], dtype=np.float64))
        highs.addRow(op.arrival, inf, 2, np.array([a_idx, d_idx], dtype=np.int32), np.array([1.0, 1.0], dtype=np.float64))
        
    # 3. For each operation i, exactly one window must be selected:
    # sum_{w} y_{i, w} = 1
    # Inactive periodic windows must be relaxed across the *entire* instance
    # horizon.  A fixed 30,000-minute bound silently makes early operations
    # infeasible on the 35-day instances because their late windows still
    # impose positive lower bounds when deselected.
    M = float(instance.latest_time + max(
        (instance.setup_time_for_point(op.point) for op in shift.operations),
        default=0,
    ))
    for i, op in enumerate(shift.operations):
        setup = instance.setup_time_for_point(op.point)
        windows = operation_windows[i]
            
        y_indices = [y_cols[(i, w)] for w in range(len(windows))]
        highs.addRow(1.0, 1.0, len(y_indices), np.array(y_indices, dtype=np.int32), np.array([1.0]*len(y_indices), dtype=np.float64))
        
        # 4. If window w is selected, arrival a_i must lie in [window.start, window.end - setup]
        for w, window in enumerate(windows):
            y_idx = y_cols[(i, w)]
            a_idx = i
            # a_i >= window.start - M * (1 - y_{i, w})  =>  a_i - M * y_{i, w} >= window.start - M
            highs.addRow(window.start - M, inf, 2, np.array([a_idx, y_idx], dtype=np.int32), np.array([1.0, -M], dtype=np.float64))
            # a_i + setup <= window.end + M * (1 - y_{i, w})  =>  a_i + M * y_{i, w} <= window.end + M - setup
            highs.addRow(-inf, window.end + M - setup, 2, np.array([a_idx, y_idx], dtype=np.int32), np.array([1.0, M], dtype=np.float64))
            
    # 5. Travel time and driver layover/rest constraints between operations:
    last_point = instance.base_index
    for i, op in enumerate(shift.operations):
        travel = instance.time_matrix[last_point][op.point]
        a_idx = i
        
        if i == 0:
            # a_0 >= shift.start + travel
            minimum_gap = travel + (
                driver.layover_duration if layover_before_index == 0 else 0
            )
            highs.changeColBounds(a_idx, shift.start + minimum_gap, inf)
            if layover_before_index != 0:
                # To ensure zero layovers, the gap from shift start to first
                # arrival must remain below the replay layover threshold.
                highs.addRow(-inf, shift.start + driver.layover_duration + travel - 1, 1, np.array([a_idx], dtype=np.int32), np.array([1.0], dtype=np.float64))
        else:
            prev_idx = i - 1
            prev_setup = instance.setup_time_for_point(shift.operations[i-1].point)
            rest = driver.layover_duration if i == layover_before_index else 0
            # Minimum time includes the represented rest at the previous
            # eligible customer when this is the selected layover split.
            highs.addRow(prev_setup + travel + rest, inf, 2, np.array([a_idx, prev_idx], dtype=np.int32), np.array([1.0, -1.0], dtype=np.float64))
            if i != layover_before_index:
                # Prevent an unselected idle gap from silently becoming a
                # second layover in deterministic replay.
                highs.addRow(-inf, driver.layover_duration + travel + prev_setup - 1, 2, np.array([a_idx, prev_idx], dtype=np.int32), np.array([1.0, -1.0], dtype=np.float64))
            
        last_point = op.point

    if latest_end is not None:
        last_setup = instance.setup_time_for_point(
            shift.operations[-1].point,
        )
        return_time = instance.time_matrix[
            shift.operations[-1].point
        ][instance.base_index]
        highs.addRow(
            -inf,
            latest_end - last_setup - return_time,
            1,
            np.array([n - 1], dtype=np.int32),
            np.array([1.0], dtype=np.float64),
        )
        
    highs.run()
    
    status = highs.getModelStatus()
    if status == highspy.HighsModelStatus.kOptimal:
        solution_info = highs.getSolution()
        col_values = solution_info.col_value
        new_ops = []
        for i, op in enumerate(shift.operations):
            new_ops.append(replace(op, arrival=int(round(col_values[i]))))
        candidate = replace(shift, operations=tuple(new_ops))
        # The timing MIP preserves operation-to-operation feasibility, but a
        # changed arrival pattern can still make the final return exceed the
        # continuous-driving limit.  Do not let a post-selection retime undo
        # route validity established by the constructor/column gate.
        if not _driving_valid(instance, candidate):
            return None
        return candidate
    else:
        logger.warning("Shift %d time optimization failed with status %s", shift.index, status)
        return None


def optimize_shift_times(instance: Instance, shift: Shift) -> Shift:
    """Best-effort compatibility wrapper used by non-transactional callers."""
    optimized = try_optimize_shift_times(instance, shift)
    return shift if optimized is None else optimized

def optimize_solution_times(instance: Instance, solution: Solution) -> Solution:
    new_shifts = []
    for shift in solution.shifts:
        new_shifts.append(optimize_shift_times(instance, shift))
    return Solution(shifts=tuple(new_shifts))


def latest_end_before_successors(
    instance: Instance,
    solution: Solution,
    shift_index: int,
) -> int:
    """Return the hard end boundary imposed by unchanged resource successors."""
    shift = next(item for item in solution.shifts if item.index == shift_index)
    bound = instance.latest_time
    driver = instance.drivers[shift.driver]
    for successor in solution.shifts:
        if successor.index == shift.index or successor.start <= shift.start:
            continue
        if successor.driver == shift.driver:
            bound = min(bound, successor.start - driver.min_inter_shift_duration)
        if successor.trailer == shift.trailer:
            bound = min(bound, successor.start)
    return bound


def _return_driving_valid(instance: Instance, shift: Shift) -> bool:
    derived = derive_solution(instance, Solution(shifts=(shift,)))[0]
    if not derived.operations:
        return True
    driver = instance.drivers[shift.driver]
    final = derived.operations[-1]
    return (
        final.driving_since_layover
        + instance.time_matrix[final.point][instance.base_index]
        <= driver.max_driving_duration
    )


def _driving_valid(instance: Instance, shift: Shift) -> bool:
    # This function sits in the innermost route-generation loop.  Calling the
    # full validator here also replays every customer tank over the horizon,
    # even though retiming cannot alter quantities.  Check exactly the DRI03
    # conditions against the already-derived shift instead.
    derived = derive_solution(instance, Solution(shifts=(shift,)))[0]
    driver = instance.drivers[shift.driver]
    previous_driving = 0
    for operation in derived.operations:
        driving = (
            previous_driving + operation.driving_before_layover
            if operation.layover_before
            else operation.driving_since_layover
        )
        if driving > driver.max_driving_duration + 1e-6:
            return False
        previous_driving = operation.driving_since_layover
    if not derived.operations:
        return True
    final = derived.operations[-1]
    return (
        final.driving_since_layover
        + instance.time_matrix[final.point][instance.base_index]
        <= driver.max_driving_duration + 1e-6
    )


def _feasible_operation_windows(
    instance: Instance,
    operation,
) -> tuple[TimeWindow, ...]:
    customer = instance.customer_by_point.get(operation.point)
    if customer is None:
        return (
            TimeWindow(
                start=0,
                end=instance.horizon * instance.unit,
            ),
        )
    if not customer.call_in or not customer.orders:
        return tuple(customer.time_windows)
    # The service-quality checker assigns a call-in delivery to an order when
    # its arrival lies in that order's interval. Intersect those intervals
    # with the physical customer windows so retiming cannot silently turn a
    # satisfied order back into QS01.
    setup = instance.setup_time_for_point(operation.point)
    intersections = {
        (
            max(window.start, order.earliest_time),
            min(window.end, order.latest_time + setup),
        )
        for window in customer.time_windows
        for order in customer.orders
    }
    return tuple(
        TimeWindow(start=start, end=end)
        for start, end in sorted(intersections)
        if start + setup <= end
    )
