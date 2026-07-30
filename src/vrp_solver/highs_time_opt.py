from __future__ import annotations

import logging
from dataclasses import replace
import numpy as np

from .model import Instance, Solution, Shift, TimeWindow

logger = logging.getLogger(__name__)

def try_optimize_shift_times(instance: Instance, shift: Shift) -> Shift | None:
    """Return a retimed shift, or ``None`` when its timing MIP is infeasible.

    This mirrors Solver.exe's affected-shift callback: a structural mutation is
    committed only when the timing model can produce a schedule.
    """
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
    M = 30000.0
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
            highs.changeColBounds(a_idx, shift.start + travel, inf)
            # To ensure 0 layovers, the gap from shift.start to first arrival must be < layover_duration + travel
            # a_0 - shift.start <= driver.layover_duration + travel - 1
            highs.addRow(-inf, shift.start + driver.layover_duration + travel - 1, 1, np.array([a_idx], dtype=np.int32), np.array([1.0], dtype=np.float64))
        else:
            prev_idx = i - 1
            prev_setup = instance.setup_time_for_point(shift.operations[i-1].point)
            # Minimum time: arrival_i >= arrival_{i-1} + setup_{i-1} + travel
            highs.addRow(prev_setup + travel, inf, 2, np.array([a_idx, prev_idx], dtype=np.int32), np.array([1.0, -1.0], dtype=np.float64))
            # Maximum time to avoid layover: arrival_i - arrival_{i-1} - setup_{i-1} <= driver.layover_duration + travel - 1
            highs.addRow(-inf, driver.layover_duration + travel + prev_setup - 1, 2, np.array([a_idx, prev_idx], dtype=np.int32), np.array([1.0, -1.0], dtype=np.float64))
            
        last_point = op.point
        
    highs.run()
    
    status = highs.getModelStatus()
    if status == highspy.HighsModelStatus.kOptimal:
        solution_info = highs.getSolution()
        col_values = solution_info.col_value
        new_ops = []
        for i, op in enumerate(shift.operations):
            new_ops.append(replace(op, arrival=int(round(col_values[i]))))
        return replace(shift, operations=tuple(new_ops))
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
