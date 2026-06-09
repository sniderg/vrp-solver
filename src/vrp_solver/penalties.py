from __future__ import annotations

from dataclasses import dataclass

from .inventory import tank_events
from .model import Instance, Solution
from .rules import derive_solution, validate_solution
from .smoothness import period_buckets, smoothness_summary


@dataclass(frozen=True)
class PenaltyWeights:
    safety_kg_min: float = 1.0
    driver_window_min: float = 100.0
    driver_rest_min: float = 100.0
    max_driving_min: float = 100.0
    timing_min: float = 100.0
    customer_window_min: float = 100.0
    smoothness_cv: float = 10_000.0
    frontload_share: float = 10_000.0
    hard_violation: float = 1_000_000_000.0


@dataclass(frozen=True)
class PenaltyBreakdown:
    hard_violations: int
    hard_penalty: float
    safety_kg_min: float
    safety_penalty: float
    driver_window_min: float
    driver_window_penalty: float
    driver_rest_min: float
    driver_rest_penalty: float
    max_driving_min: float
    max_driving_penalty: float
    timing_min: float
    timing_penalty: float
    customer_window_min: float
    customer_window_penalty: float
    smoothness_penalty: float
    total_soft_penalty: float
    total_penalty: float

    def flat(self) -> dict[str, float | int]:
        return self.__dict__.copy()


def penalty_breakdown(
    instance: Instance,
    solution: Solution,
    *,
    weights: PenaltyWeights | None = None,
) -> PenaltyBreakdown:
    weights = weights or PenaltyWeights()
    rules = validate_solution(instance, solution)
    derived = derive_solution(instance, solution)

    hard_count = sum(
        1
        for violation in rules
        if violation.code in {"DYN01", "SHI06", "REF_DRIVER", "REF_TRAILER"}
        and "TANK_SAFETY_BREACH" not in violation.message
    )
    safety_kg_min = _safety_kg_minutes(instance, solution)
    driver_window_min = _driver_window_minutes(instance, derived)
    driver_rest_min = _driver_rest_minutes(instance, derived)
    max_driving_min = _max_driving_excess_minutes(instance, derived)
    timing_min = _timing_excess_minutes(instance, derived)
    customer_window_min = _customer_window_excess_minutes(instance, derived)
    smoothness_penalty = _smoothness_penalty(instance, solution, weights)

    hard_penalty = hard_count * weights.hard_violation
    safety_penalty = safety_kg_min * weights.safety_kg_min
    driver_window_penalty = driver_window_min * weights.driver_window_min
    driver_rest_penalty = driver_rest_min * weights.driver_rest_min
    max_driving_penalty = max_driving_min * weights.max_driving_min
    timing_penalty = timing_min * weights.timing_min
    customer_window_penalty = customer_window_min * weights.customer_window_min
    total_soft = (
        safety_penalty
        + driver_window_penalty
        + driver_rest_penalty
        + max_driving_penalty
        + timing_penalty
        + customer_window_penalty
        + smoothness_penalty
    )

    return PenaltyBreakdown(
        hard_violations=hard_count,
        hard_penalty=hard_penalty,
        safety_kg_min=safety_kg_min,
        safety_penalty=safety_penalty,
        driver_window_min=driver_window_min,
        driver_window_penalty=driver_window_penalty,
        driver_rest_min=driver_rest_min,
        driver_rest_penalty=driver_rest_penalty,
        max_driving_min=max_driving_min,
        max_driving_penalty=max_driving_penalty,
        timing_min=timing_min,
        timing_penalty=timing_penalty,
        customer_window_min=customer_window_min,
        customer_window_penalty=customer_window_penalty,
        smoothness_penalty=smoothness_penalty,
        total_soft_penalty=total_soft,
        total_penalty=hard_penalty + total_soft,
    )


def _safety_kg_minutes(instance: Instance, solution: Solution) -> float:
    total = 0.0
    for event in tank_events(instance, solution):
        customer = instance.customer_by_point[event.point]
        if customer.call_in:
            continue
        deficit = max(0.0, event.safety_level - event.ending_inventory - 1e-6)
        total += deficit * instance.unit
    return total


def _driver_window_minutes(instance: Instance, derived) -> float:
    total = 0.0
    for derived_shift in derived:
        shift = derived_shift.shift
        driver = instance.drivers[shift.driver]
        total += _interval_outside_windows(shift.start, derived_shift.end, driver.time_windows)
    return total


def _driver_rest_minutes(instance: Instance, derived) -> float:
    total = 0.0
    by_driver: dict[int, list] = {}
    for derived_shift in derived:
        by_driver.setdefault(derived_shift.shift.driver, []).append(derived_shift)
    for driver_id, shifts in by_driver.items():
        driver = instance.drivers[driver_id]
        ordered = sorted(shifts, key=lambda item: (item.shift.start, item.shift.index))
        for previous, current in zip(ordered, ordered[1:]):
            required_start = previous.end + driver.min_inter_shift_duration
            total += max(0, required_start - current.shift.start)
    return float(total)


def _max_driving_excess_minutes(instance: Instance, derived) -> float:
    total = 0.0
    for derived_shift in derived:
        driver = instance.drivers[derived_shift.shift.driver]
        for operation in derived_shift.operations:
            total += max(0, operation.driving_since_layover - driver.max_driving_duration)
    return float(total)


def _timing_excess_minutes(instance: Instance, derived) -> float:
    total = 0.0
    for derived_shift in derived:
        shift = derived_shift.shift
        driver = instance.drivers[shift.driver]
        previous_departure = shift.start
        previous_point = instance.base_index
        for operation, derived_op in zip(shift.operations, derived_shift.operations):
            required = (
                previous_departure
                + instance.time_matrix[previous_point][operation.point]
                + (driver.layover_duration if derived_op.layover_before else 0)
            )
            total += max(0, required - operation.arrival)
            previous_departure = derived_op.departure
            previous_point = operation.point
    return float(total)


def _customer_window_excess_minutes(instance: Instance, derived) -> float:
    total = 0.0
    for derived_shift in derived:
        for operation, derived_op in zip(derived_shift.shift.operations, derived_shift.operations):
            customer = instance.customer_by_point.get(operation.point)
            if customer is None:
                continue
            total += _interval_outside_windows(
                derived_op.arrival,
                derived_op.departure,
                customer.time_windows,
            )
    return total


def _smoothness_penalty(
    instance: Instance,
    solution: Solution,
    weights: PenaltyWeights,
) -> float:
    summary = smoothness_summary(period_buckets(instance, solution))
    return (
        summary.delivered_cv * weights.smoothness_cv
        + summary.delivered_first_period_share * weights.frontload_share
    )


def _interval_outside_windows(start: int, end: int, windows) -> float:
    if end <= start:
        return 0.0
    covered = 0
    for window in windows:
        covered += max(0, min(end, window.end) - max(start, window.start))
    return float(max(0, end - start - covered))
