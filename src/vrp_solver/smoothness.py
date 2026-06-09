from __future__ import annotations

from dataclasses import dataclass
from math import log2
from statistics import mean, pstdev

from .model import Instance, Solution


MINUTES_PER_DAY = 1440


@dataclass(frozen=True)
class PeriodBucket:
    period: int
    start_minute: int
    end_minute: int
    delivered_quantity: float
    loaded_quantity: float
    delivery_operations: int
    loading_operations: int
    shift_starts: int
    active_shifts_started: int


@dataclass(frozen=True)
class SmoothnessSummary:
    periods: int
    total_delivered: float
    total_shift_starts: int
    delivered_mean: float
    delivered_std: float
    delivered_cv: float
    delivered_gini: float
    delivered_entropy: float
    delivered_peak: float
    delivered_peak_period: int
    delivered_peak_share: float
    delivered_first_period_share: float
    delivered_first_3_period_share: float
    shift_starts_mean: float
    shift_starts_std: float
    shift_starts_cv: float
    shift_starts_peak: int
    shift_starts_peak_period: int
    shift_starts_first_period_share: float

    def flat(self) -> dict[str, float | int]:
        return self.__dict__.copy()


def period_buckets(
    instance: Instance,
    solution: Solution,
    *,
    period_minutes: int = MINUTES_PER_DAY,
) -> list[PeriodBucket]:
    horizon_minutes = instance.horizon * instance.unit
    periods = (horizon_minutes + period_minutes - 1) // period_minutes
    delivered = [0.0 for _ in range(periods)]
    loaded = [0.0 for _ in range(periods)]
    delivery_ops = [0 for _ in range(periods)]
    loading_ops = [0 for _ in range(periods)]
    shift_starts = [0 for _ in range(periods)]
    active_shift_starts = [0 for _ in range(periods)]

    for shift in solution.shifts:
        shift_period = _period_index(shift.start, period_minutes, periods)
        shift_starts[shift_period] += 1
        if shift.operations:
            active_shift_starts[shift_period] += 1

        for operation in shift.operations:
            period = _period_index(operation.arrival, period_minutes, periods)
            if operation.quantity > 0:
                delivered[period] += operation.quantity
                delivery_ops[period] += 1
            elif operation.quantity < 0:
                loaded[period] += -operation.quantity
                loading_ops[period] += 1

    return [
        PeriodBucket(
            period=period,
            start_minute=period * period_minutes,
            end_minute=min((period + 1) * period_minutes, horizon_minutes),
            delivered_quantity=delivered[period],
            loaded_quantity=loaded[period],
            delivery_operations=delivery_ops[period],
            loading_operations=loading_ops[period],
            shift_starts=shift_starts[period],
            active_shifts_started=active_shift_starts[period],
        )
        for period in range(periods)
    ]


def smoothness_summary(buckets: list[PeriodBucket]) -> SmoothnessSummary:
    delivered = [bucket.delivered_quantity for bucket in buckets]
    starts = [bucket.shift_starts for bucket in buckets]
    total_delivered = sum(delivered)
    total_starts = sum(starts)
    delivered_peak = max(delivered) if delivered else 0.0
    delivered_peak_period = delivered.index(delivered_peak) if delivered else -1
    starts_peak = max(starts) if starts else 0
    starts_peak_period = starts.index(starts_peak) if starts else -1

    delivered_mean = mean(delivered) if delivered else 0.0
    starts_mean = mean(starts) if starts else 0.0
    delivered_std = pstdev(delivered) if len(delivered) > 1 else 0.0
    starts_std = pstdev(starts) if len(starts) > 1 else 0.0

    return SmoothnessSummary(
        periods=len(buckets),
        total_delivered=total_delivered,
        total_shift_starts=total_starts,
        delivered_mean=delivered_mean,
        delivered_std=delivered_std,
        delivered_cv=_safe_div(delivered_std, delivered_mean),
        delivered_gini=_gini(delivered),
        delivered_entropy=_normalized_entropy(delivered),
        delivered_peak=delivered_peak,
        delivered_peak_period=delivered_peak_period,
        delivered_peak_share=_safe_div(delivered_peak, total_delivered),
        delivered_first_period_share=_safe_div(delivered[0] if delivered else 0.0, total_delivered),
        delivered_first_3_period_share=_safe_div(sum(delivered[:3]), total_delivered),
        shift_starts_mean=starts_mean,
        shift_starts_std=starts_std,
        shift_starts_cv=_safe_div(starts_std, starts_mean),
        shift_starts_peak=starts_peak,
        shift_starts_peak_period=starts_peak_period,
        shift_starts_first_period_share=_safe_div(starts[0] if starts else 0, total_starts),
    )


def _period_index(time: int, period_minutes: int, periods: int) -> int:
    return min(max(time // period_minutes, 0), periods - 1)


def _safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if abs(denominator) < 1e-12 else numerator / denominator


def _gini(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(max(0.0, value) for value in values)
    total = sum(sorted_values)
    if total <= 0:
        return 0.0
    n = len(sorted_values)
    weighted_sum = sum((index + 1) * value for index, value in enumerate(sorted_values))
    return (2 * weighted_sum) / (n * total) - (n + 1) / n


def _normalized_entropy(values: list[float]) -> float:
    total = sum(value for value in values if value > 0)
    positives = [value for value in values if value > 0]
    if total <= 0 or len(values) <= 1:
        return 0.0
    entropy = -sum((value / total) * log2(value / total) for value in positives)
    return entropy / log2(len(values))
