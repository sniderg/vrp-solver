from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import mean, stdev

from .model import Instance, Solution
from .smoothness import MINUTES_PER_DAY, period_buckets


@dataclass(frozen=True)
class FleetProfile:
    drivers: int
    trailers: int
    total_trailer_capacity: float
    mean_trailer_capacity: float
    max_trailer_capacity: float
    initial_trailer_inventory: float


@dataclass(frozen=True)
class RollingDay:
    day: int
    start_minute: int
    end_minute: int
    vmi_consumption: float
    call_in_due_quantity: float
    cumulative_vmi_consumption: float
    cumulative_required_delivery: float
    cumulative_required_delivery_share: float
    smooth_required_daily_average: float
    smooth_planned_daily_average: float
    planned_delivered: float
    planned_shift_starts: int
    cumulative_planned_delivered: float
    cumulative_delivery_gap_to_required: float
    cumulative_delivery_gap_to_smooth_target: float
    cumulative_delivery_gap_to_planned_average: float
    daily_delivery_deviation_from_smooth_target: float
    daily_delivery_t_statistic: float | None


@dataclass(frozen=True)
class RollingSummary:
    days: int
    total_vmi_consumption: float
    total_call_in_due_quantity: float
    total_required_delivery: float
    smooth_required_daily_average: float
    smooth_planned_daily_average: float
    fleet: FleetProfile
    monitored_days: int
    monitored_planned_delivered: float
    monitored_required_delivery: float
    monitored_gap_to_required: float
    monitored_gap_to_smooth_target: float
    monitored_gap_to_planned_average: float
    monitored_delivery_cv: float
    monitored_delivery_t_statistic: float | None

    def flat(self) -> dict[str, object]:
        row = self.__dict__.copy()
        fleet = row.pop("fleet")
        for key, value in fleet.__dict__.items():
            row[f"fleet_{key}"] = value
        return row


def fleet_profile(instance: Instance) -> FleetProfile:
    capacities = [trailer.capacity for trailer in instance.trailers]
    return FleetProfile(
        drivers=len(instance.drivers),
        trailers=len(instance.trailers),
        total_trailer_capacity=sum(capacities),
        mean_trailer_capacity=mean(capacities) if capacities else 0.0,
        max_trailer_capacity=max(capacities) if capacities else 0.0,
        initial_trailer_inventory=sum(trailer.initial_quantity for trailer in instance.trailers),
    )


def rolling_days(
    instance: Instance,
    solution: Solution | None = None,
    *,
    monitor_days: int | None = None,
) -> list[RollingDay]:
    days = (instance.horizon * instance.unit + MINUTES_PER_DAY - 1) // MINUTES_PER_DAY
    monitor_days = days if monitor_days is None else min(monitor_days, days)
    daily_vmi_consumption = _daily_vmi_consumption(instance, days)
    daily_call_in_due = _daily_call_in_due(instance, days)
    cumulative_required = _cumulative_required_delivery(instance, days)
    total_required = cumulative_required[-1] if cumulative_required else 0.0
    smooth_average = total_required / days if days else 0.0
    planned_buckets = (
        period_buckets(instance, solution, period_minutes=MINUTES_PER_DAY)
        if solution is not None
        else []
    )
    total_planned = sum(bucket.delivered_quantity for bucket in planned_buckets)
    smooth_planned_average = total_planned / days if days and planned_buckets else 0.0

    rows: list[RollingDay] = []
    cumulative_consumption = 0.0
    cumulative_planned = 0.0
    planned_so_far: list[float] = []

    for day in range(monitor_days):
        planned_delivered = (
            planned_buckets[day].delivered_quantity
            if day < len(planned_buckets)
            else 0.0
        )
        planned_starts = (
            planned_buckets[day].shift_starts
            if day < len(planned_buckets)
            else 0
        )
        cumulative_consumption += daily_vmi_consumption[day]
        cumulative_planned += planned_delivered
        planned_so_far.append(planned_delivered)
        smooth_target_to_date = smooth_average * (day + 1)
        planned_average_target_to_date = smooth_planned_average * (day + 1)
        day_t = _one_sample_t_statistic(planned_so_far, smooth_average)

        rows.append(
            RollingDay(
                day=day,
                start_minute=day * MINUTES_PER_DAY,
                end_minute=min((day + 1) * MINUTES_PER_DAY, instance.horizon * instance.unit),
                vmi_consumption=daily_vmi_consumption[day],
                call_in_due_quantity=daily_call_in_due[day],
                cumulative_vmi_consumption=cumulative_consumption,
                cumulative_required_delivery=cumulative_required[day],
                cumulative_required_delivery_share=(
                    0.0 if total_required == 0 else cumulative_required[day] / total_required
                ),
                smooth_required_daily_average=smooth_average,
                smooth_planned_daily_average=smooth_planned_average,
                planned_delivered=planned_delivered,
                planned_shift_starts=planned_starts,
                cumulative_planned_delivered=cumulative_planned,
                cumulative_delivery_gap_to_required=cumulative_planned - cumulative_required[day],
                cumulative_delivery_gap_to_smooth_target=cumulative_planned - smooth_target_to_date,
                cumulative_delivery_gap_to_planned_average=(
                    cumulative_planned - planned_average_target_to_date
                ),
                daily_delivery_deviation_from_smooth_target=planned_delivered - smooth_average,
                daily_delivery_t_statistic=day_t,
            )
        )

    return rows


def rolling_summary(
    instance: Instance,
    solution: Solution | None = None,
    *,
    monitor_days: int | None = None,
) -> RollingSummary:
    days = (instance.horizon * instance.unit + MINUTES_PER_DAY - 1) // MINUTES_PER_DAY
    rows = rolling_days(instance, solution, monitor_days=monitor_days)
    total_vmi = sum(_daily_vmi_consumption(instance, days))
    total_call_in = sum(_daily_call_in_due(instance, days))
    total_required = _cumulative_required_delivery(instance, days)[-1]
    monitored_deliveries = [row.planned_delivered for row in rows]
    monitored_days = len(rows)
    smooth_average = total_required / days if days else 0.0
    smooth_planned_average = (
        sum(bucket.delivered_quantity for bucket in period_buckets(instance, solution))
        / days
        if solution is not None and days
        else 0.0
    )

    return RollingSummary(
        days=days,
        total_vmi_consumption=total_vmi,
        total_call_in_due_quantity=total_call_in,
        total_required_delivery=total_required,
        smooth_required_daily_average=smooth_average,
        smooth_planned_daily_average=smooth_planned_average,
        fleet=fleet_profile(instance),
        monitored_days=monitored_days,
        monitored_planned_delivered=sum(monitored_deliveries),
        monitored_required_delivery=rows[-1].cumulative_required_delivery if rows else 0.0,
        monitored_gap_to_required=rows[-1].cumulative_delivery_gap_to_required if rows else 0.0,
        monitored_gap_to_smooth_target=(
            rows[-1].cumulative_delivery_gap_to_smooth_target if rows else 0.0
        ),
        monitored_gap_to_planned_average=(
            rows[-1].cumulative_delivery_gap_to_planned_average if rows else 0.0
        ),
        monitored_delivery_cv=_cv(monitored_deliveries),
        monitored_delivery_t_statistic=_one_sample_t_statistic(
            monitored_deliveries,
            smooth_average,
        ),
    )


def _daily_vmi_consumption(instance: Instance, days: int) -> list[float]:
    values = [0.0 for _ in range(days)]
    steps_per_day = MINUTES_PER_DAY // instance.unit
    for customer in instance.customers:
        if customer.call_in:
            continue
        for step, forecast in enumerate(customer.forecast):
            day = min(step // steps_per_day, days - 1)
            values[day] += forecast
    return values


def _daily_call_in_due(instance: Instance, days: int) -> list[float]:
    values = [0.0 for _ in range(days)]
    for customer in instance.customers:
        if not customer.call_in:
            continue
        for order in customer.orders:
            day = min(max(order.latest_time // MINUTES_PER_DAY, 0), days - 1)
            values[day] += order.quantity
    return values


def _cumulative_required_delivery(instance: Instance, days: int) -> list[float]:
    steps_per_day = MINUTES_PER_DAY // instance.unit
    required_by_day = [0.0 for _ in range(days)]

    for customer in instance.customers:
        if customer.call_in:
            continue
        surplus_above_safety = customer.initial_tank_quantity - customer.safety_level
        cumulative_forecast = 0.0
        for day in range(days):
            start = day * steps_per_day
            end = min((day + 1) * steps_per_day, len(customer.forecast))
            cumulative_forecast += sum(customer.forecast[start:end])
            required_by_day[day] += max(0.0, cumulative_forecast - surplus_above_safety)

    return required_by_day


def _cv(values: list[float]) -> float:
    if not values:
        return 0.0
    avg = mean(values)
    if abs(avg) < 1e-12:
        return 0.0
    return (stdev(values) if len(values) > 1 else 0.0) / avg


def _one_sample_t_statistic(values: list[float], target_mean: float) -> float | None:
    if len(values) < 2:
        return None
    sample_std = stdev(values)
    if sample_std <= 1e-12:
        if abs(mean(values) - target_mean) <= 1e-12:
            return 0.0
        return None
    return (mean(values) - target_mean) / (sample_std / sqrt(len(values)))
