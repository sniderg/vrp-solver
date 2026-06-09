from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

from .history import RealizedConsumptionRow
from .scenario import ForecastDistribution


@dataclass(frozen=True)
class ForecastCalibrationRow:
    customer_id: int | str
    horizon_step: int | str
    count: int
    mean_bias: float
    mae: float
    underforecast_rate: float
    overforecast_rate: float
    p50_hit_rate: float | None = None
    p75_hit_rate: float | None = None
    p90_hit_rate: float | None = None
    p95_hit_rate: float | None = None

    def flat(self) -> dict[str, object]:
        return {
            "customer_id": self.customer_id,
            "horizon_step": self.horizon_step,
            "count": self.count,
            "mean_bias": self.mean_bias,
            "mae": self.mae,
            "underforecast_rate": self.underforecast_rate,
            "overforecast_rate": self.overforecast_rate,
            "p50_hit_rate": "" if self.p50_hit_rate is None else self.p50_hit_rate,
            "p75_hit_rate": "" if self.p75_hit_rate is None else self.p75_hit_rate,
            "p90_hit_rate": "" if self.p90_hit_rate is None else self.p90_hit_rate,
            "p95_hit_rate": "" if self.p95_hit_rate is None else self.p95_hit_rate,
        }


def forecast_calibration_report(
    distribution: ForecastDistribution,
    realized_rows: tuple[RealizedConsumptionRow, ...],
    *,
    known_customers: set[int] | None = None,
) -> tuple[ForecastCalibrationRow, ...]:
    groups: dict[tuple[int, int], list[tuple[float, float, dict[float, float]]]] = {}
    for row in realized_rows:
        if row.step is None:
            continue
        if known_customers is not None and row.customer_id not in known_customers:
            continue
        deterministic = distribution.deterministic.get(row.customer_id)
        if deterministic is None or row.step < 0 or row.step >= len(deterministic):
            continue
        q_values = {
            percentile: path[row.step]
            for percentile, paths in distribution.quantiles.items()
            for customer_id, path in paths.items()
            if customer_id == row.customer_id and row.step < len(path)
        }
        groups.setdefault((row.customer_id, row.step), []).append(
            (deterministic[row.step], row.realized_consumption, q_values)
        )

    step_rows = [_calibration_row(customer_id, step, samples) for (customer_id, step), samples in sorted(groups.items())]
    rows = list(step_rows)
    for customer_id in sorted({customer_id for customer_id, _step in groups}):
        customer_samples = [
            sample
            for (group_customer, _step), samples in groups.items()
            if group_customer == customer_id
            for sample in samples
        ]
        rows.append(_calibration_row(customer_id, "all", customer_samples))
    if rows:
        rows.append(_aggregate_row(step_rows))
    return tuple(rows)


def write_calibration_csv(rows: tuple[ForecastCalibrationRow, ...], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ForecastCalibrationRow("aggregate", "all", 0, 0.0, 0.0, 0.0, 0.0).flat())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.flat())


def load_calibration_csv(path: str | Path) -> tuple[ForecastCalibrationRow, ...]:
    with Path(path).open(newline="") as handle:
        return tuple(_calibration_from_dict(row) for row in csv.DictReader(handle))


def calibrate_forecast_distribution(
    distribution: ForecastDistribution,
    calibration_report: tuple[ForecastCalibrationRow, ...],
    *,
    minimum_widening: float = 0.05,
    max_widening: float = 1.0,
) -> ForecastDistribution:
    by_customer = {
        int(row.customer_id): row
        for row in calibration_report
        if str(row.customer_id).isdigit()
    }
    aggregate = next((row for row in calibration_report if str(row.customer_id) == "aggregate"), None)

    deterministic = {}
    for customer_id, path in distribution.deterministic.items():
        row = by_customer.get(customer_id, aggregate)
        multiplier = _bias_multiplier(row)
        deterministic[customer_id] = tuple(max(0.0, value * multiplier) for value in path)

    quantiles = {}
    for percentile, customer_paths in distribution.quantiles.items():
        quantiles[percentile] = {}
        for customer_id, path in customer_paths.items():
            row = by_customer.get(customer_id, aggregate)
            bias = _bias_multiplier(row)
            widening = _widening_multiplier(row, percentile, minimum_widening, max_widening)
            center = distribution.deterministic.get(customer_id, path)
            adjusted = []
            for idx, value in enumerate(path):
                base = center[idx] if idx < len(center) else value
                adjusted.append(max(0.0, base * bias + max(0.0, value - base) * widening))
            quantiles[percentile][customer_id] = tuple(adjusted)

    return replace(distribution, deterministic=deterministic, quantiles=quantiles)


def _calibration_row(
    customer_id: int,
    step: int | str,
    samples: list[tuple[float, float, dict[float, float]]],
) -> ForecastCalibrationRow:
    errors = [forecast - realized for forecast, realized, _ in samples]
    quantile_rates = {}
    for percentile in (50.0, 75.0, 90.0, 95.0):
        hits = [
            1.0 if realized <= q_values[percentile] else 0.0
            for _, realized, q_values in samples
            if percentile in q_values
        ]
        quantile_rates[percentile] = _mean(hits) if hits else None
    return ForecastCalibrationRow(
        customer_id=customer_id,
        horizon_step=step,
        count=len(samples),
        mean_bias=_mean(errors),
        mae=_mean(abs(error) for error in errors),
        underforecast_rate=_mean(1.0 if error < 0.0 else 0.0 for error in errors),
        overforecast_rate=_mean(1.0 if error > 0.0 else 0.0 for error in errors),
        p50_hit_rate=quantile_rates[50.0],
        p75_hit_rate=quantile_rates[75.0],
        p90_hit_rate=quantile_rates[90.0],
        p95_hit_rate=quantile_rates[95.0],
    )


def _aggregate_row(rows: list[ForecastCalibrationRow]) -> ForecastCalibrationRow:
    total = sum(row.count for row in rows)
    return ForecastCalibrationRow(
        customer_id="aggregate",
        horizon_step="all",
        count=total,
        mean_bias=_weighted(rows, "mean_bias"),
        mae=_weighted(rows, "mae"),
        underforecast_rate=_weighted(rows, "underforecast_rate"),
        overforecast_rate=_weighted(rows, "overforecast_rate"),
        p50_hit_rate=_weighted_optional(rows, "p50_hit_rate"),
        p75_hit_rate=_weighted_optional(rows, "p75_hit_rate"),
        p90_hit_rate=_weighted_optional(rows, "p90_hit_rate"),
        p95_hit_rate=_weighted_optional(rows, "p95_hit_rate"),
    )


def _calibration_from_dict(row: dict[str, str]) -> ForecastCalibrationRow:
    customer = row["customer_id"]
    return ForecastCalibrationRow(
        customer_id=int(customer) if customer.isdigit() else customer,
        horizon_step=int(row["horizon_step"]) if row["horizon_step"].isdigit() else row["horizon_step"],
        count=int(float(row.get("count") or 0)),
        mean_bias=float(row.get("mean_bias") or 0.0),
        mae=float(row.get("mae") or 0.0),
        underforecast_rate=float(row.get("underforecast_rate") or 0.0),
        overforecast_rate=float(row.get("overforecast_rate") or 0.0),
        p50_hit_rate=_optional_float(row.get("p50_hit_rate")),
        p75_hit_rate=_optional_float(row.get("p75_hit_rate")),
        p90_hit_rate=_optional_float(row.get("p90_hit_rate")),
        p95_hit_rate=_optional_float(row.get("p95_hit_rate")),
    )


def _bias_multiplier(row: ForecastCalibrationRow | None) -> float:
    if row is None or row.mean_bias >= 0.0:
        return 1.0
    denominator = max(1.0, row.mae)
    return 1.0 + min(0.50, abs(row.mean_bias) / denominator * 0.10)


def _widening_multiplier(
    row: ForecastCalibrationRow | None,
    percentile: float,
    minimum_widening: float,
    max_widening: float,
) -> float:
    if row is None:
        return 1.0
    hit_rate = {
        50.0: row.p50_hit_rate,
        75.0: row.p75_hit_rate,
        90.0: row.p90_hit_rate,
        95.0: row.p95_hit_rate,
    }.get(percentile)
    target = percentile / 100.0
    if hit_rate is None or hit_rate >= target:
        return 1.0
    shortfall = target - hit_rate
    return 1.0 + min(max_widening, max(minimum_widening, shortfall / max(target, 0.01)))


def _mean(values) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _weighted(rows: list[ForecastCalibrationRow], attr: str) -> float:
    total = sum(row.count for row in rows)
    return sum(getattr(row, attr) * row.count for row in rows) / total if total else 0.0


def _weighted_optional(rows: list[ForecastCalibrationRow], attr: str) -> float | None:
    usable = [row for row in rows if getattr(row, attr) is not None]
    total = sum(row.count for row in usable)
    return sum(getattr(row, attr) * row.count for row in usable) / total if total else None


def _optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)
