"""Monte Carlo consumption scenarios and hedged instance construction.

Generates physics-consistent consumption perturbations and builds a single
hedged Instance by taking time-dependent percentiles across K scenarios.
The solver operates on the hedged instance unchanged — uncertainty is handled
entirely at the input layer.
"""

from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..model import Customer, Instance, Solution

MINUTES_PER_DAY = 1440


@dataclass(frozen=True)
class ForecastDistribution:
    """Forecast uncertainty container for stochastic IRP planning.

    ``samples`` maps customer point index to sampled forecast paths.
    ``quantiles`` maps percentile to customer forecast paths.  This is the
    boundary future TabPFN outputs should satisfy; TabPFN is intentionally not a
    dependency of this module.
    """

    deterministic: dict[int, tuple[float, ...]]
    samples: dict[int, list[tuple[float, ...]]]
    quantiles: dict[float, dict[int, tuple[float, ...]]]

    @classmethod
    def from_instance(cls, instance: Instance) -> ForecastDistribution:
        deterministic = {
            customer.index: customer.forecast
            for customer in instance.customers
            if not customer.call_in
        }
        return cls(deterministic=deterministic, samples={}, quantiles={})

    @classmethod
    def from_samples(
        cls,
        instance: Instance,
        samples: dict[int, list[tuple[float, ...]]],
    ) -> ForecastDistribution:
        deterministic = {
            customer.index: customer.forecast
            for customer in instance.customers
            if not customer.call_in
        }
        return cls(deterministic=deterministic, samples=samples, quantiles={})

    def sample_count(self) -> int:
        return max((len(paths) for paths in self.samples.values()), default=0)

    def scenario_count(self) -> int:
        if self.sample_count():
            return self.sample_count()
        if self.quantiles:
            return len(_default_scenario_percentiles(self))
        return 0


def load_forecast_distribution(
    instance: Instance,
    path: str | Path,
) -> ForecastDistribution:
    """Load external forecast paths as solver input.

    Supported row-oriented schemas:
    - wide quantiles: ``item_id,step,0.5,0.75,0.9`` or ``customer_id,step,q50,q90``
    - long quantiles: ``item_id,step,quantile,value``
    - deterministic: ``item_id,step,target``

    Quantiles may be expressed as fractions (``0.9``) or percentiles (``90``).
    Missing steps fall back to the deterministic instance forecast.
    """
    rows = _read_forecast_rows(Path(path))
    return forecast_distribution_from_rows(instance, rows)


def route_wrapped_dummy_distribution(
    instance: Instance,
    solution: Solution,
    *,
    present_day: int = 0,
    quantiles: tuple[float, ...] = (50.0, 75.0, 90.0, 95.0),
    base_relative_width: float = 0.05,
    daily_relative_growth: float = 0.04,
    max_relative_width: float = 0.60,
    route_anchor_width: float = 0.08,
) -> ForecastDistribution:
    """Build dummy forecast quantiles anchored around planned customer visits.

    This is intentionally a fake CI producer for solver experiments.  It does
    not forecast; it wraps the instance's current forecast with widening
    quantile bands.  Planned visits in ``solution`` act as route anchors, so
    uncertainty tightens around an actual route day and grows as the horizon
    moves away from both the present and the nearest planned visit.
    """
    deterministic = {
        customer.index: customer.forecast
        for customer in instance.customers
        if not customer.call_in
    }
    quantile_paths: dict[float, dict[int, tuple[float, ...]]] = {
        _normalize_percentile(quantile): {} for quantile in quantiles
    }
    route_days = _customer_route_days(instance, solution)
    steps_per_day = max(1, MINUTES_PER_DAY // instance.unit)

    for customer in instance.customers:
        if customer.call_in:
            continue
        customer_route_days = route_days.get(customer.index, ())
        paths = {percentile: [] for percentile in quantile_paths}
        for step, base_value in enumerate(customer.forecast):
            day = step // steps_per_day
            horizon_days = max(0, day - present_day)
            nearest_route_gap = (
                min(abs(day - route_day) for route_day in customer_route_days)
                if customer_route_days
                else horizon_days
            )
            relative_width = min(
                max_relative_width,
                base_relative_width + daily_relative_growth * horizon_days,
            )
            if customer_route_days:
                relative_width = min(
                    relative_width,
                    route_anchor_width
                    + daily_relative_growth * max(0, nearest_route_gap),
                )
            for percentile in paths:
                scale = _quantile_scale(percentile)
                paths[percentile].append(max(0.0, base_value * (1.0 + relative_width * scale)))
        for percentile, values in paths.items():
            quantile_paths[percentile][customer.index] = tuple(values)

    return ForecastDistribution(
        deterministic=deterministic,
        samples={},
        quantiles=quantile_paths,
    )


def write_forecast_distribution_csv(
    distribution: ForecastDistribution,
    path: str | Path,
) -> None:
    """Write quantile forecasts in the external forecast-input CSV schema."""
    path = Path(path)
    quantiles = sorted(distribution.quantiles)
    fieldnames = ["item_id", "step", "target", *(_format_quantile(q) for q in quantiles)]
    rows = []
    for customer_id, deterministic_path in sorted(distribution.deterministic.items()):
        for step, target in enumerate(deterministic_path):
            row = {"item_id": customer_id, "step": step, "target": target}
            for quantile in quantiles:
                row[_format_quantile(quantile)] = distribution.quantiles.get(
                    quantile,
                    {},
                ).get(customer_id, deterministic_path)[step]
            rows.append(row)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def forecast_distribution_from_rows(
    instance: Instance,
    rows: list[dict[str, Any]],
) -> ForecastDistribution:
    rows = _rows_with_steps(rows)
    deterministic = {
        customer.index: list(customer.forecast)
        for customer in instance.customers
        if not customer.call_in
    }
    quantiles: dict[float, dict[int, list[float]]] = {}

    for row in rows:
        customer_id = _row_customer_id(row)
        if customer_id not in deterministic:
            continue
        step = _row_step(row)
        if step is None or step < 0 or step >= len(deterministic[customer_id]):
            continue
        long_quantile = _row_value(row, ("quantile", "percentile", "q"))
        if long_quantile is not None:
            value = _row_value(row, ("value", "forecast", "target", "prediction", "yhat"))
            if value is None:
                continue
            percentile = _normalize_percentile(long_quantile)
            quantiles.setdefault(percentile, {}).setdefault(
                customer_id,
                list(deterministic[customer_id]),
            )[step] = max(0.0, value)
            continue

        deterministic_value = _row_value(row, ("target", "forecast", "prediction", "yhat"))
        if deterministic_value is not None:
            deterministic[customer_id][step] = max(0.0, deterministic_value)
        for key, value in row.items():
            percentile = _parse_quantile_column(key)
            if percentile is None:
                continue
            parsed = _coerce_float(value)
            if parsed is None:
                continue
            quantiles.setdefault(percentile, {}).setdefault(
                customer_id,
                list(deterministic[customer_id]),
            )[step] = max(0.0, parsed)

    return ForecastDistribution(
        deterministic={key: tuple(value) for key, value in deterministic.items()},
        samples={},
        quantiles={
            percentile: {
                customer_id: tuple(path)
                for customer_id, path in customer_paths.items()
            }
            for percentile, customer_paths in quantiles.items()
        },
    )


def _read_forecast_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".json":
        with path.open() as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return [dict(row) for row in payload]
        if isinstance(payload, dict) and isinstance(payload.get("forecasts"), list):
            return [dict(row) for row in payload["forecasts"]]
        raise ValueError(f"Unsupported JSON forecast schema in {path}")
    if suffix == ".jsonl":
        with path.open() as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "Reading parquet forecast inputs requires pandas/pyarrow."
            ) from exc
        frame = pd.read_parquet(path)
        return frame.to_dict(orient="records")
    raise ValueError(f"Unsupported forecast input format: {path.suffix}")


def _rows_with_steps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if all(_row_step(row) is not None for row in rows):
        return rows
    timestamp_keys = ("timestamp", "date", "datetime")
    groups: dict[int, list[dict[str, Any]]] = {}
    passthrough = []
    for row in rows:
        if _row_step(row) is not None:
            passthrough.append(row)
            continue
        customer_id = _row_customer_id(row)
        if customer_id is None or not any(key in row for key in timestamp_keys):
            passthrough.append(row)
            continue
        groups.setdefault(customer_id, []).append(row)
    normalized = list(passthrough)
    for customer_rows in groups.values():
        customer_rows.sort(key=_row_timestamp_key)
        for step, row in enumerate(customer_rows):
            updated = dict(row)
            updated["step"] = step
            normalized.append(updated)
    return normalized


def _row_timestamp_key(row: dict[str, Any]) -> str:
    for key in ("timestamp", "date", "datetime"):
        if key in row:
            return str(row[key])
    return ""


def _row_customer_id(row: dict[str, Any]) -> int | None:
    for key in ("customer_id", "customer", "item_id", "point", "point_id", "index"):
        if key in row:
            return _parse_customer_id(row[key])
    return None


def _parse_customer_id(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    match = re.search(r"(\d+)$", str(value))
    return int(match.group(1)) if match else None


def _row_step(row: dict[str, Any]) -> int | None:
    for key in ("step", "time_step", "horizon_step", "t"):
        if key in row:
            value = _coerce_float(row[key])
            return int(value) if value is not None else None
    return None


def _row_value(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in row:
            value = _coerce_float(row[key])
            if value is not None:
                return value
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if value == "":
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _parse_quantile_column(key: Any) -> float | None:
    label = str(key).strip().lower()
    if label in {
        "item_id",
        "customer_id",
        "customer",
        "point",
        "point_id",
        "index",
        "step",
        "time_step",
        "horizon_step",
        "timestamp",
        "date",
        "datetime",
        "target",
        "forecast",
        "prediction",
        "yhat",
        "value",
        "quantile",
        "percentile",
        "q",
    }:
        return None
    if label.startswith(("q", "p")):
        label = label[1:]
    value = _coerce_float(label)
    if value is None:
        return None
    if 0.0 < value <= 1.0:
        return value * 100.0
    if 1.0 < value <= 100.0:
        return value
    return None


def _normalize_percentile(value: float) -> float:
    if 0.0 < value <= 1.0:
        return value * 100.0
    if 1.0 < value <= 100.0:
        return value
    raise ValueError(f"Invalid quantile/percentile value: {value}")


def _customer_route_days(instance: Instance, solution: Solution) -> dict[int, tuple[int, ...]]:
    customer_ids = {customer.index for customer in instance.customers if not customer.call_in}
    days: dict[int, set[int]] = {customer_id: set() for customer_id in customer_ids}
    for shift in solution.shifts:
        for operation in shift.operations:
            if operation.point in days and operation.quantity > 0:
                days[operation.point].add(max(0, operation.arrival // MINUTES_PER_DAY))
    return {
        customer_id: tuple(sorted(customer_days))
        for customer_id, customer_days in days.items()
        if customer_days
    }


def _quantile_scale(percentile: float) -> float:
    if percentile <= 50.0:
        return 0.0
    return min(2.0, (percentile - 50.0) / 50.0)


def _format_quantile(percentile: float) -> str:
    value = percentile / 100.0
    return f"{value:.3f}".rstrip("0").rstrip(".")


def generate_scenario_forecast(
    customer: Customer,
    unit: int,
    *,
    rng: random.Random,
    commit_end_day: int,
    day_sigma_schedule: dict[str, float],
    horizon_days: int,
) -> tuple[float, ...]:
    """Generate one physics-consistent consumption scenario for a customer.

    Perturbation is applied per-day as a multiplicative rate scaling factor,
    preserving intra-day consumption shape.  The sigma grows with distance
    from the commit boundary according to ``day_sigma_schedule``:

        commit window:  sigma = 0  (true data)
        plan window:    sigma = day_sigma_schedule["plan"]
        buffer window:  sigma = day_sigma_schedule["buffer"]

    After perturbation, a physics clamp ensures cumulative consumption never
    exceeds the available tank quantity (no negative tank levels).
    """
    steps_per_day = MINUTES_PER_DAY // unit
    base = list(customer.forecast)
    noised = list(base)

    plan_sigma = day_sigma_schedule.get("plan", 0.15)
    buffer_sigma = day_sigma_schedule.get("buffer", 0.30)

    for day in range(horizon_days):
        start = day * steps_per_day
        end = min(start + steps_per_day, len(base))
        if start >= len(base):
            break

        if day < commit_end_day:
            # commit window: true data, no noise
            continue

        # sigma grows with distance from commit boundary
        distance = day - commit_end_day
        plan_days = commit_end_day  # plan window is same width as commit
        if distance < plan_days:
            sigma = plan_sigma
        else:
            sigma = buffer_sigma

        multiplier = max(0.0, rng.gauss(1.0, sigma))
        for step in range(start, end):
            noised[step] = max(0.0, base[step] * multiplier)

    # Note: we do NOT clamp cumulative consumption to initial tank level here.
    # Consumption forecasts represent demand that will be met by deliveries;
    # the solver's inventory simulation handles tank-level physics with
    # deliveries included.

    return tuple(noised)


def generate_scenarios(
    instance: Instance,
    *,
    n_scenarios: int = 20,
    seed: int = 42,
    commit_end_day: int = 7,
    day_sigma_schedule: dict[str, float] | None = None,
) -> dict[int, list[tuple[float, ...]]]:
    """Generate K consumption scenarios for every VMI customer.

    Returns a dict mapping customer point index to a list of K forecast
    tuples, each representing one physics-consistent consumption scenario.
    """
    if day_sigma_schedule is None:
        day_sigma_schedule = {"plan": 0.15, "buffer": 0.30}

    rng = random.Random(seed)
    horizon_days = instance.horizon * instance.unit // MINUTES_PER_DAY

    scenarios: dict[int, list[tuple[float, ...]]] = {}
    for customer in instance.customers:
        if customer.call_in:
            continue
        customer_scenarios = []
        for _ in range(n_scenarios):
            scenario = generate_scenario_forecast(
                customer,
                instance.unit,
                rng=rng,
                commit_end_day=commit_end_day,
                day_sigma_schedule=day_sigma_schedule,
                horizon_days=horizon_days,
            )
            customer_scenarios.append(scenario)
        scenarios[customer.index] = customer_scenarios

    return scenarios


def build_hedged_instance(
    instance: Instance,
    scenarios: dict[int, list[tuple[float, ...]]],
    *,
    commit_end_day: int = 7,
    plan_end_day: int = 14,
    commit_percentile: float = 50.0,
    plan_percentile: float = 75.0,
    buffer_percentile: float = 90.0,
    capacity_buffer: float = 0.05,
) -> Instance:
    """Build a single hedged Instance from K scenarios.

    For each customer and timestep, the hedged forecast is the appropriate
    percentile across all K scenarios:

    - commit window (day < commit_end_day):  p50 (= true data when σ=0)
    - plan window (commit_end_day <= day < plan_end_day):  p75
    - buffer window (day >= plan_end_day):  p90

    Higher percentiles mean more pessimistic (higher) consumption, which
    causes the solver to over-deliver as a robustness buffer.

    The `capacity_buffer` (e.g., 0.05 for 5%) reduces the tank capacity in the
    hedged instance, leaving physical headroom on the true instance to
    prevent overfills during over-delivery.
    """
    steps_per_day = MINUTES_PER_DAY // instance.unit
    hedged_customers = []

    for customer in instance.customers:
        if customer.call_in or customer.index not in scenarios:
            hedged_customers.append(customer)
            continue

        customer_scenarios = scenarios[customer.index]
        n_steps = len(customer.forecast)

        # Stack scenarios: shape (K, n_steps)
        scenario_array = np.array(customer_scenarios, dtype=np.float64)

        hedged_forecast = np.empty(n_steps, dtype=np.float64)
        for step in range(n_steps):
            day = step // steps_per_day
            if day < commit_end_day:
                pct = commit_percentile
            elif day < plan_end_day:
                pct = plan_percentile
            else:
                pct = buffer_percentile
            hedged_forecast[step] = max(
                0.0,
                np.percentile(scenario_array[:, step], pct),
            )

        # Shrink capacity to leave headroom for hedging over-delivery
        hedged_capacity = customer.capacity * (1.0 - capacity_buffer)

        hedged_customers.append(
            replace(
                customer,
                forecast=tuple(hedged_forecast.tolist()),
                capacity=hedged_capacity,
            )
        )

    return replace(instance, customers=tuple(hedged_customers))


def build_hedged_instance_from_distribution(
    instance: Instance,
    distribution: ForecastDistribution,
    *,
    commit_end_day: int = 7,
    plan_end_day: int = 14,
    commit_percentile: float = 50.0,
    plan_percentile: float = 75.0,
    buffer_percentile: float = 90.0,
    capacity_buffer: float = 0.05,
) -> Instance:
    """Build a hedged instance from quantile paths or sampled paths."""
    if distribution.quantiles:
        return _build_quantile_hedged_instance(
            instance,
            distribution,
            commit_end_day=commit_end_day,
            plan_end_day=plan_end_day,
            commit_percentile=commit_percentile,
            plan_percentile=plan_percentile,
            buffer_percentile=buffer_percentile,
            capacity_buffer=capacity_buffer,
        )
    return build_hedged_instance(
        instance,
        distribution.samples,
        commit_end_day=commit_end_day,
        plan_end_day=plan_end_day,
        commit_percentile=commit_percentile,
        plan_percentile=plan_percentile,
        buffer_percentile=buffer_percentile,
        capacity_buffer=capacity_buffer,
    )


def build_scenario_instance(
    instance: Instance,
    scenarios: dict[int, list[tuple[float, ...]]],
    scenario_index: int,
) -> Instance:
    """Build an instance using one sampled forecast for each VMI customer."""
    scenario_customers = []
    for customer in instance.customers:
        customer_scenarios = scenarios.get(customer.index)
        if customer.call_in or not customer_scenarios:
            scenario_customers.append(customer)
            continue
        if not (0 <= scenario_index < len(customer_scenarios)):
            raise IndexError(
                f"scenario_index={scenario_index} out of range for customer {customer.index}"
            )
        scenario_customers.append(
            replace(customer, forecast=customer_scenarios[scenario_index])
        )
    return replace(instance, customers=tuple(scenario_customers))


def build_scenario_instance_from_distribution(
    instance: Instance,
    distribution: ForecastDistribution,
    scenario_index: int,
) -> Instance:
    return build_scenario_instance(
        instance,
        scenarios_from_distribution(distribution),
        scenario_index,
    )


def scenarios_from_distribution(
    distribution: ForecastDistribution,
    *,
    percentiles: tuple[float, ...] | None = None,
) -> dict[int, list[tuple[float, ...]]]:
    """Return sample paths for scenario validation/backtesting.

    If explicit samples are present, they are used unchanged.  Otherwise
    quantile paths are converted into deterministic stress scenarios.  This is
    deliberately not a random sampler; it gives stable p50/p75/p90/... paths
    that make backtests and solver acceptance reproducible.
    """
    if distribution.samples:
        return distribution.samples
    if not distribution.quantiles:
        return {}

    requested = percentiles or _default_scenario_percentiles(distribution)
    scenarios: dict[int, list[tuple[float, ...]]] = {
        customer_id: [] for customer_id in distribution.deterministic
    }
    for percentile in requested:
        for customer_id, deterministic_path in distribution.deterministic.items():
            scenarios[customer_id].append(
                _interpolated_quantile_path(
                    distribution,
                    percentile,
                    customer_id,
                    deterministic_path,
                )
            )
    return scenarios


def _build_quantile_hedged_instance(
    instance: Instance,
    distribution: ForecastDistribution,
    *,
    commit_end_day: int,
    plan_end_day: int,
    commit_percentile: float,
    plan_percentile: float,
    buffer_percentile: float,
    capacity_buffer: float,
) -> Instance:
    steps_per_day = MINUTES_PER_DAY // instance.unit
    percentiles = sorted(distribution.quantiles)

    def nearest_path(percentile: float, customer_id: int, fallback: tuple[float, ...]) -> tuple[float, ...]:
        if not percentiles:
            return fallback
        nearest = min(percentiles, key=lambda value: abs(value - percentile))
        return distribution.quantiles.get(nearest, {}).get(customer_id, fallback)

    customers = []
    for customer in instance.customers:
        if customer.call_in:
            customers.append(customer)
            continue
        commit_path = nearest_path(commit_percentile, customer.index, customer.forecast)
        plan_path = nearest_path(plan_percentile, customer.index, customer.forecast)
        buffer_path = nearest_path(buffer_percentile, customer.index, customer.forecast)
        forecast = []
        for step in range(len(customer.forecast)):
            day = step // steps_per_day
            if day < commit_end_day:
                forecast.append(commit_path[step])
            elif day < plan_end_day:
                forecast.append(plan_path[step])
            else:
                forecast.append(buffer_path[step])
        customers.append(
            replace(
                customer,
                forecast=tuple(forecast),
                capacity=customer.capacity * (1.0 - capacity_buffer),
            )
        )
    return replace(instance, customers=tuple(customers))


def _default_scenario_percentiles(
    distribution: ForecastDistribution,
) -> tuple[float, ...]:
    available = sorted(distribution.quantiles)
    defaults = (50.0, 75.0, 90.0, 95.0)
    selected: list[float] = []
    for percentile in defaults:
        if available:
            selected.append(min(available, key=lambda value: abs(value - percentile)))
    if not selected:
        return ()
    return tuple(dict.fromkeys(selected))


def _interpolated_quantile_path(
    distribution: ForecastDistribution,
    percentile: float,
    customer_id: int,
    fallback: tuple[float, ...],
) -> tuple[float, ...]:
    available = sorted(
        quantile
        for quantile, customer_paths in distribution.quantiles.items()
        if customer_id in customer_paths
    )
    if not available:
        return fallback
    percentile = _normalize_percentile(percentile)
    if percentile <= available[0]:
        return distribution.quantiles[available[0]][customer_id]
    if percentile >= available[-1]:
        return distribution.quantiles[available[-1]][customer_id]
    lower = max(value for value in available if value <= percentile)
    upper = min(value for value in available if value >= percentile)
    lower_path = distribution.quantiles[lower][customer_id]
    if lower == upper:
        return lower_path
    upper_path = distribution.quantiles[upper][customer_id]
    weight = (percentile - lower) / (upper - lower)
    return tuple(
        max(0.0, low + (high - low) * weight)
        for low, high in zip(lower_path, upper_path)
    )
