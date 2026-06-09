from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..model import Instance, Solution

MINUTES_PER_DAY = 1440


@dataclass(frozen=True)
class RealizedConsumptionRow:
    customer_id: int
    timestamp: str | None
    step: int | None
    realized_consumption: float
    inventory_observed: float | None = None
    delivered_quantity: float | None = None
    source: str = ""

    def flat(self) -> dict[str, object]:
        return {
            "customer_id": self.customer_id,
            "timestamp": self.timestamp or "",
            "step": "" if self.step is None else self.step,
            "realized_consumption": self.realized_consumption,
            "inventory_observed": "" if self.inventory_observed is None else self.inventory_observed,
            "delivered_quantity": "" if self.delivered_quantity is None else self.delivered_quantity,
            "source": self.source,
        }


def load_realized_consumption_history(path: str | Path) -> tuple[RealizedConsumptionRow, ...]:
    rows = _read_rows(Path(path))
    return tuple(_normalize_row(row, index) for index, row in enumerate(rows, start=2))


def write_realized_consumption_csv(rows: tuple[RealizedConsumptionRow, ...], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "customer_id",
        "timestamp",
        "step",
        "realized_consumption",
        "inventory_observed",
        "delivered_quantity",
        "source",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.flat())


def realized_history_from_solution_week(
    instance: Instance,
    solution: Solution,
    *,
    history_days: int = 7,
) -> tuple[RealizedConsumptionRow, ...]:
    """Use an instance's first-week consumption path as realized history.

    The solution contributes optional delivered-quantity context only.  It is
    not used to infer required route frequency or consumption labels.
    """
    max_step = min(instance.horizon, max(0, history_days) * MINUTES_PER_DAY // instance.unit)
    delivered = _delivered_by_customer_step(instance, solution, max_step)
    rows: list[RealizedConsumptionRow] = []
    for customer in instance.customers:
        if customer.call_in:
            continue
        for step in range(max_step):
            rows.append(
                RealizedConsumptionRow(
                    customer_id=customer.index,
                    timestamp=None,
                    step=step,
                    realized_consumption=max(0.0, customer.forecast[step]),
                    inventory_observed=None,
                    delivered_quantity=delivered.get((customer.index, step)),
                    source="solution_week_instance_consumption",
                )
            )
    return tuple(rows)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".json":
        with path.open() as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return [dict(row) for row in payload]
        if isinstance(payload, dict) and isinstance(payload.get("history"), list):
            return [dict(row) for row in payload["history"]]
    if suffix == ".jsonl":
        with path.open() as handle:
            return [json.loads(line) for line in handle if line.strip()]
    raise ValueError(f"Unsupported history input format: {path.suffix}")


def _delivered_by_customer_step(
    instance: Instance,
    solution: Solution,
    max_step: int,
) -> dict[tuple[int, int], float]:
    delivered: dict[tuple[int, int], float] = {}
    customer_ids = {customer.index for customer in instance.customers if not customer.call_in}
    for shift in solution.shifts:
        for operation in shift.operations:
            if operation.point not in customer_ids or operation.quantity <= 0.0:
                continue
            step = operation.arrival // instance.unit
            if 0 <= step < max_step:
                key = (operation.point, step)
                delivered[key] = delivered.get(key, 0.0) + operation.quantity
    return delivered


def _normalize_row(row: dict[str, Any], csv_line: int) -> RealizedConsumptionRow:
    customer_id = _row_customer_id(row)
    if customer_id is None:
        raise ValueError(f"Missing customer id on row {csv_line}")
    consumption = _row_value(
        row,
        (
            "realized_consumption",
            "consumption",
            "actual_consumption",
            "demand",
            "used_quantity",
            "estimated_consumption",
        ),
    )
    if consumption is None:
        raise ValueError(f"Missing realized consumption on row {csv_line}")
    if consumption < 0.0:
        raise ValueError(f"Negative realized consumption on row {csv_line}")
    step = _row_int(row, ("step", "time_step", "horizon_step", "t"))
    timestamp = _row_text(row, ("timestamp", "datetime", "date"))
    if step is None and not timestamp:
        raise ValueError(f"Row {csv_line} needs either step or timestamp")
    inventory = _row_value(row, ("inventory_observed", "inventory", "tank_reading", "observed_inventory"))
    delivered = _row_value(row, ("delivered_quantity", "delivered", "delivery_quantity"))
    if inventory is not None and inventory < 0.0:
        raise ValueError(f"Negative observed inventory on row {csv_line}")
    if delivered is not None and delivered < 0.0:
        raise ValueError(f"Negative delivered quantity on row {csv_line}")
    return RealizedConsumptionRow(
        customer_id=customer_id,
        timestamp=timestamp,
        step=step,
        realized_consumption=consumption,
        inventory_observed=inventory,
        delivered_quantity=delivered,
        source=_row_text(row, ("source", "origin", "method")) or "",
    )


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
    match = re.search(r"(\d+)$", str(value).strip())
    return int(match.group(1)) if match else None


def _row_int(row: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    value = _row_value(row, keys)
    return int(value) if value is not None else None


def _row_value(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key not in row:
            continue
        value = row[key]
        if value == "" or value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(parsed):
            return parsed
    return None


def _row_text(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in row and row[key] not in ("", None):
            return str(row[key])
    return None
