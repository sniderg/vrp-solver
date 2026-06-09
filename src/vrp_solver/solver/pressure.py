from __future__ import annotations

from dataclasses import dataclass

from ..inventory import tank_events
from ..model import Instance, Solution


MINUTES_PER_DAY = 1440
EPSILON = 1e-6


@dataclass(frozen=True)
class PressurePoint:
    customer: int
    first_minute: int
    deficit_area: float
    safety_steps: int
    negative_steps: int
    overfill_steps: int
    source_lead_minutes: int = 0
    cluster_accessibility: float = 0.0


def pressure_points(
    instance: Instance,
    solution: Solution,
    *,
    end_day: int,
    include_overfill: bool = True,
) -> tuple[PressurePoint, ...]:
    cutoff_minute = end_day * MINUTES_PER_DAY
    by_customer: dict[int, dict[str, float]] = {}
    source_lead = {
        customer.index: min(instance.time_matrix[source.index][customer.index] for source in instance.sources)
        if instance.sources
        else instance.time_matrix[instance.base_index][customer.index]
        for customer in instance.customers
    }
    for event in tank_events(instance, solution):
        if event.time_start >= cutoff_minute:
            continue
        if event.point not in instance.customer_by_point:
            continue
        customer = instance.customer_by_point[event.point]
        if customer.call_in:
            continue
        deficit = max(0.0, event.safety_level - event.ending_inventory)
        overfill = max(0.0, event.ending_inventory - event.capacity) if include_overfill else 0.0
        if deficit <= EPSILON and overfill <= EPSILON:
            continue
        data = by_customer.setdefault(
            event.point,
            {
                "first": float(event.time_start),
                "area": 0.0,
                "safety": 0.0,
                "negative": 0.0,
                "overfill": 0.0,
            },
        )
        data["first"] = min(data["first"], float(event.time_start))
        data["area"] += deficit * instance.unit + overfill * instance.unit
        data["safety"] += 1.0 if deficit > EPSILON else 0.0
        data["negative"] += 1.0 if event.ending_inventory < -EPSILON else 0.0
        data["overfill"] += 1.0 if overfill > EPSILON else 0.0

    return tuple(
        PressurePoint(
            customer=customer,
            first_minute=int(data["first"]),
            deficit_area=data["area"],
            safety_steps=int(data["safety"]),
            negative_steps=int(data["negative"]),
            overfill_steps=int(data["overfill"]),
            source_lead_minutes=int(source_lead.get(customer, 0)),
            cluster_accessibility=_cluster_accessibility(instance, customer),
        )
        for customer, data in sorted(
            by_customer.items(),
            key=lambda item: (
                item[1]["first"],
                -item[1]["negative"],
                -item[1]["area"],
                source_lead.get(item[0], 0),
                item[0],
            ),
        )
    )


def pressure_customers(
    instance: Instance,
    solution: Solution,
    *,
    end_day: int,
    limit: int,
) -> tuple[int, ...]:
    return tuple(point.customer for point in pressure_points(instance, solution, end_day=end_day)[:limit])


def _cluster_accessibility(instance: Instance, customer: int) -> float:
    distances = sorted(
        instance.time_matrix[customer][other.index]
        for other in instance.customers
        if other.index != customer and not other.call_in
    )
    if not distances:
        return 0.0
    nearest = distances[: min(5, len(distances))]
    return sum(nearest) / len(nearest)
