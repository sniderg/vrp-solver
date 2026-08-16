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
    hard_steps: int = 0
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
        # Soft terminal pressure: in the final 3 days of the evaluation horizon,
        # flag tanks running close to safety (< safety + 25% usable capacity) so
        # candidate generation keeps proposing replenishment shifts right up to the boundary.
        near_horizon = (event.time_start >= cutoff_minute - 3 * MINUTES_PER_DAY)
        terminal_urgency = (
            max(0.0, (event.safety_level + 0.25 * (customer.capacity - event.safety_level)) - event.ending_inventory)
            if near_horizon
            else 0.0
        )
        if deficit <= EPSILON and overfill <= EPSILON and terminal_urgency <= EPSILON:
            continue
        data = by_customer.setdefault(
            event.point,
            {
                "first": float(event.time_start),
                "area": 0.0,
                "safety": 0.0,
                "negative": 0.0,
                "overfill": 0.0,
                "hard": 0.0,
                "hard_first": float("inf"),
            },
        )
        data["first"] = min(data["first"], float(event.time_start))
        data["area"] += (deficit + terminal_urgency) * instance.unit + overfill * instance.unit
        data["safety"] += 1.0 if (deficit > EPSILON or terminal_urgency > EPSILON) else 0.0
        data["negative"] += 1.0 if event.ending_inventory < -EPSILON else 0.0
        data["overfill"] += 1.0 if overfill > EPSILON else 0.0
        data["hard"] += 1.0 if (deficit > EPSILON or overfill > EPSILON) else 0.0
        if deficit > EPSILON or overfill > EPSILON:
            data["hard_first"] = min(data["hard_first"], float(event.time_start))


    # A call-in has no inventory trajectory, so it was previously invisible
    # to every destroy/repair operator.  Treat an unmet flexible minimum as a
    # deadline-pressure point.  The deadline (rather than its opening) is the
    # relevant first minute: it is the last instant at which a route can still
    # repair the violation.
    delivered_by_order: dict[tuple[int, int], float] = {}
    for shift in solution.shifts:
        for operation in shift.operations:
            if operation.quantity <= EPSILON:
                continue
            customer = instance.customer_by_point.get(operation.point)
            if customer is None or not customer.call_in:
                continue
            for order_index, order in enumerate(customer.orders):
                if order.earliest_time <= operation.arrival <= order.latest_time:
                    key = (customer.index, order_index)
                    delivered_by_order[key] = delivered_by_order.get(key, 0.0) + operation.quantity
    for customer in instance.customers:
        if not customer.call_in:
            continue
        for order_index, order in enumerate(customer.orders):
            if order.latest_time >= cutoff_minute:
                continue
            missing = order.min_quantity_to_satisfy - delivered_by_order.get((customer.index, order_index), 0.0)
            if missing <= EPSILON:
                continue
            data = by_customer.setdefault(
                customer.index,
                {"first": float(order.latest_time), "area": 0.0, "safety": 0.0, "negative": 0.0, "overfill": 0.0, "hard": 0.0, "hard_first": float("inf")},
            )
            data["first"] = min(data["first"], float(order.latest_time))
            data["area"] += missing
            data["safety"] += 1.0
            data["hard"] += 1.0
            data["hard_first"] = min(data["hard_first"], float(order.latest_time))

    return tuple(
        PressurePoint(
            customer=customer,
            first_minute=int(
                data["hard_first"] if data["hard"] > 0 else data["first"]
            ),
            deficit_area=data["area"],
            safety_steps=int(data["safety"]),
            negative_steps=int(data["negative"]),
            overfill_steps=int(data["overfill"]),
            hard_steps=int(data["hard"]),
            source_lead_minutes=int(source_lead.get(customer, 0)),
            cluster_accessibility=_cluster_accessibility(instance, customer),
        )
        for customer, data in sorted(
            by_customer.items(),
            key=lambda item: (
                0 if item[1]["hard"] > 0 else 1,
                item[1]["hard_first"] if item[1]["hard"] > 0 else item[1]["first"],
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
