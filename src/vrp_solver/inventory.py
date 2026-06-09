from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from .model import Customer, Instance, Solution


EPSILON = 1e-6


@dataclass(frozen=True)
class TankEvent:
    point: int
    step: int
    time_start: int
    initial_inventory: float
    delivered: float
    consumed: float
    after_consumption: float
    after_delivery: float
    ending_inventory: float
    capacity: float
    safety_level: float
    overfilled_after_delivery: bool
    overfilled_ending: bool
    negative: bool
    safety_breach: bool


@dataclass(frozen=True)
class TankViolation:
    code: str
    point: int
    step: int
    time_start: int
    inventory: float
    limit: float
    message: str


def delivery_by_customer_step(solution: Solution) -> dict[int, dict[int, float]]:
    deliveries: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for shift in solution.shifts:
        for operation in shift.operations:
            if operation.quantity <= 0:
                continue
            deliveries[operation.point][operation.arrival] += operation.quantity
    return {
        point: dict(events)
        for point, events in deliveries.items()
    }


def tank_events(instance: Instance, solution: Solution) -> list[TankEvent]:
    deliveries_by_arrival = delivery_by_customer_step(solution)
    events: list[TankEvent] = []

    for customer in instance.customers:
        events.extend(
            project_customer_inventory(
                instance,
                customer,
                deliveries_by_arrival.get(customer.index, {}),
            )
        )

    return events


try:
    from .inventory_fast import project_inventory_core
    _HAS_FAST_CORE = True
except ImportError:
    _HAS_FAST_CORE = False

def project_customer_inventory(
    instance: Instance,
    customer: Customer,
    deliveries: dict[int, float],
) -> list[TankEvent]:
    """Calculate tank inventory events for a single customer over the horizon."""
    
    if _HAS_FAST_CORE and customer.forecast is not None:
        # Fast Cython Path
        deliveries_by_step = np.zeros(instance.horizon, dtype=np.float64)
        for arrival, quantity in deliveries.items():
            step = min(max(arrival // instance.unit, 0), instance.horizon - 1)
            deliveries_by_step[step] += quantity
            
        forecast_arr = np.array(customer.forecast, dtype=np.float64)
        
        inventory_arr, breach_arr = project_inventory_core(
            customer.initial_tank_quantity,
            forecast_arr,
            deliveries_by_step,
            customer.capacity,
            customer.safety_level,
            instance.horizon,
            customer.index,
            instance.unit,
            1 if customer.call_in else 0
        )
        
        # Convert back to TankEvent objects (this is the remaining bottleneck, 
        # but we can optimize this later if needed)
        events = []
        for step in range(instance.horizon):
            ending = inventory_arr[step]
            events.append(
                TankEvent(
                    point=customer.index,
                    step=step,
                    time_start=step * instance.unit,
                    initial_inventory=inventory_arr[step-1] if step > 0 else customer.initial_tank_quantity,
                    delivered=deliveries_by_step[step],
                    consumed=forecast_arr[step],
                    after_consumption=inventory_arr[step] - deliveries_by_step[step],
                    after_delivery=ending,
                    ending_inventory=ending,
                    capacity=customer.capacity,
                    safety_level=customer.safety_level,
                    overfilled_after_delivery=(not customer.call_in and ending > customer.capacity + EPSILON),
                    overfilled_ending=(not customer.call_in and ending > customer.capacity + EPSILON),
                    negative=(not customer.call_in and ending < -EPSILON),
                    safety_breach=bool(breach_arr[step])
                )
            )
        return events

    # Legacy Python Path
    inventory = customer.initial_tank_quantity
    deliveries_by_step_dict: dict[int, float] = defaultdict(float)

    for arrival, quantity in deliveries.items():
        step = min(max(arrival // instance.unit, 0), instance.horizon - 1)
        deliveries_by_step_dict[step] += quantity

    events: list[TankEvent] = []
    for step in range(instance.horizon):
        initial = inventory
        delivered = deliveries_by_step_dict.get(step, 0.0)
        consumed = customer.forecast[step] if customer.forecast else 0.0
        after_consumption = initial - consumed
        ending = after_consumption + delivered
        events.append(
            TankEvent(
                point=customer.index,
                step=step,
                time_start=step * instance.unit,
                initial_inventory=initial,
                delivered=delivered,
                consumed=consumed,
                after_consumption=after_consumption,
                after_delivery=ending,
                ending_inventory=ending,
                capacity=customer.capacity,
                safety_level=customer.safety_level,
                overfilled_after_delivery=(
                    not customer.call_in
                    and ending > customer.capacity + EPSILON
                ),
                overfilled_ending=(
                    not customer.call_in
                    and ending > customer.capacity + EPSILON
                ),
                negative=(not customer.call_in and ending < -EPSILON),
                safety_breach=(
                    not customer.call_in
                    and ending < customer.safety_level - EPSILON
                ),
            )
        )
        inventory = ending

    return events


def tank_violations(instance: Instance, solution: Solution) -> list[TankViolation]:
    violations: list[TankViolation] = []

    for event in tank_events(instance, solution):
        if event.overfilled_after_delivery:
            violations.append(
                TankViolation(
                    code="TANK_OVERFILL",
                    point=event.point,
                    step=event.step,
                    time_start=event.time_start,
                    inventory=event.ending_inventory,
                    limit=event.capacity,
                    message=(
                        f"ending inventory {event.ending_inventory:.6f} "
                        f"exceeds capacity {event.capacity:.6f}"
                    ),
                )
            )
        if event.negative:
            violations.append(
                TankViolation(
                    code="TANK_NEGATIVE",
                    point=event.point,
                    step=event.step,
                    time_start=event.time_start,
                    inventory=event.ending_inventory,
                    limit=0.0,
                    message=(
                        f"ending inventory {event.ending_inventory:.6f} "
                        "is below zero after consumption"
                    ),
                )
            )
        if event.safety_breach:
            violations.append(
                TankViolation(
                    code="TANK_SAFETY_BREACH",
                    point=event.point,
                    step=event.step,
                    time_start=event.time_start,
                    inventory=event.ending_inventory,
                    limit=event.safety_level,
                    message=(
                        f"ending inventory {event.ending_inventory:.6f} "
                        f"is below safety level {event.safety_level:.6f}"
                    ),
                )
            )

    return violations


def days_of_inventory(
    instance: Instance,
    customer: Customer,
    current_inventory: float,
    start_step: int,
    lead_time_minutes: float = 0.0,
) -> float:
    """Calculate how many days of autonomy are left before a safety breach,
    minus the logistical lead time (travel time from source).
    """
    if customer.call_in:
        return 999.0
    
    inventory = current_inventory
    steps_autonomy = 0
    for step in range(start_step, instance.horizon):
        forecast = customer.forecast[step] if step < len(customer.forecast) else 0.0
        inventory -= forecast
        if inventory < customer.safety_level - EPSILON:
            break
        steps_autonomy += 1
    
    inventory_autonomy_minutes = steps_autonomy * instance.unit
    logistical_autonomy_minutes = inventory_autonomy_minutes - lead_time_minutes
    
    return logistical_autonomy_minutes / 1440.0
