from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from .model import Instance, Shift, Solution


@dataclass(frozen=True)
class ShiftSummary:
    index: int
    driver: int
    trailer: int
    start: int
    first_arrival: int | None
    end_time: int
    operations: int
    distance: float
    travel_time: int
    service_time: int
    delivered_quantity: float
    loaded_quantity: float
    distance_cost: float
    time_cost: float
    layover_cost: float = 0.0
    estimated_cost: float = 0.0
    min_load: float = 0.0
    max_load: float = 0.0
    final_load: float = 0.0
    load_violations: int = 0


@dataclass(frozen=True)
class CustomerInventorySummary:
    point: int
    deliveries: int
    delivered_quantity: float
    min_inventory: float
    max_inventory: float
    min_margin_to_safety: float
    first_dry_step: int | None
    first_overfill_step: int | None
    first_safety_breach_step: int | None
    final_inventory: float


def summarize_shift(
    instance: Instance,
    shift: Shift,
    initial_load: float | None = None,
) -> ShiftSummary:
    trailer = instance.trailers[shift.trailer]
    driver = instance.drivers[shift.driver]

    # Start quantity and load tracking
    start_quantity = trailer.initial_quantity if initial_load is None else initial_load
    quantity = start_quantity
    last_point = instance.base_index
    last_departure = shift.start
    cumulated_driving_time = 0
    layovers = 0

    distance = 0.0
    travel_time = 0
    service_time = 0
    delivered_quantity = 0.0
    loaded_quantity = 0.0

    loads = [quantity]
    load_violations = 0
    if quantity < -1e-6 or quantity - trailer.capacity > 1e-6:
        load_violations += 1

    for operation in shift.operations:
        leg_time = instance.time_matrix[last_point][operation.point]
        leg_distance = instance.distance_matrix[last_point][operation.point]

        distance += leg_distance
        travel_time += leg_time
        setup_time = instance.setup_time_for_point(operation.point)
        service_time += setup_time

        layover_before = (
            operation.arrival - last_departure
            >= driver.layover_duration + leg_time
        )
        if layover_before:
            layovers += 1
            driving_before_layover = min(
                max(0, driver.max_driving_duration - cumulated_driving_time),
                leg_time,
            )
            cumulated_driving_time = leg_time - driving_before_layover
        else:
            cumulated_driving_time += leg_time

        quantity -= operation.quantity
        loads.append(quantity)
        if quantity < -1e-6 or quantity - trailer.capacity > 1e-6:
            load_violations += 1

        if operation.quantity > 0:
            delivered_quantity += operation.quantity
        elif operation.quantity < 0:
            loaded_quantity += -operation.quantity

        last_point = operation.point
        last_departure = operation.arrival + setup_time

    if shift.operations:
        return_time = instance.time_matrix[last_point][instance.base_index]
        return_dist = instance.distance_matrix[last_point][instance.base_index]
        distance += return_dist
        travel_time += return_time

        # Check return layover
        has_layover = layovers > 0
        if cumulated_driving_time + return_time > driver.max_driving_duration and not has_layover:
            layovers += 1
            end_time = last_departure + return_time + driver.layover_duration
        else:
            end_time = last_departure + return_time
    else:
        end_time = shift.start

    distance_cost = distance * trailer.distance_cost
    working_time = end_time - shift.start - layovers * driver.layover_duration
    time_cost = working_time * driver.time_cost
    layover_cost = layovers * driver.layover_cost

    min_load = min(loads)
    max_load = max(loads)

    return ShiftSummary(
        index=shift.index,
        driver=shift.driver,
        trailer=shift.trailer,
        start=shift.start,
        first_arrival=shift.operations[0].arrival if shift.operations else None,
        end_time=end_time,
        operations=len(shift.operations),
        distance=distance,
        travel_time=travel_time,
        service_time=service_time,
        delivered_quantity=delivered_quantity,
        loaded_quantity=loaded_quantity,
        distance_cost=distance_cost,
        time_cost=time_cost,
        layover_cost=layover_cost,
        estimated_cost=distance_cost + time_cost + layover_cost,
        min_load=min_load,
        max_load=max_load,
        final_load=quantity,
        load_violations=load_violations,
    )


def summarize_solution(instance: Instance, solution: Solution) -> list[ShiftSummary]:
    from .rules import derive_solution
    derived_shifts = derive_solution(instance, solution)

    summaries = []
    for derived in derived_shifts:
        shift = derived.shift
        driver = instance.drivers[shift.driver]
        trailer = instance.trailers[shift.trailer]

        # 1. Distance & travel_time & service_time
        distance = 0.0
        travel_time = 0
        service_time = 0
        prev_point = instance.base_index
        for op in shift.operations:
            distance += instance.distance_matrix[prev_point][op.point]
            travel_time += instance.time_matrix[prev_point][op.point]
            service_time += instance.setup_time_for_point(op.point)
            prev_point = op.point
        distance += instance.distance_matrix[prev_point][instance.base_index]
        travel_time += instance.time_matrix[prev_point][instance.base_index]

        # 2. Quantities
        delivered_quantity = sum(op.quantity for op in shift.operations if op.quantity > 0)
        loaded_quantity = sum(-op.quantity for op in shift.operations if op.quantity < 0)

        # 3. Costs
        distance_cost = distance * trailer.distance_cost
        working_time = derived.end - shift.start - derived.layovers * driver.layover_duration
        time_cost = working_time * driver.time_cost
        layover_cost = derived.layovers * driver.layover_cost

        # 4. Loads
        loads = [derived.start_trailer_quantity]
        current_load = derived.start_trailer_quantity
        load_violations = 0
        if current_load < -1e-6 or current_load - trailer.capacity > 1e-6:
            load_violations += 1

        for op in derived.operations:
            current_load = op.trailer_quantity
            loads.append(current_load)
            if current_load < -1e-6 or current_load - trailer.capacity > 1e-6:
                load_violations += 1

        min_load = min(loads)
        max_load = max(loads)

        summary = ShiftSummary(
            index=shift.index,
            driver=shift.driver,
            trailer=shift.trailer,
            start=shift.start,
            first_arrival=shift.operations[0].arrival if shift.operations else None,
            end_time=derived.end,
            operations=len(shift.operations),
            distance=distance,
            travel_time=travel_time,
            service_time=service_time,
            delivered_quantity=delivered_quantity,
            loaded_quantity=loaded_quantity,
            distance_cost=distance_cost,
            time_cost=time_cost,
            layover_cost=layover_cost,
            estimated_cost=distance_cost + time_cost + layover_cost,
            min_load=min_load,
            max_load=max_load,
            final_load=derived.end_trailer_quantity,
            load_violations=load_violations,
        )
        summaries.append(summary)

    return summaries


def delivery_events(solution: Solution) -> dict[int, list[tuple[int, float]]]:
    events: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for shift in solution.shifts:
        for operation in shift.operations:
            if operation.quantity > 0:
                events[operation.point].append((operation.arrival, operation.quantity))

    for point_events in events.values():
        point_events.sort()
    return dict(events)


def customer_inventory_summary(
    instance: Instance,
    solution: Solution | None = None,
) -> list[CustomerInventorySummary]:
    events = delivery_events(solution) if solution is not None else {}
    summaries: list[CustomerInventorySummary] = []

    for customer in instance.customers:
        inventory = customer.initial_tank_quantity
        min_inventory = inventory
        max_inventory = inventory
        first_dry_step: int | None = None
        first_overfill_step: int | None = None
        first_safety_breach_step: int | None = None
        delivered_quantity = 0.0
        deliveries = 0
        events_by_step: dict[int, float] = defaultdict(float)

        for arrival, quantity in events.get(customer.index, []):
            step = min(max(arrival // instance.unit, 0), instance.horizon - 1)
            events_by_step[step] += quantity

        for step in range(instance.horizon):
            if step in events_by_step:
                inventory += events_by_step[step]
                delivered_quantity += events_by_step[step]
                deliveries += 1
                max_inventory = max(max_inventory, inventory)
                if (
                    first_overfill_step is None
                    and inventory > customer.capacity + 1e-6
                    and not customer.call_in
                ):
                    first_overfill_step = step
            if customer.forecast:
                inventory -= customer.forecast[step]
            min_inventory = min(min_inventory, inventory)
            max_inventory = max(max_inventory, inventory)
            if first_dry_step is None and inventory < -1e-6:
                first_dry_step = step
            if (
                first_safety_breach_step is None
                and inventory < customer.safety_level - 1e-6
                and not customer.call_in
            ):
                first_safety_breach_step = step

        summaries.append(
            CustomerInventorySummary(
                point=customer.index,
                deliveries=deliveries,
                delivered_quantity=delivered_quantity,
                min_inventory=min_inventory,
                max_inventory=max_inventory,
                min_margin_to_safety=min_inventory - customer.safety_level,
                first_dry_step=first_dry_step,
                first_overfill_step=first_overfill_step,
                first_safety_breach_step=first_safety_breach_step,
                final_inventory=inventory,
            )
        )

    return summaries


def point_visit_counts(solution: Solution) -> Counter[int]:
    counter: Counter[int] = Counter()
    for shift in solution.shifts:
        for operation in shift.operations:
            counter[operation.point] += 1
    return counter
