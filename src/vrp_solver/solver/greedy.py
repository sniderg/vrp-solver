from __future__ import annotations

from dataclasses import dataclass, replace

from ..inventory import project_customer_inventory, tank_events
from ..model import Customer, Driver, Instance, Operation, Shift, Solution, TimeWindow
from ..rules import is_driving_duration_valid, is_time_window_valid, is_trailer_allowed


EPSILON = 1e-6


@dataclass(frozen=True)
class ConstructionReport:
    shifts: int
    operations: int
    delivered_quantity: float
    unscheduled_customers: tuple[int, ...]
    exhausted_resources: bool


@dataclass
class _ResourceState:
    driver: int
    trailer: int
    next_window_index: int = 0
    available_time: int = 0
    trailer_quantity: float = 0.0


@dataclass(frozen=True)
class _Candidate:
    customer: Customer
    breach_step: int
    arrival: int
    departure: int
    quantity: float
    travel_time: int
    source_arrival: int | None
    load_quantity: float


def construct_solution(
    instance: Instance,
    *,
    safety_buffer: float = 0.20,
    max_shifts: int | None = None,
) -> tuple[Solution, ConstructionReport]:
    """Build a simple route-construction VMI solution.

    This is intentionally conservative, but a shift may contain multiple
    source/customer cycles. It is a seed constructor, not a competitive solver.
    """

    resources = _initial_resources(instance)
    scheduled: dict[int, dict[int, float]] = {customer.index: {} for customer in instance.customers}
    ignore_before_step: dict[int, int] = {customer.index: 0 for customer in instance.customers}
    shifts: list[Shift] = []
    exhausted_resources = False

    while True:
        if max_shifts is not None and len(shifts) >= max_shifts:
            exhausted_resources = _next_need(instance, scheduled, ignore_before_step) is not None
            break
        if _next_need(instance, scheduled, ignore_before_step) is None:
            break

        next_window = _next_resource_window(instance, resources)
        if next_window is None:
            exhausted_resources = True
            break
        _, resource_index, window_index, window = next_window

        resource = resources[resource_index]
        shift = _build_shift_for_window(
            instance,
            resource,
            window,
            len(shifts),
            scheduled,
            ignore_before_step,
            safety_buffer,
        )
        resource.next_window_index = window_index + 1
        if shift is None:
            continue

        shifts.append(shift)

    unscheduled = tuple(
        customer.index
        for customer in instance.customers
        if _first_breach_step(instance, customer, scheduled[customer.index], 0)
        is not None
    )
    solution = _trim_overfills(instance, Solution(shifts=tuple(shifts)))
    report = ConstructionReport(
        shifts=len(solution.shifts),
        operations=sum(len(shift.operations) for shift in solution.shifts),
        delivered_quantity=sum(
            operation.quantity
            for shift in solution.shifts
            for operation in shift.operations
            if operation.quantity > 0
        ),
        unscheduled_customers=unscheduled,
        exhausted_resources=exhausted_resources,
    )
    return solution, report


def _trim_overfills(instance: Instance, solution: Solution) -> Solution:
    shifts = [list(shift.operations) for shift in solution.shifts]
    shift_meta = list(solution.shifts)

    for _ in range(500):
        shifts = _cap_source_loads(instance, shift_meta, shifts)
        current = Solution(
            shifts=tuple(
                replace(shift, operations=tuple(operations))
                for shift, operations in zip(shift_meta, shifts)
            )
        )
        overfills = [
            event
            for event in tank_events(instance, current)
            if event.overfilled_ending
        ]
        if not overfills:
            return current

        event = overfills[0]
        overflow = event.ending_inventory - event.capacity
        if overflow <= EPSILON:
            return current
        delivery_ref = _latest_delivery_ref(instance, shifts, event.point, event.step)
        if delivery_ref is None:
            return current
        shift_index, operation_index = delivery_ref
        operation = shifts[shift_index][operation_index]
        reduction = min(operation.quantity, overflow)
        new_quantity = operation.quantity - reduction
        if new_quantity <= EPSILON:
            shifts[shift_index].pop(operation_index)
        else:
            shifts[shift_index][operation_index] = replace(operation, quantity=new_quantity)
        shifts = _cap_source_loads(instance, shift_meta, shifts)

    shifts = _cap_source_loads(instance, shift_meta, shifts)
    return Solution(
        shifts=tuple(
            replace(shift, operations=tuple(operations))
            for shift, operations in zip(shift_meta, shifts)
        )
    )


def _cap_source_loads(
    instance: Instance,
    shift_meta: list[Shift],
    shifts: list[list[Operation]],
) -> list[list[Operation]]:
    trailer_quantities = {
        trailer.index: trailer.initial_quantity
        for trailer in instance.trailers
    }
    trailer_caps = {
        trailer.index: trailer.capacity
        for trailer in instance.trailers
    }
    adjusted: list[list[Operation]] = []
    for shift, operations in sorted(
        zip(shift_meta, shifts),
        key=lambda item: (item[0].start, item[0].index),
    ):
        quantity = trailer_quantities[shift.trailer]
        capacity = trailer_caps[shift.trailer]
        adjusted_operations: list[Operation] = []
        for operation in operations:
            if operation.point in instance.source_by_point and operation.quantity < -EPSILON:
                load = min(-operation.quantity, max(0.0, capacity - quantity))
                if load > EPSILON:
                    adjusted_operations.append(replace(operation, quantity=-load))
                    quantity += load
                continue
            adjusted_operations.append(operation)
            quantity -= operation.quantity
        trailer_quantities[shift.trailer] = quantity
        adjusted.append(adjusted_operations)

    by_index = {
        shift.index: operations
        for shift, operations in zip(
            sorted(shift_meta, key=lambda item: (item.start, item.index)),
            adjusted,
        )
    }
    return [by_index[shift.index] for shift in shift_meta]


def _latest_delivery_ref(
    instance: Instance,
    shifts: list[list[Operation]],
    point: int,
    step: int,
) -> tuple[int, int] | None:
    latest: tuple[int, int, int] | None = None
    for shift_index, operations in enumerate(shifts):
        for operation_index, operation in enumerate(operations):
            if operation.point != point or operation.quantity <= EPSILON:
                continue
            operation_step = min(max(operation.arrival // instance.unit, 0), instance.horizon - 1)
            if operation_step > step:
                continue
            candidate = (operation.arrival, shift_index, operation_index)
            if latest is None or candidate > latest:
                latest = candidate
    if latest is None:
        return None
    _, shift_index, operation_index = latest
    return shift_index, operation_index


def _next_resource_window(
    instance: Instance,
    resources: list[_ResourceState],
) -> tuple[int, int, int, TimeWindow] | None:
    candidates = []
    for resource_index, resource in enumerate(resources):
        driver = instance.drivers[resource.driver]
        for window_index in range(resource.next_window_index, len(driver.time_windows)):
            window = driver.time_windows[window_index]
            start = max(window.start, resource.available_time)
            if start <= window.end:
                candidates.append((start, resource_index, window_index, window))
                break
            resource.next_window_index = window_index + 1
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1]))


def _build_shift_for_window(
    instance: Instance,
    resource: _ResourceState,
    window: TimeWindow,
    shift_index: int,
    scheduled: dict[int, dict[int, float]],
    ignore_before_step: dict[int, int],
    safety_buffer: float,
) -> Shift | None:
    driver = instance.drivers[resource.driver]
    start = max(window.start, resource.available_time)
    if start >= window.end:
        return None

    operations: list[Operation] = []
    current_point = instance.base_index
    current_time = start
    driving_since_rest = 0
    end_after_return = start

    while True:
        candidate = _best_route_candidate(
            instance,
            resource,
            window,
            current_point,
            current_time,
            driving_since_rest,
            scheduled,
            ignore_before_step,
            safety_buffer,
        )
        if candidate is None:
            break

        if candidate.source_arrival is not None and candidate.load_quantity > EPSILON:
            operations.append(
                Operation(
                    point=instance.sources[0].index,
                    arrival=candidate.source_arrival,
                    quantity=-candidate.load_quantity,
                )
            )
            resource.trailer_quantity += candidate.load_quantity

        operations.append(
            Operation(
                point=candidate.customer.index,
                arrival=candidate.arrival,
                quantity=candidate.quantity,
            )
        )
        resource.trailer_quantity -= candidate.quantity
        scheduled[candidate.customer.index][candidate.arrival] = (
            scheduled[candidate.customer.index].get(candidate.arrival, 0.0)
            + candidate.quantity
        )
        arrival_step = min(max(candidate.arrival // instance.unit, 0), instance.horizon - 1)
        ignore_before_step[candidate.customer.index] = max(
            ignore_before_step[candidate.customer.index],
            arrival_step + 1,
        )

        current_point = candidate.customer.index
        current_time = candidate.departure
        driving_since_rest += candidate.travel_time
        end_after_return = (
            current_time + instance.time_matrix[current_point][instance.base_index]
        )

    if not operations:
        return None

    resource.available_time = end_after_return + driver.min_inter_shift_duration
    return Shift(
        index=shift_index,
        driver=resource.driver,
        trailer=resource.trailer,
        start=start,
        operations=tuple(operations),
    )


def _best_route_candidate(
    instance: Instance,
    resource: _ResourceState,
    window: TimeWindow,
    current_point: int,
    current_time: int,
    driving_since_rest: int,
    scheduled: dict[int, dict[int, float]],
    ignore_before_step: dict[int, int],
    safety_buffer: float,
) -> _Candidate | None:
    candidates = []
    for customer in instance.customers:
        if customer.call_in or not is_trailer_allowed(instance, customer.index, resource.trailer):
            continue

        breach = _first_breach_step(
            instance,
            customer,
            scheduled[customer.index],
            ignore_before_step[customer.index],
        )
        if breach is None:
            continue

        candidate = _candidate_for_customer(
            instance,
            resource,
            window,
            current_point,
            current_time,
            driving_since_rest,
            customer,
            breach,
            scheduled[customer.index],
            safety_buffer,
        )
        if candidate is None:
            continue

        deadline = (breach + 1) * instance.unit - 1
        lateness = max(0, candidate.arrival - deadline)
        candidates.append((lateness, breach, candidate.arrival, candidate.travel_time, customer.index, candidate))

    if not candidates:
        return None
    return min(candidates)[-1]


def _candidate_for_customer(
    instance: Instance,
    resource: _ResourceState,
    window: TimeWindow,
    current_point: int,
    current_time: int,
    driving_since_rest: int,
    customer: Customer,
    breach_step: int,
    deliveries: dict[int, float],
    safety_buffer: float,
) -> _Candidate | None:
    source = instance.sources[0]
    trailer = instance.trailers[resource.trailer]
    load_quantity = 0.0
    source_arrival = None
    time = current_time
    point = current_point
    travel_time = 0
    trailer_quantity = resource.trailer_quantity

    if trailer_quantity < min(trailer.capacity, customer.capacity) - EPSILON:
        source_arrival = time + instance.time_matrix[point][source.index]
        time = source_arrival + source.setup_time
        travel_time += instance.time_matrix[point][source.index]
        point = source.index
        load_quantity = trailer.capacity - trailer_quantity
        trailer_quantity = trailer.capacity

    arrival = time + instance.time_matrix[point][customer.index]
    departure = arrival + customer.setup_time
    travel_time += instance.time_matrix[point][customer.index]
    return_travel = instance.time_matrix[customer.index][instance.base_index]
    if departure + return_travel > window.end:
        return None
    if not is_driving_duration_valid(
        instance.drivers[resource.driver],
        driving_since_rest + travel_time + return_travel,
    ):
        return None
    if not is_time_window_valid(arrival, departure, customer.time_windows):
        return None

    arrival_step = min(max(arrival // instance.unit, 0), instance.horizon - 1)
    events = project_customer_inventory(instance, customer, deliveries)
    after_consumption = events[arrival_step].after_consumption
    already_delivered = 0.0
    for arr, qty in deliveries.items():
        if min(max(arr // instance.unit, 0), instance.horizon - 1) == arrival_step:
            already_delivered += qty

    target_inventory = _target_inventory(instance, customer, safety_buffer)
    quantity = min(
        trailer_quantity,
        customer.capacity,
        target_inventory - after_consumption - already_delivered,
    )
    quantity = max(0.0, quantity)
    if quantity <= EPSILON or quantity + EPSILON < customer.min_operation_quantity:
        return None

    return _Candidate(
        customer=customer,
        breach_step=breach_step,
        arrival=arrival,
        departure=departure,
        quantity=quantity,
        travel_time=travel_time,
        source_arrival=source_arrival,
        load_quantity=load_quantity,
    )


def _target_inventory(instance: Instance, customer: Customer, safety_buffer: float) -> float:
    if safety_buffer <= 0:
        return customer.capacity
    daily_demand = sum(customer.forecast) / max(instance.horizon / 24.0, 1.0)
    return min(customer.capacity, customer.capacity - safety_buffer * daily_demand)


def _initial_resources(instance: Instance) -> list[_ResourceState]:
    resources: list[_ResourceState] = []
    trailer_by_id = {trailer.index: trailer for trailer in instance.trailers}
    used_trailers: set[int] = set()
    for driver in instance.drivers:
        trailer_id = next((item for item in driver.trailer_ids if item not in used_trailers), None)
        if trailer_id is None:
            continue
        used_trailers.add(trailer_id)
        resources.append(
            _ResourceState(
                driver=driver.index,
                trailer=trailer_id,
                trailer_quantity=trailer_by_id[trailer_id].initial_quantity,
            )
        )
    return resources


def _next_need(
    instance: Instance,
    scheduled: dict[int, dict[int, float]],
    ignore_before_step: dict[int, int],
) -> tuple[Customer, int] | None:
    needs = []
    for customer in instance.customers:
        if customer.call_in:
            continue
        breach = _first_breach_step(
            instance,
            customer,
            scheduled[customer.index],
            ignore_before_step[customer.index],
        )
        if breach is not None:
            needs.append((breach, customer.index, customer))
    if not needs:
        return None
    breach, _, customer = min(needs)
    return customer, breach


def _first_breach_step(
    instance: Instance,
    customer: Customer,
    deliveries: dict[int, float],
    min_step: int = 0,
) -> int | None:
    events = project_customer_inventory(instance, customer, deliveries)
    for event in events:
        if event.step >= min_step and event.safety_breach:
            return event.step
    return None
