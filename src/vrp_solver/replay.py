from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .inventory import tank_events
from .model import Instance, Operation, Shift, Solution
from .rules import derive_solution


@dataclass(frozen=True)
class Segment:
    shift: int
    driver: int
    trailer: int
    kind: str
    start: int
    end: int
    origin: int | None
    destination: int | None
    operation_index: int | None
    point: int | None
    quantity: float
    trailer_quantity_start: float
    trailer_quantity_end: float
    driving_start: int
    driving_end: int


@dataclass(frozen=True)
class ResourceState:
    time: int
    shift: int | None
    driver: int | None
    trailer: int | None
    kind: str
    point: int | None
    origin: int | None
    destination: int | None
    fraction_complete: float | None
    remaining_minutes: int | None
    trailer_quantity: float | None
    driving_since_layover: int | None
    message: str


@dataclass(frozen=True)
class CustomerState:
    time: int
    step: int
    point: int
    inventory: float
    capacity: float
    safety_level: float
    margin_to_safety: float
    margin_to_capacity: float
    delivered_this_step: float
    consumed_this_step: float
    call_in: bool


def build_segments(instance: Instance, solution: Solution) -> list[Segment]:
    derived_by_shift = {derived.shift.index: derived for derived in derive_solution(instance, solution)}
    segments: list[Segment] = []

    for shift in solution.shifts:
        derived = derived_by_shift[shift.index]
        driver = instance.drivers[shift.driver]
        previous_point = instance.base_index
        previous_departure = shift.start
        previous_load = derived.start_trailer_quantity
        previous_driving = 0

        for op_index, operation in enumerate(shift.operations):
            derived_op = derived.operations[op_index]
            travel_time = instance.time_matrix[previous_point][operation.point]
            travel_start = previous_departure
            layover_duration = driver.layover_duration if derived_op.layover_before else 0

            if layover_duration:
                layover_end = travel_start + layover_duration
                segments.append(
                    Segment(
                        shift=shift.index,
                        driver=shift.driver,
                        trailer=shift.trailer,
                        kind="layover",
                        start=travel_start,
                        end=layover_end,
                        origin=previous_point,
                        destination=None,
                        operation_index=op_index,
                        point=previous_point,
                        quantity=0.0,
                        trailer_quantity_start=previous_load,
                        trailer_quantity_end=previous_load,
                        driving_start=previous_driving,
                        driving_end=0,
                    )
                )
                travel_start = layover_end
                previous_driving = 0

            travel_end = travel_start + travel_time
            if travel_time:
                segments.append(
                    Segment(
                        shift=shift.index,
                        driver=shift.driver,
                        trailer=shift.trailer,
                        kind="travel",
                        start=travel_start,
                        end=travel_end,
                        origin=previous_point,
                        destination=operation.point,
                        operation_index=op_index,
                        point=None,
                        quantity=0.0,
                        trailer_quantity_start=previous_load,
                        trailer_quantity_end=previous_load,
                        driving_start=previous_driving,
                        driving_end=derived_op.driving_since_layover,
                    )
                )

            if operation.arrival > travel_end:
                segments.append(
                    Segment(
                        shift=shift.index,
                        driver=shift.driver,
                        trailer=shift.trailer,
                        kind="wait",
                        start=travel_end,
                        end=operation.arrival,
                        origin=None,
                        destination=None,
                        operation_index=op_index,
                        point=operation.point,
                        quantity=0.0,
                        trailer_quantity_start=previous_load,
                        trailer_quantity_end=previous_load,
                        driving_start=derived_op.driving_since_layover,
                        driving_end=derived_op.driving_since_layover,
                    )
                )

            service_end = derived_op.departure
            if service_end > operation.arrival:
                segments.append(
                    Segment(
                        shift=shift.index,
                        driver=shift.driver,
                        trailer=shift.trailer,
                        kind="service",
                        start=operation.arrival,
                        end=service_end,
                        origin=None,
                        destination=None,
                        operation_index=op_index,
                        point=operation.point,
                        quantity=operation.quantity,
                        trailer_quantity_start=previous_load,
                        trailer_quantity_end=derived_op.trailer_quantity,
                        driving_start=derived_op.driving_since_layover,
                        driving_end=derived_op.driving_since_layover,
                    )
                )

            previous_point = operation.point
            previous_departure = service_end
            previous_load = derived_op.trailer_quantity
            previous_driving = derived_op.driving_since_layover

        if shift.operations:
            return_time = instance.time_matrix[previous_point][instance.base_index]
            return_start = previous_departure
            if derived.end > return_start + return_time:
                layover_end = return_start + driver.layover_duration
                segments.append(
                    Segment(
                        shift=shift.index,
                        driver=shift.driver,
                        trailer=shift.trailer,
                        kind="layover",
                        start=return_start,
                        end=layover_end,
                        origin=previous_point,
                        destination=None,
                        operation_index=None,
                        point=previous_point,
                        quantity=0.0,
                        trailer_quantity_start=previous_load,
                        trailer_quantity_end=previous_load,
                        driving_start=previous_driving,
                        driving_end=0,
                    )
                )
                return_start = layover_end
                previous_driving = 0

            segments.append(
                Segment(
                    shift=shift.index,
                    driver=shift.driver,
                    trailer=shift.trailer,
                    kind="return",
                    start=return_start,
                    end=derived.end,
                    origin=previous_point,
                    destination=instance.base_index,
                    operation_index=None,
                    point=None,
                    quantity=0.0,
                    trailer_quantity_start=previous_load,
                    trailer_quantity_end=previous_load,
                    driving_start=previous_driving,
                    driving_end=previous_driving + return_time,
                )
            )

    segments.sort(key=lambda segment: (segment.start, segment.shift, segment.operation_index or -1))
    return segments


def resource_states_at(
    instance: Instance,
    solution: Solution,
    time: int,
) -> list[ResourceState]:
    segments = build_segments(instance, solution)
    return resource_states_from_segments(instance, segments, time)


def resource_states_from_segments(
    instance: Instance,
    segments: list[Segment],
    time: int,
) -> list[ResourceState]:
    active = [segment for segment in segments if segment.start <= time < segment.end]
    states: list[ResourceState] = []

    for segment in active:
        duration = max(1, segment.end - segment.start)
        elapsed = time - segment.start
        fraction = elapsed / duration if segment.kind in {"travel", "return"} else None
        remaining = segment.end - time
        trailer_quantity = _interpolate(
            segment.trailer_quantity_start,
            segment.trailer_quantity_end,
            elapsed / duration,
        )
        driving = _interpolate(segment.driving_start, segment.driving_end, elapsed / duration)
        states.append(
            ResourceState(
                time=time,
                shift=segment.shift,
                driver=segment.driver,
                trailer=segment.trailer,
                kind=segment.kind,
                point=segment.point,
                origin=segment.origin,
                destination=segment.destination,
                fraction_complete=fraction,
                remaining_minutes=remaining,
                trailer_quantity=trailer_quantity,
                driving_since_layover=int(round(driving)),
                message=_segment_message(segment, fraction, remaining),
            )
        )

    busy_drivers = {state.driver for state in states}
    busy_trailers = {state.trailer for state in states}
    for driver in instance.drivers:
        if driver.index not in busy_drivers:
            states.append(
                ResourceState(
                    time=time,
                    shift=None,
                    driver=driver.index,
                    trailer=None,
                    kind="off_shift",
                    point=instance.base_index,
                    origin=None,
                    destination=None,
                    fraction_complete=None,
                    remaining_minutes=None,
                    trailer_quantity=None,
                    driving_since_layover=0,
                    message="driver not active in a shift",
                )
            )
    for trailer in instance.trailers:
        if trailer.index not in busy_trailers:
            states.append(
                ResourceState(
                    time=time,
                    shift=None,
                    driver=None,
                    trailer=trailer.index,
                    kind="idle_trailer",
                    point=instance.base_index,
                    origin=None,
                    destination=None,
                    fraction_complete=None,
                    remaining_minutes=None,
                    trailer_quantity=trailer.initial_quantity,
                    driving_since_layover=None,
                    message="trailer not active in a shift",
                )
            )

    return sorted(states, key=lambda state: (state.kind, state.driver or -1, state.trailer or -1))


def customer_states_at(
    instance: Instance,
    solution: Solution,
    time: int,
) -> list[CustomerState]:
    events = tank_events(instance, solution)
    return customer_states_from_events(instance, events, time)


def customer_states_from_events(
    instance: Instance,
    events: list,
    time: int,
) -> list[CustomerState]:
    step = min(max(time // instance.unit, 0), instance.horizon - 1)
    events_by_point_step = {
        (event.point, event.step): event
        for event in events
    }
    states = []
    for customer in instance.customers:
        event = events_by_point_step[(customer.index, step)]
        states.append(
            CustomerState(
                time=time,
                step=step,
                point=customer.index,
                inventory=event.ending_inventory,
                capacity=customer.capacity,
                safety_level=customer.safety_level,
                margin_to_safety=event.ending_inventory - customer.safety_level,
                margin_to_capacity=customer.capacity - event.ending_inventory,
                delivered_this_step=event.delivered,
                consumed_this_step=event.consumed,
                call_in=customer.call_in,
            )
        )
    return states


def status_overview(
    instance: Instance,
    solution: Solution,
    *,
    time: int,
    limit: int = 10,
) -> dict[str, object]:
    segments = build_segments(instance, solution)
    events = tank_events(instance, solution)
    resources = resource_states_from_segments(instance, segments, time)
    customers = customer_states_from_events(instance, events, time)
    active_resources = [
        state for state in resources
        if state.kind not in {"off_shift", "idle_trailer"}
    ]
    vmi_customers = [state for state in customers if not state.call_in]
    low_inventory = sorted(vmi_customers, key=lambda state: state.margin_to_safety)[:limit]
    near_capacity = sorted(vmi_customers, key=lambda state: state.margin_to_capacity)[:limit]
    delivered_now = [
        state for state in customers
        if abs(state.delivered_this_step) > 1e-9
    ]

    return {
        "time": time,
        "step": min(max(time // instance.unit, 0), instance.horizon - 1),
        "active_resources": active_resources,
        "low_inventory_customers": low_inventory,
        "near_capacity_customers": near_capacity,
        "delivered_this_step": delivered_now,
    }


def replay_grid(
    instance: Instance,
    solution: Solution,
    *,
    start: int = 0,
    end: int | None = None,
    step: int = 60,
) -> tuple[list[ResourceState], list[CustomerState]]:
    if end is None:
        end = instance.horizon * instance.unit
    segments = build_segments(instance, solution)
    events = tank_events(instance, solution)
    resource_rows = []
    customer_rows = []
    for time in range(start, end + 1, step):
        resource_rows.extend(resource_states_from_segments(instance, segments, time))
        customer_rows.extend(customer_states_from_events(instance, events, time))
    return resource_rows, customer_rows


def _interpolate(start: float, end: float, fraction: float) -> float:
    return start + (end - start) * max(0.0, min(1.0, fraction))


def _segment_message(segment: Segment, fraction: float | None, remaining: int) -> str:
    if segment.kind in {"travel", "return"}:
        return (
            f"{segment.kind} {segment.origin}->{segment.destination}; "
            f"{fraction or 0:.3f} complete; {remaining} min remaining"
        )
    if segment.kind == "service":
        return f"service at point {segment.point}; quantity {segment.quantity:.6f}"
    if segment.kind == "wait":
        return f"waiting at point {segment.point}; {remaining} min remaining"
    if segment.kind == "layover":
        return f"layover at/after point {segment.point}; {remaining} min remaining"
    return segment.kind
