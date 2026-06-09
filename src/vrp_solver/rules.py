from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .inventory import tank_violations
from .model import Instance, Operation, Shift, Solution


EPSILON = 1e-6


@dataclass(frozen=True)
class DerivedOperation:
    point: int
    arrival: int
    departure: int
    quantity: float
    travel_time_from_previous: int
    driving_since_layover: int
    driving_before_layover: int
    layover_before: bool
    trailer_quantity: float


@dataclass(frozen=True)
class DerivedShift:
    shift: Shift
    operations: tuple[DerivedOperation, ...]
    end: int
    start_trailer_quantity: float
    end_trailer_quantity: float
    layovers: int


@dataclass(frozen=True)
class RuleViolation:
    code: str
    severity: str
    shift: int | None
    operation: int | None
    point: int | None
    message: str


def derive_solution(instance: Instance, solution: Solution) -> list[DerivedShift]:
    trailer_quantities = {
        trailer.index: trailer.initial_quantity
        for trailer in instance.trailers
    }
    derived_by_index: dict[int, DerivedShift] = {}

    for shift in sorted(solution.shifts, key=lambda item: (item.start, item.index)):
        driver = instance.drivers[shift.driver]
        start_quantity = trailer_quantities[shift.trailer]
        quantity = start_quantity
        last_point = instance.base_index
        last_departure = shift.start
        cumulated_driving_time = 0
        layovers = 0
        derived_operations = []

        for operation in shift.operations:
            travel_time = instance.time_matrix[last_point][operation.point]
            setup_time = instance.setup_time_for_point(operation.point)
            layover_before = (
                operation.arrival - last_departure
                >= driver.layover_duration + travel_time
            )
            if layover_before:
                layovers += 1
                driving_before_layover = min(
                    max(0, driver.max_driving_duration - cumulated_driving_time),
                    travel_time,
                )
                cumulated_driving_time = travel_time - driving_before_layover
            else:
                driving_before_layover = 0
                cumulated_driving_time += travel_time

            quantity -= operation.quantity
            derived_operations.append(
                DerivedOperation(
                    point=operation.point,
                    arrival=operation.arrival,
                    departure=operation.arrival + setup_time,
                    quantity=operation.quantity,
                    travel_time_from_previous=travel_time,
                    driving_since_layover=cumulated_driving_time,
                    driving_before_layover=driving_before_layover,
                    layover_before=layover_before,
                    trailer_quantity=quantity,
                )
            )
            last_point = operation.point
            last_departure = operation.arrival + setup_time

        if derived_operations:
            return_time = instance.time_matrix[last_point][instance.base_index]
            has_layover = any(operation.layover_before for operation in derived_operations)
            if cumulated_driving_time + return_time > driver.max_driving_duration and not has_layover:
                layovers += 1
                end = last_departure + return_time + driver.layover_duration
            else:
                end = last_departure + return_time
        else:
            end = shift.start

        trailer_quantities[shift.trailer] = quantity
        derived_by_index[shift.index] = DerivedShift(
            shift=shift,
            operations=tuple(derived_operations),
            end=end,
            start_trailer_quantity=start_quantity,
            end_trailer_quantity=quantity,
            layovers=layovers,
        )

    return [derived_by_index[shift.index] for shift in solution.shifts]


def validate_solution(instance: Instance, solution: Solution) -> list[RuleViolation]:
    violations: list[RuleViolation] = []
    derived = derive_solution(instance, solution)
    violations.extend(_validate_shift_references(instance, solution))
    violations.extend(_validate_shift_operations(instance, derived))
    violations.extend(_validate_resource_constraints(instance, derived))
    violations.extend(_validate_service_quality(instance, solution))
    violations.extend(_validate_tank_bounds(instance, solution))
    return violations


def is_time_window_valid(arrival: int, departure: int, windows: tuple[TimeWindow, ...]) -> bool:
    """Check if a service interval is within any of the provided time windows."""
    return any(window.start <= arrival and departure <= window.end for window in windows)


def is_trailer_allowed(instance: Instance, point: int, trailer: int) -> bool:
    """Check if a trailer is allowed at a given point (source or customer)."""
    if point in instance.source_by_point:
        return trailer in instance.source_by_point[point].allowed_trailers
    if point in instance.customer_by_point:
        return trailer in instance.customer_by_point[point].allowed_trailers
    return True


def is_driving_duration_valid(driver: Driver, duration: int) -> bool:
    """Check if a driving duration is within the driver's maximum allowed."""
    return duration <= driver.max_driving_duration


def _violation(
    code: str,
    message: str,
    *,
    shift: int | None = None,
    operation: int | None = None,
    point: int | None = None,
    severity: str = "error",
) -> RuleViolation:
    return RuleViolation(
        code=code,
        severity=severity,
        shift=shift,
        operation=operation,
        point=point,
        message=message,
    )


def _validate_shift_references(instance: Instance, solution: Solution) -> list[RuleViolation]:
    violations: list[RuleViolation] = []
    driver_ids = {driver.index for driver in instance.drivers}
    trailer_ids = {trailer.index for trailer in instance.trailers}

    for shift in solution.shifts:
        if shift.driver not in driver_ids:
            violations.append(
                _violation(
                    "REF_DRIVER",
                    f"driver {shift.driver} is not in instance drivers",
                    shift=shift.index,
                )
            )
        if shift.trailer not in trailer_ids:
            violations.append(
                _violation(
                    "REF_TRAILER",
                    f"trailer {shift.trailer} is not in instance trailers",
                    shift=shift.index,
                )
            )
    return violations


def _validate_shift_operations(
    instance: Instance,
    derived_shifts: list[DerivedShift],
) -> list[RuleViolation]:
    violations: list[RuleViolation] = []
    latest_point = len(instance.time_matrix) - 1

    for derived in derived_shifts:
        shift = derived.shift
        driver = instance.drivers[shift.driver]
        trailer = instance.trailers[shift.trailer]
        has_layover_customer = any(
            operation.point in instance.customer_by_point
            and instance.customer_by_point[operation.point].layover_customer
            for operation in shift.operations
        )

        if derived.layovers > 0 and not has_layover_customer:
            violations.append(
                _violation(
                    "LAY02",
                    "shift has a layover but no layover customer",
                    shift=shift.index,
                )
            )
        if derived.layovers > 1:
            violations.append(
                _violation("LAY03", "shift has more than one layover", shift=shift.index)
            )

        previous_departure = shift.start
        previous_point = instance.base_index
        previous_derived_driving = 0
        for op_index, operation in enumerate(shift.operations):
            if operation.point < 0 or operation.point > latest_point:
                violations.append(
                    _violation(
                        "SHI03",
                        f"point {operation.point} is outside valid point range",
                        shift=shift.index,
                        operation=op_index,
                        point=operation.point,
                    )
                )
                continue

            derived_op = derived.operations[op_index]
            required_arrival = (
                previous_departure
                + instance.time_matrix[previous_point][operation.point]
                + (driver.layover_duration if derived_op.layover_before else 0)
            )
            if operation.arrival + EPSILON < required_arrival:
                violations.append(
                    _violation(
                        "SHI02",
                        f"arrival {operation.arrival} is before required {required_arrival}",
                        shift=shift.index,
                        operation=op_index,
                        point=operation.point,
                    )
                )

            point_kind = instance.point_kind(operation.point)
            if point_kind == "customer":
                customer = instance.customer_by_point[operation.point]
                if not is_time_window_valid(
                    derived_op.arrival,
                    derived_op.departure,
                    customer.time_windows,
                ):
                    violations.append(
                        _violation(
                            "SHI04",
                            f"service [{derived_op.arrival}, {derived_op.departure}] is outside customer time windows",
                            shift=shift.index,
                            operation=op_index,
                            point=operation.point,
                        )
                    )
                if shift.trailer not in customer.allowed_trailers:
                    violations.append(
                        _violation(
                            "SHI05",
                            f"trailer {shift.trailer} is not allowed at customer",
                            shift=shift.index,
                            operation=op_index,
                            point=operation.point,
                        )
                    )
                if operation.quantity < -EPSILON:
                    violations.append(
                        _violation(
                            "SHI11",
                            "customer delivery quantity is negative",
                            shift=shift.index,
                            operation=op_index,
                            point=operation.point,
                        )
                    )
                if not customer.call_in:
                    if operation.quantity - customer.capacity > EPSILON:
                        violations.append(
                            _violation(
                                "SHI16",
                                f"delivery {operation.quantity} exceeds customer capacity {customer.capacity}",
                                shift=shift.index,
                                operation=op_index,
                                point=operation.point,
                            )
                        )
                    if operation.quantity + EPSILON < customer.min_operation_quantity:
                        violations.append(
                            _violation(
                                "SHI16",
                                f"delivery {operation.quantity} is below minimum {customer.min_operation_quantity}",
                                shift=shift.index,
                                operation=op_index,
                                point=operation.point,
                            )
                        )
                if customer.call_in and customer.orders:
                    if not any(
                        order.earliest_time <= operation.arrival <= order.latest_time
                        for order in customer.orders
                    ):
                        violations.append(
                            _violation(
                                "QS03",
                                "call-in delivery is outside all order windows",
                                shift=shift.index,
                                operation=op_index,
                                point=operation.point,
                            )
                        )
            elif point_kind == "source":
                source = instance.source_by_point[operation.point]
                if shift.trailer not in source.allowed_trailers:
                    violations.append(
                        _violation(
                            "SHI05",
                            f"trailer {shift.trailer} is not allowed at source",
                            shift=shift.index,
                            operation=op_index,
                            point=operation.point,
                        )
                    )
                if operation.quantity > EPSILON:
                    violations.append(
                        _violation(
                            "SHI11",
                            "source loading quantity is positive",
                            shift=shift.index,
                            operation=op_index,
                            point=operation.point,
                        )
                    )

            if derived_op.trailer_quantity < -EPSILON:
                violations.append(
                    _violation(
                        "SHI06",
                        f"trailer quantity {derived_op.trailer_quantity} is negative",
                        shift=shift.index,
                        operation=op_index,
                        point=operation.point,
                    )
                )
            if derived_op.trailer_quantity - trailer.capacity > EPSILON:
                violations.append(
                    _violation(
                        "SHI06",
                        f"trailer quantity {derived_op.trailer_quantity} exceeds capacity {trailer.capacity}",
                        shift=shift.index,
                        operation=op_index,
                        point=operation.point,
                    )
                )
            if derived_op.layover_before:
                driving_check_value = previous_derived_driving + derived_op.driving_before_layover
            else:
                driving_check_value = derived_op.driving_since_layover
            if driving_check_value - driver.max_driving_duration > EPSILON:
                violations.append(
                    _violation(
                        "DRI03",
                        f"driving between layovers {driving_check_value} exceeds max {driver.max_driving_duration}",
                        shift=shift.index,
                        operation=op_index,
                        point=operation.point,
                    )
                )

            previous_departure = derived_op.departure
            previous_point = operation.point
            previous_derived_driving = derived_op.driving_since_layover

    return violations


def _validate_resource_constraints(
    instance: Instance,
    derived_shifts: list[DerivedShift],
) -> list[RuleViolation]:
    violations: list[RuleViolation] = []
    by_driver: dict[int, list[DerivedShift]] = defaultdict(list)
    by_trailer: dict[int, list[DerivedShift]] = defaultdict(list)

    for derived in derived_shifts:
        shift = derived.shift
        driver = instance.drivers[shift.driver]
        if shift.trailer not in driver.trailer_ids:
            violations.append(
                _violation(
                    "TL03",
                    f"trailer {shift.trailer} is not allowed for driver {shift.driver}",
                    shift=shift.index,
                )
            )
        if not is_time_window_valid(shift.start, derived.end, driver.time_windows):
            violations.append(
                _violation(
                    "DRI08",
                    f"shift interval [{shift.start}, {derived.end}] is outside driver time windows",
                    shift=shift.index,
                )
            )
        by_driver[shift.driver].append(derived)
        by_trailer[shift.trailer].append(derived)

    for driver_id, shifts in by_driver.items():
        driver = instance.drivers[driver_id]
        ordered = sorted(shifts, key=lambda item: (item.shift.start, item.shift.index))
        for previous, current in zip(ordered, ordered[1:]):
            required = previous.end + driver.min_inter_shift_duration
            if current.shift.start < required:
                violations.append(
                    _violation(
                        "DRI01",
                        f"driver {driver_id} starts shift at {current.shift.start} before required {required}",
                        shift=current.shift.index,
                    )
                )

    for trailer_id, shifts in by_trailer.items():
        ordered = sorted(shifts, key=lambda item: (item.shift.start, item.shift.index))
        for previous, current in zip(ordered, ordered[1:]):
            if current.shift.start < previous.end:
                violations.append(
                    _violation(
                        "TL01",
                        f"trailer {trailer_id} starts shift at {current.shift.start} before previous end {previous.end}",
                        shift=current.shift.index,
                    )
                )

    return violations


def _validate_service_quality(instance: Instance, solution: Solution) -> list[RuleViolation]:
    violations: list[RuleViolation] = []
    delivered_by_order: dict[tuple[int, int], float] = defaultdict(float)
    for shift in solution.shifts:
        for operation in shift.operations:
            customer = instance.customer_by_point.get(operation.point)
            if customer is None or not customer.call_in:
                continue
            for order_index, order in enumerate(customer.orders):
                if order.earliest_time <= operation.arrival <= order.latest_time:
                    delivered_by_order[(customer.index, order_index)] += operation.quantity

    latest_required = instance.horizon * instance.unit
    for customer in instance.customers:
        if not customer.call_in:
            continue
        for order_index, order in enumerate(customer.orders):
            if order.latest_time > latest_required:
                continue
            delivered = delivered_by_order[(customer.index, order_index)]
            if delivered + EPSILON < order.min_quantity_to_satisfy:
                violations.append(
                    _violation(
                        "QS01",
                        f"order {order_index} delivered {delivered}, below flexible minimum {order.min_quantity_to_satisfy}",
                        point=customer.index,
                    )
                )
            elif delivered + EPSILON < order.quantity:
                violations.append(
                    _violation(
                        "QS01",
                        f"order {order_index} delivered {delivered}, below nominal {order.quantity} but above flexible minimum",
                        point=customer.index,
                        severity="warning",
                    )
                )

    return violations


def _validate_tank_bounds(instance: Instance, solution: Solution) -> list[RuleViolation]:
    violations: list[RuleViolation] = []
    for violation in tank_violations(instance, solution):
        code = {
            "TANK_OVERFILL": "DYN01",
            "TANK_NEGATIVE": "DYN01",
            "TANK_SAFETY_BREACH": "QS02",
        }[violation.code]
        violations.append(
            _violation(
                code,
                f"{violation.code}: {violation.message} at step {violation.step}",
                point=violation.point,
            )
        )
    return violations


# Atomic checks exposed for solver use
