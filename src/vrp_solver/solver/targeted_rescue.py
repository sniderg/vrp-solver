from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

from ..inventory import tank_events
from ..model import Instance, Operation, Shift, Solution
from ..rules import derive_solution, is_time_window_valid, is_trailer_allowed
from ..highs_repair import repair_quantities_with_highs
from ..highs_time_opt import optimize_solution_times
from ..route_cache import RouteCache
from .highs_selector import select_shifts_with_highs


MINUTES_PER_DAY = 1440
EPSILON = 1e-6


@lru_cache(maxsize=100000)
def _derive_single_shift(instance: Instance, shift: Shift):
    return derive_solution(instance, Solution(shifts=(shift,)))[0]


def _window_arrival_samples(
    min_arrival: int,
    max_arrival: int,
    count: int,
) -> list[int]:
    """Generate `count` evenly-spaced arrival times within [min_arrival, max_arrival].

    Always includes the endpoints. When count == 1 returns just max_arrival.
    This restores the iteration-to-iteration diversity that lets the CG loop
    grow its column pool across iterations (as opposed to hardcoded min/max/mid
    which produces identical candidates on every iteration).
    """
    if count <= 1:
        return [max_arrival]
    span = max_arrival - min_arrival
    if span <= 0:
        return [min_arrival]
    return sorted(
        {
            min_arrival + round(span * i / (count - 1))
            for i in range(count)
        }
    )



@dataclass(frozen=True)
class RescueConfig:
    start_day: int = 0
    end_day: int = 14
    replace_from_day: int = 7
    max_customers: int = 12
    samples_per_customer: int = 6
    target_fill_ratio: float = 0.95
    max_pre_service_fill_ratio: float = 0.95
    sample_lookback_days: int = 5
    max_chain_length: int = 3
    nearest_chain_neighbors: int = 4
    repair_quantities: bool = True
    variable_quantity_columns: bool = False
    pressure_pricing: bool = True
    normalize_source_loads: bool = True
    quantity_objective: str = "max-delivered"


@dataclass(frozen=True)
class RescueReport:
    failing_customers: tuple[int, ...]
    generated_candidates: int
    selected_extra_shifts: int
    quantity_repair_status: str | None = None
    quantity_repair_constraints: int | None = None


def targeted_rescue(
    instance: Instance,
    baseline: Solution,
    *,
    config: RescueConfig = RescueConfig(),
) -> tuple[Solution, RescueReport]:
    failing = _failing_customers(instance, baseline, config)
    fixed_prefix = _keep_shifts_started_before(
        baseline,
        config.replace_from_day * MINUTES_PER_DAY,
    )
    candidates = _baseline_window_shifts(baseline, config)
    candidates.extend(generate_rescue_candidates(instance, fixed_prefix, failing, config=config))
    candidates.extend(generate_carryover_rescue_candidates(instance, fixed_prefix, failing, config=config))
    candidates.extend(generate_chain_rescue_candidates(instance, fixed_prefix, failing, config=config))
    candidates.extend(generate_multi_reload_candidates(instance, fixed_prefix, failing, config=config))
    candidates = _dedupe_reindex(candidates)
    if not candidates:
        return baseline, RescueReport(tuple(failing), 0, 0)

    rescued = select_shifts_with_highs(
        instance,
        fixed_prefix,
        candidates,
        start_day=config.replace_from_day,
        end_day=config.end_day,
        variable_quantities=config.variable_quantity_columns,
        pressure_pricing=config.pressure_pricing,
        baseline=baseline,
    )
    # Restore future shifts from the baseline that start after the end day of the rescue window
    future_shifts = [
        shift for shift in baseline.shifts
        if shift.start >= config.end_day * MINUTES_PER_DAY
    ]
    all_shifts = list(rescued.shifts) + future_shifts
    rescued = Solution(
        shifts=tuple(replace(shift, index=i) for i, shift in enumerate(all_shifts))
    )
    rescued = optimize_solution_times(instance, rescued)

    if config.normalize_source_loads:
        rescued = normalize_source_loads(instance, rescued)
    selected_extra = max(0, len(rescued.shifts) - len(fixed_prefix.shifts) - len(future_shifts))
    repair_status = None
    repair_constraints = None
    if config.repair_quantities:
        full_horizon_days = (instance.horizon * instance.unit) // MINUTES_PER_DAY
        rescued, repair_report = repair_quantities_with_highs(
            instance,
            rescued,
            score_days=full_horizon_days,
            feasibility_days=full_horizon_days,
            quantity_objective=config.quantity_objective,
            baseline=baseline,
            fixed_prefix_minutes=config.replace_from_day * MINUTES_PER_DAY,
        )
        if config.normalize_source_loads:
            rescued = normalize_source_loads(instance, rescued)
        repair_status = repair_report.status
        repair_constraints = repair_report.constraints

    return rescued, RescueReport(
        tuple(failing),
        len(candidates),
        selected_extra,
        repair_status,
        repair_constraints,
    )


def normalize_source_loads(instance: Instance, solution: Solution) -> Solution:
    """Make source load quantities consistent with selected trailer histories.

    Candidate generation estimates source quantities against a fixed baseline.
    Once several candidates are selected together, the real trailer state may be
    different. This pass keeps the selected route/timing and delivery quantities,
    then turns each source operation into a fill-to-capacity operation under the
    actual selected trailer history.
    """
    trailer_quantities = {
        trailer.index: trailer.initial_quantity
        for trailer in instance.trailers
    }
    trailer_capacities = {
        trailer.index: trailer.capacity
        for trailer in instance.trailers
    }
    normalized_by_index: dict[int, Shift] = {}

    for shift in sorted(solution.shifts, key=lambda item: (item.start, item.index)):
        trailer_quantity = trailer_quantities[shift.trailer]
        trailer_capacity = trailer_capacities[shift.trailer]
        operations: list[Operation] = []

        for operation in shift.operations:
            if operation.point in instance.source_by_point:
                load = max(0.0, trailer_capacity - trailer_quantity)
                operations.append(replace(operation, quantity=-load))
                trailer_quantity += load
                continue

            operations.append(operation)
            trailer_quantity -= operation.quantity

        trailer_quantities[shift.trailer] = trailer_quantity
        normalized_by_index[shift.index] = replace(shift, operations=tuple(operations))

    return Solution(
        shifts=tuple(normalized_by_index[shift.index] for shift in solution.shifts)
    )


def _keep_shifts_started_before(solution: Solution, cutoff_minute: int) -> Solution:
    return Solution(
        shifts=tuple(shift for shift in solution.shifts if shift.start < cutoff_minute)
    )


def _baseline_window_shifts(solution: Solution, config: RescueConfig) -> list[Shift]:
    start = config.replace_from_day * MINUTES_PER_DAY
    end = config.end_day * MINUTES_PER_DAY
    return [shift for shift in solution.shifts if start <= shift.start < end]


def _dedupe_reindex(shifts: list[Shift]) -> list[Shift]:
    seen: set[tuple[object, ...]] = set()
    unique: list[Shift] = []
    for shift in shifts:
        key = _shift_key(shift)
        if key in seen:
            continue
        seen.add(key)
        unique.append(replace(shift, index=len(unique)))
    return unique


def generate_rescue_candidates(
    instance: Instance,
    baseline: Solution,
    failing_customers: list[int],
    *,
    config: RescueConfig,
) -> list[Shift]:
    start_minute = max(config.start_day, config.replace_from_day) * MINUTES_PER_DAY
    end_minute = config.end_day * MINUTES_PER_DAY
    candidates: list[Shift] = []
    seen: set[tuple[object, ...]] = set()
    event_cache = _events_by_customer(instance, baseline)
    trailer_cache = _trailer_load_cache(instance, baseline)

    for customer_id in failing_customers:
        customer = instance.customer_by_point[customer_id]
        if customer.call_in:
            continue
        breach_minute = _first_breach_minute(instance, baseline, customer_id, event_cache)
        if breach_minute is None:
            continue
        # When extending a solution (breach at or before the new window start),
        # the customer needs service throughout the new window.
        if breach_minute <= start_minute + instance.unit:
            latest_customer_arrival = end_minute - 1
        else:
            latest_customer_arrival = min(breach_minute - instance.unit, end_minute - 1)
            if latest_customer_arrival < start_minute:
                continue

        for driver in instance.drivers:
            for trailer in instance.trailers:
                if trailer.index not in driver.trailer_ids:
                    continue
                if not is_trailer_allowed(instance, customer.index, trailer.index):
                    continue
                source = next(
                    (
                        src
                        for src in instance.sources
                        if trailer.index in src.allowed_trailers
                    ),
                    None,
                )
                if source is None:
                    continue

                route_to_customer = (
                    instance.time_matrix[instance.base_index][source.index]
                    + source.setup_time
                    + instance.time_matrix[source.index][customer.index]
                )
                return_time = instance.time_matrix[customer.index][instance.base_index]

                total_driving = (
                    instance.time_matrix[instance.base_index][source.index]
                    + instance.time_matrix[source.index][customer.index]
                    + return_time
                )
                needs_return_layover = (
                    customer.layover_customer
                    and total_driving > driver.max_driving_duration
                )
                trail_time = customer.setup_time + return_time + (
                    driver.layover_duration if needs_return_layover else 0
                )
                
                target_arrivals = []
                for window in driver.time_windows:
                    min_arrival = max(start_minute + route_to_customer, window.start + route_to_customer)
                    max_arrival = min(latest_customer_arrival, window.end - trail_time)
                    if min_arrival <= max_arrival:
                        target_arrivals.extend(
                            _window_arrival_samples(min_arrival, max_arrival, config.samples_per_customer)
                        )
                
                target_arrivals = sorted(list(set(target_arrivals)))
                
                for target_arrival in target_arrivals:
                    shift_start = target_arrival - route_to_customer
                    source_arrival = (
                        shift_start + instance.time_matrix[instance.base_index][source.index]
                    )
                    arrival = source_arrival + source.setup_time + instance.time_matrix[source.index][customer.index]
                    departure = arrival + customer.setup_time
                    total_driving = (
                        instance.time_matrix[instance.base_index][source.index]
                        + instance.time_matrix[source.index][customer.index]
                        + return_time
                    )
                    needs_return_layover = (
                        customer.layover_customer
                        and total_driving > driver.max_driving_duration
                    )
                    end = departure + return_time + (
                        driver.layover_duration if needs_return_layover else 0
                    )
                    if end > end_minute:
                        continue
                    if not is_time_window_valid(shift_start, end, driver.time_windows):
                        continue
                    if not is_time_window_valid(arrival, departure, customer.time_windows):
                        continue
                    if (
                        total_driving > driver.max_driving_duration
                        and not customer.layover_customer
                    ):
                        continue

                    inventory_at_arrival = _inventory_at_arrival(
                        instance, baseline, customer_id, arrival, event_cache
                    )
                    if (
                        inventory_at_arrival
                        > customer.capacity * config.max_pre_service_fill_ratio + EPSILON
                    ):
                        continue
                    room = max(0.0, customer.capacity - inventory_at_arrival)
                    target_room = max(
                        0.0,
                        customer.capacity * config.target_fill_ratio
                        - inventory_at_arrival,
                    )
                    trailer_load = _trailer_load_at(instance, trailer_cache, trailer.index, shift_start)
                    load_quantity = max(0.0, trailer.capacity - trailer_load)
                    available_quantity = trailer_load + load_quantity
                    quantity = min(available_quantity, room, target_room)
                    if quantity < customer.min_operation_quantity - EPSILON:
                        continue

                    operations = []
                    if load_quantity > EPSILON:
                        operations.append(Operation(source.index, source_arrival, -load_quantity))
                    operations.append(Operation(customer.index, arrival, quantity))
                    shift = Shift(
                        index=0,
                        driver=driver.index,
                        trailer=trailer.index,
                        start=shift_start,
                        operations=tuple(operations),
                    )
                    key = _shift_key(shift)
                    if key in seen:
                        continue
                    seen.add(key)
                    if not _is_shift_route_valid(instance, shift):
                        continue
                    candidates.append(replace(shift, index=len(candidates)))

    return candidates


def generate_carryover_rescue_candidates(
    instance: Instance,
    baseline: Solution,
    failing_customers: list[int],
    *,
    config: RescueConfig,
) -> list[Shift]:
    """Generate direct customer visits using load already on a trailer.

    Official B solutions frequently start shifts with customer deliveries before
    any source visit. This candidate type is required for customers that are too
    far from source for a source->customer route inside a driver window, but can
    still be reached directly from base with carried trailer load.
    """
    start_minute = max(config.start_day, config.replace_from_day) * MINUTES_PER_DAY
    end_minute = config.end_day * MINUTES_PER_DAY
    event_cache = _events_by_customer(instance, baseline)
    trailer_cache = _trailer_load_cache(instance, baseline)
    candidates: list[Shift] = []
    seen: set[tuple[object, ...]] = set()

    for customer_id in failing_customers:
        customer = instance.customer_by_point[customer_id]
        if customer.call_in:
            continue
        breach_minute = _first_breach_minute(instance, baseline, customer_id, event_cache)
        if breach_minute is None:
            continue
        latest_arrival = min(breach_minute - instance.unit, end_minute - 1)
        if latest_arrival < start_minute:
            latest_arrival = end_minute - 1

        for driver in instance.drivers:
            for trailer in instance.trailers:
                if trailer.index not in driver.trailer_ids:
                    continue
                if not is_trailer_allowed(instance, customer_id, trailer.index):
                    continue
                route_time = (
                    instance.time_matrix[instance.base_index][customer_id]
                    + customer.setup_time
                    + instance.time_matrix[customer_id][instance.base_index]
                )
                if route_time > max(window.end - window.start for window in driver.time_windows):
                    continue

                lead_time = instance.time_matrix[instance.base_index][customer_id]
                trail_time = customer.setup_time + instance.time_matrix[customer_id][instance.base_index]
                
                target_arrivals = []
                for window in driver.time_windows:
                    min_arrival = max(start_minute + lead_time, window.start + lead_time)
                    max_arrival = min(latest_arrival, window.end - trail_time)
                    if min_arrival <= max_arrival:
                        target_arrivals.extend(
                            _window_arrival_samples(min_arrival, max_arrival, config.samples_per_customer)
                        )
                
                target_arrivals = sorted(list(set(target_arrivals)))
                
                for target_arrival in target_arrivals:
                    shift_start = target_arrival - lead_time
                    arrival = shift_start + instance.time_matrix[instance.base_index][customer_id]
                    departure = arrival + customer.setup_time
                    end = departure + instance.time_matrix[customer_id][instance.base_index]
                    if end > end_minute:
                        continue
                    if not is_time_window_valid(shift_start, end, driver.time_windows):
                        continue
                    if not is_time_window_valid(arrival, departure, customer.time_windows):
                        continue
                    trailer_load = _trailer_load_at(instance, trailer_cache, trailer.index, shift_start)
                    if trailer_load < customer.min_operation_quantity - EPSILON:
                        continue
                    inventory_at_arrival = _inventory_at_arrival(
                        instance,
                        baseline,
                        customer_id,
                        arrival,
                        event_cache,
                    )
                    if inventory_at_arrival > customer.capacity * config.max_pre_service_fill_ratio + EPSILON:
                        continue
                    room = max(0.0, customer.capacity - inventory_at_arrival)
                    target_room = max(
                        0.0,
                        customer.capacity * config.target_fill_ratio - inventory_at_arrival,
                    )
                    quantity = min(trailer_load, room, target_room)
                    if quantity < customer.min_operation_quantity - EPSILON:
                        continue
                    shift = Shift(
                        index=0,
                        driver=driver.index,
                        trailer=trailer.index,
                        start=shift_start,
                        operations=(Operation(customer_id, arrival, quantity),),
                    )
                    key = _shift_key(shift)
                    if key in seen:
                        continue
                    seen.add(key)
                    if not _is_shift_route_valid(instance, shift):
                        continue
                    candidates.append(replace(shift, index=len(candidates)))

    return candidates


def generate_chain_rescue_candidates(
    instance: Instance,
    baseline: Solution,
    failing_customers: list[int],
    *,
    config: RescueConfig,
) -> list[Shift]:
    start_minute = max(config.start_day, config.replace_from_day) * MINUTES_PER_DAY
    end_minute = config.end_day * MINUTES_PER_DAY
    event_cache = _events_by_customer(instance, baseline)
    trailer_cache = _trailer_load_cache(instance, baseline)
    route_cache = RouteCache(instance)
    sequences = _chain_sequences(
        instance,
        baseline,
        failing_customers,
        config,
        event_cache,
        route_cache,
    )
    candidates: list[Shift] = []
    seen: set[tuple[object, ...]] = set()

    for sequence in sequences:
        anchor = instance.customer_by_point[sequence[0]]
        anchor_breach = _first_breach_minute(instance, baseline, anchor.index, event_cache)
        if anchor_breach is None:
            continue
        # When extending a solution (breach at or before the new window start),
        # the customer needs service throughout the new window.
        if anchor_breach <= start_minute + instance.unit:
            latest_anchor_arrival = end_minute - 1
        else:
            latest_anchor_arrival = min(anchor_breach - instance.unit, end_minute - 1)
            if latest_anchor_arrival < start_minute:
                continue

        for driver in instance.drivers:
            for trailer in instance.trailers:
                if trailer.index not in driver.trailer_ids:
                    continue
                if any(
                    not is_trailer_allowed(instance, customer_id, trailer.index)
                    for customer_id in sequence
                ):
                    continue
                source = next(
                    (
                        src
                        for src in instance.sources
                        if trailer.index in src.allowed_trailers
                    ),
                    None,
                )
                if source is None:
                    continue

                lead_to_anchor = (
                    instance.time_matrix[instance.base_index][source.index]
                    + source.setup_time
                    + instance.time_matrix[source.index][anchor.index]
                )

                # Compute minimum trail time for chain
                trail_time = anchor.setup_time
                curr = anchor.index
                for c_id in sequence[1:]:
                    cust = instance.customer_by_point[c_id]
                    trail_time += instance.time_matrix[curr][c_id] + cust.setup_time
                    curr = c_id
                trail_time += instance.time_matrix[curr][instance.base_index]
                
                target_arrivals = []
                for window in driver.time_windows:
                    min_arrival = max(start_minute + lead_to_anchor, window.start + lead_to_anchor)
                    max_arrival = min(latest_anchor_arrival, window.end - trail_time)
                    if min_arrival <= max_arrival:
                        target_arrivals.extend(
                            _window_arrival_samples(min_arrival, max_arrival, config.samples_per_customer)
                        )
                
                target_arrivals = sorted(list(set(target_arrivals)))
                
                for anchor_arrival in target_arrivals:
                    shift_start = anchor_arrival - lead_to_anchor
                    shift = _build_chain_shift(
                        instance,
                        baseline,
                        event_cache,
                        trailer_cache,
                        sequence,
                        driver.index,
                        trailer.index,
                        source.index,
                        shift_start,
                        end_minute,
                        config,
                        seen=seen,
                    )
                    if shift is None:
                        continue
                    candidates.append(replace(shift, index=len(candidates)))

    return candidates


def generate_multi_reload_candidates(
    instance: Instance,
    baseline: Solution,
    failing_customers: list[int],
    *,
    config: RescueConfig,
) -> list[Shift]:
    start_minute = max(config.start_day, config.replace_from_day) * MINUTES_PER_DAY
    end_minute = config.end_day * MINUTES_PER_DAY
    event_cache = _events_by_customer(instance, baseline)
    trailer_cache = _trailer_load_cache(instance, baseline)
    route_cache = RouteCache(instance)
    sequences = _multi_reload_sequences(
        instance,
        baseline,
        failing_customers,
        config,
        event_cache,
        route_cache,
    )
    candidates: list[Shift] = []
    seen: set[tuple[object, ...]] = set()

    for first_segment, second_segment in sequences:
        anchor = instance.customer_by_point[first_segment[0]]
        anchor_breach = _first_breach_minute(instance, baseline, anchor.index, event_cache)
        if anchor_breach is None:
            continue
        if anchor_breach <= start_minute + instance.unit:
            latest_anchor_arrival = end_minute - 1
        else:
            latest_anchor_arrival = min(anchor_breach - instance.unit, end_minute - 1)
            if latest_anchor_arrival < start_minute:
                continue

        for driver in instance.drivers:
            for trailer in instance.trailers:
                if trailer.index not in driver.trailer_ids:
                    continue
                route_customers = (*first_segment, *second_segment)
                if any(
                    not is_trailer_allowed(instance, customer_id, trailer.index)
                    for customer_id in route_customers
                ):
                    continue
                source = next(
                    (
                        src
                        for src in instance.sources
                        if trailer.index in src.allowed_trailers
                    ),
                    None,
                )
                if source is None:
                    continue
                lead_to_anchor = (
                    instance.time_matrix[instance.base_index][source.index]
                    + source.setup_time
                    + instance.time_matrix[source.index][anchor.index]
                )
                
                # Compute minimum trail time for multi-reload
                trail_time = anchor.setup_time
                curr = anchor.index
                for c_id in first_segment[1:]:
                    cust = instance.customer_by_point[c_id]
                    trail_time += instance.time_matrix[curr][c_id] + cust.setup_time
                    curr = c_id
                
                # Reload at source
                trail_time += instance.time_matrix[curr][source.index] + source.setup_time
                
                # Second segment
                curr = source.index
                for c_id in second_segment:
                    cust = instance.customer_by_point[c_id]
                    trail_time += instance.time_matrix[curr][c_id] + cust.setup_time
                    curr = c_id
                
                # Return to base
                trail_time += instance.time_matrix[curr][instance.base_index]
                
                target_arrivals = []
                for window in driver.time_windows:
                    min_arrival = max(start_minute + lead_to_anchor, window.start + lead_to_anchor)
                    max_arrival = min(latest_anchor_arrival, window.end - trail_time)
                    if min_arrival <= max_arrival:
                        target_arrivals.extend(
                            _window_arrival_samples(min_arrival, max_arrival, config.samples_per_customer)
                        )
                
                target_arrivals = sorted(list(set(target_arrivals)))
                
                for anchor_arrival in target_arrivals:
                    shift_start = anchor_arrival - lead_to_anchor
                    shift = _build_multi_reload_shift(
                        instance,
                        baseline,
                        event_cache,
                        trailer_cache,
                        first_segment,
                        second_segment,
                        driver.index,
                        trailer.index,
                        source.index,
                        shift_start,
                        end_minute,
                        config,
                        seen=seen,
                    )
                    if shift is None:
                        continue
                    candidates.append(replace(shift, index=len(candidates)))

    # Generate pure reload shifts to allow pre-loading at base
    for driver in instance.drivers:
        for trailer in instance.trailers:
            if trailer.index not in driver.trailer_ids:
                continue
            source = next(
                (src for src in instance.sources if trailer.index in src.allowed_trailers),
                None
            )
            if source is None:
                continue
            duration = (
                instance.time_matrix[instance.base_index][source.index]
                + source.setup_time
                + instance.time_matrix[source.index][instance.base_index]
            )
            if duration > 720:
                continue
            for window in driver.time_windows:
                if not (start_minute <= window.start < end_minute):
                    continue
                if window.end - window.start < duration:
                    continue
                shift_start = window.start
                source_arrival = shift_start + instance.time_matrix[instance.base_index][source.index]
                # Load max capacity
                op = Operation(
                    point=source.index,
                    arrival=source_arrival,
                    quantity=-trailer.capacity
                )
                shift = Shift(
                    index=0,
                    driver=driver.index,
                    trailer=trailer.index,
                    start=shift_start,
                    operations=(op,)
                )
                key = _shift_key(shift)
                if key not in seen:
                    seen.add(key)
                    candidates.append(replace(shift, index=len(candidates)))

    # Generate single-customer direct shifts starting from base (utilizing pre-loaded trailers)
    for c_id in failing_customers:
        # Check if point is a customer
        if c_id not in instance.customer_by_point:
            continue
        customer = instance.customer_by_point[c_id]
        anchor_breach = _first_breach_minute(instance, baseline, c_id, event_cache)
        if anchor_breach is None:
            continue
        if anchor_breach <= start_minute + instance.unit:
            latest_anchor_arrival = end_minute - 1
        else:
            latest_anchor_arrival = min(anchor_breach - instance.unit, end_minute - 1)
            if latest_anchor_arrival < start_minute:
                continue

        for driver in instance.drivers:
            for trailer in instance.trailers:
                if trailer.index not in driver.trailer_ids:
                    continue
                if not is_trailer_allowed(instance, c_id, trailer.index):
                    continue

                lead_to_anchor = instance.time_matrix[instance.base_index][c_id]
                trail_time = customer.setup_time + instance.time_matrix[c_id][instance.base_index]
                
                target_arrivals = []
                for window in driver.time_windows:
                    min_arrival = max(start_minute + lead_to_anchor, window.start + lead_to_anchor)
                    max_arrival = min(latest_anchor_arrival, window.end - trail_time)
                    if min_arrival <= max_arrival:
                        target_arrivals.extend(
                            _window_arrival_samples(min_arrival, max_arrival, config.samples_per_customer)
                        )
                
                target_arrivals = sorted(list(set(target_arrivals)))
                
                for anchor_arrival in target_arrivals:
                    shift_start = anchor_arrival - lead_to_anchor
                    
                    # Construct shift
                    op = Operation(point=c_id, arrival=anchor_arrival, quantity=0.0)
                    shift = Shift(
                        index=0,
                        driver=driver.index,
                        trailer=trailer.index,
                        start=shift_start,
                        operations=(op,)
                    )
                    key = _shift_key(shift)
                    if key in seen:
                        continue
                    seen.add(key)
                    if not _is_shift_route_valid(instance, shift):
                        continue
                    candidates.append(replace(shift, index=len(candidates)))

    return candidates


def _multi_reload_sequences(
    instance: Instance,
    baseline: Solution,
    failing_customers: list[int],
    config: RescueConfig,
    event_cache: dict[int, list],
    route_cache: RouteCache | None = None,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    sequences = _chain_sequences(
        instance,
        baseline,
        failing_customers,
        config,
        event_cache,
        route_cache,
    )
    output: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for sequence in sequences:
        if len(sequence) < 2:
            continue
        for split in range(1, len(sequence)):
            first = sequence[:split]
            second = sequence[split:]
            if not first or not second:
                continue
            key = (first, second)
            if key in seen:
                continue
            seen.add(key)
            output.append(key)
    return output


def _build_multi_reload_shift(
    instance: Instance,
    baseline: Solution,
    event_cache: dict[int, list],
    trailer_cache: dict[int, list[tuple[int, float]]],
    first_segment: tuple[int, ...],
    second_segment: tuple[int, ...],
    driver_id: int,
    trailer_id: int,
    source_id: int,
    shift_start: int,
    end_minute: int,
    config: RescueConfig,
    seen: set[tuple[object, ...]] | None = None,
) -> Shift | None:
    trailer = instance.trailers[trailer_id]
    source = instance.source_by_point[source_id]
    source_arrival = shift_start + instance.time_matrix[instance.base_index][source_id]
    current_time = source_arrival + source.setup_time
    current_point = source_id
    trailer_load = _trailer_load_at(instance, trailer_cache, trailer_id, shift_start)
    load_quantity = max(0.0, trailer.capacity - trailer_load)
    operations: list[Operation] = []
    if load_quantity > EPSILON:
        operations.append(Operation(source_id, source_arrival, -load_quantity))
        trailer_load += load_quantity

    delivered_count = 0
    result = _append_customer_segment(
        instance,
        baseline,
        event_cache,
        first_segment,
        current_point,
        current_time,
        trailer_load,
        operations,
        end_minute,
        config,
    )
    if result is None:
        return None
    current_point, current_time, trailer_load, count = result
    delivered_count += count

    reload_arrival = current_time + instance.time_matrix[current_point][source_id]
    if reload_arrival + source.setup_time >= end_minute:
        return None
    reload_quantity = max(0.0, trailer.capacity - trailer_load)
    if reload_quantity <= EPSILON:
        return None
    operations.append(Operation(source_id, reload_arrival, -reload_quantity))
    trailer_load += reload_quantity
    current_point = source_id
    current_time = reload_arrival + source.setup_time

    result = _append_customer_segment(
        instance,
        baseline,
        event_cache,
        second_segment,
        current_point,
        current_time,
        trailer_load,
        operations,
        end_minute,
        config,
    )
    if result is None:
        return None
    current_point, current_time, trailer_load, count = result
    delivered_count += count
    if delivered_count < 2:
        return None

    shift = Shift(
        index=0,
        driver=driver_id,
        trailer=trailer_id,
        start=shift_start,
        operations=tuple(operations),
    )
    if seen is not None:
        key = _shift_key(shift)
        if key in seen:
            return None
        seen.add(key)
    if not _is_shift_route_valid(instance, shift):
        return None
    if _derive_single_shift(instance, shift).end > end_minute:
        return None
    return shift


def _append_customer_segment(
    instance: Instance,
    baseline: Solution,
    event_cache: dict[int, list],
    segment: tuple[int, ...],
    current_point: int,
    current_time: int,
    trailer_load: float,
    operations: list[Operation],
    end_minute: int,
    config: RescueConfig,
) -> tuple[int, int, float, int] | None:
    delivered_count = 0
    for customer_id in segment:
        customer = instance.customer_by_point[customer_id]
        arrival = current_time + instance.time_matrix[current_point][customer_id]
        departure = arrival + customer.setup_time
        breach_minute = _first_breach_minute(instance, baseline, customer_id, event_cache)
        if breach_minute is not None and arrival >= breach_minute:
            continue
        if departure >= end_minute:
            continue
        if not is_time_window_valid(arrival, departure, customer.time_windows):
            continue
        inventory_at_arrival = _inventory_at_arrival(
            instance,
            baseline,
            customer_id,
            arrival,
            event_cache,
        )
        if inventory_at_arrival > customer.capacity * config.max_pre_service_fill_ratio + EPSILON:
            continue
        target_room = max(
            0.0,
            customer.capacity * config.target_fill_ratio - inventory_at_arrival,
        )
        room = max(0.0, customer.capacity - inventory_at_arrival)
        quantity = min(trailer_load, room, target_room)
        if quantity < customer.min_operation_quantity - EPSILON:
            continue
        operations.append(Operation(customer_id, arrival, quantity))
        trailer_load -= quantity
        delivered_count += 1
        current_point = customer_id
        current_time = departure
    if delivered_count == 0:
        return None
    return current_point, current_time, trailer_load, delivered_count


def _chain_sequences(
    instance: Instance,
    baseline: Solution,
    failing_customers: list[int],
    config: RescueConfig,
    event_cache: dict[int, list],
    route_cache: RouteCache | None = None,
) -> list[tuple[int, ...]]:
    route_cache = route_cache or RouteCache(instance)
    failing = [
        customer_id
        for customer_id in failing_customers
        if not instance.customer_by_point[customer_id].call_in
    ]
    breach_order = {
        customer_id: _first_breach_minute(instance, baseline, customer_id, event_cache) or 10**12
        for customer_id in failing
    }
    sequences: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    def expand_chain(current_chain: tuple[int, ...]) -> None:
        if len(current_chain) >= config.max_chain_length:
            return
            
        last_customer = current_chain[-1]
        neighbors = sorted(
            (customer_id for customer_id in failing if customer_id not in current_chain),
            key=lambda customer_id: (
                route_cache.stats((customer_id,), start_point=last_customer).travel_time,
                breach_order[customer_id],
                customer_id,
            ),
        )[: config.nearest_chain_neighbors]
        
        for neighbor in neighbors:
            new_chain = current_chain + (neighbor,)
            if new_chain not in seen:
                seen.add(new_chain)
                sequences.append(new_chain)
            expand_chain(new_chain)

    for anchor in failing:
        expand_chain((anchor,))

    return sequences


def _build_chain_shift(
    instance: Instance,
    baseline: Solution,
    event_cache: dict[int, list],
    trailer_cache: dict[int, list[tuple[int, float]]],
    sequence: tuple[int, ...],
    driver_id: int,
    trailer_id: int,
    source_id: int,
    shift_start: int,
    end_minute: int,
    config: RescueConfig,
    seen: set[tuple[object, ...]] | None = None,
) -> Shift | None:
    driver = instance.drivers[driver_id]
    trailer = instance.trailers[trailer_id]
    source = instance.source_by_point[source_id]
    source_arrival = shift_start + instance.time_matrix[instance.base_index][source_id]
    current_time = source_arrival + source.setup_time
    current_point = source_id
    trailer_load = _trailer_load_at(instance, trailer_cache, trailer_id, shift_start)
    load_quantity = max(0.0, trailer.capacity - trailer_load)
    trailer_load += load_quantity
    operations: list[Operation] = []
    if load_quantity > EPSILON:
        operations.append(Operation(source_id, source_arrival, -load_quantity))

    total_driving = instance.time_matrix[instance.base_index][source_id]
    delivered_count = 0
    for customer_id in sequence:
        customer = instance.customer_by_point[customer_id]
        arrival = current_time + instance.time_matrix[current_point][customer_id]
        departure = arrival + customer.setup_time
        breach_minute = _first_breach_minute(instance, baseline, customer_id, event_cache)
        if breach_minute is not None and arrival >= breach_minute:
            continue
        if departure >= end_minute:
            continue
        if not is_time_window_valid(arrival, departure, customer.time_windows):
            continue
        inventory_at_arrival = _inventory_at_arrival(
            instance, baseline, customer_id, arrival, event_cache
        )
        if inventory_at_arrival > customer.capacity * config.max_pre_service_fill_ratio + EPSILON:
            continue
        room = max(0.0, customer.capacity - inventory_at_arrival)
        target_room = max(
            0.0,
            customer.capacity * config.target_fill_ratio
            - inventory_at_arrival,
        )
        quantity = min(trailer_load, room, target_room)
        if quantity < customer.min_operation_quantity - EPSILON:
            continue
        operations.append(Operation(customer_id, arrival, quantity))
        delivered_count += 1
        trailer_load -= quantity
        total_driving += instance.time_matrix[current_point][customer_id]
        current_time = departure
        current_point = customer_id

    if delivered_count < 2:
        return None
    total_driving += instance.time_matrix[current_point][instance.base_index]
    has_layover_customer = any(
        instance.customer_by_point[customer_id].layover_customer
        for customer_id in sequence
    )
    needs_return_layover = (
        has_layover_customer
        and total_driving > driver.max_driving_duration
    )
    end = (
        current_time
        + instance.time_matrix[current_point][instance.base_index]
        + (driver.layover_duration if needs_return_layover else 0)
    )
    if end > end_minute:
        return None
    if total_driving > driver.max_driving_duration and not has_layover_customer:
        return None
    if not is_time_window_valid(shift_start, end, driver.time_windows):
        return None

    shift = Shift(
        index=0,
        driver=driver_id,
        trailer=trailer_id,
        start=shift_start,
        operations=tuple(operations),
    )
    if seen is not None:
        key = _shift_key(shift)
        if key in seen:
            return None
        seen.add(key)
    if not _is_shift_route_valid(instance, shift):
        return None
    return shift


def _failing_customers(
    instance: Instance,
    solution: Solution,
    config: RescueConfig,
) -> list[int]:
    cutoff_step = min(instance.horizon, config.end_day * MINUTES_PER_DAY // instance.unit)
    first_by_customer: dict[int, int] = {}
    for event in tank_events(instance, solution):
        if event.step >= cutoff_step:
            continue
        if event.safety_breach:
            first_by_customer.setdefault(event.point, event.step)
    return [
        customer_id
        for customer_id, _step in sorted(first_by_customer.items(), key=lambda item: item[1])
    ][: config.max_customers]


def _first_breach_minute(
    instance: Instance,
    solution: Solution,
    customer_id: int,
    event_cache: dict[int, list] | None = None,
) -> int | None:
    events = event_cache.get(customer_id, ()) if event_cache is not None else tank_events(instance, solution)
    for event in events:
        if event.point == customer_id and event.safety_breach:
            return event.time_start
    return None


def _inventory_at_arrival(
    instance: Instance,
    solution: Solution,
    customer_id: int,
    arrival: int,
    event_cache: dict[int, list] | None = None,
) -> float:
    step = min(max(arrival // instance.unit, 0), instance.horizon - 1)
    events = event_cache.get(customer_id, ()) if event_cache is not None else tank_events(instance, solution)
    for event in events:
        if event.point == customer_id and event.step == step:
            return event.after_consumption
    return 0.0


def _room_at_arrival(
    instance: Instance,
    solution: Solution,
    customer_id: int,
    arrival: int,
    event_cache: dict[int, list] | None = None,
) -> float:
    customer = instance.customer_by_point[customer_id]
    inventory = _inventory_at_arrival(instance, solution, customer_id, arrival, event_cache)
    return max(0.0, customer.capacity - inventory)


def _arrival_samples(
    start_minute: int,
    latest_arrival: int,
    count: int,
    lookback_days: int,
) -> list[int]:
    if count <= 1:
        return [latest_arrival]
    earliest = max(start_minute, latest_arrival - lookback_days * MINUTES_PER_DAY)
    span = latest_arrival - earliest
    if span <= 0:
        return [latest_arrival]
    return sorted(
        {
            earliest + round(span * i / (count - 1))
            for i in range(count)
        },
        reverse=True,
    )


def _trailer_load_at(
    instance: Instance,
    trailer_cache: dict[int, list[tuple[int, float]]],
    trailer_id: int,
    minute: int,
) -> float:
    load = instance.trailers[trailer_id].initial_quantity
    for shift_start, end_quantity in trailer_cache.get(trailer_id, ()):
        if shift_start >= minute:
            break
        load = end_quantity
    return load


def _events_by_customer(instance: Instance, solution: Solution) -> dict[int, list]:
    events: dict[int, list] = {}
    for event in tank_events(instance, solution):
        events.setdefault(event.point, []).append(event)
    return events


def _trailer_load_cache(instance: Instance, solution: Solution) -> dict[int, list[tuple[int, float]]]:
    cache: dict[int, list[tuple[int, float]]] = {}
    for derived in sorted(derive_solution(instance, solution), key=lambda item: item.shift.start):
        cache.setdefault(derived.shift.trailer, []).append(
            (derived.shift.start, derived.end_trailer_quantity)
        )
    return cache


@lru_cache(maxsize=100000)
def _is_shift_route_valid(instance: Instance, shift: Shift) -> bool:
    derived = _derive_single_shift(instance, shift)
    driver = instance.drivers[shift.driver]
    has_layover_customer = any(
        operation.point in instance.customer_by_point
        and instance.customer_by_point[operation.point].layover_customer
        for operation in shift.operations
    )
    if derived.layovers > 1:
        return False
    if derived.layovers > 0 and not has_layover_customer:
        return False
    if not is_time_window_valid(shift.start, derived.end, driver.time_windows):
        return False

    previous_driving = 0
    for operation, derived_operation in zip(shift.operations, derived.operations):
        if operation.point in instance.customer_by_point:
            customer = instance.customer_by_point[operation.point]
            if not is_time_window_valid(
                derived_operation.arrival,
                derived_operation.departure,
                customer.time_windows,
            ):
                return False
        if derived_operation.layover_before:
            driving = previous_driving + derived_operation.driving_before_layover
        else:
            driving = derived_operation.driving_since_layover
        if driving > driver.max_driving_duration + EPSILON:
            return False
        previous_driving = derived_operation.driving_since_layover
    return True


def _shift_key(shift: Shift) -> tuple[object, ...]:
    return (
        shift.driver,
        shift.trailer,
        shift.start,
        tuple((op.point, op.arrival, round(op.quantity, 9)) for op in shift.operations),
    )
