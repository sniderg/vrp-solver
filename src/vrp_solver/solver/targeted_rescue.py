from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import combinations, permutations

from ..inventory import tank_events
from ..model import Instance, Operation, Shift, Solution
from ..rules import derive_solution, is_time_window_valid, is_trailer_allowed, validate_solution
from ..highs_repair import repair_quantities_with_highs
from ..highs_time_opt import optimize_solution_times, try_optimize_shift_times
from ..route_cache import RouteCache
from .highs_selector import SelectorConfig, select_shifts_with_highs


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
    prioritize_severity: bool = False
    # In a full route-selection window, incumbent future routes are optional
    # columns and their quantities can be re-optimized jointly with a new
    # early refill.  Standalone/greedy repair cannot make that assumption and
    # must reserve all future capacity conservatively.
    allow_future_rebalance: bool = False
    greedy_append: bool = False
    greedy_rounds: int = 16
    greedy_candidate_limit: int = 240


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
    if config.greedy_append:
        return greedy_append_rescue(instance, baseline, config=config, failing=failing)
    fixed_prefix = _keep_shifts_started_before(
        baseline,
        config.replace_from_day * MINUTES_PER_DAY,
    )
    # A previous incumbent is only a route-column seed, not a presumption of
    # feasibility.  In particular, old cold starts may contain an impossible
    # final return that a column selector would otherwise retain unchanged.
    candidates = _baseline_window_shifts(instance, baseline, config)
    candidates.extend(generate_rescue_candidates(instance, fixed_prefix, failing, config=config))
    candidates.extend(generate_carryover_rescue_candidates(instance, fixed_prefix, failing, config=config))
    candidates.extend(generate_chain_rescue_candidates(instance, fixed_prefix, failing, config=config))
    candidates.extend(generate_multi_reload_candidates(instance, fixed_prefix, failing, config=config))
    # Individual generators are deliberately diverse; keep the central gate
    # here so no route can bypass complete derived driving validation.
    candidates = _dedupe_reindex(
        [candidate for candidate in candidates if _is_shift_route_valid(instance, candidate)]
    )
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
        # A rescue is never allowed to exchange one physical tank violation
        # for another.  Sparse-pool column generation may use soft safety
        # stock penalties, but the hard 0..capacity tank bounds remain
        # mandatory for every selected repair schedule.
        selector_config=SelectorConfig(
            strict_trailer_inventory=True,
            strict_nonnegative_inventory=True,
        ),
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


def greedy_append_rescue(
    instance: Instance,
    baseline: Solution,
    *,
    config: RescueConfig,
    failing: list[int] | None = None,
) -> tuple[Solution, RescueReport]:
    """Fast, fail-closed final feasibility repair for a cold-start schedule.

    Global selection is useful earlier in construction, but it becomes needlessly
    expensive when a nearly feasible schedule has only a few isolated inventory
    breaches.  Generate direct source-backed services, append one only when a
    full validation proves it strictly reduces the number of official-style
    errors and introduces no route/resource error, and repeat.
    """
    current = Solution(
        shifts=tuple(replace(shift, index=index) for index, shift in enumerate(baseline.shifts))
    )
    generated_total = 0
    selected = 0
    for _round in range(config.greedy_rounds):
        targets = failing if failing is not None else _failing_customers(instance, current, config)
        candidates = generate_rescue_candidates(instance, current, targets, config=config)
        # Dense substitutions are essential when every free driver/trailer
        # interval is already occupied.  These columns replace a small group
        # of incumbent one-stop shifts while covering several pressure tanks.
        candidates.extend(
            generate_chain_rescue_candidates(
                instance,
                current,
                targets,
                config=config,
                max_candidates=max(32, config.greedy_candidate_limit // 2),
            )
        )
        candidates.extend(
            generate_multi_reload_candidates(
                instance,
                current,
                targets,
                config=config,
                max_candidates=max(32, config.greedy_candidate_limit // 2),
            )
        )
        target_set = set(targets)
        candidates = _dedupe_reindex(candidates)
        candidates.sort(
            key=lambda shift: (
                -sum(
                    operation.quantity > EPSILON and operation.point in target_set
                    for operation in shift.operations
                ),
                -sum(operation.quantity > EPSILON for operation in shift.operations),
                shift.start,
            )
        )
        candidates = candidates[: config.greedy_candidate_limit]
        generated_total += len(candidates)
        current_errors = _error_count(instance, current)
        best: Solution | None = None
        best_errors = current_errors
        current_derived = derive_solution(instance, current)
        for candidate in candidates:
            conflicts = _rescue_conflicts(instance, current_derived, candidate)
            if any(
                shift.start < config.replace_from_day * MINUTES_PER_DAY
                for shift in conflicts
            ):
                continue
            proposal = Solution(
                shifts=tuple(
                    replace(shift, index=index)
                    for index, shift in enumerate(
                        (*(
                            shift for shift in current.shifts
                            if shift not in conflicts
                        ), candidate)
                    )
                )
            )
            violations = validate_solution(instance, proposal)
            # A repair may improve service quality or safety stock, but it may
            # never buy that improvement with physically impossible tank
            # state.  In particular, comparing only the total error count
            # previously allowed a proposal to exchange negative inventory for
            # a later tank overfill.  Treat *every* dynamic tank violation as
            # a hard rejection; the released checker does too.
            if any(
                violation.severity == "error" and violation.code == "DYN01"
                for violation in violations
            ):
                continue
            if any(
                violation.severity == "error"
                and violation.code not in {"QS01", "QS02", "DYN01"}
                for violation in violations
            ):
                continue
            errors = sum(violation.severity == "error" for violation in violations)
            if errors < best_errors:
                best = proposal
                best_errors = errors
        if best is None:
            break
        current = best
        selected += 1
        failing = None

    return current, RescueReport(
        tuple(_failing_customers(instance, current, config)),
        generated_total,
        selected,
    )


def _error_count(instance: Instance, solution: Solution) -> int:
    return sum(
        violation.severity == "error"
        for violation in validate_solution(instance, solution)
    )


def _rescue_conflicts(instance: Instance, current_derived, candidate: Shift) -> tuple[Shift, ...]:
    """Shifts that cannot coexist with a direct rescue on its resources."""
    candidate_end = derive_solution(instance, Solution(shifts=(candidate,)))[0].end
    driver = instance.drivers[candidate.driver]
    conflicts: list[Shift] = []
    for derived in current_derived:
        shift = derived.shift
        if shift.driver == candidate.driver:
            left = candidate_end + driver.min_inter_shift_duration
            right = derived.end + driver.min_inter_shift_duration
            if candidate.start < right and shift.start < left:
                conflicts.append(shift)
                continue
        if shift.trailer == candidate.trailer:
            if candidate.start < derived.end and shift.start < candidate_end:
                conflicts.append(shift)
    return tuple(conflicts)


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


def _baseline_window_shifts(
    instance: Instance,
    solution: Solution,
    config: RescueConfig,
) -> list[Shift]:
    start = config.replace_from_day * MINUTES_PER_DAY
    end = config.end_day * MINUTES_PER_DAY
    return [
        shift
        for shift in solution.shifts
        if start <= shift.start < end and _is_shift_route_valid(instance, shift)
    ]


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
            candidates.extend(
                _generate_callin_rescue_candidates(
                    instance,
                    baseline,
                    customer,
                    start_minute=start_minute,
                    end_minute=end_minute,
                    samples_per_customer=config.samples_per_customer,
                    seen=seen,
                )
            )
            continue
        breach_minute = _first_breach_minute(instance, baseline, customer_id, event_cache, min_minute=start_minute)
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
                outbound_driving = (
                    instance.time_matrix[instance.base_index][source.index]
                    + instance.time_matrix[source.index][customer.index]
                )
                post_layover_driving = (
                    instance.time_matrix[customer.index][source.index]
                    + instance.time_matrix[source.index][instance.base_index]
                )
                # A layover-enabled target can legally split the outbound and
                # return driving spell.  Do not reject that topology before
                # ``_is_shift_route_valid`` derives the represented return
                # layover and proves the complete driver window.
                if (
                    total_driving > driver.max_driving_duration
                    and (
                        not customer.layover_customer
                        or outbound_driving > driver.max_driving_duration
                        or post_layover_driving > driver.max_driving_duration
                    )
                ):
                    continue
                return_layover = (
                    total_driving > driver.max_driving_duration
                    and customer.layover_customer
                )
                trail_time = (
                    customer.setup_time
                    + (
                        instance.time_matrix[customer.index][source.index]
                        + driver.layover_duration
                        + source.setup_time
                        + instance.time_matrix[source.index][instance.base_index]
                        if return_layover
                        else return_time
                    )
                )
                
                target_arrivals = []
                for driver_window in driver.time_windows:
                    for customer_window in customer.time_windows:
                        min_arrival = max(
                            start_minute + route_to_customer,
                            driver_window.start + route_to_customer,
                            customer_window.start,
                        )
                        max_arrival = min(
                            latest_customer_arrival,
                            driver_window.end - trail_time,
                            customer_window.end - customer.setup_time,
                        )
                        if min_arrival <= max_arrival:
                            target_arrivals.extend(
                                _window_arrival_samples(
                                    min_arrival,
                                    max_arrival,
                                    config.samples_per_customer,
                                )
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
                    if return_layover:
                        terminal_source_arrival = (
                            departure
                            + instance.time_matrix[customer.index][source.index]
                            + driver.layover_duration
                        )
                        end = (
                            terminal_source_arrival
                            + source.setup_time
                            + instance.time_matrix[source.index][instance.base_index]
                        )
                    else:
                        terminal_source_arrival = None
                        end = departure + return_time
                    if end > end_minute:
                        continue
                    if not is_time_window_valid(shift_start, end, driver.time_windows):
                        continue
                    if not is_time_window_valid(arrival, departure, customer.time_windows):
                        continue
                    inventory_at_arrival = _inventory_at_arrival(
                        instance, baseline, customer_id, arrival, event_cache
                    )
                    if (
                        inventory_at_arrival
                        > customer.capacity * config.max_pre_service_fill_ratio + EPSILON
                    ):
                        continue
                    # A pre-existing shortage can make ``capacity - inventory``
                    # exceed the physical tank.  No single delivery may use
                    # that artificial room.
                    room = min(
                        customer.capacity,
                        max(0.0, customer.capacity - inventory_at_arrival),
                    )
                    # An added delivery persists through all later inventory
                    # checkpoints.  Reserve capacity for every already
                    # committed future delivery before offering this column;
                    # checking the tank only at ``arrival`` can otherwise
                    # create a perfectly plausible early refill that makes a
                    # later incumbent delivery overfill the tank.
                    future_room = (
                        customer.capacity
                        if config.allow_future_rebalance
                        else _future_capacity_room(
                            instance,
                            customer_id,
                            arrival,
                            event_cache,
                        )
                    )
                    target_room = max(
                        0.0,
                        customer.capacity * config.target_fill_ratio
                        - inventory_at_arrival,
                    )
                    trailer_load = _trailer_load_at(instance, trailer_cache, trailer.index, shift_start)
                    load_quantity = max(0.0, trailer.capacity - trailer_load)
                    available_quantity = trailer_load + load_quantity
                    quantity = min(available_quantity, room, future_room, target_room)
                    if quantity < customer.min_operation_quantity - EPSILON:
                        continue

                    operations = []
                    if load_quantity > EPSILON:
                        operations.append(Operation(source.index, source_arrival, -load_quantity))
                    operations.append(Operation(customer.index, arrival, quantity))
                    if terminal_source_arrival is not None:
                        operations.append(Operation(
                            source.index,
                            terminal_source_arrival,
                            -quantity,
                        ))
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

    callin_customers = [
        customer_id
        for customer_id in failing_customers
        if instance.customer_by_point[customer_id].call_in
        and _next_unsatisfied_callin_order(instance.customer_by_point[customer_id], baseline) is not None
    ]
    candidates.extend(
        _generate_callin_chain_rescue_candidates(
            instance,
            baseline,
            callin_customers,
            start_minute=start_minute,
            end_minute=end_minute,
            samples_per_customer=config.samples_per_customer,
            seen=seen,
        )
    )
    # The executable routinely combines a deadline-bound call-in with one
    # nearby VMI service.  Those stops are not interchangeable: the call-in
    # fixes the route's time anchor, while the VMI stop uses the remaining
    # payload and avoids a separate resource interval.  Keep this family
    # deliberately bounded so cold starts remain fast.
    candidates.extend(
        _generate_callin_vmi_chain_rescue_candidates(
            instance,
            baseline,
            callin_customers,
            start_minute=start_minute,
            end_minute=end_minute,
            samples_per_customer=config.samples_per_customer,
            event_cache=event_cache,
            trailer_cache=trailer_cache,
            seen=seen,
        )
    )
    # In a saturated schedule, an order cannot necessarily be appended as a
    # new shift.  Generate atomic replacements of an incumbent route instead:
    # the inserted call-in keeps the route's other VMI stops, then the timing
    # MIP proves the complete shifted route still fits its resource chain.
    candidates.extend(
        _generate_callin_route_insert_candidates(
            instance,
            baseline,
            callin_customers,
            seen=seen,
        )
    )
    return candidates


def _generate_callin_route_insert_candidates(
    instance: Instance,
    baseline: Solution,
    customer_ids: list[int],
    *,
    seen: set[tuple[object, ...]],
) -> list[Shift]:
    """Insert an unmet call-in in place of one incumbent route column.

    These are route *replacements*, not overlapping additions.  They are
    particularly important when every compatible driver/trailer pair is busy
    inside the order's time window.
    """
    derived = derive_solution(instance, baseline)
    result: list[Shift] = []
    for customer_id in customer_ids:
        customer = instance.customer_by_point[customer_id]
        order_info = _next_unsatisfied_callin_order(customer, baseline)
        if order_info is None:
            continue
        _order_index, order = order_info
        for position, shift in enumerate(baseline.shifts):
            if shift.trailer not in customer.allowed_trailers:
                continue
            if shift.start > order.latest_time:
                continue
            # The replacement must end before every fixed successor sharing
            # either resource; derive the latest permissible endpoint once.
            latest_end: int | None = None
            driver_gap = instance.drivers[shift.driver].min_inter_shift_duration
            for other_position, other in enumerate(baseline.shifts):
                if other_position == position:
                    continue
                if other.driver == shift.driver and other.start >= shift.start:
                    bound = other.start - driver_gap
                    latest_end = bound if latest_end is None else min(latest_end, bound)
                if other.trailer == shift.trailer and other.start >= shift.start:
                    bound = other.start
                    latest_end = bound if latest_end is None else min(latest_end, bound)
            for insert_at in range(len(shift.operations) + 1):
                operations = (
                    shift.operations[:insert_at]
                    + (Operation(customer_id, order.earliest_time, order.min_quantity_to_satisfy),)
                    + shift.operations[insert_at:]
                )
                tentative = replace(shift, operations=operations)
                retimed = try_optimize_shift_times(
                    instance, tentative, latest_end=latest_end,
                )
                if retimed is None:
                    continue
                if not any(
                    order.earliest_time <= operation.arrival <= order.latest_time
                    and operation.point == customer_id
                    for operation in retimed.operations
                ):
                    continue
                key = _shift_key(retimed)
                if key in seen or not _is_shift_route_valid(instance, retimed):
                    continue
                seen.add(key)
                result.append(replace(retimed, index=len(result)))
    return result


def _generate_callin_rescue_candidates(
    instance: Instance,
    baseline: Solution,
    customer,
    *,
    start_minute: int,
    end_minute: int,
    samples_per_customer: int,
    seen: set[tuple[object, ...]],
) -> list[Shift]:
    """Generate source-backed columns for the first unmet call-in order."""
    order_info = _next_unsatisfied_callin_order(customer, baseline)
    if order_info is None:
        return []
    _order_index, order = order_info
    quantity = order.min_quantity_to_satisfy
    candidates: list[Shift] = []
    for driver in instance.drivers:
        for trailer in instance.trailers:
            if trailer.index not in driver.trailer_ids or trailer.index not in customer.allowed_trailers:
                continue
            if quantity > trailer.capacity + EPSILON:
                continue
            for source in instance.sources:
                if trailer.index not in source.allowed_trailers:
                    continue
                lead = (
                    instance.time_matrix[instance.base_index][source.index]
                    + source.setup_time
                    + instance.time_matrix[source.index][customer.index]
                )
                trail = customer.setup_time + instance.time_matrix[customer.index][instance.base_index]
                for driver_window in driver.time_windows:
                    earliest = max(
                        order.earliest_time,
                        start_minute + lead,
                        driver_window.start + lead,
                    )
                    latest = min(
                        order.latest_time - customer.setup_time,
                        end_minute - trail,
                        driver_window.end - trail,
                    )
                    if earliest > latest:
                        continue
                    for arrival in _window_arrival_samples(earliest, latest, samples_per_customer):
                        shift_start = arrival - lead
                        source_arrival = shift_start + instance.time_matrix[instance.base_index][source.index]
                        departure = arrival + customer.setup_time
                        end = departure + instance.time_matrix[customer.index][instance.base_index]
                        if not is_time_window_valid(shift_start, end, driver.time_windows):
                            continue
                        if not is_time_window_valid(arrival, departure, customer.time_windows):
                            continue
                        shift = Shift(
                            index=0,
                            driver=driver.index,
                            trailer=trailer.index,
                            start=shift_start,
                            operations=(
                                Operation(source.index, source_arrival, -quantity),
                                Operation(customer.index, arrival, quantity),
                            ),
                        )
                        key = _shift_key(shift)
                        if key in seen or not _is_shift_route_valid(instance, shift):
                            continue
                        seen.add(key)
                        candidates.append(replace(shift, index=len(candidates)))
    return candidates


def _next_unsatisfied_callin_order(customer, solution: Solution):
    for order_index, order in enumerate(customer.orders):
        delivered = sum(
            operation.quantity
            for shift in solution.shifts
            for operation in shift.operations
            if operation.point == customer.index
            and operation.quantity > EPSILON
            and order.earliest_time <= operation.arrival <= order.latest_time
        )
        if delivered + EPSILON < order.min_quantity_to_satisfy:
            return order_index, order
    return None


def _generate_callin_chain_rescue_candidates(
    instance: Instance,
    baseline: Solution,
    customer_ids: list[int],
    *,
    start_minute: int,
    end_minute: int,
    samples_per_customer: int,
    seen: set[tuple[object, ...]],
) -> list[Shift]:
    """Build two/three-stop call-in routes, reloading whenever capacity requires.

    Direct call-in columns cannot express V2.12's early 136 -> reload ->
    314/124 pattern.  These are still ordinary native route columns: every
    stop, source visit, time window, and driving bound is checked locally.
    """
    tasks = [
        (customer_id, order_info[0], order_info[1])
        for customer_id in customer_ids
        if (order_info := _next_unsatisfied_callin_order(instance.customer_by_point[customer_id], baseline)) is not None
    ]
    candidates: list[Shift] = []
    trailer_cache = _trailer_load_cache(instance, baseline)
    # Singleton chains matter too: a trailer may legally leave base carrying
    # its current stock, serve a deadline-bound call-in, and reload only
    # afterwards. Restricting this family to two call-ins omitted exactly that
    # topology and forced an unnecessary source visit before the first order.
    for size in range(1, min(3, len(tasks)) + 1):
        for subset in combinations(tasks, size):
            allowed = set(instance.customer_by_point[subset[0][0]].allowed_trailers)
            for customer_id, _order_index, _order in subset[1:]:
                allowed &= set(instance.customer_by_point[customer_id].allowed_trailers)
            if not allowed:
                continue
            for route in permutations(subset):
                first_customer = instance.customer_by_point[route[0][0]]
                first_order = route[0][2]
                for driver in instance.drivers:
                    for trailer_id in driver.trailer_ids:
                        if trailer_id not in allowed:
                            continue
                        trailer = instance.trailers[trailer_id]
                        if any(order.min_quantity_to_satisfy > trailer.capacity + EPSILON for _, _, order in route):
                            continue
                        for source in instance.sources:
                            if trailer_id not in source.allowed_trailers:
                                continue
                            # A trailer which has not yet been used can leave
                            # base preloaded.  V2.12's critical t14 route does
                            # exactly that: it delivers 136, then refills
                            # before 314/124.  Starting every chain at source
                            # makes that physically impossible once trailer
                            # state is enforced.
                            # Explore both legal starts.  Trailer state is
                            # sampled at the resulting start minute; using the
                            # XML initial quantity here silently creates route
                            # columns that strict state flow can never select.
                            for lead in (
                                instance.time_matrix[instance.base_index][first_customer.index],
                                instance.time_matrix[instance.base_index][source.index]
                                + source.setup_time
                                + instance.time_matrix[source.index][first_customer.index],
                            ):
                                for window in driver.time_windows:
                                    earliest = max(first_order.earliest_time, start_minute + lead, window.start + lead)
                                    latest = min(first_order.latest_time - first_customer.setup_time, end_minute - lead, window.end - lead)
                                    if earliest > latest:
                                        continue
                                    for first_arrival in _window_arrival_samples(earliest, latest, samples_per_customer):
                                        shift_start = first_arrival - lead
                                        shift = _build_callin_chain(
                                            instance,
                                            route,
                                            driver.index,
                                            trailer_id,
                                            source.index,
                                            shift_start,
                                            _trailer_load_at(instance, trailer_cache, trailer_id, shift_start),
                                        )
                                        if shift is None:
                                            continue
                                        derived = _derive_single_shift(instance, shift)
                                        if derived.end > end_minute or not is_time_window_valid(shift.start, derived.end, driver.time_windows):
                                            continue
                                        key = _shift_key(shift)
                                        if key in seen:
                                            continue
                                        seen.add(key)
                                        candidates.append(replace(shift, index=len(candidates)))
    return candidates


def _generate_callin_vmi_chain_rescue_candidates(
    instance: Instance,
    baseline: Solution,
    customer_ids: list[int],
    *,
    start_minute: int,
    end_minute: int,
    samples_per_customer: int,
    event_cache: dict[int, list],
    trailer_cache: dict[int, list[tuple[int, float]]],
    seen: set[tuple[object, ...]],
) -> list[Shift]:
    """Generate short call-in-then-VMI routes from native state projections.

    A support stop is selected from low projected VMI tanks, ranked by urgency
    and travel from the call-in anchor.  This is a generic topology rule, not
    an instance-specific route list.
    """
    tasks = [
        (customer_id, order_info[0], order_info[1])
        for customer_id in customer_ids
        if (order_info := _next_unsatisfied_callin_order(
            instance.customer_by_point[customer_id], baseline
        )) is not None
    ]
    # A useful companion is often not yet failing: it is a tank which will
    # otherwise consume another shift later in the horizon.  Rank all VMI
    # tanks by their lowest projected fill ratio, then use geography only as a
    # tie-breaker for each call-in route.
    support_margin: dict[int, float] = {}
    for customer in instance.customers:
        if customer.call_in:
            continue
        events = event_cache.get(customer.index, ())
        relevant = [
            event.ending_inventory / max(customer.capacity, EPSILON)
            for event in events
            if start_minute <= event.time_start < end_minute
        ]
        if relevant:
            support_margin[customer.index] = min(relevant)
    candidates: list[Shift] = []
    # One or two call-ins plus one VMI stop captures the useful shared-shift
    # shape without an exponential route enumeration.
    for size in range(1, min(2, len(tasks)) + 1):
        for subset in combinations(tasks, size):
            allowed = set(instance.customer_by_point[subset[0][0]].allowed_trailers)
            for customer_id, _order_index, _order in subset[1:]:
                allowed &= set(instance.customer_by_point[customer_id].allowed_trailers)
            if not allowed:
                continue
            for callin_route in permutations(subset):
                last_callin = callin_route[-1][0]
                nearby_support = sorted(
                    (
                        customer_id for customer_id in support_margin
                        if customer_id not in {task[0] for task in callin_route}
                    ),
                    key=lambda customer_id: (
                        support_margin[customer_id],
                        instance.time_matrix[last_callin][customer_id],
                        customer_id,
                    ),
                )[:3]
                for support_id in nearby_support:
                    support_customer = instance.customer_by_point[support_id]
                    compatible = allowed & set(support_customer.allowed_trailers)
                    if not compatible:
                        continue
                    first_customer = instance.customer_by_point[callin_route[0][0]]
                    first_order = callin_route[0][2]
                    for driver in instance.drivers:
                        for trailer_id in driver.trailer_ids:
                            if trailer_id not in compatible:
                                continue
                            trailer = instance.trailers[trailer_id]
                            if any(order.min_quantity_to_satisfy > trailer.capacity + EPSILON for _, _, order in callin_route):
                                continue
                            for source in instance.sources:
                                if trailer_id not in source.allowed_trailers:
                                    continue
                                for window in driver.time_windows:
                                    # Actual state at the candidate start is
                                    # determined below; this conservative lead
                                    # leaves room for an initial reload.
                                    lead = (
                                        instance.time_matrix[instance.base_index][source.index]
                                        + source.setup_time
                                        + instance.time_matrix[source.index][first_customer.index]
                                    )
                                    earliest = max(first_order.earliest_time, start_minute + lead, window.start + lead)
                                    latest = min(first_order.latest_time - first_customer.setup_time, end_minute - 1)
                                    if earliest > latest:
                                        continue
                                    for first_arrival in _window_arrival_samples(earliest, latest, samples_per_customer):
                                        start = first_arrival - lead
                                        initial_carried = _trailer_load_at(
                                            instance, trailer_cache, trailer_id, start
                                        )
                                        shift = _build_callin_vmi_chain(
                                            instance,
                                            baseline,
                                            event_cache,
                                            callin_route,
                                            support_id,
                                            driver.index,
                                            trailer_id,
                                            source.index,
                                            start,
                                            initial_carried,
                                        )
                                        if shift is None:
                                            continue
                                        derived = _derive_single_shift(instance, shift)
                                        if derived.end > end_minute or not is_time_window_valid(shift.start, derived.end, driver.time_windows):
                                            continue
                                        key = _shift_key(shift)
                                        if key in seen:
                                            continue
                                        seen.add(key)
                                        candidates.append(replace(shift, index=len(candidates)))
    return candidates


def _build_callin_vmi_chain(
    instance: Instance,
    baseline: Solution,
    event_cache: dict[int, list],
    callin_route,
    support_id: int,
    driver_id: int,
    trailer_id: int,
    source_id: int,
    start: int,
    initial_carried: float,
) -> Shift | None:
    """Build one mixed chain, refilling whenever the next stop needs it."""
    trailer = instance.trailers[trailer_id]
    operations: list[Operation] = []
    current_point = instance.base_index
    current_time = start
    carried = initial_carried

    def refill() -> tuple[int, int, float]:
        nonlocal current_point, current_time, carried
        source_arrival = current_time + instance.time_matrix[current_point][source_id]
        load = max(0.0, trailer.capacity - carried)
        if load > EPSILON:
            operations.append(Operation(source_id, source_arrival, -load))
            carried += load
        current_time = source_arrival + instance.setup_time_for_point(source_id)
        current_point = source_id
        return current_point, current_time, carried

    for customer_id, _order_index, order in callin_route:
        quantity = order.min_quantity_to_satisfy
        if carried + EPSILON < quantity:
            refill()
        customer = instance.customer_by_point[customer_id]
        raw_arrival = max(current_time + instance.time_matrix[current_point][customer_id], order.earliest_time)
        arrival = next(
            (
                max(raw_arrival, window.start)
                for window in customer.time_windows
                if max(raw_arrival, window.start) + customer.setup_time <= window.end
                and max(raw_arrival, window.start) + customer.setup_time <= order.latest_time
            ),
            None,
        )
        if arrival is None:
            return None
        operations.append(Operation(customer_id, arrival, quantity))
        carried -= quantity
        current_time = arrival + customer.setup_time
        current_point = customer_id

    support_customer = instance.customer_by_point[support_id]
    # A full refill before the VMI stop is inexpensive in column generation
    # and lets the selector decide whether sharing this route is worthwhile.
    if carried < support_customer.min_operation_quantity - EPSILON:
        refill()
    arrival = current_time + instance.time_matrix[current_point][support_id]
    departure = arrival + support_customer.setup_time
    breach = _first_breach_minute(instance, baseline, support_id, event_cache)
    if (breach is not None and arrival >= breach) or not is_time_window_valid(arrival, departure, support_customer.time_windows):
        return None
    inventory = _inventory_at_arrival(instance, baseline, support_id, arrival, event_cache)
    room = min(support_customer.capacity, max(0.0, support_customer.capacity - inventory))
    target = max(0.0, support_customer.capacity * 0.95 - inventory)
    quantity = min(carried, room, target)
    if quantity < support_customer.min_operation_quantity - EPSILON:
        return None
    operations.append(Operation(support_id, arrival, quantity))
    shift = Shift(index=0, driver=driver_id, trailer=trailer_id, start=start, operations=tuple(operations))
    return shift if _is_shift_route_valid(instance, shift) else None


def _build_callin_chain(
    instance,
    route,
    driver_id: int,
    trailer_id: int,
    source_id: int,
    start: int,
    initial_carried: float,
) -> Shift | None:
    trailer = instance.trailers[trailer_id]
    operations: list[Operation] = []
    current_point = instance.base_index
    current_time = start
    carried = initial_carried
    for customer_id, _order_index, order in route:
        customer = instance.customer_by_point[customer_id]
        quantity = order.min_quantity_to_satisfy
        if carried + EPSILON < quantity:
            source_arrival = current_time + instance.time_matrix[current_point][source_id]
            load_quantity = trailer.capacity - carried
            operations.append(Operation(source_id, source_arrival, -load_quantity))
            current_time = source_arrival + instance.setup_time_for_point(source_id)
            current_point = source_id
            carried += load_quantity
        raw_arrival = max(current_time + instance.time_matrix[current_point][customer_id], order.earliest_time)
        arrival = next(
            (
                max(raw_arrival, window.start)
                for window in customer.time_windows
                if max(raw_arrival, window.start) + customer.setup_time <= window.end
                and max(raw_arrival, window.start) + customer.setup_time <= order.latest_time
            ),
            None,
        )
        if arrival is None:
            return None
        operations.append(Operation(customer_id, arrival, quantity))
        current_time = arrival + customer.setup_time
        current_point = customer_id
        carried -= quantity
    shift = Shift(index=0, driver=driver_id, trailer=trailer_id, start=start, operations=tuple(operations))
    return shift if _is_shift_route_valid(instance, shift) else None


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
    max_candidates: int | None = None,
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
        max_sequences=(max(128, max_candidates * 8) if max_candidates else None),
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
                    if max_candidates is not None and len(candidates) >= max_candidates:
                        return candidates

    return candidates


def generate_multi_reload_candidates(
    instance: Instance,
    baseline: Solution,
    failing_customers: list[int],
    *,
    config: RescueConfig,
    max_candidates: int | None = None,
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
        max_sequences=(max(128, max_candidates * 8) if max_candidates else None),
    )
    candidates: list[Shift] = []
    seen: set[tuple[object, ...]] = set()

    for first_segment, second_segment in sequences:
        anchor = instance.customer_by_point[first_segment[0]]
        anchor_breach = _first_breach_minute(
            instance, baseline, anchor.index, event_cache, min_minute=start_minute
        )
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
                    if max_candidates is not None and len(candidates) >= max_candidates:
                        return candidates

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
                shift_start = max(start_minute, window.start)
                if shift_start + duration > min(end_minute, window.end):
                    continue
                source_arrival = shift_start + instance.time_matrix[instance.base_index][source.index]
                carried = _trailer_load_at(
                    instance, trailer_cache, trailer.index, shift_start,
                )
                load_quantity = max(0.0, trailer.capacity - carried)
                if load_quantity <= EPSILON:
                    continue
                op = Operation(
                    point=source.index,
                    arrival=source_arrival,
                    quantity=-load_quantity,
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
                    if max_candidates is not None and len(candidates) >= max_candidates:
                        return candidates

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
                    inventory_at_arrival = _inventory_at_arrival(
                        instance,
                        baseline,
                        c_id,
                        anchor_arrival,
                        event_cache,
                    )
                    room = min(
                        customer.capacity,
                        max(0.0, customer.capacity - inventory_at_arrival),
                    )
                    target_room = max(
                        0.0,
                        customer.capacity * config.target_fill_ratio - inventory_at_arrival,
                    )
                    # This path intentionally starts from an existing carried
                    # load.  It is how a 510-minute direct round-trip can
                    # serve a remote tank that is infeasible via a source.
                    trailer_load = _trailer_load_at(
                        instance, trailer_cache, trailer.index, shift_start,
                    )
                    quantity = min(trailer_load, room, target_room)
                    if quantity < customer.min_operation_quantity - EPSILON:
                        continue

                    op = Operation(point=c_id, arrival=anchor_arrival, quantity=quantity)
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
                    if max_candidates is not None and len(candidates) >= max_candidates:
                        return candidates

    return candidates


def _multi_reload_sequences(
    instance: Instance,
    baseline: Solution,
    failing_customers: list[int],
    config: RescueConfig,
    event_cache: dict[int, list],
    route_cache: RouteCache | None = None,
    max_sequences: int | None = None,
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
            if max_sequences is not None and len(output) >= max_sequences:
                return output
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
    max_sequences: int | None = None,
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
        if (
            len(current_chain) >= config.max_chain_length
            or (max_sequences is not None and len(sequences) >= max_sequences)
        ):
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
                if max_sequences is not None and len(sequences) >= max_sequences:
                    return
            expand_chain(new_chain)
            if max_sequences is not None and len(sequences) >= max_sequences:
                return

    for anchor in failing:
        expand_chain((anchor,))
        if max_sequences is not None and len(sequences) >= max_sequences:
            break

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

    delivered_count = 0
    for customer_id in sequence:
        customer = instance.customer_by_point[customer_id]
        # A reload is an extension arc, not a once-per-route special case.
        # Refill whenever the current trailer stock cannot cover the useful
        # fill at the next customer.  This is the essential primitive behind
        # the long source -> several customers -> source -> several customers
        # chains observed in the executable's cold starts.
        inventory_at_current_time = _inventory_at_arrival(
            instance, baseline, customer_id, current_time, event_cache
        )
        desired_quantity = min(
            max(0.0, customer.capacity - inventory_at_current_time),
            max(0.0, customer.capacity * config.target_fill_ratio - inventory_at_current_time),
        )
        if (
            desired_quantity >= customer.min_operation_quantity - EPSILON
            and trailer_load + EPSILON < desired_quantity
        ):
            reload_arrival = current_time + instance.time_matrix[current_point][source_id]
            reload_quantity = max(0.0, trailer.capacity - trailer_load)
            if reload_quantity <= EPSILON:
                return None
            operations.append(Operation(source_id, reload_arrival, -reload_quantity))
            trailer_load += reload_quantity
            current_time = reload_arrival + source.setup_time
            current_point = source_id
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
        current_time = departure
        current_point = customer_id

    if delivered_count < 2:
        return None
    end = current_time + instance.time_matrix[current_point][instance.base_index]
    if end > end_minute:
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
    severity_by_customer: dict[int, tuple[int, float]] = {}
    for event in tank_events(instance, solution):
        if event.step >= cutoff_step:
            continue
        if event.safety_breach:
            first_by_customer.setdefault(event.point, event.step)
            negative_steps, deficit_area = severity_by_customer.get(event.point, (0, 0.0))
            severity_by_customer[event.point] = (
                negative_steps + int(event.ending_inventory < -EPSILON),
                deficit_area + max(0.0, event.safety_level - event.ending_inventory),
            )
    # Call-ins have no inventory trajectory, so a VMI-only scan leaves a
    # near-feasible solution with missed orders invisible to route repair.
    # Include each unsatisfied flexible minimum at its order deadline.  This
    # lets the same native candidate generator build the coupled resource
    # alternatives needed to insert a call-in without treating it as a tank
    # runout.
    cutoff_minute = cutoff_step * instance.unit
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
                    delivered_by_order[key] = (
                        delivered_by_order.get(key, 0.0) + operation.quantity
                    )
    for customer in instance.customers:
        if not customer.call_in:
            continue
        for order_index, order in enumerate(customer.orders):
            if order.latest_time >= cutoff_minute:
                continue
            missing = order.min_quantity_to_satisfy - delivered_by_order.get(
                (customer.index, order_index), 0.0
            )
            if missing <= EPSILON:
                continue
            deadline_step = min(order.latest_time // instance.unit, cutoff_step - 1)
            first_by_customer[customer.index] = min(
                first_by_customer.get(customer.index, deadline_step), deadline_step
            )
            negative_steps, deficit_area = severity_by_customer.get(customer.index, (0, 0.0))
            severity_by_customer[customer.index] = (negative_steps, deficit_area + missing)
    if config.prioritize_severity:
        ordered = sorted(
            first_by_customer,
            key=lambda customer_id: (
                -severity_by_customer[customer_id][0],
                -severity_by_customer[customer_id][1],
                first_by_customer[customer_id],
                customer_id,
            ),
        )
    else:
        ordered = [
            customer_id
            for customer_id, _step in sorted(first_by_customer.items(), key=lambda item: item[1])
        ]
    return [
        customer_id for customer_id in ordered[: config.max_customers]
    ]


def _first_breach_minute(
    instance: Instance,
    solution: Solution,
    customer_id: int,
    event_cache: dict[int, list] | None = None,
    min_minute: int = 0,
) -> int | None:
    events = event_cache.get(customer_id, ()) if event_cache is not None else tank_events(instance, solution)
    for event in events:
        if event.point == customer_id and event.safety_breach and event.time_start >= min_minute:
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


def _future_capacity_room(
    instance: Instance,
    customer_id: int,
    arrival: int,
    event_cache: dict[int, list],
) -> float:
    """Maximum extra quantity that can persist after ``arrival`` safely.

    A new delivery raises the inventory at its own step and every subsequent
    step by the same amount.  Therefore the available room is the smallest
    residual capacity in the *existing* future trajectory, not just the room
    when the truck arrives.  This is deliberately conservative: a joint
    quantity optimizer can later reduce a future delivery, but a standalone
    route column must never assume that it will.
    """
    customer = instance.customer_by_point[customer_id]
    first_step = min(max(arrival // instance.unit, 0), instance.horizon - 1)
    residuals = [
        customer.capacity - event.ending_inventory
        for event in event_cache.get(customer_id, ())
        if event.step >= first_step
    ]
    return max(0.0, min(residuals)) if residuals else customer.capacity


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
    if derived.operations:
        return_driving = (
            derived.operations[-1].driving_since_layover
            + instance.time_matrix[derived.operations[-1].point][instance.base_index]
        )
        if return_driving > driver.max_driving_duration + EPSILON:
            return False
    return True


def _shift_key(shift: Shift) -> tuple[object, ...]:
    return (
        shift.driver,
        shift.trailer,
        shift.start,
        tuple((op.point, op.arrival, round(op.quantity, 9)) for op in shift.operations),
    )
