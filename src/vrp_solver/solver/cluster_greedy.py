from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Set, Tuple

from ..inventory import days_of_inventory, project_customer_inventory, tank_events
from ..model import Customer, Instance, Operation, Shift, Solution, TimeWindow
from ..rules import is_driving_duration_valid, is_time_window_valid, is_trailer_allowed
from ..movement import nearest_neighbors


EPSILON = 1e-6
ECONOMIC_SERVICE_FILL_RATIO = 0.75
WEEKEND_DELIVERY_WEIGHT = 0.65

@dataclass(frozen=True)
class ConstructionReport:
    shifts: int
    operations: int
    delivered_quantity: float
    unscheduled_customers: tuple[int, ...]
    exhausted_resources: bool

@dataclass
class _DriverState:
    driver: int
    next_window_index: int = 0
    available_time: int = 0

@dataclass
class _TrailerState:
    trailer: int
    available_time: int = 0
    trailer_quantity: float = 0.0

@dataclass
class _ResourceState:
    driver: int
    trailer: int
    trailer_quantity: float = 0.0
    available_time: int = 0
    trailer_available_time: int = 0

@dataclass(frozen=True)
class _Candidate:
    customer: Customer
    arrival: int
    departure: int
    quantity: float
    travel_time: int
    source_arrival: int | None
    load_quantity: float
    source_index: int | None = None

def construct_cluster_solution(
    instance: Instance,
    *,
    safety_buffer: float = 0.20,
    neighborhood_size: int = 5,
    max_shifts: int | None = None,
    score_cutoff_minute: int | None = None,
    terminal_buffer_days: float = 0.0,
) -> tuple[Solution, ConstructionReport]:
    drivers, trailers = _initial_resources(instance)
    scheduled: dict[int, dict[int, float]] = {customer.index: {} for customer in instance.customers}
    ignore_before_step: dict[int, int] = {customer.index: 0 for customer in instance.customers}
    planned_volume_by_day: dict[int, float] = {}
    daily_delivery_targets = _daily_delivery_targets(instance)
    shifts: list[Shift] = []
    exhausted_resources = False

    neighborhoods = _compute_neighborhoods(instance, k=neighborhood_size)
    source_lead_minutes = {
        customer.index: min(
            instance.time_matrix[source.index][customer.index]
            for source in instance.sources
        )
        for customer in instance.customers
    }

    while True:
        if max_shifts is not None and len(shifts) >= max_shifts:
            exhausted_resources = len(_all_needs(instance, scheduled, ignore_before_step, score_cutoff_minute)) > 0
            break
        
        all_needs = _all_needs(instance, scheduled, ignore_before_step, score_cutoff_minute)
        if not all_needs:
            break
        all_needs.sort(key=lambda x: x[1])

        next_window_infos = _resource_window_candidates(instance, drivers, trailers, score_cutoff_minute)
        if not next_window_infos:
            exhausted_resources = True
            break

        shift = None
        selected_driver_state_index = None
        selected_trailer_state_index = None
        selected_resource = None
        for start_time, driver_state_index, trailer_state_index, window_index, window, resource in next_window_infos:
            for target_customer, breach_step in all_needs:
                if (breach_step + 1) * instance.unit < start_time:
                    continue

                shift = _build_cluster_shift(
                    instance,
                    resource,
                    window,
                    len(shifts),
                    target_customer,
                    neighborhoods,
                    source_lead_minutes,
                    scheduled,
                    ignore_before_step,
                    planned_volume_by_day,
                    daily_delivery_targets,
                    safety_buffer,
                    score_cutoff_minute,
                    terminal_buffer_days,
                )
                if shift:
                    selected_driver_state_index = driver_state_index
                    selected_trailer_state_index = trailer_state_index
                    selected_resource = resource
                    break
            if shift:
                break
        
        if shift is None:
            earliest_start = next_window_infos[0][0]
            for start_time, driver_state_index, _, window_index, _, _ in next_window_infos:
                if start_time != earliest_start:
                    break
                drivers[driver_state_index].next_window_index = max(
                    drivers[driver_state_index].next_window_index,
                    window_index + 1,
                )
            continue

        drivers[selected_driver_state_index].available_time = selected_resource.available_time
        trailers[selected_trailer_state_index].available_time = selected_resource.trailer_available_time
        trailers[selected_trailer_state_index].trailer_quantity = selected_resource.trailer_quantity
        shifts.append(shift)

    unscheduled = tuple(
        customer.index
        for customer in instance.customers
        if _first_breach_step(instance, customer, scheduled[customer.index], 0)
        is not None
    )
    
    solution = Solution(shifts=tuple(shifts))
    return solution, ConstructionReport(
        shifts=len(solution.shifts),
        operations=sum(len(shift.operations) for shift in solution.shifts),
        delivered_quantity=sum(op.quantity for s in solution.shifts for op in s.operations if op.quantity > 0),
        unscheduled_customers=unscheduled,
        exhausted_resources=exhausted_resources,
    )

def _all_needs(instance: Instance, scheduled, ignore_before_step, score_cutoff_minute=None):
    needs = []
    for customer in instance.customers:
        if customer.call_in:
            order = _next_unsatisfied_order(customer, scheduled[customer.index], score_cutoff_minute)
            if order is not None:
                order_index, order_due = order
                needs.append((customer, max(0, order_due.latest_time // instance.unit - 1)))
            continue
        breach = _first_breach_step(
            instance,
            customer,
            scheduled[customer.index],
            ignore_before_step[customer.index],
        )
        if breach is not None:
            needs.append((customer, breach))
    return needs

def _next_unsatisfied_order(customer, deliveries, score_cutoff_minute=None):
    for order_index, order in enumerate(customer.orders):
        if score_cutoff_minute is not None and order.earliest_time >= score_cutoff_minute:
            continue
        delivered = sum(
            quantity
            for arrival, quantity in deliveries.items()
            if order.earliest_time <= arrival <= order.latest_time
        )
        if delivered + EPSILON < order.min_quantity_to_satisfy:
            return order_index, order
    return None

def _compute_neighborhoods(instance: Instance, k: int):
    rows = nearest_neighbors(instance, k=len(instance.time_matrix), metric="distance")
    nb_dict = {}
    for row in rows:
        o = row["origin"]
        if o not in nb_dict: nb_dict[o] = []
        if row["destination_kind"] == "customer" and len(nb_dict[o]) < k:
            nb_dict[o].append(row["destination"])
    return nb_dict

def _build_cluster_shift(
    instance,
    resource,
    window,
    shift_idx,
    target,
    neighborhoods,
    source_lead_minutes,
    scheduled,
    ignore,
    planned_volume_by_day,
    daily_delivery_targets,
    buffer,
    score_cutoff_minute,
    terminal_buffer_days,
):
    driver = instance.drivers[resource.driver]
    start = max(window.start, resource.available_time)
    
    operations = []
    current_pt = instance.base_index
    current_time = start
    driving = 0
    end_after_return = start

    served_this_shift = set()

    while True:
        candidates_to_try = _candidate_customer_ids(
            instance,
            target,
            current_pt=current_pt,
            current_time=current_time,
            neighborhoods=neighborhoods,
            source_lead_minutes=source_lead_minutes,
            scheduled=scheduled,
            ignore=ignore,
            planned_volume_by_day=planned_volume_by_day,
            daily_delivery_targets=daily_delivery_targets,
            score_cutoff_minute=score_cutoff_minute,
        )
        best_cand = None
        best_score = float("-inf")
        for customer in candidates_to_try:
            if customer.index in served_this_shift:
                continue
            c = _candidate_for_customer(
                instance,
                resource,
                window,
                current_pt,
                current_time,
                driving,
                customer,
                scheduled[customer.index],
                buffer,
                score_cutoff_minute,
                terminal_buffer_days,
            )
            if c:
                score = _candidate_priority(
                    instance,
                    customer=customer,
                    candidate=c,
                    target=target,
                    current_pt=current_pt,
                    current_time=current_time,
                    neighborhoods=neighborhoods,
                    source_lead_minutes=source_lead_minutes,
                    scheduled=scheduled,
                    planned_volume_by_day=planned_volume_by_day,
                    daily_delivery_targets=daily_delivery_targets,
                    trailer_capacity=instance.trailers[resource.trailer].capacity,
                )
                if score > best_score:
                    best_cand = c
                    best_score = score
        
        if best_cand is None:
            break
            
        _apply_cand(operations, resource, scheduled, ignore, planned_volume_by_day, best_cand, instance)
        served_this_shift.add(best_cand.customer.index)
        current_pt = best_cand.customer.index
        current_time = best_cand.departure
        driving += best_cand.travel_time
        end_after_return = current_time + instance.time_matrix[current_pt][instance.base_index]

    if not operations: return None
    resource.available_time = end_after_return + driver.min_inter_shift_duration
    resource.trailer_available_time = end_after_return
    return Shift(index=shift_idx, driver=resource.driver, trailer=resource.trailer, start=start, operations=tuple(operations))


def _candidate_customer_ids(
    instance: Instance,
    target: Customer,
    *,
    current_pt: int,
    current_time: int,
    neighborhoods,
    source_lead_minutes: dict[int, int],
    scheduled,
    ignore,
    planned_volume_by_day: dict[int, float],
    daily_delivery_targets: dict[int, float],
    score_cutoff_minute: int | None,
    fill_doi_days: float = 2.5,
    max_fill: int = 16,
    max_smoothing: int = 20,
) -> list[Customer]:
    current_step = min(max(current_time // instance.unit, 0), instance.horizon - 1)
    candidate_ids: list[int] = [target.index]
    seen = {target.index}
    local_pool: list[int] = []
    ring_one: list[int] = []

    for point in (target.index, current_pt):
        for neighbor in neighborhoods.get(point, []):
            if neighbor not in seen:
                candidate_ids.append(neighbor)
                seen.add(neighbor)
            if neighbor not in local_pool:
                local_pool.append(neighbor)
            if neighbor not in ring_one:
                ring_one.append(neighbor)

    for point in ring_one:
        for neighbor in neighborhoods.get(point, []):
            if neighbor != target.index and neighbor not in local_pool:
                local_pool.append(neighbor)

    fill_candidates: list[tuple[float, float, int, int]] = []
    for customer_id in local_pool:
        customer = instance.customer_by_point[customer_id]
        if customer.call_in:
            continue
        if customer.index == target.index:
            continue
        inventory = _inventory_at_step(instance, customer, scheduled[customer.index], current_step)
        doi = days_of_inventory(
            instance,
            customer,
            inventory,
            min(current_step + 1, instance.horizon - 1),
            lead_time_minutes=source_lead_minutes[customer.index],
        )
        lower_doi, upper_doi = _service_window_days(
            instance,
            customer,
            current_inventory=inventory,
            step=current_step,
            neighborhoods=neighborhoods,
            lead_time_minutes=source_lead_minutes[customer.index],
        )
        fit_time = min(
            instance.time_matrix[target.index][customer.index],
            instance.time_matrix[current_pt][customer.index],
        )
        if lower_doi < fill_doi_days or doi < fill_doi_days:
            fill_candidates.append((lower_doi, upper_doi, fit_time, customer.index))

    fill_candidates.sort(key=lambda item: (item[0], item[2], item[1], item[3]))
    for _, _, _, customer_id in fill_candidates[:max_fill]:
        if customer_id not in seen:
            candidate_ids.append(customer_id)
            seen.add(customer_id)

    smooth_candidates = _smoothing_customer_ids(
        instance,
        scheduled=scheduled,
        current_pt=current_pt,
        current_step=current_step,
        planned_volume_by_day=planned_volume_by_day,
        daily_delivery_targets=daily_delivery_targets,
        max_count=max_smoothing,
    )
    for customer_id in smooth_candidates:
        if customer_id not in seen:
            candidate_ids.append(customer_id)
            seen.add(customer_id)

    return [instance.customer_by_point[cid] for cid in candidate_ids]


def _candidate_priority(
    instance: Instance,
    *,
    customer: Customer,
    candidate: _Candidate,
    target: Customer,
    current_pt: int,
    current_time: int,
    neighborhoods,
    source_lead_minutes: dict[int, int],
    scheduled,
    planned_volume_by_day: dict[int, float],
    daily_delivery_targets: dict[int, float],
    trailer_capacity: float,
) -> float:
    if customer.call_in:
        next_order = _next_unsatisfied_order(customer, scheduled[customer.index])
        due_slack = 0 if next_order is None else next_order[1].latest_time - candidate.arrival
        return 20_000.0 - max(0, due_slack) / 10.0 - candidate.travel_time

    current_step = min(max(current_time // instance.unit, 0), instance.horizon - 1)
    inventory = _inventory_at_step(instance, customer, scheduled[customer.index], current_step)
    doi = days_of_inventory(
        instance,
        customer,
        inventory,
        min(current_step + 1, instance.horizon - 1),
        lead_time_minutes=source_lead_minutes[customer.index],
    )
    lower_doi, upper_doi = _service_window_days(
        instance,
        customer,
        current_inventory=inventory,
        step=current_step,
        neighborhoods=neighborhoods,
        lead_time_minutes=source_lead_minutes[customer.index],
    )
    window_width = max(0.0, upper_doi - lower_doi)

    if lower_doi < 0.0:
        urgency = 12_000.0 + (-lower_doi) * 3_000.0
    elif lower_doi < 0.5:
        urgency = 9_000.0 + (0.5 - lower_doi) * 2_000.0
    elif lower_doi < 1.0:
        urgency = 7_000.0 + (1.0 - lower_doi) * 1_500.0
    else:
        urgency = max(0.0, 2.5 - lower_doi) * 700.0

    route_fit = 0.0
    if customer.index == target.index:
        route_fit += 800.0
    if customer.index in neighborhoods.get(target.index, []):
        route_fit += 400.0
    if customer.index in neighborhoods.get(current_pt, []):
        route_fit += 350.0
    route_fit += 250.0 * min(1.0, candidate.quantity / max(trailer_capacity, 1.0))
    route_fit += 180.0 * min(2.0, max(0.0, 2.0 - upper_doi))
    route_fit += _opening_tightness_bonus(instance, customer, candidate.arrival)
    route_fit += _small_tank_priority_bonus(instance, customer)
    route_fit -= 120.0 * min(2.0, window_width)
    route_fit -= 6.0 * candidate.travel_time
    route_fit -= 180.0 if candidate.source_arrival is not None else 0.0

    return (
        urgency
        + route_fit
        + _smoothing_score(
            candidate,
            planned_volume_by_day=planned_volume_by_day,
            daily_delivery_targets=daily_delivery_targets,
        )
    )


def _smoothing_customer_ids(
    instance: Instance,
    *,
    scheduled,
    current_pt: int,
    current_step: int,
    planned_volume_by_day: dict[int, float],
    daily_delivery_targets: dict[int, float],
    max_count: int,
) -> list[int]:
    candidates: list[tuple[float, int]] = []
    for customer in instance.customers:
        if customer.call_in:
            continue
        events = project_customer_inventory(instance, customer, scheduled[customer.index])
        breach = next((event for event in events[current_step:] if event.safety_breach), None)
        if breach is None:
            continue
        economic_step = _first_economic_service_step(events, customer, current_step)
        if economic_step is None:
            continue
        economic_day = economic_step * instance.unit // 1440
        breach_day = breach.time_start // 1440
        target_day = _target_service_day(economic_day, breach_day, daily_delivery_targets, planned_volume_by_day)
        day_deficit = daily_delivery_targets.get(target_day, 0.0) - planned_volume_by_day.get(target_day, 0.0)
        distance = instance.time_matrix[current_pt][customer.index]
        score = (
            target_day * 10_000.0
            + max(0.0, -day_deficit)
            + distance
            + customer.index / 10_000.0
        )
        candidates.append((score, customer.index))
    candidates.sort()
    return [customer_id for _, customer_id in candidates[:max_count]]


def _smoothing_score(
    candidate: _Candidate,
    *,
    planned_volume_by_day: dict[int, float],
    daily_delivery_targets: dict[int, float],
) -> float:
    day = candidate.arrival // 1440
    target = daily_delivery_targets.get(day)
    if target is None or target <= EPSILON:
        return 0.0
    planned = planned_volume_by_day.get(day, 0.0)
    before_gap = abs(target - planned)
    after_gap = abs(target - planned - max(0.0, candidate.quantity))
    overload = max(0.0, planned + max(0.0, candidate.quantity) - target)
    return 2.5 * (before_gap - after_gap) - 1.5 * overload


def _opening_tightness_bonus(instance: Instance, customer: Customer, arrival: int) -> float:
    horizon_minutes = max(instance.unit, instance.horizon * instance.unit)
    open_minutes = sum(
        max(0, min(window.end, horizon_minutes) - max(window.start, 0))
        for window in customer.time_windows
    )
    if open_minutes <= 0:
        return 0.0
    open_share = min(1.0, open_minutes / horizon_minutes)
    next_close = min(
        (
            window.end
            for window in customer.time_windows
            if window.start <= arrival <= window.end
        ),
        default=arrival,
    )
    close_slack_hours = max(0.0, (next_close - arrival) / 60.0)
    return 900.0 * (1.0 - open_share) + 80.0 * max(0.0, 3.0 - close_slack_hours)


def _small_tank_priority_bonus(instance: Instance, customer: Customer) -> float:
    capacities = sorted(c.capacity for c in instance.customers if not c.call_in and c.capacity > EPSILON)
    if not capacities or customer.capacity <= EPSILON:
        return 0.0
    median_capacity = capacities[len(capacities) // 2]
    size_ratio = min(4.0, median_capacity / customer.capacity)
    trailer_restriction = max(0, 3 - len(customer.allowed_trailers))
    return 450.0 * max(0.0, size_ratio - 1.0) + 180.0 * trailer_restriction


def _target_service_day(
    economic_day: int,
    breach_day: int,
    daily_delivery_targets: dict[int, float],
    planned_volume_by_day: dict[int, float],
) -> int:
    if breach_day <= economic_day:
        return breach_day
    candidate_days = range(economic_day, breach_day + 1)
    return max(
        candidate_days,
        key=lambda day: (
            daily_delivery_targets.get(day, 0.0) - planned_volume_by_day.get(day, 0.0),
            -day,
        ),
    )


def _daily_delivery_targets(instance: Instance) -> dict[int, float]:
    days = max(1, (instance.horizon * instance.unit + 1439) // 1440)
    weights = {
        day: (WEEKEND_DELIVERY_WEIGHT if day % 7 in {5, 6} else 1.0)
        for day in range(days)
    }
    total_weight = sum(weights.values()) or 1.0
    total_required = 0.0
    for customer in instance.customers:
        if customer.call_in:
            continue
        required = customer.safety_level + sum(customer.forecast) - customer.initial_tank_quantity
        total_required += max(0.0, min(customer.capacity, required))
    return {
        day: total_required * weight / total_weight
        for day, weight in weights.items()
    }


def _service_window_days(
    instance: Instance,
    customer: Customer,
    *,
    current_inventory: float,
    step: int,
    neighborhoods,
    lead_time_minutes: int,
) -> tuple[float, float]:
    base_doi = days_of_inventory(
        instance,
        customer,
        current_inventory,
        min(step + 1, instance.horizon - 1),
        lead_time_minutes=lead_time_minutes,
    )
    demand_uncertainty = _demand_uncertainty_days(
        instance,
        customer,
        current_inventory=current_inventory,
        step=step,
    )
    route_flexibility = _route_flexibility_days(instance, customer.index, neighborhoods)
    lower = base_doi - demand_uncertainty
    upper = base_doi + demand_uncertainty + route_flexibility
    return lower, upper


def _demand_uncertainty_days(
    instance: Instance,
    customer: Customer,
    *,
    current_inventory: float,
    step: int,
) -> float:
    steps_per_day = max(1, 1440 // instance.unit)
    remaining = list(customer.forecast[step:])
    if not remaining:
        return 0.0
    daily_demands = [
        sum(remaining[i:i + steps_per_day])
        for i in range(0, min(len(remaining), steps_per_day * 5), steps_per_day)
        if remaining[i:i + steps_per_day]
    ]
    if len(daily_demands) < 2:
        return 0.0
    mean_daily = sum(daily_demands) / len(daily_demands)
    if mean_daily <= EPSILON:
        return 0.0
    spread_ratio = (max(daily_demands) - min(daily_demands)) / mean_daily
    usable_inventory = max(0.0, current_inventory - customer.safety_level)
    max_daily = max(daily_demands)
    one_service_per_day_cap = max(0.0, usable_inventory / max(max_daily, EPSILON) - 1.0)
    return min(1.0, one_service_per_day_cap, 0.5 * spread_ratio)


def _route_flexibility_days(
    instance: Instance,
    point: int,
    neighborhoods,
) -> float:
    neighbor_points = neighborhoods.get(point, [])[:3]
    if not neighbor_points:
        return 0.0
    mean_minutes = sum(instance.time_matrix[point][neighbor] for neighbor in neighbor_points) / len(neighbor_points)
    return max(0.0, min(0.35, (180.0 - min(180.0, mean_minutes)) / 1440.0 * 2.0))


def _inventory_at_step(
    instance: Instance,
    customer: Customer,
    deliveries: dict[int, float],
    step: int,
) -> float:
    events = project_customer_inventory(instance, customer, deliveries)
    return events[min(step, len(events) - 1)].ending_inventory

def _apply_cand(ops, resource, scheduled, ignore, planned_volume_by_day, cand, instance):
    if cand.source_arrival is not None and cand.load_quantity > EPSILON:
        ops.append(Operation(point=cand.source_index, arrival=cand.source_arrival, quantity=-cand.load_quantity))
        resource.trailer_quantity += cand.load_quantity
    ops.append(Operation(point=cand.customer.index, arrival=cand.arrival, quantity=cand.quantity))
    resource.trailer_quantity -= cand.quantity
    scheduled[cand.customer.index][cand.arrival] = scheduled[cand.customer.index].get(cand.arrival, 0.0) + cand.quantity
    if cand.quantity > EPSILON and not cand.customer.call_in:
        day = cand.arrival // 1440
        planned_volume_by_day[day] = planned_volume_by_day.get(day, 0.0) + cand.quantity
    arrival_step = min(max(cand.arrival // instance.unit, 0), instance.horizon - 1)
    ignore[cand.customer.index] = max(ignore[cand.customer.index], arrival_step + 1)

def _candidate_for_customer(
    instance,
    resource,
    window,
    current_pt,
    current_time,
    driving,
    customer,
    deliveries,
    buffer,
    score_cutoff_minute=None,
    terminal_buffer_days=0.0,
):
    if not is_trailer_allowed(instance, customer.index, resource.trailer): return None
    if customer.call_in:
        return _candidate_for_call_in(
            instance,
            resource,
            window,
            current_pt,
            current_time,
            driving,
            customer,
            deliveries,
            score_cutoff_minute,
        )
    source = next((s for s in instance.sources if resource.trailer in s.allowed_trailers), None)
    if source is None: return None
    trailer = instance.trailers[resource.trailer]
    load_qty, source_arr, time, pt, travel, trailer_qty = 0.0, None, current_time, current_pt, 0, resource.trailer_quantity

    # Potential reload
    if trailer_qty < customer.min_operation_quantity - EPSILON:
        source_arr = time + instance.time_matrix[pt][source.index]
        if score_cutoff_minute is not None and source_arr >= score_cutoff_minute:
            return None
        time = source_arr + source.setup_time
        travel += instance.time_matrix[pt][source.index]
        pt = source.index
        load_qty = trailer.capacity - trailer_qty
        trailer_qty = trailer.capacity

    raw_arrival = time + instance.time_matrix[pt][customer.index]
    arrival = raw_arrival
    if score_cutoff_minute is not None and arrival >= score_cutoff_minute:
        return None
    total_travel = travel + instance.time_matrix[pt][customer.index]
    ret_travel = instance.time_matrix[customer.index][instance.base_index]
    
    if not is_driving_duration_valid(instance.drivers[resource.driver], driving + total_travel + ret_travel): return None

    arrival_step = min(max(arrival // instance.unit, 0), instance.horizon - 1)
    events = project_customer_inventory(instance, customer, deliveries)
    economic_step = _first_economic_service_step(events, customer, arrival_step)
    if economic_step is None:
        return None
    economic_arrival = max(arrival, economic_step * instance.unit)
    if economic_arrival - raw_arrival >= instance.drivers[resource.driver].layover_duration:
        return None
    arrival = economic_arrival
    if score_cutoff_minute is not None and arrival >= score_cutoff_minute:
        return None
    departure = arrival + customer.setup_time
    if departure + ret_travel > window.end: return None
    if not is_time_window_valid(arrival, departure, customer.time_windows): return None
    arrival_step = min(max(arrival // instance.unit, 0), instance.horizon - 1)
    inv_at_arr = events[arrival_step].after_consumption

    already_delivered = sum(
        quantity
        for delivery_arrival, quantity in deliveries.items()
        if min(max(delivery_arrival // instance.unit, 0), instance.horizon - 1) == arrival_step
    )
    
    max_room = customer.capacity - inv_at_arr - already_delivered
    if max_room < customer.min_operation_quantity - EPSILON: return None
    
    target = _target_inventory(
        instance,
        customer,
        buffer,
        arrival_step=arrival_step,
        terminal_buffer_days=terminal_buffer_days,
    )
    qty = min(trailer_qty, max_room, target - inv_at_arr - already_delivered)
    
    if qty < customer.min_operation_quantity - EPSILON:
        if max_room >= customer.min_operation_quantity - EPSILON and trailer_qty >= customer.min_operation_quantity - EPSILON:
             qty = customer.min_operation_quantity
        else: return None

    qty = _cap_quantity_without_future_overfill(instance, customer, deliveries, arrival, qty)
    if qty < customer.min_operation_quantity - EPSILON:
        return None

    return _Candidate(customer=customer, arrival=arrival, departure=departure, quantity=qty, travel_time=total_travel, source_arrival=source_arr, load_quantity=load_qty, source_index=source.index)

def _first_economic_service_step(events, customer, start_step: int) -> int | None:
    threshold = customer.capacity * ECONOMIC_SERVICE_FILL_RATIO
    for event in events[start_step:]:
        if event.after_consumption <= threshold + EPSILON:
            return event.step
    return None

def _candidate_for_call_in(
    instance,
    resource,
    window,
    current_pt,
    current_time,
    driving,
    customer,
    deliveries,
    score_cutoff_minute=None,
):
    order_info = _next_unsatisfied_order(customer, deliveries, score_cutoff_minute)
    if order_info is None:
        return None
    _, order = order_info
    source = next((s for s in instance.sources if resource.trailer in s.allowed_trailers), None)
    if source is None:
        return None

    delivered = sum(
        quantity
        for arrival, quantity in deliveries.items()
        if order.earliest_time <= arrival <= order.latest_time
    )
    remaining = order.min_quantity_to_satisfy - delivered
    if remaining <= EPSILON:
        return None

    trailer = instance.trailers[resource.trailer]
    trailer_qty = resource.trailer_quantity
    load_qty = 0.0
    source_arr = None
    time = current_time
    point = current_pt
    travel = 0
    if trailer_qty < min(remaining, trailer.capacity) - EPSILON:
        source_arr = time + instance.time_matrix[point][source.index]
        if score_cutoff_minute is not None and source_arr >= score_cutoff_minute:
            return None
        time = source_arr + source.setup_time
        travel += instance.time_matrix[point][source.index]
        point = source.index
        load_qty = trailer.capacity - trailer_qty
        trailer_qty = trailer.capacity

    raw_arrival = time + instance.time_matrix[point][customer.index]
    arrival = max(raw_arrival, order.earliest_time)
    if arrival - time >= instance.drivers[resource.driver].layover_duration + instance.time_matrix[point][customer.index]:
        return None
    if score_cutoff_minute is not None and arrival >= score_cutoff_minute:
        return None
    if arrival > order.latest_time:
        return None
    departure = arrival + customer.setup_time
    total_travel = travel + instance.time_matrix[point][customer.index]
    ret_travel = instance.time_matrix[customer.index][instance.base_index]

    if departure + ret_travel > window.end:
        return None
    if not is_driving_duration_valid(instance.drivers[resource.driver], driving + total_travel + ret_travel):
        return None
    if not is_time_window_valid(arrival, departure, customer.time_windows):
        return None

    qty = min(trailer_qty, remaining)
    if qty <= EPSILON:
        return None
    return _Candidate(
        customer=customer,
        arrival=arrival,
        departure=departure,
        quantity=qty,
        travel_time=total_travel,
        source_arrival=source_arr,
        load_quantity=load_qty,
        source_index=source.index,
    )

def _cap_quantity_without_future_overfill(instance, customer, deliveries, arrival, quantity):
    capped = quantity
    for _ in range(10):
        proposed = dict(deliveries)
        proposed[arrival] = proposed.get(arrival, 0.0) + capped
        overflow = max(
            (
                event.ending_inventory - customer.capacity
                for event in project_customer_inventory(instance, customer, proposed)
            ),
            default=0.0,
        )
        if overflow <= EPSILON:
            return capped
        capped -= overflow
        if capped <= EPSILON:
            return 0.0
    return max(0.0, capped)

def _target_inventory(
    instance,
    customer,
    buffer,
    *,
    arrival_step: int | None = None,
    terminal_buffer_days: float = 0.0,
):
    if arrival_step is not None:
        remaining_forecast = sum(customer.forecast[arrival_step:])
        steps_per_day = max(1, 1440 // instance.unit)
        buffer_steps = max(0, int(round(terminal_buffer_days * steps_per_day)))
        if buffer_steps:
            tail = list(customer.forecast[-steps_per_day:]) or [0.0]
            remaining_forecast += sum(tail[i % len(tail)] for i in range(buffer_steps))
        return min(customer.capacity, customer.safety_level + remaining_forecast)
    demand = sum(customer.forecast) / max(instance.horizon / 24.0, 1.0)
    return min(customer.capacity, customer.capacity - buffer * demand)

def _initial_resources(instance):
    drivers = [_DriverState(driver=driver.index) for driver in instance.drivers]
    trailers = [
        _TrailerState(
            trailer=trailer.index,
            trailer_quantity=trailer.initial_quantity,
        )
        for trailer in instance.trailers
    ]
    return drivers, trailers

def _first_breach_step(instance, customer, deliveries, min_step = 0):
    for e in project_customer_inventory(instance, customer, deliveries):
        if e.step >= min_step and e.safety_breach: return e.step
    return None

def _resource_window_candidates(instance, drivers, trailers, score_cutoff_minute=None):
    cands = []
    for di, driver_state in enumerate(drivers):
        d = instance.drivers[driver_state.driver]
        for wi in range(driver_state.next_window_index, len(d.time_windows)):
            w = d.time_windows[wi]
            has_candidate_for_window = False
            for ti, trailer_state in enumerate(trailers):
                if trailer_state.trailer not in d.trailer_ids:
                    continue
                s = max(w.start, driver_state.available_time, trailer_state.available_time)
                if score_cutoff_minute is not None and s >= score_cutoff_minute:
                    continue
                if s <= w.end:
                    resource = _ResourceState(
                        driver=driver_state.driver,
                        trailer=trailer_state.trailer,
                        trailer_quantity=trailer_state.trailer_quantity,
                        available_time=s,
                        trailer_available_time=s,
                    )
                    cands.append((s, di, ti, wi, w, resource))
                    has_candidate_for_window = True
            if has_candidate_for_window:
                break
    return sorted(cands, key=lambda i: (i[0], i[1], i[2]))
