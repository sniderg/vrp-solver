from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Dict, List, Set, Tuple

from ..inventory import (
    days_of_inventory,
    project_customer_inventory,
    project_customer_inventory_arrays,
    tank_events,
)
from ..model import Customer, Instance, Operation, Shift, Solution, TimeWindow
from ..rules import (
    is_driving_duration_valid,
    is_time_window_valid,
    is_trailer_allowed,
    validate_solution,
)
from ..movement import nearest_neighbors


EPSILON = 1e-6
# Cold-start routes must build enough inventory headroom to survive the next
# resource window.  At 0.75 the constructor deferred service too long and
# then made many small, capacity-limited deliveries on Set B instances.
ECONOMIC_SERVICE_FILL_RATIO = 0.90
WEEKEND_DELIVERY_WEIGHT = 0.65
# A route that waits until its remaining trailer stock is below the *next*
# customer's minimum operation before reloading tends to strand otherwise
# compatible customers in later shifts.  The original cold starts make
# deliberate mid-route top-ups; retain enough stock for a short detour, but
# top up once the trailer is materially depleted so the current cluster can
# be completed in the same driver window.
PROACTIVE_RELOAD_RATIO = 0.40

@dataclass(frozen=True)
class ConstructionReport:
    shifts: int
    operations: int
    delivered_quantity: float
    unscheduled_customers: tuple[int, ...]
    exhausted_resources: bool
    attempts: int = 1

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
    driving_after: int
    layover_before: bool
    return_layover: bool
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
    max_smoothing: int = 0,
    global_pressure_fill: int = 0,
    global_pressure_offset: int = 0,
    tie_break_seed: int = 0,
    limit_reload_after_empty_start: bool = False,
    terminal_preload: bool = True,
    first_stop_targeted: bool = True,
    prioritize_early_callins: bool = True,
) -> tuple[Solution, ConstructionReport]:
    drivers, trailers = _initial_resources(instance)
    scheduled: dict[int, dict[int, float]] = {customer.index: {} for customer in instance.customers}
    ignore_before_step: dict[int, int] = {customer.index: 0 for customer in instance.customers}
    planned_volume_by_day: dict[int, float] = {}
    daily_delivery_targets = _daily_delivery_targets(instance)
    # Candidate ranking asks for the same customer inventory repeatedly while
    # evaluating a route.  A customer's projection only changes when that
    # customer's delivery dictionary changes, so retain the exact ending
    # levels for the duration of this construction.
    inventory_cache: dict[tuple[int, tuple[tuple[int, float], ...]], tuple[float, ...]] = {}
    shifts: list[Shift] = []
    exhausted_resources = False

    neighborhoods = _compute_neighborhoods(instance, k=neighborhood_size)
    tie_break = None
    if tie_break_seed:
        tie_rng = random.Random(tie_break_seed)
        tie_break = {customer.index: tie_rng.random() for customer in instance.customers}
    source_lead_minutes = {
        customer.index: min(
            instance.time_matrix[source.index][customer.index]
            for source in instance.sources
        )
        for customer in instance.customers
    }

    while True:
        if max_shifts is not None and len(shifts) >= max_shifts:
            exhausted_resources = len(_all_needs(instance, scheduled, ignore_before_step, score_cutoff_minute, inventory_cache)) > 0
            break
        
        all_needs = _all_needs(instance, scheduled, ignore_before_step, score_cutoff_minute, inventory_cache)
        if not all_needs:
            break
        # The first few days contain contractual call-ins whose windows cannot
        # be recovered once a flexible VMI route has consumed the compatible
        # driver/trailer.  Reserve those tight early commitments before the
        # normal safety-breach ordering, but do not globally starve VMI for
        # later-horizon call-ins.
        early_callin_limit = min(
            score_cutoff_minute if score_cutoff_minute is not None else 3 * 1440,
            3 * 1440,
        )
        callin_driver_scarcity = {
            customer.index: sum(
                1
                for driver in instance.drivers
                if set(driver.trailer_ids) & set(customer.allowed_trailers)
            )
            for customer, _ in all_needs
            if customer.call_in
        }
        # Breaches separated by only one or two planning buckets compete for
        # the same outgoing shift.  Treat that as a tie and favour the smaller
        # tank: it has less recovery room if a neighbouring, larger tank gets
        # the route first.  V2.14's C11/C16 pair is the motivating case.
        first_vmi_breach = min(
            (
                breach_step
                for customer, breach_step in all_needs
                if not customer.call_in
            ),
            default=0,
        )
        all_needs.sort(
            key=lambda item: (
                (
                    0
                    if (
                        not prioritize_early_callins
                        and not item[0].call_in
                    )
                    else (
                        0
                        if (
                            prioritize_early_callins
                            and item[0].call_in
                            and (item[1] + 1) * instance.unit <= early_callin_limit
                        )
                        else 1
                    )
                ),
                callin_driver_scarcity.get(item[0].index, len(instance.drivers) + 1),
                (
                    0
                    if item[0].call_in
                    else max(0, item[1] - first_vmi_breach - 2)
                ),
                item[1] if item[0].call_in else item[0].capacity,
                item[1],
                0.0 if tie_break is None else tie_break[item[0].index],
                item[0].index,
            )
        )
        ranked_vmi_pressure = tuple(
            customer.index
            for customer, _ in all_needs
            if not customer.call_in
        )
        offset = max(0, global_pressure_offset)
        global_pressure_ids = ranked_vmi_pressure[
            offset:offset + max(0, global_pressure_fill)
        ]

        next_window_infos = _resource_window_candidates(instance, drivers, trailers, score_cutoff_minute)
        if not next_window_infos:
            exhausted_resources = True
            break

        shift = None
        selected_driver_state_index = None
        selected_trailer_state_index = None
        selected_resource = None
        for start_time, driver_state_index, trailer_state_index, window_index, window, resource in next_window_infos:
            target_needs = all_needs
            for target_customer, breach_step in target_needs:
                # An expired call-in cannot be repaired by a late delivery,
                # but an overdue VMI tank still needs service to stop the
                # accumulating runout.  Skipping both kinds made a cold start
                # abandon dozens of tanks after the early call-in reservation.
                if (
                    target_customer.call_in
                    and (breach_step + 1) * instance.unit < start_time
                ):
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
                    inventory_cache,
                    max_smoothing,
                    global_pressure_ids,
                    limit_reload_after_empty_start,
                    terminal_preload,
                    first_stop_targeted,
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
        if (
            _next_unsatisfied_order(
                customer,
                scheduled[customer.index],
                score_cutoff_minute,
            )
            is not None
            if customer.call_in
            else _first_breach_step(instance, customer, scheduled[customer.index], 0)
            is not None
        )
    )
    
    solution = Solution(shifts=tuple(shifts))
    return solution, ConstructionReport(
        shifts=len(solution.shifts),
        operations=sum(len(shift.operations) for shift in solution.shifts),
        delivered_quantity=sum(op.quantity for s in solution.shifts for op in s.operations if op.quantity > 0),
        unscheduled_customers=unscheduled,
        exhausted_resources=exhausted_resources,
    )

def _all_needs(instance, scheduled, ignore_before_step, score_cutoff_minute=None, inventory_cache=None):
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
            cache=inventory_cache,
        )
        if breach is not None:
            needs.append((customer, breach))
    return needs

def _next_unsatisfied_order(customer, deliveries, score_cutoff_minute=None):
    """Return the next order whose required flexible minimum is outstanding."""
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


def _latest_vmi_service_arrival(
    instance: Instance,
    customer: Customer,
    breach_step: int,
) -> int:
    """Latest legal arrival that can still prevent this projected breach.

    A raw inventory-breach step is not a sufficient dispatch deadline.  A
    tank may have a much earlier closing customer window, in which case an
    apparently later breach has to be handled first.  This is particularly
    important for B instances with short recurring windows.
    """
    latest_by_inventory = min(
        instance.horizon * instance.unit - 1,
        (breach_step + 1) * instance.unit - 1,
    )
    candidates = [
        min(latest_by_inventory, window.end - customer.setup_time)
        for window in customer.time_windows
        if window.start <= latest_by_inventory
        and window.end - customer.setup_time >= window.start
    ]
    return max(candidates, default=latest_by_inventory)

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
    inventory_cache,
    max_smoothing,
    global_pressure_ids,
    limit_reload_after_empty_start,
    terminal_preload,
    first_stop_targeted,
):
    driver = instance.drivers[resource.driver]
    start = max(window.start, resource.available_time)
    
    operations = []
    current_pt = instance.base_index
    current_time = start
    driving = 0
    layover_used = False
    return_layover = False
    end_after_return = start

    served_this_shift = set()
    # A trailer which began the shift empty has already paid the source detour
    # needed to make its first cluster viable.  Chaining a second reload for
    # a marginal filler can consume the remaining driver window and strand a
    # much more urgent tank before the next window.  This is not a ban on the
    # terminal preload (which stages stock for the following shift); it only
    # prevents a non-target, mid-route second load after the mandatory first
    # load.  V2.14 with trailer 3 empty exposes this distinction.
    started_with_source_reload = False
    trailer_started_empty = resource.trailer_quantity <= EPSILON

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
            inventory_cache=inventory_cache,
            max_smoothing=max_smoothing,
            global_pressure_ids=global_pressure_ids,
        )
        # The outer constructor is assigning this route to satisfy ``target``.
        # Accepting an unrelated filler as its first stop consumes this
        # resource window and prevents the outer loop from trying a compatible
        # resource.  That was first visible for V2.12 call-ins, but it is just
        # as destructive for an ordinary VMI tank whose first safety breach is
        # imminent: the historical breach cannot be repaired by serving a
        # different customer now.
        if not operations and first_stop_targeted:
            candidates_to_try = [target]
            # A few tanks (V2.14 customer 9 is a representative example) are
            # not reachable from base and back within one driving spell, even
            # though a nearby layover-eligible customer makes a legal route to
            # them.  Permit only those local layover access points before an
            # otherwise unreachable VMI target; the next iteration still gives
            # the original target its decisive priority.
            if not target.call_in:
                # This is deliberately a *geometric* lookup.  Calling the
                # general filler generator here also invokes its global
                # smoothing pass (full-horizon inventory projections for all
                # tanks) merely to find a legal local rest point.  On a cold
                # construction that was both costly and irrelevant: an
                # accessor must be close to the target and layover eligible,
                # not an economically attractive filler.
                accessor_ids = [
                    point
                    for point in neighborhoods.get(target.index, [])
                    if point in instance.customer_by_point
                    and instance.customer_by_point[point].layover_customer
                ]
                accessors = [instance.customer_by_point[point] for point in accessor_ids]
                candidates_to_try.extend(
                    candidate
                    for candidate in accessors
                    if candidate.index != target.index
                    and not candidate.call_in
                    and candidate.layover_customer
                )
        best_cand = None
        best_score = float("-inf")
        best_start_time = current_time
        for customer in candidates_to_try:
            if customer.index in served_this_shift:
                continue
            candidate_start_time = current_time
            if not operations and customer.call_in:
                candidate_start_time = _first_callin_departure_time(
                    instance,
                    resource,
                    window,
                    customer,
                    scheduled[customer.index],
                    current_time,
                    score_cutoff_minute,
                )
            c = _candidate_for_customer(
                instance,
                resource,
                window,
                current_pt,
                candidate_start_time,
                driving,
                customer,
                scheduled[customer.index],
                buffer,
                score_cutoff_minute,
                terminal_buffer_days,
                has_layover_customer=any(
                    op.point in instance.customer_by_point
                    and instance.customer_by_point[op.point].layover_customer
                    for op in operations
                ),
                layover_used=layover_used,
            )
            if c:
                if (
                    limit_reload_after_empty_start
                    and started_with_source_reload
                    and c.source_arrival is not None
                ):
                    continue
                score = _candidate_priority(
                    instance,
                    customer=customer,
                    candidate=c,
                    target=target,
                    current_pt=current_pt,
                    current_time=candidate_start_time,
                    neighborhoods=neighborhoods,
                    source_lead_minutes=source_lead_minutes,
                    scheduled=scheduled,
                    planned_volume_by_day=planned_volume_by_day,
                    daily_delivery_targets=daily_delivery_targets,
                    trailer_capacity=instance.trailers[resource.trailer].capacity,
                    is_first_service=not operations,
                    prefer_target_first=first_stop_targeted,
                    inventory_cache=inventory_cache,
                )
                if score > best_score:
                    best_cand = c
                    best_score = score
                    best_start_time = candidate_start_time
        
        if best_cand is None:
            break

        if not operations and best_start_time != current_time:
            start = best_start_time
            current_time = best_start_time
            
        first_service = not operations
        _apply_cand(operations, resource, scheduled, ignore, planned_volume_by_day, best_cand, instance)
        if first_service and trailer_started_empty and best_cand.source_arrival is not None:
            started_with_source_reload = True
        served_this_shift.add(best_cand.customer.index)
        current_pt = best_cand.customer.index
        current_time = best_cand.departure
        driving = best_cand.driving_after
        layover_used = layover_used or best_cand.layover_before
        return_layover = best_cand.return_layover
        end_after_return = current_time + instance.time_matrix[current_pt][instance.base_index]
        if return_layover:
            if best_cand.customer.layover_customer:
                # Keep driving through the target cluster after taking the
                # legal rest at this eligible stop.  The wait is represented
                # by the gap before the following operation, which is how the
                # checker derives a layover.  If no further stop fits, the
                # ordinary return-time validation remains conservative.
                current_time += driver.layover_duration
                driving = 0
                layover_used = True
                return_layover = False
                end_after_return = current_time + instance.time_matrix[current_pt][instance.base_index]
                continue
            end_after_return += driver.layover_duration
            break

    if not operations: return None
    if not return_layover and terminal_preload:
        end_after_return = _preload_terminal_source(
            instance,
            resource,
            window,
            operations,
            current_pt,
            current_time,
            driving,
            score_cutoff_minute,
        )
    resource.available_time = end_after_return + driver.min_inter_shift_duration
    resource.trailer_available_time = end_after_return
    return Shift(index=shift_idx, driver=resource.driver, trailer=resource.trailer, start=start, operations=tuple(operations))


def _first_callin_departure_time(
    instance: Instance,
    resource: _ResourceState,
    window: TimeWindow,
    customer: Customer,
    deliveries: dict[int, float],
    current_time: int,
    score_cutoff_minute: int | None,
) -> int:
    """Delay a first call-in departure instead of waiting at the customer.

    Waiting across a long call-in release is interpreted as a layover unless
    the customer supports one.  A cold-start truck is still at base, so it can
    leave later and arrive at the same legal instant without inventing rest.
    """
    order_info = _next_unsatisfied_order(customer, deliveries, score_cutoff_minute)
    if order_info is None:
        return current_time
    _, order = order_info
    trailer = instance.trailers[resource.trailer]
    if resource.trailer_quantity + EPSILON >= order.min_quantity_to_satisfy:
        travel = instance.time_matrix[instance.base_index][customer.index]
    else:
        source = next(
            (item for item in instance.sources if resource.trailer in item.allowed_trailers),
            None,
        )
        if source is None:
            return current_time
        travel = (
            instance.time_matrix[instance.base_index][source.index]
            + source.setup_time
            + instance.time_matrix[source.index][customer.index]
        )
    delayed = max(current_time, window.start, order.earliest_time - travel)
    return delayed if delayed <= window.end else current_time


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
    inventory_cache: dict[tuple[int, tuple[tuple[int, float], ...]], tuple[float, ...]] | None = None,
    fill_doi_days: float = 2.5,
    max_fill: int = 16,
    max_smoothing: int = 0,
    global_pressure_ids: tuple[int, ...] = (),
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
        inventory = _inventory_at_step(
            instance, customer, scheduled[customer.index], current_step, cache=inventory_cache,
        )
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

    # The outer planner already paid to rank the tanks needing service.  Let a
    # route use a small prefix of that ranking as possible additional stops
    # instead of rerunning a full-horizon projection for every tank after each
    # delivery.  This is deliberately optional: the V2.14 default retains its
    # previously validated local-only construction.
    for customer_id in global_pressure_ids:
        if customer_id not in seen and customer_id in instance.customer_by_point:
            candidate_ids.append(customer_id)
            seen.add(customer_id)

    if max_smoothing:
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
    is_first_service: bool,
    prefer_target_first: bool = True,
    inventory_cache: dict[tuple[int, tuple[tuple[int, float], ...]], tuple[float, ...]] | None = None,
) -> float:
    if customer.call_in:
        next_order = _next_unsatisfied_order(customer, scheduled[customer.index])
        due_slack = 0 if next_order is None else next_order[1].latest_time - candidate.arrival
        score = 20_000.0 - max(0, due_slack) / 10.0 - candidate.travel_time
        # In opportunity-first mode, a future-release call-in should not
        # consume an otherwise useful early departure.  Once a route is under
        # way it regains its normal deadline priority and can be served after
        # a reload, as in the recovered day-one construction cadence.
        if not prefer_target_first and is_first_service:
            score = -candidate.travel_time
        if (
            prefer_target_first
            and is_first_service
            and customer.index == target.index
        ):
            score += 100_000.0
        return score

    current_step = min(max(current_time // instance.unit, 0), instance.horizon - 1)
    inventory = _inventory_at_step(
        instance, customer, scheduled[customer.index], current_step, cache=inventory_cache,
    )
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

    score = (
        urgency
        + route_fit
        + _smoothing_score(
            candidate,
            planned_volume_by_day=planned_volume_by_day,
            daily_delivery_targets=daily_delivery_targets,
        )
    )
    # The outer constructor selected ``target`` as the earliest feasible
    # safety breach for this resource.  Do not let a fill-in customer displace
    # it before that critical service has happened.
    if (
        prefer_target_first
        and is_first_service
        and customer.index == target.index
    ):
        score += 100_000.0
    return score


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
    *,
    cache: dict[tuple[int, tuple[tuple[int, float], ...]], tuple[float, ...]] | None = None,
) -> float:
    key = (customer.index, tuple(sorted(deliveries.items())))
    projection = cache.get(key) if cache is not None else None
    if projection is None:
        levels_array, breaches_array = project_customer_inventory_arrays(instance, customer, deliveries)
        projection = (tuple(levels_array), tuple(bool(value) for value in breaches_array))
        if cache is not None:
            cache[key] = projection
    levels, _ = projection
    return levels[min(step, len(levels) - 1)]

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


def _preload_terminal_source(
    instance: Instance,
    resource: _ResourceState,
    window: TimeWindow,
    operations: list[Operation],
    current_pt: int,
    current_time: int,
    driving: int,
    score_cutoff_minute: int | None,
) -> int:
    """Finish at a source when it can safely stage a full trailer for next shift.

    The original executable makes substantial use of inventory carried between
    shifts.  Returning a loaded trailer is valid in the ROADEF model and avoids
    forcing every new shift to begin with a source detour.
    """
    driver = instance.drivers[resource.driver]
    trailer = instance.trailers[resource.trailer]
    load_quantity = trailer.capacity - resource.trailer_quantity
    direct_end = current_time + instance.time_matrix[current_pt][instance.base_index]
    if load_quantity <= EPSILON:
        return direct_end

    source_candidates = sorted(
        (
            source
            for source in instance.sources
            if resource.trailer in source.allowed_trailers
        ),
        key=lambda source: (
            instance.time_matrix[current_pt][source.index]
            + instance.time_matrix[source.index][instance.base_index],
            source.index,
        ),
    )
    for source in source_candidates:
        source_arrival = current_time + instance.time_matrix[current_pt][source.index]
        if score_cutoff_minute is not None and source_arrival >= score_cutoff_minute:
            continue
        end = source_arrival + source.setup_time + instance.time_matrix[source.index][instance.base_index]
        if end > window.end:
            continue
        source_driving = driving + instance.time_matrix[current_pt][source.index]
        total_driving = source_driving + instance.time_matrix[source.index][instance.base_index]
        if not is_driving_duration_valid(driver, total_driving):
            continue
        operations.append(Operation(point=source.index, arrival=source_arrival, quantity=-load_quantity))
        resource.trailer_quantity += load_quantity
        return end
    return direct_end

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
    *,
    has_layover_customer=False,
    layover_used=False,
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
            has_layover_customer=has_layover_customer,
            layover_used=layover_used,
        )
    source = next((s for s in instance.sources if resource.trailer in s.allowed_trailers), None)
    if source is None: return None
    trailer = instance.trailers[resource.trailer]
    load_qty, source_arr, time, pt, travel, trailer_qty = 0.0, None, current_time, current_pt, 0, resource.trailer_quantity

    # Potential reload.  A purely reactive reload rule produced many short
    # V2 routes: it served the next customer with the last residual stock,
    # then had insufficient driving/window slack to revisit the source and
    # continue through the nearby cluster.  Top up at a legal source once the
    # trailer is substantially depleted.  The quantity calculation below and
    # all normal timing/driving checks still decide whether this detour is
    # actually usable.
    needs_reload = trailer_qty < customer.min_operation_quantity - EPSILON
    proactive_reload = (
        current_pt != instance.base_index
        and trailer_qty < trailer.capacity * PROACTIVE_RELOAD_RATIO - EPSILON
    )
    if needs_reload or proactive_reload:
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
    
    arrival_step = min(max(arrival // instance.unit, 0), instance.horizon - 1)
    events = project_customer_inventory(instance, customer, deliveries)
    economic_step = _first_economic_service_step(events, customer, arrival_step)
    if economic_step is None:
        return None
    economic_arrival = max(arrival, economic_step * instance.unit)
    arrival = economic_arrival
    arrival = _align_arrival_to_customer_window(customer, arrival)
    if arrival is None:
        return None
    if score_cutoff_minute is not None and arrival >= score_cutoff_minute:
        return None
    departure = arrival + customer.setup_time
    if departure + ret_travel > window.end: return None
    if not is_time_window_valid(arrival, departure, customer.time_windows): return None

    layover_before = arrival - raw_arrival >= instance.drivers[resource.driver].layover_duration
    return_layover = False
    if layover_before:
        if layover_used or not (has_layover_customer or customer.layover_customer):
            return None
        # A reload is an operation before the rest, so it must itself be
        # reachable within the current driving spell.
        driving_before_last_leg = driving + travel
        if not is_driving_duration_valid(instance.drivers[resource.driver], driving_before_last_leg):
            return None
        last_leg = instance.time_matrix[pt][customer.index]
        driving_before_layover = min(
            max(0, instance.drivers[resource.driver].max_driving_duration - driving_before_last_leg),
            last_leg,
        )
        driving_after = last_leg - driving_before_layover
        if not is_driving_duration_valid(instance.drivers[resource.driver], driving_after + ret_travel):
            return None
    else:
        driving_after = driving + total_travel
        if not is_driving_duration_valid(instance.drivers[resource.driver], driving_after + ret_travel):
            if (
                not layover_used
                and (has_layover_customer or customer.layover_customer)
                and is_driving_duration_valid(instance.drivers[resource.driver], driving_after)
                and departure + ret_travel + instance.drivers[resource.driver].layover_duration <= window.end
            ):
                return_layover = True
            else:
                return None
    arrival_step = min(max(arrival // instance.unit, 0), instance.horizon - 1)
    inv_at_arr = events[arrival_step].after_consumption

    already_delivered = sum(
        quantity
        for delivery_arrival, quantity in deliveries.items()
        if min(max(delivery_arrival // instance.unit, 0), instance.horizon - 1) == arrival_step
    )
    
    # If an earlier scheduling miss has already taken the projected tank below
    # zero, ``capacity - inventory`` is larger than a physical tank.  A single
    # operation may never exceed the tank capacity, even though a later repair
    # route could otherwise use that artificial room.
    max_room = min(customer.capacity, customer.capacity - inv_at_arr - already_delivered)
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

    return _Candidate(customer=customer, arrival=arrival, departure=departure, quantity=qty, travel_time=total_travel, driving_after=driving_after, layover_before=layover_before, return_layover=return_layover, source_arrival=source_arr, load_quantity=load_qty, source_index=source.index)

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
    *,
    has_layover_customer=False,
    layover_used=False,
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
    arrival = _align_arrival_to_customer_window(customer, arrival)
    if arrival is None:
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
    if not is_time_window_valid(arrival, departure, customer.time_windows):
        return None

    layover_before = arrival - raw_arrival >= instance.drivers[resource.driver].layover_duration
    return_layover = False
    if layover_before:
        if layover_used or not (has_layover_customer or customer.layover_customer):
            return None
        driving_before_last_leg = driving + travel
        if not is_driving_duration_valid(instance.drivers[resource.driver], driving_before_last_leg):
            return None
        last_leg = instance.time_matrix[point][customer.index]
        driving_before_layover = min(
            max(0, instance.drivers[resource.driver].max_driving_duration - driving_before_last_leg),
            last_leg,
        )
        driving_after = last_leg - driving_before_layover
        if not is_driving_duration_valid(instance.drivers[resource.driver], driving_after + ret_travel):
            return None
    else:
        driving_after = driving + total_travel
        if not is_driving_duration_valid(instance.drivers[resource.driver], driving_after + ret_travel):
            if (
                not layover_used
                and (has_layover_customer or customer.layover_customer)
                and is_driving_duration_valid(instance.drivers[resource.driver], driving_after)
                and departure + ret_travel + instance.drivers[resource.driver].layover_duration <= window.end
            ):
                return_layover = True
            else:
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
        driving_after=driving_after,
        layover_before=layover_before,
        return_layover=return_layover,
        source_arrival=source_arr,
        load_quantity=load_qty,
        source_index=source.index,
    )


def _align_arrival_to_customer_window(customer: Customer, arrival: int) -> int | None:
    """Return the first customer-opening time reachable at or after arrival."""
    for window in customer.time_windows:
        candidate = max(arrival, window.start)
        if candidate + customer.setup_time <= window.end:
            return candidate
    return None

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
        # A cold start should use the available trailer capacity while a route
        # is already visiting a tank.  Planning only for the remaining demand
        # made the constructor emit hundreds of tiny Set B deliveries and run
        # out of driver windows.  The quantity cap below still prevents any
        # projected overfill.
        return customer.capacity
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

def _first_breach_step(instance, customer, deliveries, min_step=0, *, cache=None):
    key = (customer.index, tuple(sorted(deliveries.items())))
    projection = cache.get(key) if cache is not None else None
    if projection is None:
        levels_array, breaches_array = project_customer_inventory_arrays(instance, customer, deliveries)
        projection = (tuple(levels_array), tuple(bool(value) for value in breaches_array))
        if cache is not None:
            cache[key] = projection
    _, breaches = projection
    for step in range(max(0, min_step), len(breaches)):
        if breaches[step]:
            return step
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


# The following constructor implements Section 5.2 of Su et al. (2020),
# “A Matheuristic Algorithm for the Inventory Routing Problem.”  It is kept
# separate from ``construct_cluster_solution`` because the latter is an
# intentionally more prescriptive native policy developed during recovery.
# This is the reproducible paper-faithful cold-start baseline: its job is to
# produce a feasible starting point, not to reproduce later local search.
def construct_paper_solution(
    instance: Instance,
    *,
    seed: int = 1,
    retries: int = 100,
    selection_range: float = 1.5,
    refill_coefficient: float = 2.0,
    candidate_pool_size: int = 64,
    economic_urgency_minutes: int = 0,
) -> tuple[Solution, ConstructionReport]:
    """Construct a cold start following Su et al. (2020), Section 5.2.

    Shifts are built chronologically.  Each route receives a randomly chosen
    compatible available driver/trailer pair; reachable customers are ranked
    by projected shortage and a random member of the top ``tau * k`` is
    selected.  Refilling is probabilistic, deliveries use an order-up-to
    level, and economically premature VMI visits are deferred.  The complete
    construction is restarted until it is feasible, exactly as described in
    the paper.  The official validator, rather than Solver.exe, is the
    acceptance criterion.
    """
    if retries < 1:
        raise ValueError("retries must be at least one")
    if selection_range < 1.0:
        raise ValueError("selection_range must be at least one")
    if candidate_pool_size < 1:
        raise ValueError("candidate_pool_size must be at least one")
    if economic_urgency_minutes < 0:
        raise ValueError("economic_urgency_minutes must be non-negative")

    best_solution: Solution | None = None
    best_report: ConstructionReport | None = None
    best_error_count = float("inf")
    for attempt in range(retries):
        solution, report = _construct_paper_attempt(
            instance,
            rng=random.Random(seed + attempt),
            selection_range=selection_range,
            refill_coefficient=refill_coefficient,
            candidate_pool_size=candidate_pool_size,
            economic_urgency_minutes=economic_urgency_minutes,
        )
        violations = validate_solution(instance, solution)
        errors = sum(item.severity == "error" for item in violations)
        report = replace(report, attempts=attempt + 1)
        if errors < best_error_count:
            best_solution, best_report, best_error_count = solution, report, errors
        if errors == 0:
            return solution, report

    assert best_solution is not None and best_report is not None
    return best_solution, best_report


def _construct_paper_attempt(
    instance: Instance,
    *,
    rng: random.Random,
    selection_range: float,
    refill_coefficient: float,
    candidate_pool_size: int,
    economic_urgency_minutes: int,
) -> tuple[Solution, ConstructionReport]:
    drivers, trailers = _initial_resources(instance)
    scheduled: dict[int, dict[int, float]] = {customer.index: {} for customer in instance.customers}
    inventory_cache: dict[tuple[int, tuple[tuple[int, float], ...]], tuple] = {}
    shifts: list[Shift] = []
    exhausted = False
    # A conservative hard ceiling prevents an accidental empty-route loop on
    # malformed input while remaining well above the number of legal resource
    # windows on the challenge instances.
    # A single long driver window can legally contain multiple shifts.  Bound
    # only the recovery loop, not the number of shifts per availability
    # window; the normal resource-time progression remains the real limit.
    max_shifts = max(1, sum(len(driver.time_windows) for driver in instance.drivers) * max(4, len(instance.customers)))

    while len(shifts) < max_shifts:
        if not _paper_unsatisfied(instance, scheduled, inventory_cache):
            break
        windows = _resource_window_candidates(instance, drivers, trailers)
        if not windows:
            exhausted = True
            break
        earliest = windows[0][0]
        available = [item for item in windows if item[0] == earliest]
        start, driver_i, trailer_i, window_i, window, resource = rng.choice(available)
        shift = _build_paper_shift(
            instance,
            resource,
            window,
            len(shifts),
            scheduled,
            inventory_cache,
            rng=rng,
            selection_range=selection_range,
            refill_coefficient=refill_coefficient,
            candidate_pool_size=candidate_pool_size,
            economic_urgency_minutes=economic_urgency_minutes,
        )
        if shift is None:
            # No legal next operation for this pair in this driver window.
            # Move it forward; other chronological resources remain eligible.
            drivers[driver_i].next_window_index = max(drivers[driver_i].next_window_index, window_i + 1)
            continue
        drivers[driver_i].available_time = resource.available_time
        trailers[trailer_i].available_time = resource.trailer_available_time
        trailers[trailer_i].trailer_quantity = resource.trailer_quantity
        shifts.append(shift)
    else:
        exhausted = True

    unscheduled = tuple(customer.index for customer in _paper_unsatisfied(instance, scheduled, inventory_cache))
    solution = Solution(shifts=tuple(shifts))
    return solution, ConstructionReport(
        shifts=len(shifts),
        operations=sum(len(shift.operations) for shift in shifts),
        delivered_quantity=sum(
            operation.quantity
            for shift in shifts
            for operation in shift.operations
            if operation.quantity > 0
        ),
        unscheduled_customers=unscheduled,
        exhausted_resources=exhausted,
    )


def _paper_events(
    instance: Instance,
    customer: Customer,
    deliveries: dict[int, float],
    cache: dict[tuple[int, tuple[tuple[int, float], ...]], tuple],
) -> tuple:
    key = (customer.index, tuple(sorted(deliveries.items())))
    if key not in cache:
        cache[key] = tuple(project_customer_inventory(instance, customer, deliveries))
    return cache[key]


def _paper_unsatisfied(
    instance: Instance,
    scheduled: dict[int, dict[int, float]],
    inventory_cache: dict[tuple[int, tuple[tuple[int, float], ...]], tuple],
) -> list[Customer]:
    unsatisfied: list[Customer] = []
    for customer in instance.customers:
        deliveries = scheduled[customer.index]
        if customer.call_in:
            if _next_unsatisfied_order(customer, deliveries) is not None:
                unsatisfied.append(customer)
        elif next((event for event in _paper_events(instance, customer, deliveries, inventory_cache) if event.safety_breach), None) is not None:
            unsatisfied.append(customer)
    return unsatisfied


def _paper_shortage_time(
    instance: Instance,
    customer: Customer,
    deliveries: dict[int, float],
    inventory_cache: dict[tuple[int, tuple[tuple[int, float], ...]], tuple],
) -> int:
    if customer.call_in:
        order = _next_unsatisfied_order(customer, deliveries)
        return order[1].latest_time if order is not None else instance.latest_time
    breach = next((event.step for event in _paper_events(instance, customer, deliveries, inventory_cache) if event.safety_breach), None)
    return instance.latest_time if breach is None else breach * instance.unit


def _build_paper_shift(
    instance: Instance,
    resource: _ResourceState,
    window: TimeWindow,
    shift_index: int,
    scheduled: dict[int, dict[int, float]],
    inventory_cache: dict[tuple[int, tuple[tuple[int, float], ...]], tuple],
    *,
    rng: random.Random,
    selection_range: float,
    refill_coefficient: float,
    candidate_pool_size: int,
    economic_urgency_minutes: int,
) -> Shift | None:
    driver = instance.drivers[resource.driver]
    trailer = instance.trailers[resource.trailer]
    start = max(window.start, resource.available_time, resource.trailer_available_time)
    current_point = instance.base_index
    current_time = start
    driving = 0
    operations: list[Operation] = []
    visited: set[int] = set()
    has_layover_customer = False

    while True:
        candidates = []
        # A route may profitably absorb another VMI customer whose inventory
        # has reached its economic level even when that customer is not yet a
        # formal safety violation.  Restricting this pool to currently
        # unsatisfied tanks made the constructor waste scarce driver windows
        # on one-stop routes, unlike the published order-up-to construction.
        ranked_pool = sorted(
            instance.customers,
            key=lambda customer: _paper_shortage_time(
                instance,
                customer,
                scheduled[customer.index],
                inventory_cache,
            ),
        )[:candidate_pool_size]
        for customer in ranked_pool:
            if customer.index in visited or resource.trailer not in customer.allowed_trailers:
                continue
            if customer.call_in and _next_unsatisfied_order(customer, scheduled[customer.index]) is None:
                continue
            candidate = _paper_customer_candidate(
                instance,
                resource,
                window,
                current_point=current_point,
                current_time=current_time,
                driving=driving,
                customer=customer,
                deliveries=scheduled[customer.index],
                has_layover_customer=has_layover_customer,
                force_reload=False,
                at_route_start=not operations,
                inventory_cache=inventory_cache,
            )
            if candidate is None:
                candidate = _paper_customer_candidate(
                    instance,
                    resource,
                    window,
                    current_point=current_point,
                    current_time=current_time,
                    driving=driving,
                    customer=customer,
                    deliveries=scheduled[customer.index],
                    has_layover_customer=has_layover_customer,
                    force_reload=True,
                    at_route_start=not operations,
                    inventory_cache=inventory_cache,
                )
            if candidate is not None:
                candidates.append(candidate)
        if not candidates:
            break

        # Economic level is a soft gate, not a prohibition: favour a customer
        # below its level whenever one is reachable, but retain above-level
        # candidates as route fillers when no such service is available.  This
        # is the behaviour seen in the cold-start oracle and reconciles the
        # paper's “postponed or cancelled” wording with its dense routes.
        economical = [
            candidate
            for candidate in candidates
            if (
                _paper_economic_candidate(instance, candidate, scheduled[candidate.customer.index], resource.trailer, inventory_cache)
                or _paper_shortage_time(
                    instance,
                    candidate.customer,
                    scheduled[candidate.customer.index],
                    inventory_cache,
                ) <= current_time + economic_urgency_minutes
            )
        ]
        if economical:
            candidates = economical

        candidates.sort(
            key=lambda candidate: _paper_rank_key(instance, candidate, scheduled[candidate.customer.index], inventory_cache))
        # The paper selects among the first tau*k candidates at the kth visit.
        # At a route's kth visit, ``k`` is one-based and the range is capped by
        # the available reachable operations.
        choice_count = min(len(candidates), max(1, int(selection_range * (len(operations) + 1))))
        selected = rng.choice(candidates[:choice_count])

        # If residual stock is becoming scarce, choose a reachable source
        # probabilistically.  A mandatory reload is performed when the chosen
        # customer cannot receive its required quantity from residual stock.
        # ``eta`` controls how aggressively a residual load is retained for
        # deliveries.  With the published value eta=2, optional refilling
        # starts only after the trailer is below half full; a reload remains
        # mandatory when the selected delivery cannot be made otherwise.
        refill_probability = max(
            0.0,
            1.0 - refill_coefficient * resource.trailer_quantity / max(trailer.capacity, 1.0),
        )
        requires_reload = resource.trailer_quantity + EPSILON < selected.quantity
        if requires_reload or rng.random() < refill_probability:
            reloaded = _paper_customer_candidate(
                instance,
                resource,
                window,
                current_point=current_point,
                current_time=current_time,
                driving=driving,
                customer=selected.customer,
                deliveries=scheduled[selected.customer.index],
                has_layover_customer=has_layover_customer,
                force_reload=True,
                at_route_start=not operations,
                inventory_cache=inventory_cache,
            )
            if reloaded is not None:
                selected = reloaded
            elif requires_reload:
                candidates.remove(selected)
                if not candidates:
                    break
                continue

        if not operations:
            if selected.source_arrival is None:
                raw_arrival = current_time + selected.travel_time
            else:
                source = instance.source_by_point[selected.source_index]
                raw_arrival = (
                    selected.source_arrival
                    + source.setup_time
                    + instance.time_matrix[selected.source_index][selected.customer.index]
                )
            departure_delay = max(0, selected.arrival - raw_arrival)
            if departure_delay:
                start += departure_delay
                current_time += departure_delay
                if selected.source_arrival is not None:
                    selected = replace(selected, source_arrival=selected.source_arrival + departure_delay)

        if selected.source_arrival is not None:
            operations.append(
                Operation(
                    point=selected.source_index,
                    arrival=selected.source_arrival,
                    quantity=-selected.load_quantity,
                )
            )
        operations.append(Operation(point=selected.customer.index, arrival=selected.arrival, quantity=selected.quantity))
        scheduled[selected.customer.index][selected.arrival] = (
            scheduled[selected.customer.index].get(selected.arrival, 0.0) + selected.quantity
        )
        resource.trailer_quantity += selected.load_quantity - selected.quantity
        current_point = selected.customer.index
        current_time = selected.departure
        driving = selected.driving_after
        visited.add(selected.customer.index)
        has_layover_customer = has_layover_customer or selected.customer.layover_customer
        # A rest before the return leg ends this shift.  Continuing would make
        # that implicit layover disappear from the actual operation sequence.
        if selected.return_layover:
            break

    if not operations:
        return None
    end = current_time + instance.time_matrix[current_point][instance.base_index]
    if driving + instance.time_matrix[current_point][instance.base_index] > driver.max_driving_duration:
        if not has_layover_customer or end + driver.layover_duration > window.end:
            return None
        end += driver.layover_duration
    resource.available_time = end + driver.min_inter_shift_duration
    resource.trailer_available_time = end
    return Shift(index=shift_index, driver=resource.driver, trailer=resource.trailer, start=start, operations=tuple(operations))


def _paper_rank_key(
    instance: Instance,
    candidate: _Candidate,
    deliveries: dict[int, float],
    inventory_cache: dict[tuple[int, tuple[tuple[int, float], ...]], tuple],
) -> tuple[float, float, float, int]:
    """Shortage first, then the paper's cost/compatibility tie-breakers."""
    customer = candidate.customer
    shortage = _paper_shortage_time(instance, customer, deliveries, inventory_cache)
    # Su et al. use a tolerance ``mu`` for close shortage times and resolve
    # those near-ties through routing cost.  The instance time unit is the
    # natural published granularity: it preserves urgency by hour while
    # favouring the cheap next leg needed for dense routes.
    shortage_bucket = shortage // max(1, instance.unit)
    return (shortage_bucket, candidate.travel_time, len(customer.allowed_trailers), customer.index)


def _paper_economic_candidate(
    instance: Instance,
    candidate: _Candidate,
    deliveries: dict[int, float],
    trailer_index: int,
    inventory_cache: dict[tuple[int, tuple[tuple[int, float], ...]], tuple],
) -> bool:
    """Whether a route candidate is at/below the paper's economic level."""
    customer = candidate.customer
    if customer.call_in:
        return True
    step = min(instance.horizon - 1, max(0, candidate.arrival // instance.unit))
    inventory = _paper_events(instance, customer, deliveries, inventory_cache)[step].after_consumption
    trailer = instance.trailers[trailer_index]
    return inventory <= min(0.3 * customer.capacity, 0.4 * trailer.capacity) + EPSILON


def _paper_customer_candidate(
    instance: Instance,
    resource: _ResourceState,
    window: TimeWindow,
    *,
    current_point: int,
    current_time: int,
    driving: int,
    customer: Customer,
    deliveries: dict[int, float],
    has_layover_customer: bool,
    force_reload: bool,
    at_route_start: bool,
    inventory_cache: dict[tuple[int, tuple[tuple[int, float], ...]], tuple],
) -> _Candidate | None:
    """One legal paper-construction delivery, optionally preceded by a load."""
    trailer = instance.trailers[resource.trailer]
    source = min(
        (source for source in instance.sources if resource.trailer in source.allowed_trailers),
        key=lambda source: instance.time_matrix[current_point][source.index] + instance.time_matrix[source.index][customer.index],
        default=None,
    )
    if source is None:
        return None
    load_quantity = 0.0
    source_arrival = None
    point = current_point
    time = current_time
    added_travel = 0
    stock = resource.trailer_quantity
    if force_reload:
        source_arrival = time + instance.time_matrix[point][source.index]
        time = source_arrival + source.setup_time
        added_travel = instance.time_matrix[point][source.index]
        point = source.index
        load_quantity = trailer.capacity - stock
        stock = trailer.capacity

    raw_arrival = time + instance.time_matrix[point][customer.index]
    if customer.call_in:
        order_info = _next_unsatisfied_order(customer, deliveries)
        if order_info is None:
            return None
        _, order = order_info
        arrival = _align_arrival_to_customer_window(customer, max(raw_arrival, order.earliest_time))
        if arrival is None or arrival > order.latest_time:
            return None
        already = sum(quantity for when, quantity in deliveries.items() if order.earliest_time <= when <= order.latest_time)
        quantity = min(stock, max(0.0, order.min_quantity_to_satisfy - already))
    else:
        arrival = _align_arrival_to_customer_window(customer, raw_arrival)
        if arrival is None:
            return None
        step = min(instance.horizon - 1, max(0, arrival // instance.unit))
        events = _paper_events(instance, customer, deliveries, inventory_cache)
        inventory = events[step].after_consumption
        # The paper describes this as an economic *postponement* rule.  The
        # cold-start oracle still uses above-level customers as efficient
        # route fillers, so keep them admissible and let the shortage ranking
        # defer them unless they are useful in the selected top range.
        room = min(customer.capacity, customer.capacity - inventory)
        quantity = min(stock, room)
        quantity = _cap_quantity_without_future_overfill(instance, customer, deliveries, arrival, quantity)
        if quantity + EPSILON < customer.min_operation_quantity:
            return None
    if quantity <= EPSILON:
        return None

    departure = arrival + customer.setup_time
    return_travel = instance.time_matrix[customer.index][instance.base_index]
    travel = added_travel + instance.time_matrix[point][customer.index]
    # The construction only admits a node from which the pair can return to
    # base inside its driver window, as specified in the paper.
    if departure + return_travel > window.end:
        return None
    if not is_time_window_valid(arrival, departure, customer.time_windows):
        return None
    layover_before = (
        not at_route_start
        and arrival - raw_arrival >= instance.drivers[resource.driver].layover_duration
    )
    if layover_before and not (has_layover_customer or customer.layover_customer):
        return None
    # A source reload is itself part of the same continuous driving spell;
    # neither a later customer window nor a possible return layover can make
    # an already-overlong preceding leg legal.
    if driving + added_travel > instance.drivers[resource.driver].max_driving_duration:
        return None
    driving_after = driving + travel
    if layover_before:
        driving_after = instance.time_matrix[point][customer.index]
    elif driving_after > instance.drivers[resource.driver].max_driving_duration:
        return None
    return_layover = driving_after + return_travel > instance.drivers[resource.driver].max_driving_duration
    if return_layover and not (has_layover_customer or customer.layover_customer):
        return None
    if return_layover and departure + return_travel + instance.drivers[resource.driver].layover_duration > window.end:
        return None
    if not return_layover and not is_driving_duration_valid(instance.drivers[resource.driver], driving_after + return_travel):
        return None
    return _Candidate(
        customer=customer,
        arrival=arrival,
        departure=departure,
        quantity=quantity,
        travel_time=travel,
        driving_after=driving_after,
        layover_before=layover_before,
        return_layover=return_layover,
        source_arrival=source_arrival,
        load_quantity=load_quantity,
        source_index=source.index,
    )
