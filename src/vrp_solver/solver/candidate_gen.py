from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Dict, List, Set, Tuple

from ..model import Instance, Shift, Operation, Solution, Customer, Driver, Trailer, TimeWindow
from ..inventory import project_customer_inventory, days_of_inventory
from ..rules import is_driving_duration_valid, is_time_window_valid, is_trailer_allowed
from .cluster_greedy import (
    _Candidate,
    _compute_neighborhoods,
    _apply_cand,
    _cap_quantity_without_future_overfill,
)

EPSILON = 1e-6

@dataclass(frozen=True)
class GeneratorConfig:
    safety_buffer: float = 0.20
    neighborhood_size: int = 15
    max_candidates_per_window: int = 10
    seed: int = 42

@dataclass
class DriverState:
    driver: int
    available_time: int = 0
    next_window_index: int = 0

@dataclass
class TrailerState:
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

class InventoryCache:
    def __init__(self, instance: Instance, scheduled: Dict[int, Dict[int, float]]):
        self.instance = instance
        self.scheduled = scheduled
        self.cache = {}

    def get_events(self, customer_idx: int, local_scheduled: Dict[int, float]):
        if not local_scheduled:
            if customer_idx not in self.cache:
                customer = self.instance.customer_by_point[customer_idx]
                self.cache[customer_idx] = project_customer_inventory(self.instance, customer, self.scheduled[customer_idx])
            return self.cache[customer_idx]
        customer = self.instance.customer_by_point[customer_idx]
        total_scheduled = dict(self.scheduled[customer_idx])
        total_scheduled.update(local_scheduled)
        return project_customer_inventory(self.instance, customer, total_scheduled)

def generate_shift_candidates(
    instance: Instance,
    solution_prefix: Solution,
    *,
    start_day: int,
    end_day: int,
    config: GeneratorConfig = GeneratorConfig(),
) -> List[Shift]:
    from ..contest import truncate_solution
    cutoff_minute = start_day * 1440
    working_prefix = truncate_solution(solution_prefix, cutoff_minute)
    drivers, trailers, scheduled = _state_after_prefix(instance, working_prefix, cutoff_minute)
    inv_cache = InventoryCache(instance, scheduled)
    neighborhoods = _compute_neighborhoods(instance, k=config.neighborhood_size)
    source_lead_minutes = {c.index: min(instance.time_matrix[s.index][c.index] for s in instance.sources) for c in instance.customers}
    
    # Dynamically find fragile and remote tanks to avoid index out-of-bounds on smaller instances
    cust_list = [c for c in instance.customers if not c.call_in]
    if cust_list:
        def avg_f(c):
            return sum(c.forecast) / len(c.forecast) if c.forecast else 1.0
        cust_sorted_fragile = sorted(cust_list, key=lambda c: c.capacity / max(avg_f(c), 1e-3))
        FRAGILE_TANKS = {c.index for c in cust_sorted_fragile[:4]}
        cust_sorted_remote = sorted(cust_list, key=lambda c: instance.time_matrix[instance.base_index][c.index], reverse=True)
        REMOTE_CLUSTER = {c.index for c in cust_sorted_remote[:4]}
    else:
        FRAGILE_TANKS = set()
        REMOTE_CLUSTER = set()

    candidates: List[Shift] = []

    for day in range(start_day, end_day):
        day_start, day_end = day * 1440, (day + 1) * 1440
        print(f"Generating candidates for Day {day}...")
        doi_at_day_start = {}
        for customer in instance.customers:
            if customer.call_in: continue
            events = inv_cache.get_events(customer.index, {})
            inv_at_day_start = events[day_start // instance.unit].after_consumption
            doi = days_of_inventory(instance, customer, inv_at_day_start, day_start)
            doi_at_day_start[customer.index] = doi

        for d_state in drivers:
            driver = instance.drivers[d_state.driver]
            for wi in range(d_state.next_window_index, len(driver.time_windows)):
                window = driver.time_windows[wi]
                if window.start >= day_end: break
                if window.end <= day_start: continue
                print(f"    Window {wi} for Driver {d_state.driver}...")
                for t_state in trailers:
                    if t_state.trailer not in driver.trailer_ids: continue
                    
                    strategies = [
                        (config.max_candidates_per_window // 4, None, 0.2), # Greedy noisy
                        (config.max_candidates_per_window // 4, "JUGGLER", 0.1), # Multi-circuit Juggler
                        (config.max_candidates_per_window // 4, "CLUSTER_CIRCUIT", 0.0), # Group circuit
                        (config.max_candidates_per_window // 4, REMOTE_CLUSTER, 0.1), # Remote focus
                    ]
                    
                    for count, focus, noise in strategies:
                        for _ in range(count):
                            if focus == "JUGGLER":
                                cluster = random.choice([FRAGILE_TANKS, REMOTE_CLUSTER])
                                shift = _build_juggler_shift(
                                    instance, d_state, t_state, window, len(candidates),
                                    scheduled, cluster, doi_at_day_start, inv_cache
                                )
                            elif focus == "CLUSTER_CIRCUIT":
                                cluster = random.choice([FRAGILE_TANKS, REMOTE_CLUSTER])
                                shift = _build_cluster_circuit_shift(
                                    instance, d_state, t_state, window, len(candidates),
                                    scheduled, cluster, doi_at_day_start, inv_cache
                                )
                            else:
                                shift = _build_stochastic_shift(
                                    instance, d_state, t_state, window, len(candidates), 
                                    neighborhoods, scheduled, FRAGILE_TANKS, REMOTE_CLUSTER, 
                                    source_lead_minutes, doi_at_day_start, inv_cache, 
                                    noise=noise, focus_set=focus
                                )
                            if shift: candidates.append(shift)
    return candidates

def _first_economic_service_step(events, customer, start_step: int, fragile_tanks: Set[int]) -> int | None:
    threshold = customer.capacity * 0.95
    for event in events[start_step:]:
        if event.after_consumption <= threshold + EPSILON: return event.step
    return None

def _cap_qty_local(instance, customer, events, arrival, qty):
    arrival_step = arrival // instance.unit
    for event in events[arrival_step:]:
        if event.after_consumption + qty > customer.capacity + EPSILON:
            qty = max(0.0, customer.capacity - event.after_consumption)
    return qty

def _candidate_for_customer_local(
    instance, resource, window, current_pt, current_time, 
    driving, customer, local_scheduled, fragile_tanks, inv_cache, score_cutoff_minute=None,
):
    if not is_trailer_allowed(instance, customer.index, resource.trailer): return None
    if customer.call_in: return None
    source = next((s for s in instance.sources if resource.trailer in s.allowed_trailers), None)
    if source is None: return None
    trailer = instance.trailers[resource.trailer]
    load_qty, source_arr, time, pt, travel, trailer_qty = 0.0, None, current_time, current_pt, 0, resource.trailer_quantity

    if trailer_qty < customer.min_operation_quantity - EPSILON:
        source_arr = time + instance.time_matrix[pt][source.index]
        if score_cutoff_minute is not None and source_arr >= score_cutoff_minute: return None
        time = source_arr + source.setup_time
        travel += instance.time_matrix[pt][source.index]
        pt = source.index
        load_qty = trailer.capacity - trailer_qty
        trailer_qty = trailer.capacity

    raw_arrival = time + instance.time_matrix[pt][customer.index]
    arrival = raw_arrival
    if score_cutoff_minute is not None and arrival >= score_cutoff_minute: return None
    total_travel = travel + instance.time_matrix[pt][customer.index]
    ret_travel = instance.time_matrix[customer.index][instance.base_index]
    
    if not is_driving_duration_valid(instance.drivers[resource.driver], driving + total_travel + ret_travel): return None

    events = inv_cache.get_events(customer.index, local_scheduled)
    arrival_step = min(max(arrival // instance.unit, 0), instance.horizon - 1)
    
    economic_step = _first_economic_service_step(events, customer, arrival_step, fragile_tanks)
    if economic_step is None: return None
    economic_arrival = max(arrival, economic_step * instance.unit)
    if economic_arrival - raw_arrival >= instance.drivers[resource.driver].layover_duration: return None
    arrival = economic_arrival
    
    if score_cutoff_minute is not None and arrival >= score_cutoff_minute: return None
    departure = arrival + customer.setup_time
    if departure + ret_travel > window.end: return None
    if not is_time_window_valid(arrival, departure, customer.time_windows): return None
        
    arrival_step = min(max(arrival // instance.unit, 0), instance.horizon - 1)
    inv_at_arr = events[arrival_step].after_consumption
    already_delivered = sum(q for t, q in local_scheduled.items() if t // instance.unit == arrival_step)
    
    max_room = customer.capacity - inv_at_arr - already_delivered
    if max_room < customer.min_operation_quantity - EPSILON: return None
    
    qty = min(trailer_qty, max_room)
    qty = _cap_qty_local(instance, customer, events, arrival, qty)
    if qty < customer.min_operation_quantity - EPSILON: return None
    return _Candidate(customer=customer, arrival=arrival, departure=departure, quantity=qty, travel_time=total_travel, source_arrival=source_arr, load_quantity=load_qty, source_index=source.index)

def _simple_score(customer, candidate, doi_at_day_start, fragile_tanks, remote_cluster, noise, focus_set):
    score = 1000.0 / (doi_at_day_start.get(customer.index, 10.0) + 0.1)
    if focus_set and customer.index in focus_set: score += 20000.0
    if customer.index in fragile_tanks: score += 10000.0
    if customer.index in remote_cluster: score += 5000.0
    score -= candidate.travel_time * 0.5
    if noise > 0: score *= (1.0 + (random.random() - 0.5) * 2.0 * noise)
    return score

def _build_stochastic_shift(
    instance, d_state, t_state, window, shift_idx, neighborhoods, scheduled, 
    fragile_tanks, remote_cluster, source_lead_minutes, doi_at_day_start, inv_cache,
    noise=0.0, focus_set=None
) -> Shift | None:
    start = max(window.start, d_state.available_time, t_state.available_time)
    if start >= window.end: return None
    resource = _ResourceState(d_state.driver, t_state.trailer, t_state.trailer_quantity, start, start)
    local_scheduled = {c.index: {} for c in instance.customers}
    local_planned_vol = {}
    operations, current_pt, current_time, driving, served_this_shift = [], instance.base_index, start, 0, set()

    max_driving = instance.drivers[resource.driver].max_driving_duration
    while driving < max_driving:
        potential_customers = []
        if max_driving - driving < 30: break

        for customer in instance.customers:
            if customer.call_in or customer.index in served_this_shift: continue
            if focus_set and customer.index in focus_set: pass
            elif doi_at_day_start.get(customer.index, 10.0) > 5.0: continue

            c = _candidate_for_customer_local(instance, resource, window, current_pt, current_time, driving, customer, local_scheduled[customer.index], fragile_tanks, inv_cache)
            if c:
                score = _simple_score(customer, c, doi_at_day_start, fragile_tanks, remote_cluster, noise, focus_set)
                potential_customers.append((score, c))

        if not potential_customers: break
        potential_customers.sort(key=lambda x: x[0], reverse=True)
        best_cand = potential_customers[0][1] if random.random() < 0.7 else random.choice(potential_customers[:3])[1]
        _apply_cand(operations, resource, local_scheduled, {c.index: 0 for c in instance.customers}, local_planned_vol, best_cand, instance)
        served_this_shift.add(best_cand.customer.index)
        current_pt, current_time, driving = best_cand.customer.index, best_cand.departure, driving + best_cand.travel_time
    if not operations: return None
    return Shift(shift_idx, resource.driver, resource.trailer, start, tuple(operations))

def _build_juggler_shift(
    instance: Instance, d_state, t_state, window, shift_idx,
    scheduled, cluster: Set[int], doi_at_day_start, inv_cache
) -> Shift | None:
    start = max(window.start, d_state.available_time, t_state.available_time)
    if start >= window.end: return None
    resource = _ResourceState(d_state.driver, t_state.trailer, t_state.trailer_quantity, start, start)
    local_scheduled = {c.index: {} for c in instance.customers}
    operations, current_pt, current_time, driving = [], instance.base_index, start, 0
    max_driving = instance.drivers[resource.driver].max_driving_duration
    cluster_list = list(cluster)
    while driving < max_driving - 60:
        best_cand = None
        best_score = -1e9
        for cid in cluster_list:
            c = _candidate_for_customer_local(instance, resource, window, current_pt, current_time, driving, instance.customer_by_point[cid], local_scheduled[cid], set(), inv_cache)
            if c:
                score = 1.0 / (doi_at_day_start.get(cid, 10.0) + 0.1)
                if score > best_score:
                    best_score = score
                    best_cand = c
        if not best_cand: break
        _apply_cand(operations, resource, local_scheduled, {c.index: 0 for c in instance.customers}, {}, best_cand, instance)
        current_pt, current_time, driving = best_cand.customer.index, best_cand.departure, driving + best_cand.travel_time
    if not operations: return None
    return Shift(shift_idx, resource.driver, resource.trailer, start, tuple(operations))

def _build_cluster_circuit_shift(
    instance: Instance, d_state, t_state, window, shift_idx,
    scheduled, cluster: Set[int], doi_at_day_start, inv_cache
) -> Shift | None:
    start = max(window.start, d_state.available_time, t_state.available_time)
    if start >= window.end: return None
    resource = _ResourceState(d_state.driver, t_state.trailer, t_state.trailer_quantity, start, start)
    local_scheduled = {c.index: {} for c in instance.customers}
    operations, current_pt, current_time, driving = [], instance.base_index, start, 0
    targets = sorted(list(cluster), key=lambda cid: doi_at_day_start.get(cid, 10.0))
    for cid in targets:
        customer = instance.customer_by_point[cid]
        c = _candidate_for_customer_local(instance, resource, window, current_pt, current_time, driving, customer, local_scheduled[cid], set(), inv_cache)
        if c:
            _apply_cand(operations, resource, local_scheduled, {c.index: 0 for c in instance.customers}, {}, c, instance)
            current_pt, current_time, driving = c.customer.index, c.departure, driving + c.travel_time
    if not operations: return None
    return Shift(shift_idx, resource.driver, resource.trailer, start, tuple(operations))

def _state_after_prefix(instance, prefix, cutoff_minute):
    from ..rules import derive_solution
    derived = derive_solution(instance, prefix)
    driver_avail = {d.index: 0 for d in instance.drivers}
    driver_windows = {d.index: 0 for d in instance.drivers}
    trailer_avail = {t.index: 0 for t in instance.trailers}
    trailer_qty = {t.index: t.initial_quantity for t in instance.trailers}
    for d_shift in derived:
        driver_avail[d_shift.shift.driver] = max(driver_avail[d_shift.shift.driver], d_shift.end + instance.drivers[d_shift.shift.driver].min_inter_shift_duration)
        trailer_avail[d_shift.shift.trailer] = max(trailer_avail[d_shift.shift.trailer], d_shift.end)
        trailer_qty[d_shift.shift.trailer] = d_shift.end_trailer_quantity
    for driver in instance.drivers:
        avail = driver_avail[driver.index]
        for wi, window in enumerate(driver.time_windows):
            if window.end > avail:
                driver_windows[driver.index] = wi
                break
        else: driver_windows[driver.index] = len(driver.time_windows)
    drivers = [DriverState(d.index, driver_avail[d.index], driver_windows[d.index]) for d in instance.drivers]
    trailers = [TrailerState(t.index, trailer_avail[t.index], trailer_qty[t.index]) for t in instance.trailers]
    scheduled = {c.index: {} for c in instance.customers}
    for shift in prefix.shifts:
        for op in shift.operations:
            if op.quantity > EPSILON: scheduled[op.point][op.arrival] = scheduled[op.point].get(op.arrival, 0.0) + op.quantity
    return drivers, trailers, scheduled
