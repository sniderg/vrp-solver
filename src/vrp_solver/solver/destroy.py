from __future__ import annotations

import random
from dataclasses import dataclass

from ..model import Instance, Shift, Solution
from ..rules import derive_solution, is_time_window_valid, is_trailer_allowed
from .pressure import EPSILON, MINUTES_PER_DAY, pressure_points


@dataclass(frozen=True)
class DestroyResult:
    solution: Solution
    removed_shifts: tuple[Shift, ...]
    operator: str
    focus_customers: tuple[int, ...]
    start_minute: int
    end_minute: int


@dataclass(frozen=True)
class DestroyConfig:
    end_day: int = 21
    replace_from_day: int = 3
    max_removed_shifts: int = 8
    related_customer_count: int = 8
    time_band_days: int = 3


@dataclass(frozen=True)
class _ResourceConflictMove:
    customer: int
    driver: int
    trailer: int
    start_minute: int
    end_minute: int
    pressure_minute: int
    trailer_load: float
    conflicts: tuple[Shift, ...]


def resource_conflict_destroy(
    instance: Instance,
    solution: Solution,
    rng: random.Random,
    *,
    config: DestroyConfig,
) -> DestroyResult:
    """Remove shifts blocking a promising loaded-trailer rescue interval."""
    moves = _resource_conflict_moves(instance, solution, config)
    if not moves:
        return pressure_band_destroy(instance, solution, rng, config=config)

    best_group = moves[: min(5, len(moves))]
    move = rng.choice(best_group)
    removed = tuple(
        sorted(
            move.conflicts[: config.max_removed_shifts],
            key=lambda shift: (shift.start, shift.index),
        )
    )
    if not removed:
        return pressure_band_destroy(instance, solution, rng, config=config)

    removed_ids = {shift.index for shift in removed}
    kept = tuple(shift for shift in solution.shifts if shift.index not in removed_ids)
    customers = _related_customers(instance, move.customer, config.related_customer_count)
    start = min([move.start_minute, *(shift.start for shift in removed)])
    end = max([move.end_minute, *(shift.start for shift in removed)])
    return DestroyResult(
        solution=Solution(shifts=kept),
        removed_shifts=removed,
        operator="resource_conflict",
        focus_customers=customers,
        start_minute=start,
        end_minute=end,
    )


def pressure_band_destroy(
    instance: Instance,
    solution: Solution,
    rng: random.Random,
    *,
    config: DestroyConfig,
) -> DestroyResult:
    points = pressure_points(instance, solution, end_day=config.end_day)
    if not points:
        return route_block_destroy(instance, solution, rng, config=config)
    focus = points[0]
    half_width = config.time_band_days * MINUTES_PER_DAY // 2
    start = max(config.replace_from_day * MINUTES_PER_DAY, focus.first_minute - half_width)
    end = min(config.end_day * MINUTES_PER_DAY, focus.first_minute + half_width)
    customers = _related_customers(instance, focus.customer, config.related_customer_count)
    return _remove_matching_shifts(
        solution,
        operator="pressure_band",
        focus_customers=customers,
        start_minute=start,
        end_minute=end,
        max_removed=config.max_removed_shifts,
        rng=rng,
    )


def related_customer_destroy(
    instance: Instance,
    solution: Solution,
    rng: random.Random,
    *,
    config: DestroyConfig,
) -> DestroyResult:
    points = pressure_points(instance, solution, end_day=config.end_day)
    if not points:
        return route_block_destroy(instance, solution, rng, config=config)
    seed = rng.choice(points[: min(6, len(points))]).customer
    customers = _related_customers(instance, seed, config.related_customer_count)
    return _remove_matching_shifts(
        solution,
        operator="related_customer",
        focus_customers=customers,
        start_minute=config.replace_from_day * MINUTES_PER_DAY,
        end_minute=config.end_day * MINUTES_PER_DAY,
        max_removed=config.max_removed_shifts,
        rng=rng,
    )


def route_block_destroy(
    instance: Instance,
    solution: Solution,
    rng: random.Random,
    *,
    config: DestroyConfig,
) -> DestroyResult:
    del instance
    start_day = rng.randint(config.replace_from_day, max(config.replace_from_day, config.end_day - 1))
    start = start_day * MINUTES_PER_DAY
    end = min(config.end_day * MINUTES_PER_DAY, start + config.time_band_days * MINUTES_PER_DAY)
    return _remove_matching_shifts(
        solution,
        operator="route_block",
        focus_customers=(),
        start_minute=start,
        end_minute=end,
        max_removed=config.max_removed_shifts,
        rng=rng,
    )


def _related_customers(instance: Instance, seed: int, count: int) -> tuple[int, ...]:
    return tuple(
        customer.index
        for customer in sorted(
            (customer for customer in instance.customers if not customer.call_in),
            key=lambda customer: (instance.time_matrix[seed][customer.index], customer.index),
        )[:count]
    )


def _resource_conflict_moves(
    instance: Instance,
    solution: Solution,
    config: DestroyConfig,
) -> list[_ResourceConflictMove]:
    pressure = pressure_points(instance, solution, end_day=config.end_day)
    if not pressure:
        return []

    derived = derive_solution(instance, solution)
    trailer_cache = _trailer_load_cache(instance, solution)
    start_minute = config.replace_from_day * MINUTES_PER_DAY
    end_minute = config.end_day * MINUTES_PER_DAY
    top_pressure = pressure[: max(6, config.related_customer_count)]
    moves: list[_ResourceConflictMove] = []

    for point in top_pressure:
        customer = instance.customer_by_point[point.customer]
        if customer.call_in:
            continue
        latest_arrival = min(point.first_minute - instance.unit, end_minute - 1)
        if latest_arrival < start_minute:
            latest_arrival = end_minute - 1

        arrival_samples = _late_arrival_samples(
            start_minute,
            latest_arrival,
            config.time_band_days,
        )
        for target_arrival in arrival_samples:
            for driver in instance.drivers:
                to_customer = instance.time_matrix[instance.base_index][customer.index]
                from_customer = instance.time_matrix[customer.index][instance.base_index]
                shift_start = target_arrival - to_customer
                departure = target_arrival + customer.setup_time
                shift_end = departure + from_customer
                if shift_start < start_minute or shift_end > end_minute:
                    continue
                if to_customer + from_customer > driver.max_driving_duration:
                    continue
                if not is_time_window_valid(shift_start, shift_end, driver.time_windows):
                    continue
                if not is_time_window_valid(target_arrival, departure, customer.time_windows):
                    continue

                for trailer in instance.trailers:
                    if trailer.index not in driver.trailer_ids:
                        continue
                    if not is_trailer_allowed(instance, customer.index, trailer.index):
                        continue
                    trailer_load = _trailer_load_at(instance, trailer_cache, trailer.index, shift_start)
                    if trailer_load < customer.min_operation_quantity - EPSILON:
                        continue
                    conflicts = _resource_conflicts(
                        instance,
                        derived,
                        driver_id=driver.index,
                        trailer_id=trailer.index,
                        start_minute=shift_start,
                        end_minute=shift_end,
                        removable_start=start_minute,
                        removable_end=end_minute,
                    )
                    if not conflicts or len(conflicts) > config.max_removed_shifts:
                        continue
                    moves.append(
                        _ResourceConflictMove(
                            customer=customer.index,
                            driver=driver.index,
                            trailer=trailer.index,
                            start_minute=shift_start,
                            end_minute=shift_end,
                            pressure_minute=point.first_minute,
                            trailer_load=trailer_load,
                            conflicts=conflicts,
                        )
                    )

    return sorted(
        moves,
        key=lambda move: (
            len(move.conflicts),
            move.pressure_minute,
            -move.trailer_load,
            move.start_minute,
            move.customer,
            move.driver,
            move.trailer,
        ),
    )


def _resource_conflicts(
    instance: Instance,
    derived_shifts,
    *,
    driver_id: int,
    trailer_id: int,
    start_minute: int,
    end_minute: int,
    removable_start: int,
    removable_end: int,
) -> tuple[Shift, ...]:
    conflicts: dict[int, Shift] = {}
    driver = instance.drivers[driver_id]
    driver_start = start_minute - driver.min_inter_shift_duration
    driver_end = end_minute + driver.min_inter_shift_duration

    for derived in derived_shifts:
        shift = derived.shift
        conflicts_driver = (
            shift.driver == driver_id
            and _intervals_overlap(driver_start, driver_end, shift.start, derived.end)
        )
        conflicts_trailer = (
            shift.trailer == trailer_id
            and _intervals_overlap(start_minute, end_minute, shift.start, derived.end)
        )
        if not conflicts_driver and not conflicts_trailer:
            continue
        if shift.start < removable_start or shift.start >= removable_end:
            return ()
        conflicts[shift.index] = shift

    return tuple(sorted(conflicts.values(), key=lambda shift: (shift.start, shift.index)))


def _late_arrival_samples(start_minute: int, latest_arrival: int, time_band_days: int) -> tuple[int, ...]:
    lookback = max(MINUTES_PER_DAY, time_band_days * MINUTES_PER_DAY)
    local_earliest = max(start_minute, latest_arrival - lookback)
    if start_minute >= latest_arrival:
        return (latest_arrival,)
    local_offsets = (0.0, 0.15, 0.35, 0.6, 1.0)
    broad_offsets = tuple(i / 19 for i in range(20))
    local = {
        latest_arrival - round((latest_arrival - local_earliest) * offset)
        for offset in local_offsets
    }
    broad = {
        latest_arrival - round((latest_arrival - start_minute) * offset)
        for offset in broad_offsets
    }
    return tuple(
        sorted(
            local | broad,
            reverse=True,
        )
    )


def _trailer_load_cache(instance: Instance, solution: Solution) -> dict[int, list[tuple[int, float]]]:
    cache: dict[int, list[tuple[int, float]]] = {}
    for derived in sorted(derive_solution(instance, solution), key=lambda item: item.shift.start):
        cache.setdefault(derived.shift.trailer, []).append(
            (derived.shift.start, derived.end_trailer_quantity)
        )
    return cache


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


def _intervals_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return start_a < end_b and start_b < end_a


def _remove_matching_shifts(
    solution: Solution,
    *,
    operator: str,
    focus_customers: tuple[int, ...],
    start_minute: int,
    end_minute: int,
    max_removed: int,
    rng: random.Random,
) -> DestroyResult:
    focus = set(focus_customers)
    candidates: list[Shift] = []
    for shift in solution.shifts:
        if shift.start < start_minute or shift.start >= end_minute:
            continue
        if focus and not any(operation.point in focus for operation in shift.operations):
            continue
        candidates.append(shift)
    if not candidates and focus:
        return _remove_matching_shifts(
            solution,
            operator=operator,
            focus_customers=(),
            start_minute=start_minute,
            end_minute=end_minute,
            max_removed=max_removed,
            rng=rng,
        )
    rng.shuffle(candidates)
    removed = tuple(sorted(candidates[:max_removed], key=lambda shift: (shift.start, shift.index)))
    removed_ids = {shift.index for shift in removed}
    kept = tuple(shift for shift in solution.shifts if shift.index not in removed_ids)
    return DestroyResult(
        solution=Solution(shifts=kept),
        removed_shifts=removed,
        operator=operator,
        focus_customers=focus_customers,
        start_minute=start_minute,
        end_minute=end_minute,
    )
