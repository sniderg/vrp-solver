from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

from ..model import Instance, Operation, Shift, Solution
from ..xml_io import load_solution

MINUTES_PER_DAY = 1440


@dataclass(frozen=True)
class PriorRouteCandidate:
    shift: Shift
    source: str = "historical_prior"


@dataclass(frozen=True)
class RoutePriorDiagnostics:
    loaded: int = 0
    structurally_valid: int = 0
    inserted: int = 0
    selected: int = 0
    rejected: int = 0
    skeletons_regenerated: int = 0
    rejected_pressure_cover: int = 0
    rejected_route_summary: str = ""

    def flat(self) -> dict[str, int | str]:
        return self.__dict__.copy()


def load_route_prior_candidates(
    instance: Instance,
    path: str | Path,
    *,
    start_day: int = 0,
    end_day: int | None = None,
) -> tuple[PriorRouteCandidate, ...]:
    path = Path(path)
    if path.suffix.lower() == ".xml":
        return route_priors_from_solution(
            instance,
            load_solution(path),
            start_day=start_day,
            end_day=end_day,
        )
    if path.suffix.lower() == ".csv":
        return route_priors_from_csv(instance, path, start_day=start_day, end_day=end_day)
    raise ValueError(f"Unsupported route prior format: {path.suffix}")


def route_priors_from_solution(
    instance: Instance,
    solution: Solution,
    *,
    start_day: int = 0,
    end_day: int | None = None,
) -> tuple[PriorRouteCandidate, ...]:
    start = start_day * MINUTES_PER_DAY
    end = (end_day * MINUTES_PER_DAY) if end_day is not None else None
    candidates = []
    skeletons = []
    for shift in solution.shifts:
        if shift.start < start or (end is not None and shift.start >= end):
            continue
        if _candidate_is_structurally_valid(instance, shift):
            candidates.append(PriorRouteCandidate(replace(shift, index=len(candidates))))
            regenerated = regenerate_prior_route_skeleton(instance, shift, index=len(candidates) + len(skeletons))
            if regenerated is not None and (
                regenerated.start != shift.start
                or regenerated.driver != shift.driver
                or regenerated.trailer != shift.trailer
                or regenerated.operations != shift.operations
            ):
                skeletons.append(
                    PriorRouteCandidate(
                        shift=regenerated,
                        source="historical_prior_skeleton",
                    )
                )
    return tuple([*candidates, *skeletons])


def route_priors_from_csv(
    instance: Instance,
    path: str | Path,
    *,
    start_day: int = 0,
    end_day: int | None = None,
) -> tuple[PriorRouteCandidate, ...]:
    grouped: dict[str, list[dict[str, str]]] = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row.get("route_id") or row.get("shift_id") or str(len(grouped)), []).append(row)
    shifts = []
    for rows in grouped.values():
        rows.sort(key=lambda row: int(float(row.get("sequence") or row.get("op_index") or 0)))
        first = rows[0]
        start = int(float(first.get("start") or first.get("start_minute") or 0))
        if start < start_day * MINUTES_PER_DAY:
            continue
        if end_day is not None and start >= end_day * MINUTES_PER_DAY:
            continue
        shift = Shift(
            index=len(shifts),
            driver=int(float(first.get("driver") or 0)),
            trailer=int(float(first.get("trailer") or 0)),
            start=start,
            operations=tuple(
                Operation(
                    point=int(float(row.get("point") or row.get("customer_id") or row.get("source_id") or 0)),
                    arrival=int(float(row.get("arrival") or row.get("arrival_minute") or start)),
                    quantity=float(row.get("quantity") or row.get("delivered_quantity") or 0.0),
                )
                for row in rows
            ),
        )
        if _candidate_is_structurally_valid(instance, shift):
            shifts.append(shift)
    return tuple(PriorRouteCandidate(shift=shift) for shift in shifts)


def prior_shifts(priors: tuple[PriorRouteCandidate, ...] | None) -> tuple[Shift, ...]:
    if not priors:
        return ()
    return tuple(prior.shift for prior in priors)


def prior_sources(priors: tuple[PriorRouteCandidate, ...] | None) -> tuple[str, ...]:
    if not priors:
        return ()
    return tuple(prior.source for prior in priors)


def regenerate_prior_route_skeleton(
    instance: Instance,
    shift: Shift,
    *,
    index: int = 0,
) -> Shift | None:
    """Retain a historical route's point order while refreshing timing/quantities.

    This is intentionally conservative: it does not force the skeleton into the
    solution, and it only emits routes that satisfy basic resource and point
    structure. Quantity repair later can tune exact deliveries.
    """
    if not _candidate_is_structurally_valid(instance, shift):
        return None
    source_points = {source.index for source in instance.sources}
    customer_by_point = instance.customer_by_point
    operations: list[Operation] = []
    trailer = instance.trailers[shift.trailer]
    trailer_qty = 0.0
    previous = instance.base_index
    current_time = _nearest_driver_window_start(instance, shift)

    for original in shift.operations:
        point = original.point
        arrival = current_time + instance.time_matrix[previous][point]
        if point in source_points:
            quantity = -max(0.0, min(trailer.capacity, abs(original.quantity) or trailer.capacity))
            trailer_qty = min(trailer.capacity, trailer_qty - quantity)
        elif point in customer_by_point:
            customer = customer_by_point[point]
            raw_qty = max(original.quantity, customer.min_operation_quantity)
            quantity = min(customer.capacity, trailer.capacity, raw_qty)
            if trailer_qty > 0.0:
                quantity = min(quantity, trailer_qty)
                trailer_qty -= quantity
        else:
            quantity = original.quantity
        operations.append(Operation(point=point, arrival=arrival, quantity=quantity))
        current_time = arrival + instance.setup_time_for_point(point)
        previous = point

    candidate = replace(shift, index=index, start=_nearest_driver_window_start(instance, shift), operations=tuple(operations))
    return candidate if _candidate_is_structurally_valid(instance, candidate) else None


def route_signature(instance: Instance, shift: Shift) -> tuple[int, int, int, tuple[int, ...]]:
    customers = tuple(
        operation.point
        for operation in shift.operations
        if operation.quantity > 0 and operation.point in instance.customer_by_point
    )
    return (shift.driver, shift.trailer, shift.start // 240, customers)


def _nearest_driver_window_start(instance: Instance, shift: Shift) -> int:
    driver = instance.drivers[shift.driver]
    day_start = (shift.start // MINUTES_PER_DAY) * MINUTES_PER_DAY
    windows = sorted(driver.time_windows, key=lambda window: abs(window.start - shift.start))
    for window in windows:
        if day_start <= window.start < day_start + MINUTES_PER_DAY:
            return max(window.start, day_start)
    return shift.start


def _candidate_is_structurally_valid(instance: Instance, shift: Shift) -> bool:
    if shift.driver < 0 or shift.driver >= len(instance.drivers):
        return False
    if shift.trailer < 0 or shift.trailer >= len(instance.trailers):
        return False
    driver = instance.drivers[shift.driver]
    if shift.trailer not in driver.trailer_ids:
        return False
    known_points = {instance.base_index}
    known_points.update(source.index for source in instance.sources)
    known_points.update(customer.index for customer in instance.customers)
    for operation in shift.operations:
        if operation.point not in known_points:
            return False
        if operation.point in instance.customer_by_point and operation.quantity < 0.0:
            return False
    return True
