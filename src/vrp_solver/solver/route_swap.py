from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations

from ..analysis import summarize_solution
from ..contest import score_prefix_with_feasibility_tail
from ..model import Instance, Shift, Solution


MINUTES_PER_DAY = 1440
EPSILON = 1e-9


@dataclass(frozen=True)
class RouteSwapMove:
    removed_indices: tuple[int, ...]
    added_reference_indices: tuple[int, ...]
    feasible: bool
    errors: int
    hard: int
    cost: float
    delivered: float
    ratio: float


@dataclass(frozen=True)
class RouteSwapResult:
    solution: Solution
    moves: tuple[RouteSwapMove, ...]
    evaluated_moves: int
    initial_ratio: float
    final_ratio: float


def transactional_route_swap_search(
    instance: Instance,
    incumbent: Solution,
    reference: Solution,
    *,
    horizon_days: int,
    max_remove: int = 2,
    max_add: int = 2,
    max_passes: int = 3,
    max_evaluations: int = 5000,
    min_delivered_fraction: float = 1.0,
    customer_bundles: bool = False,
    ignore_tail_call_ins: bool = True,
) -> RouteSwapResult:
    """Greedy feasible transaction search over route bundles.

    Candidate moves are evaluated atomically. The incumbent is updated only when
    a whole remove/add bundle is feasible and improves cost per delivered unit.
    """
    cutoff = horizon_days * MINUTES_PER_DAY
    current = _window_solution(incumbent, cutoff)
    reference_window = _window_shifts(reference, cutoff)
    initial_score = _score(instance, current, horizon_days, ignore_tail_call_ins)
    best_ratio = _ratio(initial_score)
    min_delivered = initial_score.scored_delivered_quantity * min_delivered_fraction
    accepted: list[RouteSwapMove] = []
    evaluated = 0

    for _pass in range(max_passes):
        current_exact = {_exact_signature(shift) for shift in current.shifts}
        reference_candidates = [
            shift
            for shift in reference_window
            if _exact_signature(shift) not in current_exact
        ]
        reference_exact = {_exact_signature(ref) for ref in reference_window}
        removable = [
            shift
            for shift in current.shifts
            if _exact_signature(shift) not in reference_exact
        ]
        current_score = _score(instance, current, horizon_days, ignore_tail_call_ins)
        ranker = _rank_customer_bundle_candidates if customer_bundles else _rank_transaction_candidates
        candidate_moves = ranker(
            instance,
            current,
            current_score.scored_estimated_cost,
            current_score.scored_delivered_quantity,
            best_ratio,
            min_delivered,
            removable,
            reference_candidates,
            max_remove=max_remove,
            max_add=max_add,
        )
        best_move: tuple[float, RouteSwapMove, Solution] | None = None
        for _estimated_ratio, removed, added in candidate_moves:
            if evaluated >= max_evaluations:
                return RouteSwapResult(
                    solution=current,
                    moves=tuple(accepted),
                    evaluated_moves=evaluated,
                    initial_ratio=_ratio(initial_score),
                    final_ratio=best_ratio,
                )
            removed_indices = {shift.index for shift in removed}
            remaining = [
                shift for shift in current.shifts if shift.index not in removed_indices
            ]
            evaluated += 1
            candidate = _reindex_solution([*remaining, *added])
            score = _score(instance, candidate, horizon_days, ignore_tail_call_ins)
            ratio = _ratio(score)
            move = RouteSwapMove(
                removed_indices=tuple(shift.index for shift in removed),
                added_reference_indices=tuple(shift.index for shift in added),
                feasible=score.feasible,
                errors=score.feasibility_errors,
                hard=score.hard_violations,
                cost=score.scored_estimated_cost,
                delivered=score.scored_delivered_quantity,
                ratio=ratio,
            )
            if not score.feasible:
                continue
            if score.scored_delivered_quantity + EPSILON < min_delivered:
                continue
            improvement = best_ratio - ratio
            if improvement <= EPSILON:
                continue
            if best_move is None or improvement > best_move[0]:
                best_move = (improvement, move, candidate)
        if best_move is None:
            break
        _improvement, move, current = best_move
        best_ratio = move.ratio
        accepted.append(move)

    return RouteSwapResult(
        solution=current,
        moves=tuple(accepted),
        evaluated_moves=evaluated,
        initial_ratio=_ratio(initial_score),
        final_ratio=best_ratio,
    )


def _score(instance: Instance, solution: Solution, horizon_days: int, ignore_tail_call_ins: bool):
    return score_prefix_with_feasibility_tail(
        instance,
        solution,
        score_days=horizon_days,
        feasibility_days=horizon_days,
        ignore_tail_call_ins=ignore_tail_call_ins,
    )


def _ratio(score) -> float:
    return score.scored_estimated_cost / max(1.0, score.scored_delivered_quantity)


def _rank_transaction_candidates(
    instance: Instance,
    current: Solution,
    current_cost: float,
    current_delivered: float,
    best_ratio: float,
    min_delivered: float,
    removable: list[Shift],
    reference_candidates: list[Shift],
    *,
    max_remove: int,
    max_add: int,
) -> list[tuple[float, tuple[Shift, ...], tuple[Shift, ...]]]:
    current_metrics = {
        summary.index: (summary.estimated_cost, summary.delivered_quantity)
        for summary in summarize_solution(instance, current)
    }
    reference_metrics = {
        summary.index: (summary.estimated_cost, summary.delivered_quantity)
        for summary in summarize_solution(instance, Solution(shifts=tuple(reference_candidates)))
    }
    candidates: list[tuple[float, tuple[Shift, ...], tuple[Shift, ...]]] = []
    for remove_count in range(0, min(max_remove, len(removable)) + 1):
        for add_count in range(0, min(max_add, len(reference_candidates)) + 1):
            if remove_count == 0 and add_count == 0:
                continue
            for removed in combinations(removable, remove_count):
                removed_cost = sum(current_metrics[shift.index][0] for shift in removed)
                removed_delivered = sum(current_metrics[shift.index][1] for shift in removed)
                for added in combinations(reference_candidates, add_count):
                    added_cost = sum(reference_metrics[shift.index][0] for shift in added)
                    added_delivered = sum(reference_metrics[shift.index][1] for shift in added)
                    delivered = current_delivered - removed_delivered + added_delivered
                    if delivered + EPSILON < min_delivered:
                        continue
                    cost = current_cost - removed_cost + added_cost
                    estimated_ratio = cost / max(1.0, delivered)
                    if estimated_ratio + EPSILON >= best_ratio:
                        continue
                    candidates.append((estimated_ratio, removed, added))
    candidates.sort(key=lambda item: item[0])
    return candidates


def _rank_customer_bundle_candidates(
    instance: Instance,
    current: Solution,
    current_cost: float,
    current_delivered: float,
    best_ratio: float,
    min_delivered: float,
    removable: list[Shift],
    reference_candidates: list[Shift],
    *,
    max_remove: int,
    max_add: int,
) -> list[tuple[float, tuple[Shift, ...], tuple[Shift, ...]]]:
    current_summaries = summarize_solution(instance, current)
    current_metrics = {
        summary.index: (summary.estimated_cost, summary.delivered_quantity)
        for summary in current_summaries
    }
    current_ends = {summary.index: summary.end_time for summary in current_summaries}
    reference_summaries = summarize_solution(instance, Solution(shifts=tuple(reference_candidates)))
    reference_metrics = {
        summary.index: (summary.estimated_cost, summary.delivered_quantity)
        for summary in reference_summaries
    }
    reference_ends = {summary.index: summary.end_time for summary in reference_summaries}
    removable_by_index = {shift.index: shift for shift in removable}
    candidates: dict[
        tuple[tuple[int, ...], tuple[int, ...]],
        tuple[float, tuple[Shift, ...], tuple[Shift, ...]],
    ] = {}

    for seed in reference_candidates:
        bundle: list[Shift] = []
        covered_customers: set[int] = set()
        remaining = sorted(reference_candidates, key=lambda shift: (shift.start, shift.driver, shift.trailer))
        while remaining and len(bundle) < max_add:
            if not bundle:
                chosen = seed
            else:
                chosen = max(
                    remaining,
                    key=lambda shift: (
                        len(_customer_points(shift) & covered_customers),
                        -abs(shift.start - seed.start),
                        -shift.index,
                    ),
                )
                if not (_customer_points(chosen) & covered_customers):
                    break
            if chosen not in remaining:
                break
            bundle.append(chosen)
            covered_customers.update(_customer_points(chosen))
            remaining.remove(chosen)
            _record_customer_bundle_candidate(
                instance,
                current_cost,
                current_delivered,
                best_ratio,
                min_delivered,
                current_metrics,
                current_ends,
                reference_metrics,
                reference_ends,
                removable_by_index,
                tuple(bundle),
                max_remove,
                candidates,
            )
    ordered = list(candidates.values())
    ordered.sort(key=lambda item: item[0])
    return ordered


def _record_customer_bundle_candidate(
    instance: Instance,
    current_cost: float,
    current_delivered: float,
    best_ratio: float,
    min_delivered: float,
    current_metrics: dict[int, tuple[float, float]],
    current_ends: dict[int, int],
    reference_metrics: dict[int, tuple[float, float]],
    reference_ends: dict[int, int],
    removable_by_index: dict[int, Shift],
    added: tuple[Shift, ...],
    max_remove: int,
    candidates: dict[
        tuple[tuple[int, ...], tuple[int, ...]],
        tuple[float, tuple[Shift, ...], tuple[Shift, ...]],
    ],
) -> None:
    bundle_customers: set[int] = set()
    for shift in added:
        bundle_customers.update(_customer_points(shift))
    removed = [
        shift
        for shift in removable_by_index.values()
        if _customer_points(shift) & bundle_customers
        or any(_resource_time_conflict(instance, shift, ref, current_ends, reference_ends) for ref in added)
    ]
    removed.sort(
        key=lambda shift: (
            -(len(_customer_points(shift) & bundle_customers)),
            shift.start,
            shift.driver,
            shift.trailer,
        )
    )
    removed = removed[:max_remove]
    removed_cost = sum(current_metrics[shift.index][0] for shift in removed)
    removed_delivered = sum(current_metrics[shift.index][1] for shift in removed)
    added_cost = sum(reference_metrics[shift.index][0] for shift in added)
    added_delivered = sum(reference_metrics[shift.index][1] for shift in added)
    delivered = current_delivered - removed_delivered + added_delivered
    if delivered + EPSILON < min_delivered:
        return
    cost = current_cost - removed_cost + added_cost
    estimated_ratio = cost / max(1.0, delivered)
    if estimated_ratio + EPSILON >= best_ratio:
        return
    key = (
        tuple(sorted(shift.index for shift in removed)),
        tuple(sorted(shift.index for shift in added)),
    )
    if key not in candidates or estimated_ratio < candidates[key][0]:
        candidates[key] = (estimated_ratio, tuple(removed), added)


def _customer_points(shift: Shift) -> set[int]:
    return {operation.point for operation in shift.operations if operation.quantity > 0.0}


def _resource_time_conflict(
    instance: Instance,
    current: Shift,
    reference: Shift,
    current_ends: dict[int, int],
    reference_ends: dict[int, int],
) -> bool:
    if current.driver != reference.driver and current.trailer != reference.trailer:
        return False
    current_end = current_ends.get(current.index, current.start)
    reference_end = reference_ends.get(reference.index, reference.start)
    trailer_gap = 0
    if current.driver == reference.driver:
        driver = next(
            (driver for driver in instance.drivers if driver.index == current.driver),
            None,
        )
        trailer_gap = 0 if driver is None else driver.min_inter_shift_duration
    return current.start < reference_end + trailer_gap and reference.start < current_end + trailer_gap


def _window_solution(solution: Solution, cutoff: int) -> Solution:
    return _reindex_solution(_window_shifts(solution, cutoff))


def _window_shifts(solution: Solution, cutoff: int) -> list[Shift]:
    return [shift for shift in solution.shifts if shift.start < cutoff]


def _reindex_solution(shifts: list[Shift]) -> Solution:
    return Solution(
        shifts=tuple(
            replace(shift, index=index)
            for index, shift in enumerate(
                sorted(shifts, key=lambda shift: (shift.start, shift.driver, shift.trailer))
            )
        )
    )


def _structure_signature(shift: Shift) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    points = tuple(operation.point for operation in shift.operations)
    customers = tuple(operation.point for operation in shift.operations if operation.quantity > 0)
    return (shift.driver, shift.trailer, points, customers)


def _exact_signature(shift: Shift) -> tuple[int, int, int, tuple[tuple[int, int, float], ...]]:
    return (
        shift.driver,
        shift.trailer,
        shift.start,
        tuple(
            (operation.point, operation.arrival, round(operation.quantity, 6))
            for operation in shift.operations
        ),
    )
