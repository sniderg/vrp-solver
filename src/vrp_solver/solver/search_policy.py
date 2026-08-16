"""OR policy primitives for native feasibility-improvement searches.

Neighborhood modules construct routes and inventory moves.  This module owns
their common ordering, acceptance, operator-selection, and evaluation policy.
"""
from __future__ import annotations

import multiprocessing
import random
from collections.abc import Sequence

from ..contest import ContestScore, score_prefix_with_feasibility_tail
from ..diagnostics import ViolationVector
from ..model import Instance, Solution
from ..rules import validate_solution


def select_operator(
    operators: Sequence[str], rewards: Sequence[float],
    last_used: Sequence[int], iteration: int, rng: random.Random,
    feasibility_bias: bool,
) -> int:
    """Select by EWMA reward, recency, and feasibility-repair bias."""
    scores = []
    for index in range(len(operators)):
        recency = 2.0 if iteration - last_used[index] >= 33 else 1.0
        bias = 4.0 if feasibility_bias and index in (0, 1, 3) else 1.0
        scores.append(max(1.0, rewards[index]) * recency * bias)
    draw = rng.random() * sum(scores)
    for index, value in enumerate(scores):
        draw -= value
        if draw <= 0:
            return index
    return len(operators) - 1


def score(instance: Instance, solution: Solution, end_day: int) -> ContestScore:
    return score_prefix_with_feasibility_tail(
        instance, solution, score_days=end_day, feasibility_days=end_day,
        ignore_tail_call_ins=True,
    )


def candidate_frontier(
    candidates: list[Solution], *, budget: int, stagnation: int,
) -> list[Solution]:
    """Balance exploitation and topology exploration within an evaluation cap."""
    if budget <= 0 or len(candidates) <= budget:
        return candidates
    exploration_share = 1 / 3 if stagnation == 0 else 1 / 2
    explore = max(1, int(budget * exploration_share))
    exploit = max(1, budget - explore)
    head = candidates[:exploit]
    tail = candidates[exploit:]
    if explore == 1:
        return head + [tail[-1]]
    indexes = [
        round(index * (len(tail) - 1) / (explore - 1))
        for index in range(explore)
    ]
    return head + [tail[index] for index in indexes]


_WORKER_INSTANCE: Instance | None = None
_WORKER_CANDIDATES: list[Solution] | None = None
_WORKER_END_DAY: int | None = None


def _score_candidate_index(index: int) -> tuple[int, ContestScore]:
    assert _WORKER_INSTANCE is not None
    assert _WORKER_CANDIDATES is not None
    assert _WORKER_END_DAY is not None
    return index, score(
        _WORKER_INSTANCE, _WORKER_CANDIDATES[index], _WORKER_END_DAY,
    )


def score_candidates(
    instance: Instance, candidates: list[Solution], end_day: int, workers: int,
) -> list[tuple[Solution, ContestScore]]:
    if workers <= 1 or len(candidates) <= 1:
        return [(candidate, score(instance, candidate, end_day)) for candidate in candidates]
    global _WORKER_INSTANCE, _WORKER_CANDIDATES, _WORKER_END_DAY
    _WORKER_INSTANCE = instance
    _WORKER_CANDIDATES = candidates
    _WORKER_END_DAY = end_day
    context = multiprocessing.get_context("fork")
    with context.Pool(processes=min(workers, len(candidates))) as pool:
        indexed_scores = list(pool.imap_unordered(
            _score_candidate_index, range(len(candidates)), chunksize=1,
        ))
    indexed_scores.sort(key=lambda item: item[0])
    return [(candidates[index], item_score) for index, item_score in indexed_scores]


def accept_move(
    current: ContestScore, candidate: ContestScore,
    perturbation: int, rng: random.Random,
) -> bool:
    if score_key(candidate) < score_key(current):
        return True
    if perturbation <= 0 or candidate.hard_violations > current.hard_violations:
        return False
    if candidate.feasibility_errors > current.feasibility_errors + max(1, perturbation // 8):
        return False
    if candidate.safety_kg_min > current.safety_kg_min + perturbation * 50_000.0:
        return False
    return rng.random() >= 0.05


def hard_invariants_not_worse(
    before: ViolationVector, after: ViolationVector,
) -> bool:
    """Keep physical, service, and resource feasibility as commit invariants."""
    fields = (
        "non_finite_values", "reference_errors", "physical_errors",
        "missed_orders", "missed_order_deficit", "negative_quantity_minutes",
        "overfill_quantity_minutes", "resource_timing_errors", "other_errors",
    )
    return all(
        getattr(after, field) <= getattr(before, field) + 1e-6
        for field in fields
    )


def score_key(item: ContestScore) -> tuple[int, int, float, float]:
    return (
        item.hard_violations, item.feasibility_errors,
        item.safety_kg_min, logistic_ratio(item),
    )


def repair_key(
    instance: Instance, solution: Solution, item: ContestScore,
) -> tuple[int, int, int, float, float]:
    """Order infeasible seeds while retaining structural cleanup progress."""
    return (
        item.hard_violations, item.feasibility_errors,
        structural_shift_errors(instance, solution),
        item.safety_kg_min, logistic_ratio(item),
    )


def feasibility_key(item: ContestScore) -> tuple[int, int, float]:
    return item.hard_violations, item.feasibility_errors, item.safety_kg_min


def scalar_score(item: ContestScore) -> float:
    return (
        1_000_000 * item.hard_violations
        + 1_000 * item.feasibility_errors
        + item.safety_kg_min * 1e-5
        + logistic_ratio(item)
    )


def logistic_ratio(item: ContestScore) -> float:
    return item.scored_estimated_cost / max(1.0, item.scored_delivered_quantity)


def structural_shift_errors(instance: Instance, solution: Solution) -> int:
    return sum(
        1 for violation in validate_solution(instance, solution)
        if violation.severity == "error"
        and violation.shift is not None
        and violation.code not in {"QS01", "QS02"}
    )
