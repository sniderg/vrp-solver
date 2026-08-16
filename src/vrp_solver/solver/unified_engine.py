"""Unified native cold-start solver orchestration.

The constructor, exact quantity model, and topology search deliberately remain
separate components.  This module connects them through one fail-safe pipeline:

``construct frontier -> hard quantity repair -> block search -> full replay``.

Official verification remains an independent publication step because this API
accepts an in-memory :class:`Instance`, not the exact input/output XML paths.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from ..contest import score_prefix_with_feasibility_tail
from ..diagnostics import ViolationVector, solution_fingerprint, violation_vector
from ..highs_repair import repair_quantities_with_highs
from ..model import Instance, Solution
from ..rules import validate_solution
from .cluster_greedy import construct_cluster_solution
from .surgical_search import SurgicalSearchConfig, surgical_search


MINUTES_PER_DAY = 1_440


@dataclass(frozen=True)
class SolverResult:
    solution: Solution
    valid: bool
    errors: int
    warnings: int
    shifts: int
    runtime_seconds: float
    unscheduled_count: int
    provenance: str = "native-cold-start"
    validation_status: str = "locally_invalid"
    seeds_constructed: int = 0
    unique_constructed: int = 0
    quantity_repairs: int = 0
    search_steps: int = 0
    construction_strategy: str | None = None


@dataclass(frozen=True)
class _ConstructionStrategy:
    name: str
    seed: int = 0
    need_ordering: str = "scarcity"
    neighborhood_size: int | None = None
    long_window_urgency_override: bool = True
    proactive_reload_ratio: float = 0.40


@dataclass(frozen=True)
class _Candidate:
    solution: Solution
    vector: ViolationVector
    unscheduled: int
    seed: int
    strategy: str


def _construction_portfolio(num_seeds: int) -> tuple[_ConstructionStrategy, ...]:
    """Return deterministic structural strategies before random restarts.

    A random tie break is not a substitute for a different construction
    basin. These entries are general policies and depend on no instance name,
    customer ID, or copied route.
    """
    core = (
        _ConstructionStrategy(
            name="urgency-band",
            need_ordering="urgency-band",
            long_window_urgency_override=False,
        ),
        _ConstructionStrategy(
            name="urgency-band-narrow",
            need_ordering="urgency-band",
            neighborhood_size=3,
            long_window_urgency_override=False,
        ),
        _ConstructionStrategy(
            name="urgency-band-dense-reload",
            need_ordering="urgency-band",
            neighborhood_size=4,
            long_window_urgency_override=True,
            proactive_reload_ratio=0.48,
        ),
        _ConstructionStrategy(name="scarcity-chain"),
    )
    strategies = list(core[:num_seeds])
    random_seed = 1
    while len(strategies) < num_seeds:
        strategies.append(_ConstructionStrategy(
            name=f"scarcity-chain-seed-{random_seed}", seed=random_seed,
        ))
        random_seed += 1
    return tuple(strategies)


def solve_cold_start(
    instance: Instance,
    *,
    num_seeds: int = 10,
    time_limit: float = 60.0,
    frontier_size: int = 3,
    search_iterations: int = 64,
    search_workers: int = 1,
    stop_when_feasible: bool = True,
) -> SolverResult:
    """Construct and improve a solution using only ``instance`` data.

    ``valid`` means locally feasible after a complete replay.  It intentionally
    does not mean officially verified; callers must serialize the exact XML and
    use :func:`vrp_solver.official_verify.verify_v2_solution` for that claim.
    """
    if num_seeds < 1:
        raise ValueError("num_seeds must be at least 1")
    if frontier_size < 1:
        raise ValueError("frontier_size must be at least 1")
    if time_limit <= 0:
        raise ValueError("time_limit must be positive")

    started = time.monotonic()
    deadline = started + time_limit
    horizon_days = max(
        1, math.ceil(instance.horizon * instance.unit / MINUTES_PER_DAY),
    )

    constructed: list[_Candidate] = []
    seen: set[str] = set()
    seeds_attempted = 0
    for attempt, strategy in enumerate(_construction_portfolio(num_seeds)):
        if attempt > 0 and time.monotonic() >= deadline:
            break
        solution, report = construct_cluster_solution(
            instance,
            tie_break_seed=strategy.seed,
            need_ordering=strategy.need_ordering,
            neighborhood_size=strategy.neighborhood_size,
            long_window_urgency_override=strategy.long_window_urgency_override,
            proactive_reload_ratio=strategy.proactive_reload_ratio,
        )
        seeds_attempted += 1
        fingerprint = solution_fingerprint(solution)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        candidate = _Candidate(
            solution=solution,
            vector=violation_vector(instance, solution),
            unscheduled=len(report.unscheduled_customers),
            seed=strategy.seed,
            strategy=strategy.name,
        )
        constructed.append(candidate)
        if stop_when_feasible and candidate.vector.locally_feasible:
            break

    # Raw error counts hide whether a candidate has one tiny safety breach or
    # a horizon-long stockout.  Preserve a small, diverse frontier ordered by
    # the structured full-replay diagnostic instead.
    constructed.sort(key=lambda item: (
        item.vector.key(), item.unscheduled, _local_ratio(instance, item.solution, horizon_days), item.strategy,
    ))
    frontier = constructed[:frontier_size]
    best = frontier[0].solution
    best_unscheduled = frontier[0].unscheduled
    best_strategy = frontier[0].strategy
    best_key = _candidate_key(instance, best, best_unscheduled, horizon_days)

    quantity_repairs = 0
    for item in frontier:
        if stop_when_feasible and violation_vector(instance, best).locally_feasible:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        repaired, report = repair_quantities_with_highs(
            instance,
            item.solution,
            score_days=horizon_days,
            feasibility_days=horizon_days,
            quantity_objective="max-delivered",
            strict_inventory=True,
            time_limit_seconds=remaining,
        )
        quantity_repairs += 1
        if report.status != "Optimal":
            continue
        key = _candidate_key(instance, repaired, item.unscheduled, horizon_days)
        if key < best_key:
            best, best_unscheduled, best_strategy, best_key = (
                repaired, item.unscheduled, item.strategy, key,
            )
        if stop_when_feasible and violation_vector(instance, repaired).locally_feasible:
            best = repaired
            best_unscheduled = item.unscheduled
            best_strategy = item.strategy
            best_key = key
            break

    # The surgical search is the connectivity layer: its operator portfolio
    # reaches pressure blocks, ejections, route recombination, multi-reloads,
    # joint resource retiming, and hard quantity repair transactionally.
    steps = ()
    remaining = deadline - time.monotonic()
    if search_iterations > 0 and remaining > 0 and not violation_vector(instance, best).locally_feasible:
        best_vector = violation_vector(instance, best)
        safety_only = (
            best_vector.non_finite_values == 0
            and best_vector.reference_errors == 0
            and best_vector.physical_errors == 0
            and best_vector.missed_orders == 0
            and best_vector.negative_quantity_minutes == 0
            and best_vector.overfill_quantity_minutes == 0
            and best_vector.resource_timing_errors == 0
            and best_vector.other_errors == 0
            and best_vector.safety_deficit_quantity_minutes > 0
        )
        searched, steps = surgical_search(
            instance,
            best,
            config=SurgicalSearchConfig(
                end_day=horizon_days,
                iterations=search_iterations,
                seed=next(
                    item.seed for item in frontier
                    if item.strategy == best_strategy
                ),
                time_limit_seconds=remaining,
                workers=max(1, search_workers),
                first_operator=(
                    "pressure_band_resource_block" if safety_only else None
                ),
            ),
            progress=None,
        )
        searched_key = _candidate_key(instance, searched, best_unscheduled, horizon_days)
        if searched_key < best_key:
            best, best_key = searched, searched_key

    violations = validate_solution(instance, best)
    errors = sum(item.severity == "error" for item in violations)
    warnings = sum(item.severity == "warning" for item in violations)
    return SolverResult(
        solution=best,
        valid=errors == 0,
        errors=errors,
        warnings=warnings,
        shifts=len(best.shifts),
        runtime_seconds=round(time.monotonic() - started, 2),
        unscheduled_count=best_unscheduled,
        validation_status="locally_feasible" if errors == 0 else "locally_invalid",
        seeds_constructed=seeds_attempted,
        unique_constructed=len(constructed),
        quantity_repairs=quantity_repairs,
        search_steps=len(steps),
        construction_strategy=best_strategy,
    )


def _candidate_key(
    instance: Instance,
    solution: Solution,
    unscheduled: int,
    horizon_days: int,
) -> tuple[object, ...]:
    return (
        violation_vector(instance, solution).key(),
        unscheduled,
        _local_ratio(instance, solution, horizon_days),
        solution_fingerprint(solution),
    )


def _local_ratio(instance: Instance, solution: Solution, horizon_days: int) -> float:
    score = score_prefix_with_feasibility_tail(
        instance,
        solution,
        score_days=horizon_days,
        feasibility_days=horizon_days,
    )
    if score.scored_delivered_quantity <= 0:
        return math.inf
    return score.scored_estimated_cost / score.scored_delivered_quantity
