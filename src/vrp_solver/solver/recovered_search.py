"""Native reconstruction of Solver.exe's adaptive seven-neighborhood loop.

The original Windows binary dispatches seven opaque route/inventory mutations
through a table, learns their utility, and periodically restores its best
solution.  This module reproduces that controller on the native IRP mutations
that have explicit, auditable implementations in this repository.  The four
repair neighborhoods use the configured HiGHS/Gurobi selector; set
``ROADEF_SOLVER=gurobi`` to use Gurobi 13.
"""
from __future__ import annotations

import random
import time
import math
from dataclasses import dataclass
from typing import Callable

from ..contest import ContestScore, score_prefix_with_feasibility_tail
from ..improve import (
    move_single_customer_shifts,
    prune_redundant_shifts,
    remove_redundant_source_visits,
    trim_redundant_deliveries,
)
from ..model import Instance, Solution
from .alns import ALNSConfig, _repair
from .destroy import (
    DestroyConfig,
    pressure_band_destroy,
    related_customer_destroy,
    resource_conflict_destroy,
    route_block_destroy,
)


@dataclass(frozen=True)
class RecoveredSearchConfig:
    end_day: int = 21
    replace_from_day: int = 0
    iterations: int = 100
    repair_iterations: int = 2
    seed: int = 0
    max_no_improve: int = 32
    restore_every: int = 4
    max_removed_shifts: int = 8
    related_customer_count: int = 8
    time_band_days: int = 3
    max_pressure_customers: int = 12
    samples_per_customer: int = 6
    sample_lookback_days: int = 14
    max_candidates_per_iteration: int = 700
    target_fill_ratio: float = 0.95
    nearest_chain_neighbors: int = 4


@dataclass(frozen=True)
class RecoveredSearchStep:
    iteration: int
    operator: str
    accepted: bool
    improved_best: bool
    no_improve: int
    candidate_errors: int
    candidate_hard: int
    best_errors: int
    best_hard: int


@dataclass(frozen=True)
class PowerfulRepairConfig:
    """Escalating ALNS derived from the controller recovered from Solver.exe."""

    end_day: int = 21
    replace_from_day: int = 0
    iterations: int = 80
    seed: int = 0
    time_limit_seconds: float | None = None
    base_removed_shifts: int = 6
    max_removed_shifts: int = 24
    base_pressure_customers: int = 10
    max_pressure_customers: int = 32
    base_candidates: int = 700
    max_candidates: int = 2600
    selector_time_limit: float = 90.0
    stagnation_window: int = 4
    restart_every: int = 12
    elite_size: int = 4
    exploration: float = 1.25
    initial_temperature: float = 2.0
    cooling_rate: float = 0.96


@dataclass(frozen=True)
class PowerfulRepairStep:
    iteration: int
    operator: str
    level: int
    accepted: bool
    improved_best: bool
    candidate_errors: int
    candidate_hard: int
    candidate_safety_kg_min: float
    best_errors: int
    best_hard: int
    best_safety_kg_min: float


def powerful_repair_search(
    instance: Instance,
    initial: Solution,
    *,
    config: PowerfulRepairConfig = PowerfulRepairConfig(),
    progress: Callable[[str], None] | None = print,
) -> tuple[Solution, tuple[PowerfulRepairStep, ...]]:
    """Run an adaptive, severity-aware, escalating destroy/repair search.

    The EXE-inspired controller is retained, but operator choice uses UCB so
    under-tested neighborhoods cannot disappear.  Stagnation widens both the
    destroyed region and the Gurobi column-generation repair model.
    """
    rng = random.Random(config.seed)
    deadline = (
        None if config.time_limit_seconds is None
        else time.monotonic() + config.time_limit_seconds
    )
    operators = (
        "resource_conflict_repair",
        "pressure_band_repair",
        "related_customer_repair",
        "route_block_repair",
    )
    pulls = {name: 0 for name in operators}
    rewards = {name: 0.0 for name in operators}
    current = initial
    current_score = _score(instance, current, config.end_day)
    best, best_score = current, current_score
    elite: list[tuple[tuple, Solution, ContestScore]] = [(_severity_key(best_score), best, best_score)]
    stagnation = 0
    temperature = config.initial_temperature
    steps: list[PowerfulRepairStep] = []

    for iteration in range(config.iterations):
        if deadline is not None and time.monotonic() >= deadline:
            break
        level = min(3, stagnation // max(1, config.stagnation_window))
        operator = _ucb_select(operators, pulls, rewards, iteration, config.exploration)
        candidate = _powerful_apply(instance, current, operator, level, config, rng)
        candidate_score = _score(instance, candidate, config.end_day)
        before = _severity_scalar(current_score)
        after = _severity_scalar(candidate_score)
        delta = after - before
        accepted = delta <= 0 or (
            temperature > 1e-9 and rng.random() < math.exp(-min(delta, 50.0) / temperature)
        )
        improved_best = _severity_key(candidate_score) < _severity_key(best_score)
        pulls[operator] += 1
        relative_gain = max(-1.0, min(5.0, (before - after) / max(1.0, abs(before))))
        bonus = 3.0 if improved_best else (1.0 if accepted and delta < 0 else 0.0)
        rewards[operator] += relative_gain + bonus

        if accepted:
            current, current_score = candidate, candidate_score
        if improved_best:
            best, best_score = candidate, candidate_score
            stagnation = 0
            elite.append((_severity_key(candidate_score), candidate, candidate_score))
            elite = sorted(elite, key=lambda item: item[0])[: max(1, config.elite_size)]
        else:
            stagnation += 1

        # The recovered binary restores stored incumbents.  We strengthen that
        # behavior with a small elite pool, alternating intensification and
        # diversification instead of always returning to one solution.
        if stagnation and stagnation % max(1, config.restart_every) == 0:
            _, current, current_score = elite[rng.randrange(len(elite))]
            temperature = config.initial_temperature
        else:
            temperature *= config.cooling_rate

        step = PowerfulRepairStep(
            iteration=iteration,
            operator=operator,
            level=level,
            accepted=accepted,
            improved_best=improved_best,
            candidate_errors=candidate_score.feasibility_errors,
            candidate_hard=candidate_score.hard_violations,
            candidate_safety_kg_min=candidate_score.safety_kg_min,
            best_errors=best_score.feasibility_errors,
            best_hard=best_score.hard_violations,
            best_safety_kg_min=best_score.safety_kg_min,
        )
        steps.append(step)
        if progress is not None:
            progress(
                "repair_step,"
                f"{iteration},operator,{operator},level,{level},accepted,{accepted},"
                f"new_best,{improved_best},candidate_errors,{step.candidate_errors},"
                f"candidate_deficit,{step.candidate_safety_kg_min:.3f},"
                f"best_errors,{step.best_errors},best_deficit,{step.best_safety_kg_min:.3f}"
            )
        if best_score.feasible:
            break
    return best, tuple(steps)


def _powerful_apply(
    instance: Instance,
    current: Solution,
    operator: str,
    level: int,
    config: PowerfulRepairConfig,
    rng: random.Random,
) -> Solution:
    scale = (1.0, 1.5, 2.25, 3.25)[level]
    removed = min(config.max_removed_shifts, max(1, round(config.base_removed_shifts * scale)))
    pressure = min(
        config.max_pressure_customers,
        max(1, round(config.base_pressure_customers * scale)),
    )
    candidates = min(config.max_candidates, max(100, round(config.base_candidates * scale)))
    destroy_config = DestroyConfig(
        end_day=config.end_day,
        replace_from_day=config.replace_from_day,
        max_removed_shifts=removed,
        related_customer_count=pressure,
        time_band_days=min(8, 2 + level * 2),
    )
    destroyers: dict[str, Callable[..., object]] = {
        "resource_conflict_repair": resource_conflict_destroy,
        "pressure_band_repair": pressure_band_destroy,
        "related_customer_repair": related_customer_destroy,
        "route_block_repair": route_block_destroy,
    }
    destroyed = destroyers[operator](instance, current, rng, config=destroy_config)
    if not destroyed.removed_shifts:
        return current
    repair_config = ALNSConfig(
        end_day=config.end_day,
        replace_from_day=config.replace_from_day,
        repair_iterations=1 + level,
        max_removed_shifts=removed,
        related_customer_count=pressure,
        time_band_days=destroy_config.time_band_days,
        max_pressure_customers=pressure,
        samples_per_customer=6 + 2 * level,
        max_candidates_per_iteration=candidates,
        nearest_chain_neighbors=4 + level,
        multi_reload_columns=level >= 2,
        max_multi_reload_per_batch=8 + 6 * level,
        selector_time_limit=config.selector_time_limit,
    )
    return _repair(instance, destroyed, repair_config)


def _ucb_select(names, pulls, rewards, iteration: int, exploration: float) -> str:
    untried = [name for name in names if pulls[name] == 0]
    if untried:
        return untried[0]
    log_total = math.log(max(2, iteration + 1))
    return max(
        names,
        key=lambda name: (
            rewards[name] / pulls[name] + exploration * math.sqrt(log_total / pulls[name]),
            -pulls[name],
            name,
        ),
    )


def recovered_adaptive_search(
    instance: Instance,
    initial: Solution,
    *,
    config: RecoveredSearchConfig = RecoveredSearchConfig(),
    time_limit_seconds: float | None = None,
) -> tuple[Solution, tuple[RecoveredSearchStep, ...]]:
    """Run the recovered adaptive controller with seven named neighborhoods.

    It follows the executable's observable behavior: weighted operator
    selection, exponentially-smoothed rewards, no-improvement perturbation,
    incumbent restoration, and strict feasibility-first acceptance.
    """
    rng = random.Random(config.seed)
    deadline = None if time_limit_seconds is None else time.monotonic() + time_limit_seconds
    current = initial
    current_score = _score(instance, current, config.end_day)
    best, best_score = current, current_score
    names = (
        "resource_conflict_repair",
        "pressure_band_repair",
        "related_customer_repair",
        "route_block_repair",
        "prune_redundant_shift",
        "trim_redundant_delivery",
        "absorb_single_customer_shift",
    )
    weights = {name: 1.0 for name in names}
    last_used = {name: -1 for name in names}
    no_improve = 0
    steps: list[RecoveredSearchStep] = []

    for iteration in range(config.iterations):
        if deadline is not None and time.monotonic() >= deadline:
            break
        name = _select(names, weights, last_used, rng)
        candidate = _apply(instance, current, name, config, rng)
        candidate_score = _score(instance, candidate, config.end_day)
        last_used[name] = iteration
        accepted = _key(candidate_score) < _key(current_score)
        improved_best = _key(candidate_score) < _key(best_score)
        reward = 0.0
        if accepted:
            current, current_score = candidate, candidate_score
            reward = 2.0
        if improved_best:
            best, best_score = candidate, candidate_score
            reward = 5.0
            no_improve = 0
        else:
            no_improve += 1
        weights[name] = max(0.05, 0.70 * weights[name] + 0.30 * reward)

        # The binary periodically restores a stored incumbent after failed
        # mutations; doing so here keeps destructive repair neighborhoods safe.
        if no_improve and no_improve % max(1, config.restore_every) == 0:
            current, current_score = best, best_score
        if no_improve >= config.max_no_improve:
            current, current_score = best, best_score
            no_improve = 0

        steps.append(RecoveredSearchStep(
            iteration=iteration,
            operator=name,
            accepted=accepted,
            improved_best=improved_best,
            no_improve=no_improve,
            candidate_errors=candidate_score.feasibility_errors,
            candidate_hard=candidate_score.hard_violations,
            best_errors=best_score.feasibility_errors,
            best_hard=best_score.hard_violations,
        ))
    return best, tuple(steps)


def _apply(instance: Instance, current: Solution, name: str, config: RecoveredSearchConfig, rng: random.Random) -> Solution:
    destroy_config = DestroyConfig(
        end_day=config.end_day,
        replace_from_day=config.replace_from_day,
        max_removed_shifts=config.max_removed_shifts,
        related_customer_count=config.related_customer_count,
        time_band_days=config.time_band_days,
    )
    repair_config = ALNSConfig(
        end_day=config.end_day,
        replace_from_day=config.replace_from_day,
        repair_iterations=config.repair_iterations,
        max_pressure_customers=config.max_pressure_customers,
        samples_per_customer=config.samples_per_customer,
        sample_lookback_days=config.sample_lookback_days,
        max_candidates_per_iteration=config.max_candidates_per_iteration,
        target_fill_ratio=config.target_fill_ratio,
        nearest_chain_neighbors=config.nearest_chain_neighbors,
    )
    destroyers: dict[str, Callable[..., object]] = {
        "resource_conflict_repair": resource_conflict_destroy,
        "pressure_band_repair": pressure_band_destroy,
        "related_customer_repair": related_customer_destroy,
        "route_block_repair": route_block_destroy,
    }
    if name in destroyers:
        destroyed = destroyers[name](instance, current, rng, config=destroy_config)
        return _repair(instance, destroyed, repair_config)
    if name == "prune_redundant_shift":
        return prune_redundant_shifts(instance, current, score_days=config.end_day)[0]
    if name == "trim_redundant_delivery":
        return trim_redundant_deliveries(instance, current, score_days=config.end_day)
    return move_single_customer_shifts(instance, current, score_days=config.end_day)


def _select(names, weights, last_used, rng: random.Random) -> str:
    # Prefer operators with learned high reward; break close calls with the
    # least recently used one, then sample proportionally as the EXE does.
    ranked = sorted(names, key=lambda name: (-weights[name], last_used[name], name))
    pool = ranked[:max(1, min(3, len(ranked)))]
    total = sum(weights[name] for name in pool)
    draw = rng.random() * total
    for name in pool:
        draw -= weights[name]
        if draw <= 0:
            return name
    return pool[-1]


def _score(instance: Instance, solution: Solution, end_day: int) -> ContestScore:
    return score_prefix_with_feasibility_tail(
        instance, solution, score_days=end_day, feasibility_days=end_day, ignore_tail_call_ins=True
    )


def _key(score: ContestScore) -> tuple[int, int, int, float]:
    return (0 if score.feasible else 1, score.hard_violations, score.feasibility_errors, score.scored_estimated_cost)


def _severity_key(score: ContestScore) -> tuple[int, int, int, float, float]:
    return (
        0 if score.feasible else 1,
        score.hard_violations,
        score.feasibility_errors,
        score.safety_kg_min,
        score.scored_estimated_cost,
    )


def _severity_scalar(score: ContestScore) -> float:
    # Log compression keeps a severe early deficit important without making
    # simulated-annealing acceptance numerically impossible.
    return (
        1_000_000.0 * (0 if score.feasible else 1)
        + 10_000.0 * score.hard_violations
        + 100.0 * score.feasibility_errors
        + math.log1p(max(0.0, score.safety_kg_min))
        + 1e-6 * max(0.0, score.scored_estimated_cost)
    )
