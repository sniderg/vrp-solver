from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ..contest import ContestScore, score_prefix_with_feasibility_tail
from ..model import Instance, Solution
from .column_loop import ColumnLoopConfig, column_generation_rescue
from .destroy import (
    DestroyConfig,
    DestroyResult,
    pressure_band_destroy,
    related_customer_destroy,
    resource_conflict_destroy,
    route_block_destroy,
)
from ..xml_io import save_solution


@dataclass(frozen=True)
class ALNSConfig:
    start_day: int = 0
    end_day: int = 21
    replace_from_day: int = 3
    iterations: int = 20
    repair_iterations: int = 2
    seed: int = 0
    initial_temperature: float = 5000.0
    cooling_rate: float = 0.92
    max_removed_shifts: int = 8
    related_customer_count: int = 8
    time_band_days: int = 3
    max_pressure_customers: int = 12
    samples_per_customer: int = 6
    sample_lookback_days: int = 14
    max_candidates_per_iteration: int = 700
    target_fill_ratio: float = 0.95
    nearest_chain_neighbors: int = 4
    multi_reload_columns: bool = False
    max_multi_reload_per_batch: int = 8
    normalize_source_loads: bool = True
    quantity_objective: str = "max-delivered"
    output_xml: str | None = None


@dataclass(frozen=True)
class ALNSStep:
    iteration: int
    operator: str
    removed_shifts: int
    accepted: bool
    new_best: bool
    current_errors: int
    current_hard: int
    best_errors: int
    best_hard: int
    first_safety_breach_minute: int | None


def alns_rescue(
    instance: Instance,
    initial: Solution,
    *,
    config: ALNSConfig = ALNSConfig(),
) -> tuple[Solution, tuple[ALNSStep, ...]]:
    rng = random.Random(config.seed)
    destroy_config = DestroyConfig(
        end_day=config.end_day,
        replace_from_day=config.replace_from_day,
        max_removed_shifts=config.max_removed_shifts,
        related_customer_count=config.related_customer_count,
        time_band_days=config.time_band_days,
    )
    operators = [
        ("resource_conflict", resource_conflict_destroy),
        ("pressure_band", pressure_band_destroy),
        ("related_customer", related_customer_destroy),
        ("route_block", route_block_destroy),
    ]
    weights = {name: 1.0 for name, _op in operators}
    weights["resource_conflict"] = 2.0

    current = initial
    current_score = _score(instance, current, config)
    print(f"DEBUG: Initial score: feasible={current_score.feasible}, hard={current_score.hard_violations}, errors={current_score.feasibility_errors}, cost={current_score.scored_estimated_cost}")
    best = current
    best_score = current_score
    temperature = config.initial_temperature
    steps: list[ALNSStep] = []

    for iteration in range(config.iterations):
        name, destroy = _select_operator(operators, weights, rng)
        destroyed = destroy(instance, current, rng, config=destroy_config)
        if not destroyed.removed_shifts:
            if iteration % 100 == 0:
                print(f"DEBUG: Iteration {iteration}: Operator {name} removed 0 shifts")
            weights[name] *= 0.95
            continue
        candidate = _repair(instance, destroyed, config)
        candidate_score = _score(instance, candidate, config)
        if iteration % 10 == 0:
             print(f"DEBUG: Iteration {iteration}: Candidate score: feasible={candidate_score.feasible}, hard={candidate_score.hard_violations}, errors={candidate_score.feasibility_errors}, cost={candidate_score.scored_estimated_cost}")
        previous_score = current_score
        accepted = _accept(current_score, candidate_score, temperature, rng)
        new_best = _score_key(candidate_score) < _score_key(best_score)

        if accepted:
            current = candidate
            current_score = candidate_score
        if new_best:
            best = candidate
            best_score = candidate_score
            if config.output_xml:
                try:
                    save_solution(best, config.output_xml)
                except Exception as e:
                    print(f"DEBUG: Failed to save intermediate solution to {config.output_xml}: {e}")

        if new_best:
            weights[name] += 5.0
        elif accepted and _score_key(candidate_score) < _score_key(previous_score):
            weights[name] += 2.0
        elif accepted:
            weights[name] += 0.5
        else:
            weights[name] *= 0.98

        steps.append(
            ALNSStep(
                iteration=iteration,
                operator=destroyed.operator,
                removed_shifts=len(destroyed.removed_shifts),
                accepted=accepted,
                new_best=new_best,
                current_errors=current_score.feasibility_errors,
                current_hard=current_score.hard_violations,
                best_errors=best_score.feasibility_errors,
                best_hard=best_score.hard_violations,
                first_safety_breach_minute=best_score.first_safety_breach_minute,
            )
        )
        temperature *= config.cooling_rate

    return best, tuple(steps)


def _repair(instance: Instance, destroyed: DestroyResult, config: ALNSConfig) -> Solution:
    replace_from_day = max(config.replace_from_day, destroyed.start_minute // 1440)
    repair_config = ColumnLoopConfig(
        start_day=config.start_day,
        end_day=config.end_day,
        replace_from_day=replace_from_day,
        iterations=config.repair_iterations,
        max_pressure_customers=config.max_pressure_customers,
        samples_per_customer=config.samples_per_customer,
        sample_lookback_days=config.sample_lookback_days,
        max_candidates_per_iteration=config.max_candidates_per_iteration,
        target_fill_ratio=config.target_fill_ratio,
        nearest_chain_neighbors=config.nearest_chain_neighbors,
        multi_reload_columns=config.multi_reload_columns,
        max_multi_reload_per_batch=config.max_multi_reload_per_batch,
        normalize_source_loads=config.normalize_source_loads,
        quantity_objective=config.quantity_objective,
    )
    repaired, _steps = column_generation_rescue(instance, destroyed.solution, config=repair_config)
    return repaired


def _score(instance: Instance, solution: Solution, config: ALNSConfig) -> ContestScore:
    return score_prefix_with_feasibility_tail(
        instance,
        solution,
        score_days=config.end_day,
        feasibility_days=config.end_day,
    )


def _select_operator(operators, weights, rng: random.Random):
    total = sum(max(0.01, weights[name]) for name, _op in operators)
    pick = rng.random() * total
    running = 0.0
    for name, operator in operators:
        running += max(0.01, weights[name])
        if running >= pick:
            return name, operator
    return operators[-1]


def _accept(
    incumbent: ContestScore,
    candidate: ContestScore,
    temperature: float,
    rng: random.Random,
) -> bool:
    delta = _scalar_score(candidate) - _scalar_score(incumbent)
    if delta <= 0:
        return True
    if temperature <= 1e-9:
        return False
    return rng.random() < math.exp(-delta / temperature)


def _score_key(score: ContestScore) -> tuple[int, int, int, float]:
    return (
        0 if score.feasible else 1,
        score.hard_violations,
        score.feasibility_errors,
        score.scored_estimated_cost,
    )


def _scalar_score(score: ContestScore) -> float:
    return (
        1_000_000_000.0 * (0 if score.feasible else 1)
        + 1_000_000.0 * score.hard_violations
        + 10_000.0 * score.feasibility_errors
        + score.safety_kg_min
        + score.scored_estimated_cost
    )
