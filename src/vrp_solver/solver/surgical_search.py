"""Transactional local search reconstructed from the original Solver.exe."""
from __future__ import annotations

import random
import time
import multiprocessing
import logging
from dataclasses import dataclass, replace
from typing import Callable

from ..contest import ContestScore, score_prefix_with_feasibility_tail
from ..highs_time_opt import optimize_shift_times
from ..model import Instance, Operation, Shift, Solution
from ..rules import derive_solution
from .pressure import pressure_points
from .targeted_rescue import (
    RescueConfig,
    generate_chain_rescue_candidates,
    generate_rescue_candidates,
)


@dataclass(frozen=True)
class SurgicalSearchConfig:
    end_day: int
    iterations: int = 500
    candidates_per_move: int = 300
    pressure_customers: int = 20
    samples_per_customer: int = 12
    seed: int = 0
    time_limit_seconds: float | None = None
    no_improvement_limit: int = 64
    workers: int = 6
    first_operator: str | None = None
    output_xml: str | None = None


@dataclass(frozen=True)
class SurgicalStep:
    iteration: int
    operator: str
    evaluated: int
    accepted: bool
    errors: int
    hard: int
    safety_kg_min: float
    cost: float
    logistic_ratio: float


OPERATORS = (
    "create_shift",
    "insert_operation",
    "delete_operation",
    "replace_operation_point",
    "swap_operations",
    "relocate_between_shifts",
    "relocate_within_shift",
)

logging.getLogger("vrp_solver.highs_time_opt").setLevel(logging.ERROR)


def surgical_search(
    instance: Instance,
    initial: Solution,
    *,
    config: SurgicalSearchConfig,
    progress: Callable[[str], None] | None = print,
) -> tuple[Solution, tuple[SurgicalStep, ...]]:
    """Apply the recovered seven-move portfolio with transactional rollback."""
    from ..xml_io import save_solution

    rng = random.Random(config.seed)
    deadline = (
        None if config.time_limit_seconds is None
        else time.monotonic() + config.time_limit_seconds
    )
    current = _reindex(initial)
    score = _score(instance, current, config.end_day)
    rewards = [1.0] * 7
    attempts = [0] * 7
    last_used = [-10_000] * 7
    stagnation = 0
    steps: list[SurgicalStep] = []

    for iteration in range(config.iterations):
        if deadline is not None and time.monotonic() >= deadline:
            break
        # Match the EXE's recency-aware adaptive portfolio. Force creation and
        # insertion early while inventory is infeasible.
        if iteration == 0 and config.first_operator in OPERATORS:
            operator_index = OPERATORS.index(config.first_operator)
        else:
            operator_index = _select_operator(
                rewards, attempts, last_used, iteration, rng,
                feasibility_bias=not score.feasible,
            )
        operator = OPERATORS[operator_index]
        if progress:
            progress(
                f"surgical_start,{iteration},operator,{operator},"
                f"errors,{score.feasibility_errors},"
                f"deficit,{score.safety_kg_min:.3f}"
            )
        candidates = _candidates(instance, current, operator, config, rng)
        # The EXE perturbs enumeration after stagnation. Sampling a seeded
        # permutation gives the capped native scorer the same expanding
        # coverage instead of repeatedly testing the first candidate block.
        rng.shuffle(candidates)
        if progress:
            progress(
                f"surgical_generated,{iteration},operator,{operator},"
                f"candidates,{len(candidates)}"
            )
        best_candidate = current
        best_score = score
        evaluated = 0
        limited_candidates = candidates[: config.candidates_per_move]
        scored = _score_candidates(
            instance, limited_candidates, config.end_day, config.workers,
        )
        for candidate, candidate_score in scored:
            evaluated += 1
            if _key(candidate_score) < _key(best_score):
                best_candidate, best_score = candidate, candidate_score
                if progress:
                    progress(
                        f"surgical_candidate,{iteration},operator,{operator},evaluated,{evaluated},"
                        f"errors,{best_score.feasibility_errors},hard,{best_score.hard_violations},"
                        f"deficit,{best_score.safety_kg_min:.3f}"
                    )

        accepted = _key(best_score) < _key(score)
        previous_scalar = _scalar(score)
        if accepted:
            current, score = best_candidate, best_score
            stagnation = 0
            gain = max(0.0, previous_scalar - _scalar(score))
            rewards[operator_index] = 0.5 * rewards[operator_index] + 0.5 * min(3584.0, 1.0 + gain)
            if config.output_xml:
                save_solution(current, config.output_xml)
        else:
            stagnation += 1
            rewards[operator_index] *= 0.5
        attempts[operator_index] += 1
        last_used[operator_index] = iteration

        step = SurgicalStep(
            iteration, operator, evaluated, accepted,
            score.feasibility_errors, score.hard_violations,
            score.safety_kg_min, score.scored_estimated_cost,
            score.scored_estimated_cost / max(1.0, score.scored_delivered_quantity),
        )
        steps.append(step)
        if progress:
            progress(
                f"surgical_step,{iteration},operator,{operator},evaluated,{evaluated},"
                f"accepted,{accepted},errors,{step.errors},hard,{step.hard},"
                f"deficit,{step.safety_kg_min:.3f},cost,{step.cost:.3f},"
                f"lr,{step.logistic_ratio:.10f}"
            )
        if score.feasible or stagnation >= config.no_improvement_limit:
            break
    return current, tuple(steps)


def _candidates(
    instance: Instance,
    solution: Solution,
    operator: str,
    config: SurgicalSearchConfig,
    rng: random.Random,
) -> list[Solution]:
    if operator == "create_shift":
        return _create_shift_candidates(instance, solution, config)
    if operator == "insert_operation":
        return _insert_operation_candidates(instance, solution, config)
    if operator == "delete_operation":
        return _delete_operation_candidates(instance, solution, config, rng)
    if operator == "replace_operation_point":
        return _replace_point_candidates(instance, solution, config, rng)
    if operator == "swap_operations":
        return _swap_candidates(instance, solution, config, rng)
    if operator == "relocate_between_shifts":
        return _between_shift_candidates(instance, solution, config, rng)
    return _within_shift_candidates(instance, solution, config, rng)


def _create_shift_candidates(instance, solution, config) -> list[Solution]:
    pressure = [p.customer for p in pressure_points(instance, solution, end_day=config.end_day)]
    pressure = pressure[: config.pressure_customers]
    rescue = RescueConfig(
        start_day=0,
        end_day=config.end_day,
        replace_from_day=0,
        max_customers=config.pressure_customers,
        samples_per_customer=config.samples_per_customer,
        max_chain_length=3,
        nearest_chain_neighbors=5,
        target_fill_ratio=0.98,
    )
    shifts = generate_rescue_candidates(instance, solution, pressure, config=rescue)
    shifts += generate_chain_rescue_candidates(instance, solution, pressure, config=rescue)
    return [_reindex(Solution((*solution.shifts, replace(shift, index=len(solution.shifts)))))
            for shift in shifts]


def _insert_operation_candidates(instance, solution, config) -> list[Solution]:
    pressure = pressure_points(instance, solution, end_day=config.end_day)
    derived = derive_solution(instance, solution)
    result: list[Solution] = []
    for point in pressure[: config.pressure_customers]:
        customer = instance.customer_by_point[point.customer]
        for shift_pos, shift in enumerate(solution.shifts):
            if shift.start >= point.first_minute or shift.trailer not in customer.allowed_trailers:
                continue
            for op_pos in range(1, len(shift.operations) + 1):
                available = derived[shift_pos].operations[op_pos - 1].trailer_quantity
                quantity = min(available, customer.capacity * 0.5, 20_000.0)
                if quantity < customer.min_operation_quantity:
                    continue
                operations = list(shift.operations)
                anchor = operations[op_pos - 1]
                operations.insert(op_pos, Operation(customer.index, anchor.arrival, quantity))
                mutated = optimize_shift_times(instance, replace(shift, operations=tuple(operations)))
                shifts = list(solution.shifts)
                shifts[shift_pos] = mutated
                result.append(Solution(tuple(shifts)))
                if len(result) >= config.candidates_per_move * 2:
                    return result
    return result


def _delete_operation_candidates(instance, solution, config, rng) -> list[Solution]:
    result = []
    positions = [(s, o) for s, shift in enumerate(solution.shifts)
                 for o in range(len(shift.operations))]
    rng.shuffle(positions)
    for s, o in positions[: config.candidates_per_move * 2]:
        shift = solution.shifts[s]
        operations = list(shift.operations)
        operations.pop(o)
        shifts = list(solution.shifts)
        if operations:
            shifts[s] = optimize_shift_times(instance, replace(shift, operations=tuple(operations)))
        else:
            shifts.pop(s)
        result.append(_reindex(Solution(tuple(shifts))))
    return result


def _replace_point_candidates(instance, solution, config, rng) -> list[Solution]:
    targets = [p.customer for p in pressure_points(instance, solution, end_day=config.end_day)]
    result = []
    positions = [(s, o) for s, shift in enumerate(solution.shifts)
                 for o, op in enumerate(shift.operations) if op.quantity > 0]
    rng.shuffle(positions)
    for s, o in positions:
        shift = solution.shifts[s]
        for target in targets[: config.pressure_customers]:
            customer = instance.customer_by_point[target]
            if shift.trailer not in customer.allowed_trailers:
                continue
            operations = list(shift.operations)
            old = operations[o]
            operations[o] = replace(old, point=target, quantity=min(old.quantity, customer.capacity))
            shifts = list(solution.shifts)
            shifts[s] = optimize_shift_times(instance, replace(shift, operations=tuple(operations)))
            result.append(Solution(tuple(shifts)))
            if len(result) >= config.candidates_per_move * 2:
                return result
    return result


def _swap_candidates(instance, solution, config, rng) -> list[Solution]:
    result = []
    positions = [(s, a, b) for s, shift in enumerate(solution.shifts)
                 for a in range(len(shift.operations)) for b in range(a + 1, len(shift.operations))]
    rng.shuffle(positions)
    for s, a, b in positions[: config.candidates_per_move * 2]:
        shift = solution.shifts[s]
        operations = list(shift.operations)
        operations[a], operations[b] = operations[b], operations[a]
        shifts = list(solution.shifts)
        shifts[s] = optimize_shift_times(instance, replace(shift, operations=tuple(operations)))
        result.append(Solution(tuple(shifts)))
    return result


def _between_shift_candidates(instance, solution, config, rng) -> list[Solution]:
    result = []
    moves = [(a, o, b) for a, shift in enumerate(solution.shifts)
             for o in range(len(shift.operations)) for b in range(len(solution.shifts))
             if a != b and shift.trailer == solution.shifts[b].trailer]
    rng.shuffle(moves)
    for a, o, b in moves[: config.candidates_per_move * 2]:
        shifts = list(solution.shifts)
        source, destination = shifts[a], shifts[b]
        source_ops, destination_ops = list(source.operations), list(destination.operations)
        operation = source_ops.pop(o)
        destination_ops.append(operation)
        shifts[a] = optimize_shift_times(instance, replace(source, operations=tuple(source_ops)))
        shifts[b] = optimize_shift_times(instance, replace(destination, operations=tuple(destination_ops)))
        result.append(Solution(tuple(shifts)))
    return result


def _within_shift_candidates(instance, solution, config, rng) -> list[Solution]:
    result = []
    moves = [(s, a, b) for s, shift in enumerate(solution.shifts)
             for a in range(len(shift.operations)) for b in range(len(shift.operations)) if a != b]
    rng.shuffle(moves)
    for s, a, b in moves[: config.candidates_per_move * 2]:
        shift = solution.shifts[s]
        operations = list(shift.operations)
        operation = operations.pop(a)
        operations.insert(b, operation)
        shifts = list(solution.shifts)
        shifts[s] = optimize_shift_times(instance, replace(shift, operations=tuple(operations)))
        result.append(Solution(tuple(shifts)))
    return result


def _select_operator(rewards, attempts, last_used, iteration, rng, feasibility_bias):
    # The EXE maintains both learned reward and usage/recency state. Do not let
    # one high-reward move starve untried neighborhoods: once its attempt lead
    # reaches six, select among the least-used operators.
    minimum_attempts = min(attempts)
    if max(attempts) - minimum_attempts >= 6:
        least_used = [i for i, count in enumerate(attempts) if count == minimum_attempts]
        if feasibility_bias:
            preferred = [i for i in least_used if i in (0, 1, 3)]
            if preferred:
                least_used = preferred
        return min(least_used, key=lambda i: (last_used[i], i))
    scores = []
    for i in range(7):
        recency = 2.0 if iteration - last_used[i] >= 33 else 1.0
        bias = 4.0 if feasibility_bias and i in (0, 1, 3) else 1.0
        scores.append(max(0.01, rewards[i]) * recency * bias / (1.0 + 0.05 * attempts[i]))
    draw = rng.random() * sum(scores)
    for i, value in enumerate(scores):
        draw -= value
        if draw <= 0:
            return i
    return 6


def _score(instance, solution, end_day):
    return score_prefix_with_feasibility_tail(
        instance, solution, score_days=end_day, feasibility_days=end_day,
        ignore_tail_call_ins=True,
    )


_WORKER_INSTANCE = None
_WORKER_CANDIDATES = None
_WORKER_END_DAY = None


def _score_candidate_index(index):
    candidate = _WORKER_CANDIDATES[index]
    return index, _score(_WORKER_INSTANCE, candidate, _WORKER_END_DAY)


def _score_candidates(instance, candidates, end_day, workers):
    if workers <= 1 or len(candidates) <= 1:
        return [(candidate, _score(instance, candidate, end_day)) for candidate in candidates]
    global _WORKER_INSTANCE, _WORKER_CANDIDATES, _WORKER_END_DAY
    _WORKER_INSTANCE = instance
    _WORKER_CANDIDATES = candidates
    _WORKER_END_DAY = end_day
    context = multiprocessing.get_context("fork")
    with context.Pool(processes=min(workers, len(candidates))) as pool:
        indexed_scores = list(pool.imap_unordered(_score_candidate_index, range(len(candidates)), chunksize=1))
    indexed_scores.sort(key=lambda item: item[0])
    return [(candidates[index], score) for index, score in indexed_scores]


def _key(score: ContestScore):
    return (
        score.hard_violations,
        score.feasibility_errors,
        score.safety_kg_min,
        score.scored_estimated_cost,
    )


def _scalar(score: ContestScore):
    return 1_000_000 * score.hard_violations + 1_000 * score.feasibility_errors + score.safety_kg_min * 1e-5


def _reindex(solution: Solution) -> Solution:
    return Solution(tuple(replace(shift, index=i) for i, shift in enumerate(solution.shifts)))
