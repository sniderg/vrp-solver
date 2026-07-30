"""Transactional local search reconstructed from the original Solver.exe."""
from __future__ import annotations

import random
import time
import multiprocessing
import logging
from dataclasses import dataclass, replace
from typing import Callable

from ..contest import ContestScore, score_prefix_with_feasibility_tail
from ..highs_repair import repair_quantities_with_highs
from ..highs_time_opt import try_optimize_shift_times
from ..model import Instance, Operation, Shift, Solution
from ..rules import derive_solution, validate_solution
from .pressure import pressure_points
from .targeted_rescue import (
    RescueConfig,
    generate_chain_rescue_candidates,
    generate_rescue_candidates,
    normalize_source_loads,
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
    current_score = _score(instance, current, config.end_day)
    best = current
    best_score = current_score
    rewards = [1.0] * 7
    attempts = [0] * 7
    last_used = [-10_000] * 7
    stagnation = 0
    steps: list[SurgicalStep] = []
    quantity_repaired_structures: set[tuple[object, ...]] = set()

    for iteration in range(config.iterations):
        if deadline is not None and time.monotonic() >= deadline:
            break
        structural_errors = _structural_shift_errors(instance, current)
        # Match the EXE's recency-aware adaptive portfolio. Force creation and
        # insertion early while inventory is infeasible. When the constructed
        # seed itself contains invalid shifts, interleave targeted operation
        # deletion until the create/insert moves have conflict-free resources.
        if iteration == 0 and config.first_operator in OPERATORS:
            operator_index = OPERATORS.index(config.first_operator)
        elif structural_errors and iteration % 2 == 0:
            operator_index = OPERATORS.index("delete_operation")
        else:
            operator_index = _select_operator(
                rewards, attempts, last_used, iteration, rng,
                feasibility_bias=not best_score.feasible,
            )
        operator = OPERATORS[operator_index]
        if progress:
            progress(
                f"surgical_start,{iteration},operator,{operator},"
                f"errors,{current_score.feasibility_errors},"
                f"best_errors,{best_score.feasibility_errors},"
                f"deficit,{current_score.safety_kg_min:.3f},"
                f"structural,{structural_errors}"
            )
        perturbation = min(64, 8 * stagnation)
        candidates = _candidates(instance, current, operator, config, rng)
        # The EXE perturbs enumeration after stagnation. Sampling a seeded
        # permutation gives the capped native scorer expanding coverage. Keep
        # every compound delete endpoint inside the cap, though: these are the
        # scored equivalents of a neutral multi-step delete bridge.
        if operator == "delete_operation":
            compound = [
                candidate for candidate in candidates
                if len(candidate.shifts) < len(current.shifts)
            ]
            ordinary = [
                candidate for candidate in candidates
                if len(candidate.shifts) >= len(current.shifts)
            ]
            rng.shuffle(compound)
            rng.shuffle(ordinary)
            candidates = compound + ordinary
        elif operator == "create_shift":
            call_in_points = {
                customer.index
                for customer in instance.customers
                if customer.call_in
            }
            call_in_candidates = [
                candidate
                for candidate in candidates
                if (
                    len(candidate.shifts) > len(current.shifts)
                    and any(
                        operation.point in call_in_points
                        for operation in candidate.shifts[-1].operations
                    )
                )
            ]
            call_in_ids = {id(candidate) for candidate in call_in_candidates}
            ordinary = [
                candidate
                for candidate in candidates
                if id(candidate) not in call_in_ids
            ]
            rng.shuffle(call_in_candidates)
            rng.shuffle(ordinary)
            reserve = min(
                len(call_in_candidates),
                max(1, config.candidates_per_move // 2),
            )
            ordinary_cap = config.candidates_per_move - reserve
            candidates = (
                call_in_candidates[:reserve]
                + ordinary[:ordinary_cap]
                + call_in_candidates[reserve:]
                + ordinary[ordinary_cap:]
            )
        else:
            rng.shuffle(candidates)
        if progress:
            progress(
                f"surgical_generated,{iteration},operator,{operator},"
                f"candidates,{len(candidates)}"
            )
        move_candidate = None
        move_score = None
        evaluated = 0
        limited_candidates = candidates[: config.candidates_per_move]
        scored = _score_candidates(
            instance, limited_candidates, config.end_day, config.workers,
        )
        for candidate, candidate_score in scored:
            evaluated += 1
            if move_score is None or _key(candidate_score) < _key(move_score):
                move_candidate, move_score = candidate, candidate_score
                if progress:
                    progress(
                        f"surgical_candidate,{iteration},operator,{operator},evaluated,{evaluated},"
                        f"errors,{move_score.feasibility_errors},hard,{move_score.hard_violations},"
                        f"deficit,{move_score.safety_kg_min:.3f}"
                    )

        move_structural = None
        if operator == "delete_operation" and structural_errors:
            error_allowance = perturbation // 8
            compound = [
                (candidate, candidate_score)
                for candidate, candidate_score in scored
                if (
                    len(candidate.shifts) < len(current.shifts)
                    and candidate_score.hard_violations <= current_score.hard_violations
                    and candidate_score.feasibility_errors
                    <= current_score.feasibility_errors + error_allowance
                )
            ]
            if compound:
                ranked = [
                    (
                        _structural_shift_errors(instance, candidate),
                        _key(candidate_score),
                        candidate,
                        candidate_score,
                    )
                    for candidate, candidate_score in compound
                ]
                structural, _, candidate, candidate_score = min(
                    ranked, key=lambda item: (item[0], item[1]),
                )
                if structural < structural_errors:
                    move_candidate, move_score = candidate, candidate_score
                    move_structural = structural
                    if progress:
                        progress(
                            f"surgical_structural_candidate,{iteration},"
                            f"structural,{structural},"
                            f"errors,{candidate_score.feasibility_errors},"
                            f"deficit,{candidate_score.safety_kg_min:.3f}"
                        )

        # Solver.exe's insert primitive can invoke the global quantity and
        # inventory MIP after its local timing MIP. Do the same once for each
        # structurally valid route skeleton. While reconstructing from an
        # infeasible seed, retain the raw candidate if the continuation model
        # is infeasible; once feasible, this becomes the EXE's commit gateway.
        if (
            move_candidate is not None
            and move_score is not None
            and operator in {"create_shift", "insert_operation"}
            and _structural_shift_errors(instance, move_candidate) == 0
        ):
            structure = _structure_signature(move_candidate)
            if structure not in quantity_repaired_structures:
                quantity_repaired_structures.add(structure)
                repaired, report = repair_quantities_with_highs(
                    instance,
                    move_candidate,
                    score_days=config.end_day,
                    feasibility_days=config.end_day,
                    ignore_tail_call_ins=True,
                    quantity_objective="max-delivered",
                )
                repaired_score = _score(
                    instance, repaired, config.end_day,
                )
                if _key(repaired_score) < _key(move_score):
                    move_candidate, move_score = repaired, repaired_score
                    if progress:
                        progress(
                            f"surgical_quantity_repair,{iteration},"
                            f"status,{report.status},"
                            f"errors,{repaired_score.feasibility_errors},"
                            f"deficit,{repaired_score.safety_kg_min:.3f}"
                        )

        previous_scalar = _scalar(best_score)
        previous_feasibility = _feasibility_key(best_score)
        accepted = False
        if move_candidate is not None and move_score is not None:
            candidate_structural = _structural_shift_errors(
                instance, move_candidate,
            )
            structural_gateway = (
                move_structural is not None
                and move_structural < structural_errors
                and move_score.hard_violations <= current_score.hard_violations
                and move_score.feasibility_errors
                <= current_score.feasibility_errors + perturbation // 8
            )
            ordinary_gateway = (
                candidate_structural <= structural_errors
                and _accept_move(
                    current_score, move_score, perturbation, rng,
                )
            )
            accepted = structural_gateway or ordinary_gateway
        if accepted:
            current, current_score = move_candidate, move_score
        improved_best = _key(current_score) < _key(best_score)
        if improved_best:
            best, best_score = current, current_score
            if _feasibility_key(best_score) < previous_feasibility:
                stagnation = 0
            else:
                # LR-only polishing must not suppress the EXE's escalating
                # perturbation while feasibility is still unchanged.
                stagnation += 1
            gain = max(0.0, previous_scalar - _scalar(best_score))
            rewards[operator_index] = 0.5 * rewards[operator_index] + 0.5 * min(3584.0, 1.0 + gain)
            if config.output_xml:
                save_solution(best, config.output_xml)
        else:
            stagnation += 1
            rewards[operator_index] *= 0.5
            # The recovered controller restores the stored incumbent on one
            # draw in four. Otherwise it keeps walking from the perturbed plan,
            # which permits multi-step repairs across a neutral/worse bridge.
            if accepted and move_structural is None and rng.randrange(4) == 0:
                current, current_score = best, best_score
        attempts[operator_index] += 1
        last_used[operator_index] = iteration

        step = SurgicalStep(
            iteration, operator, evaluated, accepted,
            best_score.feasibility_errors, best_score.hard_violations,
            best_score.safety_kg_min, best_score.scored_estimated_cost,
            _logistic_ratio(best_score),
        )
        steps.append(step)
        if progress:
            progress(
                f"surgical_step,{iteration},operator,{operator},evaluated,{evaluated},"
                f"accepted,{accepted},new_best,{improved_best},"
                f"current_errors,{current_score.feasibility_errors},"
                f"errors,{step.errors},hard,{step.hard},"
                f"deficit,{step.safety_kg_min:.3f},cost,{step.cost:.3f},"
                f"lr,{step.logistic_ratio:.10f}"
            )
        if best_score.feasible or stagnation >= config.no_improvement_limit:
            break
    return best, tuple(steps)


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
    result = [
        _reindex(Solution((*solution.shifts, replace(shift, index=len(solution.shifts)))))
        for shift in shifts
    ]
    # Pressure points deliberately exclude call-in customers, but the EXE's
    # create-shift operator enumerates every admissible customer point. Include
    # focused call-in columns so QS01 orders are not an unreachable state.
    result.extend(_call_in_shift_candidates(instance, solution, config))
    return result


def _insert_operation_candidates(instance, solution, config) -> list[Solution]:
    pressure = pressure_points(instance, solution, end_day=config.end_day)
    derived = derive_solution(instance, solution)
    result = _call_in_insert_candidates(
        instance, solution, config, derived,
    )
    if len(result) >= config.candidates_per_move * 2:
        return result[: config.candidates_per_move * 2]
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
                mutated = try_optimize_shift_times(
                    instance, replace(shift, operations=tuple(operations)),
                )
                if mutated is None:
                    continue
                shifts = list(solution.shifts)
                shifts[shift_pos] = mutated
                result.append(Solution(tuple(shifts)))
                if len(result) >= config.candidates_per_move * 2:
                    return result
    return result


def _call_in_insert_candidates(instance, solution, config, derived) -> list[Solution]:
    """Insert missing call-in orders into existing compatible source routes."""
    result: list[Solution] = []
    cutoff = config.end_day * 1440
    missing = _unsatisfied_call_ins(
        instance, solution, cutoff,
    )
    total_cap = config.candidates_per_move * 2
    per_order_cap = max(8, total_cap // max(1, len(missing)))
    for point, order_index, remaining in missing:
        result.extend(
            _call_in_insertions_for_order(
                instance,
                solution,
                derived,
                point,
                order_index,
                remaining,
                per_order_cap,
            )
        )
    return result[:total_cap]


def _call_in_insertions_for_order(
    instance,
    solution,
    derived,
    point,
    order_index,
    remaining,
    candidate_cap,
) -> list[Solution]:
    customer = instance.customer_by_point[point]
    order = customer.orders[order_index]
    quantity = min(
        customer.capacity,
        max(remaining, customer.min_operation_quantity),
    )
    result: list[Solution] = []
    for shift_pos, shift in enumerate(solution.shifts):
        if shift.trailer not in customer.allowed_trailers:
            continue
        if not any(
            operation.point in instance.source_by_point
            for operation in shift.operations
        ):
            continue
        for op_pos in range(1, len(shift.operations) + 1):
            available = derived[shift_pos].operations[
                op_pos - 1
            ].trailer_quantity
            if available + 1e-6 < quantity:
                continue
            operations = list(shift.operations)
            desired = order.earliest_time
            if op_pos > 0:
                desired = max(
                    desired, operations[op_pos - 1].arrival,
                )
            if op_pos < len(operations):
                desired = min(
                    max(desired, order.earliest_time),
                    operations[op_pos].arrival,
                )
            operations.insert(
                op_pos, Operation(point, desired, quantity),
            )
            mutated = try_optimize_shift_times(
                instance,
                replace(shift, operations=tuple(operations)),
            )
            if mutated is None:
                continue
            inserted = mutated.operations[op_pos]
            if not (
                order.earliest_time
                <= inserted.arrival
                <= order.latest_time
            ):
                continue
            shifts = list(solution.shifts)
            shifts[shift_pos] = mutated
            candidate = normalize_source_loads(
                instance, _reindex(Solution(tuple(shifts))),
            )
            result.append(candidate)
            if len(result) >= candidate_cap:
                return result
    return result


def _delete_operation_candidates(instance, solution, config, rng) -> list[Solution]:
    result = []
    positions = [
        (s, o)
        for s, shift in enumerate(solution.shifts)
        for o in range(len(shift.operations))
    ]
    # The binary enumerates the existing operation vectors in their current
    # plan context. Put operations from locally invalid shifts first; otherwise
    # uniform capping can spend whole rounds deleting already-valid work.
    invalid_shifts = {
        violation.shift
        for violation in validate_solution(instance, solution)
        if violation.severity == "error" and violation.shift is not None
    }
    # A full-shift removal is the exact end state of repeatedly applying the
    # recovered delete-operation primitive. Emit that compound candidate when
    # perturbation is cleaning an invalid route so the search does not require
    # every neutral intermediate deletion to survive incumbent restoration.
    for shift_position in sorted(invalid_shifts):
        shifts = list(solution.shifts)
        shifts.pop(shift_position)
        result.append(normalize_source_loads(
            instance, _reindex(Solution(tuple(shifts))),
        ))
    priority = [position for position in positions if position[0] in invalid_shifts]
    remainder = [position for position in positions if position[0] not in invalid_shifts]
    rng.shuffle(priority)
    rng.shuffle(remainder)
    positions = priority + remainder
    for s, o in positions[: config.candidates_per_move * 2]:
        shift = solution.shifts[s]
        operations = list(shift.operations)
        operations.pop(o)
        shifts = list(solution.shifts)
        if operations:
            mutated = try_optimize_shift_times(
                instance, replace(shift, operations=tuple(operations)),
            )
            if mutated is None:
                continue
            shifts[s] = mutated
        else:
            shifts.pop(s)
        result.append(_reindex(Solution(tuple(shifts))))
    return result


def _call_in_shift_candidates(instance, solution, config) -> list[Solution]:
    """Construct single-order shifts for currently unsatisfied call-ins."""
    result: list[Solution] = []
    seen: set[tuple[object, ...]] = set()
    cutoff = config.end_day * 1440
    for point, order_index, remaining in _unsatisfied_call_ins(instance, solution, cutoff):
        customer = instance.customer_by_point[point]
        order = customer.orders[order_index]
        quantity = min(
            customer.capacity,
            max(remaining, customer.min_operation_quantity),
        )
        for driver in instance.drivers:
            for trailer_id in driver.trailer_ids:
                if trailer_id not in customer.allowed_trailers:
                    continue
                trailer = instance.trailers[trailer_id]
                if quantity > trailer.capacity + 1e-6:
                    continue
                for source in instance.sources:
                    if trailer_id not in source.allowed_trailers:
                        continue
                    lead = (
                        instance.time_matrix[instance.base_index][source.index]
                        + source.setup_time
                        + instance.time_matrix[source.index][point]
                    )
                    return_time = instance.time_matrix[point][instance.base_index]
                    for customer_window in customer.time_windows:
                        earliest = max(order.earliest_time, customer_window.start)
                        latest = min(
                            order.latest_time - customer.setup_time,
                            customer_window.end - customer.setup_time,
                            cutoff - 1,
                        )
                        if earliest > latest:
                            continue
                        arrivals = _even_samples(
                            earliest, latest, config.samples_per_customer,
                        )
                        for arrival in arrivals:
                            start = arrival - lead
                            end = arrival + customer.setup_time + return_time
                            if not any(
                                window.start <= start and end <= window.end
                                for window in driver.time_windows
                            ):
                                continue
                            source_arrival = (
                                start
                                + instance.time_matrix[instance.base_index][source.index]
                            )
                            shift = Shift(
                                index=len(solution.shifts),
                                driver=driver.index,
                                trailer=trailer_id,
                                start=start,
                                operations=(
                                    Operation(source.index, source_arrival, -quantity),
                                    Operation(point, arrival, quantity),
                                ),
                            )
                            signature = (
                                shift.driver, shift.trailer, shift.start,
                                tuple((op.point, op.arrival) for op in shift.operations),
                            )
                            if signature in seen:
                                continue
                            seen.add(signature)
                            candidate = _reindex(
                                Solution((*solution.shifts, shift))
                            )
                            result.append(
                                normalize_source_loads(instance, candidate)
                            )
    return result


def _unsatisfied_call_ins(instance, solution, cutoff):
    delivered: dict[tuple[int, int], float] = {}
    for shift in solution.shifts:
        for operation in shift.operations:
            customer = instance.customer_by_point.get(operation.point)
            if customer is None or not customer.call_in:
                continue
            for order_index, order in enumerate(customer.orders):
                if order.earliest_time <= operation.arrival <= order.latest_time:
                    key = (customer.index, order_index)
                    delivered[key] = delivered.get(key, 0.0) + operation.quantity
    missing = []
    for customer in instance.customers:
        if not customer.call_in:
            continue
        for order_index, order in enumerate(customer.orders):
            if order.latest_time > cutoff:
                continue
            remaining = (
                order.min_quantity_to_satisfy
                - delivered.get((customer.index, order_index), 0.0)
            )
            if remaining > 1e-6:
                missing.append((customer.index, order_index, remaining))
    return missing


def _even_samples(start: int, end: int, count: int) -> tuple[int, ...]:
    if count <= 1 or start == end:
        return (end,)
    return tuple(sorted({
        start + round((end - start) * index / (count - 1))
        for index in range(count)
    }))


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
            mutated = try_optimize_shift_times(
                instance, replace(shift, operations=tuple(operations)),
            )
            if mutated is None:
                continue
            shifts[s] = mutated
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
        mutated = try_optimize_shift_times(
            instance, replace(shift, operations=tuple(operations)),
        )
        if mutated is None:
            continue
        shifts[s] = mutated
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
        mutated_source = try_optimize_shift_times(
            instance, replace(source, operations=tuple(source_ops)),
        )
        mutated_destination = try_optimize_shift_times(
            instance, replace(destination, operations=tuple(destination_ops)),
        )
        if mutated_source is None or mutated_destination is None:
            continue
        shifts[a] = mutated_source
        shifts[b] = mutated_destination
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
        mutated = try_optimize_shift_times(
            instance, replace(shift, operations=tuple(operations)),
        )
        if mutated is None:
            continue
        shifts[s] = mutated
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


def _accept_move(
    current: ContestScore,
    candidate: ContestScore,
    perturbation: int,
    rng: random.Random,
) -> bool:
    if _key(candidate) < _key(current):
        return True
    if perturbation <= 0:
        return False
    if candidate.hard_violations > current.hard_violations:
        return False
    error_allowance = max(1, perturbation // 8)
    if candidate.feasibility_errors > current.feasibility_errors + error_allowance:
        return False
    deficit_allowance = perturbation * 50_000.0
    if candidate.safety_kg_min > current.safety_kg_min + deficit_allowance:
        return False
    # Keep a small stochastic refusal separate from the controller's verified
    # one-in-four incumbent restore draw.
    return rng.random() >= 0.05


def _key(score: ContestScore):
    return (
        score.hard_violations,
        score.feasibility_errors,
        score.safety_kg_min,
        _logistic_ratio(score),
    )


def _feasibility_key(score: ContestScore):
    return (
        score.hard_violations,
        score.feasibility_errors,
        score.safety_kg_min,
    )


def _scalar(score: ContestScore):
    return (
        1_000_000 * score.hard_violations
        + 1_000 * score.feasibility_errors
        + score.safety_kg_min * 1e-5
        + _logistic_ratio(score)
    )


def _logistic_ratio(score: ContestScore) -> float:
    return score.scored_estimated_cost / max(
        1.0, score.scored_delivered_quantity,
    )


def _reindex(solution: Solution) -> Solution:
    return Solution(tuple(replace(shift, index=i) for i, shift in enumerate(solution.shifts)))


def _structure_signature(solution: Solution) -> tuple[object, ...]:
    return tuple(
        (
            shift.driver,
            shift.trailer,
            shift.start,
            tuple(
                (operation.point, operation.arrival)
                for operation in shift.operations
            ),
        )
        for shift in solution.shifts
    )


def _structural_shift_errors(instance: Instance, solution: Solution) -> int:
    return sum(
        1
        for violation in validate_solution(instance, solution)
        if (
            violation.severity == "error"
            and violation.shift is not None
            and violation.code not in {"QS01", "QS02"}
        )
    )
