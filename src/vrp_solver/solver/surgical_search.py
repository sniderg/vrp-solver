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
        # Match the EXE's recency-aware adaptive portfolio once a valid plan
        # exists. The original starts from a feasible construction; our
        # constructed seed does not, so first combine the recovered deletion
        # primitive with resource-preserving retiming until the portfolio has
        # a structurally valid state from which to operate.
        if iteration == 0 and config.first_operator in OPERATORS:
            operator_index = OPERATORS.index(config.first_operator)
        elif structural_errors:
            structural_codes = {
                violation.code
                for violation in validate_solution(instance, current)
                if (
                    violation.severity == "error"
                    and violation.shift is not None
                    and violation.code not in {"QS01", "QS02"}
                )
            }
            if "SHI16" in structural_codes and iteration % 4 == 0:
                operator_index = OPERATORS.index("insert_operation")
            elif (
                structural_codes & {"DRI01", "TL01"}
                and iteration % 4 == 1
            ):
                operator_index = OPERATORS.index(
                    "relocate_within_shift",
                )
            elif (
                structural_codes & {"DRI01", "TL01"}
                and iteration % 4 == 3
            ):
                operator_index = OPERATORS.index(
                    "relocate_between_shifts",
                )
            else:
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
        elif operator == "relocate_within_shift":
            retimed = [
                candidate
                for candidate in candidates
                if _same_route_points(current, candidate)
            ]
            retimed_ids = {id(candidate) for candidate in retimed}
            ordinary = [
                candidate
                for candidate in candidates
                if id(candidate) not in retimed_ids
            ]
            rng.shuffle(retimed)
            rng.shuffle(ordinary)
            candidates = retimed + ordinary
        elif operator == "relocate_between_shifts":
            split = [
                candidate
                for candidate in candidates
                if len(candidate.shifts) > len(current.shifts)
            ]
            reassigned = [
                candidate
                for candidate in candidates
                if (
                    len(candidate.shifts) == len(current.shifts)
                    and _same_route_points(current, candidate)
                )
            ]
            priority_ids = {
                id(candidate) for candidate in (*split, *reassigned)
            }
            ordinary = [
                candidate
                for candidate in candidates
                if id(candidate) not in priority_ids
            ]
            rng.shuffle(split)
            rng.shuffle(reassigned)
            rng.shuffle(ordinary)
            candidates = split + reassigned + ordinary
        elif operator == "insert_operation":
            quantity_repairs = [
                candidate
                for candidate in candidates
                if _same_route_points(current, candidate)
            ]
            quantity_ids = {
                id(candidate) for candidate in quantity_repairs
            }
            ordinary = [
                candidate
                for candidate in candidates
                if id(candidate) not in quantity_ids
            ]
            rng.shuffle(quantity_repairs)
            rng.shuffle(ordinary)
            candidates = quantity_repairs + ordinary
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
            # Solver.exe starts from a constructor-feasible shift matrix and
            # validates every affected route before committing its mutation.
            # Once that invariant exists, never trade it away for a lower
            # aggregate QS error count.
            if (
                structural_errors == 0
                and _structural_shift_errors(instance, candidate) != 0
            ):
                continue
            if move_score is None or _key(candidate_score) < _key(move_score):
                move_candidate, move_score = candidate, candidate_score
                if progress:
                    progress(
                        f"surgical_candidate,{iteration},operator,{operator},evaluated,{evaluated},"
                        f"errors,{move_score.feasibility_errors},hard,{move_score.hard_violations},"
                        f"deficit,{move_score.safety_kg_min:.3f}"
                    )

        if operator == "replace_operation_point" and scored:
            repair_pool = sorted(
                scored,
                key=lambda item: (
                    item[1].feasibility_errors
                    - item[1].hard_violations,
                    item[1].safety_kg_min,
                    _logistic_ratio(item[1]),
                ),
            )[:3]
            for raw_candidate, _ in repair_pool:
                repaired, report = repair_quantities_with_highs(
                    instance,
                    raw_candidate,
                    score_days=config.end_day,
                    feasibility_days=config.end_day,
                    ignore_tail_call_ins=True,
                    quantity_objective="max-delivered",
                )
                repaired_score = _score(
                    instance, repaired, config.end_day,
                )
                if (
                    _structural_shift_errors(instance, repaired) == 0
                    and (
                        move_score is None
                        or _key(repaired_score) < _key(move_score)
                    )
                ):
                    move_candidate = repaired
                    move_score = repaired_score
                    if progress:
                        progress(
                            f"surgical_quantity_repair,{iteration},"
                            f"status,{report.status},"
                            f"errors,{repaired_score.feasibility_errors},"
                            f"deficit,{repaired_score.safety_kg_min:.3f}"
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
        elif (
            operator in {
                "relocate_between_shifts",
                "relocate_within_shift",
            }
            and structural_errors
        ):
            # A create+inter-shift relocation endpoint can remove an illegal
            # layover while exposing one temporary resource collision.  Keep
            # that one-error bridge reachable; the EXE's transactional
            # perturbation serves the same purpose around a valid constructor
            # state.
            error_allowance = max(
                int(operator == "relocate_between_shifts"),
                perturbation // 8,
            )
            structurally_safe = [
                (candidate, candidate_score)
                for candidate, candidate_score in scored
                if (
                    candidate_score.hard_violations
                    <= current_score.hard_violations
                    and candidate_score.feasibility_errors
                    <= current_score.feasibility_errors + error_allowance
                )
            ]
            if structurally_safe:
                ranked = [
                    (
                        _structural_shift_errors(instance, candidate),
                        _key(candidate_score),
                        candidate,
                        candidate_score,
                    )
                    for candidate, candidate_score in structurally_safe
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
        improved_best = _repair_key(
            instance, current, current_score,
        ) < _repair_key(
            instance, best, best_score,
        )
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
    return _resource_safe_created_candidates(
        instance, solution, result, config,
    )


def _resource_safe_created_candidates(
    instance: Instance,
    solution: Solution,
    candidates: list[Solution],
    config: SurgicalSearchConfig,
) -> list[Solution]:
    """Place each newly created route on a compatible idle resource pair."""
    current_derived = derive_solution(instance, solution)
    result: list[Solution] = []
    cap = max(64, config.candidates_per_move * 8)
    resource_pairs = tuple(
        (driver.index, trailer_id)
        for driver in instance.drivers
        for trailer_id in driver.trailer_ids
    )
    for candidate in candidates:
        if len(candidate.shifts) != len(solution.shifts) + 1:
            continue
        created = candidate.shifts[-1]
        preferred = [(created.driver, created.trailer)]
        alternatives = [
            (driver_id, trailer_id)
            for driver_id, trailer_id in resource_pairs
            if (driver_id, trailer_id) not in preferred
        ]
        for driver_id, trailer_id in (*preferred, *alternatives):
            if not _route_allows_trailer(
                instance, created, trailer_id,
            ):
                continue
            placed = _place_created_shift_in_resource_gap(
                instance,
                solution,
                current_derived,
                replace(
                    created,
                    driver=driver_id,
                    trailer=trailer_id,
                ),
            )
            if placed is None:
                continue
            result.append(normalize_source_loads(
                instance,
                _reindex(Solution((*solution.shifts, placed))),
            ))
            break
        if len(result) >= cap:
            break
    return result


def _place_created_shift_in_resource_gap(
    instance: Instance,
    solution: Solution,
    derived,
    shift: Shift,
) -> Shift | None:
    """Uniformly translate a valid new route into an idle resource interval."""
    trial = derive_solution(
        instance, Solution((replace(shift, index=0),)),
    )[0]
    duration = trial.end - shift.start
    delta_low = -shift.start
    delta_high = instance.latest_time - trial.end
    for operation, derived_operation in zip(
        shift.operations, trial.operations,
    ):
        customer = instance.customer_by_point.get(operation.point)
        if customer is None:
            continue
        containing = [
            window
            for window in customer.time_windows
            if (
                window.start <= operation.arrival
                and derived_operation.departure <= window.end
            )
        ]
        if not containing:
            return None
        delta_low = max(
            delta_low,
            min(window.start - operation.arrival for window in containing),
        )
        delta_high = min(
            delta_high,
            max(
                window.end - derived_operation.departure
                for window in containing
            ),
        )
        if customer.call_in and customer.orders:
            orders = [
                order
                for order in customer.orders
                if (
                    order.earliest_time
                    <= operation.arrival
                    <= order.latest_time
                )
            ]
            if not orders:
                return None
            delta_low = max(
                delta_low,
                min(
                    order.earliest_time - operation.arrival
                    for order in orders
                ),
            )
            delta_high = min(
                delta_high,
                max(
                    order.latest_time - operation.arrival
                    for order in orders
                ),
            )
    if delta_low > delta_high:
        return None

    driver = instance.drivers[shift.driver]
    starts: set[int] = set()
    for window in driver.time_windows:
        low = max(shift.start + delta_low, window.start)
        high = min(
            shift.start + delta_high,
            window.end - duration,
        )
        if low > high:
            continue
        starts.add(int(low))
        for other, other_derived in zip(solution.shifts, derived):
            if other.driver == shift.driver:
                candidate_start = (
                    other_derived.end
                    + driver.min_inter_shift_duration
                )
                if low <= candidate_start <= high:
                    starts.add(candidate_start)
            if other.trailer == shift.trailer:
                candidate_start = other_derived.end
                if low <= candidate_start <= high:
                    starts.add(candidate_start)
    for start in sorted(starts):
        delta = start - shift.start
        placed = replace(
            shift,
            start=start,
            operations=tuple(
                replace(
                    operation,
                    arrival=operation.arrival + delta,
                )
                for operation in shift.operations
            ),
        )
        placed_end = start + duration
        if not _resource_overlap(
            instance,
            solution,
            derived,
            -1,
            placed,
            placed_end,
        ):
            return placed
    return None


def _insert_operation_candidates(instance, solution, config) -> list[Solution]:
    pressure = pressure_points(instance, solution, end_day=config.end_day)
    derived = derive_solution(instance, solution)
    result = _minimum_quantity_candidates(instance, solution)
    result.extend(_call_in_insert_candidates(
        instance, solution, config, derived,
    ))
    if len(result) >= config.candidates_per_move * 2:
        return _repair_mutation_resource_conflicts(
            instance,
            result[: config.candidates_per_move * 2],
            config.candidates_per_move * 2,
        )
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
                    instance,
                    replace(shift, operations=tuple(operations)),
                    latest_end=_resource_slot_end(
                        instance, solution, derived, shift_pos,
                    ),
                )
                if mutated is None:
                    continue
                shifts = list(solution.shifts)
                shifts[shift_pos] = mutated
                result.append(Solution(tuple(shifts)))
                if len(result) >= config.candidates_per_move * 2:
                    return _repair_mutation_resource_conflicts(
                        instance,
                        result,
                        config.candidates_per_move * 2,
                    )
    return _repair_mutation_resource_conflicts(
        instance,
        result,
        config.candidates_per_move * 2,
    )


def _resource_slot_end(
    instance: Instance,
    solution: Solution,
    derived,
    position: int,
) -> int:
    """Latest legal return before this route's next resource commitment."""
    shift = solution.shifts[position]
    driver = instance.drivers[shift.driver]
    latest = max(
        (
            window.end
            for window in driver.time_windows
            if window.start <= shift.start <= window.end
        ),
        default=instance.latest_time,
    )
    for other_position, other in enumerate(solution.shifts):
        if (
            other_position == position
            or other.start < derived[position].end
        ):
            continue
        if other.driver == shift.driver:
            latest = min(
                latest,
                other.start - driver.min_inter_shift_duration,
            )
        if other.trailer == shift.trailer:
            latest = min(latest, other.start)
    return latest


def _repair_mutation_resource_conflicts(
    instance: Instance,
    candidates: list[Solution],
    cap: int,
) -> list[Solution]:
    """Commit an insertion only after its affected resource chain is legal."""
    result: list[Solution] = []
    for candidate in candidates:
        violations = [
            violation
            for violation in validate_solution(instance, candidate)
            if (
                violation.severity == "error"
                and violation.code not in {"QS01", "QS02"}
            )
        ]
        if not violations:
            result.append(candidate)
        elif all(
            violation.code in {"DRI01", "TL01"}
            for violation in violations
        ):
            repaired = _propagate_resource_retimes(
                instance, candidate,
            )
            if (
                repaired is not None
                and _structural_shift_errors(instance, repaired) == 0
            ):
                result.append(repaired)
        if len(result) >= cap:
            break
    return result


def _minimum_quantity_candidates(
    instance: Instance,
    solution: Solution,
) -> list[Solution]:
    """Raise one subminimum delivery to the customer's legal floor."""
    result: list[Solution] = []
    seen: set[tuple[int, int]] = set()
    for violation in validate_solution(instance, solution):
        if (
            violation.severity != "error"
            or violation.code != "SHI16"
            or violation.shift is None
            or violation.operation is None
        ):
            continue
        position = (violation.shift, violation.operation)
        if position in seen:
            continue
        seen.add(position)
        shift = solution.shifts[violation.shift]
        operations = list(shift.operations)
        operation = operations[violation.operation]
        customer = instance.customer_by_point.get(operation.point)
        if customer is None:
            continue
        operations[violation.operation] = replace(
            operation,
            quantity=customer.min_operation_quantity,
        )
        shifts = list(solution.shifts)
        shifts[violation.shift] = replace(
            shift, operations=tuple(operations),
        )
        result.append(normalize_source_loads(
            instance, _reindex(Solution(tuple(shifts))),
        ))
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
    targets = pressure_points(
        instance, solution, end_day=config.end_day,
    )
    result = []
    positions = [(s, o) for s, shift in enumerate(solution.shifts)
                 for o, op in enumerate(shift.operations) if op.quantity > 0]
    rng.shuffle(positions)
    selected_targets = targets[: config.pressure_customers]
    total_cap = config.candidates_per_move * 2
    per_target_cap = max(
        4, total_cap // max(1, len(selected_targets)),
    )
    for pressure in selected_targets:
        target = pressure.customer
        added = 0
        target_positions = sorted(
            positions,
            key=lambda position: (
                solution.shifts[position[0]].operations[
                    position[1]
                ].arrival > pressure.first_minute,
                abs(
                    solution.shifts[position[0]].operations[
                        position[1]
                    ].arrival - pressure.first_minute
                ),
                -solution.shifts[position[0]].operations[
                    position[1]
                ].quantity,
            ),
        )
        for s, o in target_positions:
            shift = solution.shifts[s]
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
            result.append(normalize_source_loads(
                instance, _reindex(Solution(tuple(shifts))),
            ))
            added += 1
            if len(result) >= total_cap:
                return (
                    result
                    + _compound_replace_point_candidates(
                        instance,
                        solution,
                        selected_targets,
                        total_cap,
                    )
                )
            if added >= per_target_cap:
                break
    return result + _compound_replace_point_candidates(
        instance,
        solution,
        selected_targets,
        total_cap,
    )


def _compound_replace_point_candidates(
    instance: Instance,
    solution: Solution,
    pressures,
    cap: int,
) -> list[Solution]:
    """Expose two sequential point replacements as one scored endpoint."""
    result: list[Solution] = []
    selected = tuple(pressures[: min(8, len(pressures))])
    for left_index, left in enumerate(selected):
        for right in selected[left_index + 1:]:
            ranked_shifts = sorted(
                enumerate(solution.shifts),
                key=lambda item: sum(
                    min(
                        (
                            abs(operation.arrival - pressure.first_minute)
                            for operation in item[1].operations
                            if operation.quantity > 0
                        ),
                        default=instance.latest_time,
                    )
                    for pressure in (left, right)
                ),
            )
            added = 0
            for shift_position, shift in ranked_shifts:
                left_customer = instance.customer_by_point[left.customer]
                right_customer = instance.customer_by_point[right.customer]
                if (
                    shift.trailer not in left_customer.allowed_trailers
                    or shift.trailer not in right_customer.allowed_trailers
                ):
                    continue
                deliveries = [
                    position
                    for position, operation in enumerate(shift.operations)
                    if operation.quantity > 0
                ]
                if len(deliveries) < 2:
                    continue
                left_position = min(
                    deliveries,
                    key=lambda position: abs(
                        shift.operations[position].arrival
                        - left.first_minute
                    ),
                )
                right_choices = [
                    position
                    for position in deliveries
                    if position != left_position
                ]
                right_position = min(
                    right_choices,
                    key=lambda position: abs(
                        shift.operations[position].arrival
                        - right.first_minute
                    ),
                )
                operations = list(shift.operations)
                for position, customer in (
                    (left_position, left_customer),
                    (right_position, right_customer),
                ):
                    operation = operations[position]
                    operations[position] = replace(
                        operation,
                        point=customer.index,
                        quantity=min(
                            operation.quantity, customer.capacity,
                        ),
                    )
                mutated = try_optimize_shift_times(
                    instance,
                    replace(shift, operations=tuple(operations)),
                )
                if mutated is None:
                    continue
                shifts = list(solution.shifts)
                shifts[shift_position] = mutated
                result.append(normalize_source_loads(
                    instance,
                    _reindex(Solution(tuple(shifts))),
                ))
                added += 1
                if len(result) >= cap:
                    return result
                if added >= 4:
                    break
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
    result = _split_invalid_shift_candidates(
        instance, solution, config,
    )
    result.extend(_resource_reassignment_candidates(
        instance, solution, config,
    ))
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


def _split_invalid_shift_candidates(
    instance: Instance,
    solution: Solution,
    config: SurgicalSearchConfig,
) -> list[Solution]:
    """Split routes whose implicit layover cannot satisfy the driving rules.

    The recovered EXE applies create-shift followed by inter-shift relocation
    transactionally.  A constructed seed may need both edits before either
    intermediate plan is acceptable, so expose the equivalent compound
    endpoint as a structural bridge while retaining the recovered primitives.
    Deliveries and their order are preserved exactly.
    """
    invalid = {
        violation.shift
        for violation in validate_solution(instance, solution)
        if (
            violation.severity == "error"
            and violation.code in {"LAY02", "DRI03"}
            and violation.shift is not None
        )
    }
    result: list[Solution] = []
    cap = max(config.candidates_per_move * 2, len(invalid) * 8)
    for position in sorted(invalid):
        shift = solution.shifts[position]
        for cut in range(1, len(shift.operations)):
            prefix = try_optimize_shift_times(
                instance,
                replace(shift, operations=shift.operations[:cut]),
            )
            if prefix is None:
                continue
            suffix_operations = shift.operations[cut:]
            for driver in instance.drivers:
                compatible_trailers = [
                    trailer_id
                    for trailer_id in driver.trailer_ids
                    if _route_allows_trailer(
                        instance,
                        replace(
                            shift,
                            driver=driver.index,
                            trailer=trailer_id,
                            operations=suffix_operations,
                        ),
                        trailer_id,
                    )
                ]
                if not compatible_trailers:
                    continue
                # Timing depends on the driver and route, not the trailer.
                suffix = try_optimize_shift_times(
                    instance,
                    replace(
                        shift,
                        index=len(solution.shifts),
                        driver=driver.index,
                        trailer=compatible_trailers[0],
                        operations=suffix_operations,
                    ),
                )
                if suffix is None:
                    continue
                for trailer_id in compatible_trailers:
                    shifts = list(solution.shifts)
                    shifts[position] = prefix
                    shifts.append(replace(suffix, trailer=trailer_id))
                    result.append(normalize_source_loads(
                        instance, _reindex(Solution(tuple(shifts))),
                    ))
                    if len(result) >= cap:
                        return result
    return result


def _resource_reassignment_candidates(
    instance: Instance,
    solution: Solution,
    config: SurgicalSearchConfig,
) -> list[Solution]:
    """Recreate a conflicted route on an available compatible resource pair."""
    derived = derive_solution(instance, solution)
    targets = {
        violation.shift
        for violation in validate_solution(instance, solution)
        if (
            violation.severity == "error"
            and violation.code in {"DRI01", "TL01"}
            and violation.shift is not None
        )
    }
    result: list[Solution] = []
    per_target_cap = max(
        4,
        config.candidates_per_move * 2 // max(1, len(targets)),
    )
    for shift_position in sorted(targets):
        shift = solution.shifts[shift_position]
        added = 0
        for driver in instance.drivers:
            for trailer_id in driver.trailer_ids:
                if (
                    driver.index == shift.driver
                    and trailer_id == shift.trailer
                ):
                    continue
                if not _route_allows_trailer(
                    instance, shift, trailer_id,
                ):
                    continue
                reassigned = replace(
                    shift,
                    driver=driver.index,
                    trailer=trailer_id,
                )
                reassigned = try_optimize_shift_times(
                    instance, reassigned,
                )
                if reassigned is None:
                    continue
                trial_derived = derive_solution(
                    instance,
                    Solution((replace(reassigned, index=0),)),
                )[0]
                if _resource_overlap(
                    instance,
                    solution,
                    derived,
                    shift_position,
                    reassigned,
                    trial_derived.end,
                ):
                    continue
                shifts = list(solution.shifts)
                shifts[shift_position] = reassigned
                result.append(normalize_source_loads(
                    instance, _reindex(Solution(tuple(shifts))),
                ))
                added += 1
                if added >= per_target_cap:
                    break
            if added >= per_target_cap:
                break
    return result


def _route_allows_trailer(
    instance: Instance,
    shift: Shift,
    trailer_id: int,
) -> bool:
    for operation in shift.operations:
        customer = instance.customer_by_point.get(operation.point)
        if customer is not None and trailer_id not in customer.allowed_trailers:
            return False
        source = instance.source_by_point.get(operation.point)
        if source is not None and trailer_id not in source.allowed_trailers:
            return False
    return True


def _resource_overlap(
    instance: Instance,
    solution: Solution,
    derived,
    replaced_position: int,
    candidate: Shift,
    candidate_end: int,
) -> bool:
    driver_gap = instance.drivers[
        candidate.driver
    ].min_inter_shift_duration
    for position, other in enumerate(solution.shifts):
        if position == replaced_position:
            continue
        other_end = derived[position].end
        if other.driver == candidate.driver and (
            candidate.start < other_end + driver_gap
            and other.start < candidate_end + driver_gap
        ):
            return True
        if other.trailer == candidate.trailer and (
            candidate.start < other_end
            and other.start < candidate_end
        ):
            return True
    return False


def _within_shift_candidates(instance, solution, config, rng) -> list[Solution]:
    result = _resource_retime_candidates(instance, solution)
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


def _resource_retime_candidates(
    instance: Instance,
    solution: Solution,
) -> list[Solution]:
    """Shift a conflicted route to its first legal resource availability."""
    reassigned = _repair_resource_assignments(instance, solution)
    derived = derive_solution(instance, solution)
    targets = {
        violation.shift
        for violation in validate_solution(instance, solution)
        if (
            violation.severity == "error"
            and violation.code in {"DRI01", "TL01"}
            and violation.shift is not None
        )
    }
    result: list[Solution] = []
    if reassigned is not None:
        result.append(reassigned)
    for shift_position in sorted(targets):
        shift = solution.shifts[shift_position]
        required_start = shift.start
        for other_position, other in enumerate(solution.shifts):
            if (
                other_position == shift_position
                or other.start > shift.start
            ):
                continue
            if other.driver == shift.driver:
                required_start = max(
                    required_start,
                    derived[other_position].end
                    + instance.drivers[shift.driver].min_inter_shift_duration,
                )
            if other.trailer == shift.trailer:
                required_start = max(
                    required_start, derived[other_position].end,
                )
        if required_start <= shift.start:
            continue
        delta = required_start - shift.start
        shifted = replace(
            shift,
            start=required_start,
            operations=tuple(
                replace(operation, arrival=operation.arrival + delta)
                for operation in shift.operations
            ),
        )
        shifted = try_optimize_shift_times(instance, shifted)
        if shifted is None:
            continue
        shifts = list(solution.shifts)
        shifts[shift_position] = shifted
        candidate = normalize_source_loads(
            instance, _reindex(Solution(tuple(shifts))),
        )
        result.append(candidate)
        propagated = _propagate_resource_retimes(instance, candidate)
        if propagated is not None:
            result.append(propagated)
    return result


def _repair_resource_assignments(
    instance: Instance,
    solution: Solution,
) -> Solution | None:
    """Recover the resource-feasible construction state expected by the EXE.

    The seven recovered neighborhoods never change a shift's driver/trailer
    pair: Solver.exe stores shifts inside a resource-pair matrix populated by
    its constructor.  A native constructed seed can violate that invariant.
    Reassign the fixed route intervals as a binary interval-colouring model
    before handing the plan to the recovered local-search portfolio.
    """
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    derived = derive_solution(instance, solution)
    options: list[tuple[int, int, int]] = []
    by_shift: list[list[int]] = [[] for _ in solution.shifts]
    by_shift_driver: dict[tuple[int, int], list[int]] = {}
    by_shift_trailer: dict[tuple[int, int], list[int]] = {}
    source_by_point = instance.source_by_point
    customer_by_point = instance.customer_by_point

    for position, (shift, derived_shift) in enumerate(
        zip(solution.shifts, derived)
    ):
        for trailer in instance.trailers:
            allowed = all(
                (
                    operation.point not in source_by_point
                    or trailer.index
                    in source_by_point[operation.point].allowed_trailers
                )
                and (
                    operation.point not in customer_by_point
                    or trailer.index
                    in customer_by_point[operation.point].allowed_trailers
                )
                for operation in shift.operations
            )
            if not allowed:
                continue
            for driver in instance.drivers:
                if trailer.index not in driver.trailer_ids:
                    continue
                if not any(
                    window.start <= shift.start
                    and derived_shift.end <= window.end
                    for window in driver.time_windows
                ):
                    continue
                variable = len(options)
                options.append((position, driver.index, trailer.index))
                by_shift[position].append(variable)
                by_shift_driver.setdefault(
                    (position, driver.index), [],
                ).append(variable)
                by_shift_trailer.setdefault(
                    (position, trailer.index), [],
                ).append(variable)
        if not by_shift[position]:
            return None

    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_constraint(
        variables: list[int],
        low: float,
        high: float,
    ) -> None:
        row = len(lower)
        row_indices.extend([row] * len(variables))
        column_indices.extend(variables)
        values.extend([1.0] * len(variables))
        lower.append(low)
        upper.append(high)

    for variables in by_shift:
        add_constraint(variables, 1.0, 1.0)

    for left in range(len(solution.shifts)):
        left_shift = solution.shifts[left]
        for right in range(left + 1, len(solution.shifts)):
            right_shift = solution.shifts[right]
            if (
                left_shift.start,
                left_shift.index,
            ) <= (
                right_shift.start,
                right_shift.index,
            ):
                early, late = left, right
            else:
                early, late = right, left
            for driver in instance.drivers:
                if (
                    solution.shifts[late].start
                    < derived[early].end
                    + driver.min_inter_shift_duration
                ):
                    variables = (
                        by_shift_driver.get((left, driver.index), [])
                        + by_shift_driver.get((right, driver.index), [])
                    )
                    if variables:
                        add_constraint(variables, 0.0, 1.0)
            if solution.shifts[late].start < derived[early].end:
                for trailer in instance.trailers:
                    variables = (
                        by_shift_trailer.get((left, trailer.index), [])
                        + by_shift_trailer.get((right, trailer.index), [])
                    )
                    if variables:
                        add_constraint(variables, 0.0, 1.0)

    matrix = coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(lower), len(options)),
    ).tocsr()
    objective = np.array(
        [
            0.0
            if (
                driver == solution.shifts[position].driver
                and trailer == solution.shifts[position].trailer
            )
            else 1.0
            + (
                0.0
                if trailer == solution.shifts[position].trailer
                else 0.25
            )
            for position, driver, trailer in options
        ],
        dtype=float,
    )
    result = milp(
        objective,
        integrality=np.ones(len(options)),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(
            matrix, np.asarray(lower), np.asarray(upper),
        ),
        options={"time_limit": 30.0, "mip_rel_gap": 0.0},
    )
    if result.x is None:
        return _greedy_rebuild_resource_assignments(instance, solution)

    shifts = list(solution.shifts)
    for position, variables in enumerate(by_shift):
        selected = max(variables, key=lambda variable: result.x[variable])
        _, driver, trailer = options[selected]
        shifts[position] = replace(
            shifts[position], driver=driver, trailer=trailer,
        )
    candidate = normalize_source_loads(
        instance, _reindex(Solution(tuple(shifts))),
    )
    if any(
        violation.severity == "error"
        and violation.code in {"DRI01", "DRI08", "TL01", "TL03"}
        for violation in validate_solution(instance, candidate)
    ):
        return None
    return candidate


def _greedy_rebuild_resource_assignments(
    instance: Instance,
    solution: Solution,
) -> Solution | None:
    """Rebuild the constructor's resource-pair invariant with local timing MIPs."""
    scheduled: list[Shift] = []
    driver_end: dict[int, int] = {}
    trailer_end: dict[int, int] = {}
    source_by_point = instance.source_by_point
    customer_by_point = instance.customer_by_point

    for original in sorted(
        solution.shifts, key=lambda shift: (shift.start, shift.index),
    ):
        choices: list[tuple[tuple[float, ...], Shift, int]] = []
        for trailer in instance.trailers:
            if not all(
                (
                    operation.point not in source_by_point
                    or trailer.index
                    in source_by_point[operation.point].allowed_trailers
                )
                and (
                    operation.point not in customer_by_point
                    or trailer.index
                    in customer_by_point[operation.point].allowed_trailers
                )
                for operation in original.operations
            ):
                continue
            for driver in instance.drivers:
                if trailer.index not in driver.trailer_ids:
                    continue
                earliest = max(
                    original.start,
                    trailer_end.get(trailer.index, -10**9),
                    driver_end.get(driver.index, -10**9)
                    + driver.min_inter_shift_duration,
                )
                delta = earliest - original.start
                trial = replace(
                    original,
                    driver=driver.index,
                    trailer=trailer.index,
                    start=earliest,
                    operations=tuple(
                        replace(
                            operation,
                            arrival=operation.arrival + delta,
                        )
                        for operation in original.operations
                    ),
                )
                timed = try_optimize_shift_times(instance, trial)
                if timed is None:
                    continue
                timed_derived = derive_solution(
                    instance, Solution((replace(timed, index=0),)),
                )[0]
                choices.append(
                    (
                        (
                            float(timed.start - original.start),
                            float(
                                (driver.index != original.driver)
                                + (trailer.index != original.trailer)
                            ),
                            float(timed_derived.end),
                        ),
                        timed,
                        timed_derived.end,
                    )
                )
        if not choices:
            return None
        _, selected, end = min(choices, key=lambda item: item[0])
        scheduled.append(selected)
        driver_end[selected.driver] = end
        trailer_end[selected.trailer] = end

    candidate = normalize_source_loads(
        instance, _reindex(Solution(tuple(scheduled))),
    )
    if any(
        violation.severity == "error"
        and violation.code in {"DRI01", "DRI08", "TL01", "TL03"}
        for violation in validate_solution(instance, candidate)
    ):
        return None
    return candidate


def _propagate_resource_retimes(
    instance: Instance,
    solution: Solution,
) -> Solution | None:
    """Push a resource-conflict repair through its successor shifts."""
    shifts = list(solution.shifts)
    for _ in range(max(1, len(shifts) * 2)):
        derived = derive_solution(instance, Solution(tuple(shifts)))
        ordered = sorted(
            range(len(shifts)),
            key=lambda position: (
                shifts[position].start,
                shifts[position].index,
            ),
        )
        changed = False
        for rank, position in enumerate(ordered):
            shift = shifts[position]
            required_start = shift.start
            for previous_position in ordered[:rank]:
                previous = shifts[previous_position]
                if previous.driver == shift.driver:
                    required_start = max(
                        required_start,
                        derived[previous_position].end
                        + instance.drivers[
                            shift.driver
                        ].min_inter_shift_duration,
                    )
                if previous.trailer == shift.trailer:
                    required_start = max(
                        required_start,
                        derived[previous_position].end,
                    )
            if required_start <= shift.start:
                continue
            delta = required_start - shift.start
            shifted = replace(
                shift,
                start=required_start,
                operations=tuple(
                    replace(
                        operation,
                        arrival=operation.arrival + delta,
                    )
                    for operation in shift.operations
                ),
            )
            shifted = try_optimize_shift_times(instance, shifted)
            if shifted is None:
                return None
            shifts[position] = shifted
            changed = True
            break
        if changed:
            continue
        candidate = normalize_source_loads(
            instance, _reindex(Solution(tuple(shifts))),
        )
        if not any(
            violation.severity == "error"
            and violation.code in {"DRI01", "TL01"}
            for violation in validate_solution(instance, candidate)
        ):
            return candidate
        return None
    return None


def _same_route_points(left: Solution, right: Solution) -> bool:
    return (
        len(left.shifts) == len(right.shifts)
        and all(
            tuple(operation.point for operation in left_shift.operations)
            == tuple(operation.point for operation in right_shift.operations)
            for left_shift, right_shift in zip(left.shifts, right.shifts)
        )
    )


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


def _repair_key(
    instance: Instance,
    solution: Solution,
    score: ContestScore,
):
    """Continuation ordering for an infeasible constructed seed.

    The EXE normally starts from a feasible plan, so structural feasibility is
    implicit. During reconstruction, preserve a neutral structural cleanup as
    an incumbent checkpoint instead of losing it on restart.
    """
    return (
        score.hard_violations,
        score.feasibility_errors,
        _structural_shift_errors(instance, solution),
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
