"""Transactional local search reconstructed from Solver.exe plus native moves."""
from __future__ import annotations

import random
import time
import multiprocessing
import logging
from itertools import combinations
from dataclasses import dataclass, replace
from typing import Callable

from ..contest import ContestScore, score_prefix_with_feasibility_tail
from ..diagnostics import (
    ViolationVector,
    solution_fingerprint,
    violation_vector,
)
from ..highs_repair import repair_quantities_with_highs
from ..highs_time_opt import try_optimize_shift_times
from ..inventory import tank_events
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
    candidates_per_move: int = 8
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
    "pressure_band_resource_block",
    "recombine_route_blocks",
    "joint_retailer_reinsert",
)

logging.getLogger("vrp_solver.highs_time_opt").setLevel(logging.ERROR)


def surgical_search(
    instance: Instance,
    initial: Solution,
    *,
    config: SurgicalSearchConfig,
    progress: Callable[[str], None] | None = print,
) -> tuple[Solution, tuple[SurgicalStep, ...]]:
    """Apply legacy and native topology moves with transactional rollback."""
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
    rewards = [1024.0] * len(OPERATORS)
    attempts = [0] * len(OPERATORS)
    last_used = [-10_000] * len(OPERATORS)
    stagnation = 0
    steps: list[SurgicalStep] = []
    quantity_repaired_structures: set[tuple[object, ...]] = set()

    for iteration in range(config.iterations):
        if deadline is not None and time.monotonic() >= deadline:
            break
        current_vector = violation_vector(instance, current)
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
        elif operator == "replace_operation_point":
            rebuilt = [
                candidate
                for candidate in candidates
                if (
                    _operation_count_difference(current, candidate)
                    or _is_direct_route_rebuild(current, candidate)
                )
            ]
            rebuilt_ids = {id(candidate) for candidate in rebuilt}
            compound = [
                candidate
                for candidate in candidates
                if (
                    id(candidate) not in rebuilt_ids
                    and _point_difference_count(current, candidate) >= 2
                )
            ]
            priority_ids = {
                id(candidate) for candidate in (*rebuilt, *compound)
            }
            ordinary = [
                candidate
                for candidate in candidates
                if id(candidate) not in priority_ids
            ]
            rng.shuffle(rebuilt)
            rng.shuffle(compound)
            rng.shuffle(ordinary)
            candidates = rebuilt + compound + ordinary
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
        move_vector = None
        evaluated = 0
        limited_candidates = candidates[: config.candidates_per_move]
        scored = _score_candidates(
            instance, limited_candidates, config.end_day, config.workers,
        )
        for candidate, candidate_score in scored:
            evaluated += 1
            candidate_vector = violation_vector(instance, candidate)
            if not _hard_invariants_not_worse(
                current_vector, candidate_vector,
            ):
                continue
            # Solver.exe starts from a constructor-feasible shift matrix and
            # validates every affected route before committing its mutation.
            # Once that invariant exists, never trade it away for a lower
            # aggregate QS error count.
            if (
                structural_errors == 0
                and _structural_shift_errors(instance, candidate) != 0
            ):
                continue
            if (
                move_score is None
                or move_vector is None
                or (candidate_vector.key(), _key(candidate_score))
                < (move_vector.key(), _key(move_score))
            ):
                move_candidate, move_score = candidate, candidate_score
                move_vector = candidate_vector
                if progress:
                    progress(
                        f"surgical_candidate,{iteration},operator,{operator},evaluated,{evaluated},"
                        f"errors,{move_score.feasibility_errors},hard,{move_score.hard_violations},"
                        f"deficit,{move_score.safety_kg_min:.3f}"
                    )

        # These operators change the trailer load path.  Their generated
        # topologies still carry seed/incumbent quantities, so raw replay may
        # show a temporary trailer or tank violation even when the topology
        # admits a hard-feasible quantity assignment.  Repair the topology
        # before applying the hard-invariant acceptance gateway.
        if operator in {
            "create_shift",
            "pressure_band_resource_block",
            "replace_operation_point",
            "recombine_route_blocks",
        } and scored:
            ranked_repair_pool = sorted(
                scored,
                key=lambda item: (
                    item[1].feasibility_errors
                    - item[1].hard_violations,
                    item[1].safety_kg_min,
                    _logistic_ratio(item[1]),
                ),
            )
            repair_pool = list(ranked_repair_pool[
                :12 if operator in {
                    "create_shift", "pressure_band_resource_block",
                    "recombine_route_blocks",
                } else 5
            ])
            repaired_targets: set[int] = set()
            pressure_targets = {
                pressure.customer
                for pressure in pressure_points(
                    instance, current, end_day=config.end_day,
                )
            }
            if operator == "replace_operation_point":
                for item in ranked_repair_pool:
                    target = _rebuilt_route_target(
                        current, item[0], pressure_targets,
                    )
                    if (
                        target is None
                        or target in repaired_targets
                    ):
                        continue
                    repaired_targets.add(target)
                    if item not in repair_pool:
                        repair_pool.append(item)
            for raw_candidate, _ in repair_pool:
                repaired, report = repair_quantities_with_highs(
                    instance,
                    raw_candidate,
                    score_days=config.end_day,
                    feasibility_days=config.end_day,
                    ignore_tail_call_ins=True,
                    quantity_objective="max-delivered",
                )
                if (
                    report.status != "Optimal"
                    and operator in {
                        "create_shift", "pressure_band_resource_block",
                    }
                ):
                    repaired, report = repair_quantities_with_highs(
                        instance,
                        raw_candidate,
                        score_days=config.end_day,
                        feasibility_days=config.end_day,
                        ignore_tail_call_ins=True,
                        quantity_objective="min-delivered",
                    )
                    if report.status == "Optimal":
                        compacted = _compact_after_activation(
                            instance, repaired,
                        )
                        if compacted is None:
                            continue
                        repaired = compacted
                if report.status != "Optimal":
                    continue
                repaired_score = _score(
                    instance, repaired, config.end_day,
                )
                repaired_vector = violation_vector(instance, repaired)
                if (
                    _structural_shift_errors(instance, repaired) == 0
                    and _hard_invariants_not_worse(
                        current_vector, repaired_vector,
                    )
                    and (
                        move_score is None
                        or move_vector is None
                        or (repaired_vector.key(), _key(repaired_score))
                        < (move_vector.key(), _key(move_score))
                    )
                ):
                    move_candidate = repaired
                    move_score = repaired_score
                    move_vector = repaired_vector
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
                    move_vector = violation_vector(instance, candidate)
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
                    move_vector = violation_vector(instance, candidate)
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
                repaired_vector = violation_vector(instance, repaired)
                if (
                    _hard_invariants_not_worse(
                        current_vector, repaired_vector,
                    )
                    and (
                        move_vector is None
                        or (repaired_vector.key(), _key(repaired_score))
                        < (move_vector.key(), _key(move_score))
                    )
                ):
                    move_candidate, move_score = repaired, repaired_score
                    move_vector = repaired_vector
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
            if move_vector is None:
                move_vector = violation_vector(instance, move_candidate)
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
                and _hard_invariants_not_worse(
                    current_vector, move_vector,
                )
                and (
                    move_vector.key() < current_vector.key()
                    or (
                        move_vector.key() == current_vector.key()
                        and _accept_move(
                            current_score, move_score, perturbation, rng,
                        )
                    )
                )
            )
            accepted = structural_gateway or ordinary_gateway
        if accepted:
            current, current_score = move_candidate, move_score
        current_best_vector = violation_vector(instance, current)
        incumbent_best_vector = violation_vector(instance, best)
        improved_best = (
            current_best_vector.key(),
            _repair_key(instance, current, current_score),
        ) < (
            incumbent_best_vector.key(),
            _repair_key(instance, best, best_score),
        )
        if improved_best:
            best, best_score = current, current_score
            if _feasibility_key(best_score) < previous_feasibility:
                stagnation = 0
            else:
                # LR-only polishing must not suppress the EXE's escalating
                # perturbation while feasibility is still unchanged.
                stagnation += 1
            # Verified EWMA constants from Ghidra (0x140023580):
            # strong reward (3584) when new best solution is found
            rewards[operator_index] = 0.5 * rewards[operator_index] + 0.5 * 3584.0
            if config.output_xml:
                save_solution(best, config.output_xml)
        else:
            stagnation += 1
            # large improvement reward (1024) for accepted move, failure reward (512) for rejected move
            target_reward = 1024.0 if accepted else 512.0
            rewards[operator_index] = 0.5 * rewards[operator_index] + 0.5 * target_reward
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
    if operator == "recombine_route_blocks":
        return _recombine_route_block_candidates(instance, solution, config, rng)
    if operator == "pressure_band_resource_block":
        return _pressure_band_resource_block_candidates(
            instance, solution, config, rng,
        )
    if operator == "joint_retailer_reinsert":
        return _joint_retailer_reinsert_candidates(instance, solution, config)
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
    ordinary = [
        _reindex(Solution((*solution.shifts, replace(shift, index=len(solution.shifts)))))
        for shift in shifts
    ]
    # Pressure points deliberately exclude call-in customers, but the EXE's
    # create-shift operator enumerates every admissible customer point. Include
    # focused call-in columns so QS01 orders are not an unreachable state.
    # Put mandatory-order columns through the bounded resource-placement gate
    # first.  Appending them after a large VMI pool made the cap exhaust before
    # a single call-in candidate was considered, despite the scorer's later
    # attempt to reserve call-in evaluation slots.
    result = _call_in_shift_candidates(instance, solution, config)
    result.extend(ordinary)
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
    # A safety delivery may fit immediately before a route's first stop when
    # that route already has a planned layover later in the day.  The generic
    # timing MIP deliberately models a no-layover chain, so it cannot discover
    # this otherwise valid mutation.  Preserve the route's existing timestamps
    # and validate the complete resource/inventory state transactionally.
    result.extend(_vmi_safety_prepend_candidates(
        instance, solution, pressure, derived, config,
    ))
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


def _vmi_safety_prepend_candidates(
    instance: Instance,
    solution: Solution,
    pressure,
    derived,
    config: SurgicalSearchConfig,
) -> list[Solution]:
    """Move a later VMI delivery into a route's preserved early layover.

    This is intentionally a paired move: inserting stock before the first
    stop without reducing a later delivery can overfill the tank.  The
    source-load normalizer then reconciles the two affected trailers from
    their real route histories.
    """
    result: list[Solution] = []
    cap = config.candidates_per_move * 2
    for point in pressure[: config.pressure_customers]:
        customer = instance.customer_by_point[point.customer]
        if customer.call_in:
            continue
        later = [
            (shift_pos, operation_pos, operation)
            for shift_pos, shift in enumerate(solution.shifts)
            for operation_pos, operation in enumerate(shift.operations)
            if (
                operation.point == customer.index
                and operation.quantity >= customer.min_operation_quantity
                and operation.arrival >= point.first_minute
            )
        ]
        if not later:
            continue
        for shift_pos, shift in enumerate(solution.shifts):
            if (
                not shift.operations
                or shift.start >= point.first_minute
                or shift.trailer not in customer.allowed_trailers
            ):
                continue
            first = shift.operations[0]
            for window in customer.time_windows:
                arrival = max(
                    window.start,
                    shift.start + instance.time_matrix[
                        instance.base_index
                    ][customer.index],
                )
                if arrival + customer.setup_time > window.end:
                    continue
                # Keep the original downstream timetable untouched.  It may
                # intentionally contain the one legal layover that the MIP
                # retimer does not model.
                if arrival > first.arrival:
                    continue
                for later_shift_pos, later_operation_pos, later_operation in later:
                    if later_shift_pos == shift_pos:
                        continue
                    quantity = min(
                        1_000.0,
                        customer.capacity * 0.25,
                        later_operation.quantity - customer.min_operation_quantity,
                    )
                    if quantity + 1e-6 < customer.min_operation_quantity:
                        continue
                    shifts = list(solution.shifts)
                    shifts[shift_pos] = replace(
                        shift,
                        operations=(
                            Operation(customer.index, arrival, quantity),
                            *shift.operations,
                        ),
                    )
                    target_shift = shifts[later_shift_pos]
                    target_operations = list(target_shift.operations)
                    target_operations[later_operation_pos] = replace(
                        later_operation,
                        quantity=later_operation.quantity - quantity,
                    )
                    shifts[later_shift_pos] = replace(
                        target_shift, operations=tuple(target_operations),
                    )
                    candidate = normalize_source_loads(
                        instance,
                        _reindex(Solution(tuple(shifts))),
                    )
                    if _structural_shift_errors(instance, candidate) == 0:
                        result.append(candidate)
                    if len(result) >= cap:
                        return result
    return result


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
            # The insertion and its source load are one atomic move.  Testing
            # only the incumbent's residual trailer quantity makes every
            # exactly-loaded route unreachable, although normalize_source_loads
            # below can legitimately increase the preceding source pickup.
            # Gate on the complete reload segment's capacity instead.
            previous_source = max(
                (
                    index for index in range(op_pos)
                    if shift.operations[index].point in instance.source_by_point
                ),
                default=None,
            )
            if previous_source is None:
                continue
            next_source = next(
                (
                    index for index in range(op_pos, len(shift.operations))
                    if shift.operations[index].point in instance.source_by_point
                ),
                len(shift.operations),
            )
            segment_deliveries = sum(
                max(0.0, operation.quantity)
                for operation in shift.operations[previous_source + 1:next_source]
            )
            needs_quantity_repair = (
                segment_deliveries + quantity
                > instance.trailers[shift.trailer].capacity + 1e-6
            )
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
            if needs_quantity_repair:
                # Free capacity from VMI deliveries in the same/global trailer
                # plan while keeping the inserted call-in amount fixed.  The
                # strict model also proves that no customer inventory is harmed.
                from ..highs_repair import repair_quantities_with_highs

                horizon_days = max(
                    1,
                    (instance.horizon * instance.unit + 1_439) // 1_440,
                )
                candidate, report = repair_quantities_with_highs(
                    instance,
                    candidate,
                    score_days=horizon_days,
                    feasibility_days=horizon_days,
                    quantity_objective="max-delivered",
                    strict_inventory=True,
                )
                if report.status != "Optimal":
                    continue
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
    """Construct single-order and compatible paired call-in shifts."""
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
    missing = _unsatisfied_call_ins(instance, solution, cutoff)
    for left_index in range(len(missing)):
        for right_index in range(left_index + 1, len(missing)):
            pair = (missing[left_index], missing[right_index])
            for ordered in (pair, pair[::-1]):
                first = instance.customer_by_point[ordered[0][0]]
                second = instance.customer_by_point[ordered[1][0]]
                first_order = first.orders[ordered[0][1]]
                second_order = second.orders[ordered[1][1]]
                first_quantity = max(
                    ordered[0][2], first.min_operation_quantity,
                )
                second_quantity = max(
                    ordered[1][2], second.min_operation_quantity,
                )
                for driver in instance.drivers:
                    for trailer_id in driver.trailer_ids:
                        if not (
                            trailer_id in first.allowed_trailers
                            and trailer_id in second.allowed_trailers
                        ):
                            continue
                        trailer_capacity = instance.trailers[trailer_id].capacity
                        if max(first_quantity, second_quantity) > trailer_capacity + 1e-6:
                            continue
                        for source in instance.sources:
                            if trailer_id not in source.allowed_trailers:
                                continue
                            for first_window in first.time_windows:
                                first_arrival = max(
                                    first_order.earliest_time,
                                    first_window.start,
                                )
                                if first_arrival + first.setup_time > min(
                                    first_order.latest_time,
                                    first_window.end,
                                ):
                                    continue
                                multi_reload = (
                                    first_quantity + second_quantity
                                    > trailer_capacity + 1e-6
                                )
                                reload_arrival = (
                                    first_arrival + first.setup_time
                                    + instance.time_matrix[first.index][source.index]
                                )
                                second_earliest = (
                                    reload_arrival + source.setup_time
                                    + instance.time_matrix[source.index][second.index]
                                    if multi_reload else
                                    first_arrival + first.setup_time
                                    + instance.time_matrix[first.index][second.index]
                                )
                                second_arrival = next((
                                    max(
                                        second_earliest,
                                        second_order.earliest_time,
                                        window.start,
                                    )
                                    for window in second.time_windows
                                    if max(
                                        second_earliest,
                                        second_order.earliest_time,
                                        window.start,
                                    ) + second.setup_time <= min(
                                        second_order.latest_time,
                                        window.end,
                                    )
                                ), None)
                                if second_arrival is None:
                                    continue
                                source_arrival = (
                                    first_arrival - source.setup_time
                                    - instance.time_matrix[source.index][first.index]
                                )
                                start = (
                                    source_arrival
                                    - instance.time_matrix[instance.base_index][source.index]
                                )
                                if start < 0:
                                    continue
                                route_operations = [
                                    Operation(
                                        source.index, source_arrival,
                                        -(
                                            first_quantity
                                            if multi_reload
                                            else first_quantity + second_quantity
                                        ),
                                    ),
                                    Operation(first.index, first_arrival, first_quantity),
                                ]
                                if multi_reload:
                                    route_operations.append(Operation(
                                        source.index, reload_arrival, -second_quantity,
                                    ))
                                route_operations.append(Operation(
                                    second.index, second_arrival, second_quantity,
                                ))
                                shift = Shift(
                                    index=len(solution.shifts),
                                    driver=driver.index,
                                    trailer=trailer_id,
                                    start=start,
                                    operations=tuple(route_operations),
                                )
                                signature = (
                                    shift.driver, shift.trailer, shift.start,
                                    tuple((op.point, op.arrival) for op in shift.operations),
                                )
                                if signature in seen:
                                    continue
                                seen.add(signature)
                                result.append(normalize_source_loads(
                                    instance,
                                    _reindex(Solution((*solution.shifts, shift))),
                                ))
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
                 for o, op in enumerate(shift.operations)
                 if (
                     op.quantity > 0
                     and (
                         op.point not in instance.customer_by_point
                         or not instance.customer_by_point[op.point].call_in
                     )
                 )]
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
                    + _extended_replace_candidates(
                        instance,
                        solution,
                        selected_targets,
                        total_cap,
                    )
                )
            if added >= per_target_cap:
                break
    return (
        result
        + _extended_replace_candidates(
            instance,
            solution,
            selected_targets,
            total_cap,
        )
    )


def _extended_replace_candidates(
    instance: Instance,
    solution: Solution,
    pressures,
    cap: int,
) -> list[Solution]:
    point_compounds = _compound_replace_point_candidates(
        instance, solution, pressures, cap,
    )
    rebuilt = _route_rebuild_candidates(
        instance, solution, pressures, cap,
    )
    return (
        point_compounds
        + rebuilt
        + _compound_rebuilt_route_candidates(
            instance, solution, rebuilt, pressures, cap,
        )
    )


def _compound_rebuilt_route_candidates(
    instance: Instance,
    solution: Solution,
    rebuilt: list[Solution],
    pressures,
    cap: int,
) -> list[Solution]:
    """Combine two independent route-slot rebuilds before quantity repair."""
    pressure_targets = {
        pressure.customer for pressure in pressures
    }
    grouped: dict[int, list[tuple[int, Shift]]] = {}
    for candidate in rebuilt:
        target = _rebuilt_route_target(
            solution, candidate, pressure_targets,
        )
        if target is None:
            continue
        for position, (old_shift, new_shift) in enumerate(
            zip(solution.shifts, candidate.shifts)
        ):
            if (
                old_shift.operations != new_shift.operations
                and len(new_shift.operations) == 2
                and new_shift.operations[0].quantity < 0
                and new_shift.operations[1].point == target
                and new_shift.operations[1].quantity > 0
            ):
                grouped.setdefault(target, []).append(
                    (position, new_shift),
                )
                break
    result: list[Solution] = []
    for left_target, right_target in combinations(
        sorted(grouped), 2,
    ):
        for left_position, left_shift in grouped[
            left_target
        ][:3]:
            for right_position, right_shift in grouped[
                right_target
            ][:3]:
                if left_position == right_position:
                    continue
                shifts = list(solution.shifts)
                shifts[left_position] = left_shift
                shifts[right_position] = right_shift
                result.append(normalize_source_loads(
                    instance,
                    _reindex(Solution(tuple(shifts))),
                ))
                if len(result) >= cap:
                    return result
    return result


def _route_rebuild_candidates(
    instance: Instance,
    solution: Solution,
    pressures,
    cap: int,
) -> list[Solution]:
    """Replace one saturated route slot with a direct pressure delivery."""
    derived = derive_solution(instance, solution)
    result: list[Solution] = []
    per_target_cap = max(4, cap // max(1, len(pressures)))
    for pressure in pressures:
        customer = instance.customer_by_point[pressure.customer]
        added = 0
        eligible_hosts = [
            item
            for item in enumerate(solution.shifts)
            if item[1].start < pressure.first_minute
        ]
        hosts: list[tuple[int, Shift]] = []
        used_positions: set[int] = set()
        for anchor in _even_samples(
            0,
            pressure.first_minute,
            per_target_cap,
        ):
            choices = [
                item
                for item in eligible_hosts
                if item[0] not in used_positions
            ]
            if not choices:
                break
            selected = min(
                choices,
                key=lambda item: abs(item[1].start - anchor),
            )
            hosts.append(selected)
            used_positions.add(selected[0])
        for position, host in hosts:
            if (
                host.start >= pressure.first_minute
                or host.trailer not in customer.allowed_trailers
            ):
                continue
            trailer = instance.trailers[host.trailer]
            quantity = min(customer.capacity, trailer.capacity)
            if quantity + 1e-6 < customer.min_operation_quantity:
                continue
            latest_end = _resource_slot_end(
                instance, solution, derived, position,
            )
            for source in instance.sources:
                if host.trailer not in source.allowed_trailers:
                    continue
                source_arrival = (
                    host.start
                    + instance.time_matrix[
                        instance.base_index
                    ][source.index]
                )
                earliest_customer = (
                    source_arrival
                    + source.setup_time
                    + instance.time_matrix[
                        source.index
                    ][customer.index]
                )
                desired = min(
                    pressure.first_minute,
                    max(
                        earliest_customer,
                        min(
                            (
                                window.start
                                for window in customer.time_windows
                                if window.end >= earliest_customer
                            ),
                            default=earliest_customer,
                        ),
                    ),
                )
                rebuilt = replace(
                    host,
                    operations=(
                        Operation(
                            source.index,
                            source_arrival,
                            -quantity,
                        ),
                        Operation(
                            customer.index,
                            desired,
                            quantity,
                        ),
                    ),
                )
                rebuilt = try_optimize_shift_times(
                    instance,
                    rebuilt,
                    latest_end=latest_end,
                )
                if rebuilt is None:
                    continue
                rebuilt_derived = derive_solution(
                    instance,
                    Solution((replace(rebuilt, index=0),)),
                )[0]
                if _resource_overlap(
                    instance,
                    solution,
                    derived,
                    position,
                    rebuilt,
                    rebuilt_derived.end,
                ):
                    continue
                shifts = list(solution.shifts)
                shifts[position] = rebuilt
                result.append(normalize_source_loads(
                    instance,
                    _reindex(Solution(tuple(shifts))),
                ))
                added += 1
                break
            if len(result) >= cap:
                return result
            if added >= per_target_cap:
                break
    return result


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
    for group in combinations(selected[:6], 3):
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
                for pressure in group
            ),
        )
        added = 0
        for shift_position, shift in ranked_shifts:
            customers = tuple(
                instance.customer_by_point[pressure.customer]
                for pressure in group
            )
            if any(
                shift.trailer not in customer.allowed_trailers
                for customer in customers
            ):
                continue
            available = [
                position
                for position, operation in enumerate(shift.operations)
                if operation.quantity > 0
            ]
            if len(available) < 3:
                continue
            replacements: list[tuple[int, object]] = []
            for pressure, customer in zip(group, customers):
                position = min(
                    available,
                    key=lambda item: abs(
                        shift.operations[item].arrival
                        - pressure.first_minute
                    ),
                )
                available.remove(position)
                replacements.append((position, customer))
            operations = list(shift.operations)
            for position, customer in replacements:
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
            if added >= 2:
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


def _pressure_band_resource_block_candidates(
    instance: Instance,
    solution: Solution,
    config: SurgicalSearchConfig,
    rng: random.Random,
) -> list[Solution]:
    """Rebuild routes around high-area inventory pressure intervals.

    Selection uses only pressure, time proximity, compatibility, and resource
    boundaries.  The quantity/activation transaction is deliberately handled
    by the caller after these route skeletons are generated.
    """
    pressures = pressure_points(
        instance, solution, end_day=config.end_day,
    )[: config.pressure_customers]
    if not pressures:
        return []
    cap = max(8, config.candidates_per_move * 2)
    radius = max(1_440, min(4_320, config.end_day * 288))
    derived = derive_solution(instance, solution)
    results: list[Solution] = []
    seen: set[str] = set()

    def append_candidate(shifts: list[Shift]) -> None:
        candidate = normalize_source_loads(
            instance, _reindex(Solution(tuple(shifts))),
        )
        fingerprint = solution_fingerprint(candidate)
        if fingerprint not in seen:
            seen.add(fingerprint)
            results.append(candidate)

    for pressure in pressures:
        target_customer = instance.customer_by_point[pressure.customer]
        target_positions = [
            (shift_position, operation_position)
            for shift_position, shift in enumerate(solution.shifts)
            for operation_position, operation in enumerate(shift.operations)
            if operation.point == pressure.customer and operation.quantity > 0
        ]
        target_positions.sort(key=lambda item: abs(
            solution.shifts[item[0]].operations[item[1]].arrival
            - pressure.first_minute
        ))
        if not target_positions:
            # Binary activation may remove every visit to a still-pressured
            # VMI customer. Price it back into an earlier optional slot using
            # only breach/window/compatibility features.
            replacement_slots = [
                (shift_position, operation_position)
                for shift_position, shift in enumerate(solution.shifts)
                if shift.trailer in target_customer.allowed_trailers
                for operation_position, operation in enumerate(shift.operations)
                if (
                    operation.point in instance.customer_by_point
                    and operation.quantity > 0
                    and operation.arrival <= pressure.first_minute
                    and pressure.first_minute - operation.arrival <= radius
                    and not instance.customer_by_point[operation.point].call_in
                    and not instance.customer_by_point[operation.point].layover_customer
                )
            ]
            replacement_slots.sort(key=lambda item: abs(
                solution.shifts[item[0]].operations[item[1]].arrival
                - pressure.first_minute
            ))
            seed_quantity = max(
                target_customer.min_operation_quantity, 10.0 * 1e-6,
            )
            for shift_position, operation_position in replacement_slots[
                : max(32, config.samples_per_customer * 8)
            ]:
                shift = solution.shifts[shift_position]
                operations = list(shift.operations)
                operations[operation_position] = Operation(
                    pressure.customer,
                    operations[operation_position].arrival,
                    seed_quantity,
                )
                retimed = try_optimize_shift_times(
                    instance,
                    replace(shift, operations=tuple(operations)),
                    latest_end=_resource_slot_end(
                        instance, solution, derived, shift_position,
                    ),
                )
                if (
                    retimed is None
                    or retimed.operations[operation_position].arrival
                    > pressure.first_minute
                ):
                    continue
                shifts = list(solution.shifts)
                shifts[shift_position] = retimed
                append_candidate(shifts)
                if len(results) >= cap:
                    return results
            continue
        for target_shift_position, target_operation_position in target_positions[:4]:
            target_shift = solution.shifts[target_shift_position]
            target_operation = target_shift.operations[target_operation_position]

            # Proactively reassign a pressure route to a compatible free
            # driver/trailer pair and advance it to the breach or a preceding
            # legal customer window. Ordinary resource reassignment only
            # reacts to existing overlaps; pressure repair must create slack
            # before a violation exists.
            target_offset = target_operation.arrival - target_shift.start
            candidate_starts = {
                max(0, pressure.first_minute - target_offset),
            }
            for window in target_customer.time_windows:
                if window.start <= pressure.first_minute:
                    candidate_starts.add(max(0, window.start - target_offset))
                if window.end <= pressure.first_minute:
                    candidate_starts.add(max(
                        0,
                        window.end - target_customer.setup_time - target_offset,
                    ))
            for driver in instance.drivers:
                for trailer_id in driver.trailer_ids:
                    if not _route_allows_trailer(
                        instance, target_shift, trailer_id,
                    ):
                        continue
                    for start in sorted(candidate_starts):
                        delta = start - target_shift.start
                        reassigned = replace(
                            target_shift,
                            driver=driver.index,
                            trailer=trailer_id,
                            start=start,
                            operations=tuple(
                                replace(
                                    operation,
                                    arrival=operation.arrival + delta,
                                )
                                for operation in target_shift.operations
                            ),
                        )
                        reassigned = try_optimize_shift_times(
                            instance, reassigned,
                        )
                        if reassigned is None:
                            continue
                        moved_arrival = reassigned.operations[
                            target_operation_position
                        ].arrival
                        if moved_arrival > pressure.first_minute:
                            continue
                        reassigned_end = derive_solution(
                            instance,
                            Solution((replace(reassigned, index=0),)),
                        )[0].end
                        if _resource_overlap(
                            instance,
                            solution,
                            derived,
                            target_shift_position,
                            reassigned,
                            reassigned_end,
                        ):
                            continue
                        shifts = list(solution.shifts)
                        shifts[target_shift_position] = reassigned
                        append_candidate(shifts)
                        if len(results) >= cap:
                            return results

            # Evacuate an immediate resource predecessor to another compatible
            # pair, then advance the pressure route on its original resources.
            # This is atomic: neither half is scored independently.
            predecessor_positions = sorted(
                (
                    position
                    for position, (other, replay) in enumerate(
                        zip(solution.shifts, derived)
                    )
                    if position != target_shift_position
                    and replay.end <= target_shift.start
                    and (
                        other.driver == target_shift.driver
                        or other.trailer == target_shift.trailer
                    )
                ),
                key=lambda position: derived[position].end,
                reverse=True,
            )[:2]
            for predecessor_position in predecessor_positions:
                predecessor = solution.shifts[predecessor_position]
                for predecessor_driver in instance.drivers:
                    for predecessor_trailer in predecessor_driver.trailer_ids:
                        if (
                            predecessor_driver.index == predecessor.driver
                            and predecessor_trailer == predecessor.trailer
                        ):
                            continue
                        if not _route_allows_trailer(
                            instance, predecessor, predecessor_trailer,
                        ):
                            continue
                        evacuated = try_optimize_shift_times(
                            instance,
                            replace(
                                predecessor,
                                driver=predecessor_driver.index,
                                trailer=predecessor_trailer,
                            ),
                        )
                        if evacuated is None:
                            continue
                        evacuated_end = derive_solution(
                            instance,
                            Solution((replace(evacuated, index=0),)),
                        )[0].end
                        if _resource_overlap(
                            instance,
                            solution,
                            derived,
                            predecessor_position,
                            evacuated,
                            evacuated_end,
                        ):
                            continue
                        evacuated_shifts = list(solution.shifts)
                        evacuated_shifts[predecessor_position] = evacuated
                        evacuated_solution = Solution(tuple(evacuated_shifts))
                        evacuated_derived = derive_solution(
                            instance, evacuated_solution,
                        )
                        for start in sorted(candidate_starts):
                            delta = start - target_shift.start
                            advanced = try_optimize_shift_times(
                                instance,
                                replace(
                                    target_shift,
                                    start=start,
                                    operations=tuple(
                                        replace(
                                            operation,
                                            arrival=operation.arrival + delta,
                                        )
                                        for operation in target_shift.operations
                                    ),
                                ),
                                latest_end=_resource_slot_end(
                                    instance,
                                    evacuated_solution,
                                    evacuated_derived,
                                    target_shift_position,
                                ),
                            )
                            if advanced is None:
                                continue
                            if advanced.operations[
                                target_operation_position
                            ].arrival > pressure.first_minute:
                                continue
                            advanced_end = derive_solution(
                                instance,
                                Solution((replace(advanced, index=0),)),
                            )[0].end
                            if _resource_overlap(
                                instance,
                                evacuated_solution,
                                evacuated_derived,
                                target_shift_position,
                                advanced,
                                advanced_end,
                            ):
                                continue
                            rebuilt = list(evacuated_shifts)
                            rebuilt[target_shift_position] = advanced
                            append_candidate(rebuilt)
                            if len(results) >= cap:
                                return results

            # Cross-reload relocation within the same route.  Moving a visit
            # across a source boundary changes carried-stock ownership and is
            # therefore followed by global activation/quantity repair.
            without_target = (
                target_shift.operations[:target_operation_position]
                + target_shift.operations[target_operation_position + 1:]
            )
            for insertion in range(len(without_target) + 1):
                operations = (
                    without_target[:insertion]
                    + (target_operation,)
                    + without_target[insertion:]
                )
                if operations == target_shift.operations:
                    continue
                retimed = try_optimize_shift_times(
                    instance,
                    replace(target_shift, operations=operations),
                    latest_end=_resource_slot_end(
                        instance, solution, derived, target_shift_position,
                    ),
                )
                if retimed is None:
                    continue
                if min(
                    operation.arrival for operation in retimed.operations
                    if operation.point == pressure.customer
                ) > pressure.first_minute:
                    continue
                shifts = list(solution.shifts)
                shifts[target_shift_position] = retimed
                append_candidate(shifts)
                if len(results) >= cap:
                    return results

            # Exchange the pressure visit with an earlier nearby customer on
            # another route. This keeps operation counts stable while moving
            # both visits and their load-path obligations atomically.
            exchange_slots = [
                (other_shift_position, other_operation_position)
                for other_shift_position, other_shift in enumerate(solution.shifts)
                if other_shift_position != target_shift_position
                for other_operation_position, other_operation in enumerate(other_shift.operations)
                if (
                    other_operation.point in instance.customer_by_point
                    and other_operation.quantity > 0
                    and other_operation.arrival <= pressure.first_minute
                    and pressure.first_minute - other_operation.arrival <= radius
                    and other_shift.trailer in target_customer.allowed_trailers
                    and target_shift.trailer in instance.customer_by_point[
                        other_operation.point
                    ].allowed_trailers
                )
            ]
            exchange_slots.sort(key=lambda item: abs(
                solution.shifts[item[0]].operations[item[1]].arrival
                - pressure.first_minute
            ))
            head = exchange_slots[: max(16, config.samples_per_customer * 4)]
            rng.shuffle(head)
            # Preserve a structurally mandatory late visit (for example one
            # enabling a layover) while pricing an additional early delivery
            # into an optional VMI slot. Quantity activation can retain only a
            # tiny positive amount at the late visit and assign useful volume
            # to the early duplicate.
            for other_shift_position, other_operation_position in head:
                other_shift = solution.shifts[other_shift_position]
                displaced = other_shift.operations[other_operation_position]
                displaced_customer = instance.customer_by_point[displaced.point]
                if displaced_customer.call_in or displaced_customer.layover_customer:
                    continue
                other_operations = list(other_shift.operations)
                other_operations[other_operation_position] = Operation(
                    pressure.customer,
                    displaced.arrival,
                    max(
                        target_customer.min_operation_quantity,
                        10.0 * 1e-6,
                    ),
                )
                retimed_other = try_optimize_shift_times(
                    instance,
                    replace(other_shift, operations=tuple(other_operations)),
                    latest_end=_resource_slot_end(
                        instance, solution, derived, other_shift_position,
                    ),
                )
                if (
                    retimed_other is None
                    or retimed_other.operations[
                        other_operation_position
                    ].arrival > pressure.first_minute
                ):
                    continue
                shifts = list(solution.shifts)
                shifts[other_shift_position] = retimed_other
                append_candidate(shifts)
                if len(results) >= cap:
                    return results
            # One-way ejection: remove the late pressure visit and replace an
            # earlier optional VMI slot with it. The displaced customer's
            # remaining topology is handled by binary activation and global
            # quantity repair, rather than being forced into an incompatible
            # late route.
            donor_operations = (
                target_shift.operations[:target_operation_position]
                + target_shift.operations[target_operation_position + 1:]
            )
            if donor_operations:
                retimed_donor = try_optimize_shift_times(
                    instance,
                    replace(target_shift, operations=donor_operations),
                    latest_end=_resource_slot_end(
                        instance, solution, derived, target_shift_position,
                    ),
                )
                if retimed_donor is not None:
                    for other_shift_position, other_operation_position in head:
                        other_shift = solution.shifts[other_shift_position]
                        displaced = other_shift.operations[other_operation_position]
                        displaced_customer = instance.customer_by_point[
                            displaced.point
                        ]
                        if displaced_customer.call_in or displaced_customer.layover_customer:
                            continue
                        other_operations = list(other_shift.operations)
                        other_operations[other_operation_position] = replace(
                            target_operation,
                            quantity=min(
                                target_operation.quantity,
                                target_customer.capacity,
                            ),
                        )
                        retimed_other = try_optimize_shift_times(
                            instance,
                            replace(other_shift, operations=tuple(other_operations)),
                            latest_end=_resource_slot_end(
                                instance, solution, derived, other_shift_position,
                            ),
                        )
                        if (
                            retimed_other is None
                            or retimed_other.operations[
                                other_operation_position
                            ].arrival > pressure.first_minute
                        ):
                            continue
                        shifts = list(solution.shifts)
                        shifts[target_shift_position] = retimed_donor
                        shifts[other_shift_position] = retimed_other
                        append_candidate(shifts)
                        if len(results) >= cap:
                            return results
            for other_shift_position, other_operation_position in head:
                other_shift = solution.shifts[other_shift_position]
                other_operation = other_shift.operations[other_operation_position]
                target_operations = list(target_shift.operations)
                other_operations = list(other_shift.operations)
                target_operations[target_operation_position] = replace(
                    other_operation, quantity=min(
                        other_operation.quantity,
                        instance.customer_by_point[other_operation.point].capacity,
                    ),
                )
                other_operations[other_operation_position] = replace(
                    target_operation,
                    quantity=min(target_operation.quantity, target_customer.capacity),
                )
                retimed_target = try_optimize_shift_times(
                    instance,
                    replace(target_shift, operations=tuple(target_operations)),
                    latest_end=_resource_slot_end(
                        instance, solution, derived, target_shift_position,
                    ),
                )
                retimed_other = try_optimize_shift_times(
                    instance,
                    replace(other_shift, operations=tuple(other_operations)),
                    latest_end=_resource_slot_end(
                        instance, solution, derived, other_shift_position,
                    ),
                )
                if retimed_target is None or retimed_other is None:
                    continue
                moved_arrival = retimed_other.operations[
                    other_operation_position
                ].arrival
                if moved_arrival > pressure.first_minute:
                    continue
                shifts = list(solution.shifts)
                shifts[target_shift_position] = retimed_target
                shifts[other_shift_position] = retimed_other
                append_candidate(shifts)
                if len(results) >= cap:
                    return results

            # Three-route ejection cycle: A(target)->B(early), B->C, C->A.
            # This frees the early slot without forcing B's displaced visit
            # into A when its window is incompatible there.
            for b_shift_position, b_operation_position in head[:32]:
                b_shift = solution.shifts[b_shift_position]
                b_operation = b_shift.operations[b_operation_position]
                b_customer = instance.customer_by_point[b_operation.point]
                c_slots = [
                    (c_shift_position, c_operation_position)
                    for c_shift_position, c_shift in enumerate(solution.shifts)
                    if c_shift_position not in {
                        target_shift_position, b_shift_position,
                    }
                    for c_operation_position, c_operation in enumerate(
                        c_shift.operations
                    )
                    if (
                        c_operation.point in instance.customer_by_point
                        and c_operation.quantity > 0
                        and abs(
                            c_operation.arrival - target_operation.arrival
                        ) <= radius
                        and c_shift.trailer in b_customer.allowed_trailers
                        and target_shift.trailer
                        in instance.customer_by_point[
                            c_operation.point
                        ].allowed_trailers
                    )
                ]
                c_slots.sort(key=lambda item: abs(
                    solution.shifts[item[0]].operations[item[1]].arrival
                    - target_operation.arrival
                ))
                for c_shift_position, c_operation_position in c_slots[:24]:
                    c_shift = solution.shifts[c_shift_position]
                    c_operation = c_shift.operations[c_operation_position]
                    a_operations = list(target_shift.operations)
                    b_operations = list(b_shift.operations)
                    c_operations = list(c_shift.operations)
                    a_operations[target_operation_position] = replace(
                        c_operation,
                        quantity=min(
                            c_operation.quantity,
                            instance.customer_by_point[c_operation.point].capacity,
                        ),
                    )
                    b_operations[b_operation_position] = replace(
                        target_operation,
                        quantity=min(
                            target_operation.quantity,
                            target_customer.capacity,
                        ),
                    )
                    c_operations[c_operation_position] = replace(
                        b_operation,
                        quantity=min(b_operation.quantity, b_customer.capacity),
                    )
                    retimed_a = try_optimize_shift_times(
                        instance,
                        replace(target_shift, operations=tuple(a_operations)),
                        latest_end=_resource_slot_end(
                            instance, solution, derived, target_shift_position,
                        ),
                    )
                    retimed_b = try_optimize_shift_times(
                        instance,
                        replace(b_shift, operations=tuple(b_operations)),
                        latest_end=_resource_slot_end(
                            instance, solution, derived, b_shift_position,
                        ),
                    )
                    retimed_c = try_optimize_shift_times(
                        instance,
                        replace(c_shift, operations=tuple(c_operations)),
                        latest_end=_resource_slot_end(
                            instance, solution, derived, c_shift_position,
                        ),
                    )
                    if (
                        retimed_a is None
                        or retimed_b is None
                        or retimed_c is None
                        or retimed_b.operations[
                            b_operation_position
                        ].arrival > pressure.first_minute
                    ):
                        continue
                    shifts = list(solution.shifts)
                    shifts[target_shift_position] = retimed_a
                    shifts[b_shift_position] = retimed_b
                    shifts[c_shift_position] = retimed_c
                    append_candidate(shifts)
                    if len(results) >= cap:
                        return results

            # Exchange equal-length contiguous customer fragments. A
            # two-customer exchange preserves downstream operation count and
            # often keeps tight window chains reachable where inserting or
            # swapping only the pressure visit does not.
            fragment_moves: list[
                tuple[int, int, int, int, int]
            ] = []
            for length in (2, 3):
                for target_start in range(
                    max(0, target_operation_position - length + 1),
                    min(
                        target_operation_position + 1,
                        len(target_shift.operations) - length + 1,
                    ),
                ):
                    target_fragment = target_shift.operations[
                        target_start:target_start + length
                    ]
                    if not all(
                        operation.point in instance.customer_by_point
                        for operation in target_fragment
                    ):
                        continue
                    for other_shift_position, other_shift in enumerate(
                        solution.shifts
                    ):
                        if other_shift_position == target_shift_position:
                            continue
                        for other_start in range(
                            len(other_shift.operations) - length + 1
                        ):
                            other_fragment = other_shift.operations[
                                other_start:other_start + length
                            ]
                            if not all(
                                operation.point in instance.customer_by_point
                                and operation.quantity > 0
                                for operation in other_fragment
                            ):
                                continue
                            if max(
                                operation.arrival
                                for operation in other_fragment
                            ) > pressure.first_minute:
                                continue
                            if pressure.first_minute - max(
                                operation.arrival
                                for operation in other_fragment
                            ) > radius:
                                continue
                            if any(
                                other_shift.trailer
                                not in instance.customer_by_point[
                                    operation.point
                                ].allowed_trailers
                                for operation in target_fragment
                            ):
                                continue
                            if any(
                                target_shift.trailer
                                not in instance.customer_by_point[
                                    operation.point
                                ].allowed_trailers
                                for operation in other_fragment
                            ):
                                continue
                            fragment_moves.append((
                                abs(
                                    max(operation.arrival for operation in other_fragment)
                                    - pressure.first_minute
                                ),
                                target_start,
                                length,
                                other_shift_position,
                                other_start,
                            ))
            fragment_moves.sort()
            for (
                _distance,
                target_start,
                length,
                other_shift_position,
                other_start,
            ) in fragment_moves[: max(32, config.samples_per_customer * 8)]:
                other_shift = solution.shifts[other_shift_position]
                target_fragment = target_shift.operations[
                    target_start:target_start + length
                ]
                other_fragment = other_shift.operations[
                    other_start:other_start + length
                ]
                target_operations = (
                    target_shift.operations[:target_start]
                    + other_fragment
                    + target_shift.operations[target_start + length:]
                )
                other_operations = (
                    other_shift.operations[:other_start]
                    + target_fragment
                    + other_shift.operations[other_start + length:]
                )
                retimed_target = try_optimize_shift_times(
                    instance,
                    replace(target_shift, operations=target_operations),
                    latest_end=_resource_slot_end(
                        instance, solution, derived, target_shift_position,
                    ),
                )
                retimed_other = try_optimize_shift_times(
                    instance,
                    replace(other_shift, operations=other_operations),
                    latest_end=_resource_slot_end(
                        instance, solution, derived, other_shift_position,
                    ),
                )
                if retimed_target is None or retimed_other is None:
                    continue
                moved_target_arrivals = [
                    operation.arrival
                    for operation in retimed_other.operations[
                        other_start:other_start + length
                    ]
                    if operation.point == pressure.customer
                ]
                if (
                    not moved_target_arrivals
                    or min(moved_target_arrivals) > pressure.first_minute
                ):
                    continue
                shifts = list(solution.shifts)
                shifts[target_shift_position] = retimed_target
                shifts[other_shift_position] = retimed_other
                append_candidate(shifts)
                if len(results) >= cap:
                    return results
    return results


def _recombine_route_block_candidates(
    instance: Instance,
    solution: Solution,
    config: SurgicalSearchConfig,
    rng: random.Random,
) -> list[Solution]:
    """Exchange complete customer blocks between two existing routes.

    A block is the consecutive customer subsequence after a source reload (or
    route start) and before its next reload.  The move family first attempts
    whole-route merges, then exchanges blocks.  A merge joins a compatible
    donor route to another route and removes the donor shift: that is the
    density-increasing move (more customer stops/reloads on one active shift).
    Both routes are retimed before the whole transaction reaches scoring.
    """
    result: list[Solution] = []
    cap = max(4, config.candidates_per_move * 2)
    derived = derive_solution(instance, solution)
    safety_margin = {
        (event.point, event.step): event.ending_inventory - event.safety_level
        for event in tank_events(instance, solution)
    }
    merge_moves = [
        (rank, recipient_position, donor_position)
        for recipient_position in range(len(solution.shifts))
        for donor_position in range(len(solution.shifts))
        if recipient_position != donor_position
        for rank in [_merge_prescreen(
            instance,
            solution,
            derived,
            safety_margin,
            recipient_position,
            donor_position,
        )]
        if rank is not None
    ]
    # This ranking cheaply prefers a short physical join and a small change to
    # the donor's original timing.  It avoids spending a timing MIP on random,
    # obviously fragile pairs.
    merge_moves.sort()
    for _rank, recipient_position, donor_position in merge_moves[:cap]:
        recipient = solution.shifts[recipient_position]
        donor = solution.shifts[donor_position]
        merged_operations = recipient.operations + donor.operations
        # The reference profile uses no more than 16 operations per shift.
        # Keep the native neighborhood bounded so a merge remains a targeted
        # topology move instead of creating an opaque mega-route.
        if len(merged_operations) > 16:
            continue
        merged = try_optimize_shift_times(
            instance, replace(recipient, operations=merged_operations),
        )
        if merged is None or not _route_allows_trailer(
            instance, merged, recipient.trailer,
        ):
            continue
        shifts = [
            merged if position == recipient_position else shift
            for position, shift in enumerate(solution.shifts)
            if position != donor_position
        ]
        result.append(normalize_source_loads(
            instance, _reindex(Solution(tuple(shifts))),
        ))
        if len(result) >= cap:
            return result

    block_positions = [
        (position, block)
        for position, shift in enumerate(solution.shifts)
        for block in _customer_blocks(instance, shift)
    ]
    moves = [
        (left_position, left_block, right_position, right_block)
        for left_position, left_block in block_positions
        for right_position, right_block in block_positions
        if left_position < right_position
    ]
    rng.shuffle(moves)
    for left_position, left_block, right_position, right_block in moves:
        shifts = list(solution.shifts)
        left, right = shifts[left_position], shifts[right_position]
        left_insert = right.operations[right_block[0]:right_block[1]]
        right_insert = left.operations[left_block[0]:left_block[1]]
        left_operations = left.operations[:left_block[0]] + left_insert + left.operations[left_block[1]:]
        right_operations = right.operations[:right_block[0]] + right_insert + right.operations[right_block[1]:]
        mutated_left = try_optimize_shift_times(instance, replace(left, operations=left_operations))
        mutated_right = try_optimize_shift_times(instance, replace(right, operations=right_operations))
        if mutated_left is None or mutated_right is None:
            continue
        if not _route_allows_trailer(instance, mutated_left, left.trailer):
            continue
        if not _route_allows_trailer(instance, mutated_right, right.trailer):
            continue
        shifts[left_position] = mutated_left
        shifts[right_position] = mutated_right
        result.append(normalize_source_loads(
            instance, _reindex(Solution(tuple(shifts))),
        ))
        if len(result) >= cap:
            break
    return result


def _merge_prescreen(
    instance: Instance,
    solution: Solution,
    derived,
    safety_margin: dict[tuple[int, int], float],
    recipient_position: int,
    donor_position: int,
) -> tuple[int, int] | None:
    """Cheap necessary checks before a merge invokes the timing MIP."""
    recipient = solution.shifts[recipient_position]
    donor = solution.shifts[donor_position]
    operations = recipient.operations + donor.operations
    if len(operations) > 16 or not _route_allows_trailer(
        instance, replace(recipient, operations=operations), recipient.trailer,
    ):
        return None
    arrivals, predicted_end = _earliest_route_arrivals(instance, recipient, operations)
    driver = instance.drivers[recipient.driver]
    if not any(
        window.start <= recipient.start and predicted_end <= window.end
        for window in driver.time_windows
    ):
        return None
    if _merge_resource_overlap(
        instance, solution, derived, recipient_position, donor_position, predicted_end,
    ):
        return None
    donor_offset = len(recipient.operations)
    for local_index, operation in enumerate(donor.operations):
        customer = instance.customer_by_point.get(operation.point)
        if customer is None:
            continue
        earliest = arrivals[donor_offset + local_index]
        latest = _latest_arrival(instance, customer)
        if earliest > latest:
            return None
        if customer.call_in:
            continue
        old_step = operation.arrival // instance.unit
        new_step = earliest // instance.unit
        if new_step <= old_step:
            continue
        # Delaying this known delivery removes its quantity from every bucket
        # before its new earliest arrival. If the incumbent buffer cannot cover
        # that loss, no retiming can make this merge tank-safe.
        if any(
            safety_margin.get((customer.index, step), float("-inf"))
            < operation.quantity - 1e-6
            for step in range(old_step, min(new_step, instance.horizon))
        ):
            return None
    join_time = instance.time_matrix[recipient.operations[-1].point][donor.operations[0].point]
    delay = max(0, arrivals[donor_offset] - donor.operations[0].arrival)
    return (join_time, delay)


def _earliest_route_arrivals(
    instance: Instance,
    shift: Shift,
    operations: tuple[Operation, ...],
) -> tuple[tuple[int, ...], int]:
    """Physical lower-bound schedule; it intentionally ignores optional waits."""
    point = instance.base_index
    departure = shift.start
    arrivals: list[int] = []
    for operation in operations:
        arrival = departure + instance.time_matrix[point][operation.point]
        arrivals.append(arrival)
        departure = arrival + instance.setup_time_for_point(operation.point)
        point = operation.point
    return tuple(arrivals), departure + instance.time_matrix[point][instance.base_index]


def _latest_arrival(instance: Instance, customer) -> int:
    setup = customer.setup_time
    if not customer.call_in:
        return max(window.end - setup for window in customer.time_windows)
    return max(
        min(window.end - setup, order.latest_time)
        for window in customer.time_windows
        for order in customer.orders
        if max(window.start, order.earliest_time) <= min(window.end - setup, order.latest_time)
    )


def _merge_resource_overlap(
    instance: Instance,
    solution: Solution,
    derived,
    recipient_position: int,
    donor_position: int,
    candidate_end: int,
) -> bool:
    candidate = solution.shifts[recipient_position]
    driver_gap = instance.drivers[candidate.driver].min_inter_shift_duration
    for position, other in enumerate(solution.shifts):
        if position in {recipient_position, donor_position}:
            continue
        other_end = derived[position].end
        if other.driver == candidate.driver and (
            candidate.start < other_end + driver_gap
            and other.start < candidate_end + driver_gap
        ):
            return True
        if other.trailer == candidate.trailer and (
            candidate.start < other_end and other.start < candidate_end
        ):
            return True
    return False


def _customer_blocks(instance: Instance, shift: Shift) -> tuple[tuple[int, int], ...]:
    """Return non-empty half-open customer runs, bounded by source visits."""
    blocks: list[tuple[int, int]] = []
    start: int | None = None
    for index, operation in enumerate(shift.operations):
        is_customer = operation.point in instance.customer_by_point
        if is_customer and start is None:
            start = index
        if not is_customer and start is not None:
            blocks.append((start, index))
            start = None
    if start is not None:
        blocks.append((start, len(shift.operations)))
    return tuple(blocks)


def _compact_after_activation(
    instance: Instance,
    solution: Solution,
) -> Solution | None:
    """Retime routes after optional VMI operations are deactivated."""
    current = _reindex(solution)
    derived = derive_solution(instance, current)
    shifts = list(current.shifts)
    for position, shift in enumerate(current.shifts):
        retimed = try_optimize_shift_times(
            instance,
            shift,
            latest_end=_resource_slot_end(
                instance, current, derived, position,
            ),
        )
        if retimed is None:
            return None
        shifts[position] = retimed
    return _reindex(Solution(tuple(shifts)))


def _joint_retailer_reinsert_candidates(
    instance: Instance,
    solution: Solution,
    config: SurgicalSearchConfig,
) -> list[Solution]:
    """Move one retailer refill earlier while rebalancing its later plan.

    This is the smallest useful IRP destroy/reinsert neighbourhood. It adds a
    time-feasible early source route and removes the same volume from the
    retailer's later deliveries, rather than creating stock from nowhere or
    overfilling its tank.  The full route plan is the candidate, so resource,
    inventory, and routing effects are evaluated atomically by the caller.
    """
    result: list[Solution] = []
    cap = max(4, config.candidates_per_move * 2)
    pressure = pressure_points(instance, solution, end_day=config.end_day)
    rescue_config = RescueConfig(
        start_day=0,
        end_day=config.end_day,
        replace_from_day=0,
        max_customers=1,
        samples_per_customer=max(4, config.samples_per_customer),
        allow_future_rebalance=True,
    )
    for point in pressure[: config.pressure_customers]:
        customer = instance.customer_by_point[point.customer]
        if customer.call_in:
            continue
        direct_columns = generate_rescue_candidates(
            instance, solution, [customer.index], config=rescue_config,
        )
        for direct in direct_columns:
            trial = _rebalance_retailer_after_early_column(
                instance, solution, customer.index, direct,
            )
            if trial is None:
                continue
            result.append(trial)
            if len(result) >= cap:
                return result
    return result


def _rebalance_retailer_after_early_column(
    instance: Instance,
    solution: Solution,
    customer_id: int,
    early_shift: Shift,
) -> Solution | None:
    """Add ``early_shift`` and offset it against later retailer deliveries."""
    early_deliveries = [
        operation for operation in early_shift.operations
        if operation.point == customer_id and operation.quantity > 0
    ]
    if len(early_deliveries) != 1:
        return None
    early = early_deliveries[0]
    customer = instance.customer_by_point[customer_id]
    # An early refill must be fully offset by future visits while preserving
    # every operation's minimum quantity.  Without this cap a direct column
    # can add more stock than the next visit is legally able to surrender,
    # creating an overfill now or a later runout after a crude repair.
    reducible_future = sum(
        max(0.0, operation.quantity - customer.min_operation_quantity)
        for shift in solution.shifts
        for operation in shift.operations
        if (
            operation.point == customer_id
            and operation.quantity > 0
            and operation.arrival > early.arrival
        )
    )
    early_quantity = min(early.quantity, reducible_future)
    # When the cap is binding, retain a small amount in the later delivery.
    # The XML quantities are decimal text but the simulator accumulates binary
    # floats; moving the theoretical exact remainder can otherwise create a
    # few-hundredths-of-a-litre capacity violation at the later arrival.
    if early_quantity < early.quantity - 1e-6:
        early_quantity = max(0.0, early_quantity - 0.1)
    if early_quantity < customer.min_operation_quantity - 1e-6:
        return None
    if early_quantity + 1e-6 < early.quantity:
        early = replace(early, quantity=early_quantity)
        early_shift = replace(
            early_shift,
            operations=tuple(
                early if operation == early_deliveries[0] else operation
                for operation in early_shift.operations
            ),
        )
    remaining = early.quantity
    adjusted: list[Shift] = []
    for shift in solution.shifts:
        operations = []
        for operation in shift.operations:
            if (
                operation.point != customer_id
                or operation.quantity <= 0
                or operation.arrival <= early.arrival
                or remaining <= 1e-6
            ):
                operations.append(operation)
                continue
            reducible = max(0.0, operation.quantity - customer.min_operation_quantity)
            reduction = min(reducible, remaining)
            operations.append(replace(operation, quantity=operation.quantity - reduction))
            remaining -= reduction
        adjusted.append(replace(shift, operations=tuple(operations)))
    if remaining > 1e-6:
        return None
    adjusted.append(replace(early_shift, index=len(adjusted)))
    candidate = _retime_joint_blockers(
        instance,
        _reindex(Solution(tuple(adjusted))),
    )
    if candidate is None:
        return None
    candidate = normalize_source_loads(instance, candidate)
    # A joint move is allowed to retain the incumbent's quantity shortages,
    # but it must never introduce a physical/routing defect elsewhere. In
    # particular, source-load normalisation can reveal a delivery before a
    # prior reload on the same trailer. Keep this gate global: checking just
    # the retailer being reinserted would leak such malformed routes into the
    # search pool.
    candidate_errors = [
        violation for violation in validate_solution(instance, candidate)
        if violation.severity == "error"
    ]
    if any(
        not violation.code.startswith(("QS01", "QS02"))
        for violation in candidate_errors
    ):
        return None
    return candidate


def _retime_joint_blockers(
    instance: Instance,
    candidate: Solution,
) -> Solution | None:
    """Push routes that conflict with the new early route as one transaction.

    Only routes linked by the early route's driver/trailer resource chain are
    touched. A shifted route then propagates its new end to its own following
    driver/trailer work, preventing a local repair from hiding a downstream
    overlap. The timing MIP verifies each rebuilt route against its customer
    windows before the final global scorer evaluates the endpoint.
    """
    early_position = len(candidate.shifts) - 1
    early = candidate.shifts[early_position]
    early_end = derive_solution(instance, Solution((replace(early, index=0),)))[0].end
    driver_ready = {
        early.driver: early_end + instance.drivers[early.driver].min_inter_shift_duration,
    }
    trailer_ready = {early.trailer: early_end}
    shifts = list(candidate.shifts)
    for position in sorted(
        range(len(shifts) - 1), key=lambda item: (shifts[item].start, item),
    ):
        shift = shifts[position]
        required_start = max(
            driver_ready.get(shift.driver, -10**9),
            trailer_ready.get(shift.trailer, -10**9),
        )
        if shift.start < required_start:
            delta = required_start - shift.start
            shifted = replace(
                shift,
                start=shift.start + delta,
                operations=tuple(
                    replace(operation, arrival=operation.arrival + delta)
                    for operation in shift.operations
                ),
            )
            shifted = try_optimize_shift_times(instance, shifted)
            if shifted is None:
                return None
            shifts[position] = shifted
            shift = shifted
        # Propagate only resources that participate in the rebuilt chain.
        if shift.driver in driver_ready or shift.trailer in trailer_ready:
            end = derive_solution(instance, Solution((replace(shift, index=0),)))[0].end
            driver_ready[shift.driver] = end + instance.drivers[shift.driver].min_inter_shift_duration
            trailer_ready[shift.trailer] = end
    return _reindex(Solution(tuple(shifts)))


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


def _point_difference_count(
    left: Solution,
    right: Solution,
) -> int:
    if len(left.shifts) != len(right.shifts):
        return max(len(left.shifts), len(right.shifts))
    return sum(
        left_operation.point != right_operation.point
        for left_shift, right_shift in zip(left.shifts, right.shifts)
        for left_operation, right_operation in zip(
            left_shift.operations, right_shift.operations,
        )
    )


def _operation_count_difference(
    left: Solution,
    right: Solution,
) -> bool:
    return (
        len(left.shifts) != len(right.shifts)
        or any(
            len(left_shift.operations) != len(right_shift.operations)
            for left_shift, right_shift in zip(
                left.shifts, right.shifts,
            )
        )
    )


def _is_direct_route_rebuild(
    current: Solution,
    candidate: Solution,
) -> bool:
    if len(current.shifts) != len(candidate.shifts):
        return False
    return any(
        old_shift.operations != new_shift.operations
        and len(new_shift.operations) == 2
        and new_shift.operations[0].quantity < 0
        and new_shift.operations[1].quantity > 0
        for old_shift, new_shift in zip(
            current.shifts, candidate.shifts,
        )
    )


def _rebuilt_route_target(
    current: Solution,
    candidate: Solution,
    pressure_targets: set[int],
) -> int | None:
    if len(current.shifts) != len(candidate.shifts):
        return None
    for current_shift, candidate_shift in zip(
        current.shifts, candidate.shifts,
    ):
        if (
            current_shift.operations == candidate_shift.operations
            or len(candidate_shift.operations) != 2
            or candidate_shift.operations[0].quantity >= 0
            or candidate_shift.operations[1].quantity <= 0
        ):
            continue
        targets = [
            operation.point
            for operation in candidate_shift.operations
            if (
                operation.quantity > 0
                and operation.point in pressure_targets
            )
        ]
        if len(targets) == 1:
            return targets[0]
    return None


def _select_operator(rewards, attempts, last_used, iteration, rng, feasibility_bias):
    # Match the adaptive selection mechanics from Ghidra (0x140023580):
    # Operator score = EWMA reward * recency multiplier (2.0 if unused for >=33 iterations) * feasibility bias
    scores = []
    for i in range(len(OPERATORS)):
        recency = 2.0 if (iteration - last_used[i]) >= 33 else 1.0
        bias = 4.0 if feasibility_bias and i in (0, 1, 3) else 1.0
        scores.append(max(1.0, rewards[i]) * recency * bias)
    draw = rng.random() * sum(scores)
    for i, value in enumerate(scores):
        draw -= value
        if draw <= 0:
            return i
    return len(OPERATORS) - 1


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


def _hard_invariants_not_worse(
    before: ViolationVector,
    after: ViolationVector,
) -> bool:
    """Keep physical/service/resource feasibility as a commit invariant."""
    fields = (
        "non_finite_values",
        "reference_errors",
        "physical_errors",
        "missed_orders",
        "missed_order_deficit",
        "negative_quantity_minutes",
        "overfill_quantity_minutes",
        "resource_timing_errors",
        "other_errors",
    )
    return all(
        getattr(after, field) <= getattr(before, field) + 1e-6
        for field in fields
    )


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
