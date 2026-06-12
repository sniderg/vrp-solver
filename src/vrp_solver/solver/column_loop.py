from __future__ import annotations

import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

from ..contest import score_prefix_with_feasibility_tail
from ..inventory import tank_events
from ..model import Instance, Shift, Solution
from ..highs_repair import repair_quantities_with_highs
from ..highs_time_opt import optimize_solution_times
from ..route_cache import RouteCache
from ..rules import derive_solution
from .ml_priors import MLRoutePriors
from .highs_selector import (
    SelectorConfig,
    _candidate_pressure_bonus,
    _inventory_pressure_by_customer,
    _inventory_checkpoint_steps,
    select_shifts_with_highs,
    _candidate_intervals,
    _intervals_overlap,
)
from .pressure import pressure_points
from .route_priors import RoutePriorDiagnostics, route_signature
from .targeted_rescue import (
    MINUTES_PER_DAY,
    RescueConfig,
    _baseline_window_shifts,
    _dedupe_reindex,
    _failing_customers,
    _keep_shifts_started_before,
    generate_chain_rescue_candidates,
    generate_carryover_rescue_candidates,
    generate_multi_reload_candidates,
    generate_rescue_candidates,
    normalize_source_loads,
)


@dataclass(frozen=True)
class ColumnLoopConfig:
    start_day: int = 0
    end_day: int = 14
    replace_from_day: int = 3
    iterations: int = 3
    max_pressure_customers: int = 12
    neighbors_per_anchor: int = 8
    batch_workers: int = 4
    samples_per_customer: int = 8
    sample_lookback_days: int = 14
    max_chain_length: int = 4
    nearest_chain_neighbors: int = 4
    max_candidates_per_iteration: int = 1200
    target_fill_ratio: float = 0.95
    max_pre_service_fill_ratio: float = 0.95
    multi_reload_columns: bool = False
    max_multi_reload_per_batch: int = 20
    normalize_source_loads: bool = True
    quantity_objective: str = "max-delivered"
    commit_end_day: int = 7
    next_after_commit_day: int | None = None
    route_prior_candidates: tuple[Shift, ...] = ()
    route_prior_sources: tuple[str, ...] = ()
    selector_time_limit: float = 300.0
    selector_mip_gap: float | None = None
    selector_threads: int | None = None
    selector_mip_focus: int | None = None
    selector_node_limit: int | None = None
    selector_phase: str = "auto"
    candidate_cache_dir: str | None = None
    bucket_anchor_cap: int = 80
    bucket_resource_time_cap: int = 12
    bucket_route_signature_cap: int = 8
    bucket_source_region_cap: int = 240
    protect_exact_prior_quantities: bool = False
    use_prior_incumbent: bool = True
    cap_new_candidate_quantities: bool = True


@dataclass(frozen=True)
class ColumnLoopStep:
    iteration: int
    generated_candidates: int
    pool_size: int
    selected_extra_shifts: int
    feasible: bool
    feasibility_errors: int
    hard_violations: int
    first_safety_breach_minute: int | None
    cost: float = 0.0
    logistic_ratio: float = 0.0
    min_commit_doi: float = 999.0
    vulnerable_commit_customers: list[tuple[int, float]] = None
    min_lookahead_doi: float = 999.0
    vulnerable_lookahead_customers: list[tuple[int, float]] = None
    next_after_commit_day: int | None = None
    min_next_after_commit_doi: float = 999.0
    vulnerable_next_after_commit_customers: list[tuple[int, float]] = None
    prior_loaded: int = 0
    prior_structurally_valid: int = 0
    prior_inserted: int = 0
    prior_selected: int = 0
    prior_rejected: int = 0
    prior_skeletons_regenerated: int = 0
    prior_rejected_pressure_cover: int = 0
    prior_rejected_route_summary: str = ""
    pressure_customers: tuple[int, ...] = ()
    generated_served_customers: int = 0
    generated_pressure_coverage: int = 0
    selected_pressure_coverage: int = 0
    selected_feasibility_errors: int = 0
    selected_hard_violations: int = 0
    repaired_feasibility_errors: int = 0
    repaired_hard_violations: int = 0
    candidate_accepted: bool = False
    best_improved: bool = False


def get_vulnerabilities(
    instance: Instance,
    solution: Solution,
    day: int,
) -> tuple[float, list[tuple[int, float]]]:
    """Calculate the remaining days of inventory (DOI) for each customer at the end of the given day."""
    from ..inventory import days_of_inventory
    # If the fast cython project_inventory is available, this call is extremely fast
    from ..inventory import tank_events
    events = list(tank_events(instance, solution))
    step_cutoff = min((day * 1440) // instance.unit, instance.horizon) - 1
    if step_cutoff < 0:
        step_cutoff = 0
    
    customer_inventories = {}
    for event in events:
        if event.step == step_cutoff:
            customer_inventories[event.point] = event.ending_inventory
            
    vulnerabilities = []
    for customer in instance.customers:
        if customer.call_in:
            continue
        current_inv = customer_inventories.get(customer.index, customer.initial_tank_quantity)
        doi = days_of_inventory(instance, customer, current_inv, start_step=step_cutoff + 1)
        vulnerabilities.append((customer.index, doi))
        
    vulnerabilities.sort(key=lambda x: x[1])
    min_doi = vulnerabilities[0][1] if vulnerabilities else 999.0
    return min_doi, vulnerabilities[:3]


def column_generation_rescue(
    instance: Instance,
    baseline: Solution,
    *,
    config: ColumnLoopConfig = ColumnLoopConfig(),
    ml_priors: MLRoutePriors | None = None,
) -> tuple[Solution, tuple[ColumnLoopStep, ...]]:
    fixed_prefix = _keep_shifts_started_before(
        baseline,
        config.replace_from_day * MINUTES_PER_DAY,
    )
    baseline_window = list(_baseline_window_shifts(baseline, _rescue_config(config)))
    raw_pool = [*baseline_window, *config.route_prior_candidates]
    pool = _dedupe_reindex(raw_pool)
    prior_diagnostics = _prior_diagnostics(instance, config, pool, ())
    steps: list[ColumnLoopStep] = []
    baseline_score = score_prefix_with_feasibility_tail(
        instance,
        baseline,
        score_days=config.end_day,
        feasibility_days=config.end_day,
    )
    prior_incumbent = _prior_incumbent_solution(instance, fixed_prefix, config)
    prior_incumbent_score = (
        score_prefix_with_feasibility_tail(
            instance,
            prior_incumbent,
            score_days=config.end_day,
            feasibility_days=config.end_day,
        )
        if prior_incumbent is not None
        else None
    )
    if (
        config.use_prior_incumbent
        and prior_incumbent is not None
        and prior_incumbent_score is not None
        and _score_key(prior_incumbent_score) <= _score_key(baseline_score)
    ):
        best_solution = prior_incumbent
        best_score = prior_incumbent_score
    else:
        best_solution = baseline
        best_score = baseline_score

    current = best_solution
    current_score = best_score
    no_improvement_streak = 0
    for iteration in range(config.iterations):
        pressure_customers = _pressure_customers(instance, current, config)
        # Phase 3: Augment pressure list with ML-predicted high-priority customers
        if ml_priors is not None:
            pressure_customers = _augment_with_ml_priorities(
                instance, current, config, ml_priors, pressure_customers,
            )
        # Use current solution as the generation baseline after the first
        # iteration so that breach times, trailer loads, and inventory states
        # evolve between iterations, producing genuinely new candidates.
        generation_baseline = fixed_prefix if iteration == 0 else current
        generated = _cached_generate_priced_batches(
            instance,
            generation_baseline,
            pressure_customers,
            config,
        )
        # Filter out candidates that conflict with the fixed prefix shifts
        generated = _filter_prefix_conflicts(instance, fixed_prefix, generated)
        pressure = _inventory_pressure_by_customer(
            instance,
            fixed_prefix,
            config.replace_from_day,
            config.end_day,
        )
        generated = _top_diverse_columns(instance, generated, pressure, config)

        # Cap delivery quantities for new candidates to avoid causing downstream
        # overfill when combined with existing baseline window deliveries.
        if config.cap_new_candidate_quantities:
            budgets = _safe_delivery_budgets(
                instance, fixed_prefix, baseline_window,
                config.replace_from_day, config.end_day,
            )
            generated = _apply_delivery_budgets(instance, generated, budgets)

        prev_pool_size = len(pool)
        pool = _dedupe_reindex([*pool, *generated])
        
        if len(pool) == prev_pool_size and no_improvement_streak >= 2:
            # Convergence: pool has stopped growing AND score hasn't improved
            # for 2 consecutive iterations — genuinely stuck.
            break

        selected = select_shifts_with_highs(
            instance,
            fixed_prefix,
            pool,
            start_day=config.replace_from_day,
            end_day=config.end_day,
            pressure_pricing=True,
            selector_config=SelectorConfig(
                time_limit=config.selector_time_limit,
                mip_gap=config.selector_mip_gap,
                threads=config.selector_threads,
                mip_focus=config.selector_mip_focus,
                node_limit=config.selector_node_limit,
                selector_phase=_selector_phase_for_iteration(config, best_score),
                mip_start_shift_indices=_mip_start_indices(
                    instance,
                    pool,
                    baseline_window,
                    config.route_prior_candidates,
                ),
                priority_shift_indices=_priority_prior_indices(
                    instance,
                    pool,
                    config.route_prior_candidates,
                    pressure_customers,
                ),
            ),
            ml_priors=ml_priors,
        )
        selected = optimize_solution_times(instance, selected)
        if config.normalize_source_loads:
            selected = normalize_source_loads(instance, selected)
        selected_score = score_prefix_with_feasibility_tail(
            instance,
            selected,
            score_days=config.end_day,
            feasibility_days=config.end_day,
        )
        repaired, repair_report = repair_quantities_with_highs(
            instance,
            selected,
            score_days=config.end_day,
            feasibility_days=config.end_day,
            quantity_objective=config.quantity_objective,
            fixed_shift_indices=(
                _fixed_prior_shift_indices(instance, selected, _exact_route_priors(config))
                if config.protect_exact_prior_quantities
                else None
            ),
        )
        if config.normalize_source_loads:
            repaired = normalize_source_loads(instance, repaired)
        repaired_score = score_prefix_with_feasibility_tail(
            instance,
            repaired,
            score_days=config.end_day,
            feasibility_days=config.end_day,
        )
        candidate = repaired if _score_key(repaired_score) <= _score_key(selected_score) else selected
        candidate_score = repaired_score if candidate is repaired else selected_score
        accepted = _score_key(candidate_score) <= _score_key(current_score)
        if accepted:
            current = candidate
            current_score = candidate_score
        score = current_score
        improved_best = _score_key(score) < _score_key(best_score)
        if improved_best:
            best_score = score
            best_solution = current
            no_improvement_streak = 0
        else:
            no_improvement_streak += 1

        logistic_ratio = score.scored_estimated_cost / max(1.0, score.scored_delivered_quantity)
        min_commit_doi, vuln_commit = get_vulnerabilities(instance, current, config.commit_end_day)
        next_after_commit_day = (
            config.next_after_commit_day
            if config.next_after_commit_day is not None
            else min(config.end_day, config.commit_end_day + 7)
        )
        min_next_doi, vuln_next = get_vulnerabilities(instance, current, next_after_commit_day)
        min_lookahead_doi, vuln_lookahead = get_vulnerabilities(instance, current, config.end_day)
        prior_diagnostics = _prior_diagnostics(instance, config, pool, selected.shifts)

        steps.append(
            ColumnLoopStep(
                iteration=iteration,
                generated_candidates=len(generated),
                pool_size=len(pool),
                selected_extra_shifts=max(0, len(selected.shifts) - len(fixed_prefix.shifts)),
                feasible=score.feasible,
                feasibility_errors=score.feasibility_errors,
                hard_violations=score.hard_violations,
                first_safety_breach_minute=score.first_safety_breach_minute,
                cost=score.scored_estimated_cost,
                logistic_ratio=logistic_ratio,
                min_commit_doi=min_commit_doi,
                vulnerable_commit_customers=vuln_commit,
                next_after_commit_day=next_after_commit_day,
                min_next_after_commit_doi=min_next_doi,
                vulnerable_next_after_commit_customers=vuln_next,
                min_lookahead_doi=min_lookahead_doi,
                vulnerable_lookahead_customers=vuln_lookahead,
                prior_loaded=prior_diagnostics.loaded,
                prior_structurally_valid=prior_diagnostics.structurally_valid,
                prior_inserted=prior_diagnostics.inserted,
                prior_selected=prior_diagnostics.selected,
                prior_rejected=prior_diagnostics.rejected,
                prior_skeletons_regenerated=prior_diagnostics.skeletons_regenerated,
                prior_rejected_pressure_cover=prior_diagnostics.rejected_pressure_cover,
                prior_rejected_route_summary=prior_diagnostics.rejected_route_summary,
                pressure_customers=tuple(pressure_customers),
                generated_served_customers=len(
                    _served_customers_in_shifts(instance, generated)
                ),
                generated_pressure_coverage=len(
                    set(pressure_customers)
                    & _served_customers_in_shifts(instance, generated)
                ),
                selected_pressure_coverage=len(
                    set(pressure_customers)
                    & _served_customers_in_solution(
                        instance,
                        selected,
                        start_minute=config.replace_from_day * MINUTES_PER_DAY,
                        end_minute=config.end_day * MINUTES_PER_DAY,
                    )
                ),
                selected_feasibility_errors=selected_score.feasibility_errors,
                selected_hard_violations=selected_score.hard_violations,
                repaired_feasibility_errors=repaired_score.feasibility_errors,
                repaired_hard_violations=repaired_score.hard_violations,
                candidate_accepted=accepted,
                best_improved=improved_best,
            )
        )
        if score.feasible:
            best_solution = current
            break

    reindexed = Solution(
        shifts=tuple(replace(shift, index=index) for index, shift in enumerate(best_solution.shifts))
    )
    return reindexed, tuple(steps)


def _rescue_config(config: ColumnLoopConfig) -> RescueConfig:
    return RescueConfig(
        start_day=config.start_day,
        end_day=config.end_day,
        replace_from_day=config.replace_from_day,
        max_customers=config.max_pressure_customers,
        samples_per_customer=config.samples_per_customer,
        target_fill_ratio=config.target_fill_ratio,
        max_pre_service_fill_ratio=config.max_pre_service_fill_ratio,
        sample_lookback_days=config.sample_lookback_days,
        max_chain_length=config.max_chain_length,
        nearest_chain_neighbors=config.nearest_chain_neighbors,
        repair_quantities=False,
        normalize_source_loads=config.normalize_source_loads,
        quantity_objective=config.quantity_objective,
    )


def _pressure_customers(
    instance: Instance,
    solution: Solution,
    config: ColumnLoopConfig,
) -> list[int]:
    ranked = pressure_points(instance, solution, end_day=config.end_day)
    if ranked:
        return [point.customer for point in ranked[: config.max_pressure_customers]]
    cutoff_step = min(instance.horizon, config.end_day * MINUTES_PER_DAY // instance.unit)
    breach_scores: dict[int, tuple[int, float]] = {}
    urgency_scores: dict[int, tuple[int, float]] = {}
    for event in tank_events(instance, solution):
        if event.step >= cutoff_step or event.point not in instance.customer_by_point:
            continue
        customer = instance.customer_by_point[event.point]
        if customer.call_in:
            continue
        deficit = max(0.0, event.safety_level - event.ending_inventory)
        if deficit > 0.0:
            first, total = breach_scores.get(event.point, (event.step, 0.0))
            breach_scores[event.point] = (min(first, event.step), total + deficit)
            continue
        urgency = max(0.0, customer.safety_level + 0.15 * customer.capacity - event.ending_inventory)
        if urgency > 0.0:
            first, total = urgency_scores.get(event.point, (event.step, 0.0))
            urgency_scores[event.point] = (min(first, event.step), total + urgency)

    breach_order = sorted(
        breach_scores,
        key=lambda point: (breach_scores[point][0], -breach_scores[point][1], point),
    )
    urgency_order = sorted(
        urgency_scores,
        key=lambda point: (urgency_scores[point][0], -urgency_scores[point][1], point),
    )
    ordered = [*breach_order]
    ordered.extend(point for point in urgency_order if point not in breach_scores)
    if not ordered:
        return _failing_customers(instance, solution, _rescue_config(config))
    return ordered[: config.max_pressure_customers]


def _augment_with_ml_priorities(
    instance: Instance,
    solution: Solution,
    config: ColumnLoopConfig,
    ml_priors: MLRoutePriors,
    pressure_customers: list[int],
) -> list[int]:
    """Augment pressure customer list with ML-predicted high-priority customers.

    Phase 3: Ensures the candidate pool contains shifts for customers the ML
    model predicts are critical, even if they haven't technically breached yet.
    """
    from .ml_priors import get_start_inventories

    existing = set(pressure_customers)
    # Compute max prize across all days in the window
    prizes_by_day = ml_priors.predict_prizes_by_day(
        instance, solution, config.replace_from_day, config.end_day,
    )
    max_prizes: dict[int, float] = {}
    for day_prizes in prizes_by_day.values():
        for cid, prize in day_prizes.items():
            max_prizes[cid] = max(max_prizes.get(cid, 0.0), prize)

    # Sort by descending max prize, take top-K not already in the list
    ml_ranked = sorted(max_prizes, key=lambda c: max_prizes[c], reverse=True)
    # Add up to half the pressure cap as ML-suggested customers
    ml_budget = max(1, config.max_pressure_customers // 2)
    augmented = list(pressure_customers)
    for cid in ml_ranked:
        if cid in existing:
            continue
        customer = instance.customer_by_point.get(cid)
        if customer is None or customer.call_in:
            continue
        augmented.append(cid)
        existing.add(cid)
        ml_budget -= 1
        if ml_budget <= 0:
            break

    return augmented


def _generate_priced_batches(
    instance: Instance,
    prefix: Solution,
    pressure_customers: list[int],
    config: ColumnLoopConfig,
) -> list[Shift]:
    if not pressure_customers:
        return []
    batches = [
        _anchor_batch(instance, pressure_customers, anchor, config)
        for anchor in pressure_customers
    ]
    rescue_config = _rescue_config(config)

    def generate(batch: list[int]) -> list[Shift]:
        candidates = generate_rescue_candidates(instance, prefix, batch, config=rescue_config)
        candidates.extend(generate_carryover_rescue_candidates(instance, prefix, batch, config=rescue_config))
        candidates.extend(generate_chain_rescue_candidates(instance, prefix, batch, config=rescue_config))
        if config.multi_reload_columns:
            candidates.extend(
                generate_multi_reload_candidates(instance, prefix, batch, config=rescue_config)[
                    : config.max_multi_reload_per_batch
                ]
            )
        return candidates

    if config.batch_workers <= 1:
        nested = [generate(batch) for batch in batches]
    else:
        with ThreadPoolExecutor(max_workers=config.batch_workers) as executor:
            nested = list(executor.map(generate, batches))
    return [candidate for batch in nested for candidate in batch]


def _cached_generate_priced_batches(
    instance: Instance,
    prefix: Solution,
    pressure_customers: list[int],
    config: ColumnLoopConfig,
) -> list[Shift]:
    if config.candidate_cache_dir is None:
        return _generate_priced_batches(instance, prefix, pressure_customers, config)
    cache_dir = Path(config.candidate_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _candidate_cache_key(instance, prefix, pressure_customers, config)
    cache_path = cache_dir / f"{key}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            return pickle.load(handle)
    candidates = _generate_priced_batches(instance, prefix, pressure_customers, config)
    with cache_path.open("wb") as handle:
        pickle.dump(candidates, handle)
    return candidates


def _anchor_batch(
    instance: Instance,
    pressure_customers: list[int],
    anchor: int,
    config: ColumnLoopConfig,
) -> list[int]:
    neighbors = sorted(
        (customer for customer in pressure_customers if customer != anchor),
        key=lambda customer: (instance.time_matrix[anchor][customer], customer),
    )[: config.neighbors_per_anchor]
    return [anchor, *neighbors]


def _filter_prefix_conflicts(
    instance: Instance,
    prefix: Solution,
    candidates: list[Shift],
) -> list[Shift]:
    if not candidates:
        return []
    intervals = _candidate_intervals(instance, candidates)
    prefix_derived = derive_solution(instance, prefix)
    prefix_driver = {}
    prefix_trailer = {}
    for derived in prefix_derived:
        prefix_driver.setdefault(derived.shift.driver, []).append(
            (derived.shift.start, derived.end)
        )
        prefix_trailer.setdefault(derived.shift.trailer, []).append(
            (derived.shift.start, derived.end)
        )
        
    filtered = []
    for i, candidate in enumerate(candidates):
        start, end = intervals[i]
        driver = instance.drivers[candidate.driver]
        conflicts = False
        
        for prefix_start, prefix_end in prefix_driver.get(candidate.driver, []):
            left_end = end + driver.min_inter_shift_duration
            right_end = prefix_end + driver.min_inter_shift_duration
            if _intervals_overlap(start, left_end, prefix_start, right_end):
                conflicts = True
                break
                
        if not conflicts:
            for prefix_start, prefix_end in prefix_trailer.get(candidate.trailer, []):
                if _intervals_overlap(start, end, prefix_start, prefix_end):
                    conflicts = True
                    break
                    
        if not conflicts:
            filtered.append(candidate)
            
    return filtered


def _top_diverse_columns(
    instance: Instance,
    candidates: list[Shift],
    pressure,
    config: ColumnLoopConfig,
) -> list[Shift]:
    route_cache = RouteCache(instance)
    ranked = sorted(
        candidates,
        key=lambda shift: (
            -_candidate_pressure_bonus(instance, shift, pressure),
            -len(_served_customers(instance, shift)),
            route_cache.shift_distance(shift),
        ),
    )
    selected: list[Shift] = []
    seen_routes: set[tuple[int, ...]] = set()
    bucket_counts: dict[tuple[int, int, int], int] = {}
    anchor_counts: dict[int, int] = {}
    route_signature_counts: dict[tuple[tuple[int, ...], int, int], int] = {}
    source_region_counts: dict[int, int] = {}
    for shift in ranked:
        customers = _served_customers(instance, shift)
        if not customers:
            continue
        route_key = (tuple(customers), shift.start // 240, shift.driver, shift.trailer)
        if route_key in seen_routes and config.bucket_route_signature_cap <= 1:
            continue
        bucket = (shift.driver, shift.trailer, shift.start // 240)
        anchor = customers[0]
        source_region = _first_source_region(instance, shift)
        sig_key = (tuple(customers), shift.driver, shift.trailer)
        route_signature_counts[sig_key] = route_signature_counts.get(sig_key, 0)
        if bucket_counts.get(bucket, 0) >= config.bucket_resource_time_cap:
            continue
        if anchor_counts.get(anchor, 0) >= config.bucket_anchor_cap:
            continue
        if route_signature_counts.get(sig_key, 0) >= config.bucket_route_signature_cap:
            continue
        if source_region_counts.get(source_region, 0) >= config.bucket_source_region_cap:
            continue
        seen_routes.add(route_key)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        anchor_counts[anchor] = anchor_counts.get(anchor, 0) + 1
        route_signature_counts[sig_key] += 1
        source_region_counts[source_region] = source_region_counts.get(source_region, 0) + 1
        selected.append(shift)
        if len(selected) >= config.max_candidates_per_iteration:
            break
    return selected


def _served_customers(instance: Instance, shift: Shift) -> tuple[int, ...]:
    return tuple(
        operation.point
        for operation in shift.operations
        if operation.quantity > 0 and operation.point in instance.customer_by_point
    )


def _served_customers_in_shifts(instance: Instance, shifts: list[Shift]) -> set[int]:
    served: set[int] = set()
    for shift in shifts:
        served.update(_served_customers(instance, shift))
    return served


def _served_customers_in_solution(
    instance: Instance,
    solution: Solution,
    *,
    start_minute: int,
    end_minute: int,
) -> set[int]:
    served: set[int] = set()
    for shift in solution.shifts:
        for operation in shift.operations:
            if (
                start_minute <= operation.arrival < end_minute
                and operation.quantity > 0
                and operation.point in instance.customer_by_point
            ):
                served.add(operation.point)
    return served


def _score_key(score) -> tuple[int, int, int, float]:
    return (
        0 if score.feasible else 1,
        score.hard_violations,
        score.feasibility_errors,
        score.safety_kg_min,
    )


def _selector_phase_for_iteration(config: ColumnLoopConfig, best_score) -> str:
    if config.selector_phase != "auto":
        return config.selector_phase
    return "cost" if best_score.feasible else "feasibility"


def _prior_diagnostics(
    instance: Instance,
    config: ColumnLoopConfig,
    pool: list[Shift] | tuple[Shift, ...],
    selected: tuple[Shift, ...],
) -> RoutePriorDiagnostics:
    prior_by_signature = {
        _prior_match_signature(instance, shift): shift
        for shift in config.route_prior_candidates
    }
    prior_signatures = set(prior_by_signature)
    pool_signatures = {_prior_match_signature(instance, shift) for shift in pool}
    selected_signatures = {_prior_match_signature(instance, shift) for shift in selected}
    inserted_signatures = prior_signatures & pool_signatures
    inserted = len(inserted_signatures)
    selected_count = len(inserted_signatures & selected_signatures)
    rejected_signatures = sorted(
        inserted_signatures - selected_signatures,
        key=lambda signature: (signature[0], signature[1], signature[2], signature[3]),
    )
    pressure_customers = set(_pressure_customers(instance, Solution(shifts=selected), config))
    rejected_pressure_cover = 0
    rejected_parts = []
    for signature in rejected_signatures:
        shift = prior_by_signature[signature]
        customers = _served_customers(instance, shift)
        overlap = tuple(customer for customer in customers if customer in pressure_customers)
        if overlap:
            rejected_pressure_cover += 1
        rejected_parts.append(
            "d{}-t{}-b{}:{}{}".format(
                shift.driver,
                shift.trailer,
                shift.start // 240,
                "/".join(str(customer) for customer in customers[:5]),
                f"*{','.join(str(customer) for customer in overlap[:3])}" if overlap else "",
            )
        )
    return RoutePriorDiagnostics(
        loaded=len(config.route_prior_candidates),
        structurally_valid=len(config.route_prior_candidates),
        inserted=inserted,
        selected=selected_count,
        rejected=max(0, inserted - selected_count),
        skeletons_regenerated=sum(1 for source in config.route_prior_sources if source == "historical_prior_skeleton"),
        rejected_pressure_cover=rejected_pressure_cover,
        rejected_route_summary=";".join(rejected_parts[:12]),
    )


def _mip_start_indices(
    instance: Instance,
    pool: list[Shift],
    baseline_window: list[Shift],
    priors: tuple[Shift, ...],
) -> tuple[int, ...]:
    warm_signatures = {
        route_signature(instance, shift)
        for shift in [*baseline_window, *priors]
    }
    return tuple(
        index
        for index, shift in enumerate(pool)
        if route_signature(instance, shift) in warm_signatures
    )


def _prior_match_signature(instance: Instance, shift: Shift) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    return (
        shift.driver,
        shift.trailer,
        tuple(operation.point for operation in shift.operations),
        _served_customers(instance, shift),
    )


def _priority_prior_indices(
    instance: Instance,
    pool: list[Shift],
    priors: tuple[Shift, ...],
    pressure_customers: list[int],
) -> tuple[int, ...]:
    pressure_set = set(pressure_customers)
    if not pressure_set:
        return ()
    priority_signatures = {
        route_signature(instance, shift)
        for shift in priors
        if pressure_set.intersection(_served_customers(instance, shift))
    }
    return tuple(
        index
        for index, shift in enumerate(pool)
        if route_signature(instance, shift) in priority_signatures
    )


def _fixed_prior_shift_indices(
    instance: Instance,
    solution: Solution,
    priors: tuple[Shift, ...],
) -> set[int]:
    prior_signatures = {
        _prior_match_signature(instance, shift)
        for shift in priors
    }
    return {
        shift.index
        for shift in solution.shifts
        if _prior_match_signature(instance, shift) in prior_signatures
    }


def _prior_incumbent_solution(
    instance: Instance,
    fixed_prefix: Solution,
    config: ColumnLoopConfig,
) -> Solution | None:
    exact_priors = _exact_route_priors(config)
    if not exact_priors:
        return None
    start = config.replace_from_day * MINUTES_PER_DAY
    end = config.end_day * MINUTES_PER_DAY
    window_priors = [
        shift
        for shift in exact_priors
        if start <= shift.start < end
    ]
    if not window_priors:
        return None
    return Solution(
        shifts=tuple(
            replace(shift, index=index)
            for index, shift in enumerate(
                sorted(
                    [*fixed_prefix.shifts, *window_priors],
                    key=lambda shift: (shift.start, shift.driver, shift.trailer),
                )
            )
        )
    )


def _exact_route_priors(config: ColumnLoopConfig) -> tuple[Shift, ...]:
    if not config.route_prior_sources:
        return config.route_prior_candidates
    return tuple(
        shift
        for shift, source in zip(config.route_prior_candidates, config.route_prior_sources)
        if source == "historical_prior"
    )


def _candidate_cache_key(
    instance: Instance,
    prefix: Solution,
    pressure_customers: list[int],
    config: ColumnLoopConfig,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(instance.name).encode())
    digest.update(repr(tuple((s.driver, s.trailer, s.start, tuple((op.point, op.arrival, round(op.quantity, 3)) for op in s.operations)) for s in prefix.shifts)).encode())
    digest.update(repr(tuple(pressure_customers)).encode())
    digest.update(
        repr(
            (
                config.start_day,
                config.end_day,
                config.replace_from_day,
                config.samples_per_customer,
                config.sample_lookback_days,
                config.max_chain_length,
                config.nearest_chain_neighbors,
                config.target_fill_ratio,
                config.max_pre_service_fill_ratio,
                config.multi_reload_columns,
                config.max_multi_reload_per_batch,
            )
        ).encode()
    )
    for customer in instance.customers:
        digest.update(repr((customer.index, customer.forecast[: config.end_day * 24])).encode())
    return digest.hexdigest()[:24]


def _first_source_region(instance: Instance, shift: Shift) -> int:
    for operation in shift.operations:
        if operation.point in instance.source_by_point:
            return operation.point
    if not _served_customers(instance, shift):
        return -1
    first_customer = _served_customers(instance, shift)[0]
    return min(instance.sources, key=lambda source: instance.time_matrix[source.index][first_customer]).index if instance.sources else -1

_BUDGET_EPSILON = 1e-6


def _safe_delivery_budgets(
    instance: Instance,
    prefix: Solution,
    baseline_window: list,
    replace_from_day: int,
    end_day: int,
) -> dict:
    """Compute the maximum *extra* quantity a new candidate can deliver to each
    customer without causing tank overfill at any inventory checkpoint, given
    the deliveries already committed by the baseline window shifts.

    Returns: customer_index -> max_extra_quantity (float, possibly inf)
    """
    events = list(tank_events(instance, prefix))
    events_by_cust_step = {(e.point, e.step): e for e in events}
    checkpoint_steps = _inventory_checkpoint_steps(
        instance, events, replace_from_day, end_day
    )

    # Collect baseline window deliveries: customer -> [(arrival_step, qty)]
    baseline_by_cust: dict = {}
    for shift in baseline_window:
        for op in shift.operations:
            if op.quantity <= _BUDGET_EPSILON:
                continue
            cust = instance.customer_by_point.get(op.point)
            if cust is None or cust.call_in:
                continue
            arrival_step = min(
                max(op.arrival // instance.unit, 0), instance.horizon - 1
            )
            baseline_by_cust.setdefault(cust.index, []).append(
                (arrival_step, op.quantity)
            )

    budgets: dict = {}
    for customer in instance.customers:
        if customer.call_in:
            continue
        steps = checkpoint_steps.get(customer.index, ())
        if not steps:
            budgets[customer.index] = float("inf")
            continue

        baseline_deliveries = baseline_by_cust.get(customer.index, [])
        min_budget = float("inf")
        for step in steps:
            event = events_by_cust_step.get((customer.index, step))
            if event is None:
                continue
            prefix_inv = event.ending_inventory
            rhs_upper = customer.capacity - prefix_inv  # max cumulative window delivery
            baseline_cum = sum(
                qty for arr_step, qty in baseline_deliveries if arr_step <= step
            )
            budget = rhs_upper - baseline_cum
            if budget < min_budget:
                min_budget = budget

        budgets[customer.index] = (
            max(0.0, min_budget) if min_budget < float("inf") else float("inf")
        )

    return budgets


def _apply_delivery_budgets(
    instance: Instance,
    candidates: list,
    budgets: dict,
) -> list:
    """Cap delivery quantities in *candidates* so that no customer's budget is
    exceeded.  Candidates whose meaningful deliveries all collapse to zero are
    dropped entirely.
    """
    result: list = []
    for shift in candidates:
        new_ops: list = []
        changed = False
        for op in shift.operations:
            if op.quantity <= _BUDGET_EPSILON:
                new_ops.append(op)
                continue
            cust = instance.customer_by_point.get(op.point)
            if cust is None:
                new_ops.append(op)
                continue
            budget = budgets.get(cust.index, float("inf"))
            if budget < op.quantity - _BUDGET_EPSILON:
                capped = max(0.0, budget)
                # Drop sub-minimum deliveries rather than leave a token amount
                if capped < cust.min_operation_quantity - _BUDGET_EPSILON:
                    capped = 0.0
                new_ops.append(replace(op, quantity=capped))
                changed = True
            else:
                new_ops.append(op)

        if not changed:
            result.append(shift)
            continue

        # Keep the candidate only when at least one meaningful delivery remains
        has_delivery = any(
            op.quantity > _BUDGET_EPSILON
            and op.point in instance.customer_by_point
            for op in new_ops
        )
        if has_delivery:
            result.append(replace(shift, operations=tuple(new_ops)))

    return result
