"""Rolling commit-and-advance loop with Monte Carlo hedged instances.

Decomposes a long-horizon IRP into overlapping short-horizon CG solves.
Each window uses a hedged instance built from consumption scenarios, then
commits near-term routes and advances.  The final stitched solution is
validated against the true (un-noised) instance.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal
from typing import Callable

from ..contest import ContestScore, score_prefix_with_feasibility_tail, truncate_solution
from ..model import Instance, Shift, Solution
from .column_loop import ColumnLoopConfig, column_generation_rescue
from .scenario import (
    ForecastDistribution,
    build_hedged_instance_from_distribution,
    build_scenario_instance_from_distribution,
    generate_scenarios,
    scenarios_from_distribution,
)

MINUTES_PER_DAY = 1440
RollingMode = Literal["deterministic", "hedged", "robust"]


@dataclass(frozen=True)
class RollingCGConfig:
    """Configuration for the robust rolling CG solver."""

    mode: RollingMode = "hedged"
    horizon_days: int = 30
    commit_days: int = 7
    lookahead_days: int = 14
    n_scenarios: int = 20
    scenario_seed: int = 42
    plan_sigma: float = 0.15
    buffer_sigma: float = 0.30
    commit_percentile: float = 50.0
    plan_percentile: float = 75.0
    buffer_percentile: float = 90.0
    # CG config for each window
    cg_iterations: int = 5
    max_pressure_customers: int = 12
    samples_per_customer: int = 8
    max_chain_length: int = 4
    nearest_chain_neighbors: int = 10
    max_candidates_per_iteration: int = 1200
    target_fill_ratio: float = 0.95
    multi_reload_columns: bool = False
    max_pre_service_fill_ratio: float = 0.95
    normalize_source_loads: bool = True
    quantity_objective: str = "max-delivered"
    capacity_buffer: float = 0.05
    max_hedge_retries: int = 2
    percentile_retry_step: float = 5.0
    capacity_buffer_retry_step: float = 0.03
    cg_iteration_retry_step: int = 2
    forecast_distribution: ForecastDistribution | None = None
    max_rounds: int | None = None
    committed_output_only: bool = False
    final_clip_capacity: bool = True
    progress_log_path: str | None = None
    route_prior_candidates: tuple[Shift, ...] = ()
    route_prior_sources: tuple[str, ...] = ()
    selector_time_limit: float = 300.0
    selector_mip_gap: float | None = None
    selector_threads: int | None = None
    selector_mip_focus: int | None = None
    selector_node_limit: int | None = None
    selector_phase: Literal["auto", "feasibility", "cost"] = "auto"
    candidate_cache_dir: str | None = None
    bucket_anchor_cap: int = 80
    bucket_resource_time_cap: int = 12
    bucket_route_signature_cap: int = 1
    bucket_source_region_cap: int = 240
    protect_exact_prior_quantities: bool = False
    use_prior_incumbent: bool = True


@dataclass(frozen=True)
class RollingCGStep:
    """Diagnostics for one rolling window."""

    round_index: int
    commit_start_day: int
    commit_end_day: int
    solve_end_day: int
    cg_iterations: int
    feasible: bool
    feasibility_errors: int
    hard_violations: int
    first_safety_breach_minute: int | None
    committed_shifts: int
    total_shifts: int
    scenario_feasible: bool
    scenario_failures: int
    retry_count: int
    accepted: bool = True
    rejection_reason: str = ""
    prior_loaded: int = 0
    prior_structurally_valid: int = 0
    prior_inserted: int = 0
    prior_selected: int = 0
    prior_rejected: int = 0
    prior_skeletons_regenerated: int = 0
    prior_rejected_pressure_cover: int = 0
    prior_rejected_route_summary: str = ""


@dataclass(frozen=True)
class WindowEvaluation:
    solution: Solution
    true_score: ContestScore
    scenario_feasible: bool
    scenario_failures: int
    scenario_failure_rate: float
    retry_count: int
    cg_iterations: int
    accepted: bool
    rejection_reason: str

    @property
    def key(self) -> tuple[int, int, int, int, float]:
        return (
            0 if self.true_score.feasible else 1,
            self.scenario_failures,
            self.true_score.hard_violations,
            self.true_score.feasibility_errors,
            self.true_score.scored_estimated_cost,
        )


ProgressCallback = Callable[[str], None]


def robust_rolling_rescue(
    instance: Instance,
    baseline: Solution,
    *,
    config: RollingCGConfig = RollingCGConfig(),
    progress: ProgressCallback | None = None,
) -> tuple[Solution, list[RollingCGStep]]:
    """Solve by rolling commit-and-advance with hedged consumption.

    Each round:
    1. Generate K consumption scenarios with noise growing from commit boundary
    2. Build hedged instance (percentile blend)
    3. Run CG on hedged instance for this window
    4. Commit near-term routes, advance window

    The final solution is stitched from all committed blocks and validated
    against the true (un-noised) instance.
    """
    emit = progress or (lambda _msg: None)
    progress_log = _ProgressLog(config.progress_log_path)
    instance_days = _instance_days(instance)
    horizon_days = min(config.horizon_days, instance_days)
    commit_stride = config.commit_days
    steps: list[RollingCGStep] = []

    current_solution = baseline
    committed_day = 0
    round_index = 0

    while committed_day < horizon_days and (
        config.max_rounds is None or round_index < config.max_rounds
    ):
        commit_end_day = min(committed_day + commit_stride, horizon_days)
        solve_end_day = min(committed_day + commit_stride + config.lookahead_days, horizon_days)
        incumbent_commit_score = score_prefix_with_feasibility_tail(
            instance,
            current_solution,
            score_days=commit_end_day,
            feasibility_days=commit_end_day,
        )

        emit(
            f"round={round_index} commit_days=[{committed_day},{commit_end_day}) "
            f"solve_days=[{committed_day},{solve_end_day})"
        )

        # Is this the last round? Use true data (no noise needed)
        is_last_round = commit_end_day >= horizon_days

        attempts = config.max_hedge_retries + 1
        best_attempt: WindowEvaluation | None = None

        for attempt in range(attempts):
            plan_percentile = min(
                99.0,
                config.plan_percentile + attempt * config.percentile_retry_step,
            )
            buffer_percentile = min(
                99.0,
                config.buffer_percentile + attempt * config.percentile_retry_step,
            )
            capacity_buffer = min(
                0.30,
                config.capacity_buffer + attempt * config.capacity_buffer_retry_step,
            )
            cg_iterations = config.cg_iterations + attempt * config.cg_iteration_retry_step

            distribution = (
                config.forecast_distribution
                if config.forecast_distribution is not None
                else ForecastDistribution.from_instance(instance)
            )
            if config.mode == "deterministic" or is_last_round:
                hedged = instance
                emit("  using true data (no stochastic hedge)")
            else:
                plan_sigma, buffer_sigma = _mode_sigmas(config)
                plan_percentile, buffer_percentile, capacity_buffer = _mode_hedge_values(
                    config,
                    attempt,
                    plan_percentile,
                    buffer_percentile,
                    capacity_buffer,
                )
                if config.forecast_distribution is None:
                    sigma_schedule = {"plan": plan_sigma, "buffer": buffer_sigma}
                    scenarios = generate_scenarios(
                        instance,
                        n_scenarios=config.n_scenarios,
                        seed=config.scenario_seed + round_index,
                        commit_end_day=commit_end_day,
                        day_sigma_schedule=sigma_schedule,
                    )
                    distribution = ForecastDistribution.from_samples(instance, scenarios)
                    source_label = f"generated {config.n_scenarios} scenarios"
                else:
                    source_label = "loaded external forecast distribution"
                hedged = build_hedged_instance_from_distribution(
                    instance,
                    distribution,
                    commit_end_day=commit_end_day,
                    plan_end_day=min(commit_end_day + commit_stride, horizon_days),
                    commit_percentile=config.commit_percentile,
                    plan_percentile=plan_percentile,
                    buffer_percentile=buffer_percentile,
                    capacity_buffer=capacity_buffer,
                )
                emit(
                    f"  attempt={attempt} {source_label}, "
                    f"plan_σ={plan_sigma}, buffer_σ={buffer_sigma}, "
                    f"p={plan_percentile:.1f}/{buffer_percentile:.1f}, "
                    f"capacity_buffer={capacity_buffer:.3f}"
                )

            window_prior_pairs = [
                (index, shift)
                for index, shift in enumerate(config.route_prior_candidates)
                if committed_day * MINUTES_PER_DAY <= shift.start < solve_end_day * MINUTES_PER_DAY
            ]
            cg_config = ColumnLoopConfig(
                start_day=0,
                end_day=solve_end_day,
                replace_from_day=committed_day,
                iterations=cg_iterations,
                max_pressure_customers=config.max_pressure_customers,
                samples_per_customer=config.samples_per_customer,
                sample_lookback_days=max(7, committed_day),
                max_chain_length=config.max_chain_length,
                nearest_chain_neighbors=config.nearest_chain_neighbors,
                max_candidates_per_iteration=config.max_candidates_per_iteration,
                target_fill_ratio=config.target_fill_ratio,
                multi_reload_columns=config.multi_reload_columns,
                max_pre_service_fill_ratio=config.max_pre_service_fill_ratio,
                normalize_source_loads=config.normalize_source_loads,
                quantity_objective=config.quantity_objective,
                commit_end_day=commit_end_day,
                next_after_commit_day=min(commit_end_day + commit_stride, solve_end_day),
                route_prior_candidates=tuple(
                    shift for _index, shift in window_prior_pairs
                ),
                route_prior_sources=tuple(
                    config.route_prior_sources[index]
                    if index < len(config.route_prior_sources)
                    else "historical_prior"
                    for index, _shift in window_prior_pairs
                ),
                selector_time_limit=config.selector_time_limit,
                selector_mip_gap=config.selector_mip_gap,
                selector_threads=config.selector_threads,
                selector_mip_focus=config.selector_mip_focus,
                selector_node_limit=config.selector_node_limit,
                selector_phase=config.selector_phase,
                candidate_cache_dir=config.candidate_cache_dir,
                bucket_anchor_cap=config.bucket_anchor_cap,
                bucket_resource_time_cap=config.bucket_resource_time_cap,
                bucket_route_signature_cap=config.bucket_route_signature_cap,
                bucket_source_region_cap=config.bucket_source_region_cap,
                protect_exact_prior_quantities=config.protect_exact_prior_quantities,
                use_prior_incumbent=config.use_prior_incumbent,
            )

            window_solution, cg_steps = column_generation_rescue(
                hedged, current_solution, config=cg_config
            )

            has_reached_feasibility = False
            best_iteration_key: tuple[int, int, int, float] | None = None
            for step in cg_steps:
                feasibility_msg = ""
                if step.feasible:
                    if not has_reached_feasibility:
                        feasibility_msg = " [MILESTONE: FEASIBILITY ACHIEVED]"
                        has_reached_feasibility = True
                        progress_log.write_step(
                            "milestone_feasible",
                            round_index,
                            attempt,
                            committed_day,
                            commit_end_day,
                            solve_end_day,
                            step,
                        )
                    else:
                        feasibility_msg = " [Feasible]"

                iteration_key = (
                    0 if step.feasible else 1,
                    step.hard_violations,
                    step.feasibility_errors,
                    step.cost,
                )
                improved = best_iteration_key is None or iteration_key < best_iteration_key
                if improved:
                    best_iteration_key = iteration_key
                
                v_commit_str = ", ".join(f"C{c}:{d:.2f}d" for c, d in step.vulnerable_commit_customers) if step.vulnerable_commit_customers else "None"
                v_next_str = ", ".join(f"C{c}:{d:.2f}d" for c, d in step.vulnerable_next_after_commit_customers) if step.vulnerable_next_after_commit_customers else "None"
                v_lookahead_str = ", ".join(f"C{c}:{d:.2f}d" for c, d in step.vulnerable_lookahead_customers) if step.vulnerable_lookahead_customers else "None"
                
                danger_status_commit = _danger_status(step.min_commit_doi)
                danger_status_next = _danger_status(step.min_next_after_commit_doi)
                danger_status_lookahead = _danger_status(step.min_lookahead_doi)
                improvement_msg = " [IMPROVED]" if improved else ""
                
                emit(
                    f"  CG Iter {step.iteration:<2} | Pool: {step.pool_size:<4} | Shifts: {step.selected_extra_shifts:<2} | "
                    f"Errors: {step.feasibility_errors:<3} | Hard: {step.hard_violations:<2}{feasibility_msg}{improvement_msg}\n"
                    f"    ↳ Priors: loaded={step.prior_loaded} inserted={step.prior_inserted} "
                    f"selected={step.prior_selected} rejected={step.prior_rejected} "
                    f"rejected_pressure={step.prior_rejected_pressure_cover} skeletons={step.prior_skeletons_regenerated}\n"
                    f"    ↳ Rejected prior routes: {step.prior_rejected_route_summary or 'None'}\n"
                    f"    ↳ KPI: Cost = {step.cost:.2f} | LogRatio = {step.logistic_ratio:.6f}\n"
                    f"    ↳ Commit (Day {commit_end_day}) Safety: min DOI = {step.min_commit_doi:.2f}d ({danger_status_commit}) | Top Vulnerable: {v_commit_str}\n"
                    f"    ↳ Next (Day {step.next_after_commit_day}) Safety: min DOI = {step.min_next_after_commit_doi:.2f}d ({danger_status_next}) | Top Vulnerable: {v_next_str}\n"
                    f"    ↳ Lookahead (Day {solve_end_day}) Safety: min DOI = {step.min_lookahead_doi:.2f}d ({danger_status_lookahead}) | Top Vulnerable: {v_lookahead_str}"
                )
                progress_log.write_step(
                    "iteration_improved" if improved else "iteration",
                    round_index,
                    attempt,
                    committed_day,
                    commit_end_day,
                    solve_end_day,
                    step,
                )

            lookahead_score = score_prefix_with_feasibility_tail(
                instance,
                window_solution,
                score_days=solve_end_day,
                feasibility_days=solve_end_day,
            )
            true_score = score_prefix_with_feasibility_tail(
                instance,
                window_solution,
                score_days=commit_end_day,
                feasibility_days=commit_end_day,
            )
            scenario_feasible, scenario_failures = _validate_committed_scenarios(
                instance,
                window_solution,
                distribution,
                commit_end_day,
            )
            scenario_count = distribution.scenario_count()
            scenario_failure_rate = (
                scenario_failures / scenario_count if scenario_count else 0.0
            )
            accepted, rejection_reason = _accept_window(
                incumbent_commit_score,
                true_score,
                scenario_failures,
            )
            emit(
                f"  commit score: errors={true_score.feasibility_errors} "
                f"hard={true_score.hard_violations} feasible={true_score.feasible}; "
                f"lookahead errors={lookahead_score.feasibility_errors} "
                f"hard={lookahead_score.hard_violations}; "
                f"scenario_feasible={scenario_feasible} failures={scenario_failures}; "
                f"accepted={accepted} {rejection_reason}"
            )
            progress_log.write_attempt(
                "attempt_accepted" if accepted else "attempt_rejected",
                round_index,
                attempt,
                committed_day,
                commit_end_day,
                solve_end_day,
                true_score,
                lookahead_score,
                scenario_failures,
                accepted,
                rejection_reason,
            )

            evaluation = WindowEvaluation(
                solution=window_solution,
                true_score=true_score,
                scenario_feasible=scenario_feasible,
                scenario_failures=scenario_failures,
                scenario_failure_rate=scenario_failure_rate,
                retry_count=attempt,
                cg_iterations=len(cg_steps),
                accepted=accepted,
                rejection_reason=rejection_reason,
            )
            if accepted and (best_attempt is None or evaluation.key < best_attempt.key):
                best_attempt = evaluation
            if accepted and true_score.feasible and scenario_feasible:
                break

        accepted_window = best_attempt is not None
        if best_attempt is None:
            true_score = incumbent_commit_score
            window_solution = current_solution
            scenario_feasible = True
            scenario_failures = 0
            retry_count = attempts - 1
            cg_iterations = 0
            rejection_reason = "kept incumbent: all candidates rejected"
            emit(f"  {rejection_reason}")
        else:
            window_solution = best_attempt.solution
            true_score = best_attempt.true_score
            scenario_feasible = best_attempt.scenario_feasible
            scenario_failures = best_attempt.scenario_failures
            retry_count = best_attempt.retry_count
            cg_iterations = best_attempt.cg_iterations
            rejection_reason = best_attempt.rejection_reason

        # Commit: keep all shifts starting before commit boundary
        commit_cutoff = commit_end_day * MINUTES_PER_DAY
        committed_shifts = [
            shift for shift in window_solution.shifts
            if shift.start < commit_cutoff
        ]
        committed_count = len([
            s for s in committed_shifts
            if s.start >= committed_day * MINUTES_PER_DAY
        ])

        # For the next round, start with the full window solution as baseline.
        # This preserves the 'plan' part of the solution so the next round
        # isn't starting from an empty schedule for its lookahead window.
        current_solution = window_solution

        steps.append(
            RollingCGStep(
                round_index=round_index,
                commit_start_day=committed_day,
                commit_end_day=commit_end_day,
                solve_end_day=solve_end_day,
                cg_iterations=len(cg_steps),
                feasible=true_score.feasible,
                feasibility_errors=true_score.feasibility_errors,
                hard_violations=true_score.hard_violations,
                first_safety_breach_minute=true_score.first_safety_breach_minute,
                committed_shifts=committed_count,
                total_shifts=len(committed_shifts),
                scenario_feasible=scenario_feasible,
                scenario_failures=scenario_failures,
                retry_count=retry_count,
                accepted=accepted_window,
                rejection_reason=rejection_reason,
                prior_loaded=sum(step.prior_loaded for step in cg_steps[-1:]),
                prior_structurally_valid=sum(step.prior_structurally_valid for step in cg_steps[-1:]),
                prior_inserted=sum(step.prior_inserted for step in cg_steps[-1:]),
                prior_selected=sum(step.prior_selected for step in cg_steps[-1:]),
                prior_rejected=sum(step.prior_rejected for step in cg_steps[-1:]),
                prior_skeletons_regenerated=sum(step.prior_skeletons_regenerated for step in cg_steps[-1:]),
                prior_rejected_pressure_cover=sum(step.prior_rejected_pressure_cover for step in cg_steps[-1:]),
                prior_rejected_route_summary=";".join(
                    step.prior_rejected_route_summary
                    for step in cg_steps[-1:]
                    if step.prior_rejected_route_summary
                ),
            )
        )

        committed_day = commit_end_day
        round_index += 1

    # Final validation against true instance over full horizon
    output_solution = (
        truncate_solution(current_solution, committed_day * MINUTES_PER_DAY)
        if config.committed_output_only
        else current_solution
    )

    final_score = score_prefix_with_feasibility_tail(
        instance,
        output_solution,
        score_days=max(1, committed_day if config.committed_output_only else horizon_days),
        feasibility_days=max(1, committed_day if config.committed_output_only else horizon_days),
    )
    emit(
        f"FINAL: shifts={len(output_solution.shifts)} "
        f"errors={final_score.feasibility_errors} "
        f"hard={final_score.hard_violations} "
        f"feasible={final_score.feasible} "
        f"cost={final_score.scored_estimated_cost:.2f}"
    )
    progress_log.write_final(horizon_days, committed_day, final_score, len(output_solution.shifts))
    progress_log.close()

    # Reindex shifts and apply final physics clipping to prevent any residual overfills
    indexed_solution = Solution(
        shifts=tuple(
            replace(shift, index=i)
            for i, shift in enumerate(
                sorted(output_solution.shifts, key=lambda s: (s.start, s.driver))
            )
        )
    )
    if not config.final_clip_capacity:
        return indexed_solution, steps

    clipped_solution = clip_to_tank_capacity(instance, indexed_solution)
    indexed_score = score_prefix_with_feasibility_tail(
        instance,
        indexed_solution,
        score_days=max(1, committed_day if config.committed_output_only else horizon_days),
        feasibility_days=max(1, committed_day if config.committed_output_only else horizon_days),
    )
    clipped_score = score_prefix_with_feasibility_tail(
        instance,
        clipped_solution,
        score_days=max(1, committed_day if config.committed_output_only else horizon_days),
        feasibility_days=max(1, committed_day if config.committed_output_only else horizon_days),
    )
    return (
        clipped_solution
        if _score_key(clipped_score) <= _score_key(indexed_score)
        else indexed_solution
    ), steps


class _ProgressLog:
    fieldnames = (
        "event",
        "round",
        "attempt",
        "commit_start_day",
        "commit_end_day",
        "solve_end_day",
        "iteration",
        "generated_candidates",
        "pool_size",
        "selected_extra_shifts",
        "feasible",
        "errors",
        "hard",
        "first_safety_breach_minute",
        "cost",
        "logistic_ratio",
        "scenario_failures",
        "accepted",
        "rejection_reason",
        "commit_min_doi",
        "commit_vulnerable",
        "next_after_commit_day",
        "next_after_commit_min_doi",
        "next_after_commit_vulnerable",
        "lookahead_min_doi",
        "lookahead_vulnerable",
        "output_shifts",
        "prior_loaded",
        "prior_structurally_valid",
        "prior_inserted",
        "prior_selected",
        "prior_rejected",
        "prior_skeletons_regenerated",
        "prior_rejected_pressure_cover",
        "prior_rejected_route_summary",
    )

    def __init__(self, path: str | None):
        self._handle = None
        self._writer = None
        if path is None:
            return
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = log_path.open("w", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.fieldnames)
        self._writer.writeheader()

    def write_step(
        self,
        event: str,
        round_index: int,
        attempt: int,
        commit_start_day: int,
        commit_end_day: int,
        solve_end_day: int,
        step,
    ) -> None:
        if self._writer is None:
            return
        self._write(
            {
                "event": event,
                "round": round_index,
                "attempt": attempt,
                "commit_start_day": commit_start_day,
                "commit_end_day": commit_end_day,
                "solve_end_day": solve_end_day,
                "iteration": step.iteration,
                "generated_candidates": step.generated_candidates,
                "pool_size": step.pool_size,
                "selected_extra_shifts": step.selected_extra_shifts,
                "feasible": step.feasible,
                "errors": step.feasibility_errors,
                "hard": step.hard_violations,
                "first_safety_breach_minute": step.first_safety_breach_minute,
                "cost": step.cost,
                "logistic_ratio": step.logistic_ratio,
                "commit_min_doi": step.min_commit_doi,
                "commit_vulnerable": _format_vulnerable(step.vulnerable_commit_customers),
                "next_after_commit_day": step.next_after_commit_day,
                "next_after_commit_min_doi": step.min_next_after_commit_doi,
                "next_after_commit_vulnerable": _format_vulnerable(step.vulnerable_next_after_commit_customers),
                "lookahead_min_doi": step.min_lookahead_doi,
                "lookahead_vulnerable": _format_vulnerable(step.vulnerable_lookahead_customers),
                "prior_loaded": step.prior_loaded,
                "prior_structurally_valid": step.prior_structurally_valid,
                "prior_inserted": step.prior_inserted,
                "prior_selected": step.prior_selected,
                "prior_rejected": step.prior_rejected,
                "prior_skeletons_regenerated": step.prior_skeletons_regenerated,
                "prior_rejected_pressure_cover": step.prior_rejected_pressure_cover,
                "prior_rejected_route_summary": step.prior_rejected_route_summary,
            }
        )

    def write_attempt(
        self,
        event: str,
        round_index: int,
        attempt: int,
        commit_start_day: int,
        commit_end_day: int,
        solve_end_day: int,
        true_score: ContestScore,
        lookahead_score: ContestScore,
        scenario_failures: int,
        accepted: bool,
        rejection_reason: str,
    ) -> None:
        if self._writer is None:
            return
        self._write(
            {
                "event": event,
                "round": round_index,
                "attempt": attempt,
                "commit_start_day": commit_start_day,
                "commit_end_day": commit_end_day,
                "solve_end_day": solve_end_day,
                "feasible": true_score.feasible,
                "errors": true_score.feasibility_errors,
                "hard": true_score.hard_violations,
                "first_safety_breach_minute": true_score.first_safety_breach_minute,
                "cost": true_score.scored_estimated_cost,
                "logistic_ratio": true_score.scored_estimated_cost
                / max(1.0, true_score.scored_delivered_quantity),
                "scenario_failures": scenario_failures,
                "accepted": accepted,
                "rejection_reason": rejection_reason,
                "lookahead_min_doi": "",
                "lookahead_vulnerable": (
                    f"errors={lookahead_score.feasibility_errors};"
                    f"hard={lookahead_score.hard_violations}"
                ),
            }
        )

    def write_final(
        self,
        horizon_days: int,
        committed_day: int,
        score: ContestScore,
        output_shifts: int,
    ) -> None:
        if self._writer is None:
            return
        self._write(
            {
                "event": "final",
                "commit_end_day": committed_day,
                "solve_end_day": horizon_days,
                "feasible": score.feasible,
                "errors": score.feasibility_errors,
                "hard": score.hard_violations,
                "first_safety_breach_minute": score.first_safety_breach_minute,
                "cost": score.scored_estimated_cost,
                "logistic_ratio": score.scored_estimated_cost
                / max(1.0, score.scored_delivered_quantity),
                "output_shifts": output_shifts,
            }
        )

    def _write(self, row: dict[str, object]) -> None:
        if self._writer is None:
            return
        self._writer.writerow({field: row.get(field, "") for field in self.fieldnames})
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()


def _format_vulnerable(customers: list[tuple[int, float]] | None) -> str:
    if not customers:
        return ""
    return ";".join(f"{customer}:{doi:.3f}" for customer, doi in customers)


def _danger_status(min_doi: float) -> str:
    if min_doi < 0.5:
        return "CRITICAL"
    if min_doi < 1.5:
        return "TIGHT"
    return "SAFE"


def _validate_committed_scenarios(
    instance: Instance,
    solution: Solution,
    distribution: ForecastDistribution | None,
    commit_end_day: int,
) -> tuple[bool, int]:
    if distribution is None:
        return True, 0
    scenarios = scenarios_from_distribution(distribution)
    scenario_count = max((len(paths) for paths in scenarios.values()), default=0)
    if scenario_count == 0:
        return True, 0
    failures = 0
    for scenario_index in range(scenario_count):
        scenario_instance = build_scenario_instance_from_distribution(
            instance,
            distribution,
            scenario_index,
        )
        score = score_prefix_with_feasibility_tail(
            scenario_instance,
            solution,
            score_days=commit_end_day,
            feasibility_days=commit_end_day,
        )
        if not score.feasible:
            failures += 1
    return failures == 0, failures


def _accept_window(
    incumbent_score: ContestScore,
    candidate_score: ContestScore,
    scenario_failures: int,
) -> tuple[bool, str]:
    if incumbent_score.feasible and not candidate_score.feasible:
        return False, "rejected: infeasible true prefix would replace feasible incumbent"
    if candidate_score.hard_violations > incumbent_score.hard_violations and not candidate_score.feasible:
        return False, "rejected: hard violations worse than incumbent"
    if scenario_failures > 0 and incumbent_score.feasible:
        return False, "rejected: scenario validation failed for feasible incumbent"
    return True, ""


def _score_key(score: ContestScore) -> tuple[int, int, int, float]:
    return (
        0 if score.feasible else 1,
        score.hard_violations,
        score.feasibility_errors,
        score.scored_estimated_cost,
    )


def _mode_sigmas(config: RollingCGConfig) -> tuple[float, float]:
    if config.mode == "robust":
        return max(config.plan_sigma, 0.20), max(config.buffer_sigma, 0.40)
    return config.plan_sigma, config.buffer_sigma


def _mode_hedge_values(
    config: RollingCGConfig,
    attempt: int,
    plan_percentile: float,
    buffer_percentile: float,
    capacity_buffer: float,
) -> tuple[float, float, float]:
    if config.mode != "robust":
        return plan_percentile, buffer_percentile, capacity_buffer
    return (
        max(plan_percentile, min(99.0, 90.0 + attempt * config.percentile_retry_step)),
        max(buffer_percentile, min(99.0, 95.0 + attempt * config.percentile_retry_step)),
        max(capacity_buffer, min(0.30, 0.10 + attempt * config.capacity_buffer_retry_step)),
    )


def clip_to_tank_capacity(instance: Instance, solution: Solution) -> Solution:
    """Ensure no delivery exceeds available tank space or violates min-quantity.

    If a planned delivery would overfill, it is truncated to the available space.
    If the truncated quantity is below the customer's min-delivery requirement,
    the delivery is canceled (set to 0).
    """
    from ..inventory import tank_events

    events = list(tank_events(instance, solution))
    # Map (shift_index, op_index) -> max_allowed_quantity
    allowed: dict[tuple[int, int], float] = {}

    for customer in instance.customers:
        inventory = customer.initial_tank_quantity
        # Get all deliveries for this customer, sorted by time
        cust_deliveries = []
        for s_idx, s in enumerate(solution.shifts):
            for op_idx, op in enumerate(s.operations):
                if op.point == customer.index and op.quantity > 0:
                    cust_deliveries.append((op.arrival, s_idx, op_idx, op.quantity))
        
        cust_deliveries.sort()
        
        last_time = 0
        for arrival, s_idx, op_idx, qty in cust_deliveries:
            # 1. Consume until arrival
            start_step = last_time // instance.unit
            end_step = arrival // instance.unit
            for step in range(start_step, min(end_step, instance.horizon)):
                inventory -= customer.forecast[step]
            
            inventory = max(0.0, inventory)
            
            # 2. Check space
            space = max(0.0, customer.capacity - inventory)
            clipped = min(qty, space)
            
            # 3. Min op guard
            if clipped < customer.min_operation_quantity - 1e-6:
                clipped = 0.0
                
            allowed[(s_idx, op_idx)] = clipped
            inventory += clipped
            last_time = arrival

    new_shifts = []
    for shift in solution.shifts:
        new_ops = []
        for op_idx, op in enumerate(shift.operations):
            if (shift.index, op_idx) in allowed:
                new_ops.append(replace(op, quantity=allowed[(shift.index, op_idx)]))
            else:
                new_ops.append(op)
        positive_quantity = sum(op.quantity for op in new_ops if op.quantity > 0)
        remaining_load = positive_quantity
        balanced_ops = []
        for op in new_ops:
            if op.quantity < 0:
                load = min(-op.quantity, remaining_load)
                remaining_load -= load
                if load > 1e-6:
                    balanced_ops.append(replace(op, quantity=-load))
            elif abs(op.quantity) > 1e-6:
                balanced_ops.append(op)
        if balanced_ops:
            new_shifts.append(replace(shift, operations=tuple(balanced_ops)))

    return Solution(shifts=tuple(new_shifts))


def _instance_days(instance: Instance) -> int:
    horizon_minutes = instance.horizon * instance.unit
    return (horizon_minutes + MINUTES_PER_DAY - 1) // MINUTES_PER_DAY
