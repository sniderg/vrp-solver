from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from ..contest import ContestScore, score_prefix_with_feasibility_tail
from ..model import Instance, Shift, Solution
from .candidate_gen import GeneratorConfig, generate_shift_candidates
from .highs_selector import select_shifts_with_highs
from ..highs_repair import repair_quantities_with_highs
from ..highs_time_opt import optimize_solution_times


MINUTES_PER_DAY = 1440


@dataclass(frozen=True)
class RollingHighsConfig:
    start_day: int = 0
    end_day: int = 14
    lookahead_days: int = 2
    commit_days: int = 1
    candidates_per_window: int = 12
    neighborhood_size: int = 15
    feasibility_tail_days: int = 1
    candidate_source: str = "generated"
    variable_quantities: bool = True


@dataclass(frozen=True)
class RollingHighsStep:
    day: int
    window_end_day: int
    commit_end_day: int
    generated_candidates: int
    committed_shifts: int
    score: ContestScore | None


ProgressCallback = Callable[[str], None]


def rolling_highs_select(
    instance: Instance,
    *,
    initial_solution: Solution | None = None,
    seed_candidate_solution: Solution | None = None,
    config: RollingHighsConfig = RollingHighsConfig(),
    progress: ProgressCallback | None = None,
) -> tuple[Solution, list[RollingHighsStep]]:
    """Build a solution by repeatedly selecting a short HiGHS lookahead window.

    The selector may choose shifts across the full lookahead window, but only the
    first ``commit_days`` are locked. The next iteration regenerates candidates from
    the newly committed inventory/resource state.
    """

    if config.lookahead_days <= 0:
        raise ValueError("lookahead_days must be positive")
    if config.commit_days <= 0:
        raise ValueError("commit_days must be positive")
    if config.commit_days > config.lookahead_days:
        raise ValueError("commit_days must be less than or equal to lookahead_days")
    if config.end_day <= config.start_day:
        raise ValueError("end_day must be greater than start_day")
    if config.candidate_source not in {"generated", "seed", "both"}:
        raise ValueError("candidate_source must be one of: generated, seed, both")
    if config.candidate_source in {"seed", "both"} and seed_candidate_solution is None:
        raise ValueError("seed_candidate_solution is required for seed candidate source")

    emit = progress or (lambda _message: None)
    accepted = _keep_shifts_started_before(
        initial_solution or Solution(shifts=()),
        config.start_day * MINUTES_PER_DAY,
    )
    steps: list[RollingHighsStep] = []
    day = config.start_day

    instance_horizon_days = (instance.horizon * instance.unit) // MINUTES_PER_DAY
    while day < config.end_day:
        window_end_day = min(day + config.lookahead_days, instance_horizon_days)
        commit_end_day = min(day + config.commit_days, config.end_day)
        emit(
            f"rolling_step,day={day},window_end_day={window_end_day},"
            f"commit_end_day={commit_end_day}"
        )

        candidate_config = GeneratorConfig(
            max_candidates_per_window=config.candidates_per_window,
            neighborhood_size=config.neighborhood_size,
        )
        candidates = _window_candidates(
            instance,
            accepted,
            seed_candidate_solution,
            day=day,
            window_end_day=window_end_day,
            config=config,
            candidate_config=candidate_config,
        )
        emit(f"generated_candidates,{len(candidates)}")

        if not candidates:
            score = _score_commit_tail(instance, accepted, commit_end_day, config)
            steps.append(
                RollingHighsStep(
                    day=day,
                    window_end_day=window_end_day,
                    commit_end_day=commit_end_day,
                    generated_candidates=0,
                    committed_shifts=0,
                    score=score,
                )
            )
            day = commit_end_day
            continue

        selected = select_shifts_with_highs(
            instance,
            accepted,
            candidates,
            start_day=day,
            end_day=window_end_day,
            variable_quantities=config.variable_quantities,
        )
        selected = optimize_solution_times(instance, selected)
        selected, _ = repair_quantities_with_highs(
            instance,
            selected,
            score_days=window_end_day,
            feasibility_days=window_end_day,
            fixed_prefix_minutes=day * MINUTES_PER_DAY,
        )

        previous_shift_count = len(accepted.shifts)
        committed = _reindex_solution(
            _keep_shifts_started_before(selected, commit_end_day * MINUTES_PER_DAY)
        )
        score = _score_commit_tail(instance, committed, commit_end_day, config)

        if score is not None and not score.feasible and window_end_day > commit_end_day:
            emit(
                f"commit_tail_failed,day={day},errors={score.feasibility_errors},"
                "retry=commit_window_only"
            )
            commit_candidates = _window_candidates(
                instance,
                accepted,
                seed_candidate_solution,
                day=day,
                window_end_day=commit_end_day,
                config=config,
                candidate_config=candidate_config,
            )
            emit(f"commit_window_candidates,{len(commit_candidates)}")
            if commit_candidates:
                fallback_end_day = (
                    score.feasibility_days if score is not None else commit_end_day
                )
                commit_selected = select_shifts_with_highs(
                    instance,
                    accepted,
                    commit_candidates,
                    start_day=day,
                    end_day=fallback_end_day,
                    variable_quantities=config.variable_quantities,
                )
                commit_selected = optimize_solution_times(instance, commit_selected)
                commit_repaired, _ = repair_quantities_with_highs(
                    instance,
                    commit_selected,
                    score_days=fallback_end_day,
                    feasibility_days=fallback_end_day,
                    fixed_prefix_minutes=day * MINUTES_PER_DAY,
                )
                commit_solution = _reindex_solution(
                    _keep_shifts_started_before(
                        commit_repaired, commit_end_day * MINUTES_PER_DAY
                    )
                )
                commit_score = _score_commit_tail(
                    instance, commit_solution, commit_end_day, config
                )
                if commit_score is not None and (
                    commit_score.feasible
                    or commit_score.feasibility_errors <= score.feasibility_errors
                ):
                    committed = commit_solution
                    score = commit_score

        accepted = committed
        committed_shifts = max(0, len(accepted.shifts) - previous_shift_count)
        emit(
            f"committed_shifts,{committed_shifts},"
            f"feasible,{score.feasible if score else ''},"
            f"errors,{score.feasibility_errors if score else ''},"
            f"hard,{score.hard_violations if score else ''}"
        )

        steps.append(
            RollingHighsStep(
                day=day,
                window_end_day=window_end_day,
                commit_end_day=commit_end_day,
                generated_candidates=len(candidates),
                committed_shifts=committed_shifts,
                score=score,
            )
        )
        day = commit_end_day

    return _reindex_solution(accepted), steps


def _window_candidates(
    instance: Instance,
    accepted: Solution,
    seed_candidate_solution: Solution | None,
    *,
    day: int,
    window_end_day: int,
    config: RollingHighsConfig,
    candidate_config: GeneratorConfig,
) -> list[Shift]:
    candidates: list[Shift] = []
    if config.candidate_source in {"seed", "both"}:
        candidates.extend(
            _seed_shifts_in_window(
                seed_candidate_solution or Solution(shifts=()),
                start_minute=day * MINUTES_PER_DAY,
                end_minute=window_end_day * MINUTES_PER_DAY,
                existing=accepted,
            )
        )
    if config.candidate_source in {"generated", "both"}:
        candidates.extend(
            generate_shift_candidates(
                instance,
                accepted,
                start_day=day,
                end_day=window_end_day,
                config=candidate_config,
            )
        )
    return _dedupe_reindex(candidates)


def _seed_shifts_in_window(
    seed_solution: Solution,
    *,
    start_minute: int,
    end_minute: int,
    existing: Solution,
) -> list[Shift]:
    existing_keys = {_shift_key(shift) for shift in existing.shifts}
    return [
        shift
        for shift in seed_solution.shifts
        if start_minute <= shift.start < end_minute
        and _shift_key(shift) not in existing_keys
    ]


def _dedupe_reindex(shifts: list[Shift]) -> list[Shift]:
    seen: set[tuple[object, ...]] = set()
    unique: list[Shift] = []
    for shift in shifts:
        key = _shift_key(shift)
        if key in seen:
            continue
        seen.add(key)
        unique.append(replace(shift, index=len(unique)))
    return unique


def _shift_key(shift: Shift) -> tuple[object, ...]:
    return (
        shift.driver,
        shift.trailer,
        shift.start,
        tuple((op.point, op.arrival, round(op.quantity, 9)) for op in shift.operations),
    )


def _score_commit_tail(
    instance: Instance,
    solution: Solution,
    commit_end_day: int,
    config: RollingHighsConfig,
) -> ContestScore | None:
    instance_horizon_days = (instance.horizon * instance.unit) // MINUTES_PER_DAY
    feasibility_days = min(instance_horizon_days, commit_end_day + config.feasibility_tail_days)
    if commit_end_day <= 0 or feasibility_days < commit_end_day:
        return None
    return score_prefix_with_feasibility_tail(
        instance,
        solution,
        score_days=feasibility_days,
        feasibility_days=feasibility_days,
    )


def _keep_shifts_started_before(solution: Solution, cutoff_minute: int) -> Solution:
    return Solution(
        shifts=tuple(shift for shift in solution.shifts if shift.start < cutoff_minute)
    )


def _reindex_solution(solution: Solution) -> Solution:
    return Solution(
        shifts=tuple(replace(shift, index=index) for index, shift in enumerate(solution.shifts))
    )
