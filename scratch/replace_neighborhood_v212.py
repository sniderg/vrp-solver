from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

from vrp_solver.contest import score_prefix_with_feasibility_tail
from vrp_solver.inventory import tank_violations
from vrp_solver.model import Shift, Solution
from vrp_solver.rules import derive_solution
from vrp_solver.solver.column_loop import (
    ColumnLoopConfig,
    _apply_delivery_budgets,
    _cached_generate_priced_batches,
    _filter_prefix_conflicts,
    _pressure_customers,
    _rescue_config,
    _safe_delivery_budgets,
    _top_diverse_columns,
)
from vrp_solver.solver.highs_selector import _inventory_pressure_by_customer
from vrp_solver.solver.targeted_rescue import (
    _baseline_window_shifts,
    _keep_shifts_started_before,
    normalize_source_loads,
)
from vrp_solver.xml_io import load_instance, load_solution, save_solution


ROOT = Path(__file__).resolve().parents[1]
INSTANCE = ROOT / "roadef_2016_data/set_B/Instances_B_V25-11042016/V2.12.xml"
START = ROOT / "scratch/b_v2_build_neighborhood_8/V2.12/V2.12_cluster_seed.xml"
OUT = ROOT / "scratch/b_v2_replace_neighborhood/V2.12"
RESUME = OUT / "best.xml"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    instance = load_instance(INSTANCE)
    current = load_solution(RESUME if RESUME.exists() else START)
    current_score = score(instance, current)
    print("start", current_score.feasibility_errors, current_score.hard_violations)
    save_solution(current, OUT / "start.xml")

    accepted = 0
    for pass_index in range(8):
        failing = failing_points(instance, current)
        if not failing:
            break
        candidates = rescue_candidates(instance, current, pass_index)
        ranked = sorted(
            candidates,
            key=lambda shift: (
                -len(served_points(shift) & failing),
                conflict_count(instance, current, shift),
                shift.start,
                shift.driver,
                shift.trailer,
            ),
        )
        print(
            "pass",
            pass_index,
            "errors",
            current_score.feasibility_errors,
            "hard",
            current_score.hard_violations,
            "failing",
            len(failing),
            "candidates",
            len(ranked),
            "top_fail",
            top_failing_points(instance, current),
        )
        improved = False
        for candidate in ranked:
            if not (served_points(candidate) & failing):
                continue
            trial = replace_one(instance, current, candidate)
            if trial is None:
                continue
            trial_score = score(instance, trial)
            if trial_score.hard_violations != 0:
                continue
            if trial_score.feasibility_errors < current_score.feasibility_errors:
                accepted += 1
                current = trial
                current_score = trial_score
                out = OUT / f"accepted_{accepted:03d}_{current_score.feasibility_errors}.xml"
                save_solution(current, out)
                print(
                    "accept",
                    accepted,
                    "errors",
                    current_score.feasibility_errors,
                    "hard",
                    current_score.hard_violations,
                    "served",
                    sorted(served_points(candidate) & failing),
                    "removed",
                    conflict_count(instance, current, candidate),
                    out,
                )
                improved = True
                break
        if not improved:
            pair = best_pair_replacement(instance, current, ranked, failing, current_score.feasibility_errors)
            if pair is None:
                print("no_replacement_improvement", pass_index)
                break
            current, current_score, pair_served = pair
            accepted += 1
            out = OUT / f"accepted_{accepted:03d}_{current_score.feasibility_errors}.xml"
            save_solution(current, out)
            print(
                "accept_pair",
                accepted,
                "errors",
                current_score.feasibility_errors,
                "hard",
                current_score.hard_violations,
                "served",
                sorted(pair_served),
                out,
            )

    final_path = OUT / "best.xml"
    save_solution(current, final_path)
    final_score = score(instance, current)
    print("final", final_score.feasibility_errors, final_score.hard_violations, final_score.feasible, final_path)


def score(instance, solution: Solution):
    return score_prefix_with_feasibility_tail(
        instance,
        solution,
        score_days=10,
        feasibility_days=10,
        ignore_tail_call_ins=True,
    )


def rescue_candidates(instance, solution: Solution, pass_index: int) -> list[Shift]:
    pressure_cap = 8 + 2 * min(pass_index, 2)
    config = ColumnLoopConfig(
        start_day=0,
        end_day=10,
        replace_from_day=0,
        iterations=1,
        max_pressure_customers=pressure_cap,
        samples_per_customer=3,
        max_chain_length=3,
        nearest_chain_neighbors=6,
        max_candidates_per_iteration=240,
        multi_reload_columns=True,
        selector_time_limit=60.0,
        selector_phase="feasibility",
        commit_end_day=10,
    )
    fixed_prefix = _keep_shifts_started_before(solution, 0)
    baseline_window = list(_baseline_window_shifts(solution, _rescue_config(config)))
    pressure_customers = _pressure_customers(instance, solution, config)
    generated = _cached_generate_priced_batches(instance, fixed_prefix, pressure_customers, config)
    generated = _filter_prefix_conflicts(instance, fixed_prefix, generated)
    pressure = _inventory_pressure_by_customer(instance, fixed_prefix, config.replace_from_day, config.end_day)
    generated = _top_diverse_columns(instance, generated, pressure, config)
    budgets = _safe_delivery_budgets(
        instance,
        fixed_prefix,
        baseline_window,
        config.replace_from_day,
        config.end_day,
    )
    return _apply_delivery_budgets(instance, generated, budgets)


def replace_one(instance, solution: Solution, candidate: Shift) -> Solution | None:
    candidate = replace(candidate, index=999_999)
    candidate_interval = interval_for(instance, candidate)
    kept: list[Shift] = []
    removed = 0
    for shift in solution.shifts:
        if conflicts(instance, shift, candidate, candidate_interval):
            removed += 1
            continue
        kept.append(shift)
    if removed == 0:
        return None
    shifts = sorted([*kept, candidate], key=lambda shift: (shift.start, shift.index))
    reindexed = tuple(replace(shift, index=index) for index, shift in enumerate(shifts))
    return normalize_source_loads(instance, Solution(shifts=reindexed))


def best_pair_replacement(
    instance,
    solution: Solution,
    candidates: list[Shift],
    failing: set[int],
    current_errors: int,
) -> tuple[Solution, object, set[int]] | None:
    top = [
        candidate
        for candidate in candidates
        if served_points(candidate) & failing
    ][:20]
    best: tuple[Solution, object, set[int]] | None = None
    best_errors = current_errors
    for left_index, left in enumerate(top):
        print("pair_left", left_index, "of", len(top))
        left_served = served_points(left) & failing
        for right in top[left_index + 1:]:
            pair_served = left_served | (served_points(right) & failing)
            if len(pair_served) < 2:
                continue
            if conflicts(instance, left, right, interval_for(instance, right)):
                continue
            trial = replace_many(instance, solution, (left, right))
            if trial is None:
                continue
            trial_score = score(instance, trial)
            if trial_score.hard_violations != 0:
                continue
            if trial_score.feasibility_errors < best_errors:
                best = (trial, trial_score, pair_served)
                best_errors = trial_score.feasibility_errors
    return best


def replace_many(instance, solution: Solution, candidates: tuple[Shift, ...]) -> Solution | None:
    candidate_items = [
        (replace(candidate, index=999_000 + index), interval_for(instance, candidate))
        for index, candidate in enumerate(candidates)
    ]
    kept: list[Shift] = []
    removed = 0
    for shift in solution.shifts:
        if any(conflicts(instance, shift, candidate, interval) for candidate, interval in candidate_items):
            removed += 1
            continue
        kept.append(shift)
    if removed == 0:
        return None
    shifts = sorted(
        [*kept, *(candidate for candidate, _ in candidate_items)],
        key=lambda shift: (shift.start, shift.index),
    )
    reindexed = tuple(replace(shift, index=index) for index, shift in enumerate(shifts))
    return normalize_source_loads(instance, Solution(shifts=reindexed))


def conflict_count(instance, solution: Solution, candidate: Shift) -> int:
    interval = interval_for(instance, candidate)
    return sum(1 for shift in solution.shifts if conflicts(instance, shift, candidate, interval))


def conflicts(instance, existing: Shift, candidate: Shift, candidate_interval: tuple[int, int]) -> bool:
    if existing.driver != candidate.driver and existing.trailer != candidate.trailer:
        return False
    existing_start, existing_end = interval_for(instance, existing)
    candidate_start, candidate_end = candidate_interval
    if existing.driver == candidate.driver:
        rest = instance.drivers[existing.driver].min_inter_shift_duration
        existing_end += rest
        candidate_end += rest
    return existing_start < candidate_end and candidate_start < existing_end


def interval_for(instance, shift: Shift) -> tuple[int, int]:
    derived = derive_solution(instance, Solution(shifts=(replace(shift, index=0),)))[0]
    return derived.shift.start, derived.end


def served_points(shift: Shift) -> set[int]:
    return {
        operation.point
        for operation in shift.operations
        if operation.quantity > 0
    }


def failing_points(instance, solution: Solution) -> set[int]:
    return {violation.point for violation in tank_violations(instance, solution)}


def top_failing_points(instance, solution: Solution) -> list[tuple[int, int]]:
    counts = Counter(violation.point for violation in tank_violations(instance, solution))
    return counts.most_common(8)


if __name__ == "__main__":
    main()
