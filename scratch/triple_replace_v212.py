from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import combinations
from pathlib import Path

from vrp_solver.contest import score_prefix_with_feasibility_tail
from vrp_solver.inventory import tank_violations
from vrp_solver.model import Shift, Solution
from vrp_solver.rules import derive_solution
from vrp_solver.solver.column_loop import ColumnLoopConfig, _rescue_config
from vrp_solver.solver.targeted_rescue import (
    generate_chain_rescue_candidates,
    generate_carryover_rescue_candidates,
    generate_multi_reload_candidates,
    generate_rescue_candidates,
    normalize_source_loads,
)
from vrp_solver.xml_io import load_instance, load_solution, save_solution


ROOT = Path(__file__).resolve().parents[1]
INSTANCE = ROOT / "roadef_2016_data/set_B/Instances_B_V25-11042016/V2.12.xml"
START = ROOT / "scratch/b_v2_replace_neighborhood/V2.12/best.xml"
OUT = ROOT / "scratch/b_v2_triple_replace/V2.12"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    instance = load_instance(INSTANCE)
    current = load_solution(START)
    current_score = score(instance, current)
    failures = failing_counts(instance, current)
    failing = set(failures)
    print("start", current_score.feasibility_errors, current_score.hard_violations, failures)

    candidates = generate_candidates(instance, current, list(failures))
    candidates = [candidate for candidate in candidates if served_points(candidate) & failing]
    candidates.sort(
        key=lambda shift: (
            -sum(failures.get(point, 0) for point in served_points(shift)),
            conflict_count(instance, current, shift),
            shift.start,
            shift.driver,
            shift.trailer,
        )
    )
    top = candidates[:18]
    print("candidates", len(candidates), "top", len(top))

    best = current
    best_score = current_score
    checked = 0
    for combo in combinations(top, 3):
        checked += 1
        if checked % 50 == 0:
            print("checked", checked, "best", best_score.feasibility_errors, best_score.hard_violations)
        pairwise_ok = True
        for left, right in combinations(combo, 2):
            if conflicts(instance, left, right, interval_for(instance, right)):
                pairwise_ok = False
                break
        if not pairwise_ok:
            continue
        served = set().union(*(served_points(candidate) for candidate in combo)) & failing
        if len(served) < 3:
            continue
        trial = replace_many(instance, current, combo)
        if trial is None:
            continue
        trial_score = score(instance, trial)
        if trial_score.hard_violations == 0 and trial_score.feasibility_errors < best_score.feasibility_errors:
            best = trial
            best_score = trial_score
            out = OUT / f"best_{best_score.feasibility_errors}.xml"
            save_solution(best, out)
            print("accept", best_score.feasibility_errors, sorted(served), out)

    final = OUT / "best.xml"
    save_solution(best, final)
    print("final", best_score.feasibility_errors, best_score.hard_violations, final)


def generate_candidates(instance, solution: Solution, points: list[int]) -> list[Shift]:
    config = ColumnLoopConfig(
        start_day=0,
        end_day=10,
        replace_from_day=0,
        samples_per_customer=12,
        max_chain_length=5,
        nearest_chain_neighbors=14,
        multi_reload_columns=True,
        max_multi_reload_per_batch=50,
    )
    rescue_config = _rescue_config(config)
    candidates: list[Shift] = []
    for point in points[:8]:
        batch = [point, *[other for other in points[:12] if other != point][:5]]
        candidates.extend(generate_rescue_candidates(instance, solution, batch, config=rescue_config))
        candidates.extend(generate_carryover_rescue_candidates(instance, solution, batch, config=rescue_config))
        candidates.extend(generate_chain_rescue_candidates(instance, solution, batch, config=rescue_config))
        candidates.extend(generate_multi_reload_candidates(instance, solution, batch, config=rescue_config)[:50])
    unique: dict[tuple, Shift] = {}
    for candidate in candidates:
        unique.setdefault(signature(candidate), candidate)
    return list(unique.values())


def score(instance, solution: Solution):
    return score_prefix_with_feasibility_tail(
        instance,
        solution,
        score_days=10,
        feasibility_days=10,
        ignore_tail_call_ins=True,
    )


def failing_counts(instance, solution: Solution) -> dict[int, int]:
    return dict(Counter(violation.point for violation in tank_violations(instance, solution)).most_common())


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
    shifts = sorted([*kept, *(candidate for candidate, _ in candidate_items)], key=lambda shift: (shift.start, shift.index))
    return normalize_source_loads(
        instance,
        Solution(shifts=tuple(replace(shift, index=index) for index, shift in enumerate(shifts))),
    )


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
    return {operation.point for operation in shift.operations if operation.quantity > 0}


def signature(shift: Shift) -> tuple:
    return (
        shift.driver,
        shift.trailer,
        shift.start,
        tuple((operation.point, operation.arrival, round(operation.quantity, 3)) for operation in shift.operations),
    )


if __name__ == "__main__":
    main()
