from __future__ import annotations

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
from dataclasses import replace
from collections import Counter


ROOT = Path(__file__).resolve().parents[1]
INSTANCE = ROOT / "roadef_2016_data/set_B/Instances_B_V25-11042016/V2.12.xml"
START = ROOT / "scratch/b_v2_replace_neighborhood/V2.12/best.xml"
OUT = ROOT / "scratch/b_v2_replace_top_points/V2.12"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    instance = load_instance(INSTANCE)
    current = load_solution(START)
    current_score = score(instance, current)
    print("start", current_score.feasibility_errors, current_score.hard_violations)
    points = [point for point, _ in Counter(v.point for v in tank_violations(instance, current)).most_common()]
    config = ColumnLoopConfig(
        start_day=0,
        end_day=10,
        replace_from_day=0,
        max_pressure_customers=12,
        samples_per_customer=10,
        max_chain_length=5,
        nearest_chain_neighbors=12,
        multi_reload_columns=True,
        max_multi_reload_per_batch=40,
    )
    candidates = []
    rescue_config = _rescue_config(config)
    for point in points[:6]:
        batch = [point, *[other for other in points[:10] if other != point][:4]]
        candidates.extend(generate_rescue_candidates(instance, current, batch, config=rescue_config))
        candidates.extend(generate_carryover_rescue_candidates(instance, current, batch, config=rescue_config))
        candidates.extend(generate_chain_rescue_candidates(instance, current, batch, config=rescue_config))
        candidates.extend(generate_multi_reload_candidates(instance, current, batch, config=rescue_config)[:40])
    unique = {}
    for candidate in candidates:
        unique.setdefault(signature(candidate), candidate)
    candidates = list(unique.values())
    candidates.sort(key=lambda shift: (-len(served_points(shift) & set(points)), conflict_count(instance, current, shift), shift.start))
    print("points", points)
    print("candidates", len(candidates))
    best = current
    best_score = current_score
    for index, candidate in enumerate(candidates[:500]):
        if index % 50 == 0:
            print("try", index, "best", best_score.feasibility_errors, best_score.hard_violations)
        trial = replace_one(instance, best, candidate)
        if trial is None:
            continue
        trial_score = score(instance, trial)
        if trial_score.hard_violations == 0 and trial_score.feasibility_errors < best_score.feasibility_errors:
            best = trial
            best_score = trial_score
            out = OUT / f"best_{best_score.feasibility_errors}.xml"
            save_solution(best, out)
            print("accept", best_score.feasibility_errors, sorted(served_points(candidate) & set(points)), out)
    pair = best_pair(instance, best, candidates, set(points), best_score.feasibility_errors)
    if pair is not None:
        best, best_score, pair_served = pair
        out = OUT / f"best_pair_{best_score.feasibility_errors}.xml"
        save_solution(best, out)
        print("accept_pair", best_score.feasibility_errors, sorted(pair_served), out)
    final = OUT / "best.xml"
    save_solution(best, final)
    print("final", best_score.feasibility_errors, best_score.hard_violations, final)


def score(instance, solution):
    return score_prefix_with_feasibility_tail(instance, solution, score_days=10, feasibility_days=10, ignore_tail_call_ins=True)


def replace_one(instance, solution: Solution, candidate: Shift) -> Solution | None:
    candidate = replace(candidate, index=999_999)
    candidate_interval = interval_for(instance, candidate)
    kept = []
    removed = 0
    for shift in solution.shifts:
        if conflicts(instance, shift, candidate, candidate_interval):
            removed += 1
            continue
        kept.append(shift)
    if removed == 0:
        return None
    shifts = sorted([*kept, candidate], key=lambda shift: (shift.start, shift.index))
    return normalize_source_loads(instance, Solution(shifts=tuple(replace(shift, index=i) for i, shift in enumerate(shifts))))


def best_pair(instance, solution, candidates, failing, current_errors):
    top = [candidate for candidate in candidates if served_points(candidate) & failing][:24]
    best = None
    best_errors = current_errors
    for left_index, left in enumerate(top):
        print("pair_left", left_index, "of", len(top), "best", best_errors)
        for right in top[left_index + 1:]:
            pair_served = (served_points(left) | served_points(right)) & failing
            if len(pair_served) < 2:
                continue
            if conflicts(instance, left, right, interval_for(instance, right)):
                continue
            trial = replace_many(instance, solution, (left, right))
            if trial is None:
                continue
            trial_score = score(instance, trial)
            if trial_score.hard_violations == 0 and trial_score.feasibility_errors < best_errors:
                best = (trial, trial_score, pair_served)
                best_errors = trial_score.feasibility_errors
    return best


def replace_many(instance, solution, candidates):
    candidate_items = [
        (replace(candidate, index=999_000 + index), interval_for(instance, candidate))
        for index, candidate in enumerate(candidates)
    ]
    kept = []
    removed = 0
    for shift in solution.shifts:
        if any(conflicts(instance, shift, candidate, interval) for candidate, interval in candidate_items):
            removed += 1
            continue
        kept.append(shift)
    if removed == 0:
        return None
    shifts = sorted([*kept, *(candidate for candidate, _ in candidate_items)], key=lambda shift: (shift.start, shift.index))
    return normalize_source_loads(instance, Solution(shifts=tuple(replace(shift, index=i) for i, shift in enumerate(shifts))))


def conflict_count(instance, solution, candidate):
    interval = interval_for(instance, candidate)
    return sum(1 for shift in solution.shifts if conflicts(instance, shift, candidate, interval))


def conflicts(instance, existing, candidate, candidate_interval):
    if existing.driver != candidate.driver and existing.trailer != candidate.trailer:
        return False
    existing_start, existing_end = interval_for(instance, existing)
    candidate_start, candidate_end = candidate_interval
    if existing.driver == candidate.driver:
        rest = instance.drivers[existing.driver].min_inter_shift_duration
        existing_end += rest
        candidate_end += rest
    return existing_start < candidate_end and candidate_start < existing_end


def interval_for(instance, shift):
    return (lambda derived: (derived.shift.start, derived.end))(derive_solution(instance, Solution(shifts=(replace(shift, index=0),)))[0])


def served_points(shift):
    return {operation.point for operation in shift.operations if operation.quantity > 0}


def signature(shift):
    return (shift.driver, shift.trailer, shift.start, tuple((op.point, op.arrival, round(op.quantity, 3)) for op in shift.operations))


if __name__ == "__main__":
    main()
