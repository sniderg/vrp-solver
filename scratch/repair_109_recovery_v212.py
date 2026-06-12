from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import combinations
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scratch.direct_topup_v212 import (
    conflict_count,
    conflicts,
    direct_topup_candidates,
    interval_for,
    pairwise_compatible,
    score,
    served_points,
)
from vrp_solver.inventory import tank_violations
from vrp_solver.model import Shift, Solution
from vrp_solver.solver.targeted_rescue import normalize_source_loads
from vrp_solver.xml_io import load_instance, load_solution, save_solution

INSTANCE = ROOT / "roadef_2016_data/set_B/Instances_B_V25-11042016/V2.12.xml"
START = ROOT / "scratch/b_v2_replace_neighborhood/V2.12/best.xml"
OUT = ROOT / "scratch/b_v2_repair_109_recovery/V2.12"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    instance = load_instance(INSTANCE)
    incumbent = load_solution(START)
    base_score = score(instance, incumbent)
    print("start", base_score.feasibility_errors, base_score.hard_violations)

    forced = early_109_candidates(instance, incumbent)
    print("forced", len(forced))
    best = incumbent
    best_score = base_score
    forced_trials = []
    for forced_shift in forced:
        trial, removed_points = force_shift(instance, incumbent, forced_shift)
        if trial is None:
            continue
        trial_score = score(instance, trial)
        forced_trials.append((trial_score.hard_violations, trial_score.feasibility_errors, forced_shift, trial, removed_points))
    forced_trials.sort(key=lambda item: (item[0], item[1]))
    for forced_index, (_, _, forced_shift, base_trial, removed_points) in enumerate(forced_trials[:8]):
        trial_score = score(instance, base_trial)
        print(
            "forced_try",
            forced_index,
            "score",
            trial_score.feasibility_errors,
            trial_score.hard_violations,
            "removed_points",
            sorted(removed_points),
        )
        recovery_points = sorted(set(removed_points) | set(failing_points(instance, base_trial)))
        recovery_candidates = []
        for point in recovery_points[:18]:
            recovery_candidates.extend(direct_topup_candidates(instance, base_trial, [point]))
        unique = {}
        for candidate in recovery_candidates:
            if served_points(candidate) & set(recovery_points):
                unique.setdefault(signature(candidate), candidate)
        recovery_candidates = list(unique.values())
        recovery_candidates.sort(
            key=lambda shift: (
                -sum(failing_counts(instance, base_trial).get(point, 0) for point in served_points(shift)),
                conflict_count(instance, base_trial, shift),
                shift.start,
            )
        )
        print(" recovery_candidates", len(recovery_candidates))
        candidate_best = base_trial
        candidate_best_score = trial_score
        for size in (1, 2):
            top = recovery_candidates[:10]
            checked = 0
            for combo in combinations(top, size):
                checked += 1
                if checked % 200 == 0:
                    print("  combo", size, checked, "candidate_best", candidate_best_score.feasibility_errors, candidate_best_score.hard_violations)
                if not pairwise_compatible(instance, combo):
                    continue
                recovered = add_recovery(instance, base_trial, combo)
                recovered_score = score(instance, recovered)
                if (
                    recovered_score.hard_violations == 0
                    and recovered_score.feasibility_errors < candidate_best_score.feasibility_errors
                ):
                    candidate_best = recovered
                    candidate_best_score = recovered_score
                    out = OUT / f"forced{forced_index}_combo{size}_{candidate_best_score.feasibility_errors}.xml"
                    save_solution(candidate_best, out)
                    print("  accept_recovery", candidate_best_score.feasibility_errors, sorted(set().union(*(served_points(c) for c in combo))), out)
        if candidate_best_score.hard_violations == 0 and candidate_best_score.feasibility_errors < best_score.feasibility_errors:
            best = candidate_best
            best_score = candidate_best_score
            out = OUT / f"best_{best_score.feasibility_errors}.xml"
            save_solution(best, out)
            print("accept_forced", best_score.feasibility_errors, out)

    final = OUT / "best.xml"
    save_solution(best, final)
    print("final", best_score.feasibility_errors, best_score.hard_violations, final)


def early_109_candidates(instance, solution):
    candidates = [
        candidate
        for candidate in direct_topup_candidates(instance, solution, [109])
        if any(op.point == 109 and op.arrival <= 4020 for op in candidate.operations)
    ]
    candidates.sort(key=lambda shift: (abs(next(op.arrival for op in shift.operations if op.point == 109) - 3690), conflict_count(instance, solution, shift), shift.driver))
    return candidates


def force_shift(instance, solution: Solution, forced: Shift) -> tuple[Solution | None, set[int]]:
    forced = replace(forced, index=999_999)
    interval = interval_for(instance, forced)
    kept = []
    removed_points: set[int] = set()
    removed = 0
    for shift in solution.shifts:
        if conflicts(instance, shift, forced, interval):
            removed += 1
            removed_points.update(served_points(shift))
            continue
        kept.append(shift)
    if removed == 0:
        return None, set()
    shifts = sorted([*kept, forced], key=lambda shift: (shift.start, shift.index))
    reindexed = tuple(replace(shift, index=index) for index, shift in enumerate(shifts))
    return normalize_source_loads(instance, Solution(shifts=reindexed)), removed_points


def add_recovery(instance, solution: Solution, candidates: tuple[Shift, ...]) -> Solution:
    candidate_items = [(replace(candidate, index=999_000 + index), interval_for(instance, candidate)) for index, candidate in enumerate(candidates)]
    kept = []
    for shift in solution.shifts:
        if any(conflicts(instance, shift, candidate, interval) for candidate, interval in candidate_items):
            continue
        kept.append(shift)
    shifts = sorted([*kept, *(candidate for candidate, _ in candidate_items)], key=lambda shift: (shift.start, shift.index))
    reindexed = tuple(replace(shift, index=index) for index, shift in enumerate(shifts))
    return normalize_source_loads(instance, Solution(shifts=reindexed))


def failing_counts(instance, solution):
    return Counter(v.point for v in tank_violations(instance, solution))


def failing_points(instance, solution):
    return set(failing_counts(instance, solution))


def signature(shift):
    return (
        shift.driver,
        shift.trailer,
        shift.start,
        tuple((op.point, op.arrival, round(float(op.quantity), 3)) for op in shift.operations),
    )


if __name__ == "__main__":
    main()
