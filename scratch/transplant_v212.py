from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from vrp_solver.contest import score_prefix_with_feasibility_tail
from vrp_solver.highs_repair import repair_quantities_with_highs
from vrp_solver.model import Shift, Solution
from vrp_solver.rules import derive_solution
from vrp_solver.solver.targeted_rescue import normalize_source_loads
from vrp_solver.xml_io import load_instance, load_solution, save_solution


ROOT = Path(__file__).resolve().parents[1]
INSTANCE = ROOT / "roadef_2016_data/set_B/Instances_B_V25-11042016/V2.12.xml"
START = ROOT / "scratch/b_v2_build_neighborhood_8/V2.12/V2.12_cluster_seed.xml"
SOURCES = [
    ROOT / "scratch/b_v2_build_diag_layer_check/V2.12/V2.12_cluster_seed.xml",
    ROOT / "scratch/b_v2_build_terminal_buffer_1/V2.12/V2.12_cluster_seed.xml",
    ROOT / "scratch/b_v2_build_terminal_buffer_2/V2.12/V2.12_cluster_seed.xml",
    ROOT / "scratch/b_v2_build_neighborhood_7/V2.12/V2.12_cluster_seed.xml",
    ROOT / "scratch/b_v2_build_neighborhood_9/V2.12/V2.12_cluster_seed.xml",
    ROOT / "scratch/b_v2_build_neighborhood_12/V2.12/V2.12_cluster_seed.xml",
]
OUT = ROOT / "scratch/b_v2_transplant/V2.12"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    instance = load_instance(INSTANCE)
    current = load_solution(START)
    current_score = score(instance, current)
    print("start", current_score.feasibility_errors, current_score.hard_violations)

    candidates = candidate_shifts(instance)
    print("candidates", len(candidates))
    seen_attempts: set[tuple] = set()
    improved = True
    passes = 0
    while improved and passes < 8:
        improved = False
        passes += 1
        failing = failing_points(instance, current)
        ranked = sorted(
            candidates,
            key=lambda shift: (
                -len(served_points(shift) & failing),
                shift.start,
                shift.driver,
                shift.trailer,
            ),
        )
        for candidate in ranked:
            if not (served_points(candidate) & failing):
                continue
            key = (passes, signature(candidate), current_score.feasibility_errors)
            if key in seen_attempts:
                continue
            seen_attempts.add(key)
            trial = transplant(instance, current, candidate)
            if trial is None:
                continue
            trial_score = score(instance, trial)
            if (
                trial_score.hard_violations == 0
                and trial_score.feasibility_errors < current_score.feasibility_errors
            ):
                current = trial
                current_score = trial_score
                save_solution(current, OUT / f"best_p{passes}_{current_score.feasibility_errors}.xml")
                print(
                    "accept",
                    "pass",
                    passes,
                    "errors",
                    current_score.feasibility_errors,
                    "hard",
                    current_score.hard_violations,
                    "shift",
                    signature(candidate),
                )
                improved = True
                break

    repaired, _ = repair_quantities_with_highs(
        instance,
        current,
        score_days=10,
        feasibility_days=10,
        quantity_objective="max-delivered",
    )
    repaired = normalize_source_loads(instance, repaired)
    repaired_score = score(instance, repaired)
    if repaired_score.hard_violations == 0 and repaired_score.feasibility_errors <= current_score.feasibility_errors:
        current = repaired
        current_score = repaired_score
    final_path = OUT / "best.xml"
    save_solution(current, final_path)
    print("final", current_score.feasibility_errors, current_score.hard_violations, final_path)


def score(instance, solution):
    return score_prefix_with_feasibility_tail(
        instance,
        solution,
        score_days=10,
        feasibility_days=10,
        ignore_tail_call_ins=True,
    )


def candidate_shifts(instance) -> list[Shift]:
    by_sig: dict[tuple, Shift] = {}
    for path in SOURCES:
        if not path.exists():
            continue
        solution = load_solution(path)
        for shift in solution.shifts:
            if not served_points(shift):
                continue
            by_sig.setdefault(signature(shift), shift)
    return list(by_sig.values())


def failing_points(instance, solution) -> set[int]:
    from vrp_solver.inventory import tank_violations

    violations = tank_violations(instance, solution)
    return {violation.point for violation in violations}


def served_points(shift: Shift) -> set[int]:
    return {
        operation.point
        for operation in shift.operations
        if operation.quantity > 0
    }


def signature(shift: Shift) -> tuple:
    return (
        shift.driver,
        shift.trailer,
        shift.start,
        tuple((op.point, op.arrival, round(op.quantity, 3)) for op in shift.operations),
    )


def transplant(instance, solution: Solution, candidate: Shift) -> Solution | None:
    candidate = replace(candidate, index=999_999)
    candidate_interval = interval_for(instance, candidate)
    kept = []
    for shift in solution.shifts:
        if conflicts(instance, shift, candidate, candidate_interval):
            continue
        kept.append(shift)
    if len(kept) == len(solution.shifts):
        return None
    shifts = sorted([*kept, candidate], key=lambda shift: (shift.start, shift.index))
    reindexed = tuple(replace(shift, index=index) for index, shift in enumerate(shifts))
    return normalize_source_loads(instance, Solution(shifts=reindexed))


def interval_for(instance, shift: Shift) -> tuple[int, int]:
    derived = derive_solution(instance, Solution(shifts=(replace(shift, index=0),)))[0]
    return derived.shift.start, derived.end


def conflicts(
    instance,
    existing: Shift,
    candidate: Shift,
    candidate_interval: tuple[int, int],
) -> bool:
    if existing.driver != candidate.driver and existing.trailer != candidate.trailer:
        return False
    existing_start, existing_end = interval_for(instance, existing)
    candidate_start, candidate_end = candidate_interval
    if existing.driver == candidate.driver:
        driver = instance.drivers[existing.driver]
        existing_end += driver.min_inter_shift_duration
        candidate_end += driver.min_inter_shift_duration
    return existing_start < candidate_end and candidate_start < existing_end


if __name__ == "__main__":
    main()
