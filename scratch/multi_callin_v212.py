from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import combinations, permutations
from pathlib import Path

from vrp_solver.contest import score_prefix_with_feasibility_tail
from vrp_solver.model import Operation, Shift, Solution
from vrp_solver.rules import derive_solution, is_time_window_valid, validate_solution
from vrp_solver.solver.targeted_rescue import normalize_source_loads
from vrp_solver.xml_io import load_instance, load_solution, save_solution


ROOT = Path(__file__).resolve().parents[1]
INSTANCE = ROOT / "roadef_2016_data/set_B/Instances_B_V25-11042016/V2.12.xml"
START = ROOT / "scratch/b_v2_replace_neighborhood/V2.12/best.xml"
OUT = ROOT / "scratch/b_v2_multi_callin/V2.12"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    instance = load_instance(INSTANCE)
    best = load_solution(START)
    best_score = score(instance, best)
    points = callin_errors(instance, best)
    print("start", best_score.feasibility_errors, best_score.hard_violations, points, flush=True)

    candidates = multi_callin_candidates(instance, points)
    candidates.sort(
        key=lambda shift: (
            -len(served_points(shift) & set(points)),
            conflict_count(instance, best, shift),
            shift.start,
            route_duration(instance, shift),
        )
    )
    print("candidates", len(candidates), flush=True)

    seen_scores: set[tuple[int, int, tuple[int, ...]]] = set()
    for index, candidate in enumerate(candidates[:1000]):
        if index % 50 == 0:
            print("try", index, "best", best_score.feasibility_errors, flush=True)
        trial = apply_candidate(instance, best, candidate)
        trial_score = score(instance, trial)
        if trial_score.hard_violations != 0:
            continue
        if structural_errors(instance, trial):
            continue
        key = (
            trial_score.feasibility_errors,
            trial_score.hard_violations,
            tuple(sorted(served_points(candidate))),
        )
        if key in seen_scores:
            continue
        seen_scores.add(key)
        if trial_score.feasibility_errors < best_score.feasibility_errors:
            best = trial
            best_score = trial_score
            out = OUT / f"best_{best_score.feasibility_errors}.xml"
            save_solution(best, out)
            print(
                "accept",
                best_score.feasibility_errors,
                sorted(served_points(candidate)),
                out,
                flush=True,
            )
            if best_score.feasibility_errors == 0:
                break

    final = OUT / "best.xml"
    save_solution(best, final)
    print("final", best_score.feasibility_errors, best_score.hard_violations, final, flush=True)


def multi_callin_candidates(instance, points: list[int]) -> list[Shift]:
    candidates: list[Shift] = []
    seen: set[tuple] = set()
    for size in range(1, min(4, len(points)) + 1):
        before = len(candidates)
        for subset in combinations(points, size):
            min_total = sum(instance.customer_by_point[p].orders[0].min_quantity_to_satisfy for p in subset)
            allowed_trailers = set(instance.customer_by_point[subset[0]].allowed_trailers)
            for point in subset[1:]:
                allowed_trailers &= set(instance.customer_by_point[point].allowed_trailers)
            if not allowed_trailers:
                continue
            for order in permutations(subset):
                for driver in instance.drivers:
                    for trailer_id in driver.trailer_ids:
                        if trailer_id not in allowed_trailers:
                            continue
                        trailer = instance.trailers[trailer_id]
                        if min_total > trailer.capacity:
                            continue
                        for source in instance.sources:
                            if trailer_id not in source.allowed_trailers:
                                continue
                            for start in start_samples(instance, source.index, order, driver.time_windows):
                                shift = build_route(instance, order, driver.index, trailer_id, source.index, start)
                                if shift is None:
                                    continue
                                key = signature(shift)
                                if key in seen:
                                    continue
                                seen.add(key)
                                candidates.append(replace(shift, index=len(candidates)))
        print("generated_size", size, len(candidates) - before, "total", len(candidates), flush=True)
    return candidates


def start_samples(instance, source: int, order: tuple[int, ...], driver_windows) -> list[int]:
    first = instance.customer_by_point[order[0]]
    lead = (
        instance.time_matrix[instance.base_index][source]
        + instance.setup_time_for_point(source)
        + instance.time_matrix[source][first.index]
    )
    raw: set[int] = set()
    for window in driver_windows:
        raw.add(window.start)
        raw.add(first.orders[0].earliest_time - lead)
        raw.add(first.time_windows[0].start - lead)
        raw.add(max(window.start, first.orders[0].latest_time - lead - 120))
        for offset in (0, 120, 360, 720):
            raw.add(first.orders[0].earliest_time - lead + offset)
    return sorted(t for t in raw if any(w.start <= t <= w.end for w in driver_windows))


def build_route(instance, order: tuple[int, ...], driver: int, trailer: int, source: int, start: int) -> Shift | None:
    quantity = sum(instance.customer_by_point[p].orders[0].min_quantity_to_satisfy for p in order)
    operations: list[Operation] = []
    current_time = start
    current_point = instance.base_index
    source_arrival = current_time + instance.time_matrix[current_point][source]
    operations.append(Operation(source, source_arrival, -quantity))
    current_time = source_arrival + instance.setup_time_for_point(source)
    current_point = source
    for point in order:
        customer = instance.customer_by_point[point]
        order_info = customer.orders[0]
        arrival = current_time + instance.time_matrix[current_point][point]
        arrival = max(arrival, order_info.earliest_time)
        window_arrival = None
        for window in customer.time_windows:
            candidate = max(arrival, window.start)
            if (
                candidate + customer.setup_time <= window.end
                and order_info.earliest_time <= candidate
                and candidate + customer.setup_time <= order_info.latest_time
            ):
                window_arrival = candidate
                break
        if window_arrival is None:
            return None
        operations.append(Operation(point, window_arrival, order_info.min_quantity_to_satisfy))
        current_time = window_arrival + customer.setup_time
        current_point = point

    shift = Shift(index=0, driver=driver, trailer=trailer, start=start, operations=tuple(operations))
    derived = derive_solution(instance, Solution((replace(shift, index=0),)))[0]
    driver_info = instance.drivers[driver]
    if derived.layovers:
        return None
    if any(operation.driving_since_layover > driver_info.max_driving_duration for operation in derived.operations):
        return None
    if derived.end > max(window.end for window in driver_info.time_windows):
        return None
    return shift


def apply_candidate(instance, solution: Solution, candidate: Shift) -> Solution:
    candidate = replace(candidate, index=999_999)
    candidate_interval = interval_for(instance, candidate)
    kept = []
    for shift in solution.shifts:
        if conflicts(instance, shift, candidate, candidate_interval):
            continue
        kept.append(shift)
    shifts = sorted([*kept, candidate], key=lambda shift: (shift.start, shift.index))
    return normalize_source_loads(
        instance,
        Solution(shifts=tuple(replace(shift, index=index) for index, shift in enumerate(shifts))),
    )


def callin_errors(instance, solution):
    return [
        violation.point
        for violation in validate_solution(instance, solution)
        if violation.severity == "error" and violation.code == "QS01" and violation.point is not None
    ]


def structural_errors(instance, solution):
    return [
        violation
        for violation in validate_solution(instance, solution)
        if violation.severity == "error" and violation.code not in {"QS01", "QS02"}
    ]


def score(instance, solution):
    return score_prefix_with_feasibility_tail(
        instance,
        solution,
        score_days=10,
        feasibility_days=10,
        ignore_tail_call_ins=True,
    )


def interval_for(instance, shift):
    derived = derive_solution(instance, Solution(shifts=(replace(shift, index=0),)))[0]
    return derived.shift.start, derived.end


def route_duration(instance, shift):
    start, end = interval_for(instance, shift)
    return end - start


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


def conflict_count(instance, solution, candidate):
    interval = interval_for(instance, candidate)
    return sum(1 for shift in solution.shifts if conflicts(instance, shift, candidate, interval))


def served_points(shift):
    return {operation.point for operation in shift.operations if operation.quantity > 0}


def signature(shift):
    return (
        shift.driver,
        shift.trailer,
        shift.start,
        tuple((operation.point, operation.arrival, round(operation.quantity, 3)) for operation in shift.operations),
    )


if __name__ == "__main__":
    main()
