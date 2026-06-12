from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import combinations
from pathlib import Path

from vrp_solver.contest import score_prefix_with_feasibility_tail
from vrp_solver.inventory import delivery_by_customer_step, project_customer_inventory, tank_violations
from vrp_solver.model import Operation, Shift, Solution
from vrp_solver.rules import derive_solution, is_time_window_valid
from vrp_solver.solver.targeted_rescue import normalize_source_loads
from vrp_solver.xml_io import load_instance, load_solution, save_solution


ROOT = Path(__file__).resolve().parents[1]
INSTANCE = ROOT / "roadef_2016_data/set_B/Instances_B_V25-11042016/V2.12.xml"
START = ROOT / "scratch/b_v2_replace_neighborhood/V2.12/best.xml"
OUT = ROOT / "scratch/b_v2_direct_topup/V2.12"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    instance = load_instance(INSTANCE)
    current = load_solution(START)
    current_score = score(instance, current)
    failures = dict(Counter(v.point for v in tank_violations(instance, current)).most_common())
    print("start", current_score.feasibility_errors, current_score.hard_violations, failures)
    candidates = direct_topup_candidates(instance, current, list(failures)[:8])
    candidates.sort(
        key=lambda shift: (
            -sum(failures.get(point, 0) for point in served_points(shift)),
            conflict_count(instance, current, shift),
            shift.start,
            shift.driver,
            shift.trailer,
        )
    )
    print("candidates", len(candidates))
    best = current
    best_score = current_score
    for index, candidate in enumerate(candidates[:500]):
        if index % 50 == 0:
            print("single", index, "best", best_score.feasibility_errors, best_score.hard_violations)
        trial = replace_many(instance, best, (candidate,))
        if trial is None:
            continue
        trial_score = score(instance, trial)
        if trial_score.hard_violations == 0 and trial_score.feasibility_errors < best_score.feasibility_errors:
            best = trial
            best_score = trial_score
            out = OUT / f"single_{best_score.feasibility_errors}.xml"
            save_solution(best, out)
            print("accept_single", best_score.feasibility_errors, sorted(served_points(candidate)), out)

    if True:
        final = OUT / "best.xml"
        save_solution(best, final)
        print("final", best_score.feasibility_errors, best_score.hard_violations, final)
        return

    top = candidates[:28]
    for combo_size in (2, 3):
        checked = 0
        for combo in combinations(top, combo_size):
            checked += 1
            if checked % 100 == 0:
                print("combo", combo_size, checked, "best", best_score.feasibility_errors)
            if not pairwise_compatible(instance, combo):
                continue
            trial = replace_many(instance, best, combo)
            if trial is None:
                continue
            trial_score = score(instance, trial)
            if trial_score.hard_violations == 0 and trial_score.feasibility_errors < best_score.feasibility_errors:
                best = trial
                best_score = trial_score
                out = OUT / f"combo{combo_size}_{best_score.feasibility_errors}.xml"
                save_solution(best, out)
                print(
                    "accept_combo",
                    combo_size,
                    best_score.feasibility_errors,
                    sorted(set().union(*(served_points(candidate) for candidate in combo))),
                    out,
                )

    final = OUT / "best.xml"
    save_solution(best, final)
    print("final", best_score.feasibility_errors, best_score.hard_violations, final)


def direct_topup_candidates(instance, solution: Solution, points: list[int]) -> list[Shift]:
    deliveries = delivery_by_customer_step(solution)
    candidates: list[Shift] = []
    for point in points:
        customer = instance.customer_by_point[point]
        if customer.call_in:
            continue
        events = project_customer_inventory(instance, customer, deliveries.get(point, {}))
        breach_events = [event for event in events if event.safety_breach]
        if not breach_events:
            continue
        first_breach = breach_events[0]
        candidate_arrivals = sorted(
            {
                max(window.start, min(first_breach.time_start - offset, window.end - customer.setup_time))
                for window in customer.time_windows
                for offset in (0, 120, 360, 720, 1080, 1440)
                if window.start <= first_breach.time_start <= window.end + 1440
            }
        )
        # Also try the starts of later windows for late-horizon residual breaches.
        candidate_arrivals.extend(window.start for window in customer.time_windows if window.start < 10 * 1440)
        for arrival in sorted(set(candidate_arrivals)):
            if arrival < 0 or arrival >= 10 * 1440:
                continue
            if not is_time_window_valid(arrival, arrival + customer.setup_time, customer.time_windows):
                continue
            arrival_step = min(max(arrival // instance.unit, 0), instance.horizon - 1)
            inv = events[arrival_step].after_consumption
            room = customer.capacity - inv
            if room < customer.min_operation_quantity:
                continue
            needed = max(
                customer.safety_level - inv,
                sum(event.safety_level - event.ending_inventory for event in breach_events if event.time_start >= arrival) / max(1, len(breach_events)),
            )
            quantity = min(room, max(customer.min_operation_quantity, needed + 0.5 * (customer.capacity - customer.safety_level)))
            if quantity < customer.min_operation_quantity:
                continue
            for driver in instance.drivers:
                for trailer_id in driver.trailer_ids:
                    if trailer_id not in customer.allowed_trailers:
                        continue
                    trailer = instance.trailers[trailer_id]
                    if quantity > trailer.capacity:
                        continue
                    for source in instance.sources:
                        if trailer_id not in source.allowed_trailers:
                            continue
                        travel_bs = instance.time_matrix[instance.base_index][source.index]
                        travel_sc = instance.time_matrix[source.index][point]
                        travel_cb = instance.time_matrix[point][instance.base_index]
                        source_arrival = arrival - source.setup_time - travel_sc
                        shift_start = source_arrival - travel_bs
                        if shift_start < 0:
                            continue
                        end = arrival + customer.setup_time + travel_cb
                        if not any(window.start <= shift_start and end <= window.end for window in driver.time_windows):
                            continue
                        candidates.append(
                            Shift(
                                index=len(candidates),
                                driver=driver.index,
                                trailer=trailer_id,
                                start=shift_start,
                                operations=(
                                    Operation(point=source.index, arrival=source_arrival, quantity=-quantity),
                                    Operation(point=point, arrival=arrival, quantity=quantity),
                                ),
                            )
                        )
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


def pairwise_compatible(instance, candidates: tuple[Shift, ...]) -> bool:
    for left, right in combinations(candidates, 2):
        if conflicts(instance, left, right, interval_for(instance, right)):
            return False
    return True


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
