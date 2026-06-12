from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from vrp_solver.contest import score_prefix_with_feasibility_tail
from vrp_solver.model import Operation, Shift, Solution
from vrp_solver.rules import derive_solution, validate_solution
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
OUT = ROOT / "scratch/b_v2_replace_callins/V2.12"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    instance = load_instance(INSTANCE)
    current = load_solution(START)
    current_score = score(instance, current)
    callins = callin_errors(instance, current)
    print("start", current_score.feasibility_errors, current_score.hard_violations, callins)
    candidates = direct_callin_candidates(instance, callins)
    unique = {}
    for candidate in candidates:
        if served_points(candidate) & set(callins):
            unique.setdefault(signature(candidate), candidate)
    candidates = list(unique.values())
    candidates.sort(key=lambda shift: (-len(served_points(shift) & set(callins)), conflict_count(instance, current, shift), shift.start))
    print("candidates", len(candidates))
    best = current
    best_score = current_score
    for index, candidate in enumerate(candidates[:400]):
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
            print("accept", best_score.feasibility_errors, sorted(served_points(candidate) & set(callins)), out)
    final = OUT / "best.xml"
    save_solution(best, final)
    print("final", best_score.feasibility_errors, best_score.hard_violations, final)


def direct_callin_candidates(instance, callins) -> list[Shift]:
    candidates: list[Shift] = []
    for point in callins:
        customer = instance.customer_by_point[point]
        if not customer.orders:
            continue
        order = customer.orders[0]
        quantity = order.min_quantity_to_satisfy
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
                    for window in driver.time_windows:
                        travel_to_source = instance.time_matrix[instance.base_index][source.index]
                        travel_to_customer = instance.time_matrix[source.index][point]
                        travel_home = instance.time_matrix[point][instance.base_index]
                        earliest_source_arrival = max(
                            window.start + travel_to_source,
                            order.earliest_time - source.setup_time - travel_to_customer,
                        )
                        source_arrival = earliest_source_arrival
                        delivery_arrival = source_arrival + source.setup_time + travel_to_customer
                        if delivery_arrival < order.earliest_time:
                            delivery_arrival = order.earliest_time
                            source_arrival = delivery_arrival - source.setup_time - travel_to_customer
                        shift_start = source_arrival - travel_to_source
                        if shift_start < window.start:
                            shift_start = window.start
                            source_arrival = shift_start + travel_to_source
                            delivery_arrival = source_arrival + source.setup_time + travel_to_customer
                        departure = delivery_arrival + customer.setup_time
                        end = departure + travel_home
                        if not (order.earliest_time <= delivery_arrival <= order.latest_time):
                            continue
                        if end > window.end:
                            continue
                        if delivery_arrival + customer.setup_time > order.latest_time:
                            continue
                        operations = (
                            Operation(point=source.index, arrival=source_arrival, quantity=-quantity),
                            Operation(point=point, arrival=delivery_arrival, quantity=quantity),
                        )
                        candidates.append(
                            Shift(
                                index=len(candidates),
                                driver=driver.index,
                                trailer=trailer_id,
                                start=shift_start,
                                operations=operations,
                            )
                        )
    unique = {}
    for candidate in candidates:
        unique.setdefault(signature(candidate), candidate)
    return list(unique.values())


def score(instance, solution):
    return score_prefix_with_feasibility_tail(instance, solution, score_days=10, feasibility_days=10, ignore_tail_call_ins=True)


def callin_errors(instance, solution):
    return [
        violation.point
        for violation in validate_solution(instance, solution)
        if violation.severity == "error" and violation.code == "QS01" and violation.point is not None
    ]


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
