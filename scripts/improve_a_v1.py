#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vrp_solver.analysis import summarize_solution
from vrp_solver.contest import score_prefix_with_feasibility_tail
from vrp_solver.evaluate import run_checker
from vrp_solver.highs_time_opt import optimize_shift_times
from vrp_solver.model import Instance, Operation, Shift, Solution
from vrp_solver.xml_io import load_instance, load_solution, save_solution


DATA_DIR = PROJECT_ROOT / "roadef_2016_data"
RESULTS_DIR = DATA_DIR / "hust_smart_results"
INSTANCES_DIR = DATA_DIR / "set_A_v1_1" / "Instances V1.1"
V1_CHECKER = (
    DATA_DIR
    / "checker_v1_1"
    / "Checker V1 v1.1.0.0"
    / "Challenge_Roadef_EURO_Checker_V1"
    / "bin"
    / "Release"
    / "IRP_Roadef_Challenge_Checker.exe"
)

SELECTED_SOLUTIONS = {
    "V_1.1": "v1_1.1_cached_expand3_pruned_maxfill.xml",
    "V_1.2": "v1_1.2_improved.xml",
    "V_1.3": "v1_1.3_improved_squeezed.xml",
    "V_1.4": "v1_1.4_official_greedy.xml",
    "V_1.5": "v1_1.5_improved_squeezed.xml",
    "V_1.6": "v1_1.6_improved_squeezed.xml",
    "V_1.7": "v1_1.7_improved_squeezed.xml",
    "V_1.8": "v1_1.8_improved_squeezed.xml",
    "V_1.9": "v1_1.9_rescued_feasible.xml",
    "V_1.10": "v1_1.10_official_greedy.xml",
    "V_1.11": "v1_1.11_rescued.xml",
}

EPSILON = 1e-7


@dataclass(frozen=True)
class CandidateScore:
    feasible: bool
    local_errors: int
    local_ratio: float
    official_valid: bool
    official_ratio: float
    official_first_rule: str


@dataclass(frozen=True)
class MoveResult:
    move: str
    score: CandidateScore
    solution: Solution


@dataclass(frozen=True)
class PhaseStats:
    tried: int = 0
    local_feasible: int = 0
    local_better: int = 0
    official_checked: int = 0
    official_valid: int = 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="V1-only Set A polishing hillclimber gated by the official checker."
    )
    parser.add_argument(
        "--instance",
        choices=sorted(SELECTED_SOLUTIONS, key=_instance_sort_key),
        help="Single Set A V1 instance to improve. Defaults to all selected instances.",
    )
    parser.add_argument("--seed", type=Path, help="Seed solution XML for a single instance.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "scratch" / "a_v1_tuning",
        help="Directory for improved XMLs and leaderboard CSV.",
    )
    parser.add_argument("--max-passes", type=int, default=3)
    parser.add_argument("--max-reorder-edits", type=int, default=1000)
    parser.add_argument("--max-shift-deletions", type=int, default=250)
    parser.add_argument("--max-source-edits", type=int, default=500)
    parser.add_argument("--max-trim-edits", type=int, default=500)
    parser.add_argument(
        "--trim-fractions",
        default="0.50,0.25,0.10",
        help="Comma-separated delivery trim fractions tried per operation.",
    )
    args = parser.parse_args()

    if shutil.which("mono") is None or not V1_CHECKER.exists():
        raise SystemExit("official V1 checker requires mono and the bundled checker executable")

    instances = [args.instance] if args.instance else sorted(SELECTED_SOLUTIONS, key=_instance_sort_key)
    if args.seed is not None and len(instances) != 1:
        raise SystemExit("--seed requires --instance")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for instance_name in instances:
        seed = args.seed or RESULTS_DIR / SELECTED_SOLUTIONS[instance_name]
        result = improve_instance(
            instance_name,
            seed,
            args.output_dir / instance_name,
            max_passes=args.max_passes,
            max_reorder_edits=args.max_reorder_edits,
            max_shift_deletions=args.max_shift_deletions,
            max_source_edits=args.max_source_edits,
            max_trim_edits=args.max_trim_edits,
            trim_fractions=tuple(float(value) for value in args.trim_fractions.split(",") if value.strip()),
        )
        rows.append(result)

    leaderboard = args.output_dir / "leaderboard.csv"
    _write_leaderboard(rows, leaderboard)
    print(f"wrote,{leaderboard}")
    return 0


def improve_instance(
    instance_name: str,
    seed_path: Path,
    output_dir: Path,
    *,
    max_passes: int,
    max_reorder_edits: int,
    max_shift_deletions: int,
    max_source_edits: int,
    max_trim_edits: int,
    trim_fractions: tuple[float, ...],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    instance_path = _instance_path(instance_name)
    instance = load_instance(instance_path)
    current = _reindex(load_solution(seed_path))
    score = score_solution(instance, instance_path, current)
    if not score.feasible:
        raise RuntimeError(f"seed is not V1-valid for {instance_name}: {seed_path}")

    moves: list[str] = []
    print(f"instance,{instance_name},seed,{seed_path},ratio,{score.official_ratio:.6f}")
    for pass_index in range(max_passes):
        accepted = None
        for phase, generator in (
            ("reorder", lambda: route_reorder_moves(instance, current, limit=max_reorder_edits)),
            ("delete", lambda: route_deletion_moves(instance, current, limit=max_shift_deletions)),
            ("source", lambda: source_cleanup_moves(instance, current, limit=max_source_edits)),
            ("trim", lambda: quantity_trim_moves(
                instance,
                current,
                fractions=trim_fractions,
                limit=max_trim_edits,
            )),
        ):
            accepted, stats = first_improving_move(instance, instance_path, score, generator())
            print(
                "progress,{},{},{},tried,{},local_feasible,{},local_better,{},official_checked,{},official_valid,{}".format(
                    instance_name,
                    pass_index + 1,
                    phase,
                    stats.tried,
                    stats.local_feasible,
                    stats.local_better,
                    stats.official_checked,
                    stats.official_valid,
                )
            )
            if accepted is not None:
                break
        if accepted is None:
            break
        current = _reindex(accepted.solution)
        score = accepted.score
        moves.append(accepted.move)
        checkpoint = output_dir / f"{instance_name}_pass{pass_index + 1}_{accepted.move}.xml"
        save_solution(current, checkpoint)
        print(f"accepted,{instance_name},{accepted.move},{score.official_ratio:.6f},{checkpoint}")

    output_xml = output_dir / f"{instance_name}_best.xml"
    save_solution(current, output_xml)
    return {
        "instance": instance_name,
        "seed": str(seed_path),
        "output_xml": str(output_xml),
        "official_ratio": f"{score.official_ratio:.6f}",
        "local_ratio": f"{score.local_ratio:.6f}",
        "moves": ";".join(moves),
        "move_count": len(moves),
        "official_valid": score.official_valid,
        "local_errors": score.local_errors,
    }


def first_improving_move(
    instance: Instance,
    instance_path: Path,
    incumbent_score: CandidateScore,
    candidates,
) -> tuple[MoveResult | None, PhaseStats]:
    tried = 0
    local_feasible = 0
    local_better = 0
    official_checked = 0
    official_valid_count = 0
    for move_name, candidate in candidates:
        tried += 1
        candidate = _reindex(candidate)
        local_score = score_solution_local(instance, candidate)
        if not local_score.feasible:
            continue
        local_feasible += 1
        if local_score.local_ratio + EPSILON >= incumbent_score.local_ratio:
            continue
        local_better += 1
        official_checked += 1
        official_score = score_solution_official(instance, instance_path, candidate, local_score)
        if not official_score.feasible:
            continue
        official_valid_count += 1
        if official_score.official_ratio + EPSILON < incumbent_score.official_ratio:
            return (
                MoveResult(move=move_name, score=official_score, solution=candidate),
                PhaseStats(
                    tried=tried,
                    local_feasible=local_feasible,
                    local_better=local_better,
                    official_checked=official_checked,
                    official_valid=official_valid_count,
                ),
            )
    return (
        None,
        PhaseStats(
            tried=tried,
            local_feasible=local_feasible,
            local_better=local_better,
            official_checked=official_checked,
            official_valid=official_valid_count,
        ),
    )


def route_deletion_moves(instance: Instance, solution: Solution, *, limit: int):
    summaries = {summary.index: summary for summary in summarize_solution(instance, solution)}
    ordered = sorted(
        solution.shifts,
        key=lambda shift: (
            -summaries.get(shift.index).estimated_cost if shift.index in summaries else 0.0,
            len(shift.operations),
        ),
    )
    for count, shift in enumerate(ordered):
        if count >= limit:
            break
        candidate = Solution(tuple(item for item in solution.shifts if item.index != shift.index))
        yield f"delete_shift_{shift.index}", candidate


def route_reorder_moves(instance: Instance, solution: Solution, *, limit: int):
    count = 0
    for shift in sorted(solution.shifts, key=lambda item: (-_shift_cost(instance, item), item.index)):
        seen: set[tuple[Operation, ...]] = set()
        for segment_start, segment_end in _customer_delivery_segments(instance, shift):
            if segment_end - segment_start < 1:
                continue
            for left in range(segment_start, segment_end):
                if count >= limit:
                    return
                operations = list(shift.operations)
                operations[left], operations[left + 1] = operations[left + 1], operations[left]
                candidate_ops = tuple(operations)
                if candidate_ops == shift.operations or candidate_ops in seen:
                    continue
                seen.add(candidate_ops)
                count += 1
                candidate_shift = optimize_shift_times(
                    instance,
                    Shift(shift.index, shift.driver, shift.trailer, shift.start, candidate_ops),
                )
                yield (
                    f"swap_s{shift.index}_o{left}_{left + 1}",
                    _replace_shift(solution, candidate_shift),
                )
            for left in range(segment_start, segment_end - 1):
                for right in range(left + 2, min(segment_end + 1, left + 6)):
                    if count >= limit:
                        return
                    operations = list(shift.operations)
                    operations[left : right + 1] = reversed(operations[left : right + 1])
                    candidate_ops = tuple(operations)
                    if candidate_ops == shift.operations or candidate_ops in seen:
                        continue
                    seen.add(candidate_ops)
                    count += 1
                    candidate_shift = optimize_shift_times(
                        instance,
                        Shift(shift.index, shift.driver, shift.trailer, shift.start, candidate_ops),
                    )
                    yield (
                        f"reverse_s{shift.index}_o{left}_{right}",
                        _replace_shift(solution, candidate_shift),
                    )


def _customer_delivery_segments(instance: Instance, shift: Shift) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    previous: int | None = None
    for index, operation in enumerate(shift.operations):
        is_delivery = operation.point in instance.customer_by_point and operation.quantity > 0
        if is_delivery:
            if start is None:
                start = index
            previous = index
            continue
        if start is not None and previous is not None:
            segments.append((start, previous))
        start = None
        previous = None
    if start is not None and previous is not None:
        segments.append((start, previous))
    return segments


def source_cleanup_moves(instance: Instance, solution: Solution, *, limit: int):
    count = 0
    for shift in sorted(solution.shifts, key=lambda item: (-_shift_cost(instance, item), item.index)):
        for op_index, operation in enumerate(shift.operations):
            if count >= limit:
                return
            if operation.point not in instance.source_by_point or operation.quantity >= 0:
                continue
            candidate_shift = _remove_source_and_rebalance_shift(instance, shift, op_index)
            if candidate_shift is None:
                continue
            candidate = _replace_shift(solution, candidate_shift)
            count += 1
            yield f"remove_source_s{shift.index}_o{op_index}", candidate


def quantity_trim_moves(
    instance: Instance,
    solution: Solution,
    *,
    fractions: tuple[float, ...],
    limit: int,
):
    count = 0
    for shift in sorted(solution.shifts, key=lambda item: (-_shift_cost(instance, item), item.index)):
        for op_index, operation in enumerate(shift.operations):
            customer = instance.customer_by_point.get(operation.point)
            if customer is None or customer.call_in or operation.quantity <= customer.min_operation_quantity:
                continue
            for fraction in fractions:
                if count >= limit:
                    return
                new_quantity = operation.quantity * (1.0 - fraction)
                if new_quantity < customer.min_operation_quantity:
                    continue
                candidate_shift = _replace_operation(
                    shift,
                    op_index,
                    Operation(operation.point, operation.arrival, new_quantity),
                )
                candidate = _replace_shift(solution, candidate_shift)
                count += 1
                yield f"trim_s{shift.index}_o{op_index}_{fraction:.2f}", candidate


def _remove_source_and_rebalance_shift(
    instance: Instance,
    shift: Shift,
    source_operation_index: int,
) -> Shift | None:
    operations = [
        operation
        for index, operation in enumerate(shift.operations)
        if index != source_operation_index
    ]
    trailer = instance.trailers[shift.trailer]
    load = trailer.initial_quantity
    balanced: list[Operation] = []
    changed = False
    for operation in operations:
        if operation.point in instance.source_by_point and operation.quantity < 0:
            load = min(trailer.capacity, load - operation.quantity)
            balanced.append(operation)
            continue
        customer = instance.customer_by_point.get(operation.point)
        if customer is not None and operation.quantity > 0:
            quantity = min(operation.quantity, load)
            if quantity + EPSILON < operation.quantity:
                changed = True
            if quantity + EPSILON < customer.min_operation_quantity:
                changed = True
                continue
            load -= quantity
            balanced.append(Operation(operation.point, operation.arrival, quantity))
            continue
        balanced.append(operation)
    if not changed and len(balanced) == len(shift.operations) - 1:
        return Shift(shift.index, shift.driver, shift.trailer, shift.start, tuple(balanced))
    if not balanced:
        return None
    return Shift(shift.index, shift.driver, shift.trailer, shift.start, tuple(balanced))


def score_solution(instance: Instance, instance_path: Path, solution: Solution) -> CandidateScore:
    local_score = score_solution_local(instance, solution)
    if not local_score.feasible:
        return local_score
    return score_solution_official(instance, instance_path, solution, local_score)


def score_solution_local(instance: Instance, solution: Solution) -> CandidateScore:
    days = (instance.horizon * instance.unit + 1439) // 1440
    local = score_prefix_with_feasibility_tail(
        instance,
        solution,
        score_days=days,
        feasibility_days=days,
        ignore_tail_call_ins=True,
    )
    local_ratio = local.scored_estimated_cost / max(1.0, local.scored_delivered_quantity)
    if not local.feasible:
        return CandidateScore(
            feasible=False,
            local_errors=local.feasibility_errors,
            local_ratio=local_ratio,
            official_valid=False,
            official_ratio=float("inf"),
            official_first_rule="local infeasible",
        )
    return CandidateScore(
        feasible=True,
        local_errors=local.feasibility_errors,
        local_ratio=local_ratio,
        official_valid=False,
        official_ratio=float("inf"),
        official_first_rule="official checker not run",
    )


def score_solution_official(
    instance: Instance,
    instance_path: Path,
    solution: Solution,
    local_score: CandidateScore,
) -> CandidateScore:
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        save_solution(solution, temp_path)
        official_valid, official_ratio, first_rule = run_checker(instance_path, temp_path, V1_CHECKER)
    finally:
        temp_path.unlink(missing_ok=True)
    return CandidateScore(
        feasible=local_score.feasible and official_valid and official_ratio is not None,
        local_errors=local_score.local_errors,
        local_ratio=local_score.local_ratio,
        official_valid=official_valid,
        official_ratio=official_ratio if official_ratio is not None else float("inf"),
        official_first_rule=first_rule,
    )


def _replace_shift(solution: Solution, replacement: Shift) -> Solution:
    return Solution(tuple(replacement if shift.index == replacement.index else shift for shift in solution.shifts))


def _replace_operation(shift: Shift, operation_index: int, operation: Operation) -> Shift:
    operations = list(shift.operations)
    operations[operation_index] = operation
    return Shift(shift.index, shift.driver, shift.trailer, shift.start, tuple(operations))


def _shift_cost(instance: Instance, shift: Shift) -> float:
    distance = 0.0
    last = instance.base_index
    for operation in shift.operations:
        distance += instance.distance_matrix[last][operation.point]
        last = operation.point
    distance += instance.distance_matrix[last][instance.base_index]
    return distance * instance.trailers[shift.trailer].distance_cost


def _reindex(solution: Solution) -> Solution:
    return Solution(
        tuple(
            Shift(index, shift.driver, shift.trailer, shift.start, shift.operations)
            for index, shift in enumerate(sorted(solution.shifts, key=lambda item: (item.start, item.index)))
        )
    )


def _write_leaderboard(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "instance",
        "seed",
        "output_xml",
        "official_ratio",
        "local_ratio",
        "moves",
        "move_count",
        "official_valid",
        "local_errors",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _instance_path(instance_name: str) -> Path:
    return INSTANCES_DIR / f"Instance_{instance_name}.xml"


def _instance_sort_key(instance_name: str) -> tuple[int, ...]:
    return tuple(int(part) for part in instance_name.removeprefix("V_").split("."))


if __name__ == "__main__":
    raise SystemExit(main())
