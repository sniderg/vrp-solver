from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from .evaluate import evaluate_solution
from .geo import GeoPoint
from .model import Instance, Operation, Shift, Solution
from .penalties import PenaltyWeights
from .xml_io import save_solution


@dataclass(frozen=True)
class PerturbationScope:
    clusters: tuple[int, ...]
    start_minute: int
    end_minute: int
    max_operations: int = 20


@dataclass(frozen=True)
class RemovedOperation:
    shift_index: int
    operation_index: int
    operation: Operation


@dataclass(frozen=True)
class IRPState:
    instance: Instance
    solution: Solution
    geo_points: dict[int, GeoPoint]
    scope: PerturbationScope
    removed: tuple[RemovedOperation, ...] = ()
    objective_scale: float = 1.0
    penalty_weight: float = 1.0

    def objective(self) -> float:
        evaluation = evaluate_solution(self.instance, self.solution)
        route_proxy = evaluation.estimated_distance_cost_plus_time_cost / max(
            evaluation.delivered_quantity,
            1.0,
        )
        return route_proxy * self.objective_scale + evaluation.total_penalty * self.penalty_weight

    def get_context(self):
        evaluation = evaluate_solution(self.instance, self.solution)
        return [
            evaluation.local_errors,
            evaluation.total_soft_penalty,
            evaluation.delivered_cv_daily,
            evaluation.delivered_first_day_share,
        ]

    def with_solution(
        self,
        solution: Solution,
        removed: tuple[RemovedOperation, ...] | None = None,
    ) -> "IRPState":
        return replace(self, solution=solution, removed=self.removed if removed is None else removed)


def point_in_scope(state: IRPState, operation: Operation) -> bool:
    geo = state.geo_points.get(operation.point)
    if geo is None or geo.cluster not in state.scope.clusters:
        return False
    return state.scope.start_minute <= operation.arrival < state.scope.end_minute


def remove_targeted_operations(current: IRPState, rng) -> IRPState:
    candidates: list[RemovedOperation] = []
    for shift in current.solution.shifts:
        for op_index, operation in enumerate(shift.operations):
            if operation.quantity > 0 and point_in_scope(current, operation):
                candidates.append(RemovedOperation(shift.index, op_index, operation))

    if not candidates:
        return current

    rng.shuffle(candidates)
    selected = tuple(candidates[: current.scope.max_operations])
    by_shift = {shift.index: list(shift.operations) for shift in current.solution.shifts}
    for removed in sorted(selected, key=lambda item: item.operation_index, reverse=True):
        by_shift[removed.shift_index].pop(removed.operation_index)

    shifts = tuple(
        replace(shift, operations=tuple(by_shift[shift.index]))
        for shift in current.solution.shifts
    )
    return current.with_solution(Solution(shifts=shifts), removed=selected)


def restore_removed_operations(current: IRPState, rng) -> IRPState:
    if not current.removed:
        return current

    by_shift = {shift.index: list(shift.operations) for shift in current.solution.shifts}
    for removed in sorted(current.removed, key=lambda item: item.operation_index):
        by_shift[removed.shift_index].insert(removed.operation_index, removed.operation)

    shifts = tuple(
        replace(shift, operations=tuple(by_shift[shift.index]))
        for shift in current.solution.shifts
    )
    return current.with_solution(Solution(shifts=shifts), removed=())


def jitter_targeted_arrivals(current: IRPState, rng) -> IRPState:
    shifts = []
    for shift in current.solution.shifts:
        operations = []
        for operation in shift.operations:
            if operation.quantity > 0 and point_in_scope(current, operation):
                delta = int(rng.integers(-15, 16))
                operations.append(replace(operation, arrival=max(0, operation.arrival + delta)))
            else:
                operations.append(operation)
        shifts.append(replace(shift, operations=tuple(operations)))
    return current.with_solution(Solution(shifts=tuple(shifts)))


def highs_repair_operator(current: IRPState, rng) -> IRPState:
    from .highs_repair import repair_with_highs_selection

    # Restore removed operations first so they can be optimized by MILP
    state_with_all = restore_removed_operations(current, rng)

    # Use the whole horizon for repair in this probe
    score_days = (current.instance.horizon * current.instance.unit + 1439) // 1440
    
    repaired_solution, report = repair_with_highs_selection(
        current.instance,
        state_with_all.solution,
        score_days=score_days,
    )

    return state_with_all.with_solution(repaired_solution)


def save_state_solution(state: IRPState, output_xml: str | Path) -> None:
    save_solution(state.solution, output_xml)
