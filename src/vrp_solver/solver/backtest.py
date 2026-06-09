from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from ..contest import score_prefix_with_feasibility_tail
from ..model import Instance, Solution
from ..penalties import penalty_breakdown
from .scenario import (
    ForecastDistribution,
    build_scenario_instance,
    scenarios_from_distribution,
)


@dataclass(frozen=True)
class ScenarioBacktestRow:
    scenario_index: int
    feasible: bool
    feasibility_errors: int
    hard_violations: int
    safety_kg_min: float
    tank_safety_breach_steps: int
    tank_negative_steps: int
    tank_overfill_steps: int
    cost: float
    delivered_quantity: float
    logistic_ratio: float

    def flat(self) -> dict[str, object]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class ScenarioBacktestSummary:
    scenario_count: int
    feasible_scenarios: int
    infeasible_scenarios: int
    failure_rate: float
    mean_cost: float
    worst_cost: float
    mean_logistic_ratio: float
    worst_logistic_ratio: float
    mean_safety_kg_min: float
    worst_safety_kg_min: float
    total_hard_violations: int
    total_tank_safety_breach_steps: int
    total_tank_negative_steps: int
    total_tank_overfill_steps: int
    robust_score: float = 0.0
    cost: float = 0.0
    safety_severity: float = 0.0
    overfill_severity: float = 0.0
    route_churn: float = 0.0

    def flat(self) -> dict[str, object]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class ScenarioBacktestResult:
    summary: ScenarioBacktestSummary
    rows: tuple[ScenarioBacktestRow, ...]


def backtest_solution_against_distribution(
    instance: Instance,
    solution: Solution,
    distribution: ForecastDistribution,
    *,
    horizon_days: int,
    percentiles: tuple[float, ...] | None = None,
    ignore_tail_call_ins: bool = True,
) -> ScenarioBacktestResult:
    """Evaluate one route plan against sampled or quantile forecast scenarios."""
    scenarios = scenarios_from_distribution(distribution, percentiles=percentiles)
    scenario_count = max((len(paths) for paths in scenarios.values()), default=0)
    if scenario_count == 0:
        deterministic = {
            customer.index: [distribution.deterministic.get(customer.index, customer.forecast)]
            for customer in instance.customers
            if not customer.call_in
        }
        scenarios = deterministic
        scenario_count = 1

    rows = tuple(
        _evaluate_scenario(
            instance,
            solution,
            scenarios,
            scenario_index,
            horizon_days=horizon_days,
            ignore_tail_call_ins=ignore_tail_call_ins,
        )
        for scenario_index in range(scenario_count)
    )
    return ScenarioBacktestResult(
        summary=_summarize_backtest(rows),
        rows=rows,
    )


def write_backtest_csv(result: ScenarioBacktestResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(result.rows[0].flat()) if result.rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.rows:
            writer.writerow(row.flat())


def write_backtest_summary_csv(result: ScenarioBacktestResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(result.summary.flat())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(result.summary.flat())


def route_stability_metrics(previous: Solution, current: Solution) -> dict[str, float]:
    previous_customers = _served_customers(previous)
    current_customers = _served_customers(current)
    previous_quantities = _delivered_by_customer(previous)
    current_quantities = _delivered_by_customer(current)
    customer_ids = set(previous_quantities) | set(current_quantities)
    return {
        "changed_customers": float(len(previous_customers ^ current_customers)),
        "changed_shift_count": float(abs(len(previous.shifts) - len(current.shifts))),
        "changed_delivered_quantity": sum(
            abs(previous_quantities.get(customer_id, 0.0) - current_quantities.get(customer_id, 0.0))
            for customer_id in customer_ids
        ),
    }


def _evaluate_scenario(
    instance: Instance,
    solution: Solution,
    scenarios: dict[int, list[tuple[float, ...]]],
    scenario_index: int,
    *,
    horizon_days: int,
    ignore_tail_call_ins: bool,
) -> ScenarioBacktestRow:
    scenario_instance = build_scenario_instance(instance, scenarios, scenario_index)
    score = score_prefix_with_feasibility_tail(
        scenario_instance,
        solution,
        score_days=horizon_days,
        feasibility_days=horizon_days,
        ignore_tail_call_ins=ignore_tail_call_ins,
    )
    penalties = penalty_breakdown(scenario_instance, solution)
    ratio = score.scored_estimated_cost / max(1.0, score.scored_delivered_quantity)
    return ScenarioBacktestRow(
        scenario_index=scenario_index,
        feasible=score.feasible,
        feasibility_errors=score.feasibility_errors,
        hard_violations=score.hard_violations,
        safety_kg_min=penalties.safety_kg_min,
        tank_safety_breach_steps=score.tank_safety_breach_steps,
        tank_negative_steps=score.tank_negative_steps,
        tank_overfill_steps=score.tank_overfill_steps,
        cost=score.scored_estimated_cost,
        delivered_quantity=score.scored_delivered_quantity,
        logistic_ratio=ratio,
    )


def _summarize_backtest(
    rows: tuple[ScenarioBacktestRow, ...],
) -> ScenarioBacktestSummary:
    scenario_count = len(rows)
    feasible = sum(1 for row in rows if row.feasible)
    infeasible = scenario_count - feasible
    mean_cost = _mean(row.cost for row in rows)
    failure_rate = infeasible / scenario_count if scenario_count else 0.0
    mean_safety = _mean(row.safety_kg_min for row in rows)
    overfill_steps = sum(row.tank_overfill_steps for row in rows)
    robust_score = mean_cost + 1_000_000.0 * failure_rate + 1_000.0 * mean_safety + 10_000.0 * overfill_steps
    return ScenarioBacktestSummary(
        scenario_count=scenario_count,
        feasible_scenarios=feasible,
        infeasible_scenarios=infeasible,
        failure_rate=failure_rate,
        mean_cost=mean_cost,
        worst_cost=max((row.cost for row in rows), default=0.0),
        mean_logistic_ratio=_mean(row.logistic_ratio for row in rows),
        worst_logistic_ratio=max((row.logistic_ratio for row in rows), default=0.0),
        mean_safety_kg_min=mean_safety,
        worst_safety_kg_min=max((row.safety_kg_min for row in rows), default=0.0),
        total_hard_violations=sum(row.hard_violations for row in rows),
        total_tank_safety_breach_steps=sum(row.tank_safety_breach_steps for row in rows),
        total_tank_negative_steps=sum(row.tank_negative_steps for row in rows),
        total_tank_overfill_steps=overfill_steps,
        robust_score=robust_score,
        cost=mean_cost,
        safety_severity=mean_safety,
        overfill_severity=float(overfill_steps),
        route_churn=0.0,
    )


def _mean(values) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _served_customers(solution: Solution) -> set[int]:
    return {
        operation.point
        for shift in solution.shifts
        for operation in shift.operations
        if operation.quantity > 0.0
    }


def _delivered_by_customer(solution: Solution) -> dict[int, float]:
    delivered: dict[int, float] = {}
    for shift in solution.shifts:
        for operation in shift.operations:
            if operation.quantity <= 0.0:
                continue
            delivered[operation.point] = delivered.get(operation.point, 0.0) + operation.quantity
    return delivered
