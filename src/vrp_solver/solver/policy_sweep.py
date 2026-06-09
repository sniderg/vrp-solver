from __future__ import annotations

import csv
from dataclasses import dataclass, fields, replace
from pathlib import Path

from ..contest import score_prefix_with_feasibility_tail
from ..model import Instance, Solution
from ..xml_io import save_solution
from .backtest import backtest_solution_against_distribution, route_stability_metrics
from .rolling_cg import RollingCGConfig, robust_rolling_rescue
from .scenario import ForecastDistribution


SWEEP_COLUMNS = (
    "policy_id",
    "mode",
    "plan_percentile",
    "buffer_percentile",
    "capacity_buffer",
    "commit_days",
    "lookahead_days",
    "target_fill_ratio",
    "max_pressure_customers",
    "cg_iterations",
    "risk_penalty_stockout",
    "risk_penalty_safety_kg_min",
    "risk_penalty_overfill",
    "risk_penalty_route_instability",
)


@dataclass(frozen=True)
class PolicySweepRow:
    policy_id: str
    config: RollingCGConfig
    risk_penalty_stockout: float = 1_000_000.0
    risk_penalty_safety_kg_min: float = 1_000.0
    risk_penalty_overfill: float = 10_000.0
    risk_penalty_route_instability: float = 1_000.0


@dataclass(frozen=True)
class PolicySweepResult:
    policy_id: str
    feasible: bool
    mean_cost: float
    failure_rate: float
    safety_severity: float
    overfill_severity: float
    route_churn: float
    robust_score: float
    output_xml: str

    def flat(self) -> dict[str, object]:
        return self.__dict__.copy()


def load_policy_sweep_csv(path: str | Path, base_config: RollingCGConfig) -> tuple[PolicySweepRow, ...]:
    with Path(path).open(newline="") as handle:
        return tuple(_parse_policy_row(row, index, base_config) for index, row in enumerate(csv.DictReader(handle), start=1))


class ManualCsvSweepAdapter:
    """Optimizer-neutral adapter that treats a CSV as the policy proposal source."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def propose(self, base_config: RollingCGConfig) -> tuple[PolicySweepRow, ...]:
        return load_policy_sweep_csv(self.path, base_config)


def run_policy_sweep(
    instance: Instance,
    baseline: Solution,
    distribution: ForecastDistribution,
    policies: tuple[PolicySweepRow, ...],
    output_dir: str | Path,
    *,
    horizon_days: int,
    progress=None,
    resume: bool = False,
    force: bool = False,
    results_csv: str | Path | None = None,
) -> tuple[PolicySweepResult, ...]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(results_csv) if results_csv is not None else output_dir / "policy_sweep_results.csv"
    if force and csv_path.exists():
        csv_path.unlink()
    results = list(_load_existing_results(csv_path) if resume and not force else ())
    completed = {result.policy_id for result in results}
    for policy in policies:
        emit = progress or (lambda _msg: None)
        if resume and not force and policy.policy_id in completed:
            emit(f"policy={policy.policy_id} skipped_existing")
            continue
        emit(f"policy={policy.policy_id}")
        config = replace(
            policy.config,
            horizon_days=horizon_days,
            scenario_seed=policy.config.scenario_seed,
            forecast_distribution=distribution,
        )
        solution, _steps = robust_rolling_rescue(instance, baseline, config=config, progress=progress)
        output_xml = output_dir / f"{policy.policy_id}.xml"
        save_solution(solution, output_xml)
        backtest = backtest_solution_against_distribution(
            instance,
            solution,
            distribution,
            horizon_days=horizon_days,
        )
        stability = route_stability_metrics(baseline, solution)
        result = _score_policy(policy, backtest.summary, stability, str(output_xml))
        results.append(result)
        _append_policy_sweep_result_csv(result, csv_path)
    return tuple(sorted(results, key=lambda result: (not result.feasible, result.robust_score)))


def write_policy_sweep_results_csv(results: tuple[PolicySweepResult, ...], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(PolicySweepResult("", False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "").flat())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.flat())


def _append_policy_sweep_result_csv(result: PolicySweepResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(PolicySweepResult("", False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "").flat())
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(result.flat())


def _load_existing_results(path: str | Path) -> tuple[PolicySweepResult, ...]:
    path = Path(path)
    if not path.exists():
        return ()
    with path.open(newline="") as handle:
        return tuple(
            PolicySweepResult(
                policy_id=row["policy_id"],
                feasible=_parse_bool(row["feasible"]),
                mean_cost=float(row["mean_cost"]),
                failure_rate=float(row["failure_rate"]),
                safety_severity=float(row["safety_severity"]),
                overfill_severity=float(row["overfill_severity"]),
                route_churn=float(row["route_churn"]),
                robust_score=float(row["robust_score"]),
                output_xml=row["output_xml"],
            )
            for row in csv.DictReader(handle)
        )


def _parse_policy_row(row: dict[str, str], index: int, base_config: RollingCGConfig) -> PolicySweepRow:
    valid_config_fields = {field.name for field in fields(RollingCGConfig)}
    updates = {}
    for key, raw in row.items():
        if key not in valid_config_fields or raw in ("", None):
            continue
        current = getattr(base_config, key)
        updates[key] = _coerce_value(raw, current)
    policy_id = row.get("policy_id") or row.get("id") or f"policy_{index}"
    return PolicySweepRow(
        policy_id=str(policy_id),
        config=replace(base_config, **updates),
        risk_penalty_stockout=float(row.get("risk_penalty_stockout") or 1_000_000.0),
        risk_penalty_safety_kg_min=float(row.get("risk_penalty_safety_kg_min") or 1_000.0),
        risk_penalty_overfill=float(row.get("risk_penalty_overfill") or 10_000.0),
        risk_penalty_route_instability=float(row.get("risk_penalty_route_instability") or 1_000.0),
    )


def _score_policy(policy: PolicySweepRow, summary, stability: dict[str, float], output_xml: str) -> PolicySweepResult:
    route_churn = (
        stability["changed_customers"]
        + stability["changed_shift_count"]
        + stability["changed_delivered_quantity"]
    )
    robust_score = (
        summary.mean_cost
        + policy.risk_penalty_stockout * summary.failure_rate
        + policy.risk_penalty_safety_kg_min * summary.mean_safety_kg_min
        + policy.risk_penalty_overfill * summary.total_tank_overfill_steps
        + policy.risk_penalty_route_instability * route_churn
    )
    return PolicySweepResult(
        policy_id=policy.policy_id,
        feasible=summary.infeasible_scenarios == 0,
        mean_cost=summary.mean_cost,
        failure_rate=summary.failure_rate,
        safety_severity=summary.mean_safety_kg_min,
        overfill_severity=float(summary.total_tank_overfill_steps),
        route_churn=route_churn,
        robust_score=robust_score,
        output_xml=output_xml,
    )


def _coerce_value(raw: str, current):
    if isinstance(current, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(float(raw))
    if isinstance(current, float):
        return float(raw)
    return raw


def _parse_bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}
