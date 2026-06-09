from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..contest import ContestScore, score_prefix_with_feasibility_tail
from ..xml_io import load_instance, load_solution, save_solution
from .highs_selector import rebalance_drivers
from .rolling_cg import RollingCGConfig, RollingCGStep, robust_rolling_rescue


MINUTES_PER_DAY = 1440
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class RobustBatchTarget:
    name: str
    instance_xml: Path
    baseline_xml: Path
    horizon_days: int | None


@dataclass(frozen=True)
class RobustBatchResult:
    instance: str
    horizon_days: int
    output_xml: Path
    feasible: bool
    feasibility_errors: int
    hard_violations: int
    first_safety_breach_minute: int | None
    scored_estimated_cost: float
    rounds: int
    scenario_failures: int
    max_retry_count: int
    prior_loaded: int = 0
    prior_inserted: int = 0
    prior_selected: int = 0
    prior_rejected: int = 0
    prior_skeletons_regenerated: int = 0
    prior_rejected_pressure_cover: int = 0

    @classmethod
    def from_score(
        cls,
        *,
        instance: str,
        horizon_days: int,
        output_xml: Path,
        score: ContestScore,
        steps: Iterable[RollingCGStep],
    ) -> RobustBatchResult:
        step_list = list(steps)
        return cls(
            instance=instance,
            horizon_days=horizon_days,
            output_xml=output_xml,
            feasible=score.feasible,
            feasibility_errors=score.feasibility_errors,
            hard_violations=score.hard_violations,
            first_safety_breach_minute=score.first_safety_breach_minute,
            scored_estimated_cost=score.scored_estimated_cost,
            rounds=len(step_list),
            scenario_failures=sum(step.scenario_failures for step in step_list),
            max_retry_count=max((step.retry_count for step in step_list), default=0),
            prior_loaded=sum(step.prior_loaded for step in step_list),
            prior_inserted=sum(step.prior_inserted for step in step_list),
            prior_selected=sum(step.prior_selected for step in step_list),
            prior_rejected=sum(step.prior_rejected for step in step_list),
            prior_skeletons_regenerated=sum(step.prior_skeletons_regenerated for step in step_list),
            prior_rejected_pressure_cover=sum(step.prior_rejected_pressure_cover for step in step_list),
        )


def default_b_targets() -> dict[str, RobustBatchTarget]:
    data = REPO_ROOT / "roadef_2016_data"
    instances = data / "set_B" / "Instances_B_V25-11042016"
    results = data / "hust_smart_results"
    return {
        "V2.12": RobustBatchTarget(
            name="V2.12",
            instance_xml=instances / "V2.12.xml",
            baseline_xml=results / "2.12_greedy.xml",
            horizon_days=None,
        ),
        "V2.18": RobustBatchTarget(
            name="V2.18",
            instance_xml=instances / "V2.18.xml",
            baseline_xml=results / "2.18_day0_14_markov_probe.xml",
            horizon_days=14,
        ),
    }


def robust_b_config(*, quick: bool = False) -> RollingCGConfig:
    if quick:
        return RollingCGConfig(
            mode="hedged",
            commit_days=2,
            lookahead_days=2,
            n_scenarios=3,
            cg_iterations=1,
            max_pressure_customers=4,
            samples_per_customer=2,
            max_chain_length=2,
            nearest_chain_neighbors=3,
            max_candidates_per_iteration=100,
            multi_reload_columns=True,
            max_hedge_retries=0,
        )
    return RollingCGConfig(
        mode="hedged",
        commit_days=7,
        lookahead_days=7,
        n_scenarios=20,
        scenario_seed=42,
        plan_sigma=0.15,
        buffer_sigma=0.30,
        commit_percentile=50.0,
        plan_percentile=75.0,
        buffer_percentile=90.0,
        cg_iterations=5,
        max_pressure_customers=12,
        samples_per_customer=8,
        max_chain_length=4,
        nearest_chain_neighbors=10,
        max_candidates_per_iteration=1200,
        target_fill_ratio=0.95,
        multi_reload_columns=True,
        max_pre_service_fill_ratio=0.95,
        normalize_source_loads=True,
        quantity_objective="max-delivered",
        capacity_buffer=0.05,
        max_hedge_retries=2,
    )


def first_week_rescue_config(*, quick: bool = False) -> RollingCGConfig:
    config = robust_b_config(quick=quick)
    from dataclasses import replace

    return replace(
        config,
        mode="deterministic",
        horizon_days=7,
        commit_days=7,
        lookahead_days=7 if not quick else 2,
        cg_iterations=6 if not quick else 2,
        max_pressure_customers=24 if not quick else 8,
        samples_per_customer=10 if not quick else 3,
        max_candidates_per_iteration=1800 if not quick else 200,
        selector_time_limit=900.0 if not quick else 120.0,
        selector_phase="feasibility",
        max_hedge_retries=0,
        multi_reload_columns=True,
        committed_output_only=True,
        final_clip_capacity=False,
        bucket_anchor_cap=120,
        bucket_resource_time_cap=16,
        bucket_route_signature_cap=2,
        bucket_source_region_cap=360,
    )


def run_robust_batch(
    targets: Iterable[RobustBatchTarget],
    output_dir: Path,
    *,
    config: RollingCGConfig,
    progress=print,
    rebalance: bool = True,
) -> list[RobustBatchResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[RobustBatchResult] = []
    for target in targets:
        instance = load_instance(target.instance_xml)
        baseline = load_solution(target.baseline_xml)
        horizon_days = _target_horizon_days(instance, target.horizon_days)
        target_config = config if config.horizon_days == horizon_days else _with_horizon(config, horizon_days)
        progress(f"target={target.name} horizon_days={horizon_days}")
        solution, steps = robust_rolling_rescue(
            instance,
            baseline,
            config=target_config,
            progress=progress,
        )
        if rebalance:
            solution = rebalance_drivers(instance, solution)
        output_xml = output_dir / f"{target.name.replace('.', '_')}_robust.xml"
        save_solution(solution, output_xml)
        score = score_prefix_with_feasibility_tail(
            instance,
            solution,
            score_days=horizon_days,
            feasibility_days=horizon_days,
            ignore_tail_call_ins=True,
        )
        results.append(
            RobustBatchResult.from_score(
                instance=target.name,
                horizon_days=horizon_days,
                output_xml=output_xml,
                score=score,
                steps=steps,
            )
        )
    return results


def write_results_csv(results: Iterable[RobustBatchResult], path: Path) -> None:
    rows = [result.__dict__.copy() for result in results]
    fieldnames = [
        "instance",
        "horizon_days",
        "output_xml",
        "feasible",
        "feasibility_errors",
        "hard_violations",
        "first_safety_breach_minute",
        "scored_estimated_cost",
        "rounds",
        "scenario_failures",
        "max_retry_count",
        "prior_loaded",
        "prior_inserted",
        "prior_selected",
        "prior_rejected",
        "prior_skeletons_regenerated",
        "prior_rejected_pressure_cover",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row["output_xml"] = str(row["output_xml"])
            writer.writerow(row)


def _target_horizon_days(instance, requested: int | None) -> int:
    instance_days = (instance.horizon * instance.unit + MINUTES_PER_DAY - 1) // MINUTES_PER_DAY
    return instance_days if requested is None else min(requested, instance_days)


def _with_horizon(config: RollingCGConfig, horizon_days: int) -> RollingCGConfig:
    from dataclasses import replace

    return replace(config, horizon_days=horizon_days)
