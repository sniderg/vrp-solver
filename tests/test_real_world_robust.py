from __future__ import annotations

from dataclasses import replace

import pytest

from vrp_solver.cli import build_parser
from vrp_solver.model import Operation, Shift, Solution
from vrp_solver.solver.backtest import (
    ScenarioBacktestRow,
    _summarize_backtest,
    route_stability_metrics,
)
from vrp_solver.solver.calibration import (
    ForecastCalibrationRow,
    calibrate_forecast_distribution,
    forecast_calibration_report,
)
from vrp_solver.solver.history import load_realized_consumption_history
from vrp_solver.solver.history import realized_history_from_solution_week
from vrp_solver.solver.policy_sweep import ManualCsvSweepAdapter, load_policy_sweep_csv, run_policy_sweep
from vrp_solver.solver.route_priors import route_priors_from_solution
from vrp_solver.solver.scenario import ForecastDistribution

from .test_scenario import tiny_instance


def test_history_loader_normalizes_aliases_and_rejects_negative_consumption(tmp_path) -> None:
    path = tmp_path / "history.csv"
    path.write_text(
        "item_id,date,consumption,tank_reading,delivered,source\n"
        "customer_2,2026-01-01,12.5,70,5,tank\n"
    )

    rows = load_realized_consumption_history(path)

    assert rows[0].customer_id == 2
    assert rows[0].timestamp == "2026-01-01"
    assert rows[0].realized_consumption == 12.5
    assert rows[0].inventory_observed == 70.0
    assert rows[0].delivered_quantity == 5.0

    bad = tmp_path / "bad.csv"
    bad.write_text("customer_id,step,realized_consumption\n2,0,-1\n")
    with pytest.raises(ValueError, match="Negative realized consumption"):
        load_realized_consumption_history(bad)


def test_forecast_calibration_reports_undercoverage_and_ignores_unknown_customer() -> None:
    distribution = ForecastDistribution(
        deterministic={2: (10.0, 10.0)},
        samples={},
        quantiles={90.0: {2: (11.0, 11.0)}},
    )
    realized = load_realized_consumption_history_rows(
        [
            {"customer_id": "2", "step": "0", "realized_consumption": "12"},
            {"customer_id": "999", "step": "0", "realized_consumption": "99"},
        ]
    )

    rows = forecast_calibration_report(distribution, realized, known_customers={2})
    first = rows[0]
    aggregate = rows[-1]

    assert first.customer_id == 2
    assert first.mean_bias == -2.0
    assert first.p90_hit_rate == 0.0
    assert aggregate.customer_id == "aggregate"
    assert aggregate.count == 1


def test_calibrated_forecast_widens_undercovered_p90_and_keeps_capacity_external() -> None:
    distribution = ForecastDistribution(
        deterministic={2: (10.0,)},
        samples={},
        quantiles={90.0: {2: (12.0,)}},
    )
    report = (
        ForecastCalibrationRow(
            customer_id=2,
            horizon_step="all",
            count=10,
            mean_bias=-2.0,
            mae=2.0,
            underforecast_rate=1.0,
            overforecast_rate=0.0,
            p90_hit_rate=0.50,
        ),
    )

    calibrated = calibrate_forecast_distribution(distribution, report)

    assert calibrated.deterministic[2][0] > distribution.deterministic[2][0]
    assert calibrated.quantiles[90.0][2][0] > distribution.quantiles[90.0][2][0]
    assert tiny_instance().customer_by_point[2].capacity == 100.0


def test_policy_sweep_parser_sets_rolling_config_knobs(tmp_path) -> None:
    path = tmp_path / "sweep.csv"
    path.write_text(
        "policy_id,plan_percentile,buffer_percentile,capacity_buffer,commit_days,"
        "lookahead_days,target_fill_ratio,max_pressure_customers,risk_penalty_stockout\n"
        "p1,80,95,0.1,3,9,0.9,5,123\n"
    )

    policies = load_policy_sweep_csv(path, replace_default_config())

    assert policies[0].policy_id == "p1"
    assert policies[0].config.plan_percentile == 80.0
    assert policies[0].config.buffer_percentile == 95.0
    assert policies[0].config.capacity_buffer == 0.1
    assert policies[0].config.commit_days == 3
    assert policies[0].risk_penalty_stockout == 123.0
    assert ManualCsvSweepAdapter(path).propose(replace_default_config())[0].policy_id == "p1"


def test_robust_score_monotonic_and_route_churn_zero_for_identical_plans() -> None:
    low = _summarize_backtest(
        (
            ScenarioBacktestRow(0, True, 0, 0, 1.0, 0, 0, 0, 10.0, 5.0, 2.0),
        )
    )
    high = _summarize_backtest(
        (
            ScenarioBacktestRow(0, True, 0, 0, 2.0, 0, 0, 0, 10.0, 5.0, 2.0),
        )
    )
    solution = Solution(
        shifts=(Shift(0, 0, 0, 0, (Operation(2, 0, 5.0),)),)
    )

    assert high.robust_score > low.robust_score
    assert route_stability_metrics(solution, solution)["changed_customers"] == 0.0
    assert route_stability_metrics(solution, solution)["changed_shift_count"] == 0.0
    assert route_stability_metrics(solution, solution)["changed_delivered_quantity"] == 0.0


def test_route_prior_candidate_is_marked_and_not_forced() -> None:
    instance = tiny_instance()
    prior_solution = Solution(
        shifts=(Shift(0, 0, 0, 0, (Operation(2, 1, 10.0),)),)
    )

    priors = route_priors_from_solution(instance, prior_solution)

    assert len(priors) == 1
    assert priors[0].source == "historical_prior"
    assert priors[0].shift.operations[0].point == 2


def test_route_prior_skeleton_retimes_changed_driver_window() -> None:
    instance = tiny_instance()
    driver = replace(
        instance.drivers[0],
        time_windows=(replace(instance.drivers[0].time_windows[0], start=60, end=10_000),),
    )
    retimed_instance = replace(instance, drivers=(driver,))
    prior_solution = Solution(
        shifts=(
            Shift(
                0,
                0,
                0,
                0,
                (Operation(2, 1, 10.0),),
            ),
        )
    )

    priors = route_priors_from_solution(retimed_instance, prior_solution)

    skeletons = [prior for prior in priors if prior.source == "historical_prior_skeleton"]
    assert len(skeletons) == 1
    assert [operation.point for operation in skeletons[0].shift.operations] == [2]
    assert skeletons[0].shift.start == 60


def test_pressure_covering_prior_routes_get_priority_indices() -> None:
    from vrp_solver.solver.column_loop import (
        ColumnLoopConfig,
        _fixed_prior_shift_indices,
        _prior_incumbent_solution,
        _priority_prior_indices,
    )

    instance = tiny_instance()
    prior = Shift(0, 0, 0, 0, (Operation(2, 1, 10.0),))
    unrelated = Shift(1, 0, 0, 10, (Operation(2, 11, 10.0),))

    assert _priority_prior_indices(instance, [unrelated, prior], (prior,), [2]) == (0, 1)
    assert _priority_prior_indices(instance, [unrelated, prior], (prior,), [999]) == ()
    assert _fixed_prior_shift_indices(instance, Solution((unrelated, prior)), (prior,)) == {0, 1}
    incumbent = _prior_incumbent_solution(
        instance,
        Solution(()),
        ColumnLoopConfig(replace_from_day=0, end_day=1, route_prior_candidates=(prior,), route_prior_sources=("historical_prior",)),
    )
    assert incumbent is not None
    assert incumbent.shifts[0].operations[0].point == 2


def test_solution_history_extract_uses_instance_consumption_and_delivery_context() -> None:
    instance = tiny_instance(forecast=(10.0, 20.0, 30.0))
    solution = Solution(
        shifts=(Shift(0, 0, 0, 0, (Operation(2, 1440, 5.0),)),)
    )

    rows = realized_history_from_solution_week(instance, solution, history_days=2)

    assert [row.realized_consumption for row in rows] == [10.0, 20.0]
    assert rows[0].delivered_quantity is None
    assert rows[1].delivered_quantity == 5.0
    assert rows[1].source == "solution_week_instance_consumption"


def test_new_cli_commands_parse() -> None:
    parser = build_parser()

    history = parser.parse_args(["consumption-history-check", "history.csv", "--output-csv", "out.csv"])
    calibration = parser.parse_args(
        ["forecast-calibration", "i.xml", "forecast.csv", "realized.csv", "--output-csv", "cal.csv"]
    )
    solution_history = parser.parse_args(
        ["solution-history-extract", "i.xml", "s.xml", "history.csv", "--history-days", "7"]
    )
    sweep = parser.parse_args(
        [
            "robust-policy-sweep",
            "sweep.csv",
            "--instance",
            "i.xml",
            "--baseline",
            "s.xml",
            "--forecast-input",
            "f.csv",
            "--output-dir",
            "out",
            "--selector-time-limit",
            "12",
            "--selector-mip-gap",
            "0.05",
            "--selector-threads",
            "2",
            "--selector-mip-focus",
            "1",
            "--selector-phase",
            "feasibility",
            "--resume",
        ]
    )

    assert history.command == "consumption-history-check"
    assert solution_history.command == "solution-history-extract"
    assert calibration.command == "forecast-calibration"
    assert sweep.command == "robust-policy-sweep"
    assert sweep.selector_time_limit == 12.0
    assert sweep.selector_mip_gap == 0.05
    assert sweep.selector_threads == 2
    assert sweep.selector_mip_focus == 1
    assert sweep.selector_phase == "feasibility"
    assert sweep.resume is True

    rolling = parser.parse_args(
        [
            "robust-rolling-rescue",
            "i.xml",
            "s.xml",
            "out.xml",
            "--rebalance",
            "--no-final-clip-capacity",
            "--protect-exact-prior-quantities",
            "--no-prior-incumbent",
        ]
    )
    assert rolling.rebalance is True
    assert rolling.no_final_clip_capacity is True
    assert rolling.protect_exact_prior_quantities is True
    assert rolling.no_prior_incumbent is True


def test_policy_sweep_resume_skips_completed_and_churn_uses_baseline(tmp_path, monkeypatch) -> None:
    import vrp_solver.solver.policy_sweep as policy_sweep

    instance = tiny_instance()
    baseline = Solution(shifts=())
    candidate = Solution(shifts=(Shift(0, 0, 0, 0, (Operation(2, 1, 5.0),)),))
    distribution = ForecastDistribution(deterministic={2: (1.0,)}, samples={}, quantiles={})
    policies = (
        policy_sweep.PolicySweepRow("p1", replace_default_config()),
        policy_sweep.PolicySweepRow("p2", replace_default_config()),
    )
    calls = []

    def fake_rescue(_instance, _baseline, *, config, progress=None):
        calls.append(config)
        return candidate, ()

    class FakeSummary:
        mean_cost = 10.0
        failure_rate = 0.0
        mean_safety_kg_min = 0.0
        total_tank_overfill_steps = 0
        infeasible_scenarios = 0

    class FakeBacktest:
        summary = FakeSummary()

    monkeypatch.setattr(policy_sweep, "robust_rolling_rescue", fake_rescue)
    monkeypatch.setattr(policy_sweep, "backtest_solution_against_distribution", lambda *_args, **_kwargs: FakeBacktest())
    csv_path = tmp_path / "policy_sweep_results.csv"

    first = run_policy_sweep(
        instance,
        baseline,
        distribution,
        policies[:1],
        tmp_path,
        horizon_days=1,
        results_csv=csv_path,
    )
    second = run_policy_sweep(
        instance,
        baseline,
        distribution,
        policies,
        tmp_path,
        horizon_days=1,
        resume=True,
        results_csv=csv_path,
    )

    assert [result.policy_id for result in first] == ["p1"]
    assert {result.policy_id for result in second} == {"p1", "p2"}
    assert len(calls) == 2
    assert all(result.route_churn > 0.0 for result in second)


def load_realized_consumption_history_rows(rows: list[dict[str, str]]):
    path = None
    from vrp_solver.solver.history import _normalize_row

    return tuple(_normalize_row(row, index + 2) for index, row in enumerate(rows))


def replace_default_config():
    from vrp_solver.solver.rolling_cg import RollingCGConfig

    return RollingCGConfig()
