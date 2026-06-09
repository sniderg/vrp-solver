from __future__ import annotations

from vrp_solver.cli import build_parser
from vrp_solver.model import Solution
from vrp_solver.solver.backtest import backtest_solution_against_distribution
from vrp_solver.solver.scenario import ForecastDistribution

from .test_scenario import tiny_instance


def test_backtest_uses_quantile_scenarios_and_summarizes_failures() -> None:
    instance = tiny_instance(forecast=(1.0, 1.0))
    distribution = ForecastDistribution(
        deterministic={2: (1.0, 1.0)},
        samples={},
        quantiles={
            50.0: {2: (1.0, 1.0)},
            90.0: {2: (100.0, 100.0)},
        },
    )

    result = backtest_solution_against_distribution(
        instance,
        Solution(shifts=()),
        distribution,
        horizon_days=2,
        percentiles=(50.0, 90.0),
    )

    assert result.summary.scenario_count == 2
    assert result.summary.infeasible_scenarios == 1
    assert result.summary.failure_rate == 0.5
    assert result.rows[0].feasible is True
    assert result.rows[1].feasible is False


def test_scenario_backtest_cli_parses_optional_policy_knobs() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "scenario-backtest",
            "instance.xml",
            "solution.xml",
            "forecast.csv",
            "--horizon-days",
            "7",
            "--percentiles",
            "50,75,90",
            "--output-csv",
            "rows.csv",
            "--fail-on-infeasible",
        ]
    )

    assert args.command == "scenario-backtest"
    assert args.horizon_days == 7
    assert args.percentiles == "50,75,90"
    assert args.fail_on_infeasible is True
