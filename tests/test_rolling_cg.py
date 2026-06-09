from __future__ import annotations

from dataclasses import replace

from vrp_solver.contest import score_prefix_with_feasibility_tail
from vrp_solver.model import Operation, Solution, Shift
from vrp_solver.solver.column_loop import ColumnLoopStep
from vrp_solver.solver.rolling_cg import RollingCGConfig, clip_to_tank_capacity, robust_rolling_rescue
from vrp_solver.solver.rolling_cg import _accept_window, _validate_committed_scenarios
from vrp_solver.solver.scenario import ForecastDistribution

from .test_scenario import tiny_instance


def test_validate_committed_scenarios_accepts_empty_scenarios() -> None:
    assert _validate_committed_scenarios(tiny_instance(), Solution(shifts=()), None, 1) == (
        True,
        0,
    )


def test_validate_committed_scenarios_counts_infeasible_samples() -> None:
    instance = tiny_instance(forecast=(1.0, 1.0))
    scenarios = {
        2: [
            (1.0, 1.0),
            (90.0, 90.0),
            (100.0, 100.0),
        ]
    }
    distribution = ForecastDistribution.from_samples(instance, scenarios)

    feasible, failures = _validate_committed_scenarios(
        instance,
        Solution(shifts=()),
        distribution,
        commit_end_day=2,
    )

    assert feasible is False
    assert failures == 2


def test_validate_committed_scenarios_uses_quantile_stress_paths() -> None:
    instance = tiny_instance(forecast=(1.0, 1.0))
    distribution = ForecastDistribution(
        deterministic={2: (1.0, 1.0)},
        samples={},
        quantiles={
            50.0: {2: (1.0, 1.0)},
            90.0: {2: (100.0, 100.0)},
        },
    )

    feasible, failures = _validate_committed_scenarios(
        instance,
        Solution(shifts=()),
        distribution,
        commit_end_day=2,
    )

    assert feasible is False
    assert failures == 1


def test_accept_window_rejects_infeasible_replacement_for_feasible_incumbent() -> None:
    instance = tiny_instance(forecast=(1.0, 1.0))
    incumbent = Solution(shifts=())
    incumbent_score = score_prefix_with_feasibility_tail(
        instance,
        incumbent,
        score_days=1,
        feasibility_days=1,
    )
    bad_instance = tiny_instance(forecast=(100.0, 100.0))
    candidate_score = score_prefix_with_feasibility_tail(
        bad_instance,
        incumbent,
        score_days=1,
        feasibility_days=1,
    )

    accepted, reason = _accept_window(incumbent_score, candidate_score, 0)

    assert accepted is False
    assert "infeasible true prefix" in reason


def test_rolling_rescue_keeps_feasible_incumbent_when_candidate_breaks_prefix(monkeypatch) -> None:
    import vrp_solver.solver.rolling_cg as rolling_cg

    instance = tiny_instance(forecast=(1.0,))
    incumbent = Solution(shifts=())
    overfill_candidate = Solution(
        shifts=(
            Shift(
                index=0,
                driver=0,
                trailer=0,
                start=0,
                operations=(Operation(point=2, arrival=1, quantity=200.0),),
            ),
        )
    )

    def fake_column_generation_rescue(_instance, _baseline, *, config):
        return overfill_candidate, ()

    monkeypatch.setattr(
        rolling_cg,
        "column_generation_rescue",
        fake_column_generation_rescue,
    )

    solution, steps = robust_rolling_rescue(
        instance,
        incumbent,
        config=RollingCGConfig(
            mode="deterministic",
            horizon_days=1,
            commit_days=1,
            lookahead_days=0,
            cg_iterations=1,
            max_hedge_retries=0,
        ),
        progress=None,
    )
    score = score_prefix_with_feasibility_tail(
        instance,
        solution,
        score_days=1,
        feasibility_days=1,
    )

    assert score.feasible is True
    assert steps[0].accepted is False
    assert "kept incumbent" in steps[0].rejection_reason


def test_week_ahead_round_uses_lookahead_but_returns_committed_prefix(monkeypatch) -> None:
    import vrp_solver.solver.rolling_cg as rolling_cg

    base_instance = tiny_instance(forecast=(1.0, 1.0, 1.0, 1.0))
    instance = replace(
        base_instance,
        customers=(replace(base_instance.customers[0], initial_tank_quantity=99.0),),
    )
    candidate = Solution(
        shifts=(
            Shift(
                index=0,
                driver=0,
                trailer=0,
                start=0,
                operations=(
                    Operation(point=1, arrival=0, quantity=-1.0),
                    Operation(point=2, arrival=1, quantity=1.0),
                ),
            ),
            Shift(
                index=1,
                driver=0,
                trailer=0,
                start=2 * 1440,
                operations=(
                    Operation(point=1, arrival=2 * 1440, quantity=-1.0),
                    Operation(point=2, arrival=2 * 1440 + 1, quantity=1.0),
                ),
            ),
        )
    )
    seen_end_days = []

    def fake_column_generation_rescue(_instance, _baseline, *, config):
        seen_end_days.append(config.end_day)
        return candidate, ()

    monkeypatch.setattr(
        rolling_cg,
        "column_generation_rescue",
        fake_column_generation_rescue,
    )
    monkeypatch.setattr(rolling_cg, "_accept_window", lambda *_args: (True, ""))

    solution, steps = robust_rolling_rescue(
        instance,
        Solution(shifts=()),
        config=RollingCGConfig(
            mode="hedged",
            horizon_days=4,
            commit_days=1,
            lookahead_days=2,
            cg_iterations=0,
            max_hedge_retries=0,
            forecast_distribution=ForecastDistribution(
                deterministic={2: (1.0, 1.0, 1.0, 1.0)},
                samples={},
                quantiles={90.0: {2: (1.0, 1.2, 1.4, 1.6)}},
            ),
            max_rounds=1,
            committed_output_only=True,
        ),
    )

    assert seen_end_days == [3]
    assert len(steps) == 1
    assert steps[0].commit_end_day == 1
    assert steps[0].solve_end_day == 3
    assert len(solution.shifts) == 1
    assert solution.shifts[0].start == 0


def test_clip_to_tank_capacity_balances_source_load_after_delivery_clip() -> None:
    base_instance = tiny_instance(forecast=(1.0,))
    instance = replace(
        base_instance,
        customers=(replace(base_instance.customers[0], initial_tank_quantity=99.0),),
    )
    solution = Solution(
        shifts=(
            Shift(
                index=0,
                driver=0,
                trailer=0,
                start=0,
                operations=(
                    Operation(point=1, arrival=0, quantity=-50.0),
                    Operation(point=2, arrival=1, quantity=50.0),
                ),
            ),
        )
    )

    clipped = clip_to_tank_capacity(instance, solution)

    assert clipped.shifts[0].operations[0].quantity == -1.0
    assert clipped.shifts[0].operations[1].quantity == 1.0


def test_final_capacity_clip_is_scored_before_return(monkeypatch) -> None:
    import vrp_solver.solver.rolling_cg as rolling_cg

    base_instance = tiny_instance(forecast=(1.0, 90.0, 90.0))
    instance = replace(
        base_instance,
        customers=(replace(base_instance.customers[0], initial_tank_quantity=100.0),),
    )
    candidate = Solution(
        shifts=(
            Shift(
                index=0,
                driver=0,
                trailer=0,
                start=0,
                operations=(
                    Operation(point=1, arrival=0, quantity=-90.0),
                    Operation(point=2, arrival=1, quantity=90.0),
                ),
            ),
        )
    )

    def fake_column_generation_rescue(_instance, _baseline, *, config):
        return candidate, ()

    monkeypatch.setattr(
        rolling_cg,
        "column_generation_rescue",
        fake_column_generation_rescue,
    )

    solution, _steps = robust_rolling_rescue(
        instance,
        Solution(shifts=()),
        config=RollingCGConfig(
            mode="deterministic",
            horizon_days=3,
            commit_days=3,
            lookahead_days=0,
            max_hedge_retries=0,
            final_clip_capacity=True,
        ),
    )
    score = score_prefix_with_feasibility_tail(
        instance,
        solution,
        score_days=3,
        feasibility_days=3,
    )

    assert score.tank_overfill_steps == 0
    assert score.feasibility_errors == 3


def test_progress_log_records_iteration_milestone_and_next_danger(tmp_path, monkeypatch) -> None:
    import csv
    import vrp_solver.solver.rolling_cg as rolling_cg

    instance = tiny_instance(forecast=(1.0, 1.0, 1.0, 1.0))
    candidate = Solution(shifts=())
    cg_step = ColumnLoopStep(
        iteration=0,
        generated_candidates=3,
        pool_size=5,
        selected_extra_shifts=1,
        feasible=True,
        feasibility_errors=0,
        hard_violations=0,
        first_safety_breach_minute=None,
        cost=12.5,
        logistic_ratio=0.25,
        min_commit_doi=2.0,
        vulnerable_commit_customers=[(2, 2.0)],
        next_after_commit_day=2,
        min_next_after_commit_doi=0.75,
        vulnerable_next_after_commit_customers=[(2, 0.75)],
        prior_loaded=2,
        prior_structurally_valid=2,
        prior_inserted=2,
        prior_selected=1,
        prior_rejected=1,
        prior_skeletons_regenerated=1,
        prior_rejected_pressure_cover=1,
        prior_rejected_route_summary="d0-t0-b0:2*2",
        min_lookahead_doi=0.5,
        vulnerable_lookahead_customers=[(2, 0.5)],
    )

    def fake_column_generation_rescue(_instance, _baseline, *, config):
        return candidate, (cg_step,)

    monkeypatch.setattr(
        rolling_cg,
        "column_generation_rescue",
        fake_column_generation_rescue,
    )
    log_path = tmp_path / "progress.csv"

    robust_rolling_rescue(
        instance,
        Solution(shifts=()),
        config=RollingCGConfig(
            mode="hedged",
            horizon_days=4,
            commit_days=1,
            lookahead_days=2,
            max_rounds=1,
            committed_output_only=True,
            final_clip_capacity=False,
            max_hedge_retries=0,
            progress_log_path=str(log_path),
        ),
    )

    with log_path.open() as handle:
        rows = list(csv.DictReader(handle))

    assert [row["event"] for row in rows[:2]] == [
        "milestone_feasible",
        "iteration_improved",
    ]
    assert rows[0]["next_after_commit_day"] == "2"
    assert rows[0]["next_after_commit_min_doi"] == "0.75"
    assert rows[0]["prior_loaded"] == "2"
    assert rows[0]["prior_selected"] == "1"
    assert rows[0]["prior_rejected_pressure_cover"] == "1"
    assert rows[0]["prior_rejected_route_summary"] == "d0-t0-b0:2*2"
    assert rows[-1]["event"] == "final"
