from __future__ import annotations

from vrp_solver.model import Customer, Driver, Instance, Operation, Shift, Solution, Source, TimeWindow, Trailer
from vrp_solver.solver.scenario import (
    ForecastDistribution,
    build_hedged_instance,
    build_hedged_instance_from_distribution,
    build_scenario_instance,
    forecast_distribution_from_rows,
    generate_scenarios,
    load_forecast_distribution,
    route_wrapped_dummy_distribution,
    scenarios_from_distribution,
    write_forecast_distribution_csv,
)


def tiny_instance(*, forecast=(10.0, 10.0, 10.0, 10.0)) -> Instance:
    return Instance(
        name="tiny",
        unit=1440,
        horizon=len(forecast),
        time_matrix=((0, 1, 1), (1, 0, 1), (1, 1, 0)),
        distance_matrix=((0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)),
        base_index=0,
        drivers=(
            Driver(
                index=0,
                min_inter_shift_duration=0,
                max_driving_duration=100,
                trailer_ids=(0,),
                time_windows=(TimeWindow(0, 10_000),),
                layover_duration=0,
                time_cost=1.0,
                layover_cost=0.0,
            ),
        ),
        trailers=(Trailer(index=0, capacity=100.0, initial_quantity=0.0, distance_cost=1.0),),
        sources=(Source(index=1, allowed_trailers=(0,), setup_time=0),),
        customers=(
            Customer(
                index=2,
                layover_customer=False,
                call_in=False,
                orders=(),
                setup_time=0,
                time_windows=(TimeWindow(0, 10_000),),
                allowed_trailers=(0,),
                forecast=tuple(forecast),
                capacity=100.0,
                initial_tank_quantity=100.0,
                min_operation_quantity=1.0,
                safety_level=20.0,
            ),
        ),
    )


def test_generate_scenarios_is_seeded_and_keeps_commit_window_true() -> None:
    instance = tiny_instance(forecast=(10.0, 20.0, 30.0))
    scenarios_a = generate_scenarios(
        instance,
        n_scenarios=3,
        seed=7,
        commit_end_day=1,
        day_sigma_schedule={"plan": 0.25, "buffer": 0.50},
    )
    scenarios_b = generate_scenarios(
        instance,
        n_scenarios=3,
        seed=7,
        commit_end_day=1,
        day_sigma_schedule={"plan": 0.25, "buffer": 0.50},
    )

    assert scenarios_a == scenarios_b
    assert len(scenarios_a[2]) == 3
    assert all(scenario[0] == 10.0 for scenario in scenarios_a[2])
    assert all(value >= 0.0 for scenario in scenarios_a[2] for value in scenario)


def test_build_hedged_instance_uses_window_percentiles_and_capacity_buffer() -> None:
    instance = tiny_instance()
    scenarios = {
        2: [
            (10.0, 10.0, 10.0, 10.0),
            (20.0, 20.0, 20.0, 20.0),
            (30.0, 30.0, 30.0, 30.0),
        ]
    }

    hedged = build_hedged_instance(
        instance,
        scenarios,
        commit_end_day=1,
        plan_end_day=2,
        commit_percentile=50.0,
        plan_percentile=75.0,
        buffer_percentile=90.0,
        capacity_buffer=0.05,
    )

    customer = hedged.customer_by_point[2]
    assert customer.forecast == (20.0, 25.0, 28.0, 28.0)
    assert customer.capacity == 95.0


def test_build_scenario_instance_selects_one_scenario_without_capacity_shrink() -> None:
    instance = tiny_instance()
    scenario_instance = build_scenario_instance(
        instance,
        {2: [(1.0, 2.0, 3.0, 4.0), (4.0, 3.0, 2.0, 1.0)]},
        1,
    )

    customer = scenario_instance.customer_by_point[2]
    assert customer.forecast == (4.0, 3.0, 2.0, 1.0)
    assert customer.capacity == 100.0


def test_build_hedged_instance_from_quantile_distribution() -> None:
    instance = tiny_instance()
    distribution = ForecastDistribution(
        deterministic={2: (10.0, 10.0, 10.0, 10.0)},
        samples={},
        quantiles={
            50.0: {2: (10.0, 10.0, 10.0, 10.0)},
            75.0: {2: (20.0, 20.0, 20.0, 20.0)},
            95.0: {2: (30.0, 30.0, 30.0, 30.0)},
        },
    )

    hedged = build_hedged_instance_from_distribution(
        instance,
        distribution,
        commit_end_day=1,
        plan_end_day=2,
        commit_percentile=50.0,
        plan_percentile=75.0,
        buffer_percentile=95.0,
        capacity_buffer=0.10,
    )

    customer = hedged.customer_by_point[2]
    assert customer.forecast == (10.0, 20.0, 30.0, 30.0)
    assert customer.capacity == 90.0


def test_scenarios_from_quantile_distribution_interpolates_stress_paths() -> None:
    distribution = ForecastDistribution(
        deterministic={2: (10.0, 10.0)},
        samples={},
        quantiles={
            50.0: {2: (10.0, 20.0)},
            90.0: {2: (30.0, 60.0)},
        },
    )

    scenarios = scenarios_from_distribution(
        distribution,
        percentiles=(50.0, 70.0, 95.0),
    )

    assert scenarios[2][0] == (10.0, 20.0)
    assert scenarios[2][1] == (20.0, 40.0)
    assert scenarios[2][2] == (30.0, 60.0)


def test_forecast_distribution_from_wide_quantile_rows() -> None:
    instance = tiny_instance()

    distribution = forecast_distribution_from_rows(
        instance,
        [
            {"item_id": "customer_2", "step": "0", "target": "9", "0.5": "10", "q90": "20"},
            {"item_id": "customer_2", "step": "1", "target": "8", "0.5": "11", "q90": "21"},
        ],
    )

    assert distribution.deterministic[2] == (9.0, 8.0, 10.0, 10.0)
    assert distribution.quantiles[50.0][2] == (10.0, 11.0, 10.0, 10.0)
    assert distribution.quantiles[90.0][2] == (20.0, 21.0, 10.0, 10.0)


def test_forecast_distribution_from_long_quantile_rows() -> None:
    instance = tiny_instance()

    distribution = forecast_distribution_from_rows(
        instance,
        [
            {"customer_id": 2, "timestamp": "2026-01-02", "quantile": 0.9, "value": 21.0},
            {"customer_id": 2, "timestamp": "2026-01-01", "quantile": 0.9, "value": 20.0},
        ],
    )

    assert distribution.quantiles[90.0][2] == (20.0, 21.0, 10.0, 10.0)


def test_route_wrapped_dummy_distribution_grows_and_anchors_on_routes() -> None:
    instance = tiny_instance(forecast=(10.0, 10.0, 10.0, 10.0))
    solution = Solution(
        shifts=(
            Shift(
                index=0,
                driver=0,
                trailer=0,
                start=0,
                operations=(Operation(point=2, arrival=2 * 1440, quantity=10.0),),
            ),
        )
    )

    distribution = route_wrapped_dummy_distribution(
        instance,
        solution,
        quantiles=(50.0, 90.0),
        base_relative_width=0.05,
        daily_relative_growth=0.10,
        max_relative_width=0.60,
        route_anchor_width=0.02,
    )

    p90 = distribution.quantiles[90.0][2]
    assert distribution.quantiles[50.0][2] == instance.customer_by_point[2].forecast
    assert p90[1] > p90[0]
    assert p90[2] < p90[1]
    assert p90[3] > p90[2]


def test_write_route_wrapped_dummy_distribution_round_trips(tmp_path) -> None:
    instance = tiny_instance()
    distribution = route_wrapped_dummy_distribution(
        instance,
        Solution(shifts=()),
        quantiles=(50.0, 75.0, 90.0),
    )
    path = tmp_path / "dummy_ci.csv"

    write_forecast_distribution_csv(distribution, path)
    loaded = load_forecast_distribution(instance, path)

    assert loaded.quantiles[50.0][2] == distribution.quantiles[50.0][2]
    assert loaded.quantiles[75.0][2] == distribution.quantiles[75.0][2]
    assert loaded.quantiles[90.0][2] == distribution.quantiles[90.0][2]
