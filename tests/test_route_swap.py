from __future__ import annotations

from dataclasses import replace

from vrp_solver.model import Operation, Shift, Solution
from vrp_solver.solver.route_swap import transactional_route_swap_search

from .test_scenario import tiny_instance


def test_transactional_route_swap_accepts_only_feasible_ratio_improvement() -> None:
    base = tiny_instance(forecast=(1.0,))
    instance = replace(
        base,
        drivers=(replace(base.drivers[0], layover_duration=10_000),),
        trailers=(replace(base.trailers[0], capacity=200.0),),
        customers=(replace(base.customers[0], capacity=200.0),),
    )
    incumbent = Solution(
        shifts=(
            Shift(0, 0, 0, 0, (Operation(1, 1, -10.0), Operation(2, 2, 10.0)),),
        )
    )
    reference = Solution(
        shifts=(
            Shift(0, 0, 0, 0, (Operation(1, 1, -20.0), Operation(2, 2, 20.0)),),
        )
    )

    result = transactional_route_swap_search(
        instance,
        incumbent,
        reference,
        horizon_days=1,
        max_remove=1,
        max_add=1,
    )

    assert len(result.moves) == 1
    assert result.moves[0].feasible is True
    assert result.final_ratio < result.initial_ratio
    assert result.solution.shifts[0].operations[-1].quantity == 20.0


def test_customer_bundle_route_swap_accepts_feasible_bundle() -> None:
    base = tiny_instance(forecast=(1.0,))
    instance = replace(
        base,
        drivers=(replace(base.drivers[0], layover_duration=10_000),),
        trailers=(replace(base.trailers[0], capacity=200.0),),
        customers=(replace(base.customers[0], capacity=200.0),),
    )
    incumbent = Solution(
        shifts=(
            Shift(0, 0, 0, 0, (Operation(1, 1, -10.0), Operation(2, 2, 10.0)),),
        )
    )
    reference = Solution(
        shifts=(
            Shift(0, 0, 0, 0, (Operation(1, 1, -20.0), Operation(2, 2, 20.0)),),
        )
    )

    result = transactional_route_swap_search(
        instance,
        incumbent,
        reference,
        horizon_days=1,
        max_remove=1,
        max_add=1,
        customer_bundles=True,
    )

    assert len(result.moves) == 1
    assert result.moves[0].feasible is True
    assert result.solution.shifts[0].operations[-1].quantity == 20.0
