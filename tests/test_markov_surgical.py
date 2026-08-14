"""Tests for surgical search with Markov sequence selection and LAHC enabled."""
from __future__ import annotations

import pytest

from vrp_solver.model import (
    Customer,
    Driver,
    Instance,
    Operation,
    Shift,
    Solution,
    Source,
    TimeWindow,
    Trailer,
)
from vrp_solver.solver.surgical_search import (
    SurgicalSearchConfig,
    surgical_search,
)
from vrp_solver.rules import validate_solution


def _create_mini_fixture() -> tuple[Instance, Solution]:
    trailers = (
        Trailer(0, 20_000.0, 0.0, 0.0),
    )
    customers = tuple(
        Customer(
            index=2 + i,
            layover_customer=False,
            call_in=False,
            orders=(),
            setup_time=5,
            time_windows=(TimeWindow(0, 1_000),),
            allowed_trailers=(0,),
            forecast=(50.0,) * 5,
            capacity=1_000.0,
            initial_tank_quantity=60.0,
            min_operation_quantity=10.0,
            safety_level=50.0,

        )
        for i in range(5)
    )
    point_count = 7
    matrix = tuple(tuple(0 if i == j else 10 for j in range(point_count)) for i in range(point_count))
    instance = Instance(
        name="mini-fixture",
        unit=60,
        horizon=5,
        time_matrix=matrix,
        distance_matrix=tuple(tuple(float(v) for v in row) for row in matrix),
        base_index=0,
        drivers=(Driver(0, 0, 600, (0,), (TimeWindow(0, 1_000),), 60, 0.0, 0.0),),
        trailers=trailers,
        sources=(Source(1, (0,), 5),),
        customers=customers,
    )
    # Simple initial solution
    initial_solution = Solution(shifts=(
        Shift(
            index=0,
            driver=0,
            trailer=0,
            start=0,
            operations=(
                Operation(1, 10, -500.0),
                Operation(2, 25, 250.0),
                Operation(3, 40, 250.0),
            ),
        ),
    ))
    return instance, initial_solution


def test_surgical_search_with_markov_and_lahc():
    instance, solution = _create_mini_fixture()
    config = SurgicalSearchConfig(
        end_day=1,
        iterations=5,
        candidates_per_move=4,
        use_lahc=True,
        lahc_capacity=10,
        use_markov_sequence=True,
        seed=1,
    )
    best_sol, steps = surgical_search(instance, solution, config=config)
    assert len(steps) == 5
    # Verify no unhandled exceptions and valid output structure
    violations = validate_solution(instance, best_sol)
    assert isinstance(violations, tuple) or isinstance(violations, list)
