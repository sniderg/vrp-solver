from __future__ import annotations

import random
from types import SimpleNamespace

from vrp_solver.model import Customer, Driver, Instance, Operation, Shift, Solution, Source, TimeWindow, Trailer
from vrp_solver.solver.surgical_search import (
    SurgicalSearchConfig,
    _pressure_band_resource_block_candidates,
    _recombine_route_block_candidates,
    surgical_search,
)


def test_recombine_route_blocks_exchanges_whole_customer_runs() -> None:
    windows = (TimeWindow(0, 720),)
    matrix = tuple(
        tuple(0 if left == right else 5 for right in range(5))
        for left in range(5)
    )
    instance = Instance(
        name="recombine",
        unit=60,
        horizon=12,
        time_matrix=matrix,
        distance_matrix=tuple(tuple(float(value) for value in row) for row in matrix),
        base_index=0,
        drivers=(
            Driver(0, 0, 600, (0,), windows, 600, 0.0, 0.0),
            Driver(1, 0, 600, (1,), windows, 600, 0.0, 0.0),
        ),
        trailers=(Trailer(0, 100.0, 0.0, 0.0), Trailer(1, 100.0, 0.0, 0.0)),
        sources=(Source(0, (0, 1), 0),),
        customers=tuple(
            Customer(point, False, False, (), 0, windows, (0, 1), (0.0,) * 12, 100.0, 0.0, 1.0, 0.0)
            for point in range(1, 5)
        ),
    )
    solution = Solution((
        Shift(0, 0, 0, 0, (
            Operation(0, 0, -10.0), Operation(1, 5, 10.0),
            Operation(0, 10, -10.0), Operation(2, 15, 10.0),
        )),
        Shift(1, 1, 1, 0, (
            Operation(0, 0, -10.0), Operation(3, 5, 10.0),
            Operation(0, 10, -10.0), Operation(4, 15, 10.0),
        )),
    ))
    config = SurgicalSearchConfig(end_day=1, candidates_per_move=4)

    candidates = _recombine_route_block_candidates(instance, solution, config, random.Random(7))

    assert candidates
    assert any(
        [operation.point for operation in candidate.shifts[0].operations if operation.point != 0] != [1, 2]
        for candidate in candidates
    )
    assert any(
        len(candidate.shifts) == 1
        and len(candidate.shifts[0].operations) == 8
        for candidate in candidates
    )
    for candidate in candidates:
        assert all(len(shift.operations) in {4, 8} for shift in candidate.shifts)


def test_recombine_route_blocks_runs_quantity_repair(monkeypatch) -> None:
    """Load-path-changing recombinations must reach hard quantity repair."""
    windows = (TimeWindow(0, 720),)
    matrix = tuple(
        tuple(0 if left == right else 5 for right in range(3))
        for left in range(3)
    )
    instance = Instance(
        name="recombine-repair",
        unit=60,
        horizon=12,
        time_matrix=matrix,
        distance_matrix=tuple(tuple(float(value) for value in row) for row in matrix),
        base_index=0,
        drivers=(
            Driver(0, 0, 600, (0,), windows, 600, 0.0, 0.0),
            Driver(1, 0, 600, (1,), windows, 600, 0.0, 0.0),
        ),
        trailers=(Trailer(0, 100.0, 0.0, 0.0), Trailer(1, 100.0, 0.0, 0.0)),
        sources=(Source(0, (0, 1), 0),),
        customers=tuple(
            Customer(point, False, False, (), 0, windows, (0, 1),
                     (0.0,) * 12, 100.0, 0.0, 1.0, 0.0)
            for point in range(1, 3)
        ),
    )
    solution = Solution((
        Shift(0, 0, 0, 0, (Operation(0, 0, -10.0), Operation(1, 5, 10.0))),
        Shift(1, 1, 1, 0, (Operation(0, 0, -10.0), Operation(2, 5, 10.0))),
    ))
    calls = []

    class Report:
        status = "Optimal"

    def fake_repair(_instance, candidate, **_kwargs):
        calls.append(candidate)
        return candidate, Report()

    monkeypatch.setattr(
        "vrp_solver.solver.surgical_search.repair_quantities_with_highs",
        fake_repair,
    )
    surgical_search(
        instance,
        solution,
        config=SurgicalSearchConfig(
            end_day=1,
            iterations=1,
            candidates_per_move=4,
            workers=1,
            first_operator="recombine_route_blocks",
        ),
        progress=None,
    )

    assert calls


def test_pressure_band_block_uses_features_to_exchange_routes(monkeypatch) -> None:
    windows = (TimeWindow(0, 600),)
    matrix = (
        (0, 10, 10),
        (10, 0, 10),
        (10, 10, 0),
    )
    instance = Instance(
        name="feature-driven-pressure-block",
        unit=60,
        horizon=10,
        time_matrix=matrix,
        distance_matrix=tuple(tuple(float(value) for value in row) for row in matrix),
        base_index=0,
        drivers=(
            Driver(0, 0, 600, (0,), windows, 600, 0.0, 0.0),
            Driver(1, 0, 600, (1,), windows, 600, 0.0, 0.0),
        ),
        trailers=(Trailer(0, 100.0, 100.0, 0.0), Trailer(1, 100.0, 100.0, 0.0)),
        sources=(Source(0, (0, 1), 0),),
        customers=(
            Customer(1, False, False, (), 0, windows, (0, 1),
                     (0.0,) * 10, 100.0, 0.0, 1.0, 0.0),
            Customer(2, False, False, (), 0, windows, (0, 1),
                     (0.0,) * 10, 100.0, 0.0, 1.0, 0.0),
        ),
    )
    solution = Solution((
        Shift(0, 0, 0, 0, (Operation(1, 50, 10.0),)),
        Shift(1, 1, 1, 150, (Operation(2, 200, 10.0),)),
    ))
    monkeypatch.setattr(
        "vrp_solver.solver.surgical_search.pressure_points",
        lambda *_args, **_kwargs: [
            SimpleNamespace(customer=2, first_minute=100, deficit_area=1_000.0),
        ],
    )
    monkeypatch.setattr(
        "vrp_solver.solver.surgical_search.try_optimize_shift_times",
        lambda _instance, shift, **_kwargs: Shift(
            shift.index,
            shift.driver,
            shift.trailer,
            shift.start,
            tuple(
                Operation(operation.point, shift.start + 10 * (index + 1), operation.quantity)
                for index, operation in enumerate(shift.operations)
            ),
        ),
    )

    candidates = _pressure_band_resource_block_candidates(
        instance,
        solution,
        SurgicalSearchConfig(
            end_day=1,
            pressure_customers=1,
            candidates_per_move=8,
            samples_per_customer=4,
        ),
        random.Random(7),
    )

    assert candidates
    assert any(
        operation.point == 2 and operation.arrival <= 100
        for candidate in candidates
        for shift in candidate.shifts
        for operation in shift.operations
    )
    assert any(
        shift.driver == 0
        and shift.trailer == 0
        and any(operation.point == 2 for operation in shift.operations)
        for candidate in candidates
        for shift in candidate.shifts
    )


def test_pressure_band_block_can_evacuate_resource_predecessor(monkeypatch) -> None:
    windows = (TimeWindow(0, 600),)
    matrix = ((0, 10, 10), (10, 0, 10), (10, 10, 0))
    instance = Instance(
        name="pressure-predecessor-evacuation",
        unit=60,
        horizon=10,
        time_matrix=matrix,
        distance_matrix=tuple(tuple(float(value) for value in row) for row in matrix),
        base_index=0,
        drivers=(
            Driver(0, 0, 600, (0,), windows, 600, 0.0, 0.0),
            Driver(1, 0, 600, (1,), windows, 600, 0.0, 0.0),
        ),
        trailers=(Trailer(0, 100.0, 100.0, 0.0), Trailer(1, 100.0, 100.0, 0.0)),
        sources=(Source(0, (0, 1), 0),),
        customers=(
            Customer(1, False, False, (), 0, windows, (0, 1),
                     (0.0,) * 10, 100.0, 0.0, 1.0, 0.0),
            Customer(2, False, False, (), 0, windows, (1,),
                     (0.0,) * 10, 100.0, 0.0, 1.0, 0.0),
        ),
    )
    solution = Solution((
        Shift(0, 1, 1, 0, (Operation(1, 10, 10.0),)),
        Shift(1, 1, 1, 150, (Operation(2, 200, 10.0),)),
    ))
    monkeypatch.setattr(
        "vrp_solver.solver.surgical_search.pressure_points",
        lambda *_args, **_kwargs: [
            SimpleNamespace(customer=2, first_minute=10, deficit_area=2_000.0),
        ],
    )
    monkeypatch.setattr(
        "vrp_solver.solver.surgical_search.try_optimize_shift_times",
        lambda _instance, shift, **_kwargs: Shift(
            shift.index, shift.driver, shift.trailer, shift.start,
            tuple(
                Operation(operation.point, shift.start + 10 * (index + 1), operation.quantity)
                for index, operation in enumerate(shift.operations)
            ),
        ),
    )

    candidates = _pressure_band_resource_block_candidates(
        instance,
        solution,
        SurgicalSearchConfig(end_day=1, pressure_customers=1, candidates_per_move=8),
        random.Random(11),
    )

    assert any(
        any(
            shift.trailer == 0 and operation.point == 1
            for shift in candidate.shifts
            for operation in shift.operations
        )
        and any(
            shift.trailer == 1
            and operation.point == 2
            and operation.arrival <= 10
            for shift in candidate.shifts
            for operation in shift.operations
        )
        for candidate in candidates
    )
def test_pressure_band_block_exchanges_two_customer_fragments(monkeypatch) -> None:
    windows = (TimeWindow(0, 600),)
    matrix = tuple(
        tuple(0 if left == right else 10 for right in range(5))
        for left in range(5)
    )
    instance = Instance(
        name="pressure-fragment-exchange",
        unit=60,
        horizon=10,
        time_matrix=matrix,
        distance_matrix=tuple(tuple(float(value) for value in row) for row in matrix),
        base_index=0,
        drivers=(
            Driver(0, 0, 600, (0,), windows, 600, 0.0, 0.0),
            Driver(1, 0, 600, (1,), windows, 600, 0.0, 0.0),
        ),
        trailers=(Trailer(0, 100.0, 100.0, 0.0), Trailer(1, 100.0, 100.0, 0.0)),
        sources=(Source(0, (0, 1), 0),),
        customers=tuple(
            Customer(point, False, False, (), 0, windows, (0, 1),
                     (0.0,) * 10, 100.0, 0.0, 1.0, 0.0)
            for point in range(1, 5)
        ),
    )
    solution = Solution((
        Shift(0, 0, 0, 0, (Operation(1, 40, 10.0), Operation(3, 60, 10.0))),
        Shift(1, 1, 1, 150, (Operation(2, 200, 10.0), Operation(4, 220, 10.0))),
    ))
    monkeypatch.setattr(
        "vrp_solver.solver.surgical_search.pressure_points",
        lambda *_args, **_kwargs: [
            SimpleNamespace(customer=2, first_minute=100, deficit_area=3_000.0),
        ],
    )
    monkeypatch.setattr(
        "vrp_solver.solver.surgical_search.try_optimize_shift_times",
        lambda _instance, shift, **_kwargs: Shift(
            shift.index, shift.driver, shift.trailer, shift.start,
            tuple(
                Operation(operation.point, shift.start + 10 * (index + 1), operation.quantity)
                for index, operation in enumerate(shift.operations)
            ),
        ),
    )

    candidates = _pressure_band_resource_block_candidates(
        instance,
        solution,
        SurgicalSearchConfig(end_day=1, pressure_customers=1, candidates_per_move=16),
        random.Random(13),
    )

    assert any(
        any(
            [operation.point for operation in shift.operations] == [2, 4]
            and shift.start == 0
            for shift in candidate.shifts
        )
        for candidate in candidates
    )
    assert any(
        sorted(
            operation.point
            for shift in candidate.shifts
            for operation in shift.operations
        ) == [2, 3, 4]
        for candidate in candidates
    )
    assert any(
        sum(
            operation.point == 2
            for shift in candidate.shifts
            for operation in shift.operations
        ) == 2
        for candidate in candidates
    )


def test_pressure_band_block_builds_three_route_ejection_cycle(monkeypatch) -> None:
    windows = (TimeWindow(0, 600),)
    matrix = tuple(
        tuple(0 if left == right else 10 for right in range(4))
        for left in range(4)
    )
    instance = Instance(
        name="pressure-three-route-cycle",
        unit=60,
        horizon=10,
        time_matrix=matrix,
        distance_matrix=tuple(tuple(float(value) for value in row) for row in matrix),
        base_index=0,
        drivers=tuple(
            Driver(index, 0, 600, (index,), windows, 600, 0.0, 0.0)
            for index in range(3)
        ),
        trailers=tuple(Trailer(index, 100.0, 100.0, 0.0) for index in range(3)),
        sources=(Source(0, (0, 1, 2), 0),),
        customers=tuple(
            Customer(point, False, False, (), 0, windows, (0, 1, 2),
                     (0.0,) * 10, 100.0, 0.0, 1.0, 0.0)
            for point in range(1, 4)
        ),
    )
    solution = Solution((
        Shift(0, 0, 0, 150, (Operation(2, 200, 10.0),)),
        Shift(1, 1, 1, 0, (Operation(1, 50, 10.0),)),
        Shift(2, 2, 2, 160, (Operation(3, 210, 10.0),)),
    ))
    monkeypatch.setattr(
        "vrp_solver.solver.surgical_search.pressure_points",
        lambda *_args, **_kwargs: [
            SimpleNamespace(customer=2, first_minute=100, deficit_area=4_000.0),
        ],
    )
    monkeypatch.setattr(
        "vrp_solver.solver.surgical_search.try_optimize_shift_times",
        lambda _instance, shift, **_kwargs: Shift(
            shift.index, shift.driver, shift.trailer, shift.start,
            tuple(
                Operation(operation.point, shift.start + 10 * (index + 1), operation.quantity)
                for index, operation in enumerate(shift.operations)
            ),
        ),
    )

    candidates = _pressure_band_resource_block_candidates(
        instance,
        solution,
        SurgicalSearchConfig(end_day=1, pressure_customers=1, candidates_per_move=32),
        random.Random(17),
    )

    assert any(
        sorted(
            (shift.trailer, shift.operations[0].point)
            for shift in candidate.shifts
        ) == [(0, 3), (1, 2), (2, 1)]
        for candidate in candidates
    )


def test_pressure_band_block_prices_missing_vmi_topology(monkeypatch) -> None:
    windows = (TimeWindow(0, 600),)
    matrix = ((0, 10, 10), (10, 0, 10), (10, 10, 0))
    instance = Instance(
        name="missing-pressure-topology",
        unit=60,
        horizon=10,
        time_matrix=matrix,
        distance_matrix=tuple(tuple(float(value) for value in row) for row in matrix),
        base_index=0,
        drivers=(Driver(0, 0, 600, (0,), windows, 600, 0.0, 0.0),),
        trailers=(Trailer(0, 100.0, 100.0, 0.0),),
        sources=(Source(0, (0,), 0),),
        customers=(
            Customer(1, False, False, (), 0, windows, (0,),
                     (0.0,) * 10, 100.0, 0.0, 1.0, 0.0),
            Customer(2, False, False, (), 0, windows, (0,),
                     (0.0,) * 10, 100.0, 0.0, 5.0, 0.0),
        ),
    )
    solution = Solution((
        Shift(0, 0, 0, 0, (Operation(1, 50, 10.0),)),
    ))
    monkeypatch.setattr(
        "vrp_solver.solver.surgical_search.pressure_points",
        lambda *_args, **_kwargs: [
            SimpleNamespace(customer=2, first_minute=100, deficit_area=5_000.0),
        ],
    )
    monkeypatch.setattr(
        "vrp_solver.solver.surgical_search.try_optimize_shift_times",
        lambda _instance, shift, **_kwargs: shift,
    )

    candidates = _pressure_band_resource_block_candidates(
        instance,
        solution,
        SurgicalSearchConfig(end_day=1, pressure_customers=1, candidates_per_move=8),
        random.Random(19),
    )

    assert any(
        candidate.shifts[0].operations[0].point == 2
        and candidate.shifts[0].operations[0].quantity == 5.0
        for candidate in candidates
    )
