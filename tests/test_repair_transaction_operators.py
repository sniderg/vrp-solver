from __future__ import annotations

import random

from vrp_solver.model import (
    Customer,
    Driver,
    Instance,
    Operation,
    Order,
    Shift,
    Solution,
    Source,
    TimeWindow,
    Trailer,
)
from vrp_solver.solver.surgical_search import (
    SurgicalSearchConfig,
    _balanced_reload_stop_candidates,
    _merge_lone_reload_candidates,
)


def _instance(customers: tuple[Customer, ...], points: int) -> Instance:
    windows = (TimeWindow(0, 100_000),)
    matrix = tuple(
        tuple(0 if left == right else 10 for right in range(points))
        for left in range(points)
    )
    return Instance(
        name="repair-transactions",
        unit=60,
        horizon=100,
        time_matrix=matrix,
        distance_matrix=tuple(tuple(float(v) for v in row) for row in matrix),
        base_index=0,
        drivers=(Driver(0, 550, 550, (0,), windows, 550, 0.0, 0.0),),
        trailers=(Trailer(0, 100.0, 100.0, 0.0),),
        sources=(Source(1, (0,), 10),),
        customers=customers,
    )


def _customer(point: int, *, call_in: bool = False, orders: tuple[Order, ...] = ()) -> Customer:
    windows = (TimeWindow(0, 100_000),)
    return Customer(
        point, False, call_in, orders, 10, windows, (0,),
        (0.0,) * 100, 200.0, 0.0, 1.0, 0.0,
    )


def test_merge_lone_reload_absorbs_shift_into_predecessor() -> None:
    instance = _instance((_customer(2),), points=3)
    solution = Solution((
        Shift(0, 0, 0, 0, (
            Operation(2, 10, 50.0),
        )),
        # Reload-only shift: every operation is a source load.
        Shift(1, 0, 0, 100, (
            Operation(1, 110, -50.0),
        )),
    ))
    config = SurgicalSearchConfig(end_day=4, candidates_per_move=4)

    candidates = _merge_lone_reload_candidates(instance, solution, config)

    assert candidates, "operator declined an expressible merge"
    merged = candidates[0]
    assert len(merged.shifts) == 1
    tail = merged.shifts[0].operations[-1]
    assert tail.point == 1 and tail.quantity == -50.0
    # Re-timed to the earliest chain-feasible arrival after the predecessor.
    assert tail.arrival == 10 + 10 + 10


def test_merge_lone_reload_requires_same_driver_and_trailer() -> None:
    instance = _instance((_customer(2),), points=3)
    lone = Shift(1, 0, 0, 100, (Operation(1, 110, -50.0),))
    no_predecessor = Solution((lone,))
    config = SurgicalSearchConfig(end_day=4, candidates_per_move=4)

    assert _merge_lone_reload_candidates(instance, no_predecessor, config) == []


def test_balanced_reload_stop_serves_short_call_in_order() -> None:
    order = Order(quantity=60.0, earliest_time=0, latest_time=5_000, quantity_flexibility=80)
    instance = _instance(
        (_customer(2), _customer(3, call_in=True, orders=(order,))),
        points=4,
    )
    # One shift serves customer 2 only; customer 3's order is untouched.
    solution = Solution((
        Shift(0, 0, 0, 0, (
            Operation(2, 10, 50.0),
        )),
    ))
    config = SurgicalSearchConfig(end_day=4, candidates_per_move=8)

    candidates = _balanced_reload_stop_candidates(
        instance, solution, config, random.Random(3),
    )

    assert candidates, "operator declined an expressible insertion"
    floor = order.quantity * order.quantity_flexibility / 100.0
    balanced = []
    for candidate in candidates:
        for shift in candidate.shifts:
            delivered = sum(
                op.quantity for op in shift.operations
                if op.point == 3 and op.quantity > 0
            )
            loaded = -sum(
                op.quantity for op in shift.operations
                if op.point == 1 and op.quantity < 0
            )
            if delivered > floor and abs(loaded - delivered) < 1e-6:
                balanced.append(candidate)
    assert balanced, "no candidate pairs the stop with an equal reload above the floor"


def test_balanced_reload_stop_no_shortfall_no_candidates() -> None:
    instance = _instance((_customer(2),), points=3)
    solution = Solution((
        Shift(0, 0, 0, 0, (Operation(2, 10, 50.0),)),
    ))
    config = SurgicalSearchConfig(end_day=4, candidates_per_move=8)

    assert _balanced_reload_stop_candidates(
        instance, solution, config, random.Random(3),
    ) == []
