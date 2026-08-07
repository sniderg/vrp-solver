from __future__ import annotations

from vrp_solver.model import Customer, Driver, Instance, Operation, Shift, Solution, Source, TimeWindow, Trailer
from vrp_solver.solver.surgical_search import _rebalance_retailer_after_early_column


def test_joint_retailer_reinsert_offsets_early_delivery_against_later_visit() -> None:
    instance = Instance(
        name="retailer-reinsert",
        unit=60,
        horizon=12,
        time_matrix=((0, 1, 1), (1, 0, 1), (1, 1, 0)),
        distance_matrix=((0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)),
        base_index=0,
        drivers=(Driver(0, 0, 600, (0,), (TimeWindow(0, 720),), 600, 0.0, 0.0),),
        trailers=(Trailer(0, 100.0, 0.0, 0.0),),
        sources=(Source(1, (0,), 0),),
        customers=(Customer(2, False, False, (), 0, (TimeWindow(0, 720),), (0,), (0.0,) * 12, 100.0, 20.0, 10.0, 0.0),),
    )
    incumbent = Solution((
        Shift(0, 0, 0, 0, (Operation(1, 1, -80.0), Operation(2, 100, 80.0))),
    ))
    early = Shift(1, 0, 0, 0, (Operation(1, 1, -30.0), Operation(2, 10, 30.0)))

    candidate = _rebalance_retailer_after_early_column(instance, incumbent, 2, early)

    assert candidate is not None
    deliveries = [
        operation.quantity
        for shift in candidate.shifts
        for operation in shift.operations
        if operation.point == 2
    ]
    assert sorted(deliveries) == [30.0, 50.0]


def test_joint_retailer_reinsert_retimes_a_conflicting_following_route() -> None:
    instance = Instance(
        name="retailer-reinsert-retime",
        unit=60,
        horizon=12,
        time_matrix=((0, 1, 1), (1, 0, 1), (1, 1, 0)),
        distance_matrix=((0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)),
        base_index=0,
        drivers=(Driver(0, 20, 600, (0,), (TimeWindow(0, 720),), 600, 0.0, 0.0),),
        trailers=(Trailer(0, 100.0, 0.0, 0.0),),
        sources=(Source(1, (0,), 0),),
        customers=(Customer(2, False, False, (), 0, (TimeWindow(0, 720),), (0,), (0.0,) * 12, 100.0, 20.0, 10.0, 0.0),),
    )
    incumbent = Solution((
        Shift(0, 0, 0, 20, (Operation(1, 21, -80.0), Operation(2, 100, 80.0))),
    ))
    early = Shift(1, 0, 0, 0, (Operation(1, 1, -30.0), Operation(2, 10, 30.0)))

    candidate = _rebalance_retailer_after_early_column(instance, incumbent, 2, early)

    assert candidate is not None
    later = next(shift for shift in candidate.shifts if shift.start > 0)
    # Early route ends at minute 11; mandatory driver rest makes minute 31
    # the earliest valid start for the original route.
    assert later.start >= 31
