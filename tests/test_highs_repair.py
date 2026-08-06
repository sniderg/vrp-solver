from __future__ import annotations

from vrp_solver.highs_repair import repair_quantities_with_highs
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


def test_min_delivered_cannot_deactivate_call_in_visit() -> None:
    windows = (TimeWindow(0, 240),)
    instance = Instance(
        name="mandatory-call-in-activation",
        unit=60,
        horizon=4,
        time_matrix=((0, 10), (10, 0)),
        distance_matrix=((0.0, 1.0), (1.0, 0.0)),
        base_index=0,
        drivers=(Driver(0, 0, 240, (0,), windows, 240, 0.0, 0.0),),
        trailers=(Trailer(0, 10_000.0, 0.0, 0.0),),
        sources=(Source(0, (0,), 0),),
        customers=(
            Customer(
                1,
                False,
                True,
                (Order(1_000.0, 0, 180, 100),),
                0,
                windows,
                (0,),
                (0.0,) * 4,
                10_000.0,
                0.0,
                1.0,
                0.0,
            ),
        ),
    )
    solution = Solution((
        Shift(0, 0, 0, 0, (
            Operation(0, 0, -1_000.0),
            Operation(1, 10, 1_000.0),
        )),
    ))

    repaired, report = repair_quantities_with_highs(
        instance,
        solution,
        score_days=1,
        feasibility_days=1,
        quantity_objective="min-delivered",
    )

    assert report.status == "Optimal"
    deliveries = [
        operation.quantity
        for shift in repaired.shifts
        for operation in shift.operations
        if operation.point == 1
    ]
    assert deliveries == [1_000.0]
