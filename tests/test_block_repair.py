from __future__ import annotations

from vrp_solver.diagnostics import violation_vector
from vrp_solver.model import (
    Customer, Driver, Instance, Operation, Shift, Solution, Source, TimeWindow,
    Trailer,
)
from vrp_solver.solver.block_repair import (
    cross_shift_ejection_candidates, segment_reorder_candidates,
    state_preserving_split_candidates,
)


def test_state_preserving_split_keeps_trailer_stock_and_repairs_driving() -> None:
    instance = Instance(
        name="split",
        unit=60,
        horizon=20,
        time_matrix=((0, 80, 80), (80, 0, 80), (80, 80, 0)),
        distance_matrix=((0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)),
        base_index=0,
        drivers=(
            Driver(0, 0, 200, (0,), (TimeWindow(0, 1_200),), 120, 0.0, 0.0),
            Driver(1, 0, 200, (0,), (TimeWindow(0, 1_200),), 120, 0.0, 0.0),
        ),
        trailers=(Trailer(0, 1_000.0, 1_000.0, 0.0),),
        sources=(Source(0, (0,), 0),),
        customers=(
            Customer(1, False, True, (), 0, (TimeWindow(0, 1_200),), (0,), (0.0,) * 20, 1_000.0, 0.0, 1.0, 0.0),
            Customer(2, False, True, (), 0, (TimeWindow(0, 1_200),), (0,), (0.0,) * 20, 1_000.0, 0.0, 1.0, 0.0),
        ),
    )
    original = Solution((Shift(
        0, 0, 0, 0,
        (Operation(1, 80, 100.0), Operation(2, 160, 100.0)),
    ),))
    assert violation_vector(instance, original).resource_timing_errors > 0

    candidates, funnel = state_preserving_split_candidates(instance, original, (0,))

    assert funnel.strict_improvements >= 1
    repaired = candidates[0]
    assert len(repaired.shifts) == 2
    assert {shift.trailer for shift in repaired.shifts} == {0}
    assert [op.quantity for shift in repaired.shifts for op in shift.operations] == [100.0, 100.0]
    assert violation_vector(instance, repaired).resource_timing_errors == 0


def test_segment_reorder_can_move_layover_customer_to_legal_break() -> None:
    instance = Instance(
        name="reorder",
        unit=60,
        horizon=30,
        time_matrix=(
            (0, 80, 80, 80), (80, 0, 80, 20),
            (80, 80, 0, 80), (80, 20, 80, 0),
        ),
        distance_matrix=tuple(tuple(0.0 for _ in range(4)) for _ in range(4)),
        base_index=0,
        drivers=(Driver(0, 0, 220, (0,), (TimeWindow(0, 1_800),), 120, 0.0, 0.0),),
        trailers=(Trailer(0, 1_000.0, 1_000.0, 0.0),),
        sources=(Source(0, (0,), 0),),
        customers=(
            Customer(1, False, True, (), 0, (TimeWindow(0, 1_800),), (0,), (0.0,) * 30, 1_000.0, 0.0, 1.0, 0.0),
            Customer(2, False, True, (), 0, (TimeWindow(0, 1_800),), (0,), (0.0,) * 30, 1_000.0, 0.0, 1.0, 0.0),
            Customer(3, True, True, (), 0, (TimeWindow(0, 1_800),), (0,), (0.0,) * 30, 1_000.0, 0.0, 1.0, 0.0),
        ),
    )
    # 1 -> 2 -> 3 cannot rest before returning; 1 -> 3 -> 2 can rest at 3.
    original = Solution((Shift(0, 0, 0, 0, (
        Operation(1, 80, 1.0), Operation(2, 160, 1.0), Operation(3, 240, 1.0),
    )),))

    candidates, funnel = segment_reorder_candidates(instance, original, (0,))

    assert funnel.strict_improvements >= 1
    assert any(shift.operations[1].point == 3 for candidate in candidates for shift in candidate.shifts)


def test_cross_shift_ejection_repairs_topology_and_quantities() -> None:
    instance = Instance(
        name="ejection",
        unit=60,
        horizon=4,
        time_matrix=((0, 80, 20), (80, 0, 20), (20, 20, 0)),
        distance_matrix=tuple(tuple(0.0 for _ in range(3)) for _ in range(3)),
        base_index=0,
        drivers=(
            Driver(0, 0, 100, (0,), (TimeWindow(0, 240),), 120, 0.0, 0.0),
            Driver(1, 0, 100, (1,), (TimeWindow(0, 240),), 120, 0.0, 0.0),
        ),
        trailers=(Trailer(0, 1_000.0, 200.0, 0.0), Trailer(1, 1_000.0, 200.0, 0.0)),
        sources=(Source(0, (0, 1), 0),),
        customers=(
            Customer(1, False, True, (), 0, (TimeWindow(0, 240),), (0, 1), (0.0,) * 4, 1_000.0, 0.0, 1.0, 0.0),
            Customer(2, True, True, (), 0, (TimeWindow(0, 240),), (0, 1), (0.0,) * 4, 1_000.0, 0.0, 1.0, 0.0),
        ),
    )
    original = Solution((
        Shift(0, 0, 0, 0, (Operation(2, 20, 50.0), Operation(1, 100, 50.0))),
        Shift(1, 1, 1, 0, (Operation(0, 0, -100.0), Operation(2, 20, 50.0))),
    ))
    assert violation_vector(instance, original).resource_timing_errors > 0

    candidates, funnel = cross_shift_ejection_candidates(
        instance, original, (0,), arrival_radius=240,
    )

    assert funnel.strict_improvements >= 1
    assert any(violation_vector(instance, candidate).resource_timing_errors == 0 for candidate in candidates)
