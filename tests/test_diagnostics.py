from __future__ import annotations

from dataclasses import replace

import pytest

from vrp_solver.diagnostics import (
    assess_atomic_repair, solution_fingerprint, violation_vector,
)
from vrp_solver.model import (
    Customer, Driver, Instance, Operation, Order, Shift, Solution, Source,
    TimeWindow, Trailer,
)
from vrp_solver.highs_time_opt import (
    latest_end_before_successors, try_optimize_shift_times,
)
from vrp_solver.rules import derive_solution, validate_solution


@pytest.fixture
def instance() -> Instance:
    return Instance(
        name="diagnostic",
        unit=60,
        horizon=3,
        time_matrix=((0, 10), (10, 0)),
        distance_matrix=((0.0, 1.0), (1.0, 0.0)),
        base_index=0,
        drivers=(Driver(0, 0, 120, (0,), (TimeWindow(0, 240),), 60, 0.0, 0.0),),
        trailers=(Trailer(0, 1_000.0, 1_000.0, 0.0),),
        sources=(Source(0, (0,), 0),),
        customers=(
            Customer(
                1, False, False, (), 0, (TimeWindow(0, 240),), (0,),
                (100.0, 100.0, 100.0), 1_000.0, 150.0, 1.0, 100.0,
            ),
        ),
    )


def test_violation_vector_measures_inventory_amount_duration(instance: Instance) -> None:
    vector = violation_vector(instance, Solution(shifts=()))

    assert vector.negative_quantity_minutes == pytest.approx((50.0 + 150.0) * 60)
    assert vector.safety_deficit_quantity_minutes == pytest.approx((50.0 + 150.0 + 250.0) * 60)
    assert not vector.locally_feasible


def test_violation_vector_measures_callin_deficit(instance: Instance) -> None:
    callin = replace(
        instance.customers[0], call_in=True, forecast=(0.0, 0.0, 0.0),
        orders=(Order(500.0, 0, 120, 80),),
    )
    callin_instance = replace(instance, customers=(callin,))

    vector = violation_vector(callin_instance, Solution(shifts=()))

    assert vector.missed_orders == 1
    assert vector.missed_order_deficit == pytest.approx(400.0)


def test_fingerprint_covers_topology_timing_and_quantity(instance: Instance) -> None:
    shift = Shift(0, 0, 0, 0, (Operation(1, 10, 100.0),))
    base = Solution((shift,))

    assert solution_fingerprint(base) == solution_fingerprint(base)
    assert solution_fingerprint(base) != solution_fingerprint(
        Solution((replace(shift, start=1),)),
    )
    assert solution_fingerprint(base) != solution_fingerprint(
        Solution((replace(shift, operations=(replace(shift.operations[0], quantity=101.0),)),)),
    )


def test_non_finite_quantity_fails_closed(instance: Instance) -> None:
    solution = Solution((Shift(0, 0, 0, 0, (Operation(1, 10, float("nan")),)),))

    vector = violation_vector(instance, solution)

    assert vector.non_finite_values == 1
    assert not vector.locally_feasible
    with pytest.raises(ValueError):
        solution_fingerprint(solution)


def test_atomic_repair_rejects_moving_order_error_to_timing(instance: Instance) -> None:
    callin = replace(
        instance.customers[0], call_in=True, forecast=(0.0, 0.0, 0.0),
        orders=(Order(100.0, 0, 120, 100),),
    )
    callin_instance = replace(instance, customers=(callin,))
    incumbent = Solution(shifts=())
    # Covers the order, but starts outside the driver's window.
    candidate = Solution((Shift(0, 0, 0, -20, (Operation(1, 10, 100.0),)),))

    decision = assess_atomic_repair(callin_instance, incumbent, candidate)

    assert not decision.accepted
    assert decision.reason == "regressed:resource_timing_errors"


def test_atomic_repair_accepts_order_coverage_without_regression(instance: Instance) -> None:
    callin = replace(
        instance.customers[0], call_in=True, forecast=(0.0, 0.0, 0.0),
        orders=(Order(100.0, 0, 120, 100),),
    )
    callin_instance = replace(instance, customers=(callin,))
    incumbent = Solution(shifts=())
    candidate = Solution((Shift(0, 0, 0, 0, (Operation(1, 10, 100.0),)),))

    decision = assess_atomic_repair(callin_instance, incumbent, candidate)

    assert decision.accepted
    assert decision.reason == "strict_improvement"


def test_timing_optimizer_can_represent_layover_after_eligible_customer() -> None:
    instance = Instance(
        name="layover-retime",
        unit=60,
        horizon=30,
        time_matrix=((0, 60, 60), (60, 0, 60), (60, 60, 0)),
        distance_matrix=((0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)),
        base_index=0,
        drivers=(Driver(0, 0, 150, (0,), (TimeWindow(0, 1_800),), 120, 0.0, 0.0),),
        trailers=(Trailer(0, 1_000.0, 1_000.0, 0.0),),
        sources=(Source(0, (0,), 0),),
        customers=(
            Customer(1, True, False, (), 0, (TimeWindow(0, 1_800),), (0,), (0.0,) * 30, 1_000.0, 0.0, 1.0, 0.0),
            Customer(2, False, False, (), 0, (TimeWindow(0, 1_800),), (0,), (0.0,) * 30, 1_000.0, 0.0, 1.0, 0.0),
        ),
    )
    shift = Shift(0, 0, 0, 0, (Operation(1, 60, 1.0), Operation(2, 120, 1.0)))

    repaired = try_optimize_shift_times(instance, shift)

    assert repaired is not None
    assert any(operation.layover_before for operation in derive_solution(
        instance, Solution((repaired,)),
    )[0].operations)
    assert not [v for v in validate_solution(instance, Solution((repaired,))) if v.code == "DRI03"]


def test_timing_optimizer_relaxes_inactive_windows_across_long_horizon() -> None:
    instance = Instance(
        name="long-horizon-retime", unit=60, horizon=840,
        time_matrix=((0, 10), (10, 0)),
        distance_matrix=((0.0, 1.0), (1.0, 0.0)), base_index=0,
        drivers=(Driver(0, 0, 120, (0,), (TimeWindow(0, 1_000),), 1_000, 0.0, 0.0),),
        trailers=(Trailer(0, 1_000.0, 1_000.0, 0.0),),
        sources=(Source(0, (0,), 0),),
        customers=(Customer(
            1, False, False, (), 0,
            (TimeWindow(100, 200), TimeWindow(45_000, 45_100)),
            (0,), (0.0,) * 840, 1_000.0, 0.0, 1.0, 0.0,
        ),),
    )
    shift = Shift(0, 0, 0, 0, (Operation(1, 150, 1.0),))

    repaired = try_optimize_shift_times(instance, shift)

    assert repaired is not None
    assert 100 <= repaired.operations[0].arrival <= 200


def test_timing_optimizer_can_wait_for_first_layover_customer() -> None:
    instance = Instance(
        name="first-stop-layover", unit=60, horizon=40,
        time_matrix=((0, 273), (273, 0)),
        distance_matrix=((0.0, 1.0), (1.0, 0.0)), base_index=0,
        drivers=(Driver(0, 0, 535, (0,), (TimeWindow(0, 2_000),), 600, 0.0, 0.0),),
        trailers=(Trailer(0, 1_000.0, 1_000.0, 0.0),),
        sources=(Source(0, (0,), 0),),
        customers=(Customer(
            1, True, False, (), 30, (TimeWindow(900, 1_200),),
            (0,), (0.0,) * 40, 1_000.0, 0.0, 1.0, 0.0,
        ),),
    )
    shift = Shift(0, 0, 0, 0, (Operation(1, 900, 1.0),))

    repaired = try_optimize_shift_times(instance, shift)

    assert repaired is not None
    assert repaired.operations[0].arrival >= 873
    errors = [
        violation for violation in validate_solution(instance, Solution((repaired,)))
        if violation.code in {"DRI03", "LAY02", "LAY03"}
    ]
    assert not errors


def test_successor_boundary_includes_driver_rest_and_trailer_availability(
    instance: Instance,
) -> None:
    driver = replace(instance.drivers[0], min_inter_shift_duration=30)
    bounded = replace(instance, drivers=(driver,))
    solution = Solution((
        Shift(10, 0, 0, 0, (Operation(1, 10, 1.0),)),
        Shift(11, 0, 0, 100, (Operation(1, 110, 1.0),)),
    ))

    assert latest_end_before_successors(bounded, solution, 10) == 70
