from __future__ import annotations

from vrp_solver.model import Customer, Driver, Instance, Order, Solution, Source, TimeWindow, Trailer
from vrp_solver.diagnostics import ViolationVector
from vrp_solver.solver.surgical_search import (
    _hard_invariants_not_worse,
    _score,
    _search_violation_vector,
)


def _vector(**changes) -> ViolationVector:
    values = dict(
        non_finite_values=0,
        reference_errors=0,
        physical_errors=0,
        missed_orders=0,
        missed_order_deficit=0.0,
        negative_quantity_minutes=0.0,
        overfill_quantity_minutes=0.0,
        safety_deficit_quantity_minutes=1_000.0,
        resource_timing_errors=0,
        other_errors=0,
    )
    values.update(changes)
    return ViolationVector(**values)


def test_safety_improvement_cannot_reintroduce_runout() -> None:
    before = _vector(safety_deficit_quantity_minutes=1_000.0)
    after = _vector(
        physical_errors=1,
        negative_quantity_minutes=60.0,
        safety_deficit_quantity_minutes=100.0,
    )

    assert not _hard_invariants_not_worse(before, after)


def test_safety_improvement_preserving_hard_invariants_is_allowed() -> None:
    before = _vector(safety_deficit_quantity_minutes=1_000.0)
    after = _vector(safety_deficit_quantity_minutes=100.0)

    assert _hard_invariants_not_worse(before, after)


def test_missed_call_in_order_is_hard_for_surgical_acceptance() -> None:
    instance = Instance(
        name="missed-call-in",
        unit=60,
        horizon=4,
        time_matrix=((0, 10), (10, 0)),
        distance_matrix=((0.0, 1.0), (1.0, 0.0)),
        base_index=0,
        drivers=(Driver(0, 0, 120, (0,), (TimeWindow(0, 240),), 60, 0.0, 0.0),),
        trailers=(Trailer(0, 10_000.0, 0.0, 0.0),),
        sources=(Source(0, (0,), 0),),
        customers=(
            Customer(
                1, False, True,
                (Order(1_000.0, 0, 120, 100),),
                0, (TimeWindow(0, 240),), (0,),
                (0.0,) * 4, 10_000.0, 0.0, 1.0, 0.0,
            ),
        ),
    )

    score = _score(instance, Solution(shifts=()), end_day=1)

    assert score.feasibility_errors == 1
    assert score.hard_violations == 1


def test_vmi_safety_breach_is_hard_for_surgical_acceptance() -> None:
    instance = Instance(
        name="safety-breach",
        unit=60,
        horizon=4,
        time_matrix=((0, 10), (10, 0)),
        distance_matrix=((0.0, 1.0), (1.0, 0.0)),
        base_index=0,
        drivers=(Driver(0, 0, 120, (0,), (TimeWindow(0, 240),), 60, 0.0, 0.0),),
        trailers=(Trailer(0, 10_000.0, 0.0, 0.0),),
        sources=(Source(0, (0,), 0),),
        customers=(
            Customer(1, False, False, (), 0, (TimeWindow(0, 240),), (0,),
                     (100.0,) * 4, 1_000.0, 400.0, 1.0, 300.0),
        ),
    )

    score = _score(instance, Solution(shifts=()), end_day=1)

    assert score.feasibility_errors == 3
    assert score.hard_violations == 3


def test_rolling_violation_vector_ignores_uncommitted_tail() -> None:
    instance = Instance(
        name="rolling-prefix-vector",
        unit=60,
        horizon=48,
        time_matrix=((0, 10), (10, 0)),
        distance_matrix=((0.0, 1.0), (1.0, 0.0)),
        base_index=0,
        drivers=(),
        trailers=(),
        sources=(),
        customers=(
            Customer(
                1, False, False, (), 0, (TimeWindow(0, 2_880),), (),
                (30.0,) * 48, 2_000.0, 1_000.0, 1.0, 100.0,
            ),
        ),
    )

    prefix = _search_violation_vector(instance, Solution(()), end_day=1)
    full = _search_violation_vector(instance, Solution(()), end_day=2)

    assert prefix.locally_feasible
    assert not full.locally_feasible
