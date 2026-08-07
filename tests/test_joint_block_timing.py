from __future__ import annotations

from vrp_solver.joint_block_timing import (
    generate_pressure_block_insertions,
    generate_pressure_block_substitutions,
    retime_connected_resource_block,
    retime_resource_blocks,
)
from vrp_solver.model import (
    Customer, Driver, Instance, Operation, Shift, Solution, Source,
    TimeWindow, Trailer,
)
from vrp_solver.rules import validate_solution
from vrp_solver.solver.surgical_search import (
    SurgicalSearchConfig,
    _candidate_frontier,
    _compaction_order_starts,
    _early_pressure_insertion_target,
    _resource_safe_created_candidates,
)


def test_joint_block_timing_moves_a_resource_successor_with_its_predecessor() -> None:
    """A connected block repairs rest without freezing the successor start."""
    instance = Instance(
        name="joint-block",
        unit=60,
        horizon=3,
        time_matrix=((0, 10), (10, 0)),
        distance_matrix=((0.0, 1.0), (1.0, 0.0)),
        base_index=0,
        drivers=(Driver(0, 30, 120, (0,), (TimeWindow(0, 240),), 60, 0.0, 0.0),),
        trailers=(Trailer(0, 1_000.0, 1_000.0, 0.0),),
        sources=(Source(0, (0,), 0),),
        customers=(Customer(1, False, False, (), 0, (TimeWindow(0, 240),), (0,), (0.0,) * 3, 1_000.0, 1_000.0, 1.0, 0.0),),
    )
    solution = Solution((
        Shift(10, 0, 0, 0, (Operation(1, 10, 1.0),)),
        # This start conflicts with the first route's return plus mandatory
        # inter-shift rest. A one-shift retime would use it as a fixed bound.
        Shift(11, 0, 0, 40, (Operation(1, 50, 1.0),)),
    ))

    repaired = retime_connected_resource_block(instance, solution, 10)

    assert repaired is not None
    second = next(shift for shift in repaired.shifts if shift.index == 11)
    # First route: arrival 10 + return 10 + mandatory rest 30.
    assert second.start >= 50
    assert not [
        violation for violation in validate_solution(instance, repaired)
        if violation.code in {"DRI01", "TL01", "DRI08"}
    ]


def test_compact_joint_timing_reclaims_internal_idle_time() -> None:
    """Compaction can create a resource slot that fixed-interval placement misses."""
    instance = Instance(
        name="joint-compaction",
        unit=60,
        horizon=3,
        time_matrix=((0, 10), (10, 0)),
        distance_matrix=((0.0, 1.0), (1.0, 0.0)),
        base_index=0,
        drivers=(Driver(0, 0, 120, (0,), (TimeWindow(0, 240),), 60, 0.0, 0.0),),
        trailers=(Trailer(0, 1_000.0, 1_000.0, 0.0),),
        sources=(Source(0, (0,), 0),),
        customers=(Customer(1, False, False, (), 0, (TimeWindow(0, 240),), (0,), (0.0,) * 3, 1_000.0, 1_000.0, 1.0, 0.0),),
    )
    solution = Solution((
        # This route occupies [0, 120] only because it waits until minute 100.
        Shift(10, 0, 0, 0, (Operation(1, 100, 1.0),)),
        Shift(11, 0, 0, 60, (Operation(1, 70, 1.0),)),
    ))

    compacted = retime_resource_blocks(
        instance, solution, (11,), max_shifts=2, compact=True,
    )

    assert compacted is not None
    first = next(shift for shift in compacted.shifts if shift.index == 10)
    assert first.operations[0].arrival == 10
    assert not [
        violation for violation in validate_solution(instance, compacted)
        if violation.code in {"DRI01", "TL01", "DRI08"}
    ]


def test_created_shift_can_use_slot_released_by_compaction() -> None:
    instance = Instance(
        name="create-with-compaction",
        unit=60,
        horizon=3,
        time_matrix=((0, 10), (10, 0)),
        distance_matrix=((0.0, 1.0), (1.0, 0.0)),
        base_index=0,
        drivers=(Driver(0, 0, 120, (0,), (TimeWindow(0, 240),), 60, 0.0, 0.0),),
        trailers=(Trailer(0, 1_000.0, 1_000.0, 0.0),),
        sources=(Source(0, (0,), 0),),
        customers=(Customer(1, False, False, (), 0, (TimeWindow(0, 240),), (0,), (0.0,) * 3, 1_000.0, 1_000.0, 1.0, 0.0),),
    )
    incumbent = Solution((
        Shift(10, 0, 0, 0, (Operation(1, 100, 1.0),)),
    ))
    created = Solution((
        *incumbent.shifts,
        Shift(1, 0, 0, 60, (Operation(1, 70, 1.0),)),
    ))

    candidates = _resource_safe_created_candidates(
        instance,
        incumbent,
        [created],
        SurgicalSearchConfig(end_day=1),
    )

    assert len(candidates) == 1
    assert len(candidates[0].shifts) == 2
    assert not [
        violation for violation in validate_solution(instance, candidates[0])
        if violation.code in {"DRI01", "TL01", "DRI08"}
    ]


def test_compaction_enumerates_nearby_resource_orders() -> None:
    solution = Solution((
        Shift(10, 0, 0, 100, ()),
        Shift(11, 0, 1, 300, ()),
        Shift(12, 1, 0, 500, ()),
        Shift(13, 1, 1, 700, ()),
        Shift(14, 2, 2, 900, ()),
    ))
    created = Shift(99, 0, 0, 600, ())

    starts = _compaction_order_starts(solution, created)

    assert starts == (600, 499, 500, 299, 300, 99, 100)


def test_pressure_block_insertion_adds_an_early_vmi_visit() -> None:
    instance = Instance(
        name="joint-insert",
        unit=60,
        horizon=4,
        time_matrix=((0, 10, 10), (10, 0, 10), (10, 10, 0)),
        distance_matrix=((0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)),
        base_index=0,
        drivers=(Driver(0, 0, 120, (0,), (TimeWindow(0, 300),), 60, 0.0, 0.0),),
        trailers=(Trailer(0, 1_000.0, 1_000.0, 0.0),),
        sources=(Source(0, (0,), 0),),
        customers=(
            Customer(1, False, False, (), 0, (TimeWindow(0, 300),), (0,), (0.0,) * 4, 1_000.0, 1_000.0, 1.0, 0.0),
            Customer(2, False, False, (), 0, (TimeWindow(0, 300),), (0,), (0.0,) * 4, 1_000.0, 1_000.0, 1.0, 0.0),
        ),
    )
    solution = Solution((
        Shift(10, 0, 0, 0, (Operation(1, 10, 1.0),)),
        Shift(11, 0, 0, 80, (Operation(2, 90, 1.0),)),
    ))

    candidates = generate_pressure_block_insertions(
        instance, solution, customer_point=2, first_minute=50,
    )

    assert candidates
    assert any(
        any(operation.point == 2 and operation.arrival <= 50 for operation in shift.operations)
        for candidate in candidates for shift in candidate.shifts
    )


def test_pressure_repair_funnel_recognises_an_added_vmi_visit() -> None:
    before = Solution((Shift(0, 0, 0, 0, (Operation(1, 10, 1.0),)),))
    candidate = Solution((Shift(0, 0, 0, 0, (
        Operation(1, 10, 1.0), Operation(2, 20, 1.0),
    )),))

    assert _early_pressure_insertion_target(before, candidate, {2}) == 2


def test_pressure_block_substitution_replaces_optional_vmi_stop() -> None:
    instance = Instance(
        name="joint-substitute", unit=60, horizon=4,
        time_matrix=((0, 10, 10), (10, 0, 10), (10, 10, 0)),
        distance_matrix=((0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)),
        base_index=0,
        drivers=(Driver(0, 0, 120, (0,), (TimeWindow(0, 300),), 60, 0.0, 0.0),),
        trailers=(Trailer(0, 1_000.0, 1_000.0, 0.0),), sources=(Source(0, (0,), 0),),
        customers=(
            Customer(1, False, False, (), 0, (TimeWindow(0, 300),), (0,), (0.0,) * 4, 1_000.0, 1_000.0, 1.0, 0.0),
            Customer(2, False, False, (), 0, (TimeWindow(0, 300),), (0,), (0.0,) * 4, 1_000.0, 1_000.0, 1.0, 0.0),
        ),
    )
    solution = Solution((Shift(0, 0, 0, 0, (Operation(1, 10, 1.0),)),))

    candidates = generate_pressure_block_substitutions(
        instance, solution, customer_point=2, first_minute=50,
    )

    assert candidates
    assert candidates[0].shifts[0].operations[0].point == 2


def test_candidate_frontier_keeps_late_topology_families_in_budget() -> None:
    candidates = list(range(12))

    frontier = _candidate_frontier(candidates, budget=6, stagnation=0)

    assert len(frontier) == 6
    assert frontier[:4] == [0, 1, 2, 3]
    assert frontier[-1] == 11
