"""The QS01 nominal ceiling, and the empty-shift output defect.

Both are facts about the *released checker* that the internal scorers did not
model, found by running it rather than by reading the rules:

1. A call-in order's nominal quantity is a ceiling. Over-delivering an order
   makes the checker report it as a **missed** order. Measured on V2.24 order 0
   of customer 6 (flexible minimum 9600, nominal 12000) by writing four
   solutions that differ in that one operation's quantity and verifying each:
   9600, 10800 and 12000 all pass; 12001 produces
   ``[ checkQS01 MissedOrder ] : Missed Order[0] of the customer[6]``.

   This mattered: the search drove V2.15 to zero errors by its own scoring
   while the checker rejected it with three QS01 MissedOrder lines, because
   over-filling a call-in order bought logistic ratio for free.

2. The checker throws an unhandled .NET exception (0xE0434352, reported as
   ``official_status,execution_failed``) on an ``<operations />`` element.
   Operators empty shifts rather than removing them, and the official
   truncation drops operation-free shifts before scoring, so nothing internal
   ever noticed.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vrp_solver.rules import validate_solution  # noqa: E402
from vrp_solver.xml_io import (  # noqa: E402
    Operation,
    Shift,
    Solution,
    load_instance,
    load_solution,
    save_solution,
)

from test_fast_state import AVAILABLE  # noqa: E402


pytestmark = pytest.mark.skipif(not AVAILABLE, reason="Set B artifacts absent")


def _callin_orders(instance):
    """Every (customer, order) pair the QS01 check actually applies to."""
    latest_required = instance.horizon * instance.unit
    for customer in instance.customers:
        if not customer.call_in:
            continue
        for order_index, order in enumerate(customer.orders):
            if order.latest_time <= latest_required:
                yield customer, order_index, order


def _qs01_errors(instance, solution):
    return [
        v
        for v in validate_solution(instance, solution)
        if v.code == "QS01" and v.severity == "error"
    ]


def _first_pair_with_a_callin_order():
    for name, instance_path, solution_path in AVAILABLE:
        instance = load_instance(instance_path)
        if any(True for _ in _callin_orders(instance)):
            return name, instance, load_solution(solution_path)
    pytest.skip("no available instance defines a scored call-in order")


# -- the ceiling ------------------------------------------------------------


def _serve_one_order(instance, customer, order, quantity):
    """A minimal solution delivering ``quantity`` inside ``order``'s window.

    Deliberately not a *valid* solution -- it exists to isolate QS01 from
    every other rule, which is exactly what the checker probes could not do
    (the extra load drained the trailer and buried QS01 under 145 SHI06).
    """
    arrival = order.earliest_time
    return Solution(
        shifts=(
            Shift(
                index=0,
                driver=0,
                trailer=0,
                start=max(0, arrival - 1),
                operations=(
                    Operation(point=customer.index, arrival=arrival, quantity=quantity),
                ),
            ),
        )
    )


def test_delivering_the_flexible_minimum_satisfies_an_order():
    _name, instance, _solution = _first_pair_with_a_callin_order()
    customer, order_index, order = next(_callin_orders(instance))
    solution = _serve_one_order(
        instance, customer, order, order.min_quantity_to_satisfy
    )
    errors = [v for v in _qs01_errors(instance, solution) if v.point == customer.index]
    assert not errors, f"the flexible minimum should satisfy order {order_index}"


def test_delivering_exactly_the_nominal_quantity_satisfies_an_order():
    """12000 passes on the measured V2.24 case: the ceiling is inclusive."""
    _name, instance, _solution = _first_pair_with_a_callin_order()
    customer, _order_index, order = next(_callin_orders(instance))
    solution = _serve_one_order(instance, customer, order, order.quantity)
    errors = [v for v in _qs01_errors(instance, solution) if v.point == customer.index]
    assert not errors, "the nominal quantity is the ceiling, so it must be allowed"


def test_over_delivering_a_callin_order_is_a_missed_order():
    """The unmodelled rule. 12001 fails on the measured V2.24 case."""
    _name, instance, _solution = _first_pair_with_a_callin_order()
    customer, order_index, order = next(_callin_orders(instance))
    solution = _serve_one_order(instance, customer, order, order.quantity * 1.5)
    errors = [v for v in _qs01_errors(instance, solution) if v.point == customer.index]
    assert errors, (
        f"over-delivering order {order_index} of customer {customer.index} must "
        "count as a missed order -- otherwise the search buys logistic ratio for free"
    )


def test_under_delivering_the_flexible_minimum_is_still_a_missed_order():
    """The pre-existing lower bound must survive the new upper bound."""
    _name, instance, _solution = _first_pair_with_a_callin_order()
    customer, _order_index, order = next(_callin_orders(instance))
    solution = _serve_one_order(
        instance, customer, order, order.min_quantity_to_satisfy * 0.5
    )
    assert [v for v in _qs01_errors(instance, solution) if v.point == customer.index]


@pytest.mark.parametrize("name,instance_path,solution_path", AVAILABLE)
def test_the_fast_state_and_the_reference_scorer_agree_on_the_ceiling(
    name, instance_path, solution_path
):
    """The bound has to land in both scorers, or the search optimises a fiction.

    Doubling every call-in delivery makes the ceiling bind wherever it can,
    and the fast state's incremental QS01 count must still equal the
    from-scratch reference count.
    """
    from vrp_solver.contest import score_prefix_with_feasibility_tail
    from vrp_solver.fast.state import SearchState, instance_days

    instance = load_instance(instance_path)
    solution = load_solution(solution_path)
    callin_points = {c.index for c in instance.customers if c.call_in}
    if not callin_points:
        pytest.skip(f"{name} defines no call-in customers")

    inflated = Solution(
        shifts=tuple(
            replace(
                shift,
                operations=tuple(
                    replace(op, quantity=op.quantity * 2.0)
                    if op.point in callin_points
                    else op
                    for op in shift.operations
                ),
            )
            for shift in solution.shifts
        )
    )
    days = instance_days(instance)
    fast = SearchState.from_solution(instance, inflated, score_days=days).score()
    reference = score_prefix_with_feasibility_tail(
        instance,
        inflated,
        score_days=days,
        feasibility_days=days,
        ignore_tail_call_ins=True,
    )
    assert fast.feasibility_errors == reference.feasibility_errors


# -- the empty-shift output defect -----------------------------------------


def test_saved_solutions_never_contain_an_operation_free_shift(tmp_path):
    """An ``<operations />`` element crashes the checker, so it must not be written."""
    name, instance_path, solution_path = AVAILABLE[0]
    solution = load_solution(solution_path)
    with_empties = Solution(
        shifts=solution.shifts
        + (Shift(index=999, driver=0, trailer=0, start=0, operations=()),)
    )
    out = tmp_path / "with_empties.xml"
    save_solution(with_empties, out)

    root = ET.parse(out).getroot()
    written = root.findall(".//IRP_Roadef_Challenge_Shift_")
    assert len(written) == sum(1 for s in solution.shifts if s.operations)
    for shift in written:
        operations = shift.find("operations")
        assert operations is not None and len(operations) > 0, (
            "an operation-free shift reached the output XML; the checker "
            "reports execution_failed on it rather than a verdict"
        )


def test_dropping_empty_shifts_reindexes_without_a_gap(tmp_path):
    """Indices are reassigned on write, so dropping a shift must not leave a hole."""
    name, instance_path, solution_path = AVAILABLE[0]
    solution = load_solution(solution_path)
    out = tmp_path / "reindexed.xml"
    save_solution(solution, out)
    root = ET.parse(out).getroot()
    indices = [
        int(s.find("index").text)
        for s in root.findall(".//IRP_Roadef_Challenge_Shift_")
    ]
    assert indices == list(range(len(indices)))


def test_to_solution_can_drop_empty_shifts_for_output():
    """``drop_empty`` exists for the writer; the default keeps positions stable."""
    from vrp_solver.fast.state import SearchState, instance_days

    name, instance_path, solution_path = AVAILABLE[0]
    instance = load_instance(instance_path)
    solution = load_solution(solution_path)
    days = instance_days(instance)
    state = SearchState.from_solution(instance, solution, score_days=days)
    state.set_operations(0, [], [], [])

    kept = state.to_solution()
    dropped = state.to_solution(drop_empty=True)
    assert any(not s.operations for s in kept.shifts), (
        "the default must keep emptied shifts so operator positions stay stable"
    )
    assert all(s.operations for s in dropped.shifts)
    # Emptying a shift must not change the score either way -- the official
    # truncation ignores operation-free shifts.
    assert len(dropped.shifts) < len(kept.shifts)
