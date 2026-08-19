"""The quantity decoder: quantities are derived from a route, never searched.

TS2020 sec. 3.1 encodes a solution as routes only and evaluates it with two
decoders (timing, then quantity). These tests pin the parts of that decode that
a bug would silently monetize -- every one of them checks a bound whose breach
*improves* the logistic ratio, which is why the internal scorer alone was not
enough to catch the QS01 over-delivery (see ``test_callin_nominal_ceiling.py``).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vrp_solver.fast.decode import (  # noqa: E402
    decode_quantities,
    decode_shift_quantities,
)
from vrp_solver.fast.operators import OPERATORS  # noqa: E402
from vrp_solver.fast.retime import CandidateLists  # noqa: E402
from vrp_solver.fast.state import (  # noqa: E402
    EPSILON,
    _KIND_CUSTOMER,
    _KIND_SOURCE,
    SearchState,
    instance_days,
)
from vrp_solver.rules import validate_solution  # noqa: E402
from vrp_solver.xml_io import load_instance, load_solution  # noqa: E402

from test_fast_state import AVAILABLE  # noqa: E402


def _load(instance_path: str, solution_path: str):
    instance = load_instance(instance_path)
    solution = load_solution(solution_path)
    days = instance_days(instance)
    state = SearchState.from_solution(instance, solution, score_days=days)
    return instance, state, days, CandidateLists(state.fi)


_CASES = [case for case in AVAILABLE if Path(case[2]).exists()]
pytestmark = pytest.mark.skipif(not _CASES, reason="no instance/solution pair present")


# -- the bounds the decoder must never cross --------------------------------


@pytest.mark.parametrize("name,instance_path,solution_path", _CASES, ids=lambda v: str(v))
def test_no_decoded_quantity_exceeds_the_trailer_capacity(
    name, instance_path, solution_path
):
    """A trailer cannot move more than it holds, in one drop or in total."""
    _instance, state, _days, _lists = _load(instance_path, solution_path)
    state.begin()
    decode_quantities(state)
    state.commit()
    for rec in state.shifts:
        capacity = float(state.fi.trailer_capacity[rec.trailer])
        for quantity in rec.quantities:
            assert abs(quantity) <= capacity + 1e-6


@pytest.mark.parametrize("name,instance_path,solution_path", _CASES, ids=lambda v: str(v))
def test_the_decoder_never_drains_the_trailer_below_zero(
    name, instance_path, solution_path
):
    """SHI06 is unreachable by decoding, not merely priced.

    The running load along every route stays non-negative, which is what
    ``_tail_slack`` exists to guarantee: the surplus pass adds to a stop and
    that lowers the load at *every* later stop.
    """
    _instance, state, _days, _lists = _load(instance_path, solution_path)
    state.begin()
    decode_quantities(state)
    state.commit()
    fi = state.fi
    for rec in state.shifts:
        if not rec.points:
            continue
        capacity = float(fi.trailer_capacity[rec.trailer])
        # Walk from the shift's own start-of-route load, reconstructed the same
        # way the decoder does, then check every prefix.
        load = float(fi.trailer_initial[rec.trailer])
        key = (rec.start, rec.index)
        for other in state._by_trailer[rec.trailer]:
            if other is not rec and (other.start, other.index) < key:
                load -= other.net_quantity
        load = min(max(load, 0.0), capacity)
        for quantity in rec.quantities:
            load -= quantity
            assert load >= -1e-6, f"{name}: trailer load went negative ({load})"
            assert load <= capacity + 1e-6, f"{name}: trailer overfilled ({load})"


@pytest.mark.parametrize("name,instance_path,solution_path", _CASES, ids=lambda v: str(v))
def test_the_decoder_never_over_delivers_a_call_in_order(
    name, instance_path, solution_path
):
    """C23, and the defect that made a rejected solution read as valid.

    The nominal quantity is an inclusive ceiling: the released checker reports an
    over-delivered order as a *missed* order. Over-delivery is the single most
    tempting bug here, because it buys logistic-ratio denominator for free.
    """
    instance, state, _days, _lists = _load(instance_path, solution_path)
    state.begin()
    decode_quantities(state)
    state.commit()
    fi = state.fi
    for row, is_call_in in enumerate(fi.cust_is_call_in):
        if not is_call_in:
            continue
        for earliest, latest, quantity, _minimum in fi.cust_qs01_orders[row]:
            total = sum(
                op_quantity
                for op_arrival, op_quantity in state._cust_ops[row]
                if earliest <= op_arrival <= latest and op_quantity > 0.0
            )
            assert total <= float(quantity) + EPSILON, (
                f"{name}: row {row} order [{earliest},{latest}] delivered "
                f"{total} above nominal {quantity}"
            )


@pytest.mark.parametrize("name,instance_path,solution_path", _CASES, ids=lambda v: str(v))
def test_the_decoder_delivers_nothing_outside_an_order_window(
    name, instance_path, solution_path
):
    """C21: no delivery to a call-in customer without an order covering it."""
    _instance, state, _days, _lists = _load(instance_path, solution_path)
    state.begin()
    decode_quantities(state)
    state.commit()
    fi = state.fi
    for row, is_call_in in enumerate(fi.cust_is_call_in):
        if not is_call_in:
            continue
        windows = [
            (earliest, latest)
            for earliest, latest, _q, _m in fi.cust_qs01_orders[row]
        ]
        for arrival, quantity in state._cust_ops[row]:
            if quantity <= EPSILON:
                continue
            assert any(
                earliest <= arrival <= latest for earliest, latest in windows
            ), f"{name}: row {row} got {quantity} at {arrival} with no order"


# -- structure --------------------------------------------------------------


@pytest.mark.parametrize("name,instance_path,solution_path", _CASES, ids=lambda v: str(v))
def test_a_leading_source_loads_the_trailer(name, instance_path, solution_path):
    """The first gap the paper leaves, and the one our first attempt got wrong.

    Deciding a source's load from downstream demand made every route beginning
    at a source load nothing (V2.24: 0 -> 358 errors). Loading is free once the
    trip is committed, so a source fills the trailer.
    """
    _instance, state, _days, _lists = _load(instance_path, solution_path)
    state.begin()
    decode_quantities(state)
    state.commit()
    fi = state.fi
    checked = 0
    for rec in state.shifts:
        if not rec.points or fi.point_kind[rec.points[0]] != _KIND_SOURCE:
            continue
        # A leading source loads unless the trailer arrived already full.
        if rec.quantities[0] < -EPSILON:
            checked += 1
            continue
        assert abs(rec.quantities[0]) <= EPSILON
    assert checked or not any(
        rec.points and fi.point_kind[rec.points[0]] == _KIND_SOURCE
        for rec in state.shifts
    ), f"{name}: every leading source loaded zero"


@pytest.mark.parametrize("name,instance_path,solution_path", _CASES, ids=lambda v: str(v))
def test_the_surplus_pass_raises_delivered_quantity(name, instance_path, solution_path):
    """Leftover load is delivered quantity thrown away, and LR is per unit.

    The direct measurement of what the surplus pass buys: decoding with it
    delivers at least as much as the forward pass alone, on the same routes.
    Compared against the same decode with ``_spend_surplus`` stubbed out, so the
    only difference is the pass itself.
    """
    from vrp_solver.fast import decode as decode_module

    _instance, plain, _days, _lists = _load(instance_path, solution_path)
    original = decode_module._spend_surplus
    decode_module._spend_surplus = lambda *a, **k: None
    try:
        plain.begin()
        decode_quantities(plain)
        plain.commit()
        without = plain.score().scored_delivered_quantity
    finally:
        decode_module._spend_surplus = original

    _instance, full, _days, _lists = _load(instance_path, solution_path)
    full.begin()
    decode_quantities(full)
    full.commit()
    with_pass = full.score().scored_delivered_quantity

    assert with_pass >= without - 1e-6, (
        f"{name}: the surplus pass lost delivered quantity "
        f"({with_pass} < {without})"
    )


@pytest.mark.parametrize("name,instance_path,solution_path", _CASES, ids=lambda v: str(v))
def test_repeated_decoding_is_stable(name, instance_path, solution_path):
    """Repeated decoding does not drift.  It does not always reach a fixed point.

    A single sweep is not idempotent, and that is inherent rather than a bug: a
    customer's tank headroom depends on deliveries made by *other* shifts, so
    decoding shift A moves the bound on shift B, and shifts are decoded in
    trailer-chain order. Most instances then settle exactly -- V2.24 and V2.13
    by the third sweep, V2.14 by the ninth, each sweep lowering both the error
    count and the ratio.

    V2.19 does not settle: it enters a period-2 limit cycle in which two stops
    at one customer hand the same 3,322.4 kg back and forth indefinitely. That
    is bounded rather than divergent -- the error count pins at 1385 and the
    ratio wobbles in the fifth decimal -- and publication ranks by error count,
    which is stable. So the property worth pinning is stability, and the failure
    this guards against is unbounded drift: an earlier version drifted because
    the surplus pass reused ceilings captured during the forward pass, which
    cost V2.24 47 spurious QS02 breaches on the second sweep. See
    ``_spend_surplus``.
    """
    _instance, state, _days, _lists = _load(instance_path, solution_path)
    errors: list[int] = []
    delivered: list[float] = []
    for _sweep in range(14):
        state.begin()
        decode_quantities(state)
        state.commit()
        score = state.score()
        errors.append(score.feasibility_errors)
        delivered.append(score.scored_delivered_quantity)

    # The tail is the settled regime: the first few sweeps legitimately improve.
    tail_errors = errors[-5:]
    tail_delivered = delivered[-5:]
    assert min(tail_errors) == max(tail_errors), (
        f"{name}: error count still moving in the tail: {tail_errors}"
    )
    spread = max(tail_delivered) - min(tail_delivered)
    assert spread <= 0.01 * max(tail_delivered), (
        f"{name}: delivered quantity drifting by {spread:.1f} in the tail"
    )
    # Deliberately *not* asserted: that repeated decoding never raises the error
    # count. It can, and the cause is understood -- the surplus pass fills a
    # customer's tank through one shift, which can leave another shift's stop at
    # the same customer short of its `min_operation_quantity` (SHI16) or push a
    # later step past its safety window (QS02). On V2.24 that is 0 -> 19 errors
    # standalone. Reserving headroom for other stops was tried and rejected: it
    # recovered 1 error of 13 for an O(all shifts) scan in the inner loop.
    #
    # It stays on because the measurement that matters is the search, not a
    # standalone sweep, and there the pass earns its place (60 s, seed 1,
    # published `(errors, LR)`):
    #
    #     inst    surplus=True      surplus=False
    #     V2.15   1  / 0.04794      1  / 0.05640
    #     V2.26   7  / 0.04621      6  / 0.04822
    #     V2.24   0  / 0.01846      0  / 0.02249
    #     V2.13   0  / 0.04655      0  / 0.05588
    #
    # Equal or better errors on three of four and a materially better ratio on
    # all four: the search repairs what the pass breaks, and keeps the delivered
    # quantity it wins. See ``decode.SPEND_SURPLUS``.


@pytest.mark.parametrize("name,instance_path,solution_path", _CASES, ids=lambda v: str(v))
def test_rollback_after_a_decode_is_exact(name, instance_path, solution_path):
    """The decode goes through the state's primitives, so it reverts exactly.

    Exact, not approximate: the staged primitives skip the rescore but still
    snapshot the record, and the snapshot carries its own cached derivation.
    An inexact revert would let error counts drift over a long run.
    """
    _instance, state, _days, _lists = _load(instance_path, solution_path)
    before = state.score()
    state.begin()
    decode_quantities(state)
    state.rollback()
    after = state.score()
    assert before.feasibility_errors == after.feasibility_errors
    assert before.logistic_ratio == after.logistic_ratio
    assert before.scored_delivered_quantity == after.scored_delivered_quantity


@pytest.mark.parametrize("name,instance_path,solution_path", _CASES, ids=lambda v: str(v))
def test_the_fast_score_agrees_with_the_reference_after_decoding(
    name, instance_path, solution_path
):
    """The decode must not open a gap between the two scorers.

    This is the check that caught the missing QS01 ceiling: agreement here is
    what lets an internal error count stand in for a checker verdict.
    """
    instance, state, _days, _lists = _load(instance_path, solution_path)
    state.begin()
    decode_quantities(state)
    state.commit()
    score = state.score()
    violations = validate_solution(instance, state.to_solution(drop_empty=True))
    errors = sum(1 for v in violations if v.severity == "error")
    assert score.feasibility_errors == errors


@pytest.mark.parametrize("name,instance_path,solution_path", _CASES, ids=lambda v: str(v))
def test_decoding_after_a_structural_move_keeps_the_scorers_in_step(
    name, instance_path, solution_path
):
    """The path the search actually takes: move, decode touched shifts, score."""
    instance, state, _days, lists = _load(instance_path, solution_path)
    rng = random.Random(11)
    for _ in range(300):
        state.begin()
        OPERATORS[rng.randrange(len(OPERATORS))][1](state, rng, lists)
        for touched in state.touched_positions():
            decode_shift_quantities(state, touched)
        state.commit()
    score = state.score()
    violations = validate_solution(instance, state.to_solution(drop_empty=True))
    errors = sum(1 for v in violations if v.severity == "error")
    assert score.feasibility_errors == errors


@pytest.mark.parametrize("name,instance_path,solution_path", _CASES, ids=lambda v: str(v))
def test_a_customer_stop_gets_the_least_possible_not_the_most(
    name, instance_path, solution_path
):
    """The paper's rule, stated as a measurement.

    Decoding cannot deliver more in total than a fill-everything policy would,
    and on a real instance it delivers strictly less somewhere -- otherwise
    "least possible" is not being applied at all.
    """
    _instance, state, _days, _lists = _load(instance_path, solution_path)
    fi = state.fi
    state.begin()
    decode_quantities(state)
    state.commit()
    strictly_less = 0
    for rec in state.shifts:
        capacity = float(fi.trailer_capacity[rec.trailer])
        for point, quantity in zip(rec.points, rec.quantities):
            if fi.point_kind[point] != _KIND_CUSTOMER:
                continue
            if quantity < capacity - EPSILON:
                strictly_less += 1
    assert strictly_less, f"{name}: every drop was a full trailer load"
