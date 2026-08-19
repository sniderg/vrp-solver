"""Tests for the operator portfolio (step 3).

Two obligations per operator (step 3.7): a firing invocation must produce a
*valid mutation* -- one the verified scorer can still score and agree about --
and rollback must restore the state.  A third, the step-3 gate, is the firing
rate itself: an operator that cannot fire is the pathology this rebuild exists
to remove.
"""

from __future__ import annotations

import random
import sys
from hashlib import sha256
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vrp_solver.contest import score_prefix_with_feasibility_tail  # noqa: E402
from vrp_solver.fast.decode import decode_shift_quantities  # noqa: E402
from vrp_solver.fast.objective import GROUPS, Objective  # noqa: E402
from vrp_solver.fast.operators import (  # noqa: E402
    OPERATOR_NAMES,
    OPERATORS,
)
from vrp_solver.fast.retime import (  # noqa: E402
    CandidateLists,
    earliest_arrivals,
)
from vrp_solver.fast.state import SearchState, instance_days  # noqa: E402
from vrp_solver.xml_io import load_instance, load_solution  # noqa: E402

from test_fast_state import AVAILABLE  # noqa: E402

pytestmark = pytest.mark.skipif(not AVAILABLE, reason="Set B artifacts absent")

# One small and one large instance for the per-operator tests, so the suite
# stays quick; the firing-rate gate runs over everything available.
_SMALL = [p for p in AVAILABLE if p[0] in {"V2.13", "V2.24"}] or AVAILABLE[:1]
_GATE = AVAILABLE


def _seed_for(op_name: str) -> int:
    """A stable per-operator seed.

    Not ``hash(op_name)``: Python randomizes string hashing per process, so
    these tests drew a different sample every run. That is how a real
    ``two_opt_star`` weakness went from passing to failing between two runs of
    the same code -- the gate has to be reproducible to mean anything.
    """
    return int.from_bytes(sha256(op_name.encode()).digest()[:4], "big") % 10_000


def _load(instance_path, solution_path):
    instance = load_instance(instance_path)
    solution = load_solution(solution_path)
    days = instance_days(instance)
    state = SearchState.from_solution(instance, solution, score_days=days)
    return instance, state, days, CandidateLists(state.fi)


# -- retiming ---------------------------------------------------------------


@pytest.mark.parametrize("name,instance_path,solution_path", _SMALL)
def test_earliest_arrivals_are_monotone_and_reachable(
    name, instance_path, solution_path
):
    _instance, state, _days, _lists = _load(instance_path, solution_path)
    fi = state.fi
    for rec in state.shifts:
        if not rec.points:
            continue
        arrivals = earliest_arrivals(fi, rec.start, rec.points, rec.driver)
        assert len(arrivals) == len(rec.points)
        last_departure = rec.start
        last_point = fi.base
        for point, arrival in zip(rec.points, arrivals):
            required = last_departure + fi.time_matrix[last_point][point]
            assert arrival >= required, "retiming produced a physically impossible leg"
            last_departure = arrival + fi.setup_time[point]
            last_point = point


@pytest.mark.parametrize("name,instance_path,solution_path", _SMALL)
def test_retiming_never_worsens_shi02(name, instance_path, solution_path):
    """Earliest-arrival retiming should remove arrival-too-early errors.

    This is why ``_apply_route`` retimes by default: without it, every
    resequencing move would carry SHI02 noise created by the move itself.
    """
    _instance, state, _days, _lists = _load(instance_path, solution_path)
    before = state.score().shift_errors
    for position, rec in enumerate(state.shifts):
        if rec.points:
            state.set_operations(
                position,
                list(rec.points),
                earliest_arrivals(state.fi, rec.start, rec.points, rec.driver),
                list(rec.quantities),
            )
    assert state.score().shift_errors <= before


@pytest.mark.parametrize("name,instance_path,solution_path", _SMALL)
def test_candidate_lists_are_nearest_first_and_bounded(
    name, instance_path, solution_path
):
    _instance, state, _days, _lists = _load(instance_path, solution_path)
    fi = state.fi
    lists = CandidateLists(fi, fraction=0.10)
    assert lists.k >= 5
    for point in range(min(fi.n_points, 20)):
        near = lists.near(point)
        assert len(near) == lists.k
        assert point not in near
        row = fi.time_matrix[point]
        times = [row[j] for j in near]
        assert times == sorted(times), "candidate list is not nearest-first"
        # Nothing outside the list may be closer than the furthest inside it.
        outside = [
            row[j] for j in range(fi.n_points) if j != point and j not in set(near)
        ]
        if outside:
            assert min(outside) >= max(times)


# -- per-operator obligations ----------------------------------------------


@pytest.mark.parametrize("op_name,operator", OPERATORS, ids=OPERATOR_NAMES)
@pytest.mark.parametrize(
    "name,instance_path,solution_path", _SMALL, ids=lambda v: str(v)
)
def test_operator_rollback_is_exact(
    op_name, operator, name, instance_path, solution_path
):
    """Step 3.7: every operator's mutation must be undoable."""
    _instance, state, _days, lists = _load(instance_path, solution_path)
    rng = random.Random(_seed_for(op_name))

    before_score = state.score()
    before_solution = state.to_solution()

    fired = 0
    for _ in range(40):
        state.begin()
        if operator(state, rng, lists):
            fired += 1
        state.rollback()
        assert state.score() == before_score, f"{op_name}: score drift after rollback"
        assert state.to_solution() == before_solution, (
            f"{op_name}: structure drift after rollback"
        )
    assert fired > 0, f"{op_name}: never fired in 40 attempts on {name}"


@pytest.mark.parametrize("op_name,operator", OPERATORS, ids=OPERATOR_NAMES)
@pytest.mark.parametrize(
    "name,instance_path,solution_path", _SMALL, ids=lambda v: str(v)
)
def test_operator_output_agrees_with_the_official_scorer(
    op_name, operator, name, instance_path, solution_path
):
    """Step 3.7: an operator may only produce states the scorer still agrees on.

    This is the real safety net.  An operator that builds a structurally odd
    route (empty shift, source-only route, a stop outside every window) is
    allowed -- the objective prices it -- but the incremental score must still
    equal a from-scratch official score, or the search is optimising a fiction.
    """
    instance, state, days, lists = _load(instance_path, solution_path)
    rng = random.Random(_seed_for(op_name) + 1)
    objective = Objective()

    fired = 0
    for _ in range(25):
        if not operator(state, rng, lists):
            continue
        fired += 1
        fast = state.score()
        reference = score_prefix_with_feasibility_tail(
            instance, state.to_solution(),
            score_days=days, feasibility_days=days, ignore_tail_call_ins=True,
        )
        assert fast.feasibility_errors == reference.feasibility_errors, (
            f"{op_name}: error drift after {fired} applications"
        )
        assert fast.scored_estimated_cost == pytest.approx(
            reference.scored_estimated_cost, rel=1e-9, abs=1e-6
        ), f"{op_name}: cost drift after {fired} applications"
        assert fast.scored_delivered_quantity == pytest.approx(
            reference.scored_delivered_quantity, rel=1e-9, abs=1e-6
        ), f"{op_name}: delivered drift after {fired} applications"
        assert fast.safety_kg_min == pytest.approx(
            reference.safety_kg_min, rel=1e-9, abs=1e-6
        ), f"{op_name}: safety drift after {fired} applications"
        # The objective must stay usable on whatever the operator built.
        assert sum(getattr(fast, g) for g in GROUPS) == fast.feasibility_errors
        quality = objective.quality(fast)
        assert quality == quality and quality < float("inf")
    assert fired > 0, f"{op_name}: never fired on {name}"


@pytest.mark.parametrize("op_name,operator", OPERATORS, ids=OPERATOR_NAMES)
def test_operator_survives_an_empty_state(op_name, operator):
    """No operator may raise on a degenerate state.

    A portfolio member that crashes when the state has no shifts, or shifts
    with no stops, turns a rare intermediate into a dead search.
    """
    name, instance_path, solution_path = AVAILABLE[0]
    _instance, state, _days, lists = _load(instance_path, solution_path)
    for position in range(len(state.shifts)):
        state.set_operations(position, [], [], [])
    rng = random.Random(5)
    for _ in range(30):
        state.begin()
        operator(state, rng, lists)  # must not raise
        state.rollback()


# -- quantity bounds (contingency 2B) --------------------------------------

def test_the_portfolio_exposes_no_quantity_operator():
    """Contingency 2B's real fix: remove the variable, not bound it.

    Three operators used to move quantities directly, and they caused the two
    worst defects of this rebuild. ``increase_quantity`` multiplied by up to
    1.5x with no ceiling, so on V2.15 seed 4 one operation reached 107,593,087
    kg against a largest trailer capacity of 20,000 and the resulting LR of
    0.000037 looked like a breakthrough; and free upward movement on a call-in
    drop bought LR denominator by over-delivering an order, which the released
    checker scores as a *missed* order (see ``test_callin_nominal_ceiling.py``).

    Both are unreachable now that quantities are decoded from the route rather
    than searched over, per TS2020 sec. 3.1 -- none of Kheiri's LLH0-LLH18
    touches a quantity either. A ceiling patched onto a free variable is
    strictly weaker than not exposing the variable, so this test pins the
    absence.
    """
    banned = {
        "increase_quantity",
        "decrease_quantity",
        "fill_quantity",
        "minimal_quantity",
    }
    present = {name for name, _op in OPERATORS} & banned
    assert not present, f"quantity operators are back in the portfolio: {present}"


@pytest.mark.parametrize("name,instance_path,solution_path", _SMALL)
def test_the_logistic_ratio_denominator_stays_physical(
    name, instance_path, solution_path
):
    """Delivered quantity must stay within what the trailers could carry.

    The direct check for a gamed LR denominator: total delivered cannot exceed
    the trailers' initial load plus everything loaded at sources, so the ratio
    cannot be driven to zero by inflating it. Run with the decoder in the loop,
    since the decoder is now the only thing that sets a quantity.
    """
    _instance, state, _days, lists = _load(instance_path, solution_path)
    rng = random.Random(4)
    largest = max(float(c) for c in state.fi.trailer_capacity)
    for _ in range(4000):
        state.begin()
        OPERATORS[rng.randrange(len(OPERATORS))][1](state, rng, lists)
        for touched in state.touched_positions():
            decode_shift_quantities(state, touched)
        state.commit()
    for rec in state.shifts:
        for quantity in rec.quantities:
            assert abs(quantity) <= largest + 1e-6, (
                f"decoded |{quantity}| exceeds the largest trailer capacity"
            )
    score = state.score()
    headroom = sum(float(v) for v in state.fi.trailer_initial)
    assert score.scored_delivered_quantity <= (
        score.scored_loaded_quantity + headroom + 1e-6
    ), "delivered more than the trailers could ever have carried"


# -- the step 3 gate --------------------------------------------------------

# 90% per the plan's gate, applied to every operator with no exemption.
# ``create_shift`` was exempt while it drew driver windows freely and declined
# the ones opening past the score cutoff; it now draws only from windows that
# can host a visible shift and clears the target like everything else.
# The gate covers 25 operators: the four quantity movers are gone, since
# quantities are decoded rather than searched (see the absence test above).
_FIRE_TARGET = 0.90
_ATTEMPTS = 200


def _firing_rate(state, operator, lists, seed):
    rng = random.Random(seed)
    fired = 0
    for _ in range(_ATTEMPTS):
        state.begin()
        if operator(state, rng, lists):
            fired += 1
        state.rollback()
    return fired / _ATTEMPTS


@pytest.mark.parametrize("name,instance_path,solution_path", _GATE)
def test_every_operator_fires_on_at_least_ninety_percent_of_invocations(
    name, instance_path, solution_path
):
    """Step 3 gate. The direct fix for the legacy zero-candidate steps."""
    _instance, state, _days, lists = _load(instance_path, solution_path)
    rates = {
        op_name: _firing_rate(state, operator, lists, _seed_for(op_name))
        for op_name, operator in OPERATORS
    }
    weak = {
        op_name: rate for op_name, rate in rates.items() if rate < _FIRE_TARGET
    }
    assert not weak, f"{name}: operators below {_FIRE_TARGET:.0%}: {weak}"
