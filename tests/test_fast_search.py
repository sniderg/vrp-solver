"""Tests for the search loop and its controllers (step 4, and the step 2 gate).

The step 2 gate itself -- zero empty neighbourhoods and >= 20% of steps accepted
over a 60 s run on V2.15 -- is measured by ``tools/bench_search.py``; a 60 s run
has no place in a unit suite. What is pinned here are the loop's invariants,
which is what makes that measurement trustworthy: the loop never corrupts the
state, the score it reports is the official one, and the best it returns is
really the best it saw.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vrp_solver.contest import score_prefix_with_feasibility_tail  # noqa: E402
from vrp_solver.fast.objective import AdaptiveWeights, Objective  # noqa: E402
from vrp_solver.fast.retime import CandidateLists  # noqa: E402
from vrp_solver.fast.search import (  # noqa: E402
    N_OPERATORS,
    Selector,
    UniformSelector,
    run_search,
)
from vrp_solver.fast.state import SearchState, instance_days  # noqa: E402
from vrp_solver.xml_io import load_instance, load_solution  # noqa: E402

from test_fast_state import AVAILABLE  # noqa: E402

pytestmark = pytest.mark.skipif(not AVAILABLE, reason="Set B artifacts absent")

_SMALL = [p for p in AVAILABLE if p[0] in {"V2.13", "V2.24"}] or AVAILABLE[:1]
_STEPS = 3000


def _state(instance_path, solution_path):
    instance = load_instance(instance_path)
    solution = load_solution(solution_path)
    days = instance_days(instance)
    return instance, SearchState.from_solution(instance, solution, score_days=days), days


# -- controllers ------------------------------------------------------------


def test_uniform_selector_covers_the_whole_portfolio():
    selector = UniformSelector()
    rng = random.Random(7)
    seen = set()
    for _ in range(20 * N_OPERATORS):
        chosen = selector.select(rng)
        assert len(chosen) == 1
        seen.update(chosen)
    assert seen == set(range(N_OPERATORS)), "an unreachable operator is dead weight"


def test_the_base_selector_update_is_a_no_op():
    """Simple controllers must be able to ignore feedback without overriding."""
    Selector().update((0,), improved_best=True, gain=1.0, seconds=0.1)


# -- loop invariants --------------------------------------------------------


@pytest.mark.parametrize("name,instance_path,solution_path", _SMALL)
def test_the_reported_best_score_matches_the_official_scorer(
    name, instance_path, solution_path
):
    """The loop may not report a score the checker would disagree with.

    Every operator is pinned against the official scorer individually; this
    pins the composition of thousands of them, including the commit/rollback
    interleaving that no single-operator test exercises.
    """
    instance, state, days = _state(instance_path, solution_path)
    result = run_search(state, limit=30.0, seed=3, max_steps=_STEPS)
    reference = score_prefix_with_feasibility_tail(
        instance, result.best_solution,
        score_days=days, feasibility_days=days, ignore_tail_call_ins=True,
    )
    assert result.best_score.feasibility_errors == reference.feasibility_errors
    assert result.best_score.scored_estimated_cost == pytest.approx(
        reference.scored_estimated_cost, rel=1e-9, abs=1e-6
    )
    assert result.best_score.scored_delivered_quantity == pytest.approx(
        reference.scored_delivered_quantity, rel=1e-9, abs=1e-6
    )


@pytest.mark.parametrize("name,instance_path,solution_path", _SMALL)
def test_the_search_never_reports_a_best_worse_than_its_seed(
    name, instance_path, solution_path
):
    """The seed is the incumbent, so the returned best is at worst the seed."""
    _instance, state, _days = _state(instance_path, solution_path)
    seed_score = state.score()
    objective = Objective()
    seed_quality = objective.quality(seed_score)
    result = run_search(
        state, limit=30.0, seed=11, objective=objective, max_steps=_STEPS
    )
    assert result.best_quality <= seed_quality + 1e-12


@pytest.mark.parametrize("name,instance_path,solution_path", _SMALL)
def test_the_live_state_ends_on_an_accepted_solution(
    name, instance_path, solution_path
):
    """No transaction is left dangling, so the state stays usable afterwards.

    A leaked ``begin()`` would raise here, and a mis-scored commit would show as
    a mismatch between the live state and a fresh score of its own solution.
    """
    instance, state, days = _state(instance_path, solution_path)
    run_search(state, limit=30.0, seed=13, max_steps=_STEPS)
    state.begin()  # would raise "a transaction is already open" if leaked
    state.rollback()
    live = state.score()
    reference = score_prefix_with_feasibility_tail(
        instance, state.to_solution(),
        score_days=days, feasibility_days=days, ignore_tail_call_ins=True,
    )
    assert live.feasibility_errors == reference.feasibility_errors
    assert live.scored_estimated_cost == pytest.approx(
        reference.scored_estimated_cost, rel=1e-9, abs=1e-6
    )


@pytest.mark.parametrize("name,instance_path,solution_path", _SMALL)
def test_the_same_seed_reproduces_the_same_search(name, instance_path, solution_path):
    """Step 4.5 needs matched seeds, so a run has to be reproducible."""
    outcomes = []
    for _ in range(2):
        _instance, state, _days = _state(instance_path, solution_path)
        result = run_search(
            state, limit=30.0, seed=29, objective=Objective(), max_steps=_STEPS
        )
        outcomes.append((result.best_quality, result.best_solution))
    assert outcomes[0][0] == outcomes[1][0]
    assert outcomes[0][1] == outcomes[1][1]


@pytest.mark.parametrize("name,instance_path,solution_path", _SMALL)
def test_reselection_removes_empty_neighbourhood_steps(
    name, instance_path, solution_path
):
    """The step 2 gate's first half, at unit-test scale.

    ``two_opt_star`` and ``create_shift`` decline a few percent of invocations
    for real structural reasons, which the step 3 gate allows; the loop absorbs
    that by reselecting, because a step is a unit of search progress and one
    operator declining is not a reason to spend one.
    """
    _instance, state, _days = _state(instance_path, solution_path)
    result = run_search(state, limit=30.0, seed=17, max_steps=_STEPS)
    assert result.telemetry.empty_neighbourhood == 0
    assert result.telemetry.steps > 0


@pytest.mark.parametrize("name,instance_path,solution_path", _SMALL)
def test_a_dead_state_terminates_instead_of_spinning(
    name, instance_path, solution_path
):
    """The retry loop is bounded, so an unfireable portfolio still returns."""
    _instance, state, _days = _state(instance_path, solution_path)

    class _Dead(Selector):
        def select(self, rng):
            return ()

    result = run_search(
        state, limit=30.0, seed=19, selector=_Dead(), max_steps=200
    )
    assert result.telemetry.steps == 200
    assert result.telemetry.empty_neighbourhood == 200


@pytest.mark.parametrize("name,instance_path,solution_path", _SMALL)
def test_telemetry_and_operator_accounting_are_consistent(
    name, instance_path, solution_path
):
    _instance, state, _days = _state(instance_path, solution_path)
    result = run_search(state, limit=30.0, seed=23, max_steps=_STEPS)
    t = result.telemetry
    assert t.steps == _STEPS
    assert 0 <= t.accepted <= t.steps
    assert 0.0 <= t.accepted_fraction <= 1.0
    for i in range(N_OPERATORS):
        assert result.operator_fires[i] <= result.operator_calls[i]
    # Every step that was not conceded as empty applied at least one operator.
    assert sum(result.operator_fires) >= t.steps - t.empty_neighbourhood
    assert result.operator_table().isascii(), "Windows cp1252 consoles"


@pytest.mark.parametrize("name,instance_path,solution_path", _SMALL)
def test_adaptive_weights_are_repriced_against_the_same_objective(
    name, instance_path, solution_path
):
    """Moving weights must not let a stale incumbent quality leak in.

    When the landscape moves, the incumbent and the best are re-priced under
    the new weights before the comparison; otherwise the loop compares two
    numbers computed under different objectives and "improvement" is an
    artifact of the weight change.
    """
    _instance, state, _days = _state(instance_path, solution_path)
    adaptive = AdaptiveWeights(window=50)
    result = run_search(
        state, limit=30.0, seed=31, adaptive=adaptive, max_steps=_STEPS
    )
    assert result.best_quality == pytest.approx(
        adaptive.objective().quality(result.best_score)
    )


def test_candidate_lists_are_reused_across_a_run():
    """Neighbour lists are per-instance, so a run must not rebuild them."""
    name, instance_path, solution_path = _SMALL[0]
    _instance, state, _days = _state(instance_path, solution_path)
    lists = CandidateLists(state.fi)
    result = run_search(
        state, limit=30.0, seed=37, lists=lists, max_steps=500
    )
    assert result.telemetry.steps == 500


# -- the published incumbent ------------------------------------------------
#
# ``best_solution`` is best under the *live adaptive quality*; the artifact that
# gets verified has to be best under ``(errors, LR)``.  Those differ, and the
# difference was not academic: on V2.26 the adaptive best "improved" a 7-error
# seed to 32 errors, because decaying weights let the worse state out-price the
# better one.  These pin the second incumbent.


@pytest.mark.parametrize("name,instance_path,solution_path", _SMALL)
def test_the_published_score_is_never_worse_than_the_seed(
    name, instance_path, solution_path
):
    _instance, state, _days = _state(instance_path, solution_path)
    seed_score = state.score()
    result = run_search(
        state, limit=30.0, seed=41, adaptive=AdaptiveWeights(), max_steps=_STEPS
    )
    published = result.published_score
    assert published is not None
    assert (published.feasibility_errors, published.logistic_ratio) <= (
        seed_score.feasibility_errors,
        seed_score.logistic_ratio,
    ), "the published incumbent regressed below the seed it started from"


@pytest.mark.parametrize("name,instance_path,solution_path", _SMALL)
def test_the_published_score_beats_the_adaptive_best_on_errors(
    name, instance_path, solution_path
):
    """Publication order is lexicographic, so it can never lose on errors."""
    _instance, state, _days = _state(instance_path, solution_path)
    result = run_search(
        state, limit=30.0, seed=43, adaptive=AdaptiveWeights(), max_steps=_STEPS
    )
    assert (
        result.published_score.feasibility_errors
        <= result.best_score.feasibility_errors
    )


@pytest.mark.parametrize("name,instance_path,solution_path", _SMALL)
def test_the_published_solution_scores_what_the_run_reported(
    name, instance_path, solution_path
):
    """The returned XML must actually score as claimed, via the official path.

    The published incumbent is captured mid-transaction, before a possible
    rollback, so this is the test that it was captured from the right state.
    """
    instance, state, days = _state(instance_path, solution_path)
    result = run_search(
        state, limit=30.0, seed=47, adaptive=AdaptiveWeights(), max_steps=_STEPS
    )
    reference = score_prefix_with_feasibility_tail(
        instance,
        result.published_solution,
        score_days=days,
        feasibility_days=days,
        ignore_tail_call_ins=True,
    )
    assert reference.feasibility_errors == result.published_score.feasibility_errors


@pytest.mark.parametrize("name,instance_path,solution_path", _SMALL)
def test_the_published_solution_carries_no_empty_shift(
    name, instance_path, solution_path
):
    """The checker returns execution_failed on an ``<operations />`` element."""
    _instance, state, _days = _state(instance_path, solution_path)
    result = run_search(
        state, limit=30.0, seed=53, adaptive=AdaptiveWeights(), max_steps=_STEPS
    )
    assert all(shift.operations for shift in result.published_solution.shifts)
