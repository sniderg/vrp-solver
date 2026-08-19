"""Tests for the penalized objective and acceptance rule (step 2)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from vrp_solver.fast.objective import (
    DEFAULT_WEIGHTS,
    GROUPS,
    AdaptiveWeights,
    Objective,
    SearchTelemetry,
    accept,
    acceptance_threshold,
)
from vrp_solver.fast.state import SearchState, StateScore, instance_days
from vrp_solver.xml_io import load_instance, load_solution

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_fast_state import AVAILABLE, _random_mutation  # noqa: E402


def _score(**kwargs) -> StateScore:
    base = dict(
        scored_shifts=1,
        scored_operations=2,
        scored_delivered_quantity=1000.0,
        scored_loaded_quantity=1000.0,
        scored_estimated_cost=100.0,
        feasibility_errors=0,
        feasibility_warnings=0,
        safety_kg_min=0.0,
        tank_safety_breach_steps=0,
        tank_negative_steps=0,
        tank_overfill_steps=0,
        vmi_customers_below_safety=0,
    )
    base.update(kwargs)
    return StateScore(**base)


# -- the objective itself ---------------------------------------------------


def test_feasible_quality_is_exactly_the_logistic_ratio():
    score = _score()
    assert Objective().quality(score) == pytest.approx(score.logistic_ratio)


def test_every_violation_group_is_priced():
    """No group may be silently free -- that would recreate a blind spot."""
    objective = Objective()
    baseline = objective.quality(_score())
    for name in GROUPS:
        worse = objective.quality(_score(**{name: 1}))
        assert worse > baseline, f"{name} carries no penalty"


def test_safety_deficit_is_priced_on_its_own():
    objective = Objective()
    assert objective.quality(_score(safety_kg_min=1e7)) > objective.quality(_score())


def test_one_rule_error_outweighs_any_reachable_lr_gain():
    """A violation must never be buyable with a routing improvement.

    Set B logistic ratios are ~0.02-0.2.  A single structural error is
    weighted 1.0, so even collapsing cost to zero cannot pay for it.
    """
    objective = Objective()
    # A realistically poor feasible state: LR at the top of the Set B range.
    poor_but_clean = objective.quality(
        _score(scored_estimated_cost=200.0, scored_delivered_quantity=1000.0)
    )
    assert poor_but_clean == pytest.approx(0.2)
    # The best conceivable state that breaks one rule: zero cost.
    perfect_but_broken = objective.quality(
        _score(scored_estimated_cost=0.0, shift_errors=1, feasibility_errors=1)
    )
    assert perfect_but_broken > poor_but_clean


def test_quality_is_finite_for_a_badly_broken_state():
    """The whole point of step 2: no state is unscoreable."""
    quality = Objective().quality(
        _score(
            scored_delivered_quantity=0.0,
            scored_estimated_cost=5000.0,
            feasibility_errors=900,
            shift_errors=400,
            trailer_errors=100,
            driver_errors=100,
            callin_errors=100,
            tank_negative_steps=100,
            tank_overfill_steps=100,
            tank_safety_breach_steps=100,
            safety_kg_min=1e9,
        )
    )
    assert quality == pytest.approx(quality)  # not NaN
    assert quality < float("inf")


def test_scaled_multiplies_every_weight():
    scaled = Objective().scaled(10.0)
    for name, weight in DEFAULT_WEIGHTS.items():
        assert scaled.weights[name] == pytest.approx(weight * 10.0)
    assert scaled.penalty(_score(shift_errors=1)) == pytest.approx(
        Objective().penalty(_score(shift_errors=1)) * 10.0
    )


# -- adaptive weights -------------------------------------------------------


def test_persistent_violation_raises_its_own_weight_only():
    """Per-group scales are normalized, so the claim is about the *ratio*."""
    adaptive = AdaptiveWeights(window=10_000)
    for _ in range(50):
        adaptive.observe(_score(driver_errors=1, feasibility_errors=1))
    scales = adaptive.scales()
    assert scales["driver_errors"] == pytest.approx(1.0), "the top scale is held at 1"
    assert scales["shift_errors"] < scales["driver_errors"], (
        "an absent violation should decay relative to a persistent one"
    )


def test_persistent_group_ratio_saturates_instead_of_the_scale_itself():
    """The regression normalization exists for.

    At ``raise_factor = 1.05`` an un-normalized scale reaches ``max_scale=200``
    in ~108 observations, so on a 100k-step run every group that ever fires ends
    pinned at the ceiling and the scales carry no information -- measured that
    way on V2.15 before ``_normalize``. What must saturate is the *ratio*
    between a persistent group and an absent one, bounded by ``min_scale``.
    """
    adaptive = AdaptiveWeights(window=10_000, min_scale=0.05)
    for _ in range(5000):
        adaptive.observe(_score(shift_errors=1, feasibility_errors=1))
    scales = adaptive.scales()
    assert scales["shift_errors"] == pytest.approx(1.0)
    assert scales["driver_errors"] == pytest.approx(0.05)


def test_weights_are_bounded():
    adaptive = AdaptiveWeights(window=10_000, max_scale=3.0, min_scale=0.5)
    for _ in range(2000):
        adaptive.observe(_score(shift_errors=1, feasibility_errors=1))
    scales = adaptive.scales()
    assert scales["shift_errors"] == pytest.approx(1.0)
    assert scales["driver_errors"] == pytest.approx(0.5)
    # An all-feasible run decays every group to the floor; ``_normalize`` steps
    # aside once nothing exceeds 1, so the floor stays reachable.
    for _ in range(2000):
        adaptive.observe(_score())
    assert adaptive.scales()["shift_errors"] == pytest.approx(0.5)


def test_rebalance_raises_the_global_scale_when_too_often_infeasible():
    adaptive = AdaptiveWeights(window=100)
    # 90% infeasible is far above the 0.35 target ceiling.
    for i in range(100):
        adaptive.observe(
            _score(shift_errors=1, feasibility_errors=1) if i % 10 else _score()
        )
    assert adaptive.global_scale > 1.0


def test_rebalance_lowers_the_global_scale_when_rarely_infeasible():
    adaptive = AdaptiveWeights(window=100)
    for _ in range(100):
        adaptive.observe(_score())
    assert adaptive.global_scale < 1.0


def test_rebalance_is_quiet_inside_the_target_band():
    adaptive = AdaptiveWeights(window=100)
    for i in range(100):
        adaptive.observe(
            _score(callin_errors=1, feasibility_errors=1) if i % 4 == 0 else _score()
        )
    # 25% infeasible sits inside (0.15, 0.35), so the brake stays off.
    assert adaptive.global_scale == pytest.approx(1.0)


def test_global_brake_is_not_swamped_by_per_group_drift():
    """The regression that motivated splitting the two multipliers.

    Per-group scales move every observation and so travel by
    ``factor ** window`` across one window; a global adjustment folded into the
    same numbers once per window is invisible next to that. Keeping the global
    scale separate means the brake compounds once per window and is still
    legible after several windows.
    """
    adaptive = AdaptiveWeights(window=100)
    for _ in range(5):
        for i in range(100):
            adaptive.observe(
                _score(shift_errors=1, feasibility_errors=1) if i % 10 else _score()
            )
    # Five windows at 90% infeasible: the brake should have engaged every time.
    assert adaptive.global_scale == pytest.approx(1.5**5)

    # The brake multiplies whatever the per-group scale settled on, so its
    # contribution survives even a group that decayed to its floor.  (A group
    # that never fires *should* end up cheap; what must not happen is the
    # brake's effect vanishing into the per-group drift.)
    effective = adaptive.objective().weights["callin_errors"]
    unbraked = DEFAULT_WEIGHTS["callin_errors"] * adaptive.scales()["callin_errors"]
    assert effective == pytest.approx(unbraked * 1.5**5)
    assert effective > unbraked * 7.0


def test_the_global_brake_is_off_while_the_best_is_infeasible():
    """Measured on V2.15: an ungated brake ratchets to its ceiling and stays.

    From an infeasible seed the visited-infeasible fraction is ~100% because no
    feasible state has been found yet, so every window reads as "violations
    underpriced". The brake ran to 200x, quality reached 2.5e5, and the run
    ended with 33 errors against 16 for fixed weights. The band only means
    something once there is a feasible incumbent to return to.
    """
    adaptive = AdaptiveWeights(window=100)
    for _ in range(500):
        adaptive.observe(
            _score(shift_errors=1, feasibility_errors=1), best_feasible=False
        )
    assert adaptive.global_scale == pytest.approx(1.0)


def test_group_ratios_survive_the_global_scale():
    """The brake must change magnitude without reordering the groups."""
    adaptive = AdaptiveWeights(window=50)
    for _ in range(200):
        adaptive.observe(_score(driver_errors=1, feasibility_errors=1))
    weights = adaptive.objective().weights
    assert weights["driver_errors"] > weights["callin_errors"] > 0.0


# -- acceptance -------------------------------------------------------------


def test_threshold_is_flat_while_best_is_infeasible():
    for elapsed in (0.0, 30.0, 60.0):
        assert acceptance_threshold(
            best_feasible=False, elapsed=elapsed, limit=60.0
        ) == pytest.approx(0.001)


def test_threshold_anneals_once_best_is_feasible():
    start = acceptance_threshold(best_feasible=True, elapsed=0.0, limit=60.0)
    mid = acceptance_threshold(best_feasible=True, elapsed=30.0, limit=60.0)
    end = acceptance_threshold(best_feasible=True, elapsed=60.0, limit=60.0)
    assert start == pytest.approx(0.0101)
    assert mid == pytest.approx(0.0051)
    assert end == pytest.approx(0.0001)
    assert start > mid > end


def test_threshold_clamps_past_the_limit():
    assert acceptance_threshold(
        best_feasible=True, elapsed=999.0, limit=60.0
    ) == pytest.approx(0.0001)


def test_improving_and_equal_moves_are_always_accepted():
    kwargs = dict(best_quality=1.0, best_feasible=True, elapsed=0.0, limit=60.0)
    assert accept(current_quality=1.0, candidate_quality=0.9, **kwargs)
    assert accept(current_quality=1.0, candidate_quality=1.0, **kwargs)


def test_slightly_worse_than_best_is_accepted_but_far_worse_is_not():
    kwargs = dict(
        current_quality=1.0, best_quality=1.0, best_feasible=True,
        elapsed=0.0, limit=60.0,
    )
    assert accept(candidate_quality=1.005, **kwargs)
    assert not accept(candidate_quality=1.5, **kwargs)


def test_acceptance_narrows_as_time_runs_out():
    kwargs = dict(
        current_quality=1.0, candidate_quality=1.005, best_quality=1.0,
        best_feasible=True, limit=60.0,
    )
    assert accept(elapsed=0.0, **kwargs)
    assert not accept(elapsed=59.0, **kwargs)


def test_acceptance_handles_a_negative_best_quality():
    """``abs(best_quality)`` keeps the band from inverting."""
    assert accept(
        current_quality=-1.0, candidate_quality=-0.999, best_quality=-1.0,
        best_feasible=True, elapsed=0.0, limit=60.0,
    )


# -- telemetry --------------------------------------------------------------


def test_telemetry_tracks_the_fractions_the_gate_needs():
    telemetry = SearchTelemetry()
    for i in range(10):
        telemetry.record(
            _score(shift_errors=1, feasibility_errors=1) if i < 3 else _score(),
            accepted=i % 2 == 0,
        )
    assert telemetry.steps == 10
    assert telemetry.accepted_fraction == pytest.approx(0.5)
    assert telemetry.infeasible_fraction == pytest.approx(0.3)


def test_telemetry_keeps_lr_numerator_and_denominator_apart():
    """Contingency 2B: LR alone cannot reveal a gamed denominator."""
    telemetry = SearchTelemetry()
    score = _score(scored_estimated_cost=250.0, scored_delivered_quantity=9000.0)
    telemetry.record_best(score, Objective().quality(score))
    assert telemetry.best_cost == pytest.approx(250.0)
    assert telemetry.best_delivered == pytest.approx(9000.0)
    assert "best_cost" in telemetry.summary()
    assert telemetry.summary().isascii(), "Windows cp1252 consoles"


# -- against real instances -------------------------------------------------


@pytest.mark.skipif(not AVAILABLE, reason="Set B artifacts absent")
@pytest.mark.parametrize("name,instance_path,solution_path", AVAILABLE)
def test_violation_groups_sum_to_the_official_error_count(
    name, instance_path, solution_path
):
    """The objective may only price violations the verified scorer counts.

    ``test_fast_state`` already pins ``feasibility_errors`` against the
    official pipeline.  Pinning the group decomposition to that total is what
    makes the penalized objective trustworthy: it cannot double-count an error
    or miss one.
    """
    instance = load_instance(instance_path)
    solution = load_solution(solution_path)
    days = instance_days(instance)
    state = SearchState.from_solution(instance, solution, score_days=days)
    score = state.score()
    assert sum(getattr(score, name) for name in GROUPS) == score.feasibility_errors


@pytest.mark.skipif(not AVAILABLE, reason="Set B artifacts absent")
@pytest.mark.parametrize("name,instance_path,solution_path", AVAILABLE)
def test_group_sum_holds_under_mutation(name, instance_path, solution_path):
    import random

    instance = load_instance(instance_path)
    solution = load_solution(solution_path)
    days = instance_days(instance)
    state = SearchState.from_solution(instance, solution, score_days=days)
    rng = random.Random(31337)
    objective = Objective()

    applied = 0
    for _ in range(120):
        if not _random_mutation(state, rng):
            continue
        applied += 1
        score = state.score()
        assert (
            sum(getattr(score, group) for group in GROUPS)
            == score.feasibility_errors
        ), f"{name}: group decomposition drifted after {applied} moves"
        quality = objective.quality(score)
        assert quality == quality, f"{name}: quality became NaN"
        assert quality < float("inf")
    assert applied > 0
