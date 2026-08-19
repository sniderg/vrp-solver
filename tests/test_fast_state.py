"""Equivalence and revert tests for the fast search substrate.

Step 1.6/1.7 of ``REBUILD_PLAN.md``.  These tests are the gate: the fast
state is only useful if it agrees with
:func:`vrp_solver.contest.score_prefix_with_feasibility_tail` to within
floating-point noise, on real instances, after arbitrary move sequences.
"""

from __future__ import annotations

from pathlib import Path
import random

import pytest

from vrp_solver.contest import score_prefix_with_feasibility_tail
from vrp_solver.fast.state import FastInstance, SearchState, instance_days
from vrp_solver.xml_io import load_instance, load_solution


INSTANCE_DIR = Path("roadef_2016_data/set_B/Instances_B_V25-11042016")

# Instance/solution pairs that exist in the repo.  Solutions are only used as
# *inputs to score*; nothing here treats a reference solution as solver input.
PAIRS = [
    ("V2.13", Path("scratch/replicate_V2.13_native.xml")),
    ("V2.14", Path("scratch/cold_V2.14_cadence.xml")),
    ("V2.15", Path("scratch/V2.15_compact_full_master.xml")),
    ("V2.16.2", Path("scratch/cold_V2.16.2_batch.xml")),
    ("V2.19", Path("scratch/opt_V2.19_native.xml")),
    ("V2.20.2", Path("scratch/opt_V2.20.2_native.xml")),
    ("V2.21.2", Path("scratch/opt_V2.21.2_native.xml")),
    ("V2.22", Path("scratch/best_V2.22_native.xml")),
    ("V2.24", Path("scratch/replicate_V2.24_native.xml")),
    ("V2.25", Path("scratch/opt3_V2.25_native.xml")),
]


def _available_pairs():
    out = []
    for name, solution_path in PAIRS:
        instance_path = INSTANCE_DIR / f"{name}.xml"
        if instance_path.exists() and solution_path.exists():
            out.append((name, instance_path, solution_path))
    return out


AVAILABLE = _available_pairs()

pytestmark = pytest.mark.skipif(
    not AVAILABLE,
    reason="Set B instances and baseline solution artifacts are not present",
)


def _assert_matches(instance, solution, score_days):
    reference = score_prefix_with_feasibility_tail(
        instance,
        solution,
        score_days=score_days,
        feasibility_days=score_days,
        ignore_tail_call_ins=True,
    )
    state = SearchState.from_solution(instance, solution, score_days=score_days)
    fast = state.score()

    assert fast.scored_shifts == reference.scored_shifts
    assert fast.scored_operations == reference.scored_operations
    assert fast.scored_delivered_quantity == pytest.approx(
        reference.scored_delivered_quantity, rel=1e-9, abs=1e-6
    )
    assert fast.scored_loaded_quantity == pytest.approx(
        reference.scored_loaded_quantity, rel=1e-9, abs=1e-6
    )
    assert fast.scored_estimated_cost == pytest.approx(
        reference.scored_estimated_cost, rel=1e-9, abs=1e-6
    )
    assert fast.feasibility_errors == reference.feasibility_errors
    assert fast.feasibility_warnings == reference.feasibility_warnings
    assert fast.tank_negative_steps == reference.tank_negative_steps
    assert fast.tank_overfill_steps == reference.tank_overfill_steps
    assert fast.tank_safety_breach_steps == reference.tank_safety_breach_steps
    assert fast.vmi_customers_below_safety == reference.vmi_customers_below_safety
    assert fast.safety_kg_min == pytest.approx(
        reference.safety_kg_min, rel=1e-9, abs=1e-6
    )
    return state, reference


@pytest.mark.parametrize("name,instance_path,solution_path", AVAILABLE)
def test_score_matches_contest_full_horizon(name, instance_path, solution_path):
    instance = load_instance(instance_path)
    solution = load_solution(solution_path)
    _assert_matches(instance, solution, instance_days(instance))


@pytest.mark.parametrize("name,instance_path,solution_path", AVAILABLE)
def test_score_matches_contest_truncated(name, instance_path, solution_path):
    instance = load_instance(instance_path)
    solution = load_solution(solution_path)
    days = instance_days(instance)
    if days < 2:
        pytest.skip("instance horizon is a single day")
    _assert_matches(instance, solution, max(1, days // 2))


@pytest.mark.parametrize("name,instance_path,solution_path", AVAILABLE)
def test_round_trip_solution(name, instance_path, solution_path):
    instance = load_instance(instance_path)
    solution = load_solution(solution_path)
    days = instance_days(instance)
    state = SearchState.from_solution(instance, solution, score_days=days)
    rebuilt = SearchState.from_solution(instance, state.to_solution(), score_days=days)
    assert state.score() == rebuilt.score()
    assert state.to_solution() == rebuilt.to_solution()


def _python_customer_tank(fi: FastInstance, row: int, deliveries):
    """The pure-Python reference for the Cython tank aggregate loop.

    Contingency 1A moved ``_recompute_customer_tank``'s inner loop into
    ``inventory_fast.score_customer_row``.  The plan requires the Python
    version to stay as the oracle, so it lives here and is asserted against.
    """
    import numpy as np

    ending = np.cumsum(deliveries - fi.cust_forecast[row]) + fi.cust_initial[row]
    safety = fi.cust_safety[row]
    capacity = fi.cust_capacity[row]
    negative = int(np.count_nonzero(ending < -1e-6))
    overfill = int(np.count_nonzero(ending > capacity + 1e-6))
    breach = int(np.count_nonzero(ending < safety - 1e-6))
    deficit = safety - ending - 1e-6
    kg_min = float(deficit[deficit > 0.0].sum()) * fi.unit
    return breach, negative, overfill, 1 if breach else 0, kg_min


@pytest.mark.parametrize("name,instance_path,solution_path", AVAILABLE)
def test_cython_tank_row_matches_python_oracle(name, instance_path, solution_path):
    """Step 1.6, contingency 1A: the C loop must equal the Python reference."""
    score_customer_row = pytest.importorskip(
        "vrp_solver.inventory_fast"
    ).score_customer_row

    instance = load_instance(instance_path)
    solution = load_solution(solution_path)
    days = instance_days(instance)
    state = SearchState.from_solution(instance, solution, score_days=days)
    fi = state.fi

    checked = 0
    for row in range(fi.n_customers):
        if fi.cust_is_call_in[row]:
            continue
        deliveries = state._deliveries[row]
        fast = score_customer_row(
            float(fi.cust_initial[row]),
            fi.cust_forecast[row],
            deliveries,
            float(fi.cust_capacity[row]),
            float(fi.cust_safety[row]),
            fi.horizon,
            fi.unit,
        )
        oracle = _python_customer_tank(fi, row, deliveries)
        assert fast[:4] == oracle[:4], f"{name}: counter mismatch on customer row {row}"
        assert fast[4] == pytest.approx(oracle[4], rel=1e-9, abs=1e-6), (
            f"{name}: safety_kg_min mismatch on customer row {row}"
        )
        checked += 1

    assert checked > 0, "no VMI customer was checked; the test would prove nothing"


def _random_mutation(state: SearchState, rng: random.Random) -> bool:
    """Apply one random structural edit.  Returns False if nothing was done."""
    if not state.shifts:
        return False
    position = rng.randrange(len(state.shifts))
    rec = state.shifts[position]
    choice = rng.randrange(6)

    if choice == 0 and len(rec.points) >= 2:
        # Reverse a block.
        i = rng.randrange(len(rec.points) - 1)
        j = rng.randrange(i + 1, len(rec.points))
        points = list(rec.points)
        points[i : j + 1] = reversed(points[i : j + 1])
        state.set_operations(position, points, rec.arrivals, rec.quantities)
        return True
    if choice == 1 and rec.points:
        # Drop a stop.
        i = rng.randrange(len(rec.points))
        state.set_operations(
            position,
            rec.points[:i] + rec.points[i + 1 :],
            rec.arrivals[:i] + rec.arrivals[i + 1 :],
            rec.quantities[:i] + rec.quantities[i + 1 :],
        )
        return True
    if choice == 2 and rec.points:
        # Perturb one quantity.
        i = rng.randrange(len(rec.points))
        state.set_quantity(position, i, rec.quantities[i] * rng.uniform(0.2, 1.8))
        return True
    if choice == 3:
        # Move the whole shift in time.
        delta = rng.choice([-240, -60, 60, 240])
        cutoff = state.fi.cutoff
        if rec.start + delta < 0:
            return False
        arrivals = [a + delta for a in rec.arrivals]
        if arrivals and (max(arrivals) >= cutoff or min(arrivals) < 0):
            return False
        state.set_shift_timing(position, rec.start + delta, arrivals)
        return True
    if choice == 4:
        # Reassign resources, deliberately including illegal combinations so
        # the penalty path is exercised.
        state.set_shift_resources(
            position,
            driver=rng.randrange(len(state.fi.drivers)),
            trailer=rng.randrange(len(state.fi.trailers)),
        )
        return True
    if choice == 5 and len(state.shifts) > 1:
        state.remove_shift(position)
        return True
    return False


@pytest.mark.parametrize("name,instance_path,solution_path", AVAILABLE)
def test_incremental_score_matches_after_random_moves(
    name, instance_path, solution_path
):
    """Step 1.6: the equivalence gate, under mutation.

    After each accepted edit the incrementally maintained score must equal a
    from-scratch ``contest`` score of the mutated solution.  A single
    disagreement anywhere invalidates the substrate.
    """
    instance = load_instance(instance_path)
    solution = load_solution(solution_path)
    days = instance_days(instance)
    state = SearchState.from_solution(instance, solution, score_days=days)
    rng = random.Random(20260810)

    applied = 0
    for _ in range(200):
        if applied >= 40:
            break
        if not _random_mutation(state, rng):
            continue
        applied += 1
        fast = state.score()
        reference = score_prefix_with_feasibility_tail(
            instance,
            state.to_solution(),
            score_days=days,
            feasibility_days=days,
            ignore_tail_call_ins=True,
        )
        assert fast.feasibility_errors == reference.feasibility_errors, (
            f"{name}: error count drift after {applied} moves "
            f"(fast={fast.feasibility_errors} ref={reference.feasibility_errors})"
        )
        assert fast.scored_estimated_cost == pytest.approx(
            reference.scored_estimated_cost, rel=1e-9, abs=1e-6
        ), f"{name}: cost drift after {applied} moves"
        assert fast.scored_delivered_quantity == pytest.approx(
            reference.scored_delivered_quantity, rel=1e-9, abs=1e-6
        ), f"{name}: delivered drift after {applied} moves"
        assert fast.safety_kg_min == pytest.approx(
            reference.safety_kg_min, rel=1e-9, abs=1e-6
        ), f"{name}: safety_kg_min drift after {applied} moves"
        assert fast.feasibility_warnings == reference.feasibility_warnings
        assert fast.tank_negative_steps == reference.tank_negative_steps
        assert fast.tank_overfill_steps == reference.tank_overfill_steps
        assert fast.tank_safety_breach_steps == reference.tank_safety_breach_steps

    assert applied > 0, "no mutation was applied; the test would prove nothing"


@pytest.mark.parametrize("name,instance_path,solution_path", AVAILABLE)
def test_rollback_is_exact(name, instance_path, solution_path):
    """Step 1.7: apply-then-revert restores the state exactly."""
    instance = load_instance(instance_path)
    solution = load_solution(solution_path)
    days = instance_days(instance)
    state = SearchState.from_solution(instance, solution, score_days=days)
    rng = random.Random(4242)

    before_score = state.score()
    before_solution = state.to_solution()

    for _ in range(60):
        state.begin()
        touched = 0
        for _ in range(rng.randrange(1, 4)):
            if _random_mutation(state, rng):
                touched += 1
        state.rollback()
        assert state.score() == before_score, f"{name}: score drift after rollback"
        assert state.to_solution() == before_solution, (
            f"{name}: structure drift after rollback"
        )


@pytest.mark.parametrize("name,instance_path,solution_path", AVAILABLE)
def test_commit_keeps_mutation(name, instance_path, solution_path):
    instance = load_instance(instance_path)
    solution = load_solution(solution_path)
    days = instance_days(instance)
    state = SearchState.from_solution(instance, solution, score_days=days)
    rng = random.Random(7)

    before = state.to_solution()
    state.begin()
    changed = False
    for _ in range(50):
        if _random_mutation(state, rng):
            changed = True
            break
    state.commit()
    assert changed, "no mutation was applied"
    assert state.to_solution() != before

    # A committed state must still agree with a from-scratch rebuild.
    rebuilt = SearchState.from_solution(instance, state.to_solution(), score_days=days)
    assert state.score() == rebuilt.score()


@pytest.mark.parametrize("name,instance_path,solution_path", AVAILABLE)
def test_copy_is_independent(name, instance_path, solution_path):
    instance = load_instance(instance_path)
    solution = load_solution(solution_path)
    days = instance_days(instance)
    state = SearchState.from_solution(instance, solution, score_days=days)
    clone = state.copy()
    assert clone.score() == state.score()

    rng = random.Random(99)
    for _ in range(50):
        if _random_mutation(clone, rng):
            break
    rebuilt = SearchState.from_solution(instance, state.to_solution(), score_days=days)
    assert state.score() == rebuilt.score(), "mutating the copy disturbed the original"
