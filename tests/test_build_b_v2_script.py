from __future__ import annotations

from pathlib import Path
from subprocess import run

from scripts import build_b_v2
from vrp_solver.contest import ContestScore


def _score(*, feasible: bool, ratio: float) -> ContestScore:
    delivered = 1000.0
    return ContestScore(
        score_days=10,
        feasibility_days=10,
        score_cutoff_minute=14_400,
        feasibility_cutoff_minute=14_400,
        submitted_shifts=1,
        submitted_operations=2,
        scored_shifts=1,
        scored_operations=2,
        scored_delivered_quantity=delivered,
        scored_loaded_quantity=delivered,
        scored_estimated_cost=ratio * delivered,
        feasible=feasible,
        feasibility_errors=0 if feasible else 1,
        feasibility_warnings=0,
        hard_violations=0 if feasible else 1,
        safety_kg_min=0.0,
        tank_safety_breach_steps=0,
        tank_negative_steps=0,
        tank_overfill_steps=0,
        vmi_customers_below_safety=0,
        first_safety_breach_minute=None,
    )


def test_best_result_ignores_infeasible_and_official_invalid() -> None:
    valid = build_b_v2.PhaseResult(
        phase="valid",
        path=Path("valid.xml"),
        score=_score(feasible=True, ratio=0.2),
        official_valid=True,
        official_ratio=0.19,
    )
    better_local_but_invalid = build_b_v2.PhaseResult(
        phase="invalid",
        path=Path("invalid.xml"),
        score=_score(feasible=True, ratio=0.1),
        official_valid=False,
    )
    infeasible = build_b_v2.PhaseResult(
        phase="infeasible",
        path=Path("infeasible.xml"),
        score=_score(feasible=False, ratio=0.05),
    )

    assert build_b_v2.best_result([infeasible, better_local_but_invalid, valid]) == valid


def test_build_b_v2_help_runs() -> None:
    result = run(["python", "scripts/build_b_v2.py", "--help"], capture_output=True, text=True)

    assert result.returncode == 0
    assert "Build Set B V2 solutions from scratch" in result.stdout
