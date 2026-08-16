from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from vrp_solver.diagnostics import ViolationVector
from vrp_solver.model import Instance, Solution
from vrp_solver.solver import unified_engine


ZERO = ViolationVector(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0)


def _instance() -> Instance:
    return Instance(
        name="orchestration", unit=60, horizon=48,
        time_matrix=((0,),), distance_matrix=((0.0,),), base_index=0,
        drivers=(), trailers=(), sources=(), customers=(),
    )


def test_rejects_empty_seed_budget() -> None:
    with pytest.raises(ValueError, match="num_seeds"):
        unified_engine.solve_cold_start(_instance(), num_seeds=0)


def test_constructor_portfolio_uses_structural_strategies_before_random_seeds() -> None:
    portfolio = unified_engine._construction_portfolio(4)

    assert [strategy.name for strategy in portfolio] == [
        "urgency-band",
        "urgency-band-narrow",
        "urgency-band-dense-reload",
        "scarcity-chain",
    ]
    assert portfolio[0].long_window_urgency_override is False
    assert portfolio[1].neighborhood_size == 3
    assert portfolio[2].neighborhood_size == 4
    assert portfolio[2].long_window_urgency_override is True
    assert portfolio[2].proactive_reload_ratio == pytest.approx(0.48)
    assert portfolio[3].need_ordering == "scarcity"


def test_connects_constructor_quantity_repair_and_search(monkeypatch) -> None:
    instance = _instance()
    raw = Solution(())
    repaired = replace(raw)
    searched = replace(raw)
    calls: list[str] = []

    monkeypatch.setattr(
        unified_engine, "construct_cluster_solution",
        lambda *_args, **_kwargs: (
            raw, SimpleNamespace(unscheduled_customers=(7,)),
        ),
    )
    vectors = iter((
        replace(ZERO, safety_deficit_quantity_minutes=3.0),
        replace(ZERO, safety_deficit_quantity_minutes=3.0),
        replace(ZERO, safety_deficit_quantity_minutes=2.0),
        replace(ZERO, safety_deficit_quantity_minutes=2.0),
        ZERO,
        ZERO,
    ))
    monkeypatch.setattr(unified_engine, "violation_vector", lambda *_: next(vectors))

    def repair(*_args, **kwargs):
        calls.append(f"repair:{kwargs['feasibility_days']}")
        return repaired, SimpleNamespace(status="Optimal")

    def search(_instance, _initial, *, config, progress):
        calls.append(f"search:{config.end_day}:{progress}")
        return searched, (SimpleNamespace(),)

    monkeypatch.setattr(unified_engine, "repair_quantities_with_highs", repair)
    monkeypatch.setattr(unified_engine, "surgical_search", search)
    monkeypatch.setattr(unified_engine, "validate_solution", lambda *_: [])
    monkeypatch.setattr(unified_engine, "solution_fingerprint", lambda solution: str(id(solution)))
    monkeypatch.setattr(unified_engine, "_local_ratio", lambda *_: 1.0)

    result = unified_engine.solve_cold_start(
        instance, num_seeds=1, search_iterations=5, stop_when_feasible=False,
    )

    # 48 hourly steps are two days; the former implementation incorrectly
    # divided the step count itself by 1,440.
    assert calls == ["repair:2", "search:2:None"]
    assert result.valid
    assert result.validation_status == "locally_feasible"
    assert result.provenance == "native-cold-start"
    assert result.quantity_repairs == 1
    assert result.search_steps == 1


def test_local_ratio_uses_full_horizon_keyword_contract(monkeypatch) -> None:
    captured = {}

    def score(_instance, _solution, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            scored_delivered_quantity=4.0,
            scored_estimated_cost=2.0,
        )

    monkeypatch.setattr(
        unified_engine, "score_prefix_with_feasibility_tail", score,
    )

    assert unified_engine._local_ratio(_instance(), Solution(()), 2) == 0.5
    assert captured == {"score_days": 2, "feasibility_days": 2}


def test_stops_constructing_at_first_feasible_seed(monkeypatch) -> None:
    calls = []

    def construct(*_args, **kwargs):
        calls.append(kwargs["tie_break_seed"])
        return Solution(()), SimpleNamespace(unscheduled_customers=())

    monkeypatch.setattr(unified_engine, "construct_cluster_solution", construct)
    monkeypatch.setattr(unified_engine, "violation_vector", lambda *_: ZERO)
    monkeypatch.setattr(unified_engine, "validate_solution", lambda *_: [])
    monkeypatch.setattr(unified_engine, "solution_fingerprint", lambda *_: "feasible")
    monkeypatch.setattr(unified_engine, "_local_ratio", lambda *_: 1.0)

    result = unified_engine.solve_cold_start(_instance(), num_seeds=10)

    assert calls == [0]
    assert result.seeds_constructed == 1
    assert result.quantity_repairs == 0
