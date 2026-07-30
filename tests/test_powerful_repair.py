from types import SimpleNamespace

from vrp_solver.solver.recovered_search import _ucb_select
from vrp_solver.solver.surgical_search import _accept_move, _key


def test_ucb_tries_every_operator_before_reusing_one():
    names = ("a", "b", "c", "d")
    pulls = {name: 0 for name in names}
    rewards = {name: 0.0 for name in names}
    selected = []
    for iteration in range(4):
        name = _ucb_select(names, pulls, rewards, iteration, 1.25)
        selected.append(name)
        pulls[name] += 1
    assert selected == ["a", "b", "c", "d"]


def test_ucb_can_return_to_a_productive_operator():
    names = ("productive", "other")
    pulls = {"productive": 3, "other": 3}
    rewards = {"productive": 9.0, "other": 0.0}
    assert _ucb_select(names, pulls, rewards, 6, 1.25) == "productive"


def test_surgical_tie_break_uses_logistic_ratio_not_raw_cost():
    incumbent = SimpleNamespace(
        hard_violations=0,
        feasibility_errors=10,
        safety_kg_min=100.0,
        scored_estimated_cost=100.0,
        scored_delivered_quantity=10_000.0,
    )
    lower_cost_but_worse_ratio = SimpleNamespace(
        hard_violations=0,
        feasibility_errors=10,
        safety_kg_min=100.0,
        scored_estimated_cost=90.0,
        scored_delivered_quantity=8_000.0,
    )
    assert _key(incumbent) < _key(lower_cost_but_worse_ratio)


def test_surgical_perturbation_can_cross_one_error_bridge():
    current = SimpleNamespace(
        hard_violations=0,
        feasibility_errors=10,
        safety_kg_min=100.0,
        scored_estimated_cost=100.0,
        scored_delivered_quantity=10_000.0,
    )
    bridge = SimpleNamespace(
        hard_violations=0,
        feasibility_errors=11,
        safety_kg_min=100.0,
        scored_estimated_cost=100.0,
        scored_delivered_quantity=10_000.0,
    )

    class AlwaysAccept:
        @staticmethod
        def random():
            return 1.0

    assert not _accept_move(current, bridge, 0, AlwaysAccept())
    assert _accept_move(current, bridge, 8, AlwaysAccept())
