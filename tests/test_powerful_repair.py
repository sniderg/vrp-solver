from vrp_solver.solver.recovered_search import _ucb_select


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
