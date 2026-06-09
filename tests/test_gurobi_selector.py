from __future__ import annotations

import importlib.util

import pytest

from vrp_solver.solver.highs_selector import SelectorConfig, _GurobiSelectorModel


@pytest.mark.skipif(importlib.util.find_spec("gurobipy") is None, reason="gurobipy not installed")
def test_direct_gurobi_selector_model_solves_small_binary_problem() -> None:
    model = _GurobiSelectorModel(SelectorConfig(time_limit=5.0))
    model.addCol(-1.0, 0.0, 1.0, 0, [], [])
    x = model.getNumCol() - 1
    model.changeColIntegrality(x, "B")
    model.addCol(0.0, 0.0, 1.0, 0, [], [])
    y = model.getNumCol() - 1
    model.changeColIntegrality(y, "B")
    model.addRow(0.0, 1.0, 2, [x, y], [1.0, 1.0])

    status, values = model.optimize()

    assert values is not None
    assert status in {"Optimal", "TimeLimit"}
    assert values[x] > 0.5


@pytest.mark.skipif(importlib.util.find_spec("gurobipy") is None, reason="gurobipy not installed")
def test_direct_gurobi_selector_model_accepts_mip_start() -> None:
    model = _GurobiSelectorModel(SelectorConfig(time_limit=5.0))
    model.addCol(0.0, 0.0, 1.0, 0, [], [])
    x = model.getNumCol() - 1
    model.changeColIntegrality(x, "B")
    model.set_start(x, 1.0)
    model.model.update()

    assert model.vars[x].Start == 1.0
