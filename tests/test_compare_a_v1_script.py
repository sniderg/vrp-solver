from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "compare_a_v1.py"
    spec = importlib.util.spec_from_file_location("compare_a_v1", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_improve_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "improve_a_v1.py"
    spec = importlib.util.spec_from_file_location("improve_a_v1", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_scan_candidates_does_not_match_v11_to_v110(tmp_path: Path) -> None:
    module = _load_script_module()
    wanted = tmp_path / "v1_1.1_candidate.xml"
    unwanted = tmp_path / "v1_1.10_candidate.xml"
    wanted.touch()
    unwanted.touch()

    candidates = module._scan_candidates(tmp_path, "V_1.1")

    assert candidates == [wanted]


def test_scan_candidates_matches_result_minor_suffix(tmp_path: Path) -> None:
    module = _load_script_module()
    wanted = tmp_path / "v1_1.9_rescued_feasible.xml"
    wanted.touch()

    candidates = module._scan_candidates(tmp_path, "V_1.9")

    assert candidates == [wanted]


def test_source_cleanup_rebalances_delivery_to_remaining_load() -> None:
    from tests.test_scenario import tiny_instance
    from vrp_solver.model import Operation, Shift

    module = _load_improve_script_module()
    instance = tiny_instance()
    shift = Shift(
        index=0,
        driver=0,
        trailer=0,
        start=0,
        operations=(
            Operation(point=1, arrival=0, quantity=-40.0),
            Operation(point=2, arrival=1, quantity=30.0),
            Operation(point=1, arrival=2, quantity=-50.0),
            Operation(point=2, arrival=3, quantity=50.0),
        ),
    )

    cleaned = module._remove_source_and_rebalance_shift(instance, shift, 2)

    assert cleaned is not None
    assert cleaned.operations == (
        Operation(point=1, arrival=0, quantity=-40.0),
        Operation(point=2, arrival=1, quantity=30.0),
        Operation(point=2, arrival=3, quantity=10.0),
    )


def test_route_reorder_moves_preserve_operations_and_quantities() -> None:
    from tests.test_scenario import tiny_instance
    from vrp_solver.model import Operation, Shift, Solution

    module = _load_improve_script_module()
    shift = Shift(
        index=0,
        driver=0,
        trailer=0,
        start=0,
        operations=(
            Operation(point=1, arrival=0, quantity=-40.0),
            Operation(point=2, arrival=1, quantity=20.0),
            Operation(point=2, arrival=2, quantity=20.0),
        ),
    )

    moves = list(module.route_reorder_moves(tiny_instance(), Solution((shift,)), limit=2))

    assert [name for name, _solution in moves] == ["swap_s0_o1_2"]
    for _name, solution in moves:
        assert sorted(op.quantity for op in solution.shifts[0].operations) == [-40.0, 20.0, 20.0]


def test_first_improving_move_skips_official_when_local_ratio_not_better(monkeypatch) -> None:
    from tests.test_scenario import tiny_instance
    from vrp_solver.model import Solution

    module = _load_improve_script_module()
    incumbent = module.CandidateScore(
        feasible=True,
        local_errors=0,
        local_ratio=1.0,
        official_valid=True,
        official_ratio=1.0,
        official_first_rule="",
    )
    official_calls = []

    def fake_local(_instance, _solution):
        return module.CandidateScore(
            feasible=True,
            local_errors=0,
            local_ratio=1.0,
            official_valid=False,
            official_ratio=float("inf"),
            official_first_rule="official checker not run",
        )

    def fake_official(*_args):
        official_calls.append(True)
        return incumbent

    monkeypatch.setattr(module, "score_solution_local", fake_local)
    monkeypatch.setattr(module, "score_solution_official", fake_official)

    result, stats = module.first_improving_move(
        tiny_instance(),
        Path("instance.xml"),
        incumbent,
        [("noop", Solution(shifts=()))],
    )

    assert result is None
    assert stats.tried == 1
    assert stats.local_feasible == 1
    assert stats.local_better == 0
    assert stats.official_checked == 0
    assert official_calls == []
