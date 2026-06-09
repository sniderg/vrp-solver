from __future__ import annotations

from pathlib import Path

from vrp_solver.cli import build_parser
from vrp_solver.contest import score_prefix_with_feasibility_tail
from vrp_solver.model import Solution
from vrp_solver.solver.robust_batch import (
    RobustBatchResult,
    _target_horizon_days,
    write_results_csv,
)

from .test_scenario import tiny_instance


def test_target_horizon_clamps_to_instance_horizon() -> None:
    instance = tiny_instance(forecast=(1.0, 1.0, 1.0))

    assert _target_horizon_days(instance, None) == 3
    assert _target_horizon_days(instance, 30) == 3
    assert _target_horizon_days(instance, 2) == 2


def test_robust_batch_result_writes_csv(tmp_path: Path) -> None:
    score = score_prefix_with_feasibility_tail(
        tiny_instance(forecast=(1.0,)),
        Solution(shifts=()),
        score_days=1,
        feasibility_days=1,
    )
    result = RobustBatchResult.from_score(
        instance="V2.test",
        horizon_days=1,
        output_xml=tmp_path / "solution.xml",
        score=score,
        steps=(),
    )

    csv_path = tmp_path / "summary.csv"
    write_results_csv([result], csv_path)

    content = csv_path.read_text()
    assert "instance,horizon_days,output_xml" in content
    assert "V2.test,1" in content


def test_robust_batch_cli_parses_quick_mode() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "robust-batch-rescue",
            "/tmp/robust-out",
            "--instances",
            "V2.12",
            "--horizons",
            "14",
            "--quick",
            "--first-week-rescue-preset",
            "--candidate-cache-dir",
            "/tmp/robust-cache",
            "--no-rebalance",
        ]
    )

    assert args.command == "robust-batch-rescue"
    assert args.instances == "V2.12"
    assert args.horizons == "14"
    assert args.quick is True
    assert args.first_week_rescue_preset is True
    assert str(args.candidate_cache_dir) == "/tmp/robust-cache"
    assert args.no_rebalance is True
