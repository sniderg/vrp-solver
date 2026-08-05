from __future__ import annotations

import json
from pathlib import Path

from vrp_solver.audit import audit_solution
from vrp_solver.official_verify import OfficialVerification
from vrp_solver.xml_io import load_instance, load_solution


ROOT = Path(__file__).resolve().parents[1]
INSTANCE = ROOT / "roadef_2016_data/set_B/Instances_B_V25-11042016/V2.12.xml"
SOLUTION = ROOT / "scratch/native_solutions/V2.12_native.xml"


def test_audit_artifacts_keep_official_verdict_as_publication_gate(tmp_path, monkeypatch) -> None:
    def fake_verify(*_args, **_kwargs):
        return OfficialVerification(
            valid=False,
            status="invalid",
            message="rejected",
            checker_archive=None,
            checker_sha256=None,
            return_code=0,
            logistic_ratio=None,
            rule_counts={"DYN01": 1},
            output="CHECKING FAILED",
        )

    monkeypatch.setattr("vrp_solver.audit.verify_v2_solution", fake_verify)
    instance = load_instance(INSTANCE)
    solution = load_solution(SOLUTION)
    result = audit_solution(
        instance,
        solution,
        instance_xml=INSTANCE,
        solution_xml=SOLUTION,
        output_dir=tmp_path,
        checker_archive=tmp_path / "unused.zip",
    )

    assert result.valid is False
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["published"] is False
    assert (tmp_path / "simulator/tank_events.csv").is_file()
    assert (tmp_path / "native_checker/violations.csv").is_file()
    assert (tmp_path / "analyzer/shifts.csv").is_file()
    assert (tmp_path / "official_checker/checker_output.txt").is_file()
