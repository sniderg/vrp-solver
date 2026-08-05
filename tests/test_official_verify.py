from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from vrp_solver.official_verify import verify_v2_solution


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTANCE = REPO_ROOT / "roadef_2016_data/set_B/Instances_B_V25-11042016/V2.12.xml"
ARCHIVE = REPO_ROOT / "roadef_2016_data/Checker_V2.2_07032016.zip"
REFERENCE = REPO_ROOT / "scratch/set_b_solutions/V2.12.xml"
KNOWN_INVALID = REPO_ROOT / "scratch/native_solutions/V2.12_native.xml"


@pytest.mark.skipif(
    not (shutil.which("mono") and INSTANCE.is_file() and ARCHIVE.is_file()),
    reason="requires Mono plus the released ROADEF V2 checker archive and V2.12 data",
)
def test_official_checker_rejects_known_invalid_native_solution() -> None:
    result = verify_v2_solution(INSTANCE, KNOWN_INVALID, checker_archive=ARCHIVE)

    assert result.valid is False
    assert result.status == "invalid"
    assert result.rule_counts["SHI06"] > 0
    assert result.rule_counts["DYN01"] > 0


@pytest.mark.skipif(
    not (shutil.which("mono") and INSTANCE.is_file() and ARCHIVE.is_file() and REFERENCE.is_file()),
    reason="requires Mono plus the released ROADEF V2 checker archive and V2.12 data",
)
def test_official_checker_accepts_known_valid_reference_solution() -> None:
    result = verify_v2_solution(INSTANCE, REFERENCE, checker_archive=ARCHIVE)

    assert result.valid is True
    assert result.status == "valid"
    assert result.logistic_ratio == pytest.approx(0.018496)
