from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from vrp_solver.instance_manifest import OFFICIAL_INSTANCE_SHA256, classify_instance
from vrp_solver.official_verify import V2_ARCHIVE_SHA256, verify_v2_solution


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTANCE = REPO_ROOT / "roadef_2016_data/set_B/Instances_B_V25-11042016/V2.12.xml"
ARCHIVE = REPO_ROOT / "roadef_2016_data/Checker_V2.2_07032016.zip"
REFERENCE = REPO_ROOT / "scratch/set_b_solutions/V2.12.xml"
KNOWN_INVALID = REPO_ROOT / "scratch/native_solutions/V2.12_native.xml"


def test_tampered_checker_archive_fails_closed(tmp_path: Path) -> None:
    fake = tmp_path / "Checker_V2.2_07032016.zip"
    fake.write_bytes(b"PK\x03\x04 not the released checker")

    instance = tmp_path / "instance.xml"
    solution = tmp_path / "solution.xml"
    instance.write_text("<instance />")
    solution.write_text("<solution />")
    result = verify_v2_solution(instance, solution, checker_archive=fake)
    assert result.valid is False
    assert result.status == "unavailable"
    assert "SHA-256" in result.message


@pytest.mark.skipif(not ARCHIVE.is_file(), reason="checker archive not present")
def test_local_checker_archive_matches_pinned_release_hash() -> None:
    import hashlib

    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == V2_ARCHIVE_SHA256


@pytest.mark.skipif(not INSTANCE.is_file(), reason="Set B data not present")
def test_local_official_instances_match_roadef_manifest() -> None:
    set_b = INSTANCE.parent
    set_x = REPO_ROOT / "roadef_2016_data/set_X"
    for name in OFFICIAL_INSTANCE_SHA256:
        path = (set_b if name.startswith("V2") else set_x) / name
        if path.is_file():
            assert classify_instance(path) == "official", name


def test_manifest_flags_modified_official_instance(tmp_path: Path) -> None:
    fake = tmp_path / "V2.14.xml"
    fake.write_text("<IRP_Roadef_Challenge_Instance />")
    assert classify_instance(fake) == "MODIFIED-OFFICIAL"
    other = tmp_path / "my_simulated_instance.xml"
    other.write_text("<IRP_Roadef_Challenge_Instance />")
    assert classify_instance(other) == "not-in-manifest"


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
