"""Fail-closed wrapper around the official ROADEF 2016 V2 checker.

This module deliberately does not reuse any native route, inventory, or scoring
logic.  A solution is accepted only if the released checker itself prints its
exact success sentinel.  Missing Mono, a missing archive, an execution error,
or ambiguous output are all failures -- never a local "pass".
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile


V2_ARCHIVE_NAME = "Checker_V2.2_07032016.zip"
V2_EXE_MEMBER = (
    "Challenge_Roadef_EURO_Checker_V2/bin/Release/"
    "IRP_Roadef_Challenge_Checker.exe"
)
VALID_SENTINEL = "THIS OUTPUT IS VALID"
FAILED_SENTINEL = "CHECKING FAILED"
RATIO_RE = re.compile(r"Logistic Ratio\s*=\s*([0-9]+(?:[.,][0-9]+)?)")
RULE_RE = re.compile(r"^\s*\[\s*([^:\]]+)\s*:", re.MULTILINE)


@dataclass(frozen=True)
class OfficialVerification:
    """Result of one authoritative checker execution.

    ``valid`` is true only for an unambiguous official success.  ``status`` is
    intentionally explicit so callers cannot silently treat an unavailable
    checker as a valid result.
    """

    valid: bool
    status: str
    message: str
    checker_archive: Path | None
    checker_sha256: str | None
    return_code: int | None
    logistic_ratio: float | None
    rule_counts: dict[str, int]
    output: str


def default_v2_archive(project_root: Path) -> Path:
    return project_root / "roadef_2016_data" / V2_ARCHIVE_NAME


def verify_v2_solution(
    instance_xml: Path,
    solution_xml: Path,
    *,
    checker_archive: Path,
    timeout_seconds: float = 180.0,
) -> OfficialVerification:
    """Run the released V2 checker from its archive and return a fail-closed result."""
    archive = Path(checker_archive)
    instance = Path(instance_xml)
    solution = Path(solution_xml)
    if not instance.is_file():
        return _unavailable(f"instance XML does not exist: {instance}", archive)
    if not solution.is_file():
        return _unavailable(f"solution XML does not exist: {solution}", archive)
    if not archive.is_file():
        return _unavailable(f"official V2 checker archive does not exist: {archive}", archive)

    if sys.platform == "win32":
        launcher: list[str] = []
    else:
        mono = shutil.which("mono")
        if mono is None:
            return _unavailable("Mono is not installed or not on PATH", archive)
        launcher = [mono]

    archive_hash = _sha256(archive)
    try:
        with zipfile.ZipFile(archive) as bundle:
            try:
                executable = bundle.read(V2_EXE_MEMBER)
            except KeyError:
                return _unavailable(
                    f"archive does not contain the expected released V2 executable: {V2_EXE_MEMBER}",
                    archive,
                    archive_hash,
                )
    except (OSError, zipfile.BadZipFile) as exc:
        return _unavailable(f"cannot read official checker archive: {exc}", archive, archive_hash)

    with tempfile.TemporaryDirectory(prefix="roadef-v2-checker-") as temporary:
        executable_path = Path(temporary) / "IRP_Roadef_Challenge_Checker.exe"
        executable_path.write_bytes(executable)
        try:
            process = subprocess.run(
                launcher + [str(executable_path), str(instance), str(solution)],
                input="\n",
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            return OfficialVerification(
                valid=False,
                status="execution_failed",
                message=f"official checker timed out after {timeout_seconds:g} seconds",
                checker_archive=archive,
                checker_sha256=archive_hash,
                return_code=None,
                logistic_ratio=None,
                rule_counts=_rule_counts(output),
                output=output,
            )
        except OSError as exc:
            return _execution_failed(str(exc), archive, archive_hash)

    output = process.stdout or ""
    ratio_match = RATIO_RE.search(output)
    ratio = float(ratio_match.group(1).replace(",", ".")) if ratio_match else None
    has_valid_sentinel = VALID_SENTINEL in output
    has_failure_sentinel = FAILED_SENTINEL in output
    valid = process.returncode == 0 and has_valid_sentinel and not has_failure_sentinel
    if valid:
        message = "official V2 checker accepted this exact XML"
        status = "valid"
    elif process.returncode != 0:
        message = f"official checker exited with status {process.returncode}"
        status = "execution_failed"
    elif has_failure_sentinel:
        message = "official V2 checker rejected the solution"
        status = "invalid"
    else:
        message = "official checker output was ambiguous; refusing to accept the solution"
        status = "execution_failed"

    return OfficialVerification(
        valid=valid,
        status=status,
        message=message,
        checker_archive=archive,
        checker_sha256=archive_hash,
        return_code=process.returncode,
        logistic_ratio=ratio if valid else None,
        rule_counts=_rule_counts(output),
        output=output,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rule_counts(output: str) -> dict[str, int]:
    return dict(sorted(Counter(match.group(1).strip() for match in RULE_RE.finditer(output)).items()))


def _unavailable(
    message: str,
    archive: Path | None,
    archive_hash: str | None = None,
) -> OfficialVerification:
    return OfficialVerification(False, "unavailable", message, archive, archive_hash, None, None, {}, "")


def _execution_failed(message: str, archive: Path, archive_hash: str) -> OfficialVerification:
    return OfficialVerification(False, "execution_failed", message, archive, archive_hash, None, None, {}, "")
