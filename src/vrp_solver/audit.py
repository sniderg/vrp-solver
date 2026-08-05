"""Independent, file-backed audit of one candidate solution.

This module is intentionally outside ``solver``.  It never generates or
repairs a route: it only observes an already-written XML and produces four
separate artifacts.  The released checker remains the sole publication gate.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from .analysis import customer_inventory_summary, summarize_solution
from .inventory import tank_events
from .model import Instance, Solution
from .official_verify import OfficialVerification, verify_v2_solution
from .rules import validate_solution


def audit_solution(
    instance: Instance,
    solution: Solution,
    *,
    instance_xml: Path,
    solution_xml: Path,
    output_dir: Path,
    checker_archive: Path,
    official_timeout: float = 180.0,
) -> OfficialVerification:
    """Write independent simulator, native-rule, analyzer, and official artifacts.

    ``published`` in ``manifest.json`` is deliberately synonymous with the
    released checker's acceptance.  A clean native-rule or simulator report is
    diagnostic evidence only, never a substitute verdict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    simulator_dir = output_dir / "simulator"
    native_checker_dir = output_dir / "native_checker"
    analyzer_dir = output_dir / "analyzer"
    official_dir = output_dir / "official_checker"
    for directory in (simulator_dir, native_checker_dir, analyzer_dir, official_dir):
        directory.mkdir(exist_ok=True)

    events = tank_events(instance, solution)
    _write_rows(simulator_dir / "tank_events.csv", events)

    violations = validate_solution(instance, solution)
    _write_rows(native_checker_dir / "violations.csv", violations)

    shifts = summarize_solution(instance, solution)
    customers = customer_inventory_summary(instance, solution)
    _write_rows(analyzer_dir / "shifts.csv", shifts)
    _write_rows(analyzer_dir / "customers.csv", customers)

    official = verify_v2_solution(
        instance_xml,
        solution_xml,
        checker_archive=checker_archive,
        timeout_seconds=official_timeout,
    )
    (official_dir / "checker_output.txt").write_text(official.output, encoding="utf-8")

    horizon_end = instance.horizon * instance.unit
    out_of_horizon = sum(
        1
        for shift in solution.shifts
        for operation in shift.operations
        if operation.arrival < 0 or operation.arrival >= horizon_end
    )
    native_errors = sum(violation.severity == "error" for violation in violations)
    manifest = {
        "instance_xml": str(instance_xml.resolve()),
        "solution_xml": str(solution_xml.resolve()),
        "artifacts": {
            "simulator": "simulator/tank_events.csv",
            "native_checker": "native_checker/violations.csv",
            "analyzer_shifts": "analyzer/shifts.csv",
            "analyzer_customers": "analyzer/customers.csv",
            "official_checker": "official_checker/checker_output.txt",
        },
        "native_rule_errors": native_errors,
        "simulator_tank_events": len(events),
        "out_of_horizon_arrival_operations": out_of_horizon,
        "official_status": official.status,
        "official_valid": official.valid,
        "official_logistic_ratio": official.logistic_ratio,
        "official_rule_counts": official.rule_counts,
        "published": official.valid,
        "publication_rule": "published is true only when the released ROADEF checker accepts this exact XML",
        "local_warning": (
            "Native simulator diagnostics are not an acceptance oracle; "
            "out-of-horizon arrivals require special scrutiny because legacy local paths may clamp them."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return official


def _write_rows(path: Path, rows: list[object]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    serialized = [asdict(row) for row in rows]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(serialized[0]))
        writer.writeheader()
        writer.writerows(serialized)
