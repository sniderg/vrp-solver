"""Timing monitor for every HiGHS solve in the codebase.

Purpose: detect the moment we start handing HiGHS models it finds genuinely
hard (nonzero branch-and-bound nodes, minutes of runtime) — that is the
measured trigger for building a native Gurobi path for the model in
question, and not before. See NATIVE_BENCHMARK_RESULTS.md 2026-08-20 and
the selector measurement in skills/solve-roadef-irp/SKILL.md (root-solved
MIPs gain nothing from a stronger solver).

Behavior:
- Every solve taking >= ROADEF_MIP_LOG_MIN_SECONDS (default 1.0) appends a
  CSV line to ROADEF_MIP_LOG (default <cwd>/out/highs_timings.csv):
  timestamp,label,seconds,status,cols,rows,integer_cols,mip_nodes
- Every solve taking >= ROADEF_MIP_SLOW_SECONDS (default 60) additionally
  prints `highs_slow_solve,<label>,<seconds>,...` to stdout so it is
  visible in run logs without opening the CSV.

Solves in hot loops that finish in milliseconds cost one time.monotonic()
call and no I/O.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def timed_run(highs, label: str) -> None:
    """Run ``highs.run()`` with timing, logging, and a slow-solve marker."""
    started = time.monotonic()
    highs.run()
    elapsed = time.monotonic() - started

    log_min = _float_env("ROADEF_MIP_LOG_MIN_SECONDS", 1.0)
    if elapsed < log_min:
        return

    try:
        status = highs.modelStatusToString(highs.getModelStatus())
        cols = highs.getNumCol()
        rows = highs.getNumRow()
        info = highs.getInfo()
        nodes = getattr(info, "mip_node_count", -1)
        lp = highs.getLp()
        integrality = getattr(lp, "integrality_", None)
        integer_cols = (
            sum(1 for kind in integrality if int(kind) != 0) if integrality else 0
        )
    except Exception:
        status, cols, rows, nodes, integer_cols = "unknown", -1, -1, -1, -1

    line = (
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')},{label},"
        f"{elapsed:.3f},{status},{cols},{rows},{integer_cols},{nodes}"
    )
    log_path = Path(os.environ.get("ROADEF_MIP_LOG", "out/highs_timings.csv"))
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not log_path.exists()
        with log_path.open("a", encoding="utf-8") as handle:
            if new_file:
                handle.write(
                    "timestamp,label,seconds,status,cols,rows,integer_cols,mip_nodes\n"
                )
            handle.write(line + "\n")
    except OSError:
        pass

    if elapsed >= _float_env("ROADEF_MIP_SLOW_SECONDS", 60.0):
        print(f"highs_slow_solve,{line}")
