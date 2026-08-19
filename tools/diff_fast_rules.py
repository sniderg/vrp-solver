"""Locate a fast-state / legacy-scorer disagreement down to the rule code.

Replays the same random mutation sequence the equivalence test uses and, at the
first divergence, prints the per-code violation breakdown from the legacy
validator next to the fast state's group totals.
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vrp_solver.contest import score_prefix_with_feasibility_tail  # noqa: E402
from vrp_solver.fast.state import SearchState, instance_days  # noqa: E402
from vrp_solver.rules import validate_solution  # noqa: E402
from vrp_solver.xml_io import load_instance, load_solution  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_fast_state import _random_mutation  # noqa: E402


def main(instance_path: str, solution_path: str, seed: int = 20260810) -> int:
    instance = load_instance(instance_path)
    solution = load_solution(solution_path)
    days = instance_days(instance)
    state = SearchState.from_solution(instance, solution, score_days=days)
    rng = random.Random(seed)

    applied = 0
    for _ in range(400):
        if applied >= 60:
            break
        if not _random_mutation(state, rng):
            continue
        applied += 1
        current = state.to_solution()
        fast = state.score()
        reference = score_prefix_with_feasibility_tail(
            instance, current, score_days=days, feasibility_days=days,
            ignore_tail_call_ins=True,
        )
        if fast.feasibility_errors == reference.feasibility_errors:
            continue

        print(f"divergence after {applied} moves: "
              f"fast={fast.feasibility_errors} ref={reference.feasibility_errors}")
        violations = validate_solution(instance, current)
        errors = Counter(v.code for v in violations if v.severity == "error")
        print("\nlegacy per-code error counts:")
        for code, count in sorted(errors.items()):
            print(f"  {code:12s} {count}")
        print("\nfast group totals:")
        print(f"  shift-local  {state._shift_errors}")
        print(f"  trailer      {state._trailer_errors}   (SHI06 + TL01)")
        print(f"  driver       {state._driver_errors}   (DRI01)")
        print(f"  tank         {state._tank_errors}   (DYN01 + QS02)")
        print(f"  call-in      {state._callin_errors}   (QS01)")

        legacy_shift_local = sum(
            count for code, count in errors.items()
            if code in {
                "REF_DRIVER", "REF_TRAILER", "LAY02", "LAY03", "SHI02", "SHI03",
                "SHI04", "SHI05", "SHI11", "SHI16", "QS03", "DRI03", "DRI08",
                "TL03",
            }
        )
        legacy_trailer = errors.get("SHI06", 0) + errors.get("TL01", 0)
        print("\nexpected split from legacy codes:")
        print(f"  shift-local  {legacy_shift_local}"
              f"  delta={state._shift_errors - legacy_shift_local:+d}")
        print(f"  trailer      {legacy_trailer}"
              f"  delta={state._trailer_errors - legacy_trailer:+d}")
        print(f"  driver       {errors.get('DRI01', 0)}"
              f"  delta={state._driver_errors - errors.get('DRI01', 0):+d}")
        print(f"  tank         {errors.get('DYN01', 0) + errors.get('QS02', 0)}"
              f"  delta={state._tank_errors - (errors.get('DYN01', 0) + errors.get('QS02', 0)):+d}")
        print(f"  call-in      {errors.get('QS01', 0)}"
              f"  delta={state._callin_errors - errors.get('QS01', 0):+d}")
        return 1

    print(f"no divergence in {applied} applied moves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
