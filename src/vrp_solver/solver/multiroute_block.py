"""Atomic multi-route pressure repair with explicit reload relocation."""
from __future__ import annotations

from dataclasses import dataclass

from ..diagnostics import assess_atomic_repair, solution_fingerprint
from ..highs_repair import repair_quantities_with_highs
from ..joint_block_timing import generate_pressure_substitution_ejections
from ..model import Instance, Solution
from .pressure import PressurePoint


@dataclass(frozen=True)
class MultiRouteBlockFunnel:
    timed_topologies: int = 0
    unique_topologies: int = 0
    hard_quantity_feasible: int = 0
    strict_improvements: int = 0


def repair_pressure_multiroute_block(
    instance: Instance,
    solution: Solution,
    pressure: PressurePoint,
    *,
    end_day: int,
    max_topologies: int = 32,
    time_limit_per_model: float = 10.0,
) -> tuple[list[Solution], MultiRouteBlockFunnel]:
    """Rebuild, jointly retime, hard-repair, and replay one pressure block."""
    topologies = generate_pressure_substitution_ejections(
        instance,
        solution,
        customer_point=pressure.customer,
        first_minute=pressure.first_minute,
        max_candidates=max_topologies,
    )
    seen: set[str] = set()
    accepted: list[Solution] = []
    unique = feasible = improved = 0
    for topology in topologies:
        fingerprint = solution_fingerprint(topology)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique += 1
        repaired, report = repair_quantities_with_highs(
            instance,
            topology,
            score_days=end_day,
            feasibility_days=end_day,
            ignore_tail_call_ins=True,
            quantity_objective="min-delivered",
            strict_inventory=True,
            time_limit_seconds=time_limit_per_model,
        )
        if report.status != "Optimal":
            repaired, report = repair_quantities_with_highs(
                instance,
                topology,
                score_days=end_day,
                feasibility_days=end_day,
                ignore_tail_call_ins=True,
                quantity_objective="min-delivered",
                strict_inventory=False,
                time_limit_seconds=time_limit_per_model,
            )
        if report.status != "Optimal":
            continue

        feasible += 1
        if assess_atomic_repair(instance, solution, repaired).accepted:
            improved += 1
            accepted.append(repaired)
    return accepted, MultiRouteBlockFunnel(
        timed_topologies=len(topologies),
        unique_topologies=unique,
        hard_quantity_feasible=feasible,
        strict_improvements=improved,
    )
