from __future__ import annotations

import concurrent.futures
import math
import os
import time
from dataclasses import dataclass
from typing import Callable

from vrp_solver.contest import score_prefix_with_feasibility_tail, truncate_solution_atomic
from vrp_solver.diagnostics import solution_fingerprint
from vrp_solver.highs_repair import repair_quantities_with_highs
from vrp_solver.model import Instance, Solution
from vrp_solver.solver.cluster_greedy import construct_cluster_solution
from vrp_solver.solver.surgical_search import SurgicalSearchConfig, surgical_search
from vrp_solver.solver.targeted_rescue import RescueConfig, targeted_rescue

MINUTES_PER_DAY = 1440


def _evaluate_candidate_strategy(
    instance: Instance,
    committed_prefix: Solution,
    strat: dict,
    window_end_day: int,
    next_commit_day: int,
) -> tuple[Solution, Solution, object] | None:
    try:
        sol_ext, _ = construct_cluster_solution(
            instance,
            initial_solution=committed_prefix,
            need_ordering=strat["need_ordering"],
            neighborhood_size=strat["neighborhood_size"],
            proactive_reload_ratio=strat["proactive_reload_ratio"],
            long_window_urgency_override=True,
            tie_break_seed=strat["seed"],
        )
        sol_window = truncate_solution_atomic(sol_ext, window_end_day * MINUTES_PER_DAY)
        score_commit = score_prefix_with_feasibility_tail(
            instance,
            sol_window,
            score_days=next_commit_day,
            feasibility_days=next_commit_day,
        )
        if score_commit.feasibility_errors > 0 and score_commit.feasibility_errors <= 100:
            repaired_sol, _ = repair_quantities_with_highs(instance, sol_window, score_days=window_end_day)
            rep_score = score_prefix_with_feasibility_tail(
                instance,
                repaired_sol,
                score_days=next_commit_day,
                feasibility_days=next_commit_day,
            )
            if rep_score.feasible or rep_score.feasibility_errors < score_commit.feasibility_errors:
                sol_window = repaired_sol
                score_commit = rep_score
        return (sol_window, committed_prefix, score_commit)
    except Exception:
        return None


@dataclass(frozen=True)
class RollingBeamNode:
    committed_prefix: Solution
    projected_solution: Solution
    commit_day: int
    score_errors: int
    cost: float
    logistic_ratio: float


def solve_rolling_horizon_beam(
    instance: Instance,
    *,
    beam_width: int = 8,
    lookahead_days: int = 6,
    commit_step_days: int = 1,
    time_limit_seconds: float = 2400.0,
    overlap_days: int = 2,
    strict_feasibility_gate: bool = True,
    progress: Callable[[str], None] | None = print,
) -> Solution:
    """Solve long-horizon instances using fine-grained rolling beam search with lookahead carryover."""
    start_time = time.monotonic()
    deadline = start_time + time_limit_seconds if time_limit_seconds > 0 else None
    horizon_days = max(1, math.ceil(instance.horizon * instance.unit / MINUTES_PER_DAY))

    empty_sol = Solution(())
    beam = [
        RollingBeamNode(
            committed_prefix=empty_sol,
            projected_solution=empty_sol,
            commit_day=0,
            score_errors=0,
            cost=0.0,
            logistic_ratio=0.0,
        )
    ]

    exploration_strategies = [
        {"need_ordering": "urgency-band", "neighborhood_size": 8, "proactive_reload_ratio": 0.48, "seed": 0},
        {"need_ordering": "scarcity", "neighborhood_size": 8, "proactive_reload_ratio": 0.52, "seed": 1},
        {"need_ordering": "scarcity", "neighborhood_size": 12, "proactive_reload_ratio": 0.65, "seed": 2},
        {"need_ordering": "urgency-band", "neighborhood_size": 12, "proactive_reload_ratio": 0.65, "seed": 3},
        {"need_ordering": "scarcity", "neighborhood_size": 10, "proactive_reload_ratio": 0.55, "seed": 4},
        {"need_ordering": "scarcity", "neighborhood_size": 12, "proactive_reload_ratio": 0.58, "seed": 5},
    ]

    for current_commit_day in range(0, horizon_days, commit_step_days):
        if deadline is not None and time.monotonic() >= deadline:
            if progress:
                progress(f"Rolling beam deadline reached at day {current_commit_day}")
            break

        next_commit_day = min(current_commit_day + commit_step_days, horizon_days)
        window_end_day = min(current_commit_day + commit_step_days + lookahead_days, horizon_days)

        if progress:
            progress(
                f"\n--- Rolling Window: Day {current_commit_day} -> {next_commit_day} "
                f"(lookahead to day {window_end_day}, active beam: {len(beam)}) ---"
            )

        new_candidates: list[RollingBeamNode] = []
        found_feasible_warm = False

        # Phase 1: Rapid Warm-Start Sweep across all beam nodes
        for node_idx, node in enumerate(beam):
            warm_shifts = tuple(
                s for s in node.projected_solution.shifts
                if s.start < window_end_day * MINUTES_PER_DAY
            )
            if len(warm_shifts) > len(node.committed_prefix.shifts):
                sol_warm = Solution(warm_shifts)
                score_warm = score_prefix_with_feasibility_tail(
                    instance,
                    sol_warm,
                    score_days=next_commit_day,
                    feasibility_days=next_commit_day,
                )
                lr_warm = (
                    score_warm.scored_estimated_cost / score_warm.scored_delivered_quantity
                    if score_warm.scored_delivered_quantity > 0
                    else 0.0
                )
                committed_part = truncate_solution_atomic(sol_warm, next_commit_day * MINUTES_PER_DAY)
                new_candidates.append(
                    RollingBeamNode(
                        committed_prefix=committed_part,
                        projected_solution=sol_warm,
                        commit_day=next_commit_day,
                        score_errors=score_warm.feasibility_errors,
                        cost=score_warm.scored_estimated_cost,
                        logistic_ratio=lr_warm,
                    )
                )
                if score_warm.feasible:
                    found_feasible_warm = True
                    if progress:
                        progress(
                            f"  Node {node_idx} Warm-Start Feasible: Day {next_commit_day} "
                            f"(shifts={len(committed_part.shifts)}, cost={score_warm.scored_estimated_cost:.2f}, LR={lr_warm:.4f})"
                        )

        # Phase 2: Concurrent Exploratory Construction with Fix-and-Optimize Lookback
        freeze_day = max(0, current_commit_day - overlap_days)
        tasks = []
        active_strats = exploration_strategies[:1] if found_feasible_warm else exploration_strategies
        for node in beam:
            freeze_pfx = (
                truncate_solution_atomic(node.committed_prefix, freeze_day * MINUTES_PER_DAY)
                if freeze_day < current_commit_day
                else node.committed_prefix
            )
            for strat in active_strats:
                tasks.append((freeze_pfx, strat))

        raw_exploratory: list[tuple[Solution, Solution, object]] = []
        if tasks:
            with concurrent.futures.ProcessPoolExecutor(max_workers=min(len(tasks), os.cpu_count() or 4)) as executor:
                futures = [
                    executor.submit(
                        _evaluate_candidate_strategy,
                        instance,
                        committed_pfx,
                        strat,
                        window_end_day,
                        next_commit_day,
                    )
                    for (committed_pfx, strat) in tasks
                ]
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    if res is not None:
                        raw_exploratory.append(res)

        # Sort raw exploratory candidates by feasibility errors
        raw_exploratory.sort(key=lambda item: item[2].feasibility_errors)

        # Phase 3: Focused Targeted Recourse & Surgical Search
        recourse_budget = 2 if not found_feasible_warm else 0
        for cand_idx, (sol_window, committed_pfx, score_commit) in enumerate(raw_exploratory):
            if recourse_budget > 0 and not score_commit.feasible and score_commit.feasibility_errors <= 100:
                recourse_budget -= 1
                try:
                    # Step 3a: Targeted Rescue for failing customers
                    cfg = RescueConfig(
                        start_day=0,
                        end_day=window_end_day,
                        replace_from_day=current_commit_day,
                        samples_per_customer=6,
                    )
                    rescued_sol, _ = targeted_rescue(instance, sol_window, config=cfg)
                    resc_score = score_prefix_with_feasibility_tail(
                        instance,
                        rescued_sol,
                        score_days=next_commit_day,
                        feasibility_days=next_commit_day,
                    )
                    if resc_score.feasible or resc_score.feasibility_errors < score_commit.feasibility_errors:
                        score_commit = resc_score
                        sol_window = rescued_sol

                    # Step 3b: Surgical search fine tuning if still needed
                    if not score_commit.feasible:
                        time_left = max(5.0, (deadline - time.monotonic()) if deadline else 30.0)
                        surg_time = min(15.0, time_left)
                        repaired_sol, _ = surgical_search(
                            instance,
                            sol_window,
                            config=SurgicalSearchConfig(
                                end_day=next_commit_day,
                                iterations=32,
                                seed=cand_idx,
                                time_limit_seconds=surg_time,
                                workers=1,
                            ),
                            progress=None,
                        )
                        rep_score = score_prefix_with_feasibility_tail(
                            instance,
                            repaired_sol,
                            score_days=next_commit_day,
                            feasibility_days=next_commit_day,
                        )
                        if rep_score.feasible or rep_score.feasibility_errors < score_commit.feasibility_errors:
                            score_commit = rep_score
                            sol_window = repaired_sol
                except Exception:
                    pass

            lr = (
                score_commit.scored_estimated_cost / score_commit.scored_delivered_quantity
                if score_commit and score_commit.scored_delivered_quantity > 0
                else 0.0
            )
            committed_part = truncate_solution_atomic(sol_window, next_commit_day * MINUTES_PER_DAY)
            new_candidates.append(
                RollingBeamNode(
                    committed_prefix=committed_part,
                    projected_solution=sol_window,
                    commit_day=next_commit_day,
                    score_errors=score_commit.feasibility_errors,
                    cost=score_commit.scored_estimated_cost if score_commit else 0.0,
                    logistic_ratio=lr,
                )
            )

        if not new_candidates:
            if progress:
                progress(f"Beam collapsed at day {next_commit_day}")
            break

        # Deduplicate fingerprints
        seen_fps: set[tuple[object, ...]] = set()
        deduped: list[RollingBeamNode] = []
        for cand in new_candidates:
            fp = solution_fingerprint(cand.committed_prefix)
            if fp not in seen_fps:
                seen_fps.add(fp)
                deduped.append(cand)

        # Sort: prioritize 0 errors, then lowest cost (preserves fleet reserve flexibility)
        deduped.sort(key=lambda x: (x.score_errors, x.cost))

        # Diversity-preserving beam selection: retain top nodes across distinct shift topologies
        selected: list[RollingBeamNode] = []
        seen_shift_counts: set[int] = set()
        for cand in deduped:
            s_count = len(cand.committed_prefix.shifts)
            if s_count not in seen_shift_counts and len(selected) < max(2, beam_width // 2):
                seen_shift_counts.add(s_count)
                selected.append(cand)

        for cand in deduped:
            if cand not in selected and len(selected) < beam_width:
                selected.append(cand)

        beam = selected[:beam_width]
        best_node = beam[0]
        if progress:
            progress(
                f"==> Day {next_commit_day} Advance. Best candidate errors={best_node.score_errors}, "
                f"shifts={len(best_node.committed_prefix.shifts)}, cost={best_node.cost:.2f}, LR={best_node.logistic_ratio:.4f}"
            )

        if strict_feasibility_gate:
            max_allowed = 0 if next_commit_day == horizon_days else 100000
            if best_node.score_errors > max_allowed:
                if progress:
                    progress(
                        f"\n[STRICT FEASIBILITY GATE TRIGGERED] Day {next_commit_day} best candidate has "
                        f"{best_node.score_errors} errors (limit: {max_allowed}). Halting trial immediately to avoid wasted compute.\n"
                    )
                break

    return beam[0].committed_prefix if beam else empty_sol
