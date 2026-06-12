from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

from vrp_solver.contest import score_prefix_with_feasibility_tail
from vrp_solver.highs_repair import repair_quantities_with_highs
from vrp_solver.rules import validate_solution
from vrp_solver.solver.column_loop import (
    ColumnLoopConfig,
    _apply_delivery_budgets,
    _cached_generate_priced_batches,
    _filter_prefix_conflicts,
    _pressure_customers,
    _rescue_config,
    _safe_delivery_budgets,
    _top_diverse_columns,
)
from vrp_solver.solver.highs_selector import (
    SelectorConfig,
    _inventory_pressure_by_customer,
    select_shifts_with_highs,
)
from vrp_solver.solver.targeted_rescue import (
    MINUTES_PER_DAY,
    _baseline_window_shifts,
    _dedupe_reindex,
    _keep_shifts_started_before,
    normalize_source_loads,
)
from vrp_solver.xml_io import load_instance, load_solution, save_solution


ROOT = Path(__file__).resolve().parents[1]
INSTANCE = ROOT / "roadef_2016_data/set_B/Instances_B_V25-11042016/V2.12.xml"
SEED = ROOT / "scratch/b_v2_build_diag_layer_check/V2.12/V2.12_cluster_seed.xml"
OUT = ROOT / "scratch/b_v2_selected_debug"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    instance = load_instance(INSTANCE)
    baseline = load_solution(SEED)
    config = ColumnLoopConfig(
        start_day=0,
        end_day=10,
        replace_from_day=0,
        iterations=1,
        max_pressure_customers=10,
        samples_per_customer=3,
        max_chain_length=3,
        nearest_chain_neighbors=6,
        max_candidates_per_iteration=350,
        multi_reload_columns=True,
        selector_time_limit=60.0,
        selector_phase="feasibility",
        commit_end_day=10,
    )
    fixed_prefix = _keep_shifts_started_before(
        baseline,
        config.replace_from_day * MINUTES_PER_DAY,
    )
    baseline_window = list(_baseline_window_shifts(baseline, _rescue_config(config)))
    pool = _dedupe_reindex(baseline_window)
    pressure_customers = _pressure_customers(instance, baseline, config)
    generated = _cached_generate_priced_batches(instance, fixed_prefix, pressure_customers, config)
    generated = _filter_prefix_conflicts(instance, fixed_prefix, generated)
    pressure = _inventory_pressure_by_customer(instance, fixed_prefix, config.replace_from_day, config.end_day)
    generated = _top_diverse_columns(instance, generated, pressure, config)
    budgets = _safe_delivery_budgets(
        instance,
        fixed_prefix,
        baseline_window,
        config.replace_from_day,
        config.end_day,
    )
    generated = _apply_delivery_budgets(instance, generated, budgets)
    pool = _dedupe_reindex([*pool, *generated])
    additive_generated = _filter_prefix_conflicts(instance, baseline, generated)
    selected = select_shifts_with_highs(
        instance,
        baseline,
        additive_generated,
        start_day=config.replace_from_day,
        end_day=config.end_day,
        pressure_pricing=True,
        selector_config=SelectorConfig(
            time_limit=config.selector_time_limit,
            selector_phase=config.selector_phase,
        ),
    )
    selected = optimize_like_loop(instance, selected)
    repaired, _ = repair_quantities_with_highs(
        instance,
        selected,
        score_days=config.end_day,
        feasibility_days=config.end_day,
        quantity_objective=config.quantity_objective,
    )
    repaired = normalize_source_loads(instance, repaired)
    save_solution(selected, OUT / "selected.xml")
    save_solution(repaired, OUT / "repaired.xml")
    print("pressure", pressure_customers)
    print(
        "generated",
        len(generated),
        "additive_generated",
        len(additive_generated),
        "pool",
        len(pool),
        "selected",
        len(selected.shifts),
    )
    report("selected", instance, selected)
    report("repaired", instance, repaired)


def optimize_like_loop(instance, solution):
    from vrp_solver.highs_time_opt import optimize_solution_times

    return normalize_source_loads(instance, optimize_solution_times(instance, solution))


def report(label, instance, solution) -> None:
    score = score_prefix_with_feasibility_tail(
        instance,
        solution,
        score_days=10,
        feasibility_days=10,
    )
    violations = [v for v in validate_solution(instance, solution) if v.severity == "error"]
    print(
        label,
        "feasible",
        score.feasible,
        "errors",
        score.feasibility_errors,
        "hard",
        score.hard_violations,
        "shifts",
        len(solution.shifts),
    )
    print(label, "codes", Counter(v.code for v in violations).most_common(12))
    print(label, "hard-ish", Counter(v.code for v in violations if v.code in {"DYN01", "SHI06", "REF_DRIVER", "REF_TRAILER"}).most_common())
    for violation in violations[:20]:
        print(label, "sample", violation)


if __name__ == "__main__":
    main()
