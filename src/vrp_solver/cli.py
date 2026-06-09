from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .analysis import customer_inventory_summary, summarize_solution
from .contest import score_prefix_with_feasibility_tail
from .solver.greedy import construct_solution
from .solver.cluster_greedy import construct_cluster_solution
from .evaluate import evaluate_solution
from .improve import (
    prune_redundant_shifts,
    remove_redundant_source_visits,
    trim_redundant_deliveries,
)
from .highs_repair import repair_quantities_with_highs
from .inventory import tank_events, tank_violations
from .geo import mds_coordinates, plot_geo, write_geo_csv
from .movement import (
    asymmetry_outliers,
    collocation_groups,
    distance_time_outliers,
    movement_edges,
    nearest_neighbors,
    summarize_matrices,
)
from .penalties import PenaltyWeights, penalty_breakdown
from .replay import (
    build_segments,
    customer_states_at,
    replay_grid,
    resource_states_at,
    status_overview,
)
from .rules import validate_solution
from .rolling import rolling_days, rolling_summary
from .smoothness import period_buckets, smoothness_summary
from .xml_io import load_instance, load_solution, save_solution


CHECKER_EXE = (
    Path(__file__).resolve().parent.parent
    / "roadef_2016_data"
    / "checker_v2"
    / "Challenge_Roadef_EURO_Checker_V2"
    / "bin"
    / "Release"
    / "IRP_Roadef_Challenge_Checker.exe"
)
RATIO_RE = re.compile(r"Logistic Ratio\s*=\s*([0-9]+(?:[.,][0-9]+)?)")
RULES_INDEX = Path(__file__).resolve().parent.parent / "roadef_2016_data" / "rules_index.md"


def cmd_instance_summary(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    customers_by_kind = {
        "vmi": sum(1 for customer in instance.customers if not customer.call_in),
        "call_in": sum(1 for customer in instance.customers if customer.call_in),
    }
    avg_forecast = [
        sum(customer.forecast) / len(customer.forecast)
        for customer in instance.customers
        if customer.forecast
    ]

    print(f"name,{instance.name}")
    print(f"unit_minutes,{instance.unit}")
    print(f"horizon_steps,{instance.horizon}")
    print(f"horizon_hours,{instance.horizon * instance.unit / 60:.2f}")
    print(f"points,{len(instance.time_matrix)}")
    print(f"drivers,{len(instance.drivers)}")
    print(f"trailers,{len(instance.trailers)}")
    print(f"sources,{len(instance.sources)}")
    print(f"customers,{len(instance.customers)}")
    print(f"vmi_customers,{customers_by_kind['vmi']}")
    print(f"call_in_customers,{customers_by_kind['call_in']}")
    print(f"avg_customer_forecast_per_step,{sum(avg_forecast) / len(avg_forecast):.6f}")
    print(f"total_initial_customer_inventory,{sum(c.initial_tank_quantity for c in instance.customers):.3f}")
    print(f"total_customer_safety_level,{sum(c.safety_level for c in instance.customers):.3f}")
    return 0


from .solver.candidate_gen import generate_shift_candidates, GeneratorConfig
from .solver.highs_selector import select_shifts_with_highs
from .solver.rolling_highs import RollingHighsConfig, rolling_highs_select
from .solver.targeted_rescue import RescueConfig, targeted_rescue
from .solver.column_loop import ColumnLoopConfig, column_generation_rescue
from .solver.alns import ALNSConfig, alns_rescue
from .solver.rolling_cg import RollingCGConfig, robust_rolling_rescue
from .solver.scenario import (
    load_forecast_distribution,
    route_wrapped_dummy_distribution,
    write_forecast_distribution_csv,
)
from .solver.calibration import (
    calibrate_forecast_distribution,
    forecast_calibration_report,
    load_calibration_csv,
    write_calibration_csv,
)
from .solver.history import (
    load_realized_consumption_history,
    realized_history_from_solution_week,
    write_realized_consumption_csv,
)
from .solver.backtest import (
    backtest_solution_against_distribution,
    write_backtest_csv,
    write_backtest_summary_csv,
)
from .solver.policy_sweep import (
    load_policy_sweep_csv,
    run_policy_sweep,
    write_policy_sweep_results_csv,
)
from .solver.route_priors import load_route_prior_candidates, prior_shifts, prior_sources
from .solver.route_swap import transactional_route_swap_search
from .solver.robust_batch import (
    default_b_targets,
    first_week_rescue_config,
    robust_b_config,
    run_robust_batch,
    write_results_csv,
)


def cmd_highs_select(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    prefix = load_solution(args.prefix_xml)

    print(f"Generating candidates for days {args.start_day} to {args.end_day}...")
    config = GeneratorConfig(
        max_candidates_per_window=args.candidates_per_window,
        neighborhood_size=args.neighborhood_size,
    )
    candidates = generate_shift_candidates(
        instance, prefix, 
        start_day=args.start_day, 
        end_day=args.end_day,
        config=config
    )
    print(f"Generated {len(candidates)} candidate shifts.")
    
    print("Selecting shifts with HiGHS...")
    solution = select_shifts_with_highs(
        instance, prefix, candidates,
        start_day=args.start_day,
        end_day=args.end_day
    )
    
    save_solution(solution, args.output_xml)
    print(f"Saved solution to {args.output_xml}")
    return 0


def cmd_rolling_highs_select(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    initial_solution = load_solution(args.prefix_xml) if args.prefix_xml else None
    seed_candidate_solution = (
        load_solution(args.candidate_solution_xml)
        if args.candidate_solution_xml
        else None
    )
    config = RollingHighsConfig(
        start_day=args.start_day,
        end_day=args.end_day,
        lookahead_days=args.lookahead_days,
        commit_days=args.commit_days,
        candidates_per_window=args.candidates_per_window,
        neighborhood_size=args.neighborhood_size,
        feasibility_tail_days=args.feasibility_tail_days,
        candidate_source=args.candidate_source,
    )
    solution, steps = rolling_highs_select(
        instance,
        initial_solution=initial_solution,
        seed_candidate_solution=seed_candidate_solution,
        config=config,
        progress=print,
    )
    save_solution(solution, args.output_xml)
    print(f"Saved rolling solution to {args.output_xml}")
    print("day,window_end_day,commit_end_day,generated_candidates,committed_shifts,feasible,errors,hard")
    for step in steps:
        score = step.score
        print(
            f"{step.day},{step.window_end_day},{step.commit_end_day},"
            f"{step.generated_candidates},{step.committed_shifts},"
            f"{score.feasible if score else ''},"
            f"{score.feasibility_errors if score else ''},"
            f"{score.hard_violations if score else ''}"
        )
    return 0


def cmd_targeted_rescue(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    baseline = load_solution(args.solution_xml)
    config = RescueConfig(
        start_day=args.start_day,
        end_day=args.end_day,
        replace_from_day=args.replace_from_day,
        max_customers=args.max_customers,
        samples_per_customer=args.samples_per_customer,
        target_fill_ratio=args.target_fill_ratio,
        max_pre_service_fill_ratio=args.max_pre_service_fill_ratio,
        sample_lookback_days=args.sample_lookback_days,
        max_chain_length=args.max_chain_length,
        nearest_chain_neighbors=args.nearest_chain_neighbors,
        variable_quantity_columns=args.variable_quantity_columns,
        pressure_pricing=not args.no_pressure_pricing,
        normalize_source_loads=not args.no_normalize_source_loads,
        quantity_objective=args.quantity_objective,
    )
    rescued, report = targeted_rescue(instance, baseline, config=config)
    save_solution(rescued, args.output_xml)
    print(f"Saved rescued solution to {args.output_xml}")
    print(f"failing_customers,{','.join(map(str, report.failing_customers))}")
    print(f"generated_candidates,{report.generated_candidates}")
    print(f"selected_extra_shifts,{report.selected_extra_shifts}")
    if report.quantity_repair_status is not None:
        print(f"quantity_repair_status,{report.quantity_repair_status}")
        print(f"quantity_repair_constraints,{report.quantity_repair_constraints}")
    return 0


def cmd_column_generation_rescue(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    baseline = load_solution(args.solution_xml)
    
    ml_priors = None
    if getattr(args, "ml_priors_path", None) is not None:
        from .solver.ml_priors import MLRoutePriors
        ml_priors = MLRoutePriors()
        ml_priors.load(args.ml_priors_path)
        print(f"Loaded ML route priors from {args.ml_priors_path}")

    config = ColumnLoopConfig(
        start_day=args.start_day,
        end_day=args.end_day,
        replace_from_day=args.replace_from_day,
        iterations=args.iterations,
        max_pressure_customers=args.max_pressure_customers,
        neighbors_per_anchor=args.neighbors_per_anchor,
        batch_workers=args.batch_workers,
        samples_per_customer=args.samples_per_customer,
        sample_lookback_days=args.sample_lookback_days,
        max_chain_length=args.max_chain_length,
        nearest_chain_neighbors=args.nearest_chain_neighbors,
        max_candidates_per_iteration=args.max_candidates_per_iteration,
        target_fill_ratio=args.target_fill_ratio,
        max_pre_service_fill_ratio=args.max_pre_service_fill_ratio,
        multi_reload_columns=args.multi_reload_columns,
        max_multi_reload_per_batch=args.max_multi_reload_per_batch,
        normalize_source_loads=not args.no_normalize_source_loads,
        quantity_objective=args.quantity_objective,
    )
    solution, steps = column_generation_rescue(instance, baseline, config=config, ml_priors=ml_priors)
    save_solution(solution, args.output_xml)
    print(f"Saved column-loop solution to {args.output_xml}")
    print("iteration,generated_candidates,pool_size,selected_extra_shifts,feasible,errors,hard,first_safety_breach_minute")
    for step in steps:
        print(
            f"{step.iteration},{step.generated_candidates},{step.pool_size},"
            f"{step.selected_extra_shifts},{step.feasible},"
            f"{step.feasibility_errors},{step.hard_violations},"
            f"{step.first_safety_breach_minute}"
        )
    return 0


def cmd_alns_rescue(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    initial = load_solution(args.solution_xml)
    config = ALNSConfig(
        start_day=args.start_day,
        end_day=args.end_day,
        replace_from_day=args.replace_from_day,
        iterations=args.iterations,
        repair_iterations=args.repair_iterations,
        seed=args.seed,
        initial_temperature=args.initial_temperature,
        cooling_rate=args.cooling_rate,
        max_removed_shifts=args.max_removed_shifts,
        related_customer_count=args.related_customer_count,
        time_band_days=args.time_band_days,
        max_pressure_customers=args.max_pressure_customers,
        samples_per_customer=args.samples_per_customer,
        sample_lookback_days=args.sample_lookback_days,
        max_candidates_per_iteration=args.max_candidates_per_iteration,
        target_fill_ratio=args.target_fill_ratio,
        nearest_chain_neighbors=args.nearest_chain_neighbors,
        multi_reload_columns=args.multi_reload_columns,
        max_multi_reload_per_batch=args.max_multi_reload_per_batch,
        normalize_source_loads=not args.no_normalize_source_loads,
        quantity_objective=args.quantity_objective,
        output_xml=str(args.output_xml),
    )
    solution, steps = alns_rescue(instance, initial, config=config)
    save_solution(solution, args.output_xml)
    print(f"Saved ALNS solution to {args.output_xml}")
    print("iteration,operator,removed_shifts,accepted,new_best,current_errors,current_hard,best_errors,best_hard,first_safety_breach_minute")
    for step in steps:
        print(
            f"{step.iteration},{step.operator},{step.removed_shifts},"
            f"{step.accepted},{step.new_best},"
            f"{step.current_errors},{step.current_hard},"
            f"{step.best_errors},{step.best_hard},"
            f"{step.first_safety_breach_minute}"
        )
    return 0


def cmd_robust_rolling_rescue(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    baseline = load_solution(args.solution_xml)
    forecast_distribution = (
        load_forecast_distribution(instance, args.forecast_input)
        if args.forecast_input is not None
        else None
    )
    if forecast_distribution is not None and args.calibration_input is not None:
        forecast_distribution = calibrate_forecast_distribution(
            forecast_distribution,
            load_calibration_csv(args.calibration_input),
        )
    route_priors = load_route_prior_candidates(instance, args.route_priors) if args.route_priors is not None else ()
    route_prior_candidates = prior_shifts(route_priors)
    config = RollingCGConfig(
        mode=args.mode,
        horizon_days=args.horizon_days,
        commit_days=args.commit_days,
        lookahead_days=args.lookahead_days,
        n_scenarios=args.n_scenarios,
        scenario_seed=args.scenario_seed,
        plan_sigma=args.plan_sigma,
        buffer_sigma=args.buffer_sigma,
        commit_percentile=args.commit_percentile,
        plan_percentile=args.plan_percentile,
        buffer_percentile=args.buffer_percentile,
        cg_iterations=args.cg_iterations,
        max_pressure_customers=args.max_pressure_customers,
        samples_per_customer=args.samples_per_customer,
        max_chain_length=args.max_chain_length,
        nearest_chain_neighbors=args.nearest_chain_neighbors,
        max_candidates_per_iteration=args.max_candidates_per_iteration,
        target_fill_ratio=args.target_fill_ratio,
        multi_reload_columns=args.multi_reload_columns,
        max_pre_service_fill_ratio=args.max_pre_service_fill_ratio,
        normalize_source_loads=not args.no_normalize_source_loads,
        quantity_objective=args.quantity_objective,
        capacity_buffer=args.capacity_buffer,
        max_hedge_retries=args.max_hedge_retries,
        forecast_distribution=forecast_distribution,
        max_rounds=args.max_rounds,
        committed_output_only=args.committed_output_only,
        final_clip_capacity=not args.no_final_clip_capacity,
        progress_log_path=str(args.progress_log) if args.progress_log else None,
        route_prior_candidates=route_prior_candidates,
        route_prior_sources=prior_sources(route_priors),
        selector_time_limit=args.selector_time_limit,
        selector_mip_gap=args.selector_mip_gap,
        selector_threads=args.selector_threads,
        selector_mip_focus=args.selector_mip_focus,
        selector_node_limit=args.selector_node_limit,
        selector_phase=args.selector_phase,
        candidate_cache_dir=str(args.candidate_cache_dir) if args.candidate_cache_dir else None,
        bucket_anchor_cap=args.bucket_anchor_cap,
        bucket_resource_time_cap=args.bucket_resource_time_cap,
        bucket_route_signature_cap=args.bucket_route_signature_cap,
        bucket_source_region_cap=args.bucket_source_region_cap,
        protect_exact_prior_quantities=args.protect_exact_prior_quantities,
        use_prior_incumbent=not args.no_prior_incumbent,
    )
    solution, steps = robust_rolling_rescue(
        instance, baseline, config=config, progress=print
    )
    
    if args.rebalance:
        from .solver.highs_selector import rebalance_drivers

        print("Rebalancing driver workloads...")
        solution = rebalance_drivers(instance, solution)
    save_solution(solution, args.output_xml)
    print(f"Saved robust rolling solution to {args.output_xml}")
    print("round,commit_start,commit_end,solve_end,cg_iters,feasible,errors,hard,first_breach,committed_shifts,total_shifts,scenario_feasible,scenario_failures,retry_count,accepted,rejection_reason")
    for step in steps:
        print(
            f"{step.round_index},{step.commit_start_day},{step.commit_end_day},"
            f"{step.solve_end_day},{step.cg_iterations},{step.feasible},"
            f"{step.feasibility_errors},{step.hard_violations},"
            f"{step.first_safety_breach_minute},{step.committed_shifts},{step.total_shifts},"
            f"{step.scenario_feasible},{step.scenario_failures},{step.retry_count},"
            f"{step.accepted},{step.rejection_reason}"
        )
    return 0


def cmd_robust_batch_rescue(args: argparse.Namespace) -> int:
    from dataclasses import replace

    presets = default_b_targets()
    requested = [name.strip() for name in args.instances.split(",") if name.strip()]
    unknown = [name for name in requested if name not in presets]
    if unknown:
        raise SystemExit(f"Unknown B instance preset(s): {','.join(unknown)}")
    targets = [presets[name] for name in requested]
    if args.horizons != "auto":
        horizon_days = int(args.horizons)
        targets = [replace(target, horizon_days=horizon_days) for target in targets]
    elif args.first_week_rescue_preset:
        targets = [replace(target, horizon_days=7) for target in targets]
    config = first_week_rescue_config(quick=args.quick) if args.first_week_rescue_preset else robust_b_config(quick=args.quick)
    if args.seed is not None:
        config = replace(config, scenario_seed=args.seed)
    if args.candidate_cache_dir is not None:
        config = replace(config, candidate_cache_dir=str(args.candidate_cache_dir))
    results = run_robust_batch(
        targets,
        args.output_dir,
        config=config,
        progress=print,
        rebalance=not args.no_rebalance,
    )
    csv_path = args.output_dir / "robust_batch_results.csv"
    write_results_csv(results, csv_path)
    print(f"Saved robust batch summary to {csv_path}")
    print("instance,horizon_days,feasible,errors,hard,first_breach,cost,scenario_failures,max_retry_count,output_xml")
    for result in results:
        print(
            f"{result.instance},{result.horizon_days},{result.feasible},"
            f"{result.feasibility_errors},{result.hard_violations},"
            f"{result.first_safety_breach_minute},{result.scored_estimated_cost:.6f},"
            f"{result.scenario_failures},{result.max_retry_count},{result.output_xml}"
        )
    return 0


def cmd_dummy_route_ci(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    solution = load_solution(args.solution_xml)
    quantiles = tuple(float(value) for value in args.quantiles.split(",") if value.strip())
    distribution = route_wrapped_dummy_distribution(
        instance,
        solution,
        present_day=args.present_day,
        quantiles=quantiles,
        base_relative_width=args.base_relative_width,
        daily_relative_growth=args.daily_relative_growth,
        max_relative_width=args.max_relative_width,
        route_anchor_width=args.route_anchor_width,
    )
    write_forecast_distribution_csv(distribution, args.output_csv)
    print(f"Saved dummy route-wrapped forecast CI to {args.output_csv}")
    print(
        "customers,{},quantiles,{},rows,{}".format(
            len(distribution.deterministic),
            ",".join(str(q) for q in sorted(distribution.quantiles)),
            sum(len(path) for path in distribution.deterministic.values()),
        )
    )
    return 0


def cmd_week_ahead_ci_rescue(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    baseline = load_solution(args.solution_xml)
    forecast_distribution = load_forecast_distribution(instance, args.forecast_input)
    if args.calibration_input is not None:
        forecast_distribution = calibrate_forecast_distribution(
            forecast_distribution,
            load_calibration_csv(args.calibration_input),
        )
    route_priors = (
        load_route_prior_candidates(instance, args.route_priors, end_day=args.planning_horizon_days)
        if args.route_priors is not None
        else ()
    )
    route_prior_candidates = prior_shifts(route_priors)
    config = RollingCGConfig(
        mode=args.mode,
        horizon_days=args.planning_horizon_days,
        commit_days=7,
        lookahead_days=args.lookahead_days,
        n_scenarios=0,
        plan_percentile=args.plan_percentile,
        buffer_percentile=args.buffer_percentile,
        cg_iterations=args.cg_iterations,
        max_pressure_customers=args.max_pressure_customers,
        samples_per_customer=args.samples_per_customer,
        max_chain_length=args.max_chain_length,
        nearest_chain_neighbors=args.nearest_chain_neighbors,
        max_candidates_per_iteration=args.max_candidates_per_iteration,
        target_fill_ratio=args.target_fill_ratio,
        multi_reload_columns=args.multi_reload_columns,
        max_pre_service_fill_ratio=args.max_pre_service_fill_ratio,
        normalize_source_loads=not args.no_normalize_source_loads,
        quantity_objective=args.quantity_objective,
        capacity_buffer=args.capacity_buffer,
        max_hedge_retries=args.max_hedge_retries,
        forecast_distribution=forecast_distribution,
        max_rounds=1,
        committed_output_only=True,
        final_clip_capacity=False,
        progress_log_path=str(args.progress_log) if args.progress_log else None,
        route_prior_candidates=route_prior_candidates,
        route_prior_sources=prior_sources(route_priors),
        selector_time_limit=args.selector_time_limit,
        selector_mip_gap=args.selector_mip_gap,
        selector_threads=args.selector_threads,
        selector_mip_focus=args.selector_mip_focus,
        selector_node_limit=args.selector_node_limit,
        selector_phase=args.selector_phase,
        candidate_cache_dir=str(args.candidate_cache_dir) if args.candidate_cache_dir else None,
        bucket_anchor_cap=args.bucket_anchor_cap,
        bucket_resource_time_cap=args.bucket_resource_time_cap,
        bucket_route_signature_cap=args.bucket_route_signature_cap,
        bucket_source_region_cap=args.bucket_source_region_cap,
        protect_exact_prior_quantities=args.protect_exact_prior_quantities,
        use_prior_incumbent=not args.no_prior_incumbent,
    )
    solution, steps = robust_rolling_rescue(
        instance,
        baseline,
        config=config,
        progress=print,
    )
    from .solver.highs_selector import rebalance_drivers
    if not args.no_rebalance:
        print("Rebalancing driver workloads...")
        solution = rebalance_drivers(instance, solution)
    save_solution(solution, args.output_xml)
    score = score_prefix_with_feasibility_tail(
        instance,
        solution,
        score_days=7,
        feasibility_days=7,
        ignore_tail_call_ins=True,
    )
    print(f"Saved week-ahead CI solution to {args.output_xml}")
    print(
        "week_days,7,feasible,{},{},{},cost,{:.2f},shifts,{}".format(
            score.feasibility_errors,
            score.hard_violations,
            score.feasible,
            score.scored_estimated_cost,
            len(solution.shifts),
        )
    )
    print("round,commit_start,commit_end,solve_end,cg_iters,feasible,errors,hard,scenario_failures,accepted,rejection_reason")
    for step in steps:
        print(
            f"{step.round_index},{step.commit_start_day},{step.commit_end_day},"
            f"{step.solve_end_day},{step.cg_iterations},{step.feasible},"
            f"{step.feasibility_errors},{step.hard_violations},"
            f"{step.scenario_failures},{step.accepted},{step.rejection_reason}"
        )
    return 0


def cmd_scenario_backtest(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    solution = load_solution(args.solution_xml)
    distribution = load_forecast_distribution(instance, args.forecast_input)
    percentiles = (
        tuple(float(value) for value in args.percentiles.split(",") if value.strip())
        if args.percentiles
        else None
    )
    result = backtest_solution_against_distribution(
        instance,
        solution,
        distribution,
        horizon_days=args.horizon_days,
        percentiles=percentiles,
        ignore_tail_call_ins=not args.include_tail_call_ins,
    )
    if args.output_csv:
        write_backtest_csv(result, args.output_csv)
        print(f"wrote,{args.output_csv}")
    if args.output_summary_csv:
        write_backtest_summary_csv(result, args.output_summary_csv)
        print(f"wrote_summary,{args.output_summary_csv}")
    for key, value in result.summary.flat().items():
        print(f"{key},{value}")
    return 1 if args.fail_on_infeasible and result.summary.infeasible_scenarios else 0


def cmd_consumption_history_check(args: argparse.Namespace) -> int:
    rows = load_realized_consumption_history(args.history_csv)
    if args.output_csv:
        write_realized_consumption_csv(rows, args.output_csv)
        print(f"wrote,{args.output_csv}")
    print(f"rows,{len(rows)}")
    print(f"customers,{len({row.customer_id for row in rows})}")
    return 0


def cmd_solution_history_extract(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    solution = load_solution(args.solution_xml)
    rows = realized_history_from_solution_week(
        instance,
        solution,
        history_days=args.history_days,
    )
    write_realized_consumption_csv(rows, args.output_csv)
    print(f"wrote,{args.output_csv}")
    print(f"rows,{len(rows)}")
    print(f"customers,{len({row.customer_id for row in rows})}")
    print("source,solution_week_instance_consumption")
    return 0


def cmd_forecast_calibration(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    distribution = load_forecast_distribution(instance, args.forecast_csv)
    realized = load_realized_consumption_history(args.realized_csv)
    rows = forecast_calibration_report(
        distribution,
        realized,
        known_customers={customer.index for customer in instance.customers if not customer.call_in},
    )
    if args.output_csv:
        write_calibration_csv(rows, args.output_csv)
        print(f"wrote,{args.output_csv}")
    for row in rows:
        print(",".join(str(value) for value in row.flat().values()))
    return 0


def cmd_robust_policy_sweep(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance)
    baseline = load_solution(args.baseline)
    distribution = load_forecast_distribution(instance, args.forecast_input)
    if args.calibration_input is not None:
        distribution = calibrate_forecast_distribution(
            distribution,
            load_calibration_csv(args.calibration_input),
        )
    base_config = RollingCGConfig(
        mode=args.mode,
        horizon_days=args.horizon_days,
        scenario_seed=args.scenario_seed,
        forecast_distribution=distribution,
        max_rounds=args.max_rounds,
        selector_time_limit=args.selector_time_limit,
        selector_mip_gap=args.selector_mip_gap,
        selector_threads=args.selector_threads,
        selector_mip_focus=args.selector_mip_focus,
        selector_node_limit=args.selector_node_limit,
        selector_phase=args.selector_phase,
        candidate_cache_dir=str(args.candidate_cache_dir) if args.candidate_cache_dir else None,
    )
    policies = load_policy_sweep_csv(args.sweep_csv, base_config)
    results = run_policy_sweep(
        instance,
        baseline,
        distribution,
        policies,
        args.output_dir,
        horizon_days=args.horizon_days,
        progress=print,
        resume=args.resume,
        force=args.force,
        results_csv=args.output_dir / "policy_sweep_results.csv",
    )
    csv_path = args.output_dir / "policy_sweep_results.csv"
    write_policy_sweep_results_csv(results, csv_path)
    print(f"wrote,{csv_path}")
    for result in results:
        print(
            f"{result.policy_id},{result.feasible},{result.mean_cost:.6f},"
            f"{result.failure_rate:.6f},{result.robust_score:.6f},{result.output_xml}"
        )
    return 0


def cmd_solution_summary(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    solution = load_solution(args.solution_xml)
    shifts = summarize_solution(instance, solution)
    inventories = customer_inventory_summary(instance, solution)
    total_delivered = sum(summary.delivered_quantity for summary in shifts)
    total_estimated_cost = sum(summary.estimated_cost for summary in shifts)
    breached = [summary for summary in inventories if summary.first_safety_breach_step is not None]

    print(f"shifts,{len(solution.shifts)}")
    print(f"operations,{sum(summary.operations for summary in shifts)}")
    print(f"delivered_quantity,{total_delivered:.6f}")
    print(f"estimated_distance_cost_plus_time_cost,{total_estimated_cost:.6f}")
    print(f"load_violations,{sum(summary.load_violations for summary in shifts)}")
    print(f"customers_with_safety_breach_by_simple_projection,{len(breached)}")
    print(
        "worst_margin_to_safety,"
        f"{min(summary.min_margin_to_safety for summary in inventories):.6f}"
    )

    if args.shifts_csv:
        with Path(args.shifts_csv).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(shifts[0].__dict__))
            writer.writeheader()
            writer.writerows(summary.__dict__ for summary in shifts)

    if args.inventory_csv:
        with Path(args.inventory_csv).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(inventories[0].__dict__))
            writer.writeheader()
            writer.writerows(summary.__dict__ for summary in inventories)

    return 0


def cmd_customer_targets(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    solution = load_solution(args.solution_xml)
    inventories = customer_inventory_summary(instance, solution)
    customers = instance.customer_by_point

    rows = []
    for summary in inventories:
        customer = customers[summary.point]
        total_forecast = sum(customer.forecast)
        excess_delivery = summary.delivered_quantity - max(
            0.0,
            total_forecast + customer.safety_level - customer.initial_tank_quantity,
        )
        rows.append(
            {
                "point": summary.point,
                "deliveries": summary.deliveries,
                "delivered_quantity": f"{summary.delivered_quantity:.6f}",
                "total_forecast": f"{total_forecast:.6f}",
                "initial_inventory": f"{customer.initial_tank_quantity:.6f}",
                "safety_level": f"{customer.safety_level:.6f}",
                "min_margin_to_safety": f"{summary.min_margin_to_safety:.6f}",
                "final_inventory": f"{summary.final_inventory:.6f}",
                "excess_delivery_estimate": f"{excess_delivery:.6f}",
                "first_safety_breach_step": (
                    "" if summary.first_safety_breach_step is None
                    else summary.first_safety_breach_step
                ),
            }
        )

    rows.sort(
        key=lambda row: (
            -float(row["excess_delivery_estimate"]),
            -int(row["deliveries"]),
            float(row["min_margin_to_safety"]),
        )
    )

    if args.output_csv:
        with Path(args.output_csv).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    for row in rows[: args.limit]:
        print(
            f"point={row['point']} deliveries={row['deliveries']} "
            f"excess={row['excess_delivery_estimate']} "
            f"min_margin={row['min_margin_to_safety']} "
            f"final_inventory={row['final_inventory']}"
        )
    return 0


def cmd_rule_check(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    solution = load_solution(args.solution_xml)
    violations = validate_solution(instance, solution)

    counts: dict[str, int] = {}
    for violation in violations:
        counts[violation.code] = counts.get(violation.code, 0) + 1

    print(f"violations,{len(violations)}")
    for code, count in sorted(counts.items()):
        print(f"{code},{count}")
    error_count = sum(1 for violation in violations if violation.severity == "error")
    warning_count = sum(1 for violation in violations if violation.severity == "warning")
    print(f"errors,{error_count}")
    print(f"warnings,{warning_count}")

    if args.output_csv:
        with Path(args.output_csv).open("w", newline="") as file:
            fieldnames = ["code", "severity", "shift", "operation", "point", "message"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(violation.__dict__ for violation in violations)

    if args.limit:
        for violation in violations[: args.limit]:
            print(
                f"{violation.code}: shift={violation.shift} op={violation.operation} "
                f"point={violation.point} {violation.message}"
            )

    return 1 if error_count and args.fail_on_violation else 0


def cmd_rules_index(args: argparse.Namespace) -> int:
    print(RULES_INDEX.read_text())
    return 0


def cmd_construct_solution(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    solution, report = construct_solution(
        instance,
        safety_buffer=args.safety_buffer,
        max_shifts=args.max_shifts,
    )
    save_solution(solution, args.output_xml)
    print(f"wrote,{args.output_xml}")
    print(f"shifts,{report.shifts}")
    print(f"operations,{report.operations}")
    print(f"delivered_quantity,{report.delivered_quantity:.6f}")
    print(f"exhausted_resources,{report.exhausted_resources}")
    print(f"unscheduled_customers,{len(report.unscheduled_customers)}")
    if report.unscheduled_customers:
        print(
            "unscheduled_customer_ids,"
            + " ".join(str(point) for point in report.unscheduled_customers[: args.limit])
        )
    return 0


def cmd_cluster_construct_solution(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    score_cutoff_minute = (
        None
        if args.score_days is None
        else args.score_days * 1440
    )
    solution, report = construct_cluster_solution(
        instance,
        safety_buffer=args.safety_buffer,
        neighborhood_size=args.neighborhood_size,
        max_shifts=args.max_shifts,
        score_cutoff_minute=score_cutoff_minute,
        terminal_buffer_days=args.terminal_buffer_days,
    )
    save_solution(solution, args.output_xml)
    print(f"wrote,{args.output_xml}")
    print(f"shifts,{report.shifts}")
    print(f"operations,{report.operations}")
    print(f"delivered_quantity,{report.delivered_quantity:.6f}")
    print(f"exhausted_resources,{report.exhausted_resources}")
    print(f"unscheduled_customers,{len(report.unscheduled_customers)}")
    if report.unscheduled_customers:
        print(
            "unscheduled_customer_ids,"
            + " ".join(str(point) for point in report.unscheduled_customers[: args.limit])
        )
    return 0


def cmd_matrix_summary(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    summary = summarize_matrices(instance)
    for key, value in summary.__dict__.items():
        print(f"{key},{value}")
    groups = collocation_groups(instance)
    print(f"collocation_groups,{len(groups)}")
    print(f"collocated_points,{sum(len(group) for group in groups)}")
    return 0


def cmd_export_edges(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    edges = movement_edges(instance)
    with Path(args.output_csv).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(edges[0].__dict__))
        writer.writeheader()
        writer.writerows(edge.__dict__ for edge in edges)
    print(f"wrote,{args.output_csv}")
    print(f"edges,{len(edges)}")
    return 0


def cmd_nearest(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    rows = nearest_neighbors(instance, k=args.k, metric=args.metric)
    with Path(args.output_csv).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote,{args.output_csv}")
    print(f"rows,{len(rows)}")
    return 0


def cmd_collocations(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    groups = collocation_groups(instance)
    rows = [
        {
            "group_id": group_id,
            "size": len(group),
            "points": " ".join(str(point) for point in group),
            "kinds": " ".join(instance.point_kind(point) for point in group),
        }
        for group_id, group in enumerate(groups)
    ]
    if args.output_csv:
        with Path(args.output_csv).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["group_id", "size", "points", "kinds"])
            writer.writeheader()
            writer.writerows(rows)
    print(f"groups,{len(groups)}")
    print(f"points,{sum(len(group) for group in groups)}")
    for row in rows[: args.limit]:
        print(f"group={row['group_id']} size={row['size']} points={row['points']}")
    return 0


def cmd_speed_outliers(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    edges = distance_time_outliers(instance, limit=args.limit)
    if args.output_csv:
        with Path(args.output_csv).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(edges[0].__dict__))
            writer.writeheader()
            writer.writerows(edge.__dict__ for edge in edges)
    for edge in edges:
        print(
            f"{edge.origin}->{edge.destination} "
            f"distance={edge.distance} time={edge.time} "
            f"speed={edge.speed_distance_per_hour:.3f}"
        )
    return 0


def cmd_asymmetry_outliers(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    edges = asymmetry_outliers(instance, metric=args.metric, limit=args.limit)
    if args.output_csv:
        with Path(args.output_csv).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(edges[0].__dict__))
            writer.writeheader()
            writer.writerows(edge.__dict__ for edge in edges)
    for edge in edges:
        print(
            f"{edge.origin}->{edge.destination} "
            f"distance={edge.distance} reverse_distance={edge.reverse_distance} "
            f"time={edge.time} reverse_time={edge.reverse_time} "
            f"distance_delta={edge.distance_delta} time_delta={edge.time_delta}"
        )
    return 0


def cmd_tank_check(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    solution = load_solution(args.solution_xml)
    violations = tank_violations(instance, solution)
    counts: dict[str, int] = {}
    for violation in violations:
        counts[violation.code] = counts.get(violation.code, 0) + 1

    print(f"tank_violations,{len(violations)}")
    for code, count in sorted(counts.items()):
        print(f"{code},{count}")

    if args.violations_csv:
        with Path(args.violations_csv).open("w", newline="") as file:
            fieldnames = ["code", "point", "step", "time_start", "inventory", "limit", "message"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(violation.__dict__ for violation in violations)

    if args.events_csv:
        events = tank_events(instance, solution)
        with Path(args.events_csv).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(events[0].__dict__))
            writer.writeheader()
            writer.writerows(event.__dict__ for event in events)

    for violation in violations[: args.limit]:
        print(
            f"{violation.code}: point={violation.point} step={violation.step} "
            f"time={violation.time_start} inventory={violation.inventory:.6f} "
            f"limit={violation.limit:.6f}"
        )
    return 1 if violations and args.fail_on_violation else 0


def cmd_segments(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    solution = load_solution(args.solution_xml)
    segments = build_segments(instance, solution)
    with Path(args.output_csv).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(segments[0].__dict__))
        writer.writeheader()
        writer.writerows(segment.__dict__ for segment in segments)
    print(f"wrote,{args.output_csv}")
    print(f"segments,{len(segments)}")
    return 0


def cmd_replay_snapshot(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    solution = load_solution(args.solution_xml)
    resources = resource_states_at(instance, solution, args.time)
    customers = customer_states_at(instance, solution, args.time)

    if args.resources_csv:
        with Path(args.resources_csv).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(resources[0].__dict__))
            writer.writeheader()
            writer.writerows(state.__dict__ for state in resources)
    if args.customers_csv:
        with Path(args.customers_csv).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(customers[0].__dict__))
            writer.writeheader()
            writer.writerows(state.__dict__ for state in customers)

    active = [state for state in resources if state.kind not in {"off_shift", "idle_trailer"}]
    vmi_customers = [state for state in customers if not state.call_in]
    low_customers = sorted(vmi_customers, key=lambda state: state.margin_to_safety)[: args.limit]
    print(f"time,{args.time}")
    print(f"active_resource_states,{len(active)}")
    for state in active[: args.limit]:
        print(
            f"{state.kind}: shift={state.shift} driver={state.driver} trailer={state.trailer} "
            f"point={state.point} arc={state.origin}->{state.destination} {state.message}"
        )
    print("lowest_customer_margins")
    for state in low_customers:
        print(
            f"point={state.point} inventory={state.inventory:.6f} "
            f"safety_margin={state.margin_to_safety:.6f} "
            f"capacity_margin={state.margin_to_capacity:.6f}"
        )
    return 0


def cmd_replay_grid(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    solution = load_solution(args.solution_xml)
    resources, customers = replay_grid(
        instance,
        solution,
        start=args.start,
        end=args.end,
        step=args.step,
    )
    with Path(args.resources_csv).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(resources[0].__dict__))
        writer.writeheader()
        writer.writerows(state.__dict__ for state in resources)
    with Path(args.customers_csv).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(customers[0].__dict__))
        writer.writeheader()
        writer.writerows(state.__dict__ for state in customers)
    print(f"wrote_resources,{args.resources_csv}")
    print(f"wrote_customers,{args.customers_csv}")
    print(f"resource_rows,{len(resources)}")
    print(f"customer_rows,{len(customers)}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    solution = load_solution(args.solution_xml)
    status = status_overview(instance, solution, time=args.time, limit=args.limit)

    print(f"time,{status['time']}")
    print(f"forecast_step,{status['step']}")
    print(f"active_resources,{len(status['active_resources'])}")
    print()
    print("active_resources")
    for state in status["active_resources"]:
        print(
            f"shift={state.shift} driver={state.driver} trailer={state.trailer} "
            f"kind={state.kind} point={state.point} arc={state.origin}->{state.destination} "
            f"load={state.trailer_quantity:.3f} drive={state.driving_since_layover} "
            f"{state.message}"
        )
    print()
    print("lowest_vmi_inventory")
    for state in status["low_inventory_customers"]:
        print(
            f"point={state.point} inv={state.inventory:.3f} "
            f"safety_margin={state.margin_to_safety:.3f} "
            f"capacity_margin={state.margin_to_capacity:.3f}"
        )
    print()
    print("nearest_capacity")
    for state in status["near_capacity_customers"]:
        print(
            f"point={state.point} inv={state.inventory:.3f} "
            f"capacity_margin={state.margin_to_capacity:.3f} "
            f"safety_margin={state.margin_to_safety:.3f}"
        )
    print()
    print("deliveries_this_step")
    for state in status["delivered_this_step"][: args.limit]:
        print(
            f"point={state.point} delivered={state.delivered_this_step:.3f} "
            f"consumed={state.consumed_this_step:.3f} inv={state.inventory:.3f}"
        )
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    instance_path = Path(args.instance_xml)
    solution_path = Path(args.solution_xml)
    instance = load_instance(instance_path)
    solution = load_solution(solution_path)
    checker_exe = _default_checker_exe(instance_path)
    evaluation = evaluate_solution(
        instance,
        solution,
        instance_xml=instance_path,
        solution_xml=solution_path,
        checker_exe=checker_exe,
        run_official_checker=args.official,
    )
    row = evaluation.flat()

    if args.output_csv:
        with Path(args.output_csv).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

    for key, value in row.items():
        print(f"{key},{value}")
    return 1 if args.fail_on_error and evaluation.local_errors else 0


def cmd_smoothness(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    solution = load_solution(args.solution_xml)
    buckets = period_buckets(instance, solution, period_minutes=args.period_minutes)
    summary = smoothness_summary(buckets)

    if args.buckets_csv:
        with Path(args.buckets_csv).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(buckets[0].__dict__))
            writer.writeheader()
            writer.writerows(bucket.__dict__ for bucket in buckets)

    if args.summary_csv:
        with Path(args.summary_csv).open("w", newline="") as file:
            row = summary.flat()
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

    for key, value in summary.flat().items():
        print(f"{key},{value}")

    print("period_preview")
    for bucket in buckets[: args.limit]:
        print(
            f"period={bucket.period} minutes={bucket.start_minute}-{bucket.end_minute} "
            f"delivered={bucket.delivered_quantity:.3f} "
            f"shift_starts={bucket.shift_starts} "
            f"delivery_ops={bucket.delivery_operations}"
        )
    return 0


def cmd_rolling_monitor(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    solution = load_solution(args.solution_xml) if args.solution_xml else None
    days = rolling_days(instance, solution, monitor_days=args.days)
    summary = rolling_summary(instance, solution, monitor_days=args.days)

    if args.days_csv:
        with Path(args.days_csv).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(days[0].__dict__))
            writer.writeheader()
            writer.writerows(day.__dict__ for day in days)

    if args.summary_csv:
        row = summary.flat()
        with Path(args.summary_csv).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

    for key, value in summary.flat().items():
        print(f"{key},{value}")
    print("day_preview")
    for day in days[: args.limit]:
        print(
            f"day={day.day} consumption={day.vmi_consumption:.3f} "
            f"required_cum={day.cumulative_required_delivery:.3f} "
            f"planned={day.planned_delivered:.3f} "
            f"gap_required={day.cumulative_delivery_gap_to_required:.3f} "
            f"gap_smooth={day.cumulative_delivery_gap_to_smooth_target:.3f} "
            f"gap_planned_avg={day.cumulative_delivery_gap_to_planned_average:.3f} "
            f"starts={day.planned_shift_starts}"
        )
    return 0


def cmd_penalties(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    solution = load_solution(args.solution_xml)
    weights = PenaltyWeights(
        safety_kg_min=args.safety_kg_min,
        driver_window_min=args.driver_window_min,
        driver_rest_min=args.driver_rest_min,
        max_driving_min=args.max_driving_min,
        timing_min=args.timing_min,
        customer_window_min=args.customer_window_min,
        smoothness_cv=args.smoothness_cv,
        frontload_share=args.frontload_share,
        hard_violation=args.hard_violation,
    )
    penalties = penalty_breakdown(instance, solution, weights=weights)
    row = penalties.flat()

    if args.output_csv:
        with Path(args.output_csv).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

    for key, value in row.items():
        print(f"{key},{value}")
    return 1 if args.fail_on_hard and penalties.hard_violations else 0


def cmd_contest_score(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    solution = load_solution(args.solution_xml)
    score = score_prefix_with_feasibility_tail(
        instance,
        solution,
        score_days=args.score_days,
        feasibility_days=args.feasibility_days,
        ignore_tail_call_ins=args.ignore_tail_call_ins,
    )
    row = score.flat()

    if args.output_csv:
        with Path(args.output_csv).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

    for key, value in row.items():
        print(f"{key},{value}")
    return 1 if args.fail_on_infeasible and not score.feasible else 0


def cmd_contest_prune(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    solution = load_solution(args.solution_xml)
    improved, report = prune_redundant_shifts(
        instance,
        solution,
        score_days=args.score_days,
        feasibility_days=args.feasibility_days,
        ignore_tail_call_ins=args.ignore_tail_call_ins,
        max_passes=args.max_passes,
    )
    if args.trim_deliveries:
        improved = trim_redundant_deliveries(
            instance,
            improved,
            score_days=args.score_days,
            feasibility_days=args.feasibility_days,
            ignore_tail_call_ins=args.ignore_tail_call_ins,
            max_rounds=args.trim_rounds,
        )
    if args.remove_sources:
        improved = remove_redundant_source_visits(
            instance,
            improved,
            score_days=args.score_days,
            feasibility_days=args.feasibility_days,
            ignore_tail_call_ins=args.ignore_tail_call_ins,
            max_rounds=args.source_rounds,
        )
    save_solution(improved, args.output_xml)
    row = report.flat()
    print(f"wrote,{args.output_xml}")
    for key, value in row.items():
        print(f"{key},{value}")
    return 0


def cmd_route_swap_search(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    incumbent = load_solution(args.incumbent_xml)
    reference = load_solution(args.reference_xml)
    result = transactional_route_swap_search(
        instance,
        incumbent,
        reference,
        horizon_days=args.horizon_days,
        max_remove=args.max_remove,
        max_add=args.max_add,
        max_passes=args.max_passes,
        max_evaluations=args.max_evaluations,
        min_delivered_fraction=args.min_delivered_fraction,
        customer_bundles=args.customer_bundles,
        ignore_tail_call_ins=args.ignore_tail_call_ins,
    )
    save_solution(result.solution, args.output_xml)
    score = score_prefix_with_feasibility_tail(
        instance,
        result.solution,
        score_days=args.horizon_days,
        feasibility_days=args.horizon_days,
        ignore_tail_call_ins=args.ignore_tail_call_ins,
    )
    print(f"wrote,{args.output_xml}")
    print(f"evaluated_moves,{result.evaluated_moves}")
    print(f"accepted_moves,{len(result.moves)}")
    print(f"initial_ratio,{result.initial_ratio:.12f}")
    print(f"final_ratio,{result.final_ratio:.12f}")
    print(
        "final_score,feasible,{},{},hard,{},cost,{:.6f},delivered,{:.6f},shifts,{}".format(
            score.feasible,
            score.feasibility_errors,
            score.hard_violations,
            score.scored_estimated_cost,
            score.scored_delivered_quantity,
            score.scored_shifts,
        )
    )
    print("move,removed_indices,added_reference_indices,cost,delivered,ratio")
    for move_index, move in enumerate(result.moves):
        print(
            "{},{},{},{:.6f},{:.6f},{:.12f}".format(
                move_index,
                " ".join(str(index) for index in move.removed_indices),
                " ".join(str(index) for index in move.added_reference_indices),
                move.cost,
                move.delivered,
                move.ratio,
            )
        )
    return 0


def _default_checker_exe(instance_path: Path) -> Path:
    data_dir = Path(__file__).resolve().parent.parent / "roadef_2016_data"
    if "_ConvertedTo_V2" not in instance_path.name and "Instance_V_1." in instance_path.name:
        return (
            data_dir
            / "checker_v1_1"
            / "Checker V1 v1.1.0.0"
            / "Challenge_Roadef_EURO_Checker_V1"
            / "bin"
            / "Release"
            / "IRP_Roadef_Challenge_Checker.exe"
        )
    return (
        data_dir
        / "checker_v2"
        / "Challenge_Roadef_EURO_Checker_V2"
        / "bin"
        / "Release"
        / "IRP_Roadef_Challenge_Checker.exe"
    )


def cmd_contest_highs_repair(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    solution = load_solution(args.solution_xml)
    repaired, report = repair_quantities_with_highs(
        instance,
        solution,
        score_days=args.score_days,
        feasibility_days=args.feasibility_days,
        ignore_tail_call_ins=args.ignore_tail_call_ins,
        quantity_objective=args.quantity_objective,
    )
    save_solution(repaired, args.output_xml)
    print(f"wrote,{args.output_xml}")
    for key, value in report.flat().items():
        print(f"{key},{value}")
    return 1 if args.fail_on_infeasible and not report.after_feasible else 0


def cmd_mds_map(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance_xml)
    points = mds_coordinates(
        instance,
        clusters=args.clusters,
        random_state=args.seed,
        dissimilarity=args.dissimilarity,
    )
    write_geo_csv(points, args.output_csv)
    if args.output_png:
        plot_geo(points, args.output_png)

    counts: dict[int, int] = {}
    for point in points:
        counts[point.cluster] = counts.get(point.cluster, 0) + 1
    print(f"wrote_csv,{args.output_csv}")
    if args.output_png:
        print(f"wrote_png,{args.output_png}")
    print(f"points,{len(points)}")
    print(f"clusters,{args.clusters}")
    for cluster, count in sorted(counts.items()):
        print(f"cluster_{cluster},{count}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    mono = shutil.which("mono")
    if mono is None:
        raise SystemExit("`mono` is not installed or not on PATH.")

    process = subprocess.run(
        [mono, str(CHECKER_EXE), args.instance_xml, args.solution_xml],
        input="\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(process.stdout)
    if process.returncode != 0:
        return process.returncode

    match = RATIO_RE.search(process.stdout)
    if match:
        print(f"parsed_logistic_ratio,{float(match.group(1).replace(',', '.')):.12f}")
    return 0


def cmd_doi_report(args: argparse.Namespace) -> int:
    from .inventory import days_of_inventory, project_customer_inventory, delivery_by_customer_step
    from .model import Solution
    instance = load_instance(args.instance_xml)
    solution = load_solution(args.solution_xml) if args.solution_xml else Solution(shifts=())
    deliveries_by_arrival = delivery_by_customer_step(solution)
    
    # Pre-calculate nearest source travel time for each customer
    source_travel = {}
    for customer in instance.customers:
        min_travel = min(instance.time_matrix[source.index][customer.index] for source in instance.sources)
        source_travel[customer.index] = min_travel

    step = args.minute // instance.unit
    print("point,kind,inventory,safety_level,lead_time_min,logistical_doi,status")
    for customer in instance.customers:
        events = project_customer_inventory(instance, customer, deliveries_by_arrival.get(customer.index, {}))
        current_event = events[min(step, len(events)-1)]
        lead_time = source_travel[customer.index]
        doi = days_of_inventory(instance, customer, current_event.ending_inventory, step + 1, lead_time_minutes=lead_time)
        
        status = "OK"
        if doi < 0: status = "INSOLVENT"
        elif doi < 0.5: status = "CRITICAL"
        elif doi < 1.0: status = "URGENT"
        
        print(f"{customer.index},{'call-in' if customer.call_in else 'vmi'},{current_event.ending_inventory:.1f},{customer.safety_level:.1f},{lead_time},{doi:.2f},{status}")
    return 0


def cmd_resilience_report(args: argparse.Namespace) -> int:
    """Stress test: If all deliveries stop at `blackout-start`, who fails and when?"""
    from .inventory import project_customer_inventory, delivery_by_customer_step
    from .model import Solution
    
    instance = load_instance(args.instance_xml)
    solution = load_solution(args.solution_xml) if args.solution_xml else Solution(shifts=())
    
    # Only keep deliveries BEFORE the blackout
    deliveries_by_arrival = delivery_by_customer_step(solution)
    pre_blackout_deliveries = {}
    for point, events in deliveries_by_arrival.items():
        pre_blackout_deliveries[point] = {arr: qty for arr, qty in events.items() if arr < args.blackout_start}
    
    start_step = args.blackout_start // instance.unit
    end_step = start_step + (args.duration_days * 1440 // instance.unit)
    
    print(f"Stress Test: Blackout starting at minute {args.blackout_start}")
    print("point,kind,hours_until_breach,status")
    
    breach_counts = 0
    for customer in instance.customers:
        if customer.call_in: continue
        
        events = project_customer_inventory(instance, customer, pre_blackout_deliveries.get(customer.index, {}))
        
        first_breach_step = None
        for step in range(start_step, min(end_step, len(events))):
            if events[step].safety_breach:
                first_breach_step = step
                break
        
        if first_breach_step is not None:
            hours = (first_breach_step * instance.unit - args.blackout_start) / 60.0
            status = "FAIL" if hours < 24 else "WARNING"
            print(f"{customer.index},vmi,{hours:.1f},{status}")
            breach_counts += 1
            
    if breach_counts == 0:
        print(f"SYSTEM SECURE: All customers survived a {args.duration_days}-day blackout.")
    else:
        print(f"SYSTEM VULNERABLE: {breach_counts} customers breached safety levels within {args.duration_days} days.")
    return 0


def cmd_clone_solution(args: argparse.Namespace) -> int:
    solution = load_solution(args.solution_xml)
    save_solution(solution, args.output_xml)
    print(f"wrote,{args.output_xml}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ROADEF/EURO 2016 IRP helper tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    instance_summary = subparsers.add_parser("instance-summary")
    instance_summary.add_argument("instance_xml")
    instance_summary.set_defaults(func=cmd_instance_summary)

    solution_summary = subparsers.add_parser("solution-summary")
    solution_summary.add_argument("instance_xml")
    solution_summary.add_argument("solution_xml")
    solution_summary.add_argument("--shifts-csv")
    solution_summary.add_argument("--inventory-csv")
    solution_summary.set_defaults(func=cmd_solution_summary)

    customer_targets = subparsers.add_parser("customer-targets")
    customer_targets.add_argument("instance_xml")
    customer_targets.add_argument("solution_xml")
    customer_targets.add_argument("--output-csv")
    customer_targets.add_argument("--limit", type=int, default=20)
    customer_targets.set_defaults(func=cmd_customer_targets)

    rule_check = subparsers.add_parser("rule-check")
    rule_check.add_argument("instance_xml")
    rule_check.add_argument("solution_xml")
    rule_check.add_argument("--output-csv")
    rule_check.add_argument("--limit", type=int, default=20)
    rule_check.add_argument("--fail-on-violation", action="store_true")
    rule_check.set_defaults(func=cmd_rule_check)

    rules_index = subparsers.add_parser("rules-index")
    rules_index.set_defaults(func=cmd_rules_index)

    construct = subparsers.add_parser("construct-solution")
    construct.add_argument("instance_xml")
    construct.add_argument("output_xml")
    construct.add_argument("--safety-buffer", type=float, default=0.20)
    construct.add_argument("--max-shifts", type=int)
    construct.add_argument("--limit", type=int, default=25)
    construct.set_defaults(func=cmd_construct_solution)

    cluster_construct = subparsers.add_parser("cluster-construct-solution")
    cluster_construct.add_argument("instance_xml")
    cluster_construct.add_argument("output_xml")
    cluster_construct.add_argument("--safety-buffer", type=float, default=0.20)
    cluster_construct.add_argument("--neighborhood-size", type=int, default=5)
    cluster_construct.add_argument("--max-shifts", type=int)
    cluster_construct.add_argument("--score-days", type=int)
    cluster_construct.add_argument("--terminal-buffer-days", type=float, default=0.0)
    cluster_construct.add_argument("--limit", type=int, default=25)
    cluster_construct.set_defaults(func=cmd_cluster_construct_solution)

    matrix_summary = subparsers.add_parser("matrix-summary")
    matrix_summary.add_argument("instance_xml")
    matrix_summary.set_defaults(func=cmd_matrix_summary)

    export_edges = subparsers.add_parser("export-edges")
    export_edges.add_argument("instance_xml")
    export_edges.add_argument("output_csv")
    export_defaults = {"func": cmd_export_edges}
    export_edges.set_defaults(**export_defaults)

    nearest = subparsers.add_parser("nearest")
    nearest.add_argument("instance_xml")
    nearest.add_argument("output_csv")
    nearest.add_argument("--metric", choices=["distance", "time"], default="distance")
    nearest.add_argument("-k", type=int, default=10)
    nearest.set_defaults(func=cmd_nearest)

    collocations = subparsers.add_parser("collocations")
    collocations.add_argument("instance_xml")
    collocations.add_argument("--output-csv")
    collocations.add_argument("--limit", type=int, default=20)
    collocations.set_defaults(func=cmd_collocations)

    speed_outliers = subparsers.add_parser("speed-outliers")
    speed_outliers.add_argument("instance_xml")
    speed_outliers.add_argument("--output-csv")
    speed_outliers.add_argument("--limit", type=int, default=25)
    speed_outliers.set_defaults(func=cmd_speed_outliers)

    asymmetry = subparsers.add_parser("asymmetry-outliers")
    asymmetry.add_argument("instance_xml")
    asymmetry.add_argument("--metric", choices=["distance", "time"], default="time")
    asymmetry.add_argument("--output-csv")
    asymmetry.add_argument("--limit", type=int, default=25)
    asymmetry.set_defaults(func=cmd_asymmetry_outliers)

    tank_check = subparsers.add_parser("tank-check")
    tank_check.add_argument("instance_xml")
    tank_check.add_argument("solution_xml")
    tank_check.add_argument("--violations-csv")
    tank_check.add_argument("--events-csv")
    tank_check.add_argument("--limit", type=int, default=25)
    tank_check.add_argument("--fail-on-violation", action="store_true")
    tank_check.set_defaults(func=cmd_tank_check)

    segments = subparsers.add_parser("segments")
    segments.add_argument("instance_xml")
    segments.add_argument("solution_xml")
    segments.add_argument("output_csv")
    segments.set_defaults(func=cmd_segments)

    replay_snapshot = subparsers.add_parser("replay-snapshot")
    replay_snapshot.add_argument("instance_xml")
    replay_snapshot.add_argument("solution_xml")
    replay_snapshot.add_argument("--time", type=int, required=True)
    replay_snapshot.add_argument("--resources-csv")
    replay_snapshot.add_argument("--customers-csv")
    replay_snapshot.add_argument("--limit", type=int, default=12)
    replay_snapshot.set_defaults(func=cmd_replay_snapshot)

    replay = subparsers.add_parser("replay-grid")
    replay.add_argument("instance_xml")
    replay.add_argument("solution_xml")
    replay.add_argument("--resources-csv", required=True)
    replay.add_argument("--customers-csv", required=True)
    replay.add_argument("--start", type=int, default=0)
    replay.add_argument("--end", type=int)
    replay.add_argument("--step", type=int, default=60)
    replay.set_defaults(func=cmd_replay_grid)

    status = subparsers.add_parser("status")
    status.add_argument("instance_xml")
    status.add_argument("solution_xml")
    status.add_argument("--time", type=int, required=True)
    status.add_argument("--limit", type=int, default=10)
    status.set_defaults(func=cmd_status)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("instance_xml")
    evaluate.add_argument("solution_xml")
    evaluate.add_argument("--official", action="store_true")
    evaluate.add_argument("--output-csv")
    evaluate.add_argument("--fail-on-error", action="store_true")
    evaluate.set_defaults(func=cmd_evaluate)

    smoothness = subparsers.add_parser("smoothness")
    smoothness.add_argument("instance_xml")
    smoothness.add_argument("solution_xml")
    smoothness.add_argument("--period-minutes", type=int, default=1440)
    smoothness.add_argument("--buckets-csv")
    smoothness.add_argument("--summary-csv")
    smoothness.add_argument("--limit", type=int, default=10)
    smoothness.set_defaults(func=cmd_smoothness)

    rolling_monitor = subparsers.add_parser("rolling-monitor")
    rolling_monitor.add_argument("instance_xml")
    rolling_monitor.add_argument("solution_xml", nargs="?")
    rolling_monitor.add_argument("--days", type=int)
    rolling_monitor.add_argument("--days-csv")
    rolling_monitor.add_argument("--summary-csv")
    rolling_monitor.add_argument("--limit", type=int, default=10)
    rolling_monitor.set_defaults(func=cmd_rolling_monitor)

    penalties = subparsers.add_parser("penalties")
    penalties.add_argument("instance_xml")
    penalties.add_argument("solution_xml")
    penalties.add_argument("--output-csv")
    penalties.add_argument("--fail-on-hard", action="store_true")
    penalties.add_argument("--safety-kg-min", type=float, default=1.0)
    penalties.add_argument("--driver-window-min", type=float, default=100.0)
    penalties.add_argument("--driver-rest-min", type=float, default=100.0)
    penalties.add_argument("--max-driving-min", type=float, default=100.0)
    penalties.add_argument("--timing-min", type=float, default=100.0)
    penalties.add_argument("--customer-window-min", type=float, default=100.0)
    penalties.add_argument("--smoothness-cv", type=float, default=10_000.0)
    penalties.add_argument("--frontload-share", type=float, default=10_000.0)
    penalties.add_argument("--hard-violation", type=float, default=1_000_000_000.0)
    penalties.set_defaults(func=cmd_penalties)

    contest_score = subparsers.add_parser("contest-score")
    contest_score.add_argument("instance_xml")
    contest_score.add_argument("solution_xml")
    contest_score.add_argument("--score-days", type=int, required=True)
    contest_score.add_argument("--feasibility-days", type=int)
    contest_score.add_argument("--ignore-tail-call-ins", action="store_true")
    contest_score.add_argument("--output-csv")
    contest_score.add_argument("--fail-on-infeasible", action="store_true")
    contest_score.set_defaults(func=cmd_contest_score)

    contest_prune = subparsers.add_parser("contest-prune")
    contest_prune.add_argument("instance_xml")
    contest_prune.add_argument("solution_xml")
    contest_prune.add_argument("output_xml")
    contest_prune.add_argument("--score-days", type=int, required=True)
    contest_prune.add_argument("--feasibility-days", type=int)
    contest_prune.add_argument("--ignore-tail-call-ins", action="store_true")
    contest_prune.add_argument("--max-passes", type=int, default=3)
    contest_prune.add_argument("--trim-deliveries", action="store_true")
    contest_prune.add_argument("--trim-rounds", type=int, default=5)
    contest_prune.add_argument("--remove-sources", action="store_true")
    contest_prune.add_argument("--source-rounds", type=int, default=10)
    contest_prune.set_defaults(func=cmd_contest_prune)

    route_swap = subparsers.add_parser("route-swap-search")
    route_swap.add_argument("instance_xml", type=Path)
    route_swap.add_argument("incumbent_xml", type=Path)
    route_swap.add_argument("reference_xml", type=Path)
    route_swap.add_argument("output_xml", type=Path)
    route_swap.add_argument("--horizon-days", type=int, required=True)
    route_swap.add_argument("--max-remove", type=int, default=2)
    route_swap.add_argument("--max-add", type=int, default=2)
    route_swap.add_argument("--max-passes", type=int, default=3)
    route_swap.add_argument("--max-evaluations", type=int, default=5000)
    route_swap.add_argument("--min-delivered-fraction", type=float, default=1.0)
    route_swap.add_argument("--customer-bundles", action="store_true")
    route_swap.add_argument("--ignore-tail-call-ins", action="store_true")
    route_swap.set_defaults(func=cmd_route_swap_search)

    highs_repair = subparsers.add_parser("contest-highs-repair")
    highs_repair.add_argument("instance_xml")
    highs_repair.add_argument("solution_xml")
    highs_repair.add_argument("output_xml")
    highs_repair.add_argument("--score-days", type=int, required=True)
    highs_repair.add_argument("--feasibility-days", type=int)
    highs_repair.add_argument("--ignore-tail-call-ins", action="store_true")
    highs_repair.add_argument(
        "--quantity-objective",
        choices=("min-delivered", "max-delivered"),
        default="max-delivered",
    )
    highs_repair.add_argument("--fail-on-infeasible", action="store_true")
    highs_repair.set_defaults(func=cmd_contest_highs_repair)

    highs_select = subparsers.add_parser("highs-select")
    highs_select.add_argument("instance_xml", type=Path)
    highs_select.add_argument("prefix_xml", type=Path)
    highs_select.add_argument("output_xml", type=Path)
    highs_select.add_argument("--start-day", type=int, default=7)
    highs_select.add_argument("--end-day", type=int, default=14)
    highs_select.add_argument("--candidates-per-window", type=int, default=10)
    highs_select.add_argument("--neighborhood-size", type=int, default=5)
    highs_select.set_defaults(func=cmd_highs_select)

    rolling_highs = subparsers.add_parser("rolling-highs-select")
    rolling_highs.add_argument("instance_xml", type=Path)
    rolling_highs.add_argument("output_xml", type=Path)
    rolling_highs.add_argument("--prefix-xml", type=Path)
    rolling_highs.add_argument("--candidate-solution-xml", type=Path)
    rolling_highs.add_argument(
        "--candidate-source",
        choices=["generated", "seed", "both"],
        default="generated",
    )
    rolling_highs.add_argument("--start-day", type=int, default=0)
    rolling_highs.add_argument("--end-day", type=int, default=14)
    rolling_highs.add_argument("--lookahead-days", type=int, default=2)
    rolling_highs.add_argument("--commit-days", type=int, default=1)
    rolling_highs.add_argument("--feasibility-tail-days", type=int, default=1)
    rolling_highs.add_argument("--candidates-per-window", type=int, default=12)
    rolling_highs.add_argument("--neighborhood-size", type=int, default=15)
    rolling_highs.set_defaults(func=cmd_rolling_highs_select)

    targeted_rescue_cmd = subparsers.add_parser("targeted-rescue")
    targeted_rescue_cmd.add_argument("instance_xml", type=Path)
    targeted_rescue_cmd.add_argument("solution_xml", type=Path)
    targeted_rescue_cmd.add_argument("output_xml", type=Path)
    targeted_rescue_cmd.add_argument("--start-day", type=int, default=0)
    targeted_rescue_cmd.add_argument("--end-day", type=int, default=14)
    targeted_rescue_cmd.add_argument("--replace-from-day", type=int, default=7)
    targeted_rescue_cmd.add_argument("--max-customers", type=int, default=12)
    targeted_rescue_cmd.add_argument("--samples-per-customer", type=int, default=6)
    targeted_rescue_cmd.add_argument("--target-fill-ratio", type=float, default=0.95)
    targeted_rescue_cmd.add_argument("--max-pre-service-fill-ratio", type=float, default=0.95)
    targeted_rescue_cmd.add_argument("--sample-lookback-days", type=int, default=5)
    targeted_rescue_cmd.add_argument("--max-chain-length", type=int, default=3)
    targeted_rescue_cmd.add_argument("--nearest-chain-neighbors", type=int, default=4)
    targeted_rescue_cmd.add_argument("--variable-quantity-columns", action="store_true")
    targeted_rescue_cmd.add_argument("--no-pressure-pricing", action="store_true")
    targeted_rescue_cmd.add_argument("--no-normalize-source-loads", action="store_true")
    targeted_rescue_cmd.add_argument(
        "--quantity-objective",
        choices=("min-delivered", "max-delivered"),
        default="max-delivered",
    )
    targeted_rescue_cmd.set_defaults(func=cmd_targeted_rescue)

    column_loop = subparsers.add_parser("column-generation-rescue")
    column_loop.add_argument("instance_xml", type=Path)
    column_loop.add_argument("solution_xml", type=Path)
    column_loop.add_argument("output_xml", type=Path)
    column_loop.add_argument("--start-day", type=int, default=0)
    column_loop.add_argument("--end-day", type=int, default=14)
    column_loop.add_argument("--replace-from-day", type=int, default=3)
    column_loop.add_argument("--iterations", type=int, default=3)
    column_loop.add_argument("--max-pressure-customers", type=int, default=12)
    column_loop.add_argument("--neighbors-per-anchor", type=int, default=8)
    column_loop.add_argument("--batch-workers", type=int, default=4)
    column_loop.add_argument("--samples-per-customer", type=int, default=8)
    column_loop.add_argument("--sample-lookback-days", type=int, default=14)
    column_loop.add_argument("--max-chain-length", type=int, default=4)
    column_loop.add_argument("--nearest-chain-neighbors", type=int, default=10)
    column_loop.add_argument("--max-candidates-per-iteration", type=int, default=1200)
    column_loop.add_argument("--target-fill-ratio", type=float, default=0.95)
    column_loop.add_argument("--max-pre-service-fill-ratio", type=float, default=0.95)
    column_loop.add_argument("--multi-reload-columns", action="store_true")
    column_loop.add_argument("--max-multi-reload-per-batch", type=int, default=20)
    column_loop.add_argument("--no-normalize-source-loads", action="store_true")
    column_loop.add_argument(
        "--quantity-objective",
        choices=("min-delivered", "max-delivered"),
        default="max-delivered",
    )
    column_loop.add_argument("--ml-priors-path", type=Path, default=None)
    column_loop.set_defaults(func=cmd_column_generation_rescue)

    alns_cmd = subparsers.add_parser("alns-rescue")
    alns_cmd.add_argument("instance_xml", type=Path)
    alns_cmd.add_argument("solution_xml", type=Path)
    alns_cmd.add_argument("output_xml", type=Path)
    alns_cmd.add_argument("--start-day", type=int, default=0)
    alns_cmd.add_argument("--end-day", type=int, default=21)
    alns_cmd.add_argument("--replace-from-day", type=int, default=3)
    alns_cmd.add_argument("--iterations", type=int, default=20)
    alns_cmd.add_argument("--repair-iterations", type=int, default=2)
    alns_cmd.add_argument("--seed", type=int, default=0)
    alns_cmd.add_argument("--initial-temperature", type=float, default=5000.0)
    alns_cmd.add_argument("--cooling-rate", type=float, default=0.92)
    alns_cmd.add_argument("--max-removed-shifts", type=int, default=8)
    alns_cmd.add_argument("--related-customer-count", type=int, default=8)
    alns_cmd.add_argument("--time-band-days", type=int, default=3)
    alns_cmd.add_argument("--max-pressure-customers", type=int, default=12)
    alns_cmd.add_argument("--samples-per-customer", type=int, default=6)
    alns_cmd.add_argument("--sample-lookback-days", type=int, default=14)
    alns_cmd.add_argument("--max-candidates-per-iteration", type=int, default=700)
    alns_cmd.add_argument("--target-fill-ratio", type=float, default=0.95)
    alns_cmd.add_argument("--nearest-chain-neighbors", type=int, default=4)
    alns_cmd.add_argument("--multi-reload-columns", action="store_true")
    alns_cmd.add_argument("--max-multi-reload-per-batch", type=int, default=8)
    alns_cmd.add_argument("--no-normalize-source-loads", action="store_true")
    alns_cmd.add_argument(
        "--quantity-objective",
        choices=("min-delivered", "max-delivered"),
        default="max-delivered",
    )
    alns_cmd.set_defaults(func=cmd_alns_rescue)

    rolling_cg = subparsers.add_parser("robust-rolling-rescue")
    rolling_cg.add_argument("instance_xml", type=Path)
    rolling_cg.add_argument("solution_xml", type=Path)
    rolling_cg.add_argument("output_xml", type=Path)
    rolling_cg.add_argument(
        "--mode",
        choices=("deterministic", "hedged", "robust"),
        default="hedged",
    )
    rolling_cg.add_argument("--horizon-days", type=int, default=30)
    rolling_cg.add_argument("--commit-days", type=int, default=7)
    rolling_cg.add_argument("--lookahead-days", type=int, default=7)
    rolling_cg.add_argument("--n-scenarios", type=int, default=20)
    rolling_cg.add_argument("--scenario-seed", type=int, default=42)
    rolling_cg.add_argument("--plan-sigma", type=float, default=0.15)
    rolling_cg.add_argument("--buffer-sigma", type=float, default=0.30)
    rolling_cg.add_argument("--commit-percentile", type=float, default=50.0)
    rolling_cg.add_argument("--plan-percentile", type=float, default=75.0)
    rolling_cg.add_argument("--buffer-percentile", type=float, default=90.0)
    rolling_cg.add_argument("--cg-iterations", type=int, default=5)
    rolling_cg.add_argument("--max-pressure-customers", type=int, default=12)
    rolling_cg.add_argument("--samples-per-customer", type=int, default=8)
    rolling_cg.add_argument("--max-chain-length", type=int, default=4)
    rolling_cg.add_argument("--nearest-chain-neighbors", type=int, default=10)
    rolling_cg.add_argument("--max-candidates-per-iteration", type=int, default=1200)
    rolling_cg.add_argument("--target-fill-ratio", type=float, default=0.95)
    rolling_cg.add_argument("--multi-reload-columns", action="store_true")
    rolling_cg.add_argument("--max-pre-service-fill-ratio", type=float, default=0.95)
    rolling_cg.add_argument("--no-normalize-source-loads", action="store_true")
    rolling_cg.add_argument("--quantity-objective", choices=("min-delivered", "max-delivered"), default="max-delivered")
    rolling_cg.add_argument("--capacity-buffer", type=float, default=0.05)
    rolling_cg.add_argument("--max-hedge-retries", type=int, default=2)
    rolling_cg.add_argument("--max-rounds", type=int)
    rolling_cg.add_argument("--committed-output-only", action="store_true")
    rolling_cg.add_argument("--no-final-clip-capacity", action="store_true")
    rolling_cg.add_argument("--progress-log", type=Path)
    rolling_cg.add_argument("--calibration-input", type=Path)
    rolling_cg.add_argument("--route-priors", type=Path)
    rolling_cg.add_argument("--selector-time-limit", type=float, default=300.0)
    rolling_cg.add_argument("--selector-mip-gap", type=float)
    rolling_cg.add_argument("--selector-threads", type=int)
    rolling_cg.add_argument("--selector-mip-focus", type=int)
    rolling_cg.add_argument("--selector-node-limit", type=int)
    rolling_cg.add_argument("--selector-phase", choices=("auto", "feasibility", "cost"), default="auto")
    rolling_cg.add_argument("--candidate-cache-dir", type=Path)
    rolling_cg.add_argument("--bucket-anchor-cap", type=int, default=80)
    rolling_cg.add_argument("--bucket-resource-time-cap", type=int, default=12)
    rolling_cg.add_argument("--bucket-route-signature-cap", type=int, default=1)
    rolling_cg.add_argument("--bucket-source-region-cap", type=int, default=240)
    rolling_cg.add_argument("--rebalance", action="store_true")
    rolling_cg.add_argument("--protect-exact-prior-quantities", action="store_true")
    rolling_cg.add_argument("--no-prior-incumbent", action="store_true")
    rolling_cg.add_argument(
        "--forecast-input",
        type=Path,
        help="External forecast CSV/JSON/JSONL/parquet with item_id/customer_id, step or timestamp, and quantile columns.",
    )
    rolling_cg.set_defaults(func=cmd_robust_rolling_rescue)

    robust_batch = subparsers.add_parser("robust-batch-rescue")
    robust_batch.add_argument("output_dir", type=Path)
    robust_batch.add_argument("--instances", default="V2.12,V2.18")
    robust_batch.add_argument("--horizons", choices=("auto", "14", "21", "30"), default="auto")
    robust_batch.add_argument("--seed", type=int)
    robust_batch.add_argument("--quick", action="store_true")
    robust_batch.add_argument("--first-week-rescue-preset", action="store_true")
    robust_batch.add_argument("--candidate-cache-dir", type=Path)
    robust_batch.add_argument("--no-rebalance", action="store_true")
    robust_batch.set_defaults(func=cmd_robust_batch_rescue)

    dummy_route_ci = subparsers.add_parser("dummy-route-ci")
    dummy_route_ci.add_argument("instance_xml", type=Path)
    dummy_route_ci.add_argument("solution_xml", type=Path)
    dummy_route_ci.add_argument("output_csv", type=Path)
    dummy_route_ci.add_argument("--present-day", type=int, default=0)
    dummy_route_ci.add_argument("--quantiles", default="50,75,90,95")
    dummy_route_ci.add_argument("--base-relative-width", type=float, default=0.05)
    dummy_route_ci.add_argument("--daily-relative-growth", type=float, default=0.04)
    dummy_route_ci.add_argument("--max-relative-width", type=float, default=0.60)
    dummy_route_ci.add_argument("--route-anchor-width", type=float, default=0.08)
    dummy_route_ci.set_defaults(func=cmd_dummy_route_ci)

    week_ahead_ci = subparsers.add_parser("week-ahead-ci-rescue")
    week_ahead_ci.add_argument("instance_xml", type=Path)
    week_ahead_ci.add_argument("solution_xml", type=Path)
    week_ahead_ci.add_argument("forecast_input", type=Path)
    week_ahead_ci.add_argument("output_xml", type=Path)
    week_ahead_ci.add_argument("--mode", choices=("hedged", "robust"), default="hedged")
    week_ahead_ci.add_argument("--planning-horizon-days", type=int, default=30)
    week_ahead_ci.add_argument("--lookahead-days", type=int, default=21)
    week_ahead_ci.add_argument("--plan-percentile", type=float, default=75.0)
    week_ahead_ci.add_argument("--buffer-percentile", type=float, default=90.0)
    week_ahead_ci.add_argument("--cg-iterations", type=int, default=5)
    week_ahead_ci.add_argument("--max-pressure-customers", type=int, default=12)
    week_ahead_ci.add_argument("--samples-per-customer", type=int, default=8)
    week_ahead_ci.add_argument("--max-chain-length", type=int, default=4)
    week_ahead_ci.add_argument("--nearest-chain-neighbors", type=int, default=10)
    week_ahead_ci.add_argument("--max-candidates-per-iteration", type=int, default=1200)
    week_ahead_ci.add_argument("--target-fill-ratio", type=float, default=0.95)
    week_ahead_ci.add_argument("--multi-reload-columns", action="store_true")
    week_ahead_ci.add_argument("--max-pre-service-fill-ratio", type=float, default=0.95)
    week_ahead_ci.add_argument("--no-normalize-source-loads", action="store_true")
    week_ahead_ci.add_argument("--quantity-objective", choices=("min-delivered", "max-delivered"), default="max-delivered")
    week_ahead_ci.add_argument("--capacity-buffer", type=float, default=0.05)
    week_ahead_ci.add_argument("--max-hedge-retries", type=int, default=2)
    week_ahead_ci.add_argument("--progress-log", type=Path)
    week_ahead_ci.add_argument("--calibration-input", type=Path)
    week_ahead_ci.add_argument("--route-priors", type=Path)
    week_ahead_ci.add_argument("--selector-time-limit", type=float, default=300.0)
    week_ahead_ci.add_argument("--selector-mip-gap", type=float)
    week_ahead_ci.add_argument("--selector-threads", type=int)
    week_ahead_ci.add_argument("--selector-mip-focus", type=int)
    week_ahead_ci.add_argument("--selector-node-limit", type=int)
    week_ahead_ci.add_argument("--selector-phase", choices=("auto", "feasibility", "cost"), default="auto")
    week_ahead_ci.add_argument("--candidate-cache-dir", type=Path)
    week_ahead_ci.add_argument("--bucket-anchor-cap", type=int, default=80)
    week_ahead_ci.add_argument("--bucket-resource-time-cap", type=int, default=12)
    week_ahead_ci.add_argument("--bucket-route-signature-cap", type=int, default=1)
    week_ahead_ci.add_argument("--bucket-source-region-cap", type=int, default=240)
    week_ahead_ci.add_argument("--protect-exact-prior-quantities", action="store_true")
    week_ahead_ci.add_argument("--no-prior-incumbent", action="store_true")
    week_ahead_ci.add_argument("--no-rebalance", action="store_true")
    week_ahead_ci.set_defaults(func=cmd_week_ahead_ci_rescue)

    history_check = subparsers.add_parser("consumption-history-check")
    history_check.add_argument("history_csv", type=Path)
    history_check.add_argument("--output-csv", type=Path)
    history_check.set_defaults(func=cmd_consumption_history_check)

    solution_history = subparsers.add_parser("solution-history-extract")
    solution_history.add_argument("instance_xml", type=Path)
    solution_history.add_argument("solution_xml", type=Path)
    solution_history.add_argument("output_csv", type=Path)
    solution_history.add_argument("--history-days", type=int, default=7)
    solution_history.set_defaults(func=cmd_solution_history_extract)

    forecast_calibration = subparsers.add_parser("forecast-calibration")
    forecast_calibration.add_argument("instance_xml", type=Path)
    forecast_calibration.add_argument("forecast_csv", type=Path)
    forecast_calibration.add_argument("realized_csv", type=Path)
    forecast_calibration.add_argument("--output-csv", type=Path)
    forecast_calibration.set_defaults(func=cmd_forecast_calibration)

    policy_sweep = subparsers.add_parser("robust-policy-sweep")
    policy_sweep.add_argument("sweep_csv", type=Path)
    policy_sweep.add_argument("--instance", type=Path, required=True)
    policy_sweep.add_argument("--baseline", type=Path, required=True)
    policy_sweep.add_argument("--forecast-input", type=Path, required=True)
    policy_sweep.add_argument("--output-dir", type=Path, required=True)
    policy_sweep.add_argument("--horizon-days", type=int, default=30)
    policy_sweep.add_argument("--scenario-seed", type=int, default=42)
    policy_sweep.add_argument("--mode", choices=("deterministic", "hedged", "robust"), default="hedged")
    policy_sweep.add_argument("--max-rounds", type=int)
    policy_sweep.add_argument("--calibration-input", type=Path)
    policy_sweep.add_argument("--selector-time-limit", type=float, default=300.0)
    policy_sweep.add_argument("--selector-mip-gap", type=float)
    policy_sweep.add_argument("--selector-threads", type=int)
    policy_sweep.add_argument("--selector-mip-focus", type=int)
    policy_sweep.add_argument("--selector-node-limit", type=int)
    policy_sweep.add_argument("--selector-phase", choices=("auto", "feasibility", "cost"), default="auto")
    policy_sweep.add_argument("--candidate-cache-dir", type=Path)
    policy_sweep.add_argument("--resume", action="store_true")
    policy_sweep.add_argument("--force", action="store_true")
    policy_sweep.set_defaults(func=cmd_robust_policy_sweep)

    scenario_backtest = subparsers.add_parser("scenario-backtest")
    scenario_backtest.add_argument("instance_xml", type=Path)
    scenario_backtest.add_argument("solution_xml", type=Path)
    scenario_backtest.add_argument("forecast_input", type=Path)
    scenario_backtest.add_argument("--horizon-days", type=int, required=True)
    scenario_backtest.add_argument(
        "--percentiles",
        help="Comma-separated quantile stress paths to evaluate, e.g. 50,75,90,95.",
    )
    scenario_backtest.add_argument("--output-csv", type=Path)
    scenario_backtest.add_argument("--output-summary-csv", type=Path)
    scenario_backtest.add_argument("--include-tail-call-ins", action="store_true")
    scenario_backtest.add_argument("--fail-on-infeasible", action="store_true")
    scenario_backtest.set_defaults(func=cmd_scenario_backtest)

    mds_map = subparsers.add_parser("mds-map")

    mds_map.add_argument("instance_xml")
    mds_map.add_argument("output_csv")
    mds_map.add_argument("--output-png")
    mds_map.add_argument("--clusters", type=int, default=12)
    mds_map.add_argument("--seed", type=int, default=42)
    mds_map.add_argument("--dissimilarity", choices=["distance", "time"], default="distance")
    mds_map.set_defaults(func=cmd_mds_map)

    check = subparsers.add_parser("check")
    check.add_argument("instance_xml")
    check.add_argument("solution_xml")
    check.set_defaults(func=cmd_check)

    clone = subparsers.add_parser("clone-solution")
    clone.add_argument("solution_xml")
    clone.add_argument("output_xml")
    clone.set_defaults(func=cmd_clone_solution)

    doi_report = subparsers.add_parser("doi-report")
    doi_report.add_argument("instance_xml")
    doi_report.add_argument("solution_xml", nargs="?")
    doi_report.add_argument("--minute", type=int, default=0)
    doi_report.set_defaults(func=cmd_doi_report)

    resilience = subparsers.add_parser("resilience-report")
    resilience.add_argument("instance_xml")
    resilience.add_argument("solution_xml", nargs="?")
    resilience.add_argument("--blackout-start", type=int, default=0)
    resilience.add_argument("--duration-days", type=int, default=2)
    resilience.set_defaults(func=cmd_resilience_report)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
