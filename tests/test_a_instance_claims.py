from __future__ import annotations

from vrp_solver.contest import score_prefix_with_feasibility_tail
from vrp_solver.xml_io import load_instance, load_solution


RAW_A11 = "roadef_2016_data/set_A_v1_1/Instances V1.1/Instance_V_1.1.xml"
V2_A11 = "roadef_2016_data/set_A/Instance_V_1.1_ConvertedTo_V2.xml"
BEST_A11 = "roadef_2016_data/hust_smart_results/v1_1.1_cached_expand3_pruned_maxfill.xml"
INFEASIBLE_A11 = "roadef_2016_data/hust_smart_results/v1_1.1_rescued_full_horizon.xml"
RAW_A19 = "roadef_2016_data/set_A_v1_1/Instances V1.1/Instance_V_1.9.xml"
BEST_A19 = "roadef_2016_data/hust_smart_results/v1_1.9_rescued_feasible.xml"
INFEASIBLE_A19 = "roadef_2016_data/hust_smart_results/v1_1.9_rescued_w8.xml"


def _ratio(instance_path: str, solution_path: str) -> tuple[bool, int, float]:
    instance = load_instance(instance_path)
    solution = load_solution(solution_path)
    days = (instance.horizon * instance.unit + 1439) // 1440
    score = score_prefix_with_feasibility_tail(
        instance,
        solution,
        score_days=days,
        feasibility_days=days,
        ignore_tail_call_ins=True,
    )
    return (
        score.feasible,
        score.feasibility_errors,
        score.scored_estimated_cost / max(1.0, score.scored_delivered_quantity),
    )


def test_a11_best_claim_uses_feasible_raw_v1_ratio() -> None:
    feasible, errors, ratio = _ratio(RAW_A11, BEST_A11)

    assert feasible is True
    assert errors == 0
    assert round(ratio, 6) == 0.031523


def test_a11_rescued_full_horizon_is_not_a_benchmark_claim() -> None:
    feasible, errors, ratio = _ratio(RAW_A11, INFEASIBLE_A11)

    assert feasible is False
    assert errors == 514
    assert round(ratio, 6) == 0.032369
    assert round(ratio / 2.0, 6) == 0.016184


def test_a11_v2_ratio_is_double_raw_ratio_for_same_solution() -> None:
    raw_feasible, _raw_errors, raw_ratio = _ratio(RAW_A11, BEST_A11)
    v2_feasible, _v2_errors, v2_ratio = _ratio(V2_A11, BEST_A11)

    assert raw_feasible is True
    assert v2_feasible is True
    assert round(v2_ratio / 2.0, 6) == round(raw_ratio, 6)


def test_a19_best_claim_uses_error_free_rescue_not_near_miss() -> None:
    feasible, errors, ratio = _ratio(RAW_A19, BEST_A19)
    near_miss_feasible, near_miss_errors, _near_miss_ratio = _ratio(
        RAW_A19,
        INFEASIBLE_A19,
    )

    assert feasible is True
    assert errors == 0
    assert round(ratio, 6) == 0.026426
    assert near_miss_feasible is False
    assert near_miss_errors == 100
