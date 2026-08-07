from __future__ import annotations

from dataclasses import replace

import pytest

from vrp_solver.model import Customer, Driver, Instance, Operation, Order, Source, TimeWindow, Trailer
from vrp_solver.solver.cluster_greedy import (
    _ResourceState,
    _align_arrival_to_customer_window,
    _candidate_for_customer,
    _next_unsatisfied_order,
    _paper_customer_candidate,
    _preload_terminal_source,
    construct_paper_solution,
)
from vrp_solver.rules import validate_solution


@pytest.fixture
def base_instance() -> Instance:
    return Instance(
        name="cluster-cold-start",
        unit=60,
        horizon=48,
        time_matrix=((0, 60), (60, 0)),
        distance_matrix=((0.0, 50.0), (50.0, 0.0)),
        base_index=0,
        drivers=(
            Driver(
                index=0,
                min_inter_shift_duration=0,
                max_driving_duration=540,
                trailer_ids=(0,),
                time_windows=(TimeWindow(start=0, end=2_880),),
                layover_duration=540,
                time_cost=0.0,
                layover_cost=0.0,
            ),
        ),
        trailers=(Trailer(index=0, capacity=20_000.0, initial_quantity=10_000.0, distance_cost=0.0),),
        sources=(Source(index=0, allowed_trailers=(0,), setup_time=30),),
        customers=(
            Customer(
                index=1,
                layover_customer=False,
                call_in=False,
                orders=(),
                setup_time=30,
                time_windows=(TimeWindow(start=0, end=2_880),),
                allowed_trailers=(0,),
                forecast=(10.0,) * 48,
                capacity=15_000.0,
                initial_tank_quantity=5_000.0,
                min_operation_quantity=2_000.0,
                safety_level=1_000.0,
            ),
        ),
    )


def test_terminal_source_preload_carries_inventory_to_next_shift(base_instance) -> None:
    resource = _ResourceState(driver=0, trailer=0, trailer_quantity=0.0)
    operations = [Operation(point=1, arrival=100, quantity=10_000.0)]

    end = _preload_terminal_source(
        base_instance,
        resource,
        TimeWindow(start=0, end=500),
        operations,
        current_pt=1,
        current_time=130,
        driving=60,
        score_cutoff_minute=None,
    )

    assert operations[-1] == Operation(point=0, arrival=190, quantity=-20_000.0)
    assert resource.trailer_quantity == 20_000.0
    assert end == 220


def test_layover_customer_allows_economic_wait_and_resets_driving(base_instance) -> None:
    driver = replace(base_instance.drivers[0], layover_duration=60)
    customer = replace(
        base_instance.customers[0],
        layover_customer=True,
        initial_tank_quantity=15_000.0,
        forecast=(100.0,) * base_instance.horizon,
        min_operation_quantity=1.0,
        time_windows=(TimeWindow(start=300, end=2_880),),
    )
    instance = replace(base_instance, drivers=(driver,), customers=(customer,))
    resource = _ResourceState(driver=0, trailer=0, trailer_quantity=10_000.0)

    candidate = _candidate_for_customer(
        instance,
        resource,
        TimeWindow(start=0, end=2_880),
        current_pt=0,
        current_time=0,
        driving=0,
        customer=customer,
        deliveries={},
        buffer=0.2,
    )

    assert candidate is not None
    assert candidate.layover_before
    assert candidate.driving_after == 0


def test_economic_wait_needs_an_eligible_layover_customer(base_instance) -> None:
    driver = replace(base_instance.drivers[0], layover_duration=60)
    customer = replace(
        base_instance.customers[0],
        layover_customer=False,
        initial_tank_quantity=15_000.0,
        forecast=(100.0,) * base_instance.horizon,
        min_operation_quantity=1.0,
    )
    instance = replace(base_instance, drivers=(driver,), customers=(customer,))
    resource = _ResourceState(driver=0, trailer=0, trailer_quantity=10_000.0)

    candidate = _candidate_for_customer(
        instance,
        resource,
        TimeWindow(start=0, end=2_880),
        current_pt=0,
        current_time=0,
        driving=0,
        customer=customer,
        deliveries={},
        buffer=0.2,
    )

    assert candidate is None


def test_customer_window_wait_uses_next_opening(base_instance) -> None:
    customer = replace(
        base_instance.customers[0],
        setup_time=30,
        time_windows=(TimeWindow(start=300, end=400), TimeWindow(start=500, end=700)),
    )

    assert _align_arrival_to_customer_window(customer, 100) == 300
    assert _align_arrival_to_customer_window(customer, 390) == 500
    assert _align_arrival_to_customer_window(customer, 680) is None


def test_layover_customer_return_rest_requires_representable_successor(base_instance) -> None:
    """A final-stop rest is legal only when a successor source stop fits.

    The checker derives a rest from the following operation's leading gap, so
    the candidate must flag ``return_layover`` (forcing the terminal source
    visit) rather than being rejected outright.
    """
    driver = replace(base_instance.drivers[0], max_driving_duration=100, layover_duration=60)
    customer = replace(base_instance.customers[0], layover_customer=True)
    instance = replace(base_instance, drivers=(driver,), customers=(customer,))
    resource = _ResourceState(driver=0, trailer=0, trailer_quantity=10_000.0)

    candidate = _candidate_for_customer(
        instance,
        resource,
        TimeWindow(start=0, end=2_880),
        current_pt=0,
        current_time=0,
        driving=0,
        customer=customer,
        deliveries={},
        buffer=0.2,
    )

    assert candidate is not None
    assert candidate.return_layover

    # Without a compatible source there is no successor operation to carry
    # the rest, so the candidate must be rejected as before.
    no_source = replace(
        instance,
        sources=(replace(instance.sources[0], allowed_trailers=()),),
    )
    rejected = _candidate_for_customer(
        no_source,
        resource,
        TimeWindow(start=0, end=2_880),
        current_pt=0,
        current_time=0,
        driving=0,
        customer=customer,
        deliveries={},
        buffer=0.2,
    )

    assert rejected is None


def test_paper_constructor_does_not_use_a_second_layover(base_instance) -> None:
    driver = replace(base_instance.drivers[0], layover_duration=60)
    customer = replace(
        base_instance.customers[0],
        layover_customer=True,
        initial_tank_quantity=15_000.0,
        forecast=(100.0,) * base_instance.horizon,
        min_operation_quantity=1.0,
        time_windows=(TimeWindow(start=300, end=2_880),),
    )
    instance = replace(base_instance, drivers=(driver,), customers=(customer,))
    resource = _ResourceState(driver=0, trailer=0, trailer_quantity=10_000.0)

    candidate = _paper_customer_candidate(
        instance,
        resource,
        TimeWindow(start=0, end=2_880),
        current_point=0,
        current_time=0,
        driving=0,
        customer=customer,
        deliveries={},
        has_layover_customer=True,
        layover_used=True,
        force_reload=False,
        at_route_start=False,
        inventory_cache={},
    )

    assert candidate is None


def test_call_in_is_feasible_after_flexible_minimum(base_instance) -> None:
    customer = replace(
        base_instance.customers[0],
        call_in=True,
        orders=(Order(quantity=10_000.0, earliest_time=0, latest_time=1_000, quantity_flexibility=80),),
    )

    assert _next_unsatisfied_order(customer, {100: 7_999.0}) is not None
    assert _next_unsatisfied_order(customer, {100: 8_000.0}) is None


def test_paper_constructor_is_seeded_and_returns_a_valid_cold_start() -> None:
    instance = Instance(
        name="paper-construction",
        unit=60,
        horizon=12,
        time_matrix=((0, 10, 20), (10, 0, 10), (20, 10, 0)),
        distance_matrix=((0.0, 1.0, 2.0), (1.0, 0.0, 1.0), (2.0, 1.0, 0.0)),
        base_index=0,
        drivers=(Driver(0, 0, 120, (0,), (TimeWindow(0, 1_000),), 60, 0.0, 0.0),),
        trailers=(Trailer(0, 10_000.0, 0.0, 0.0),),
        sources=(Source(1, (0,), 5),),
        customers=(
            Customer(2, False, False, (), 5, (TimeWindow(0, 1_000),), (0,), (500.0,) * 12, 10_000.0, 3_000.0, 1_000.0, 500.0),
        ),
    )

    first, first_report = construct_paper_solution(instance, seed=7, retries=3)
    second, second_report = construct_paper_solution(instance, seed=7, retries=3)

    assert first == second
    assert first_report == second_report
    assert first_report.attempts == 1
    assert not [item for item in validate_solution(instance, first) if item.severity == "error"]
