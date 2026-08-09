from dataclasses import replace

from vrp_solver.model import (
    Customer,
    Driver,
    Instance,
    Operation,
    Order,
    Shift,
    Solution,
    Source,
    TimeWindow,
    Trailer,
)
from vrp_solver.solver.cluster_greedy import derive_cluster_construction_policy
from vrp_solver.solver.surgical_search import (
    SurgicalSearchConfig,
    _coverage_rebuild_operator,
)
from vrp_solver.solver.targeted_rescue import (
    RescueConfig,
    generate_rescue_candidates,
)
from vrp_solver.rules import derive_solution


def _instance(*, allowed_trailers: tuple[int, ...] = (0,)) -> Instance:
    trailers = (
        Trailer(0, 20_000.0, 0.0, 0.0),
        Trailer(1, 20_000.0, 0.0, 0.0),
    )
    customers = tuple(
        Customer(
            index=2 + index,
            layover_customer=False,
            call_in=False,
            orders=(),
            setup_time=5,
            time_windows=(TimeWindow(0, 1_000),),
            allowed_trailers=allowed_trailers,
            forecast=(10.0,) * 10,
            capacity=1_000.0,
            initial_tank_quantity=100.0,
            min_operation_quantity=10.0,
            safety_level=50.0,
        )
        for index in range(25)
    )
    point_count = 27
    matrix = tuple(tuple(0 if i == j else 10 for j in range(point_count)) for i in range(point_count))
    return Instance(
        name="feature-policy-fixture",
        unit=60,
        horizon=10,
        time_matrix=matrix,
        distance_matrix=tuple(tuple(float(value) for value in row) for row in matrix),
        base_index=0,
        drivers=(Driver(0, 0, 600, (0, 1), (TimeWindow(0, 1_000),), 60, 0.0, 0.0),),
        trailers=trailers,
        sources=(Source(1, (0, 1), 5),),
        customers=customers,
    )


def test_cluster_policy_expands_breadth_for_sparse_compatibility() -> None:
    sparse = derive_cluster_construction_policy(_instance(allowed_trailers=(0,)))
    dense = derive_cluster_construction_policy(_instance(allowed_trailers=(0, 1)))

    assert sparse.neighborhood_size >= dense.neighborhood_size
    assert sparse.global_pressure_fill >= dense.global_pressure_fill
    assert 5 <= sparse.neighborhood_size <= 16
    assert 4 <= sparse.global_pressure_fill <= 16


def test_missing_coverage_rotates_across_topology_surfaces() -> None:
    instance = _instance()
    config = SurgicalSearchConfig(end_day=1)
    empty = Solution(())

    assert _coverage_rebuild_operator(
        instance, empty, config, iteration=0, stagnation=0,
    ) == "create_shift"
    assert _coverage_rebuild_operator(
        instance, empty, config, iteration=1, stagnation=1,
    ) == "insert_operation"
    assert _coverage_rebuild_operator(
        instance, empty, config, iteration=2, stagnation=2,
    ) == "pressure_band_resource_block"
    assert _coverage_rebuild_operator(
        instance, empty, config, iteration=3, stagnation=3,
    ) == "multiroute_pressure_block"
    assert _coverage_rebuild_operator(
        instance, empty, config, iteration=4, stagnation=4,
    ) == "replace_operation_point"

    development = replace(config, coverage_include_ejection=False)
    assert _coverage_rebuild_operator(
        instance, empty, development, iteration=3, stagnation=3,
    ) == "multiroute_pressure_block"
    assert _coverage_rebuild_operator(
        instance, empty, development, iteration=4, stagnation=4,
    ) == "recombine_route_blocks"


def test_timely_visit_disables_coverage_override() -> None:
    instance = replace(_instance(), customers=(_instance().customers[0],))
    solution = Solution((
        Shift(
            index=0,
            driver=0,
            trailer=0,
            start=0,
            operations=(Operation(point=2, arrival=20, quantity=100.0),),
        ),
    ))

    assert _coverage_rebuild_operator(
        instance,
        solution,
        SurgicalSearchConfig(end_day=1),
        iteration=0,
        stagnation=0,
    ) is None


def test_rescue_represents_return_layover_with_terminal_source() -> None:
    matrix = (
        (0, 10, 60),
        (10, 0, 60),
        (60, 60, 0),
    )
    customer = Customer(
        index=2,
        layover_customer=True,
        call_in=False,
        orders=(),
        setup_time=5,
        time_windows=(TimeWindow(70, 180),),
        allowed_trailers=(0,),
        forecast=(10.0,) * 5,
        capacity=100.0,
        initial_tank_quantity=10.0,
        min_operation_quantity=1.0,
        safety_level=5.0,
    )
    instance = Instance(
        name="represented-return-layover",
        unit=60,
        horizon=5,
        time_matrix=matrix,
        distance_matrix=tuple(tuple(float(v) for v in row) for row in matrix),
        base_index=0,
        drivers=(Driver(
            0, 0, 100, (0,), (TimeWindow(0, 300),), 30, 0.0, 0.0,
        ),),
        trailers=(Trailer(0, 100.0, 0.0, 0.0),),
        sources=(Source(1, (0,), 5),),
        customers=(customer,),
    )

    candidates = generate_rescue_candidates(
        instance,
        Solution(()),
        [customer.index],
        config=RescueConfig(
            start_day=0,
            end_day=1,
            replace_from_day=0,
            samples_per_customer=4,
        ),
    )

    assert candidates
    route = candidates[0]
    assert tuple(operation.point for operation in route.operations) == (1, 2, 1)
    derived = derive_solution(instance, Solution((route,)))[0]
    assert derived.layovers == 1
    assert derived.operations[-1].layover_before


def test_rescue_prices_every_unsatisfied_callin_order() -> None:
    base = _instance()
    customer = replace(
        base.customers[0],
        call_in=True,
        orders=(
            Order(100.0, 100, 200, 100),
            Order(100.0, 500, 600, 100),
        ),
    )
    instance = replace(base, customers=(customer, *base.customers[1:]))

    candidates = generate_rescue_candidates(
        instance,
        Solution(()),
        [customer.index],
        config=RescueConfig(
            start_day=0,
            end_day=1,
            replace_from_day=0,
            samples_per_customer=2,
        ),
    )

    arrivals = [
        operation.arrival
        for shift in candidates
        for operation in shift.operations
        if operation.point == customer.index
    ]
    assert any(100 <= arrival <= 200 for arrival in arrivals)
    assert any(500 <= arrival <= 600 for arrival in arrivals)
