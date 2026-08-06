from dataclasses import replace

from vrp_solver.model import (
    Customer,
    Driver,
    Instance,
    Operation,
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
    ) == "replace_operation_point"

    development = replace(config, coverage_include_ejection=False)
    assert _coverage_rebuild_operator(
        instance, empty, development, iteration=3, stagnation=3,
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
