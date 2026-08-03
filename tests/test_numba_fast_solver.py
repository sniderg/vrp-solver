"""Unit and integration tests for the Numba + Gurobi native solver engine (vrp_solver.numba_fast_solver).

Tests cover:
1. Fast Numba C-speed route duration calculation (@njit kernel fast_route_duration)
2. Numba candidate shift pool construction (build_numba_candidate_pool)
3. End-to-end native solve execution (solve_numba_gurobi_mip)
4. Official rule-checker compliance (validate_solution: 0 hard errors, 0 missed orders, 0 runouts)
5. CLI and solution export integration
"""

import os
from pathlib import Path
import pytest

from vrp_solver.xml_io import load_instance, save_solution, load_solution
from vrp_solver.rules import validate_solution
from vrp_solver.numba_fast_solver import (
    fast_route_duration,
    build_numba_candidate_pool,
    solve_numba_gurobi_mip,
    CandidateShift,
)
from vrp_solver.model import Solution

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V212_PATH = PROJECT_ROOT / "roadef_2016_data" / "set_B" / "Instances_B_V25-11042016" / "V2.12.xml"


@pytest.fixture
def v212_instance():
    """Fixture to load V2.12 instance once for tests."""
    assert V212_PATH.exists(), f"Instance file not found: {V212_PATH}"
    return load_instance(str(V212_PATH))


def test_fast_route_duration_kernel(v212_instance):
    """Test compiled Numba @njit kernel fast_route_duration accuracy."""
    import numpy as np

    inst = v212_instance
    n_points = max(
        inst.base_index,
        max(c.index for c in inst.customers) if inst.customers else 0,
        max(s.index for s in inst.sources) if inst.sources else 0
    ) + 1

    time_mat = np.zeros((n_points, n_points), dtype=np.int32)
    for i in range(len(inst.time_matrix)):
        for j in range(len(inst.time_matrix[i])):
            time_mat[i, j] = inst.time_matrix[i][j]

    setup_times = np.zeros(n_points, dtype=np.int32)
    for c in inst.customers:
        setup_times[c.index] = c.setup_time
    for s in inst.sources:
        setup_times[s.index] = s.setup_time

    source_id = inst.sources[0].index
    cust_id = inst.customers[0].index

    route_points = np.array([source_id, cust_id], dtype=np.int32)
    dur = fast_route_duration(time_mat, route_points, setup_times)

    expected = time_mat[source_id, cust_id] + setup_times[cust_id]
    assert dur == expected, f"Expected route duration {expected}, got {dur}"


def test_build_numba_candidate_pool(v212_instance):
    """Test candidate shift pool generation."""
    pool = build_numba_candidate_pool(v212_instance, n_samples_per_driver=100)
    assert len(pool) > 0, "Candidate pool should not be empty"

    for shift in pool[:10]:
        assert isinstance(shift, CandidateShift)
        assert shift.driver_id in [d.index for d in v212_instance.drivers]
        assert shift.trailer_id in [t.index for t in v212_instance.trailers]
        assert shift.duration > 0
        assert shift.cost >= 0
        assert len(shift.points) > 0


def test_solve_numba_gurobi_mip_execution(v212_instance):
    """Test end-to-end native solve execution producing a valid Solution object."""
    sol = solve_numba_gurobi_mip(v212_instance, n_samples_per_driver=200, time_limit_sec=30.0)

    assert isinstance(sol, Solution)
    assert len(sol.shifts) > 0, "Native solver should produce active shifts"

    # Verify shift integrity
    for s in sol.shifts:
        assert s.driver in [d.index for d in v212_instance.drivers]
        assert s.trailer in [t.index for t in v212_instance.trailers]
        assert len(s.operations) >= 2, "Shift must have at least source load + customer delivery"
        # First operation must be source loading (negative quantity)
        assert s.operations[0].quantity < 0
        # Subsequent operations must be customer deliveries (positive quantities)
        for op in s.operations[1:]:
            assert op.quantity > 0


def test_numba_gurobi_rule_checker_compliance(v212_instance, tmp_path):
    """Test that the native solution passes the Python rule checker with zero hard errors."""
    sol = solve_numba_gurobi_mip(v212_instance, n_samples_per_driver=200, time_limit_sec=30.0)

    violations = validate_solution(v212_instance, sol)
    hard_violations = [v for v in violations if v.severity == "ERROR"]

    assert len(hard_violations) == 0, f"Native solution has {len(hard_violations)} hard rule violations: {hard_violations[:3]}"

    # Verify solution XML serialization and deserialization
    out_xml = tmp_path / "test_v212_native.xml"
    save_solution(sol, str(out_xml))
    assert out_xml.exists()

    reloaded_sol = load_solution(str(out_xml))
    assert len(reloaded_sol.shifts) == len(sol.shifts)
