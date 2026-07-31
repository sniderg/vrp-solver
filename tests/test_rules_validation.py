"""Comprehensive unit tests for rules.py validation and checker rule enforcement."""
import pytest
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
from vrp_solver.rules import (
    _validate_shift_references,
    is_driving_duration_valid,
    is_time_window_valid,
    is_trailer_allowed,
    validate_solution,
)


@pytest.fixture
def base_instance():
    """Create a minimal 2-day VRP instance with 1 driver, 1 trailer, 1 source, and 2 customers."""
    driver = Driver(
        index=0,
        min_inter_shift_duration=540,
        max_driving_duration=540,
        trailer_ids=(0,),
        time_windows=(TimeWindow(start=0, end=2880),),
        layover_duration=540,
        time_cost=10.0,
        layover_cost=50.0,
    )
    trailer = Trailer(
        index=0,
        capacity=20_000.0,
        initial_quantity=10_000.0,
        distance_cost=1.0,
    )
    source = Source(
        index=0,
        allowed_trailers=(0,),
        setup_time=30,
    )
    customer1 = Customer(
        index=1,
        layover_customer=False,
        call_in=False,
        orders=(),
        setup_time=30,
        time_windows=(TimeWindow(start=0, end=2880),),
        allowed_trailers=(0,),
        forecast=(10.0,) * 48,
        capacity=15_000.0,
        initial_tank_quantity=5_000.0,
        min_operation_quantity=2_000.0,
        safety_level=1_000.0,
    )
    customer2 = Customer(
        index=2,
        layover_customer=False,
        call_in=False,
        orders=(),
        setup_time=30,
        time_windows=(TimeWindow(start=0, end=2880),),
        allowed_trailers=(0,),
        forecast=(5.0,) * 48,
        capacity=10_000.0,
        initial_tank_quantity=8_000.0,
        min_operation_quantity=1_000.0,
        safety_level=500.0,
    )
    time_matrix = (
        (0, 60, 120),
        (60, 0, 90),
        (120, 90, 0),
    )
    dist_matrix = (
        (0.0, 50.0, 100.0),
        (50.0, 0.0, 80.0),
        (100.0, 80.0, 0.0),
    )
    return Instance(
        name="TestInstance",
        unit=60,
        horizon=48,
        time_matrix=time_matrix,
        distance_matrix=dist_matrix,
        base_index=0,
        drivers=(driver,),
        trailers=(trailer,),
        sources=(source,),
        customers=(customer1, customer2),
    )


def test_atomic_helpers(base_instance):
    """Test atomic rule checkers."""
    tw = (TimeWindow(0, 100), TimeWindow(200, 300))
    assert is_time_window_valid(10, 50, tw)
    assert not is_time_window_valid(50, 150, tw)

    assert is_trailer_allowed(base_instance, 1, 0)
    driver = base_instance.drivers[0]
    assert is_driving_duration_valid(driver, 500)
    assert not is_driving_duration_valid(driver, 600)


def test_valid_solution_has_no_errors(base_instance):
    """Verify a valid solution produces zero rule violations."""
    shift = Shift(
        index=0,
        driver=0,
        trailer=0,
        start=100,
        operations=(
            Operation(point=0, arrival=160, quantity=-10_000.0),
            Operation(point=1, arrival=250, quantity=10_000.0),
        ),
    )
    sol = Solution(shifts=(shift,))
    violations = validate_solution(base_instance, sol)
    errors = [v for v in violations if v.severity == "error"]
    assert len(errors) == 0


def test_invalid_driver_and_trailer_reference(base_instance):
    """Test REF_DRIVER and REF_TRAILER rules."""
    shift = Shift(
        index=0,
        driver=99,
        trailer=99,
        start=100,
        operations=(),
    )
    sol = Solution(shifts=(shift,))
    violations = _validate_shift_references(base_instance, sol)
    codes = {v.code for v in violations}
    assert "REF_DRIVER" in codes
    assert "REF_TRAILER" in codes


def test_layover_without_layover_customer(base_instance):
    """Test LAY02 rule (layover without layover customer)."""
    shift = Shift(
        index=0,
        driver=0,
        trailer=0,
        start=100,
        operations=(
            Operation(point=0, arrival=160, quantity=-5_000.0),
            Operation(point=1, arrival=1500, quantity=5_000.0),
        ),
    )
    sol = Solution(shifts=(shift,))
    codes = {v.code for v in validate_solution(base_instance, sol)}
    assert "LAY02" in codes


def test_travel_time_arrival_violation(base_instance):
    """Test SHI02 rule (arrival before required travel time)."""
    shift = Shift(
        index=0,
        driver=0,
        trailer=0,
        start=100,
        operations=(
            Operation(point=1, arrival=110, quantity=5_000.0), # Required arrival from base is 100 + 60 = 160
        ),
    )
    sol = Solution(shifts=(shift,))
    codes = {v.code for v in validate_solution(base_instance, sol)}
    assert "SHI02" in codes


def test_disallowed_trailer_at_customer(base_instance):
    """Test SHI05 rule (trailer not allowed at customer/source)."""
    cust_restricted = Customer(
        index=1,
        layover_customer=False,
        call_in=False,
        orders=(),
        setup_time=30,
        time_windows=(TimeWindow(start=0, end=2880),),
        allowed_trailers=(1,), # Only trailer 1 allowed
        forecast=(10.0,) * 48,
        capacity=15_000.0,
        initial_tank_quantity=5_000.0,
        min_operation_quantity=2_000.0,
        safety_level=1_000.0,
    )
    inst = Instance(
        name="TestInstance",
        unit=60,
        horizon=48,
        time_matrix=base_instance.time_matrix,
        distance_matrix=base_instance.distance_matrix,
        base_index=0,
        drivers=base_instance.drivers,
        trailers=base_instance.trailers,
        sources=base_instance.sources,
        customers=(cust_restricted, base_instance.customers[1]),
    )

    shift = Shift(
        index=0,
        driver=0,
        trailer=0,
        start=100,
        operations=(
            Operation(point=0, arrival=160, quantity=-5_000.0),
            Operation(point=1, arrival=250, quantity=5_000.0),
        ),
    )
    sol = Solution(shifts=(shift,))
    codes = {v.code for v in validate_solution(inst, sol)}
    assert "SHI05" in codes


def test_delivery_quantity_bounds(base_instance):
    """Test SHI11 and SHI16 rules (negative delivery, exceeding capacity, below min operation quantity)."""
    shift_neg = Shift(
        index=0,
        driver=0,
        trailer=0,
        start=100,
        operations=(
            Operation(point=1, arrival=250, quantity=-500.0),
        ),
    )
    codes_neg = {v.code for v in validate_solution(base_instance, Solution((shift_neg,)))}
    assert "SHI11" in codes_neg

    shift_over = Shift(
        index=0,
        driver=0,
        trailer=0,
        start=100,
        operations=(
            Operation(point=0, arrival=160, quantity=-10_000.0),
            Operation(point=1, arrival=250, quantity=18_000.0),
        ),
    )
    codes_over = {v.code for v in validate_solution(base_instance, Solution((shift_over,)))}
    assert "SHI16" in codes_over

    shift_under = Shift(
        index=0,
        driver=0,
        trailer=0,
        start=100,
        operations=(
            Operation(point=0, arrival=160, quantity=-500.0),
            Operation(point=1, arrival=250, quantity=500.0),
        ),
    )
    codes_under = {v.code for v in validate_solution(base_instance, Solution((shift_under,)))}
    assert "SHI16" in codes_under


def test_trailer_inventory_overfill(base_instance):
    """Test SHI06 rule (trailer inventory exceeds capacity)."""
    shift = Shift(
        index=0,
        driver=0,
        trailer=0,
        start=100,
        operations=(
            Operation(point=0, arrival=160, quantity=-15_000.0), # Loading 15k on initial 10k exceeds capacity 20k
        ),
    )
    sol = Solution(shifts=(shift,))
    codes = {v.code for v in validate_solution(base_instance, sol)}
    assert "SHI06" in codes


def test_driver_inter_shift_rest_violation(base_instance):
    """Test DRI01 rule (driver rest duration between shifts)."""
    shift1 = Shift(
        index=0,
        driver=0,
        trailer=0,
        start=100,
        operations=(
            Operation(point=0, arrival=160, quantity=-5_000.0),
            Operation(point=1, arrival=250, quantity=5_000.0),
        ),
    )
    shift2 = Shift(
        index=1,
        driver=0,
        trailer=0,
        start=300,
        operations=(
            Operation(point=0, arrival=360, quantity=-5_000.0),
        ),
    )
    sol = Solution(shifts=(shift1, shift2))
    codes = {v.code for v in validate_solution(base_instance, sol)}
    assert "DRI01" in codes


def test_trailer_overlap_violation(base_instance):
    """Test TL01 rule (trailer used in overlapping shifts by different drivers)."""
    driver2 = Driver(
        index=1,
        min_inter_shift_duration=540,
        max_driving_duration=540,
        trailer_ids=(0,),
        time_windows=(TimeWindow(start=0, end=2880),),
        layover_duration=540,
        time_cost=10.0,
        layover_cost=50.0,
    )
    inst = Instance(
        name="TestInstance",
        unit=60,
        horizon=48,
        time_matrix=base_instance.time_matrix,
        distance_matrix=base_instance.distance_matrix,
        base_index=0,
        drivers=(base_instance.drivers[0], driver2),
        trailers=base_instance.trailers,
        sources=base_instance.sources,
        customers=base_instance.customers,
    )

    shift1 = Shift(
        index=0,
        driver=0,
        trailer=0,
        start=100,
        operations=(
            Operation(point=0, arrival=160, quantity=-5_000.0),
            Operation(point=1, arrival=250, quantity=5_000.0),
        ),
    )
    shift2 = Shift(
        index=1,
        driver=1,
        trailer=0,
        start=200,
        operations=(
            Operation(point=0, arrival=260, quantity=-5_000.0),
        ),
    )
    sol = Solution(shifts=(shift1, shift2))
    codes = {v.code for v in validate_solution(inst, sol)}
    assert "TL01" in codes


def test_call_in_order_rules(base_instance):
    """Test QS01 and QS03 rules for call-in customers."""
    order = Order(
        quantity=5_000.0,
        earliest_time=100,
        latest_time=500,
        quantity_flexibility=80,
    )
    call_in_cust = Customer(
        index=1,
        layover_customer=False,
        call_in=True,
        orders=(order,),
        setup_time=30,
        time_windows=(TimeWindow(start=0, end=2880),),
        allowed_trailers=(0,),
        forecast=(0.0,) * 48,
        capacity=15_000.0,
        initial_tank_quantity=5_000.0,
        min_operation_quantity=2_000.0,
        safety_level=1_000.0,
    )
    inst = Instance(
        name="TestInstance",
        unit=60,
        horizon=48,
        time_matrix=base_instance.time_matrix,
        distance_matrix=base_instance.distance_matrix,
        base_index=0,
        drivers=base_instance.drivers,
        trailers=base_instance.trailers,
        sources=base_instance.sources,
        customers=(call_in_cust, base_instance.customers[1]),
    )

    sol_empty = Solution(shifts=())
    codes_qs01 = {v.code for v in validate_solution(inst, sol_empty)}
    assert "QS01" in codes_qs01

    shift_late = Shift(
        index=0,
        driver=0,
        trailer=0,
        start=500,
        operations=(
            Operation(point=0, arrival=560, quantity=-5_000.0),
            Operation(point=1, arrival=650, quantity=5_000.0),
        ),
    )
    codes_qs03 = {v.code for v in validate_solution(inst, Solution((shift_late,)))}
    assert "QS03" in codes_qs03
