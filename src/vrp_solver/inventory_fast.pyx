# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Fast inventory projection and violation counting for the scoring hot path.

This module replaces the Python TankEvent-based scoring pipeline with pure
array-based computation. The key insight is that the scoring functions
(tank_violations count, safety_kg_minutes, hard_violation count) only need
aggregate numeric results -- not per-step TankEvent objects.
"""
import numpy as np
cimport numpy as cnp

def project_inventory_core(
    double initial_qty,
    double[:] forecast,
    double[:] deliveries_by_step,
    double capacity,
    double safety_level,
    int horizon,
    int point_index,
    int unit_mins,
    int is_call_in
):
    """Original per-customer inventory projection returning arrays."""
    cdef double[:] inventory_out = np.empty(horizon, dtype=np.float64)
    cdef int[:] breach_out = np.zeros(horizon, dtype=np.int32)

    cdef double inventory = initial_qty
    cdef double ending = 0.0
    cdef double consumed = 0.0
    cdef double delivered = 0.0
    cdef double EPS = 1e-6
    cdef int step = 0

    for step in range(horizon):
        delivered = deliveries_by_step[step]
        consumed = forecast[step]

        # Ending = (Inventory - Consumed) + Delivered
        ending = (inventory - consumed) + delivered
        inventory_out[step] = ending

        # Safety breach check
        if is_call_in == 0:
            if ending < (safety_level - EPS):
                breach_out[step] = 1

        inventory = ending

    return np.asarray(inventory_out), np.asarray(breach_out)


def score_all_customers(
    double[:] initial_quantities,
    double[:,:] forecasts,
    double[:] capacities,
    double[:] safety_levels,
    int[:] is_call_in,
    int num_customers,
    int horizon,
    int unit_mins,
    double[:,:] deliveries_matrix,
):
    """Compute aggregate violation counts and safety deficit for ALL customers.

    This replaces the entire tank_events() -> tank_violations() -> 
    _safety_kg_minutes() pipeline with a single pass over all customers,
    returning only the aggregate numbers the scoring functions need.

    Parameters
    ----------
    initial_quantities : array of initial tank quantities per customer
    forecasts : 2D array [num_customers x horizon] of consumption forecasts
    capacities : array of tank capacities per customer
    safety_levels : array of safety levels per customer
    is_call_in : int array (1 = call-in customer, 0 = VMI)
    num_customers : number of customers
    horizon : number of time steps
    unit_mins : minutes per time step
    deliveries_matrix : 2D array [num_customers x horizon] of deliveries

    Returns
    -------
    tuple of (safety_breach_count, negative_count, overfill_count,
              hard_violations, safety_kg_min,
              breach_points_count, negative_points_count)
    """
    cdef int total_safety_breaches = 0
    cdef int total_negatives = 0
    cdef int total_overfills = 0
    cdef int hard_violations = 0
    cdef double safety_kg_min = 0.0

    cdef int breach_points = 0
    cdef int negative_points = 0

    cdef double inventory, ending, consumed, delivered, deficit
    cdef double EPS = 1e-6
    cdef int c, step
    cdef int customer_had_breach, customer_had_negative

    for c in range(num_customers):
        if is_call_in[c] == 1:
            continue

        inventory = initial_quantities[c]
        customer_had_breach = 0
        customer_had_negative = 0

        for step in range(horizon):
            delivered = deliveries_matrix[c, step]
            consumed = forecasts[c, step]
            ending = (inventory - consumed) + delivered

            # Tank negative -> hard violation (DYN01)
            if ending < -EPS:
                total_negatives += 1
                hard_violations += 1
                if customer_had_negative == 0:
                    customer_had_negative = 1
                    negative_points += 1

            # Tank overfill
            if ending > capacities[c] + EPS:
                total_overfills += 1

            # Safety breach
            if ending < safety_levels[c] - EPS:
                total_safety_breaches += 1
                if customer_had_breach == 0:
                    customer_had_breach = 1
                    breach_points += 1

            # Safety deficit accumulation (for penalty)
            deficit = safety_levels[c] - ending - EPS
            if deficit > 0.0:
                safety_kg_min += deficit * unit_mins

            inventory = ending

    return (
        total_safety_breaches,
        total_negatives,
        total_overfills,
        hard_violations,
        safety_kg_min,
        breach_points,
        negative_points,
    )
