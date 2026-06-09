# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
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
    # Output arrays for the fields we care about
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
