"""Fast mutable search substrate.

This package is the rebuild described in ``REBUILD_PLAN.md``.  It exists
alongside the legacy immutable ``Solution`` pipeline and does not modify it.

The contract is narrow and testable: :class:`~vrp_solver.fast.state.SearchState`
mirrors, exactly, the numbers that
:func:`vrp_solver.contest.score_prefix_with_feasibility_tail` reports for the
same solution, while supporting in-place mutation with incremental rescoring.
"""

from .objective import (
    AdaptiveWeights,
    Objective,
    SearchTelemetry,
    accept,
    acceptance_threshold,
)
from .state import FastInstance, SearchState, StateScore, instance_days

__all__ = [
    "AdaptiveWeights",
    "FastInstance",
    "Objective",
    "SearchState",
    "SearchTelemetry",
    "StateScore",
    "accept",
    "acceptance_threshold",
    "instance_days",
]
