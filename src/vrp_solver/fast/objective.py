"""Single penalized objective and acceptance rule.

Step 2 of ``REBUILD_PLAN.md``.

The premise of the rebuild is that infeasibility must be *priced*, never used
as a generation gate.  The legacy search discarded candidates that broke a
hard rule, which is why 96% of its wall time went into candidate generation
that returned nothing: the only reachable states were the feasible ones, and
from a tight incumbent there are almost none.

Here every state has a finite quality::

    quality = LR + sum_i w_i * violation_i

with ``LR = cost / max(1, delivered)``.  A move is therefore always
evaluable, and the search decides what to do with a violation instead of
being forbidden from seeing it.

Weight scale
------------
The starting weights come from the magnitudes already encoded in
``surgical_search._scalar``, which lexicographically ordered
``(hard_violations, feasibility_errors, safety_kg_min, LR)`` by using
``1e6 * hard + 1e3 * errors + 1e-5 * safety_kg_min + LR``.  Set B logistic
ratios sit around 0.02-0.2, so a unit weight of 1.0 per rule error already
makes one violation dominate any LR gain -- which is the intent: a violation
should be worth trading only against a large routing improvement, and never
free.  These are *starting* values; :class:`AdaptiveWeights` moves them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .state import StateScore

# Violation groups the objective prices, in the order used everywhere below.
GROUPS = (
    "shift_errors",
    "trailer_errors",
    "driver_errors",
    "tank_negative_steps",
    "tank_overfill_steps",
    "tank_safety_breach_steps",
    "callin_errors",
)

# Starting weights, per unit of the corresponding counter.
#
# Rationale for the relative order, strongest first:
#   * ``tank_negative_steps``  -- a customer ran dry; the deepest failure and
#     the one the checker treats as unrecoverable in practice.
#   * ``shift_errors``, ``trailer_errors``, ``driver_errors`` -- structural
#     rule breaks (resource compatibility, driving time, load carry-over).
#     Repairable by a later move, so priced below running dry.
#   * ``callin_errors`` -- an unmet call-in order; a missed obligation rather
#     than a broken route.
#   * ``tank_overfill_steps`` -- overfill is a quantity error, usually fixed
#     by one quantity move, so it is the cheapest counted error.
#   * ``tank_safety_breach_steps`` -- QS02.  Counted per *step*, so a single
#     bad delivery can contribute hundreds; weighted far lower per unit than
#     the others for that reason, with the depth of the breach carried by
#     ``safety_weight`` below instead.
DEFAULT_WEIGHTS = {
    "shift_errors": 1.0,
    "trailer_errors": 1.0,
    "driver_errors": 1.0,
    "tank_negative_steps": 2.0,
    "tank_overfill_steps": 0.5,
    "tank_safety_breach_steps": 0.05,
    "callin_errors": 0.75,
}

# ``safety_kg_min`` is an integral in kg-minutes and runs to 1e6-1e8, so it
# needs a small coefficient to sit alongside an LR of order 0.1.  1e-8 makes a
# 1e7 kg-min deficit worth about 0.1 -- comparable to the whole LR, i.e. worth
# fixing, without swamping the rule counters.
DEFAULT_SAFETY_WEIGHT = 1e-8


def group_counts(score: StateScore) -> dict[str, int]:
    """The violation counters the objective prices, by group name."""
    return {name: getattr(score, name) for name in GROUPS}


@dataclass(frozen=True)
class Objective:
    """A weighting of the violation groups against the logistic ratio."""

    weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_WEIGHTS)
    )
    safety_weight: float = DEFAULT_SAFETY_WEIGHT

    def penalty(self, score: StateScore) -> float:
        total = self.safety_weight * score.safety_kg_min
        for name, weight in self.weights.items():
            count = getattr(score, name)
            if count:
                total += weight * count
        return total

    def quality(self, score: StateScore) -> float:
        """Lower is better.  Finite for every state, feasible or not."""
        return score.logistic_ratio + self.penalty(score)

    def scaled(self, factor: float) -> "Objective":
        """This objective with every violation weight multiplied by ``factor``."""
        return replace(
            self,
            weights={name: w * factor for name, w in self.weights.items()},
            safety_weight=self.safety_weight * factor,
        )


class AdaptiveWeights:
    """Raise the weight of a violation while it persists; decay when absent.

    Step 2.2.  The mechanism is deliberately per-group: a search that keeps
    breaking driving-time limits should find driving-time limits expensive
    without also being told that overfill is expensive.

    ``target_infeasible_fraction`` is a global brake on top of that.  The
    analysis document suggests keeping 15-35% of *visited* states infeasible:
    below that band the search is over-constrained and behaves like the legacy
    gated version; above it, it wanders. Both bounds are starting guesses to be
    measured (step 2.4), not trusted.

    The two mechanisms are kept on *separate* multipliers -- a per-group scale
    and one global scale -- because they operate on different timescales and
    would otherwise fight.  Per-group scales move every observation, so over a
    200-step window they travel by ``factor ** 200``; a global adjustment
    applied to the same numbers once per window is swamped by three orders of
    magnitude and the brake never actually engages.  Multiplying the two
    independently lets the per-group ratios express *which* violation is
    expensive while the global scale sets *how* expensive violations are
    overall.
    """

    def __init__(
        self,
        objective: Objective | None = None,
        *,
        raise_factor: float = 1.05,
        decay_factor: float = 0.97,
        min_scale: float = 0.05,
        max_scale: float = 200.0,
        global_step: float = 1.5,
        min_global_scale: float = 0.05,
        max_global_scale: float = 200.0,
        target_infeasible_fraction: tuple[float, float] = (0.15, 0.35),
        window: int = 200,
    ) -> None:
        base = objective or Objective()
        self._base_weights = dict(base.weights)
        self._base_safety = base.safety_weight
        self._scale = {name: 1.0 for name in self._base_weights}
        self._safety_scale = 1.0
        self._global_scale = 1.0
        self.raise_factor = raise_factor
        self.decay_factor = decay_factor
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.global_step = global_step
        self.min_global_scale = min_global_scale
        self.max_global_scale = max_global_scale
        self.target_low, self.target_high = target_infeasible_fraction
        self.window = window
        self._visited = 0
        self._infeasible = 0

    # -- observation -------------------------------------------------------

    def observe(self, score: StateScore, *, best_feasible: bool = True) -> None:
        """Update weights from one visited state.

        ``best_feasible`` gates the global brake only.  Measured on V2.15 from
        an infeasible seed, leaving it ungated ran the brake to its ceiling and
        kept it there: the visited-infeasible fraction was 99.6% simply because
        no feasible state had been found yet, so every window read as
        "violations underpriced" and the scale ratcheted by 1.5 each time.  With
        weights 200x up, quality reached 2.5e5 and the run *lost* ground against
        fixed weights (33 errors versus 16).

        The band is only meaningful once there is a feasible incumbent to
        return to.  While the best is infeasible, the search is already spending
        everything it has on repair -- LR is negligible next to the penalty --
        and raising the weights adds nothing but scale.  Per-group weights keep
        moving throughout, since which violation is expensive is useful
        information either way.
        """
        self._visited += 1
        if not score.feasible:
            self._infeasible += 1

        for name in self._base_weights:
            if getattr(score, name):
                self._scale[name] = min(
                    self.max_scale, self._scale[name] * self.raise_factor
                )
            else:
                self._scale[name] = max(
                    self.min_scale, self._scale[name] * self.decay_factor
                )
        if score.safety_kg_min > 0.0:
            self._safety_scale = min(
                self.max_scale, self._safety_scale * self.raise_factor
            )
        else:
            self._safety_scale = max(
                self.min_scale, self._safety_scale * self.decay_factor
            )
        self._normalize()

        if self._visited >= self.window:
            if best_feasible:
                self._rebalance()
            self._visited = 0
            self._infeasible = 0

    def _normalize(self) -> None:
        """Hold the largest per-group scale at 1, so only *ratios* persist.

        Without this the mechanism is useless over a long run: at
        ``raise_factor = 1.05`` a group saturates ``max_scale = 200`` in about
        108 observations, so on a 100k-step V2.15 run every group that ever
        fires ends pinned at 200 and the scales carry no information -- measured
        exactly that way before this was added. Overall magnitude is the global
        scale's job; this keeps the two mechanisms from duplicating each other.

        Skipped while every scale is still at or below 1, so an all-feasible run
        can decay to the floor as :meth:`scales` promises.
        """
        top = max(self._scale.values())
        if self._safety_scale > top:
            top = self._safety_scale
        if top <= 1.0:
            return
        inverse = 1.0 / top
        for name in self._scale:
            self._scale[name] = max(self.min_scale, self._scale[name] * inverse)
        self._safety_scale = max(self.min_scale, self._safety_scale * inverse)

    def _rebalance(self) -> None:
        """Move the global scale so the infeasible fraction returns to the band.

        Too many infeasible states visited means violations are underpriced,
        so the scale goes up; too few means the search is over-constrained and
        is behaving like the legacy gated version, so it comes down.
        """
        fraction = self._infeasible / max(1, self._visited)
        if fraction > self.target_high:
            factor = self.global_step
        elif fraction < self.target_low:
            factor = 1.0 / self.global_step
        else:
            return
        self._global_scale = min(
            self.max_global_scale,
            max(self.min_global_scale, self._global_scale * factor),
        )

    # -- readout -----------------------------------------------------------

    @property
    def infeasible_fraction(self) -> float:
        return self._infeasible / max(1, self._visited)

    @property
    def global_scale(self) -> float:
        return self._global_scale

    def objective(self) -> Objective:
        g = self._global_scale
        return Objective(
            weights={
                name: self._base_weights[name] * self._scale[name] * g
                for name in self._base_weights
            },
            safety_weight=self._base_safety * self._safety_scale * g,
        )

    def scales(self) -> dict[str, float]:
        """Per-group multipliers, for the step 2.4 weight trajectory.

        These exclude the global scale; read that from :attr:`global_scale`.
        """
        out = dict(self._scale)
        out["safety_kg_min"] = self._safety_scale
        return out


def acceptance_threshold(
    *, best_feasible: bool, elapsed: float, limit: float
) -> float:
    """Kheiri's ``T`` from TS2020 eq. 4.

    While the incumbent is infeasible, ``T`` is a flat 0.001: the search is
    still looking for anything valid and can afford a wide band.  Once the
    incumbent is feasible, ``T`` starts near 0.0101 and anneals to 0.0001 as
    the time limit approaches, so late steps are close to hill-climbing.
    """
    if not best_feasible:
        return 0.001
    if limit <= 0.0:
        return 0.0001
    remaining = 1.0 - min(1.0, max(0.0, elapsed / limit))
    return 0.0001 + 0.01 * remaining


def accept(
    *,
    current_quality: float,
    candidate_quality: float,
    best_quality: float,
    best_feasible: bool,
    elapsed: float,
    limit: float,
) -> bool:
    """Step 2.3: accept an equal-or-better move, or a near-best worse one.

    Note the asymmetry that makes this work as a diversifier: the first clause
    compares against *current*, so plateau moves are always taken; the second
    compares against *best*, so a worsening move is only allowed while the
    state is still close to the best ever seen.
    """
    if candidate_quality <= current_quality:
        return True
    threshold = acceptance_threshold(
        best_feasible=best_feasible, elapsed=elapsed, limit=limit
    )
    return candidate_quality < best_quality + threshold * abs(best_quality)


@dataclass
class SearchTelemetry:
    """Step 2.4 instrumentation.

    Kept as plain counters so it costs nothing in the inner loop.  The
    numerator and denominator of LR are tracked separately on purpose: it is
    the check for contingency 2B, where LR is gamed by inflating delivered
    quantity instead of reducing cost.
    """

    steps: int = 0
    accepted: int = 0
    improved_best: int = 0
    visited_infeasible: int = 0
    empty_neighbourhood: int = 0

    best_quality: float = float("inf")
    best_cost: float = 0.0
    best_delivered: float = 0.0
    best_errors: int = 0

    def record(self, score: StateScore, *, accepted: bool) -> None:
        self.steps += 1
        if accepted:
            self.accepted += 1
        if not score.feasible:
            self.visited_infeasible += 1

    def record_best(self, score: StateScore, quality: float) -> None:
        self.improved_best += 1
        self.best_quality = quality
        self.best_cost = score.scored_estimated_cost
        self.best_delivered = score.scored_delivered_quantity
        self.best_errors = score.feasibility_errors

    @property
    def accepted_fraction(self) -> float:
        return self.accepted / max(1, self.steps)

    @property
    def infeasible_fraction(self) -> float:
        return self.visited_infeasible / max(1, self.steps)

    def summary(self) -> str:
        # ASCII only: Windows consoles use cp1252.
        return (
            f"steps={self.steps} accepted={self.accepted_fraction:.1%} "
            f"infeasible={self.infeasible_fraction:.1%} "
            f"empty={self.empty_neighbourhood} "
            f"best_q={self.best_quality:.6f} "
            f"best_cost={self.best_cost:.1f} "
            f"best_delivered={self.best_delivered:.1f} "
            f"best_errors={self.best_errors}"
        )
