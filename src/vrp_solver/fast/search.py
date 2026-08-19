"""The search loop, and the selection controllers that drive it.

Step 4 of ``REBUILD_PLAN.md``, plus the measurement harness the step 2 gate
needs (that gate is a property of the neighbourhood, so it could not be
measured until step 3's portfolio existed).

The loop itself is deliberately thin.  Everything that decides *what* to try
lives in a :class:`Selector`, and everything that decides *whether to keep it*
lives in :mod:`vrp_solver.fast.objective`.  Step 4's whole point is to show
that each increment of controller complexity earns its place, which is only
possible if the controllers are interchangeable at one seam.

One step is:

1. ask the selector for a *sequence* of operator ids (length 1 for every
   controller except SSHH, which learns over pairs),
2. open a transaction, apply them in order,
3. score, price with the live objective, and apply Kheiri's acceptance rule,
4. commit or roll back, and tell the selector what happened.

A sequence that changes nothing counts as an empty neighbourhood and is
reported: it is the pathology this rebuild exists to remove, so it is measured
rather than assumed absent.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Sequence

from ..xml_io import Solution
from .decode import decode_shift_quantities
from .objective import AdaptiveWeights, Objective, SearchTelemetry, accept
from .operators import OPERATORS, OPERATOR_NAMES
from .retime import CandidateLists
from .state import SearchState, StateScore

N_OPERATORS = len(OPERATORS)

#: How many times a step reselects before conceding an empty neighbourhood.
#: With every operator firing on >= 90% of invocations (the step 3 gate), eight
#: independent draws make a genuinely empty step a ~1e-8 event, so the count
#: that remains means the *state* is dead, not the draw.
_EMPTY_RETRIES = 8


# --------------------------------------------------------------------------
# controllers
# --------------------------------------------------------------------------


class Selector:
    """Chooses which low-level heuristics to apply, and learns from outcomes.

    ``select`` returns a sequence of operator ids so that SSHH (step 4.4) fits
    the same interface as uniform random (4.1); the simpler controllers return
    a one-element sequence.
    """

    name = "selector"

    def select(self, rng: random.Random) -> Sequence[int]:  # pragma: no cover
        raise NotImplementedError

    def update(
        self,
        chosen: Sequence[int],
        *,
        improved_best: bool,
        gain: float,
        seconds: float,
    ) -> None:
        """Report the outcome of the last :meth:`select`.

        ``gain`` is ``quality(before) - quality(after)``, so positive is an
        improvement.  ``seconds`` is the wall time the step cost, which 4.2
        onwards use to avoid preferring an operator that improves twice as much
        at a hundred times the price.
        """


class UniformSelector(Selector):
    """Step 4.1: pick one operator uniformly at random. The baseline.

    Every later controller has to beat this, and the honest outcome is that it
    might not: a portfolio whose members are all individually reasonable can be
    hard to improve on by weighting.
    """

    name = "uniform"

    def __init__(self, n: int = N_OPERATORS) -> None:
        self.n = n

    def select(self, rng: random.Random) -> Sequence[int]:
        return (rng.randrange(self.n),)


class RouletteSelector(Selector):
    """Step 4.2: roulette on exponentially averaged reward per CPU second.

    The credit for one step is ``max(gain, 0) / seconds`` — an operator that
    improves twice as much but costs 100x is not a better default, so reward
    is a *rate*.  Both the reward and its second moment are tracked as
    exponential moving averages and the roulette weight is the mean divided by
    the RMS (Adam-style): the operators' raw gains differ by three orders of
    magnitude (``delete_shift`` swings thousands, ``shift_one_arrival``
    fractions), so an unnormalized mean would rank by amplitude when what
    predicts usefulness is signal-to-noise.

    ``floor`` keeps every operator selectable forever.  The V2.17 evidence for
    it is direct: ``insert_unsatisfied`` produced zero new bests over 46k calls
    on that instance yet is the operator that closed V2.15, so starving an
    operator on one instance's evidence is exactly the mistake a per-instance
    controller exists to avoid.
    """

    name = "roulette"

    def __init__(
        self,
        n: int = N_OPERATORS,
        *,
        beta1: float = 0.995,
        beta2: float = 0.999,
        floor: float = 0.05,
    ) -> None:
        self.n = n
        self.beta1 = beta1
        self.beta2 = beta2
        self.floor = floor
        self._mean = [0.0] * n
        self._second = [0.0] * n
        self._steps = [0] * n

    def select(self, rng: random.Random) -> Sequence[int]:
        weights = []
        for i in range(self.n):
            if self._steps[i] == 0:
                # Unsampled operators carry the average weight so the first
                # sweep is close to uniform rather than an artifact of
                # initialisation.
                weights.append(-1.0)
                continue
            # Bias-corrected EMAs, exactly Adam's m-hat / sqrt(v-hat).
            correct1 = 1.0 - self.beta1 ** self._steps[i]
            correct2 = 1.0 - self.beta2 ** self._steps[i]
            mean = self._mean[i] / correct1
            rms = (self._second[i] / correct2) ** 0.5
            weights.append(mean / (rms + 1e-12))
        sampled = [w for w in weights if w >= 0.0]
        fill = (sum(sampled) / len(sampled)) if sampled else 1.0
        top = max(max(weights), fill, 1e-12)
        floor_w = self.floor * top
        weights = [max(w if w >= 0.0 else fill, floor_w) for w in weights]
        total = sum(weights)
        draw = rng.random() * total
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if draw < acc:
                return (i,)
        return (self.n - 1,)

    def update(
        self,
        chosen: Sequence[int],
        *,
        improved_best: bool,
        gain: float,
        seconds: float,
    ) -> None:
        reward = max(gain, 0.0) / max(seconds, 1e-9)
        share = reward / len(chosen)
        for i in chosen:
            self._steps[i] += 1
            self._mean[i] = self.beta1 * self._mean[i] + (1 - self.beta1) * share
            self._second[i] = (
                self.beta2 * self._second[i] + (1 - self.beta2) * share * share
            )


@dataclass
class SearchResult:
    """What a run produced, and enough instrumentation to judge it."""

    best_solution: Solution
    best_score: StateScore
    best_quality: float
    telemetry: SearchTelemetry
    elapsed: float
    #: The best state under the lexicographic ``(errors, LR)`` order, which is
    #: the order that actually decides publication.  ``best_solution`` is best
    #: under the *live adaptive quality*, and those two differ: the adaptive
    #: weights decay, so a 32-error state can out-price a 7-error one and be
    #: reported as an improvement.  That is correct as a search driver and wrong
    #: as an artifact, so both are kept and the caller picks by purpose.
    published_solution: Solution | None = None
    published_score: StateScore | None = None
    operator_calls: list[int] = field(default_factory=list)
    operator_fires: list[int] = field(default_factory=list)
    operator_gains: list[float] = field(default_factory=list)
    operator_seconds: list[float] = field(default_factory=list)
    operator_bests: list[int] = field(default_factory=list)

    @property
    def steps(self) -> int:
        return self.telemetry.steps

    @property
    def steps_per_second(self) -> float:
        return self.telemetry.steps / max(1e-9, self.elapsed)

    def operator_table(self) -> str:
        """Per-operator call/fire/new-best/gain table. ASCII only (cp1252)."""
        rows = [
            f"{'operator':26s} {'calls':>7s} {'fire%':>6s} {'bests':>6s} "
            f"{'gain':>11s} {'ms/call':>8s}"
        ]
        for i, name in enumerate(OPERATOR_NAMES):
            calls = self.operator_calls[i]
            if not calls:
                continue
            rows.append(
                f"{name:26s} {calls:7d} "
                f"{self.operator_fires[i] / calls:6.1%} "
                f"{self.operator_bests[i]:6d} "
                f"{self.operator_gains[i]:11.5f} "
                f"{1000.0 * self.operator_seconds[i] / calls:8.3f}"
            )
        return "\n".join(rows)


def run_search(
    state: SearchState,
    *,
    limit: float,
    seed: int = 1,
    selector: Selector | None = None,
    lists: CandidateLists | None = None,
    adaptive: AdaptiveWeights | None = None,
    objective: Objective | None = None,
    max_steps: int | None = None,
    decode: bool = True,
) -> SearchResult:
    """Run the search on ``state`` for ``limit`` seconds. Mutates ``state``.

    ``adaptive`` and ``objective`` are alternatives: pass an
    :class:`AdaptiveWeights` for step 2.2's moving weights, or a fixed
    :class:`Objective` to hold them still (which is what the ablation needs, so
    two controllers are compared on the same landscape).

    ``decode`` re-derives quantities on the shifts a step touched, following
    TS2020 sec. 3.1 where a solution is routes only and quantities are a decode
    step rather than decision variables. Keep it on; the flag exists so the
    ablation can measure what the decoder is worth.
    """
    rng = random.Random(seed)
    selector = selector or UniformSelector()
    lists = lists or CandidateLists(state.fi)
    if adaptive is None and objective is None:
        adaptive = AdaptiveWeights()
    live = objective if adaptive is None else adaptive.objective()

    telemetry = SearchTelemetry()
    calls = [0] * N_OPERATORS
    fires = [0] * N_OPERATORS
    gains = [0.0] * N_OPERATORS
    seconds = [0.0] * N_OPERATORS
    bests = [0] * N_OPERATORS

    score = state.score()
    current_quality = live.quality(score)
    best_score = score
    best_quality = current_quality
    best_solution = state.to_solution()
    telemetry.record_best(best_score, best_quality)

    def _rank(s: StateScore) -> tuple[int, float]:
        """Publication order: fewer errors first, then lower ratio."""
        return (s.feasibility_errors, s.logistic_ratio)

    published_score = score
    published_solution = best_solution

    started = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - started
        if elapsed >= limit:
            break
        if max_steps is not None and telemetry.steps >= max_steps:
            break

        step_started = time.perf_counter()
        # Reselect while the chosen sequence produces nothing.  Two operators
        # legitimately decline a few percent of the time (`two_opt_star` needs
        # two shifts with a splittable route, `create_shift` needs a driver
        # window that opens before the score cutoff), and the step 2 gate asks
        # for *zero* steps with an empty neighbourhood.  Absorbing that here is
        # the right place: a step is a unit of search progress, and one
        # operator declining is not a reason to spend one.  The retry count is
        # bounded so a genuinely dead state still terminates the loop rather
        # than spinning, and `empty_neighbourhood` then records it honestly.
        chosen: Sequence[int] = ()
        changed = False
        for _attempt in range(_EMPTY_RETRIES):
            chosen = selector.select(rng)
            state.begin()
            for index in chosen:
                calls[index] += 1
                if OPERATORS[index][1](state, rng, lists):
                    fires[index] += 1
                    changed = True
            if changed:
                break
            state.rollback()
            selector.update(chosen, improved_best=False, gain=0.0, seconds=0.0)
        if not changed:
            telemetry.empty_neighbourhood += 1
            telemetry.steps += 1
            step_cost = time.perf_counter() - step_started
            for index in chosen:
                seconds[index] += step_cost / max(1, len(chosen))
            continue

        # Quantities are derived, so a structural move leaves them stale: an
        # inserted stop carries zero and a resequenced route may now be able to
        # give more or less at each stop. Decode before scoring, and only on the
        # shifts this step actually touched -- decoding every shift would put an
        # O(all shifts) pass in the inner loop, which is the cost this rebuild
        # exists to remove.
        if decode:
            for touched in state.touched_positions():
                decode_shift_quantities(state, touched)

        candidate = state.score()
        # Captured here, while the candidate state is still live: a rejected
        # step is rolled back below, and a state can be the best one ever seen
        # by error count without being accepted under the live weights.
        if _rank(candidate) < _rank(published_score):
            published_score = candidate
            published_solution = state.to_solution(drop_empty=True)
        if adaptive is not None:
            adaptive.observe(candidate, best_feasible=best_score.feasible)
            live = adaptive.objective()
            # The landscape just moved, so the incumbent has to be re-priced
            # under the new weights or the comparison is between two different
            # objectives.
            current_quality = live.quality(score)
            best_quality = live.quality(best_score)
        candidate_quality = live.quality(candidate)

        keep = accept(
            current_quality=current_quality,
            candidate_quality=candidate_quality,
            best_quality=best_quality,
            best_feasible=best_score.feasible,
            elapsed=elapsed,
            limit=limit,
        )
        gain = current_quality - candidate_quality
        improved_best = candidate_quality < best_quality

        if keep:
            state.commit()
            score = candidate
            current_quality = candidate_quality
            if improved_best:
                best_score = candidate
                best_quality = candidate_quality
                best_solution = state.to_solution()
                telemetry.record_best(best_score, best_quality)
                for index in chosen:
                    bests[index] += 1
        else:
            state.rollback()

        telemetry.record(candidate, accepted=keep)
        step_cost = time.perf_counter() - step_started
        for index in chosen:
            seconds[index] += step_cost / len(chosen)
            gains[index] += gain / len(chosen)
        selector.update(
            chosen, improved_best=improved_best, gain=gain, seconds=step_cost
        )

    return SearchResult(
        best_solution=best_solution,
        best_score=best_score,
        published_solution=published_solution,
        published_score=published_score,
        best_quality=best_quality,
        telemetry=telemetry,
        elapsed=time.perf_counter() - started,
        operator_calls=calls,
        operator_fires=fires,
        operator_gains=gains,
        operator_seconds=seconds,
        operator_bests=bests,
    )
