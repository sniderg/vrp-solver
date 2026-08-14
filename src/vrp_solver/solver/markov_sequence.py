"""Adaptive Markov Sequence Selection (SSHH) and Late Acceptance (LAHC).

Implements the sequence-based selection hyper-heuristic paradigm inspired by
Ahmed Kheiri (Transportation Science 2020) for the ROADEF 2016 IRP.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import random
from typing import Callable, List, Optional, Sequence, Tuple


class LLHTier(Enum):
    """Categorization of low-level heuristics."""
    RUIN = auto()         # Disruption / mutational operators
    RECREATE = auto()     # Construction / insertion operators
    LOCAL_SEARCH = auto() # Polish / shift-level refinement operators


@dataclass(frozen=True)
class LLHDescriptor:
    """Metadata and tier for a low-level heuristic operator."""
    name: str
    tier: LLHTier
    description: str


# Standard catalog of operators
DEFAULT_LLH_CATALOG: Tuple[LLHDescriptor, ...] = (
    # Tier 1: Ruin / Mutational
    LLHDescriptor("ruin_random_shifts", LLHTier.RUIN, "Unassign 1-3 random shifts"),
    LLHDescriptor("ruin_pressure_customers", LLHTier.RUIN, "Eject deliveries for highest runout-pressure customers"),
    LLHDescriptor("eject_customer_operation", LLHTier.RUIN, "Remove a single customer delivery from a shift"),
    LLHDescriptor("state_preserving_split", LLHTier.RUIN, "Split an over-extended shift preserving trailer continuity"),
    # Tier 2: Recreate / Insertion
    LLHDescriptor("cluster_greedy_insert", LLHTier.RECREATE, "Insert unassigned customers using spatiotemporal clustering"),
    LLHDescriptor("urgency_pressure_insert", LLHTier.RECREATE, "Insert deliveries for customers nearest stockout"),
    LLHDescriptor("cross_shift_reinsert", LLHTier.RECREATE, "Reinsert ejected customer operations into adjacent shifts"),
    LLHDescriptor("trailer_block_recombine", LLHTier.RECREATE, "Recombine trailer blocks across driver shift chains"),
    # Tier 3: Local Search / Repair
    LLHDescriptor("swap_operations", LLHTier.LOCAL_SEARCH, "Swap operation order within a shift"),
    LLHDescriptor("shift_timing_2opt", LLHTier.LOCAL_SEARCH, "Optimize arrival timestamps and break placement"),
    LLHDescriptor("fast_quantity_repair", LLHTier.LOCAL_SEARCH, "Adjust continuous delivery quantities"),
)


@dataclass
class MarkovConfig:
    """Configuration for Markov Transition Matrix and Sequence Generation."""
    initial_weight: float = 1.0
    min_weight: float = 0.05
    reward_accept: float = 0.5
    reward_best: float = 2.0
    decay_rate: float = 0.02
    min_sequence_length: int = 2
    max_sequence_length: int = 4
    temperature: float = 1.0


class MarkovSequenceSelector:
    """Adaptive Markov chain managing transition probabilities between LLHs."""

    def __init__(
        self,
        catalog: Sequence[LLHDescriptor] = DEFAULT_LLH_CATALOG,
        config: Optional[MarkovConfig] = None,
        seed: Optional[int] = None,
    ):
        self.catalog = tuple(catalog)
        self.config = config or MarkovConfig()
        self.rng = random.Random(seed)
        self.name_to_idx = {desc.name: i for i, desc in enumerate(self.catalog)}
        self.n = len(self.catalog)
        
        # Initialize N x N transition matrix
        self.matrix: List[List[float]] = [
            [self.config.initial_weight for _ in range(self.n)]
            for _ in range(self.n)
        ]

    def sample_next(self, current_idx: int, allowed_indices: Optional[Sequence[int]] = None) -> int:
        """Sample the next operator index using softmax / roulette wheel from row current_idx."""
        weights = [self.matrix[current_idx][j] for j in range(self.n)]
        if allowed_indices is not None:
            allowed_set = set(allowed_indices)
            weights = [w if i in allowed_set else 0.0 for i, w in enumerate(weights)]
        
        total = sum(weights)
        if total <= 1e-9:
            # Fallback to uniform among allowed
            candidates = list(allowed_indices) if allowed_indices is not None else list(range(self.n))
            return self.rng.choice(candidates)
        
        # Roulette selection
        r = self.rng.uniform(0.0, total)
        accum = 0.0
        for idx, w in enumerate(weights):
            accum += w
            if accum >= r:
                return idx
        return len(weights) - 1

    def generate_sequence(self, start_tier: LLHTier = LLHTier.RUIN) -> List[str]:
        """Generate a meaningful sequence: Ruin -> Recreate -> [Optional Local Search]."""
        tier1_indices = [i for i, desc in enumerate(self.catalog) if desc.tier == LLHTier.RUIN]
        tier2_indices = [i for i, desc in enumerate(self.catalog) if desc.tier == LLHTier.RECREATE]
        tier3_indices = [i for i, desc in enumerate(self.catalog) if desc.tier == LLHTier.LOCAL_SEARCH]

        # 1. Start with a Ruin operator
        start_idx = self.rng.choice(tier1_indices)
        sequence_indices = [start_idx]

        # 2. Transition to Recreate operator
        next_idx = self.sample_next(start_idx, allowed_indices=tier2_indices)
        sequence_indices.append(next_idx)

        # 3. Optionally transition to Local Search operator or another Recreate
        target_len = self.rng.randint(self.config.min_sequence_length, self.config.max_sequence_length)
        while len(sequence_indices) < target_len:
            curr = sequence_indices[-1]
            # Next can be Tier 2 or Tier 3
            allowed = tier2_indices + tier3_indices
            step_idx = self.sample_next(curr, allowed_indices=allowed)
            sequence_indices.append(step_idx)
            if self.catalog[step_idx].tier == LLHTier.LOCAL_SEARCH:
                # Local search naturally terminates sequence
                break

        return [self.catalog[idx].name for idx in sequence_indices]

    def reward_sequence(self, sequence_names: Sequence[str], is_best: bool = False) -> None:
        """Reinforce transitions that formed the successful sequence."""
        if len(sequence_names) < 2:
            return
        delta = self.config.reward_best if is_best else self.config.reward_accept
        for i in range(len(sequence_names) - 1):
            src = self.name_to_idx.get(sequence_names[i])
            dst = self.name_to_idx.get(sequence_names[i + 1])
            if src is not None and dst is not None:
                self.matrix[src][dst] += delta

    def decay_matrix(self) -> None:
        """Apply periodic decay to transition weights to prevent stagnation."""
        for i in range(self.n):
            for j in range(self.n):
                self.matrix[i][j] = max(
                    self.config.min_weight,
                    self.matrix[i][j] * (1.0 - self.config.decay_rate)
                )


class LateAcceptanceBuffer:
    """Late Acceptance Hill Climbing (LAHC) circular acceptance buffer.

    Accepts candidate S' if Score(S') <= Buffer[t % L], allowing the solver
    to cross non-improving intermediate structural valleys without greedy stagnation.
    """

    def __init__(self, capacity: int = 50, initial_score: float = float("inf")):
        self.capacity = max(1, capacity)
        self.buffer = [initial_score] * self.capacity
        self.iteration = 0

    def threshold(self) -> float:
        """Get the current comparison threshold from L iterations ago."""
        return self.buffer[self.iteration % self.capacity]

    def assess_and_record(self, candidate_score: float) -> Tuple[bool, float]:
        """Assess candidate score against threshold, record current score, and advance index.

        Returns (accepted, threshold_used).
        """
        curr_thresh = self.threshold()
        # Accept if candidate is better than or equal to the historical state
        accepted = candidate_score <= curr_thresh + 1e-9
        
        # Update buffer at current slot with current accepted/incumbent score
        effective_score = candidate_score if accepted else curr_thresh
        self.buffer[self.iteration % self.capacity] = effective_score
        self.iteration += 1
        return accepted, curr_thresh

    def reset(self, initial_score: float) -> None:
        """Reset buffer with a new baseline score."""
        self.buffer = [initial_score] * self.capacity
        self.iteration = 0
