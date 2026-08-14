"""Unit tests for MarkovSequenceSelector and LateAcceptanceBuffer."""
from __future__ import annotations

import pytest

from vrp_solver.solver.markov_sequence import (
    DEFAULT_LLH_CATALOG,
    LLHTier,
    LateAcceptanceBuffer,
    MarkovConfig,
    MarkovSequenceSelector,
)


def test_markov_sequence_generation():
    selector = MarkovSequenceSelector(seed=42)
    for _ in range(20):
        seq = selector.generate_sequence()
        assert len(seq) >= 2
        # First operator must be Ruin
        first_desc = next(d for d in DEFAULT_LLH_CATALOG if d.name == seq[0])
        assert first_desc.tier == LLHTier.RUIN
        # Second operator must be Recreate
        second_desc = next(d for d in DEFAULT_LLH_CATALOG if d.name == seq[1])
        assert second_desc.tier == LLHTier.RECREATE


def test_markov_reward_and_decay():
    config = MarkovConfig(initial_weight=1.0, reward_accept=1.0, reward_best=3.0, decay_rate=0.1)
    selector = MarkovSequenceSelector(config=config, seed=42)
    
    seq = ["ruin_random_shifts", "cluster_greedy_insert", "swap_operations"]
    src_idx = selector.name_to_idx["ruin_random_shifts"]
    dst_idx = selector.name_to_idx["cluster_greedy_insert"]
    
    initial_w = selector.matrix[src_idx][dst_idx]
    assert initial_w == 1.0
    
    # Reward sequence
    selector.reward_sequence(seq, is_best=False)
    assert selector.matrix[src_idx][dst_idx] == 2.0
    
    selector.reward_sequence(seq, is_best=True)
    assert selector.matrix[src_idx][dst_idx] == 5.0
    
    # Decay matrix
    selector.decay_matrix()
    assert pytest.approx(selector.matrix[src_idx][dst_idx], rel=1e-5) == 4.5


def test_late_acceptance_buffer():
    # Buffer of size 3
    lahc = LateAcceptanceBuffer(capacity=3, initial_score=100.0)
    
    # Iteration 0: compared to 100.0. Candidate = 90.0 (Accept)
    acc, thresh = lahc.assess_and_record(90.0)
    assert acc is True
    assert thresh == 100.0
    
    # Iteration 1: compared to 100.0. Candidate = 110.0 (Reject)
    acc, thresh = lahc.assess_and_record(110.0)
    assert acc is False
    assert thresh == 100.0
    
    # Iteration 2: compared to 100.0. Candidate = 95.0 (Accept)
    acc, thresh = lahc.assess_and_record(95.0)
    assert acc is True
    assert thresh == 100.0
    
    # Iteration 3: compared to Buffer[0] which is 90.0!
    # Candidate = 92.0 -> should be rejected because 92.0 > 90.0
    acc, thresh = lahc.assess_and_record(92.0)
    assert acc is False
    assert thresh == 90.0
    
    # Iteration 4: compared to Buffer[1] which is 100.0!
    # Candidate = 98.0 -> should be accepted because 98.0 <= 100.0
    acc, thresh = lahc.assess_and_record(98.0)
    assert acc is True
    assert thresh == 100.0
