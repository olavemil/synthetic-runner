"""Tests for library.tools.voting."""

from __future__ import annotations

import random

from library.tools.voting import borda_tally, approval_weights, weighted_sample
from library.tools.identity import Identity


class TestBordaTally:
    def test_correct_winner(self):
        candidates = {"a": "option a", "b": "option b", "c": "option c"}
        votes = {
            "v1": ["b", "a", "c"],
            "v2": ["b", "c", "a"],
            "v3": ["a", "b", "c"],
        }
        tally = borda_tally(candidates, votes)
        assert tally["winner_member"] == "b"
        assert tally["candidate_count"] == 3
        assert tally["vote_count"] == 3

    def test_returns_correct_fields(self):
        candidates = {"x": "text x", "y": "text y"}
        votes = {"v1": ["x", "y"]}
        tally = borda_tally(candidates, votes)
        assert "winner_member" in tally
        assert "winner_message" in tally
        assert "scores" in tally
        assert "is_tie" in tally
        assert tally["winner_message"] == "text x"

    def test_no_votes_all_score_one(self):
        candidates = {"a": "aa", "b": "bb"}
        tally = borda_tally(candidates, {})
        assert tally["scores"]["a"] == 1
        assert tally["scores"]["b"] == 1

    def test_single_candidate_wins(self):
        candidates = {"only": "text"}
        tally = borda_tally(candidates, {})
        assert tally["winner_member"] == "only"

    def test_is_tie_true_when_all_equal(self):
        candidates = {"a": "aa", "b": "bb"}
        # No votes → both score 1 → tie
        tally = borda_tally(candidates, {})
        assert tally["is_tie"] is True

    def test_is_tie_false_when_clear_winner(self):
        candidates = {"a": "aa", "b": "bb"}
        votes = {"v1": ["a", "b"], "v2": ["a", "b"]}
        tally = borda_tally(candidates, votes)
        assert tally["is_tie"] is False
        assert tally["winner_member"] == "a"

    def test_is_tie_false_for_single_candidate(self):
        candidates = {"only": "text"}
        tally = borda_tally(candidates, {})
        assert tally["is_tie"] is False


class TestApprovalWeights:
    def test_basic_weights(self):
        identities = [
            Identity(name="a", approval=2),
            Identity(name="b", approval=0),
            Identity(name="c", approval=-1),
        ]
        weights = approval_weights(identities)
        # floor=1 by default: max(0, approval+1)
        assert weights[0] == 3.0  # 2+1
        assert weights[1] == 1.0  # 0+1
        assert weights[2] == 0.0  # max(0, -1+1)

    def test_custom_floor(self):
        identities = [Identity(name="x", approval=5)]
        weights = approval_weights(identities, floor=0)
        assert weights[0] == 5.0

    def test_all_negative_gives_zeros(self):
        identities = [Identity(name="x", approval=-5), Identity(name="y", approval=-5)]
        weights = approval_weights(identities)
        assert all(w == 0.0 for w in weights)


class TestWeightedSample:
    def test_returns_n_items(self):
        pop = list(range(10))
        weights = [1.0] * 10
        result = weighted_sample(pop, weights, 5)
        assert len(result) == 5

    def test_no_duplicates(self):
        pop = list(range(10))
        weights = [float(i + 1) for i in range(10)]
        rng = random.Random(42)
        for _ in range(20):
            result = weighted_sample(pop, weights, 5, rng)
            assert len(result) == len(set(result))

    def test_n_greater_than_population_caps(self):
        pop = [1, 2, 3]
        weights = [1.0, 1.0, 1.0]
        result = weighted_sample(pop, weights, 10)
        assert len(result) == 3

    def test_all_zero_weights_falls_back_to_uniform(self):
        pop = [1, 2, 3, 4, 5]
        weights = [0.0, 0.0, 0.0, 0.0, 0.0]
        rng = random.Random(7)
        result = weighted_sample(pop, weights, 3, rng)
        assert len(result) == 3

    def test_empty_population(self):
        assert weighted_sample([], [], 3) == []

    def test_high_weight_item_selected_more(self):
        pop = [0, 1]
        weights = [1.0, 100.0]
        rng = random.Random(0)
        counts = {0: 0, 1: 0}
        for _ in range(200):
            result = weighted_sample(pop, weights, 1, rng)
            counts[result[0]] += 1
        assert counts[1] > counts[0]
