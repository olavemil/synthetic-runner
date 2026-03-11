"""Tests for probabilistic decision utilities."""

from __future__ import annotations

import random
from unittest.mock import patch

import pytest

from library.tools.decisions import probabilistic


class TestProbabilistic:
    def test_always_true_at_one(self):
        """Value of 1.0 should always return True."""
        for _ in range(100):
            assert probabilistic(1.0) is True

    def test_always_false_at_zero(self):
        """Value of 0.0 should always return False."""
        for _ in range(100):
            assert probabilistic(0.0) is False

    def test_clamps_above_one(self):
        """Values > 1.0 should be clamped to 1.0 (always True)."""
        for _ in range(100):
            assert probabilistic(1.5) is True

    def test_clamps_below_zero(self):
        """Values < 0.0 should be clamped to 0.0 (always False)."""
        for _ in range(100):
            assert probabilistic(-0.5) is False

    def test_statistical_distribution(self):
        """Value of 0.7 should fire roughly 70% of the time."""
        random.seed(42)
        trials = 10000
        fires = sum(1 for _ in range(trials) if probabilistic(0.7))
        ratio = fires / trials
        assert 0.65 < ratio < 0.75

    def test_midpoint_distribution(self):
        """Value of 0.5 should fire roughly 50% of the time."""
        random.seed(42)
        trials = 10000
        fires = sum(1 for _ in range(trials) if probabilistic(0.5))
        ratio = fires / trials
        assert 0.45 < ratio < 0.55

    def test_low_value_fires_less(self):
        """0.1 should fire less than 0.9."""
        random.seed(42)
        trials = 5000
        fires_low = sum(1 for _ in range(trials) if probabilistic(0.1))
        random.seed(42)
        fires_high = sum(1 for _ in range(trials) if probabilistic(0.9))
        assert fires_low < fires_high

    def test_label_triggers_logging(self):
        """When label is provided, should log the decision."""
        with patch("library.tools.decisions.logger") as mock_logger:
            random.seed(42)
            probabilistic(0.5, label="test_decision")
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            assert "test_decision" in str(call_args)

    def test_no_label_no_logging(self):
        """When no label, should not log."""
        with patch("library.tools.decisions.logger") as mock_logger:
            probabilistic(0.5)
            mock_logger.info.assert_not_called()

    def test_deterministic_with_seed(self):
        """Same seed should produce same result."""
        random.seed(123)
        r1 = probabilistic(0.5)
        random.seed(123)
        r2 = probabilistic(0.5)
        assert r1 == r2
