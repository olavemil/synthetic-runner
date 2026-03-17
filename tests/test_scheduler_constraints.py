"""Tests for scheduling constraint budget tracking in checker.py."""

from __future__ import annotations

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from library.harness.checker import Checker
from library.harness.config import SchedulingConstraints
from library.harness.store import open_store, NamespacedStore


@pytest.fixture
def temp_db():
    """Create a temporary SQLite store for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store_db = open_store(db_path)
        yield store_db
        store_db.close()


@pytest.fixture
def checker_store(temp_db):
    """Create a namespaced store for checker state."""
    return NamespacedStore(temp_db, "checker")


class TestReplyBudget:
    """Tests for per-window reply budget checking."""

    def test_check_reply_budget_allows_when_no_constraints(self, checker_store):
        """Reply is allowed when no constraints configured."""
        result = Checker.check_reply_budget(checker_store, "test-instance", None)
        assert result is True

    def test_check_reply_budget_allows_under_limit(self, checker_store):
        """Reply is allowed when under the max_replies_per_window limit."""
        constraints = SchedulingConstraints(max_replies_per_window=3)
        checker_store.put("replies_this_window:test-instance", 2)

        result = Checker.check_reply_budget(checker_store, "test-instance", constraints)
        assert result is True

    def test_check_reply_budget_denies_at_limit(self, checker_store):
        """Reply is denied when at the max_replies_per_window limit."""
        constraints = SchedulingConstraints(max_replies_per_window=3)
        checker_store.put("replies_this_window:test-instance", 3)

        result = Checker.check_reply_budget(checker_store, "test-instance", constraints)
        assert result is False

    def test_check_reply_budget_denies_over_limit(self, checker_store):
        """Reply is denied when over the max_replies_per_window limit."""
        constraints = SchedulingConstraints(max_replies_per_window=3)
        checker_store.put("replies_this_window:test-instance", 5)

        result = Checker.check_reply_budget(checker_store, "test-instance", constraints)
        assert result is False

    def test_increment_reply_count(self, checker_store):
        """Incrementing reply count works correctly."""
        Checker.increment_reply_count(checker_store, "test-instance")
        assert checker_store.get("replies_this_window:test-instance") == 1

        Checker.increment_reply_count(checker_store, "test-instance")
        assert checker_store.get("replies_this_window:test-instance") == 2

    def test_increment_reply_count_handles_missing(self, checker_store):
        """Incrementing from no prior count starts at 1."""
        Checker.increment_reply_count(checker_store, "new-instance")
        assert checker_store.get("replies_this_window:new-instance") == 1


class TestReactiveThinkingBudget:
    """Tests for reactive thinking session budget checking."""

    def test_check_reactive_thinking_allows_when_no_constraints(self, checker_store):
        """Reactive thinking is allowed when no constraints configured."""
        result = Checker.check_reactive_thinking_budget(
            checker_store, "test-instance", None, 1000.0
        )
        assert result is True

    def test_check_reactive_thinking_allows_under_session_limit(self, checker_store):
        """Reactive thinking is allowed when under session limit."""
        constraints = SchedulingConstraints(
            reactive_thinking_max_sessions=2,
            reactive_thinking_cooldown=900,
        )
        checker_store.put("reactive_sessions:test-instance", 1)
        checker_store.put("last_reactive_session:test-instance", 500.0)

        result = Checker.check_reactive_thinking_budget(
            checker_store, "test-instance", constraints, 2000.0
        )
        assert result is True

    def test_check_reactive_thinking_denies_at_session_limit(self, checker_store):
        """Reactive thinking is denied when at session limit."""
        constraints = SchedulingConstraints(
            reactive_thinking_max_sessions=2,
            reactive_thinking_cooldown=900,
        )
        checker_store.put("reactive_sessions:test-instance", 2)

        result = Checker.check_reactive_thinking_budget(
            checker_store, "test-instance", constraints, 2000.0
        )
        assert result is False

    def test_check_reactive_thinking_denies_in_cooldown(self, checker_store):
        """Reactive thinking is denied when in cooldown period."""
        constraints = SchedulingConstraints(
            reactive_thinking_max_sessions=2,
            reactive_thinking_cooldown=900,
        )
        checker_store.put("reactive_sessions:test-instance", 1)
        checker_store.put("last_reactive_session:test-instance", 1500.0)

        # Current time is 2000, last session was at 1500, cooldown is 900
        # 2000 - 1500 = 500 < 900, so should be denied
        result = Checker.check_reactive_thinking_budget(
            checker_store, "test-instance", constraints, 2000.0
        )
        assert result is False

    def test_check_reactive_thinking_allows_after_cooldown(self, checker_store):
        """Reactive thinking is allowed after cooldown expires."""
        constraints = SchedulingConstraints(
            reactive_thinking_max_sessions=2,
            reactive_thinking_cooldown=900,
        )
        checker_store.put("reactive_sessions:test-instance", 1)
        checker_store.put("last_reactive_session:test-instance", 1000.0)

        # Current time is 2000, last session was at 1000, cooldown is 900
        # 2000 - 1000 = 1000 >= 900, so should be allowed
        result = Checker.check_reactive_thinking_budget(
            checker_store, "test-instance", constraints, 2000.0
        )
        assert result is True

    def test_record_reactive_session(self, checker_store):
        """Recording a reactive session updates counters correctly."""
        now = 1500.0
        Checker.record_reactive_session(checker_store, "test-instance", now)

        assert checker_store.get("reactive_sessions:test-instance") == 1
        assert checker_store.get("last_reactive_session:test-instance") == now

        Checker.record_reactive_session(checker_store, "test-instance", 2000.0)
        assert checker_store.get("reactive_sessions:test-instance") == 2
        assert checker_store.get("last_reactive_session:test-instance") == 2000.0


class TestWindowBudgetReset:
    """Tests for resetting window budgets on heartbeat."""

    def test_reset_window_budgets(self, checker_store):
        """Reset clears all budget counters."""
        checker_store.put("replies_this_window:test-instance", 5)
        checker_store.put("reactive_sessions:test-instance", 3)
        checker_store.put("last_reactive_session:test-instance", 1000.0)

        now = 2000.0
        Checker.reset_window_budgets(checker_store, "test-instance", now)

        assert checker_store.get("replies_this_window:test-instance") == 0
        assert checker_store.get("reactive_sessions:test-instance") == 0
        assert checker_store.get("last_guaranteed_thinking:test-instance") == now


class TestOnMessagePhase:
    """Tests for on_message phase restriction helper."""

    def test_get_on_message_phase_returns_none_when_no_constraints(self):
        """Returns None when no constraints configured."""
        result = Checker.get_on_message_phase(None)
        assert result is None

    def test_get_on_message_phase_returns_none_when_no_phases(self):
        """Returns None when no phase restrictions set."""
        constraints = SchedulingConstraints(on_message_thinking_phases=None)
        result = Checker.get_on_message_phase(constraints)
        assert result is None

    def test_get_on_message_phase_returns_phase(self):
        """Returns the phase when restrictions are set."""
        constraints = SchedulingConstraints(
            on_message_thinking_phases={"THINKING"}
        )
        result = Checker.get_on_message_phase(constraints)
        assert result == "THINKING"

    def test_get_on_message_phase_returns_first_from_set(self):
        """Returns one phase when multiple are set."""
        constraints = SchedulingConstraints(
            on_message_thinking_phases={"THINKING", "COMPOSING"}
        )
        result = Checker.get_on_message_phase(constraints)
        assert result in {"THINKING", "COMPOSING"}
