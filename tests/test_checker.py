"""Tests for Checker — polling and schedule checking."""

from __future__ import annotations

import logging
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest
import yaml

from library.harness.checker import Checker
from library.harness.adapters import Event
from library.harness.config import (
    HarnessConfig,
    AdapterConfig,
    InstanceConfig,
    MessagingConfig,
    SpaceMapping,
)
from library.harness.jobqueue import JobQueue
from library.harness.registry import Registry
from library.harness.store import open_store, NamespacedStore
from library.species import Species, SpeciesManifest, EntryPoint


_SPECIES_ID = "test-species"


def _make_species(schedule=None):
    eps = []
    if schedule:
        eps.append(EntryPoint(name="heartbeat", schedule=schedule, trigger=None, handler=MagicMock()))
    manifest = SpeciesManifest(
        species_id=_SPECIES_ID,
        entry_points=eps,
        tools=[],
        default_files={},
        spawn=lambda ctx: None,
    )
    s = MagicMock(spec=Species)
    s.manifest.return_value = manifest
    return s


def _make_instance(
    instance_id="inst-1",
    species_id=_SPECIES_ID,
    schedule=None,
    with_messaging=False,
    entity_id="@bot:matrix.org",
):
    messaging = None
    if with_messaging:
        messaging = MessagingConfig(
            adapter="test-adapter",
            entity_id=entity_id,
            access_token="token",
            spaces=[SpaceMapping(name="main", handle="!room:matrix.org")],
        )
    return InstanceConfig(
        instance_id=instance_id,
        species=species_id,
        provider="test",
        model="test-model",
        messaging=messaging,
        schedule=schedule or {},
    )


def _make_harness_config(adapter_type="local_file"):
    return HarnessConfig(
        providers=[],
        adapters=[AdapterConfig(id="test-adapter", type=adapter_type, base_dir="/tmp/test")],
    )


class TestCheckerSchedule:
    def test_run_logs_start_and_scheduled_count(self, caplog):
        db = open_store()
        species = _make_species(schedule="* * * * *")
        instance = _make_instance(schedule={})
        registry = Registry()
        registry.register_species(species)
        registry.register_instance(instance)

        checker = Checker(
            harness_config=_make_harness_config(),
            registry=registry,
            store_db=db,
            base_dir=None,
        )
        checker._store.put("schedule_next:inst-1:heartbeat", time.time() - 1)

        with caplog.at_level(logging.INFO):
            checker.run()

        messages = [r.getMessage() for r in caplog.records]
        assert any("Checker cycle started (instances=1)" in m for m in messages)
        assert any("Checker cycle finished" in m and "scheduled=1" in m for m in messages)

    def test_enqueues_due_scheduled_ep(self):
        db = open_store()
        species = _make_species(schedule="* * * * *")
        instance = _make_instance(schedule={})
        registry = Registry()
        registry.register_species(species)
        registry.register_instance(instance)

        checker = Checker(
            harness_config=_make_harness_config(),
            registry=registry,
            store_db=db,
            base_dir=None,
        )

        # First call registers next_fire
        checker._check_schedules(set())
        queue = JobQueue(db)
        assert queue.pending_count() == 0  # not yet due

        # Manually set next_fire to past
        checker._store.put("schedule_next:inst-1:heartbeat", time.time() - 1)

        checker._check_schedules(set())
        assert queue.pending_count() == 1

    def test_no_duplicate_enqueue(self):
        db = open_store()
        species = _make_species(schedule="* * * * *")
        instance = _make_instance(schedule={})
        registry = Registry()
        registry.register_species(species)
        registry.register_instance(instance)

        checker = Checker(
            harness_config=_make_harness_config(),
            registry=registry,
            store_db=db,
            base_dir=None,
        )

        checker._store.put("schedule_next:inst-1:heartbeat", time.time() - 1)
        checker._check_schedules(set())
        checker._store.put("schedule_next:inst-1:heartbeat", time.time() - 1)
        checker._check_schedules(set())  # should not enqueue again

        queue = JobQueue(db)
        assert queue.pending_count() == 1

    def test_run_prioritizes_messages_before_scheduled_jobs(self):
        db = open_store()
        species = _make_species(schedule="* * * * *")
        instance = _make_instance(with_messaging=True)
        registry = Registry()
        registry.register_species(species)
        registry.register_instance(instance)

        checker = Checker(
            harness_config=_make_harness_config(adapter_type="matrix"),
            registry=registry,
            store_db=db,
            base_dir=None,
        )
        queue = JobQueue(db)

        # Make heartbeat due now.
        checker._store.put("schedule_next:inst-1:heartbeat", time.time() - 1)
        # Pretend we already synced once so polling can produce reactive events.
        checker._store.put("inst-1:main", "tok0")

        evt = Event(
            event_id="$evt1",
            sender="@user:matrix.org",
            body="hello",
            timestamp=1,
            room="!room:matrix.org",
        )
        adapter = MagicMock()
        adapter.poll.return_value = ([evt], "tok1")

        with patch.object(checker, "_get_adapter", return_value=adapter):
            checker.run()

        # Messages are polled first, so on_message gets enqueued.
        # The instance guard then blocks the heartbeat.
        pending = queue.list_pending()
        assert len(pending) == 1
        assert pending[0].entry_point == "on_message"
        assert checker._store.get("external_count:inst-1") == 1

    def test_idle_throttling(self):
        db = open_store()
        species = _make_species(schedule="* * * * *")
        instance = _make_instance(schedule={"max_idle_heartbeats": 2, "max_thinks_per_reply": 10})
        registry = Registry()
        registry.register_species(species)
        registry.register_instance(instance)

        checker = Checker(
            harness_config=_make_harness_config(),
            registry=registry,
            store_db=db,
            base_dir=None,
        )
        queue = JobQueue(db)

        def _trigger():
            checker._store.put("schedule_next:inst-1:heartbeat", time.time() - 1)
            checker._check_schedules(set())  # no reactive instances
            if queue.has_active("inst-1"):
                job = queue.claim_next("w")
                queue.complete(job, "w")

        _trigger()  # idle=1, enqueues
        _trigger()  # idle=2, enqueues
        count_before = queue.pending_count()
        _trigger()  # idle=3 >= max_idle=2, skips
        assert queue.pending_count() == count_before  # nothing added


    def test_heartbeats_increment_thinks_counter(self):
        """With inverted throttling, heartbeats always run and increment thinks counter."""
        db = open_store()
        species = _make_species(schedule="* * * * *")
        instance = _make_instance(schedule={"min_thinks_per_reply": 3})
        registry = Registry()
        registry.register_species(species)
        registry.register_instance(instance)

        checker = Checker(
            harness_config=_make_harness_config(),
            registry=registry,
            store_db=db,
            base_dir=None,
        )
        queue = JobQueue(db)

        def _trigger():
            checker._store.put("schedule_next:inst-1:heartbeat", time.time() - 1)
            checker._check_schedules(set())
            if queue.has_active("inst-1"):
                job = queue.claim_next("w")
                queue.complete(job, "w")

        # Heartbeats always enqueue and increment counter (no throttling)
        thinks = checker._store.get("thinks_since_reply:inst-1")
        assert thinks is None or thinks == 0
        _trigger()  # thinks: 0 -> 1
        assert checker._store.get("thinks_since_reply:inst-1") == 1
        _trigger()  # thinks: 1 -> 2
        assert checker._store.get("thinks_since_reply:inst-1") == 2
        _trigger()  # thinks: 2 -> 3
        assert checker._store.get("thinks_since_reply:inst-1") == 3

    def test_thinks_counter_persists_on_message_arrival(self):
        """With inverted throttling, counter does NOT reset on message arrival."""
        db = open_store()
        species = _make_species(schedule="* * * * *")
        instance = _make_instance(schedule={"min_thinks_per_reply": 3}, with_messaging=True)
        registry = Registry()
        registry.register_species(species)
        registry.register_instance(instance)

        checker = Checker(
            harness_config=_make_harness_config(adapter_type="matrix"),
            registry=registry,
            store_db=db,
            base_dir=None,
        )
        queue = JobQueue(db)

        # Set thinks counter to 5
        checker._store.put("thinks_since_reply:inst-1", 5)

        # Simulate a message arriving
        checker._store.put("inst-1:main", "tok0")
        evt = Event(
            event_id="$evt1",
            sender="@user:matrix.org",
            body="hello",
            timestamp=1,
            room="!room:matrix.org",
        )
        adapter = MagicMock()
        adapter.poll.return_value = ([evt], "tok1")

        with patch.object(checker, "_get_adapter", return_value=adapter):
            checker._poll_instance(instance)

        # Counter should NOT be reset (only ctx.send resets it)
        assert checker._store.get("thinks_since_reply:inst-1") == 5


class TestCheckerReactive:
    def test_reactive_instances_reset_idle(self):
        db = open_store()
        species = _make_species(schedule="* * * * *")
        instance = _make_instance(schedule={"max_idle_heartbeats": 1})
        registry = Registry()
        registry.register_species(species)
        registry.register_instance(instance)

        checker = Checker(
            harness_config=_make_harness_config(),
            registry=registry,
            store_db=db,
            base_dir=None,
        )
        # Saturate idle count
        checker._store.put("idle:inst-1", 5)

        queue = JobQueue(db)
        checker._store.put("schedule_next:inst-1:heartbeat", time.time() - 1)
        # Pass inst-1 as reactive (just got messages)
        checker._check_schedules({"inst-1"})

        # max_idle throttle should NOT apply since inst-1 is in reactive_instances
        # (the idle_count check is skipped for reactive instances)
        # Verify idle was not the blocking factor by checking the queue
        # The schedule check passes because inst-1 is in reactive_instances
        assert queue.pending_count() == 1

    def test_poll_instance_enqueues_on_message_and_stores_events_in_inbox(self):
        db = open_store()
        species = _make_species(schedule="* * * * *")
        instance = _make_instance(with_messaging=True)
        registry = Registry()
        registry.register_species(species)
        registry.register_instance(instance)

        checker = Checker(
            harness_config=_make_harness_config(adapter_type="matrix"),
            registry=registry,
            store_db=db,
            base_dir=None,
        )
        queue = JobQueue(db)

        evt = Event(
            event_id="$evt1",
            sender="@user:matrix.org",
            body="hello",
            timestamp=1,
            room="!room:matrix.org",
        )
        adapter = MagicMock()
        # First poll initializes token and should not enqueue.
        # Second poll should enqueue on_message and store events in inbox.
        adapter.poll.side_effect = [([evt], "tok1"), ([evt], "tok2")]

        with patch.object(checker, "_get_adapter", return_value=adapter):
            got_messages_1, enqueued_1 = checker._poll_instance(instance)
            got_messages_2, enqueued_2 = checker._poll_instance(instance)

        assert got_messages_1 is False
        assert enqueued_1 is False
        assert got_messages_2 is True
        assert enqueued_2 is True

        pending = queue.list_pending()
        assert len(pending) == 1

        # Events are in the persistent inbox, not the job payload
        checker_store = NamespacedStore(db, "checker")
        inbox_events = Checker.drain_pending_events(checker_store, "inst-1")
        assert len(inbox_events) == 1
        assert inbox_events[0]["event_id"] == "$evt1"
        # Event room should be normalized to logical space name by checker.
        assert inbox_events[0]["room"] == "main"
        assert checker._store.get("external_count:inst-1") == 1

    def test_poll_instance_resolves_missing_entity_id_via_adapter_and_filters_self(self):
        db = open_store()
        species = _make_species(schedule="* * * * *")
        instance = _make_instance(with_messaging=True, entity_id="")
        registry = Registry()
        registry.register_species(species)
        registry.register_instance(instance)

        checker = Checker(
            harness_config=_make_harness_config(adapter_type="matrix"),
            registry=registry,
            store_db=db,
            base_dir=None,
        )
        queue = JobQueue(db)

        evt_self = Event(
            event_id="$self",
            sender="@bot:matrix.org",
            body="my own message",
            timestamp=1,
            room="!room:matrix.org",
        )
        evt_external = Event(
            event_id="$ext",
            sender="@user:matrix.org",
            body="hello",
            timestamp=2,
            room="!room:matrix.org",
        )
        adapter = MagicMock()
        adapter.get_entity_id.return_value = "@bot:matrix.org"
        adapter.poll.side_effect = [
            ([evt_self, evt_external], "tok1"),
            ([evt_self, evt_external], "tok2"),
        ]

        with patch.object(checker, "_get_adapter", return_value=adapter):
            got_messages_1, enqueued_1 = checker._poll_instance(instance)
            got_messages_2, enqueued_2 = checker._poll_instance(instance)

        assert got_messages_1 is False
        assert enqueued_1 is False
        assert got_messages_2 is True
        assert enqueued_2 is True
        pending = queue.list_pending()
        assert len(pending) == 1

        # Events are in the persistent inbox, not the job payload
        checker_store = NamespacedStore(db, "checker")
        inbox_events = Checker.drain_pending_events(checker_store, "inst-1")
        assert len(inbox_events) == 1
        assert inbox_events[0]["event_id"] == "$ext"
        assert checker._store.get("external_count:inst-1") == 1
        assert checker._resolved_entity_ids["inst-1"] == "@bot:matrix.org"

    def test_events_survive_when_job_is_running(self):
        """Events appended to the inbox are preserved even when a job is already running."""
        db = open_store()
        species = _make_species(schedule="* * * * *")
        instance = _make_instance(with_messaging=True)
        registry = Registry()
        registry.register_species(species)
        registry.register_instance(instance)

        checker = Checker(
            harness_config=_make_harness_config(adapter_type="matrix"),
            registry=registry,
            store_db=db,
            base_dir=None,
        )
        queue = JobQueue(db)

        evt1 = Event(event_id="$1", sender="@user:matrix.org", body="first", timestamp=1, room="!room:matrix.org")
        evt2 = Event(event_id="$2", sender="@user:matrix.org", body="second", timestamp=2, room="!room:matrix.org")

        adapter = MagicMock()
        # Poll 1: init sync (skipped), Poll 2: first message, Poll 3: second message while running
        adapter.poll.side_effect = [([evt1], "tok1"), ([evt1], "tok2"), ([evt2], "tok3")]

        with patch.object(checker, "_get_adapter", return_value=adapter):
            checker._poll_instance(instance)  # init sync
            checker._poll_instance(instance)  # enqueues on_message, evt1 in inbox

        # Simulate the worker claiming the job (now running)
        job = queue.claim_next("worker-1")
        assert job is not None

        # Poll again while job is running — events should go to inbox
        with patch.object(checker, "_get_adapter", return_value=adapter):
            got_messages, enqueued = checker._poll_instance(instance)

        assert got_messages is True
        # Can't enqueue or merge (running), but events are safe in inbox
        checker_store = NamespacedStore(db, "checker")
        inbox = checker_store.get("pending_events:inst-1")
        assert isinstance(inbox, list)
        # evt1 was already drained by... wait, nobody drained it yet.
        # Both evt1 and evt2 should be in the inbox.
        event_ids = [e["event_id"] for e in inbox]
        assert "$1" in event_ids
        assert "$2" in event_ids

    def test_poll_instance_uses_fallback_when_entity_id_unknown(self, caplog):
        """When entity_id lookup fails, use instance_id as fallback to allow reactive events."""
        db = open_store()
        species = _make_species(schedule="* * * * *")
        instance = _make_instance(with_messaging=True, entity_id="")
        registry = Registry()
        registry.register_species(species)
        registry.register_instance(instance)

        checker = Checker(
            harness_config=_make_harness_config(adapter_type="matrix"),
            registry=registry,
            store_db=db,
            base_dir=None,
        )
        queue = JobQueue(db)

        evt = Event(
            event_id="$evt1",
            sender="@someone:matrix.org",  # Different from fallback "inst-1"
            body="hello",
            timestamp=1,
            room="!room:matrix.org",
        )
        adapter = MagicMock()
        adapter.get_entity_id.return_value = ""
        adapter.poll.side_effect = [([evt], "tok1"), ([evt], "tok2")]

        with patch.object(checker, "_get_adapter", return_value=adapter):
            with caplog.at_level(logging.WARNING):
                got_messages_1, enqueued_1 = checker._poll_instance(instance)
                got_messages_2, enqueued_2 = checker._poll_instance(instance)

        # First poll is initial sync
        assert got_messages_1 is False
        assert enqueued_1 is False
        # Second poll uses fallback entity_id and processes event
        assert got_messages_2 is True
        assert enqueued_2 is True
        assert queue.pending_count() == 1
        # Warning mentions fallback
        messages = [r.getMessage() for r in caplog.records]
        assert any("using instance_id 'inst-1' as fallback" in m for m in messages)

    def test_poll_instance_timeout_is_logged_without_exception_trace(self, caplog):
        db = open_store()
        species = _make_species(schedule="* * * * *")
        instance = _make_instance(with_messaging=True)
        registry = Registry()
        registry.register_species(species)
        registry.register_instance(instance)

        checker = Checker(
            harness_config=_make_harness_config(adapter_type="matrix"),
            registry=registry,
            store_db=db,
            base_dir=None,
        )
        adapter = MagicMock()
        adapter.poll.side_effect = httpx.ReadTimeout("timed out")

        with patch.object(checker, "_get_adapter", return_value=adapter):
            with caplog.at_level(logging.WARNING):
                got_messages, enqueued = checker._poll_instance(instance)

        assert got_messages is False
        assert enqueued is False
        messages = [r.getMessage() for r in caplog.records]
        assert any("Timeout polling inst-1/main" in m for m in messages)


class TestReplyRateLimit:
    """Tests for reply rate limiting."""

    def _make_store(self):
        store_db = open_store(":memory:")
        return NamespacedStore(store_db, "checker")

    def test_no_limits_configured(self):
        store = self._make_store()
        assert Checker.check_reply_rate(store, "inst-1", {}) is True

    def test_cooldown_allows_after_elapsed(self):
        store = self._make_store()
        store.put("last_reply_time:inst-1", time.time() - 200)
        assert Checker.check_reply_rate(store, "inst-1", {"reply_cooldown_seconds": 120}) is True

    def test_cooldown_blocks_within_window(self):
        store = self._make_store()
        store.put("last_reply_time:inst-1", time.time() - 30)
        assert Checker.check_reply_rate(store, "inst-1", {"reply_cooldown_seconds": 120}) is False

    def test_hourly_cap_allows_under_limit(self):
        store = self._make_store()
        store.put("reply_hour_start:inst-1", time.time())
        store.put("reply_count_hour:inst-1", 5)
        assert Checker.check_reply_rate(store, "inst-1", {"max_replies_per_hour": 10}) is True

    def test_hourly_cap_blocks_at_limit(self):
        store = self._make_store()
        store.put("reply_hour_start:inst-1", time.time())
        store.put("reply_count_hour:inst-1", 10)
        assert Checker.check_reply_rate(store, "inst-1", {"max_replies_per_hour": 10}) is False

    def test_hourly_cap_resets_after_hour(self):
        store = self._make_store()
        store.put("reply_hour_start:inst-1", time.time() - 4000)  # > 1 hour ago
        store.put("reply_count_hour:inst-1", 100)
        assert Checker.check_reply_rate(store, "inst-1", {"max_replies_per_hour": 10}) is True

    def test_record_reply_sent(self):
        store = self._make_store()
        Checker.record_reply_sent(store, "inst-1")
        assert store.get("last_reply_time:inst-1") is not None
        assert store.get("reply_count_hour:inst-1") == 1

    def test_record_increments_count(self):
        store = self._make_store()
        Checker.record_reply_sent(store, "inst-1")
        Checker.record_reply_sent(store, "inst-1")
        assert store.get("reply_count_hour:inst-1") == 2

    def test_both_limits_combined(self):
        store = self._make_store()
        # Within cooldown
        store.put("last_reply_time:inst-1", time.time() - 30)
        config = {"reply_cooldown_seconds": 120, "max_replies_per_hour": 100}
        assert Checker.check_reply_rate(store, "inst-1", config) is False

        # Past cooldown but at hourly limit
        store.put("last_reply_time:inst-1", time.time() - 200)
        store.put("reply_hour_start:inst-1", time.time())
        store.put("reply_count_hour:inst-1", 100)
        assert Checker.check_reply_rate(store, "inst-1", config) is False
