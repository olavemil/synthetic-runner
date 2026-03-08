"""Tests for Checker — polling and schedule checking."""

from __future__ import annotations

import logging
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest
import yaml

from symbiosis.harness.checker import Checker
from symbiosis.harness.adapters import Event
from symbiosis.harness.config import (
    HarnessConfig,
    AdapterConfig,
    InstanceConfig,
    MessagingConfig,
    SpaceMapping,
)
from symbiosis.harness.jobqueue import JobQueue
from symbiosis.harness.registry import Registry
from symbiosis.harness.store import open_store
from symbiosis.species import Species, SpeciesManifest, EntryPoint


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


    def test_max_thinks_per_reply_throttles_heartbeat(self):
        db = open_store()
        species = _make_species(schedule="* * * * *")
        instance = _make_instance(schedule={"max_thinks_per_reply": 1})
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

        _trigger()  # thinks=1, enqueues (0 < 1)
        count_after_first = queue.pending_count()
        _trigger()  # thinks=1 >= 1, skips
        assert queue.pending_count() == count_after_first  # nothing added

    def test_max_thinks_per_reply_resets_on_message(self):
        db = open_store()
        species = _make_species(schedule="* * * * *")
        instance = _make_instance(schedule={"max_thinks_per_reply": 1}, with_messaging=True)
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

        # Saturate thinks_since_reply
        checker._store.put("thinks_since_reply:inst-1", 5)

        # Simulate a message arriving (poll_instance resets counter)
        checker._store.put("inst-1:main", "tok0")  # pre-seeded token so poll produces events
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

        # Counter should be reset to 0
        assert checker._store.get("thinks_since_reply:inst-1") == 0

        # Now heartbeat should be schedulable again
        checker._store.put("schedule_next:inst-1:heartbeat", time.time() - 1)
        # Drain on_message job first
        if queue.has_active("inst-1"):
            job = queue.claim_next("w")
            queue.complete(job, "w")
        checker._check_schedules(set())
        assert queue.pending_count() == 1


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

    def test_poll_instance_enqueues_on_message_with_serialized_events(self):
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
        # Second poll should enqueue on_message with event payload.
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
        payload_events = pending[0].payload.get("events", [])
        assert len(payload_events) == 1
        assert payload_events[0]["event_id"] == "$evt1"
        # Event room should be normalized to logical space name by checker.
        assert payload_events[0]["room"] == "main"
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
        payload_events = pending[0].payload.get("events", [])
        assert len(payload_events) == 1
        assert payload_events[0]["event_id"] == "$ext"
        assert checker._store.get("external_count:inst-1") == 1
        assert checker._resolved_entity_ids["inst-1"] == "@bot:matrix.org"

    def test_poll_instance_skips_reactive_events_when_entity_id_unknown(self, caplog):
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
            sender="@someone:matrix.org",
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

        assert got_messages_1 is False
        assert enqueued_1 is False
        assert got_messages_2 is False
        assert enqueued_2 is False
        assert queue.pending_count() == 0
        messages = [r.getMessage() for r in caplog.records]
        assert any("No entity_id configured for inst-1" in m for m in messages)
        assert any("Skipping reactive events for inst-1/main" in m for m in messages)

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
