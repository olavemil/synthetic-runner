"""Tests for Worker — job execution and provider concurrency."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, call, patch

import pytest

from library.harness.adapters import Event
from library.harness.config import HarnessConfig, ProviderConfig, AdapterConfig, InstanceConfig
from library.harness.jobqueue import JobQueue
from library.harness.registry import Registry
from library.harness.store import open_store, NamespacedStore
from library.harness.worker import Worker
from library.species import Species, SpeciesManifest, EntryPoint


def _make_instance(instance_id, provider="default"):
    return InstanceConfig(
        instance_id=instance_id,
        species="test",
        provider=provider,
        model="m",
    )


def _make_species(handler=None):
    h = handler or MagicMock()
    manifest = SpeciesManifest(
        species_id="test",
        entry_points=[
            EntryPoint(name="on_message", handler=h, trigger="message", schedule=None),
            EntryPoint(name="heartbeat", handler=h, trigger=None, schedule="0 * * * *"),
        ],
        tools=[],
        default_files={},
        spawn=lambda ctx: None,
    )
    s = MagicMock(spec=Species)
    s.manifest.return_value = manifest
    return s


def _build_worker(db, instances, max_concurrency=None, handler=None):
    harness_config = HarnessConfig(
        providers=[
            ProviderConfig(
                id="default",
                type="openai_compat",
                max_concurrency=max_concurrency,
            )
        ],
        adapters=[],
    )
    registry = Registry()
    species = _make_species(handler=handler)
    registry.register_species(species)
    for inst in instances:
        registry.register_instance(inst)

    mock_provider = MagicMock()
    mock_provider.create.return_value = MagicMock(
        message="hi", tool_calls=[], finish_reason="stop", usage={}
    )

    worker = Worker(
        harness_config=harness_config,
        registry=registry,
        providers={"default": mock_provider},
        adapters={},
        store_db=db,
        base_dir="/tmp",
    )
    return worker


class TestWorkerRun:
    def test_logs_enqueued_instances_and_completion(self, caplog):
        db = open_store()
        instances = [_make_instance("a"), _make_instance("b")]
        worker = _build_worker(db, instances)
        queue = JobQueue(db)

        queue.enqueue("a", "on_message")
        queue.enqueue("b", "on_message")

        with patch.object(worker, "_build_context") as mock_ctx:
            mock_ctx.return_value = MagicMock()
            with caplog.at_level(logging.INFO):
                worker.run()

        messages = [r.getMessage() for r in caplog.records]
        assert any("Worker cycle started (enqueued_instances=a,b)" in m for m in messages)
        assert any("Completed a " in m and "success=True" in m for m in messages)
        assert any("Completed b " in m and "success=True" in m for m in messages)

    def test_drains_queue(self):
        db = open_store()
        instances = [_make_instance("a"), _make_instance("b")]
        worker = _build_worker(db, instances)
        queue = JobQueue(db)

        queue.enqueue("a", "on_message")
        queue.enqueue("b", "on_message")

        with patch.object(worker, "_run_job", wraps=worker._run_job) as mock_run:
            with patch.object(worker, "_build_context") as mock_ctx:
                mock_ctx.return_value = MagicMock()
                worker.run()
                assert mock_run.call_count == 2

        assert queue.pending_count() == 0
        assert queue.running_count() == 0

    def test_empty_queue_is_noop(self):
        db = open_store()
        worker = _build_worker(db, [])
        # Should not raise
        worker.run()

    def test_completes_job_on_handler_error(self):
        db = open_store()
        instances = [_make_instance("a")]
        worker = _build_worker(db, instances)
        queue = JobQueue(db)
        queue.enqueue("a", "on_message")

        with patch.object(worker, "_build_context", side_effect=RuntimeError("boom")):
            worker.run()  # should not raise

        assert not queue.has_active("a")

    def test_rehydrates_events_from_inbox(self):
        db = open_store()
        instances = [_make_instance("a")]
        captured = {}

        def handler(ctx, events=None):  # noqa: ARG001
            captured["events"] = events

        worker = _build_worker(db, instances, handler=handler)
        queue = JobQueue(db)

        # Store events in the persistent inbox (as checker would)
        worker._checker_store.put("pending_events:a", [
            {
                "event_id": "$evt",
                "sender": "@u:matrix.org",
                "body": "hello",
                "timestamp": 123,
                "room": "main",
            }
        ])
        queue.enqueue("a", "on_message")

        with patch.object(worker, "_build_context") as mock_ctx:
            mock_ctx.return_value = MagicMock()
            worker.run()

        events = captured.get("events")
        assert isinstance(events, list)
        assert len(events) == 1
        assert isinstance(events[0], Event)
        assert events[0].event_id == "$evt"
        assert events[0].room == "main"

        # Inbox should be drained
        assert worker._checker_store.get("pending_events:a") is None

    def test_on_message_defaults_missing_events_to_empty_list(self):
        db = open_store()
        instances = [_make_instance("a")]
        captured = {}

        def handler(ctx, events=None):  # noqa: ARG001
            captured["events"] = events

        worker = _build_worker(db, instances, handler=handler)
        queue = JobQueue(db)
        queue.enqueue("a", "on_message")

        with patch.object(worker, "_build_context") as mock_ctx:
            mock_ctx.return_value = MagicMock()
            worker.run()

        assert captured.get("events") == []

    def test_logs_failure_reason_summary_on_handler_error(self, caplog):
        db = open_store()
        instances = [_make_instance("a")]

        def handler(ctx, **kwargs):  # noqa: ARG001
            raise ValueError("boom")

        worker = _build_worker(db, instances, handler=handler)
        queue = JobQueue(db)
        queue.enqueue("a", "heartbeat")

        with patch.object(worker, "_build_context") as mock_ctx:
            mock_ctx.return_value = MagicMock()
            with caplog.at_level(logging.INFO):
                worker.run()

        messages = [r.getMessage() for r in caplog.records]
        assert any("Failed a " in m and "ValueError: boom" in m for m in messages)
        assert any("Completed a " in m and "success=False" in m for m in messages)

    def test_on_message_send_policy_allows_one_send_with_events(self):
        db = open_store()
        instances = [_make_instance("a")]
        captured = {}

        class FakeCtx:
            def __init__(self):
                self._sent = 0

            def configure_send_policy(self, **kwargs):
                captured["policy"] = kwargs

            @property
            def sent_message_count(self):
                return self._sent

        def handler(ctx, events=None):  # noqa: ARG001
            captured["events"] = events

        worker = _build_worker(db, instances, handler=handler)
        # Store events in the persistent inbox
        worker._checker_store.put("pending_events:a", [
            {
                "event_id": "$evt",
                "sender": "@u:matrix.org",
                "body": "hello",
                "timestamp": 123,
                "room": "main",
            }
        ])
        queue = JobQueue(db)
        queue.enqueue("a", "on_message")

        with patch.object(worker, "_build_context", return_value=FakeCtx()):
            worker.run()

        assert isinstance(captured.get("events"), list)
        assert captured["policy"]["allow_send"] is True
        assert captured["policy"]["max_sends"] == 1

    def test_on_message_send_policy_blocks_without_events(self):
        db = open_store()
        instances = [_make_instance("a")]
        captured = {}

        class FakeCtx:
            def __init__(self):
                self._sent = 0

            def configure_send_policy(self, **kwargs):
                captured["policy"] = kwargs

            @property
            def sent_message_count(self):
                return self._sent

        def handler(ctx, events=None):  # noqa: ARG001
            captured["events"] = events

        worker = _build_worker(db, instances, handler=handler)
        queue = JobQueue(db)
        queue.enqueue("a", "on_message")

        with patch.object(worker, "_build_context", return_value=FakeCtx()):
            worker.run()

        assert captured["events"] == []
        assert captured["policy"]["allow_send"] is False
        assert captured["policy"]["max_sends"] == 0

    def test_heartbeat_send_policy_requires_new_external_messages(self):
        db = open_store()
        instances = [_make_instance("a")]
        captured = {}

        class FakeCtx:
            def __init__(self):
                self._sent = 0

            def configure_send_policy(self, **kwargs):
                captured["policy"] = kwargs

            @property
            def sent_message_count(self):
                return self._sent

        def handler(ctx):  # noqa: ARG001
            captured["handled"] = True

        worker = _build_worker(db, instances, handler=handler)
        worker._checker_store.put("external_count:a", 3)
        worker._checker_store.put("send_gate_seen:a:heartbeat", 3)
        queue = JobQueue(db)
        queue.enqueue("a", "heartbeat")

        with patch.object(worker, "_build_context", return_value=FakeCtx()):
            worker.run()

        assert captured.get("handled") is True
        assert captured["policy"]["allow_send"] is False
        assert captured["policy"]["max_sends"] == 0

    def test_heartbeat_send_gate_consumed_when_message_sent(self):
        db = open_store()
        instances = [_make_instance("a")]
        captured = {}

        class FakeCtx:
            def __init__(self):
                self._sent = 0

            def configure_send_policy(self, **kwargs):
                captured["policy"] = kwargs

            @property
            def sent_message_count(self):
                return self._sent

        def handler(ctx):  # noqa: ARG001
            ctx._sent = 1

        worker = _build_worker(db, instances, handler=handler)
        worker._checker_store.put("external_count:a", 5)
        queue = JobQueue(db)
        queue.enqueue("a", "heartbeat")

        with patch.object(worker, "_build_context", return_value=FakeCtx()):
            worker.run()

        assert captured["policy"]["allow_send"] is True
        assert captured["policy"]["max_sends"] == 1
        assert worker._checker_store.get("send_gate_seen:a:heartbeat") == 5


    def test_combined_job_runs_on_message_then_heartbeat(self):
        db = open_store()
        instances = [_make_instance("a")]
        call_order = []

        def on_message_handler(ctx, events=None):  # noqa: ARG001
            call_order.append("on_message")

        def heartbeat_handler(ctx):  # noqa: ARG001
            call_order.append("heartbeat")

        # Create species with separate handlers for on_message and heartbeat
        manifest = SpeciesManifest(
            species_id="test",
            entry_points=[
                EntryPoint(name="on_message", handler=on_message_handler, trigger="message", schedule=None),
                EntryPoint(name="heartbeat", handler=heartbeat_handler, trigger=None, schedule="0 * * * *"),
            ],
            tools=[],
            default_files={},
            spawn=lambda ctx: None,
        )
        s = MagicMock(spec=Species)
        s.manifest.return_value = manifest

        harness_config = HarnessConfig(
            providers=[ProviderConfig(id="default", type="openai_compat")],
            adapters=[],
        )
        registry = Registry()
        registry.register_species(s)
        for inst in instances:
            registry.register_instance(inst)
        mock_provider = MagicMock()
        worker = Worker(
            harness_config=harness_config,
            registry=registry,
            providers={"default": mock_provider},
            adapters={},
            store_db=db,
            base_dir="/tmp",
        )

        queue = JobQueue(db)
        # Store events in the persistent inbox
        worker._checker_store.put("pending_events:a", [
            {"event_id": "$1", "sender": "@u:m", "body": "hi", "timestamp": 1, "room": "main"}
        ])
        queue.enqueue("a", "on_message")
        queue.merge_into_pending("a", "heartbeat", {"heartbeat": True})

        with patch.object(worker, "_build_context") as mock_ctx:
            mock_ctx.return_value = MagicMock()
            worker.run()

        assert call_order == ["on_message", "heartbeat"]


class TestProviderSlots:
    def test_can_run_respects_max_concurrency(self):
        db = open_store()
        instances = [_make_instance("a"), _make_instance("b")]
        worker = _build_worker(db, instances, max_concurrency=1)

        # Claim the single slot manually
        worker._slots_store.claim("default:slot:0", "other-worker")

        assert not worker._can_run("a")
        assert not worker._can_run("b")

    def test_can_run_when_slot_free(self):
        db = open_store()
        instances = [_make_instance("a")]
        worker = _build_worker(db, instances, max_concurrency=2)

        assert worker._can_run("a")

    def test_no_limit_always_can_run(self):
        db = open_store()
        instances = [_make_instance("a")]
        worker = _build_worker(db, instances, max_concurrency=None)

        # Artificially fill all slots (none configured)
        assert worker._can_run("a")

    def test_slot_released_after_job(self):
        db = open_store()
        instances = [_make_instance("a")]
        worker = _build_worker(db, instances, max_concurrency=1)
        queue = JobQueue(db)
        queue.enqueue("a", "on_message")

        with patch.object(worker, "_build_context") as mock_ctx:
            mock_ctx.return_value = MagicMock()
            worker.run()

        # Slot should be released
        slots_items = worker._slots_store.scan_items("default:slot:")
        running_slots = [k for k, v, owner in slots_items if owner is not None]
        assert len(running_slots) == 0
