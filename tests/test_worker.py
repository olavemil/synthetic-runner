"""Tests for Worker — job execution and provider concurrency."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from symbiosis.harness.config import HarnessConfig, ProviderConfig, AdapterConfig, InstanceConfig
from symbiosis.harness.jobqueue import JobQueue
from symbiosis.harness.registry import Registry
from symbiosis.harness.store import open_store, NamespacedStore
from symbiosis.harness.worker import Worker
from symbiosis.species import Species, SpeciesManifest, EntryPoint


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


def _build_worker(db, instances, max_concurrency=None):
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
    species = _make_species()
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
