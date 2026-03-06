"""Tests for Checker — polling and schedule checking."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
import yaml

from symbiosis.harness.checker import Checker
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


def _make_instance(instance_id="inst-1", species_id=_SPECIES_ID, schedule=None, with_messaging=False):
    messaging = None
    if with_messaging:
        messaging = MessagingConfig(
            adapter="test-adapter",
            entity_id="@bot:matrix.org",
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

    def test_idle_throttling(self):
        db = open_store()
        species = _make_species(schedule="* * * * *")
        instance = _make_instance(schedule={"max_idle_heartbeats": 2})
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
