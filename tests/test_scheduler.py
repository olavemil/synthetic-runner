"""Tests for scheduler — scheduling, locking, dispatch."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from symbiosis.harness.config import (
    HarnessConfig,
    ProviderConfig,
    AdapterConfig,
    InstanceConfig,
    MessagingConfig,
    SpaceMapping,
)
from symbiosis.harness.registry import Registry
from symbiosis.harness.scheduler import Scheduler, build_providers, build_adapters
from symbiosis.species import Species, SpeciesManifest, EntryPoint


class MockSpecies(Species):
    def __init__(self):
        self.on_message_calls = []
        self.heartbeat_calls = []

    def manifest(self) -> SpeciesManifest:
        return SpeciesManifest(
            species_id="mock",
            entry_points=[
                EntryPoint(name="on_message", handler=self._on_message, trigger="message"),
                EntryPoint(name="heartbeat", handler=self._heartbeat, schedule="0 * * * *"),
            ],
        )

    def _on_message(self, ctx, events=None):
        self.on_message_calls.append(events)

    def _heartbeat(self, ctx):
        self.heartbeat_calls.append(True)


class TestRegistry:
    def test_register_and_lookup(self):
        registry = Registry()
        species = MockSpecies()
        registry.register_species(species)

        config = InstanceConfig(
            instance_id="test-1", species="mock", provider="p", model="m"
        )
        registry.register_instance(config)

        handler = registry.get_handler("test-1", "on_message")
        assert handler is not None

    def test_missing_species(self):
        registry = Registry()
        with pytest.raises(KeyError, match="mock"):
            registry.get_manifest("mock")

    def test_missing_instance(self):
        registry = Registry()
        with pytest.raises(KeyError, match="test-1"):
            registry.get_instance_config("test-1")

    def test_missing_entry_point(self):
        registry = Registry()
        species = MockSpecies()
        registry.register_species(species)
        config = InstanceConfig(
            instance_id="test-1", species="mock", provider="p", model="m"
        )
        registry.register_instance(config)
        with pytest.raises(KeyError, match="nonexistent"):
            registry.get_handler("test-1", "nonexistent")


class TestScheduler:
    def _make_scheduler(self, tmp_path):
        harness_config = HarnessConfig(
            providers=[ProviderConfig(id="mock-provider", type="openai_compat", base_url="http://localhost", api_key="key")],
            adapters=[],
            storage_dir="instances",
            store_path="test.db",
            poll_interval=1,
        )

        registry = Registry()
        species = MockSpecies()
        registry.register_species(species)
        config = InstanceConfig(
            instance_id="test-1", species="mock", provider="mock-provider", model="m"
        )
        registry.register_instance(config)

        mock_provider = MagicMock()
        providers = {"mock-provider": mock_provider}

        scheduler = Scheduler(
            harness_config=harness_config,
            registry=registry,
            providers=providers,
            adapters={},
            base_dir=tmp_path,
        )
        return scheduler, species

    def test_build_context(self, tmp_path):
        scheduler, _ = self._make_scheduler(tmp_path)
        config = scheduler._registry.get_instance_config("test-1")
        ctx = scheduler._build_context(config)
        assert ctx.instance_id == "test-1"
        assert ctx.species_id == "mock"

    def test_dispatch(self, tmp_path):
        scheduler, species = self._make_scheduler(tmp_path)
        scheduler._dispatch("test-1", "heartbeat")
        assert len(species.heartbeat_calls) == 1

    def test_dispatch_with_events(self, tmp_path):
        scheduler, species = self._make_scheduler(tmp_path)
        from symbiosis.harness.adapters import Event
        events = [Event(event_id="1", sender="alice", body="hi", timestamp=1000)]
        scheduler._dispatch("test-1", "on_message", events=events)
        assert len(species.on_message_calls) == 1
        assert species.on_message_calls[0] == events

    def test_instance_locking(self, tmp_path):
        scheduler, species = self._make_scheduler(tmp_path)

        # Simulate slow handler
        def slow_handler(ctx, **kwargs):
            time.sleep(0.2)
            species.heartbeat_calls.append(True)

        manifest = scheduler._registry.get_manifest("mock")
        for ep in manifest.entry_points:
            if ep.name == "heartbeat":
                ep.handler = slow_handler

        # Dispatch two concurrent calls
        t1 = threading.Thread(target=scheduler._dispatch, args=("test-1", "heartbeat"))
        t2 = threading.Thread(target=scheduler._dispatch, args=("test-1", "heartbeat"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Both should complete (serially)
        assert len(species.heartbeat_calls) == 2

    def test_stop(self, tmp_path):
        scheduler, _ = self._make_scheduler(tmp_path)
        scheduler.stop()
        assert scheduler._running is False

    def test_check_schedules_ignores_non_cron_and_unknown_schedule_keys(self, tmp_path):
        scheduler, _ = self._make_scheduler(tmp_path)
        config = scheduler._registry.get_instance_config("test-1")
        config.schedule = {
            "heartbeat": "*/5 * * * *",
            "max_idle_heartbeats": 3,   # metadata, not cron
            "unknown_entrypoint": "*/5 * * * *",  # not in manifest
        }

        scheduler._check_schedules()

        assert "test-1:heartbeat" in scheduler._schedule_state
        assert "test-1:max_idle_heartbeats" not in scheduler._schedule_state
        assert "test-1:unknown_entrypoint" not in scheduler._schedule_state

    def test_poll_reactive_normalizes_room_to_logical_space(self, tmp_path):
        from symbiosis.harness.adapters import Event

        harness_config = HarnessConfig(
            providers=[ProviderConfig(id="mock-provider", type="openai_compat", base_url="http://localhost", api_key="key")],
            adapters=[AdapterConfig(id="matrix-main", type="matrix", homeserver="https://matrix.org")],
            storage_dir="instances",
            store_path="test.db",
            poll_interval=1,
        )
        registry = Registry()
        species = MockSpecies()
        registry.register_species(species)
        registry.register_instance(
            InstanceConfig(
                instance_id="test-1",
                species="mock",
                provider="mock-provider",
                model="m",
                messaging=MessagingConfig(
                    adapter="matrix-main",
                    entity_id="@bot:matrix.org",
                    access_token="token",
                    spaces=[SpaceMapping(name="main", handle="!room:matrix.org")],
                ),
            )
        )

        scheduler = Scheduler(
            harness_config=harness_config,
            registry=registry,
            providers={"mock-provider": MagicMock()},
            adapters={},
            base_dir=tmp_path,
        )

        adapter = MagicMock()
        adapter.poll.return_value = (
            [
                Event(event_id="1", sender="@user:matrix.org", body="hello", timestamp=1, room="!room:matrix.org"),
                Event(event_id="2", sender="@bot:matrix.org", body="self", timestamp=2, room="!room:matrix.org"),
            ],
            "next-token",
        )
        scheduler._build_adapter = MagicMock(return_value=adapter)  # type: ignore[method-assign]

        captured: list[list[Event]] = []

        def fake_dispatch(instance_id, entry_point_name, **kwargs):  # noqa: ARG001
            captured.append(kwargs["events"])

        scheduler._dispatch = fake_dispatch  # type: ignore[method-assign]
        scheduler._sync_tokens["test-1"] = {"main": "start-token"}

        scheduler._poll_reactive()

        assert len(captured) == 1
        assert len(captured[0]) == 1
        assert captured[0][0].sender == "@user:matrix.org"
        assert captured[0][0].room == "main"


class TestBuildProviders:
    def test_unknown_type_skipped(self):
        config = HarnessConfig(
            providers=[ProviderConfig(id="x", type="unknown")]
        )
        providers = build_providers(config)
        assert "x" not in providers

    def test_empty_config(self):
        config = HarnessConfig()
        providers = build_providers(config)
        assert providers == {}


class TestBuildAdapters:
    def test_unknown_type_skipped(self):
        config = HarnessConfig(
            adapters=[AdapterConfig(id="x", type="unknown")]
        )
        adapters = build_adapters(config)
        assert "x" not in adapters

    def test_empty_config(self):
        config = HarnessConfig()
        adapters = build_adapters(config)
        assert adapters == {}
