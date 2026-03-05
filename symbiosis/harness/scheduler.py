"""Unified scheduler — manages all instances and entry points from a single process."""

from __future__ import annotations

import logging
import signal
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from croniter import croniter

from symbiosis.harness.config import (
    HarnessConfig,
    InstanceConfig,
    load_harness_config,
    load_instance_config,
)
from symbiosis.harness.context import InstanceContext
from symbiosis.harness.mailbox import Mailbox
from symbiosis.harness.registry import Registry
from symbiosis.harness.storage import NamespacedStorage
from symbiosis.harness.store import open_store, StoreDB

if TYPE_CHECKING:
    from symbiosis.harness.adapters import MessagingAdapter
    from symbiosis.harness.providers import LLMProvider

logger = logging.getLogger(__name__)


class Scheduler:
    def __init__(
        self,
        harness_config: HarnessConfig,
        registry: Registry,
        providers: dict[str, LLMProvider],
        adapters: dict[str, MessagingAdapter],
        base_dir: str | Path = ".",
    ):
        self._config = harness_config
        self._registry = registry
        self._providers = providers
        self._adapters = adapters
        self._base_dir = Path(base_dir)
        self._store_db = open_store(self._base_dir / harness_config.store_path)
        self._instance_locks: dict[str, threading.Lock] = {}
        self._instance_adapters: dict[str, MessagingAdapter] = {}
        self._sync_tokens: dict[str, dict[str, str]] = {}  # instance_id -> {space: token}
        self._running = False
        self._schedule_state: dict[str, float] = {}  # entry key -> next fire time

    def _get_lock(self, instance_id: str) -> threading.Lock:
        if instance_id not in self._instance_locks:
            self._instance_locks[instance_id] = threading.Lock()
        return self._instance_locks[instance_id]

    def _build_adapter(self, instance_config: InstanceConfig) -> MessagingAdapter | None:
        """Build or retrieve a per-instance messaging adapter.

        Combines shared adapter config (homeserver) with per-instance
        credentials (access_token from instance messaging config).
        """
        instance_id = instance_config.instance_id
        if instance_id in self._instance_adapters:
            return self._instance_adapters[instance_id]

        if not instance_config.messaging:
            return None

        adapter_id = instance_config.messaging.adapter
        adapter_config = self._config.get_adapter(adapter_id)

        if adapter_config.type == "matrix":
            from symbiosis.harness.adapters.matrix import MatrixAdapter
            token = instance_config.messaging.access_token or ""
            adapter = MatrixAdapter(
                homeserver=adapter_config.homeserver or "",
                access_token=token,
            )
        elif adapter_config.type == "local_file":
            from symbiosis.harness.adapters.local_file import LocalFileAdapter
            adapter = LocalFileAdapter(base_dir=adapter_config.base_dir or "messages")
        else:
            logger.warning("Unknown adapter type: %s", adapter_config.type)
            return None

        self._instance_adapters[instance_id] = adapter
        return adapter

    def _build_context(self, instance_config: InstanceConfig) -> InstanceContext:
        """Construct an InstanceContext for an instance."""
        storage = NamespacedStorage(
            self._base_dir / self._config.storage_dir,
            instance_config.instance_id,
        )

        provider = self._providers.get(instance_config.provider)
        if provider is None:
            raise KeyError(f"Provider '{instance_config.provider}' not available")

        adapter = None
        space_map: dict[str, str] = {}
        if instance_config.messaging:
            adapter = self._build_adapter(instance_config)
            for sp in instance_config.messaging.spaces:
                space_map[sp.name] = sp.handle

        mailbox = Mailbox(
            self._base_dir / self._config.storage_dir,
            instance_config.instance_id,
        )

        return InstanceContext(
            instance_id=instance_config.instance_id,
            species_id=instance_config.species,
            storage=storage,
            provider=provider,
            default_model=instance_config.model,
            adapter=adapter,
            space_map=space_map,
            store_db=self._store_db,
            mailbox=mailbox,
            instance_config=instance_config,
        )

    def _dispatch(self, instance_id: str, entry_point_name: str, **kwargs) -> None:
        """Dispatch an entry point for an instance, with per-instance locking."""
        lock = self._get_lock(instance_id)
        if not lock.acquire(blocking=False):
            logger.info(
                "Instance %s is busy, queuing %s", instance_id, entry_point_name
            )
            lock.acquire()

        try:
            config = self._registry.get_instance_config(instance_id)
            ctx = self._build_context(config)
            handler = self._registry.get_handler(instance_id, entry_point_name)
            logger.info("Dispatching %s.%s", instance_id, entry_point_name)
            handler(ctx, **kwargs)
        except Exception:
            logger.exception(
                "Error in %s.%s", instance_id, entry_point_name
            )
        finally:
            lock.release()

    def _poll_reactive(self) -> None:
        """Poll all messaging adapters for new events and dispatch reactive handlers."""
        for instance_config in self._registry.list_instances():
            if not instance_config.messaging:
                continue

            instance_id = instance_config.instance_id
            adapter = self._build_adapter(instance_config)
            if adapter is None:
                continue

            if instance_id not in self._sync_tokens:
                self._sync_tokens[instance_id] = {}

            for space_mapping in instance_config.messaging.spaces:
                space_name = space_mapping.name
                handle = space_mapping.handle
                token = self._sync_tokens[instance_id].get(space_name)

                try:
                    events, next_token = adapter.poll(handle, token)
                    # Normalize adapter-level room handles to logical space names
                    # before dispatching to species handlers.
                    normalized_events = [
                        type(evt)(
                            event_id=evt.event_id,
                            sender=evt.sender,
                            body=evt.body,
                            timestamp=evt.timestamp,
                            room=space_name,
                        )
                        for evt in events
                    ]
                    self._sync_tokens[instance_id][space_name] = next_token

                    if normalized_events and token is not None:
                        # Only dispatch if we had a previous token (skip initial sync)
                        entity_id = instance_config.messaging.entity_id
                        own_events = [e for e in normalized_events if e.sender != entity_id]
                        if own_events:
                            self._dispatch(instance_id, "on_message", events=own_events)
                except Exception:
                    logger.exception(
                        "Error polling %s/%s", instance_id, space_name
                    )

    def _check_schedules(self) -> None:
        """Check cron schedules and dispatch due entry points."""
        now = time.time()

        for instance_config in self._registry.list_instances():
            instance_id = instance_config.instance_id
            manifest = self._registry.get_manifest(instance_config.species)

            for ep in manifest.entry_points:
                if not ep.schedule:
                    continue

                key = f"{instance_id}:{ep.name}"

                if key not in self._schedule_state:
                    cron = croniter(ep.schedule, now)
                    self._schedule_state[key] = cron.get_next(float)
                    continue

                next_fire = self._schedule_state[key]
                if now >= next_fire:
                    cron = croniter(ep.schedule, now)
                    self._schedule_state[key] = cron.get_next(float)
                    threading.Thread(
                        target=self._dispatch,
                        args=(instance_id, ep.name),
                        daemon=True,
                    ).start()

            # Also check instance-level schedule overrides
            for ep_name, cron_expr in instance_config.schedule.items():
                key = f"{instance_id}:{ep_name}"
                if key not in self._schedule_state:
                    cron = croniter(cron_expr, now)
                    self._schedule_state[key] = cron.get_next(float)
                    continue

                next_fire = self._schedule_state[key]
                if now >= next_fire:
                    cron = croniter(cron_expr, now)
                    self._schedule_state[key] = cron.get_next(float)
                    threading.Thread(
                        target=self._dispatch,
                        args=(instance_id, ep_name),
                        daemon=True,
                    ).start()

    def run_forever(self) -> None:
        """Main loop — poll for events and check schedules."""
        self._running = True

        def handle_signal(sig, frame):
            logger.info("Received signal %s, shutting down", sig)
            self._running = False

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        logger.info("Scheduler started with %d instances", len(self._registry.list_instances()))

        while self._running:
            try:
                self._poll_reactive()
                self._check_schedules()
            except Exception:
                logger.exception("Error in scheduler loop")

            time.sleep(self._config.poll_interval)

        logger.info("Scheduler stopped")
        self._store_db.close()

    def stop(self) -> None:
        self._running = False


def build_providers(harness_config: HarnessConfig) -> dict[str, LLMProvider]:
    """Build LLM provider instances from config."""
    providers: dict[str, LLMProvider] = {}

    for pc in harness_config.providers:
        if pc.type == "openai_compat":
            from symbiosis.harness.providers.openai_compat import OpenAICompatProvider
            providers[pc.id] = OpenAICompatProvider(
                base_url=pc.base_url,
                api_key=pc.api_key,
            )
        elif pc.type == "anthropic":
            from symbiosis.harness.providers.anthropic import AnthropicProvider
            providers[pc.id] = AnthropicProvider(api_key=pc.api_key)
        else:
            logger.warning("Unknown provider type: %s", pc.type)

    return providers


def build_adapters(harness_config: HarnessConfig) -> dict[str, MessagingAdapter]:
    """Build shared messaging adapter instances from config.

    Note: adapters that require per-instance credentials (like Matrix)
    are built on demand by the scheduler via _build_adapter(). This
    function only pre-builds stateless adapters (like local_file).
    """
    adapters: dict[str, MessagingAdapter] = {}

    for ac in harness_config.adapters:
        if ac.type == "local_file":
            from symbiosis.harness.adapters.local_file import LocalFileAdapter
            adapters[ac.id] = LocalFileAdapter(base_dir=ac.base_dir or "messages")
        elif ac.type == "matrix":
            pass  # built per-instance by scheduler (needs per-instance access_token)
        else:
            logger.warning("Unknown adapter type: %s", ac.type)

    return adapters
