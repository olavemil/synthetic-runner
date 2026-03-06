"""Checker — lightweight poller that enqueues jobs without running pipelines."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from croniter import croniter

from symbiosis.harness.config import HarnessConfig, InstanceConfig, load_harness_config, load_instance_config
from symbiosis.harness.jobqueue import JobQueue
from symbiosis.harness.registry import Registry
from symbiosis.harness.store import StoreDB, NamespacedStore, open_store

logger = logging.getLogger(__name__)

_INSTANCE_TEMPLATE_STEMS = {"example", "sample", "template"}


def _is_template_instance_file(path: Path) -> bool:
    stem = path.stem.lower()
    return (
        stem in _INSTANCE_TEMPLATE_STEMS
        or stem.endswith(".example")
        or stem.endswith(".sample")
        or stem.endswith(".template")
    )


class Checker:
    """
    Polls messaging adapters and checks cron schedules.

    State persisted in SQLite (checker namespace):
      sync_tokens:    {instance_id}:{space_name} -> token
      schedule_next:  {instance_id}:{entry_point} -> next_fire_timestamp
      idle_counts:    {instance_id} -> int (heartbeats since last message)
    """

    _NS = "checker"

    def __init__(
        self,
        harness_config: HarnessConfig,
        registry: Registry,
        store_db: StoreDB,
        base_dir: Path,
    ):
        self._config = harness_config
        self._registry = registry
        self._store = NamespacedStore(store_db, self._NS)
        self._base_dir = base_dir
        self._queue = JobQueue(store_db)
        self._instance_adapters: dict[str, object] = {}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute one check cycle: poll adapters + check schedules."""
        reactive_instances: set[str] = set()

        for instance_config in self._registry.list_instances():
            if self._poll_instance(instance_config):
                reactive_instances.add(instance_config.instance_id)

        self._check_schedules(reactive_instances)

    # ------------------------------------------------------------------
    # Reactive polling
    # ------------------------------------------------------------------

    def _poll_instance(self, instance_config: InstanceConfig) -> bool:
        """Poll all spaces for an instance. Returns True if new events found."""
        if not instance_config.messaging:
            return False

        instance_id = instance_config.instance_id
        adapter = self._get_adapter(instance_config)
        if adapter is None:
            return False

        got_messages = False

        for space_mapping in instance_config.messaging.spaces:
            space_name = space_mapping.name
            handle = space_mapping.handle
            token_key = f"{instance_id}:{space_name}"
            token = self._store.get(token_key)

            try:
                events, next_token = adapter.poll(handle, token)
                self._store.put(token_key, next_token)

                if events and token is not None:
                    # Filter out own events
                    entity_id = instance_config.messaging.entity_id
                    own_events = [e for e in events if e.sender != entity_id]
                    if own_events:
                        got_messages = True
            except Exception:
                logger.exception("Error polling %s/%s", instance_id, space_name)

        if got_messages and not self._queue.has_active(instance_id):
            job_id = self._queue.enqueue(instance_id, "on_message", {})
            if job_id:
                logger.info("Enqueued on_message for %s (job %s)", instance_id, job_id)
            # Reset idle count on new messages
            self._store.put(f"idle:{instance_id}", 0)

        return got_messages

    # ------------------------------------------------------------------
    # Schedule checking
    # ------------------------------------------------------------------

    def _check_schedules(self, reactive_instances: set[str]) -> None:
        now = time.time()

        for instance_config in self._registry.list_instances():
            instance_id = instance_config.instance_id
            manifest = self._registry.get_manifest(instance_config.species)

            # Collect all scheduled entry points from manifest + instance overrides
            schedule_map: dict[str, str] = {}
            for ep in manifest.entry_points:
                if ep.schedule:
                    schedule_map[ep.name] = ep.schedule

            for ep_name, value in instance_config.schedule.items():
                if isinstance(value, str):
                    schedule_map[ep_name] = value

            max_idle = instance_config.schedule.get("max_idle_heartbeats")
            if max_idle is not None:
                try:
                    max_idle = int(max_idle)
                except (ValueError, TypeError):
                    max_idle = None

            for ep_name, cron_expr in schedule_map.items():
                key = f"schedule_next:{instance_id}:{ep_name}"
                next_fire = self._store.get(key)

                if next_fire is None:
                    cron = croniter(cron_expr, now)
                    self._store.put(key, cron.get_next(float))
                    continue

                if now < next_fire:
                    continue

                # Advance to next fire time
                cron = croniter(cron_expr, now)
                self._store.put(key, cron.get_next(float))

                # Idle throttle for heartbeat-type entry points
                if max_idle is not None and instance_id not in reactive_instances:
                    idle_key = f"idle:{instance_id}"
                    idle_count = self._store.get(idle_key) or 0
                    if idle_count >= max_idle:
                        logger.debug(
                            "Skipping %s.%s (idle=%d >= max=%d)",
                            instance_id, ep_name, idle_count, max_idle,
                        )
                        continue
                    self._store.put(idle_key, idle_count + 1)

                # Enqueue if not already active
                if not self._queue.has_active(instance_id):
                    job_id = self._queue.enqueue(instance_id, ep_name, {})
                    if job_id:
                        logger.info(
                            "Enqueued %s.%s (job %s)", instance_id, ep_name, job_id
                        )
                else:
                    logger.debug(
                        "Skipping %s.%s — instance already has active job",
                        instance_id, ep_name,
                    )

    # ------------------------------------------------------------------
    # Adapter management
    # ------------------------------------------------------------------

    def _get_adapter(self, instance_config: InstanceConfig):
        instance_id = instance_config.instance_id
        if instance_id in self._instance_adapters:
            return self._instance_adapters[instance_id]

        if not instance_config.messaging:
            return None

        adapter_id = instance_config.messaging.adapter
        try:
            adapter_config = self._config.get_adapter(adapter_id)
        except KeyError:
            logger.warning("Adapter '%s' not in harness config", adapter_id)
            return None

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


def build_checker(base_dir: Path, config_path: Path, registry: Registry, store_db: StoreDB) -> Checker:
    harness_config = load_harness_config(config_path)
    return Checker(
        harness_config=harness_config,
        registry=registry,
        store_db=store_db,
        base_dir=base_dir,
    )
