"""Checker — lightweight poller that enqueues jobs without running pipelines."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from croniter import croniter
import httpx

from symbiosis.harness.adapters import Event
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
        self._resolved_entity_ids: dict[str, str | None] = {}
        self._warned_missing_entity_id: set[str] = set()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute one check cycle: poll adapters + check schedules."""
        instances = self._registry.list_instances()
        logger.info("Checker cycle started (instances=%d)", len(instances))
        # Enqueue due scheduled jobs first so heartbeat-like tasks are not
        # indefinitely starved by frequent on_message enqueues.
        scheduled_instances: set[str] = self._check_schedules(set())
        reactive_instances: set[str] = set()

        for instance_config in instances:
            got_messages, enqueued = self._poll_instance(instance_config)
            if got_messages:
                reactive_instances.add(instance_config.instance_id)
            if enqueued:
                scheduled_instances.add(instance_config.instance_id)

        # Run schedule pass again after polling to keep idle counters aligned
        # with this cycle's reactive activity and to catch any entry points that
        # became due during long polling loops.
        scheduled_instances.update(self._check_schedules(reactive_instances))
        logger.info("Checker cycle finished (scheduled_instances=%d)", len(scheduled_instances))

    # ------------------------------------------------------------------
    # Reactive polling
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_events(events: list[Event]) -> list[dict]:
        return [
            {
                "event_id": e.event_id,
                "sender": e.sender,
                "body": e.body,
                "timestamp": e.timestamp,
                "room": e.room,
            }
            for e in events
        ]

    def _poll_instance(self, instance_config: InstanceConfig) -> tuple[bool, bool]:
        """Poll all spaces for an instance. Returns (got_messages, enqueued_job)."""
        if not instance_config.messaging:
            return False, False

        instance_id = instance_config.instance_id
        adapter = self._get_adapter(instance_config)
        if adapter is None:
            return False, False
        entity_id = self._resolve_entity_id(instance_config, adapter)

        got_messages = False
        enqueued = False
        pending_events: list[Event] = []

        for space_mapping in instance_config.messaging.spaces:
            space_name = space_mapping.name
            handle = space_mapping.handle
            token_key = f"{instance_id}:{space_name}"
            token = self._store.get(token_key)

            try:
                events, next_token = adapter.poll(handle, token)
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
                self._store.put(token_key, next_token)

                if normalized_events and token is None:
                    logger.info(
                        "Initial sync %s/%s received events=%d; skipping enqueue until next poll",
                        instance_id,
                        space_name,
                        len(normalized_events),
                    )

                if normalized_events and token is not None:
                    if not entity_id:
                        logger.warning(
                            "Skipping reactive events for %s/%s: entity_id is unknown "
                            "(configure messaging.entity_id or use adapter identity lookup)",
                            instance_id,
                            space_name,
                        )
                        continue
                    # Filter out own events
                    external_events = [e for e in normalized_events if e.sender != entity_id]
                    filtered_self = len(normalized_events) - len(external_events)
                    if external_events:
                        got_messages = True
                        pending_events.extend(external_events)
                    else:
                        logger.info(
                            "No external events for %s/%s (events=%d filtered_self=%d)",
                            instance_id,
                            space_name,
                            len(normalized_events),
                            filtered_self,
                        )
            except httpx.TimeoutException as exc:
                logger.warning("Timeout polling %s/%s: %s", instance_id, space_name, exc)
            except Exception:
                logger.exception("Error polling %s/%s", instance_id, space_name)

        if got_messages:
            counter_key = f"external_count:{instance_id}"
            try:
                prior_count = int(self._store.get(counter_key) or 0)
            except (TypeError, ValueError):
                prior_count = 0
            new_count = prior_count + len(pending_events)
            self._store.put(counter_key, new_count)
            logger.info(
                "Recorded external messages for %s (delta=%d total=%d)",
                instance_id,
                len(pending_events),
                new_count,
            )
            if not self._queue.has_active(instance_id):
                payload = {"events": self._serialize_events(pending_events)}
                job_id = self._queue.enqueue(instance_id, "on_message", payload)
                if job_id:
                    logger.info(
                        "Enqueued on_message for %s (job %s, events=%d)",
                        instance_id,
                        job_id,
                        len(pending_events),
                    )
                    enqueued = True
                # Reset idle and thinks-since-reply counters on new messages
                self._store.put(f"idle:{instance_id}", 0)
                self._store.put(f"thinks_since_reply:{instance_id}", 0)
            else:
                logger.info("Skipping on_message enqueue for %s; active job already exists", instance_id)

        return got_messages, enqueued

    def _resolve_entity_id(self, instance_config: InstanceConfig, adapter) -> str:
        """Resolve sender identity used to filter out self-generated events."""
        instance_id = instance_config.instance_id
        messaging = instance_config.messaging
        if messaging is None:
            return ""

        configured = (messaging.entity_id or "").strip()
        if configured:
            self._resolved_entity_ids[instance_id] = configured
            return configured

        if instance_id in self._resolved_entity_ids:
            return self._resolved_entity_ids[instance_id] or ""

        resolved = ""
        lookup = getattr(adapter, "get_entity_id", None)
        if callable(lookup):
            try:
                resolved = str(lookup() or "").strip()
            except Exception:
                logger.exception("Failed adapter entity_id lookup for %s", instance_id)

        if resolved:
            self._resolved_entity_ids[instance_id] = resolved
            logger.info("Resolved entity_id for %s via adapter lookup: %s", instance_id, resolved)
            return resolved

        self._resolved_entity_ids[instance_id] = None
        if instance_id not in self._warned_missing_entity_id:
            logger.warning(
                "No entity_id configured for %s and adapter lookup failed; "
                "reactive events will be skipped to avoid self-trigger loops",
                instance_id,
            )
            self._warned_missing_entity_id.add(instance_id)
        return ""

    # ------------------------------------------------------------------
    # Schedule checking
    # ------------------------------------------------------------------

    def _check_schedules(self, reactive_instances: set[str]) -> set[str]:
        now = time.time()
        scheduled_instances: set[str] = set()

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

            max_thinks = instance_config.schedule.get("max_thinks_per_reply")
            if max_thinks is not None:
                try:
                    max_thinks = int(max_thinks)
                except (ValueError, TypeError):
                    max_thinks = None

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

                if instance_id not in reactive_instances:
                    # Idle throttle: cap consecutive idle heartbeats
                    if max_idle is not None:
                        idle_key = f"idle:{instance_id}"
                        idle_count = self._store.get(idle_key) or 0
                        if idle_count >= max_idle:
                            logger.debug(
                                "Skipping %s.%s (idle=%d >= max_idle=%d)",
                                instance_id, ep_name, idle_count, max_idle,
                            )
                            continue
                        self._store.put(idle_key, idle_count + 1)

                    # Thinks-per-reply throttle: cap heartbeats between replies
                    if max_thinks is not None:
                        thinks_key = f"thinks_since_reply:{instance_id}"
                        thinks_count = self._store.get(thinks_key) or 0
                        if thinks_count >= max_thinks:
                            logger.debug(
                                "Skipping %s.%s (thinks_since_reply=%d >= max_thinks_per_reply=%d)",
                                instance_id, ep_name, thinks_count, max_thinks,
                            )
                            continue
                        self._store.put(thinks_key, thinks_count + 1)

                # Enqueue if not already active
                if not self._queue.has_active(instance_id):
                    job_id = self._queue.enqueue(instance_id, ep_name, {})
                    if job_id:
                        logger.info(
                            "Enqueued %s.%s (job %s)", instance_id, ep_name, job_id
                        )
                        scheduled_instances.add(instance_id)
                else:
                    logger.debug(
                        "Skipping %s.%s — instance already has active job",
                        instance_id, ep_name,
                    )

        return scheduled_instances

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
