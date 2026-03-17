"""Checker — lightweight poller that enqueues jobs without running pipelines."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from croniter import croniter
import httpx

from library.harness.adapters import Event
from library.harness.config import HarnessConfig, InstanceConfig, load_harness_config, load_instance_config
from library.harness.jobqueue import JobQueue
from library.harness.registry import Registry
from library.harness.store import StoreDB, NamespacedStore, open_store

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

    Events are stored in a persistent inbox in SQLite, independent of job
    state.  The worker reads and clears the inbox at job start, so events
    can never be lost even if a job is already running.

    State persisted in SQLite (checker namespace):
      pending_events: {instance_id} -> list[dict]  (event inbox)
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
        """Execute one check cycle: poll adapters + check schedules.

        Messages are always prioritised over scheduled tasks. We poll first,
        enqueue any on_message jobs, then check schedules — so an instance
        with pending messages will never have a heartbeat enqueued instead.
        """
        instances = self._registry.list_instances()
        logger.debug("Checker cycle: found instances=%s", [i.instance_id for i in instances])
        logger.info("Checker cycle started (instances=%d)", len(instances))
        reactive_instances: set[str] = set()

        # Poll for messages first — reactive work takes priority.
        for instance_config in instances:
            got_messages, _enqueued = self._poll_instance(instance_config)
            if got_messages:
                reactive_instances.add(instance_config.instance_id)

        # Now check schedules, knowing which instances already have messages.
        scheduled_instances = self._check_schedules(reactive_instances)
        logger.info("Checker cycle finished (reactive=%d, scheduled=%d)",
                     len(reactive_instances), len(scheduled_instances))

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
            # Append events to the persistent inbox — always succeeds
            self._append_pending_events(instance_id, pending_events)

            counter_key = f"external_count:{instance_id}"
            try:
                prior_count = int(self._store.get(counter_key) or 0)
            except (TypeError, ValueError):
                prior_count = 0
            new_count = prior_count + len(pending_events)
            self._store.put(counter_key, new_count)
            logger.info(
                "Recorded external messages for %s (delta=%d total=%d inbox=%d)",
                instance_id,
                len(pending_events),
                new_count,
                len(self._load_pending_events(instance_id)),
            )

            # Ensure a job exists to process the inbox
            enqueued = self._ensure_job(instance_id, "on_message")

            # Reset idle counter on new messages
            self._store.put(f"idle:{instance_id}", 0)

        return got_messages, enqueued

    def _ensure_job(self, instance_id: str, entry_point: str) -> bool:
        """Ensure instance has an active job that includes the given entry point.

        Returns True if a job was enqueued or an existing one was updated.
        """
        has_active = self._queue.has_active(instance_id)
        logger.debug("%s.%s has_active=%s", instance_id, entry_point, has_active)
        if not has_active:
            payload = {"heartbeat": True} if entry_point == "heartbeat" else {}
            job_id = self._queue.enqueue(instance_id, entry_point, payload)
            if job_id:
                logger.info("Enqueued %s.%s (job %s)", instance_id, entry_point, job_id)
                return True
            logger.warning("Failed to enqueue %s.%s (returned None)", instance_id, entry_point)
            return False

        # Job exists — try to merge entry point into it (only works if pending)
        merge_payload = {"heartbeat": True} if entry_point == "heartbeat" else None
        if self._queue.merge_into_pending(instance_id, entry_point, merge_payload):
            logger.info("Merged %s.%s into pending job", instance_id, entry_point)
            return True

        # Job is running — that's fine. Events are in the inbox and will be
        # picked up by the next job after this one completes.
        logger.debug(
            "Job running for %s; %s will be handled next cycle",
            instance_id, entry_point,
        )
        return False

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
    # Persistent event inbox
    # ------------------------------------------------------------------

    def _append_pending_events(self, instance_id: str, events: list[Event]) -> None:
        """Append events to the persistent inbox for an instance."""
        key = f"pending_events:{instance_id}"
        existing = self._store.get(key)
        if not isinstance(existing, list):
            existing = []
        existing.extend(self._serialize_events(events))
        self._store.put(key, existing)

    def _load_pending_events(self, instance_id: str) -> list[dict]:
        """Load the raw pending events for an instance."""
        raw = self._store.get(f"pending_events:{instance_id}")
        if isinstance(raw, list):
            return raw
        return []

    @staticmethod
    def drain_pending_events(store: NamespacedStore, instance_id: str) -> list[dict]:
        """Read and clear the event inbox for an instance.

        Called by the worker at job start.  Returns serialized event dicts.
        """
        key = f"pending_events:{instance_id}"
        raw = store.get(key)
        if not isinstance(raw, list) or not raw:
            return []
        store.delete(key)
        return raw

    # ------------------------------------------------------------------
    # Schedule checking
    # ------------------------------------------------------------------

    def _check_schedules(self, reactive_instances: set[str]) -> set[str]:
        now = time.time()
        scheduled_instances: set[str] = set()
        all_instances = self._registry.list_instances()
        logger.debug("Checking schedules for instances: %s", [ic.instance_id for ic in all_instances])

        for instance_config in all_instances:
            instance_id = instance_config.instance_id
            manifest = self._registry.get_manifest(instance_config.species)
            logger.debug("Evaluating schedules for %s (species=%s)", instance_id, instance_config.species)

            # Collect all scheduled entry points from manifest + instance overrides
            schedule_map: dict[str, str] = {}
            for ep in manifest.entry_points:
                if ep.schedule:
                    schedule_map[ep.name] = ep.schedule

            for ep_name, value in instance_config.schedule.items():
                if isinstance(value, str):
                    schedule_map[ep_name] = value

            logger.debug("%s collected schedule_map from manifest+instance: %s", instance_id, schedule_map)
            if not schedule_map:
                logger.debug("%s has no schedules configured, skipping schedule check", instance_id)
                continue

            max_idle = instance_config.schedule.get("max_idle_heartbeats")
            if max_idle is not None:
                try:
                    max_idle = int(max_idle)
                except (ValueError, TypeError):
                    max_idle = None

            logger.debug("%s schedule_map keys: %s", instance_id, list(schedule_map.keys()))
            for ep_name, cron_expr in schedule_map.items():
                key = f"schedule_next:{instance_id}:{ep_name}"
                next_fire = self._store.get(key)

                if next_fire is None:
                    cron = croniter(cron_expr, now)
                    next_scheduled = cron.get_next(float)
                    self._store.put(key, next_scheduled)
                    logger.debug("%s.%s initialized (cron=%s, next_fire=%.1f)", instance_id, ep_name, cron_expr, next_scheduled)
                    continue

                if now < next_fire:
                    logger.debug("%s.%s not ready (now=%.1f < next_fire=%.1f, gap=%.1f sec)", instance_id, ep_name, now, next_fire, next_fire - now)
                    continue

                # Advance to next fire time
                cron = croniter(cron_expr, now)
                next_scheduled = cron.get_next(float)
                self._store.put(key, next_scheduled)
                logger.info("%s.%s fired (cron=%s, next_fire=%.1f)", instance_id, ep_name, cron_expr, next_scheduled)

                if instance_id not in reactive_instances:
                    # Idle throttle: cap consecutive idle heartbeats
                    throttled = False
                    if max_idle is not None:
                        idle_key = f"idle:{instance_id}"
                        idle_count = self._store.get(idle_key) or 0
                        if idle_count >= max_idle:
                            logger.debug(
                                "Skipping %s.%s (idle=%d >= max_idle=%d)",
                                instance_id, ep_name, idle_count, max_idle,
                            )
                            throttled = True
                        else:
                            self._store.put(idle_key, idle_count + 1)

                    # If heartbeat is throttled, check for pending messages instead
                    if throttled:
                        pending_key = f"pending_events:{instance_id}"
                        pending = self._store.get(pending_key)
                        if isinstance(pending, list) and pending:
                            logger.info(
                                "%s.%s throttled but has pending messages (count=%d), enqueueing on_message",
                                instance_id, ep_name, len(pending),
                            )
                            self._ensure_job(instance_id, "on_message")
                        continue

                # Enqueue heartbeat and increment thinks counter
                if ep_name == "heartbeat":
                    thinks_key = f"thinks_since_reply:{instance_id}"
                    thinks_count = self._store.get(thinks_key) or 0
                    self._store.put(thinks_key, thinks_count + 1)
                    logger.debug("%s heartbeat thinks_since_reply: %d -> %d", instance_id, thinks_count, thinks_count + 1)

                    # Reset window budgets on guaranteed thinking (heartbeat)
                    self.reset_window_budgets(self._store, instance_id, now)

                ensured = self._ensure_job(instance_id, ep_name)
                logger.debug("%s.%s ensure_job returned: %s", instance_id, ep_name, ensured)
                if ensured:
                    scheduled_instances.add(instance_id)

        return scheduled_instances

    # ------------------------------------------------------------------
    # Reply rate limiting
    # ------------------------------------------------------------------

    @staticmethod
    def check_reply_rate(store: NamespacedStore, instance_id: str, config: dict) -> bool:
        """Check if the instance is allowed to reply based on rate limits.

        Args:
            store: The checker namespace store.
            instance_id: The instance to check.
            config: The instance's schedule config dict.

        Returns:
            True if the instance can reply, False if rate-limited.
        """
        now = time.time()

        # Check cooldown
        cooldown = config.get("reply_cooldown_seconds")
        if cooldown is not None:
            try:
                cooldown = int(cooldown)
            except (ValueError, TypeError):
                cooldown = None

        if cooldown is not None:
            last_reply = store.get(f"last_reply_time:{instance_id}")
            if last_reply is not None:
                try:
                    elapsed = now - float(last_reply)
                    if elapsed < cooldown:
                        return False
                except (ValueError, TypeError):
                    pass

        # Check hourly cap
        max_per_hour = config.get("max_replies_per_hour")
        if max_per_hour is not None:
            try:
                max_per_hour = int(max_per_hour)
            except (ValueError, TypeError):
                max_per_hour = None

        if max_per_hour is not None:
            hour_start = store.get(f"reply_hour_start:{instance_id}")
            hour_count = store.get(f"reply_count_hour:{instance_id}") or 0
            try:
                hour_start = float(hour_start) if hour_start is not None else 0.0
                hour_count = int(hour_count)
            except (ValueError, TypeError):
                hour_start = 0.0
                hour_count = 0

            # Reset if hour has elapsed
            if now - hour_start > 3600:
                hour_start = now
                hour_count = 0
                store.put(f"reply_hour_start:{instance_id}", hour_start)
                store.put(f"reply_count_hour:{instance_id}", 0)

            if hour_count >= max_per_hour:
                return False

        return True

    @staticmethod
    def record_reply_sent(store: NamespacedStore, instance_id: str) -> None:
        """Record that a reply was sent, updating rate limit counters."""
        now = time.time()
        store.put(f"last_reply_time:{instance_id}", now)

        hour_start = store.get(f"reply_hour_start:{instance_id}")
        hour_count = store.get(f"reply_count_hour:{instance_id}") or 0
        try:
            hour_start = float(hour_start) if hour_start is not None else 0.0
            hour_count = int(hour_count)
        except (ValueError, TypeError):
            hour_start = 0.0
            hour_count = 0

        if now - hour_start > 3600:
            hour_start = now
            hour_count = 0

        store.put(f"reply_hour_start:{instance_id}", hour_start)
        store.put(f"reply_count_hour:{instance_id}", hour_count + 1)

    # ------------------------------------------------------------------
    # Scheduling constraints (per-window budgets)
    # ------------------------------------------------------------------

    @staticmethod
    def check_reply_budget(
        store: NamespacedStore,
        instance_id: str,
        constraints,  # SchedulingConstraints or None
    ) -> bool:
        """Check if the instance can send another reply in the current window.

        Returns True if allowed, False if budget exhausted.
        """
        if constraints is None:
            return True  # No constraints configured
        replies = store.get(f"replies_this_window:{instance_id}") or 0
        try:
            replies = int(replies)
        except (TypeError, ValueError):
            replies = 0
        return replies < constraints.max_replies_per_window

    @staticmethod
    def increment_reply_count(store: NamespacedStore, instance_id: str) -> None:
        """Increment the reply counter for the current window."""
        key = f"replies_this_window:{instance_id}"
        current = store.get(key) or 0
        try:
            current = int(current)
        except (TypeError, ValueError):
            current = 0
        store.put(key, current + 1)

    @staticmethod
    def check_reactive_thinking_budget(
        store: NamespacedStore,
        instance_id: str,
        constraints,  # SchedulingConstraints or None
        now: float,
    ) -> bool:
        """Check if the instance can run a reactive thinking session.

        Returns True if allowed, False if budget exhausted or in cooldown.
        """
        if constraints is None:
            return True  # No constraints configured

        sessions = store.get(f"reactive_sessions:{instance_id}") or 0
        last_session_time = store.get(f"last_reactive_session:{instance_id}") or 0
        try:
            sessions = int(sessions)
            last_session_time = float(last_session_time)
        except (TypeError, ValueError):
            sessions = 0
            last_session_time = 0

        if sessions >= constraints.reactive_thinking_max_sessions:
            return False
        if (now - last_session_time) < constraints.reactive_thinking_cooldown:
            return False
        return True

    @staticmethod
    def record_reactive_session(store: NamespacedStore, instance_id: str, now: float) -> None:
        """Record that a reactive thinking session was triggered."""
        sessions_key = f"reactive_sessions:{instance_id}"
        sessions = store.get(sessions_key) or 0
        try:
            sessions = int(sessions)
        except (TypeError, ValueError):
            sessions = 0
        store.put(sessions_key, sessions + 1)
        store.put(f"last_reactive_session:{instance_id}", now)

    @staticmethod
    def reset_window_budgets(store: NamespacedStore, instance_id: str, now: float) -> None:
        """Reset all budgets for a new guaranteed thinking window.

        Called when heartbeat fires (guaranteed thinking).
        """
        store.put(f"replies_this_window:{instance_id}", 0)
        store.put(f"reactive_sessions:{instance_id}", 0)
        store.put(f"last_guaranteed_thinking:{instance_id}", now)
        logger.debug("Reset window budgets for %s", instance_id)

    @staticmethod
    def get_on_message_phase(constraints) -> str | None:
        """Get the restricted phase for on_message handlers.

        Returns the phase name if constraints specify phase restrictions,
        None otherwise.
        """
        if constraints is None:
            return None
        if constraints.on_message_thinking_phases:
            # Return the first (and typically only) phase in the set
            phases = constraints.on_message_thinking_phases
            if phases:
                return next(iter(phases))
        return None

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
            from library.harness.adapters.matrix import MatrixAdapter
            token = instance_config.messaging.access_token or ""
            adapter = MatrixAdapter(
                homeserver=adapter_config.homeserver or "",
                access_token=token,
            )
        elif adapter_config.type == "local_file":
            from library.harness.adapters.local_file import LocalFileAdapter
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
