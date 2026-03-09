"""Worker — drains the job queue, respecting provider concurrency limits."""

from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path

from library.harness.adapters import Event
from library.harness.checker import Checker
from library.harness.config import HarnessConfig, ProviderConfig
from library.harness.context import InstanceContext
from library.harness.jobqueue import Job, JobQueue
from library.harness.mailbox import Mailbox
from library.harness.registry import Registry
from library.harness.storage import NamespacedStorage
from library.harness.store import StoreDB, NamespacedStore

logger = logging.getLogger(__name__)


class Worker:
    """
    Drains the job queue run-to-empty, with:
    - Provider concurrency limits (slot claiming via SQLite)
    - Per-instance serialization (instance guard in JobQueue)
    - Parallel execution across different instances/providers
    """

    _SLOTS_NS = "provider:slots"
    _CHECKER_NS = "checker"

    def __init__(
        self,
        harness_config: HarnessConfig,
        registry: Registry,
        providers: dict,
        adapters: dict,
        store_db: StoreDB,
        base_dir: str | Path = ".",
    ):
        self._config = harness_config
        self._registry = registry
        self._providers = providers
        self._adapters = adapters
        self._store_db = store_db
        self._base_dir = Path(base_dir)
        self._queue = JobQueue(store_db)
        self._slots_store = NamespacedStore(store_db, self._SLOTS_NS)
        self._checker_store = NamespacedStore(store_db, self._CHECKER_NS)
        self._instance_adapters: dict[str, object] = {}
        self._worker_id = f"worker:{uuid.uuid4().hex[:8]}"

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Drain the queue run-to-empty using threads for parallelism."""
        # Clean up stale guards from crashed workers (jobs stuck for >1 hour)
        stale_cleaned = self._queue.cleanup_stale_guards(max_age_seconds=3600)
        if stale_cleaned:
            logger.info("Cleaned up %d stale job guard(s)", stale_cleaned)

        pending_jobs = self._queue.list_pending()
        pending_instances = sorted({job.instance_id for job in pending_jobs})
        logger.info(
            "Worker cycle started (enqueued_instances=%s)",
            ",".join(pending_instances) if pending_instances else "-",
        )
        logger.debug(
            "Pending jobs breakdown: %s",
            {inst: len([j for j in pending_jobs if j.instance_id == inst]) for inst in pending_instances}
        )
        threads = []
        claimed_jobs = 0

        while True:
            job = self._queue.claim_next(
                self._worker_id,
                can_run=self._can_run,
            )
            if job is None:
                logger.debug("No more jobs to claim (claimed_jobs=%d)", claimed_jobs)
                break
            claimed_jobs += 1
            logger.debug("Claimed job for %s (job_id=%s, entry_point=%s, can_run=%s)", 
                        job.instance_id, job.job_id, job.entry_point, self._can_run(job.instance_id))

            provider_id = self._get_provider_id(job.instance_id)
            slot_key = self._claim_provider_slot(provider_id)

            t = threading.Thread(
                target=self._run_job,
                args=(job, slot_key),
                daemon=True,
                name=f"job-{job.instance_id}-{job.entry_point}",
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        logger.info("Worker cycle finished (claimed_jobs=%d)", claimed_jobs)

    # ------------------------------------------------------------------
    # Job execution
    # ------------------------------------------------------------------

    def _run_job(self, job: Job, slot_key: str | None) -> None:
        success = True
        failure_reason = ""
        heartbeat_gate_used = False
        heartbeat_external_count = 0
        ctx: InstanceContext | None = None
        try:
            config = self._registry.get_instance_config(job.instance_id)
            ctx = self._build_context(config)
            payload = dict(job.payload or {})
            entry_points = payload.pop("entry_points", [job.entry_point])

            # Drain the persistent event inbox (written by checker)
            inbox_events_raw = Checker.drain_pending_events(
                self._checker_store, job.instance_id
            )
            inbox_events = [
                Event(**evt) if isinstance(evt, dict) else evt
                for evt in inbox_events_raw
            ]

            has_heartbeat = bool(payload.pop("heartbeat", False)) or "heartbeat" in entry_points

            # Build ordered phases: messages first, then heartbeat
            phases: list[tuple[str, dict]] = []
            if inbox_events or "on_message" in entry_points:
                phases.append(("on_message", {"events": inbox_events}))
            if has_heartbeat:
                phases.append(("heartbeat", {}))

            if not phases:
                # Fallback: run the primary entry point as before
                phases.append((job.entry_point, payload))

            logger.info(
                "Running %s (job %s, phases=%s, events=%d)",
                job.instance_id,
                job.job_id,
                "+".join(ep for ep, _ in phases),
                len(inbox_events),
            )

            for ep_name, ep_payload in phases:
                handler = self._registry.get_handler(job.instance_id, ep_name)
                heartbeat_gate_used_phase, heartbeat_external_count_phase = (
                    self._configure_send_policy(ctx, job, ep_name, ep_payload)
                )
                if ep_name == "heartbeat":
                    heartbeat_gate_used = heartbeat_gate_used_phase
                    heartbeat_external_count = heartbeat_external_count_phase

                logger.info(
                    "Running phase %s.%s (job %s)",
                    job.instance_id, ep_name, job.job_id,
                )
                handler(ctx, **ep_payload)
                logger.info(
                    "Completed phase %s.%s (job %s)",
                    job.instance_id, ep_name, job.job_id,
                )

        except Exception as exc:
            success = False
            failure_reason = f"{type(exc).__name__}: {exc}"
            logger.exception("Error in %s", job.instance_id)
        finally:
            sent_count = self._coerce_int(getattr(ctx, "sent_message_count", 0) if ctx is not None else 0, 0)
            if ctx is not None and heartbeat_gate_used and sent_count > 0:
                seen_key = self._heartbeat_seen_key(job.instance_id)
                self._checker_store.put(seen_key, heartbeat_external_count)
                logger.info(
                    "Heartbeat send gate consumed for %s (external_count=%d)",
                    job.instance_id,
                    heartbeat_external_count,
                )
            self._queue.complete(job, self._worker_id)
            if slot_key:
                self._release_provider_slot(slot_key)
            if not success:
                logger.error(
                    "Failed %s (job %s, reason=%s)",
                    job.instance_id,
                    job.job_id,
                    failure_reason or "(unknown)",
                )
            logger.info(
                "Completed %s (job %s, success=%s)",
                job.instance_id,
                job.job_id,
                success,
            )

    def _configure_send_policy(self, ctx: InstanceContext, job: Job, entry_point: str, payload: dict) -> tuple[bool, int]:
        """Configure per-phase messaging limits.

        Returns:
            (uses_heartbeat_gate, external_count_snapshot)
        """
        if entry_point == "on_message":
            events = payload.get("events")
            event_count = len(events) if isinstance(events, list) else 0
            allow = event_count > 0
            ctx.configure_send_policy(
                allow_send=allow,
                max_sends=1 if allow else 0,
                reason="on_message sends require external events and are limited to one message",
            )
            logger.info(
                "Configured send policy for %s.%s (allow=%s max_sends=%d events=%d)",
                job.instance_id,
                entry_point,
                allow,
                1 if allow else 0,
                event_count,
            )
            return False, 0

        if entry_point == "heartbeat":
            external_count = self._coerce_int(self._checker_store.get(f"external_count:{job.instance_id}"), 0)
            seen_count = self._coerce_int(self._checker_store.get(self._heartbeat_seen_key(job.instance_id)), 0)
            allow = external_count > seen_count
            ctx.configure_send_policy(
                allow_send=allow,
                max_sends=1 if allow else 0,
                reason=(
                    "heartbeat sends require unseen external messages from other entities "
                    "(excluding self messages)"
                ),
            )
            logger.info(
                "Configured send policy for %s.%s (allow=%s max_sends=%d external_count=%d seen=%d)",
                job.instance_id,
                entry_point,
                allow,
                1 if allow else 0,
                external_count,
                seen_count,
            )
            return True, external_count

        return False, 0

    @staticmethod
    def _coerce_int(value, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _heartbeat_seen_key(instance_id: str) -> str:
        return f"send_gate_seen:{instance_id}:heartbeat"

    # ------------------------------------------------------------------
    # Provider slot management
    # ------------------------------------------------------------------

    def _get_provider_id(self, instance_id: str) -> str | None:
        try:
            config = self._registry.get_instance_config(instance_id)
            return config.provider
        except KeyError:
            return None

    def _get_max_concurrency(self, provider_id: str | None) -> int | None:
        if not provider_id:
            return None
        try:
            pc = self._config.get_provider(provider_id)
            return pc.max_concurrency
        except KeyError:
            return None

    def _can_run(self, instance_id: str) -> bool:
        """Check if a provider slot is available for the instance's provider."""
        provider_id = self._get_provider_id(instance_id)
        max_conc = self._get_max_concurrency(provider_id)
        if max_conc is None:
            logger.debug("%s can_run=True (no concurrency limit for provider %s)", instance_id, provider_id)
            return True

        used = sum(
            1 for _, _, owner in self._slots_store.scan_items(f"{provider_id}:slot:")
            if owner is not None
        )
        can_run = used < max_conc
        logger.debug("%s can_run=%s (provider=%s, used=%d/%d slots)", instance_id, can_run, provider_id, used, max_conc)
        return can_run

    def _claim_provider_slot(self, provider_id: str | None) -> str | None:
        if not provider_id:
            return None
        max_conc = self._get_max_concurrency(provider_id)
        if max_conc is None:
            return None

        for i in range(max_conc):
            slot_key = f"{provider_id}:slot:{i}"
            if self._slots_store.claim(slot_key, self._worker_id):
                return slot_key
        return None

    def _release_provider_slot(self, slot_key: str) -> None:
        self._slots_store.release(slot_key, self._worker_id)

    # ------------------------------------------------------------------
    # Context construction
    # ------------------------------------------------------------------

    def _build_context(self, instance_config) -> InstanceContext:
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

        ctx = InstanceContext(
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
        ctx._sync_config = self._config.sync
        return ctx

    def _build_adapter(self, instance_config):
        instance_id = instance_config.instance_id
        if instance_id in self._instance_adapters:
            return self._instance_adapters[instance_id]

        if not instance_config.messaging:
            return None

        adapter_id = instance_config.messaging.adapter
        adapter_config = self._config.get_adapter(adapter_id)

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
