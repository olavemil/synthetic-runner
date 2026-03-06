"""Worker — drains the job queue, respecting provider concurrency limits."""

from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path

from symbiosis.harness.config import HarnessConfig, ProviderConfig
from symbiosis.harness.context import InstanceContext
from symbiosis.harness.jobqueue import Job, JobQueue
from symbiosis.harness.mailbox import Mailbox
from symbiosis.harness.registry import Registry
from symbiosis.harness.storage import NamespacedStorage
from symbiosis.harness.store import StoreDB, NamespacedStore

logger = logging.getLogger(__name__)


class Worker:
    """
    Drains the job queue run-to-empty, with:
    - Provider concurrency limits (slot claiming via SQLite)
    - Per-instance serialization (instance guard in JobQueue)
    - Parallel execution across different instances/providers
    """

    _SLOTS_NS = "provider:slots"

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
        self._instance_adapters: dict[str, object] = {}
        self._worker_id = f"worker:{uuid.uuid4().hex[:8]}"

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Drain the queue run-to-empty using threads for parallelism."""
        threads = []

        while True:
            job = self._queue.claim_next(
                self._worker_id,
                can_run=self._can_run,
            )
            if job is None:
                break

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

    # ------------------------------------------------------------------
    # Job execution
    # ------------------------------------------------------------------

    def _run_job(self, job: Job, slot_key: str | None) -> None:
        try:
            config = self._registry.get_instance_config(job.instance_id)
            ctx = self._build_context(config)
            handler = self._registry.get_handler(job.instance_id, job.entry_point)
            logger.info("Running %s.%s (job %s)", job.instance_id, job.entry_point, job.job_id)
            handler(ctx, **job.payload)
        except Exception:
            logger.exception("Error in %s.%s", job.instance_id, job.entry_point)
        finally:
            self._queue.complete(job, self._worker_id)
            if slot_key:
                self._release_provider_slot(slot_key)

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
            return True

        used = sum(
            1 for _, _, owner in self._slots_store.scan_items(f"{provider_id}:slot:")
            if owner is not None
        )
        return used < max_conc

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

    def _build_adapter(self, instance_config):
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
