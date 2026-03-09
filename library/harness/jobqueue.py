"""Job queue — SQLite-backed queue with instance deduplication and provider slots."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Callable

from library.harness.store import StoreDB, NamespacedStore


@dataclass
class Job:
    job_id: str        # sortable: "{timestamp_us:020d}:{uuid4}"
    instance_id: str
    entry_point: str
    payload: dict
    created_at: float


class JobQueue:
    """
    Manages pending/running jobs with two invariants:
    - At most one pending-or-running job per instance (instance guard).
    - Jobs within the same instance are processed in enqueue order.

    Storage layout (NamespacedStore):
      jobs namespace:   job_id -> {instance_id, entry_point, payload, created_at}
      guards namespace: instance_id -> job_id  (owner = worker_id while running)
    """

    _JOBS_NS = "jobqueue:jobs"
    _GUARDS_NS = "jobqueue:guards"

    def __init__(self, db: StoreDB):
        self._jobs = NamespacedStore(db, self._JOBS_NS)
        self._guards = NamespacedStore(db, self._GUARDS_NS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(self, instance_id: str, entry_point: str, payload: dict | None = None) -> str | None:
        """
        Enqueue a job for instance_id/entry_point.

        Returns the new job_id, or None if the instance already has an
        active (pending or running) job.
        """
        if self.has_active(instance_id):
            return None

        job_id = self._make_job_id()
        data = {
            "instance_id": instance_id,
            "entry_point": entry_point,
            "payload": payload or {},
            "created_at": time.time(),
        }
        self._jobs.put(job_id, data)
        # Register the guard (unclaimed = pending)
        self._guards.put(instance_id, job_id)
        return job_id

    def has_active(self, instance_id: str) -> bool:
        """True if instance_id has a pending or running job."""
        return self._guards.get(instance_id) is not None

    def claim_next(
        self,
        worker_id: str,
        can_run: Callable[[str], bool] | None = None,
    ) -> Job | None:
        """
        Find the oldest pending job whose instance guard is unclaimed
        and whose instance passes can_run (e.g. provider slot available).

        Atomically marks it as running by claiming the guard with worker_id.
        Returns the Job or None if nothing is available.
        """
        # Scan all jobs in insertion order (job_id is sortable by creation time)
        all_jobs = self._jobs.scan()  # [(job_id, data), ...]
        guards_items = self._guards.scan_items()  # [(instance_id, job_id, owner), ...]
        guard_map = {inst: (jid, owner) for inst, jid, owner in guards_items}

        for job_id, data in all_jobs:
            instance_id = data["instance_id"]
            guard = guard_map.get(instance_id)
            if guard is None:
                continue  # no guard — stale job, skip
            _, owner = guard
            if owner is not None:
                continue  # already running

            if can_run and not can_run(instance_id):
                continue  # provider slots full

            # Try to atomically claim the guard
            if self._guards.claim(instance_id, worker_id):
                return Job(
                    job_id=job_id,
                    instance_id=instance_id,
                    entry_point=data["entry_point"],
                    payload=data.get("payload", {}),
                    created_at=data["created_at"],
                )

        return None

    def complete(self, job: Job, worker_id: str) -> None:
        """Mark a job as done: remove it and release the instance guard."""
        self._jobs.delete(job.job_id)
        self._guards.release(job.instance_id, worker_id)
        # Clear the guard entry entirely so has_active() returns False
        self._guards.delete(job.instance_id)

    def pending_count(self) -> int:
        """Number of pending (unclaimed) jobs."""
        items = self._guards.scan_items()
        return sum(1 for _, _, owner in items if owner is None)

    def running_count(self) -> int:
        """Number of currently running (claimed) jobs."""
        items = self._guards.scan_items()
        return sum(1 for _, _, owner in items if owner is not None)

    def list_pending(self) -> list[Job]:
        """Return all pending jobs in enqueue order."""
        all_jobs = self._jobs.scan()
        guards_items = self._guards.scan_items()
        pending_instances = {inst for inst, _, owner in guards_items if owner is None}

        jobs = []
        for job_id, data in all_jobs:
            if data["instance_id"] in pending_instances:
                jobs.append(Job(
                    job_id=job_id,
                    instance_id=data["instance_id"],
                    entry_point=data["entry_point"],
                    payload=data.get("payload", {}),
                    created_at=data["created_at"],
                ))
        return jobs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_job_id() -> str:
        ts = int(time.time() * 1_000_000)
        return f"{ts:020d}:{uuid.uuid4().hex}"
