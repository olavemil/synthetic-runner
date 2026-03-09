"""Job queue — SQLite-backed queue with instance deduplication and provider slots."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

from library.harness.store import StoreDB, NamespacedStore

logger = logging.getLogger(__name__)


@dataclass
class Job:
    job_id: str        # sortable: "{timestamp_us:020d}:{uuid4}"
    instance_id: str
    entry_point: str   # primary entry point (kept for logging/compat)
    payload: dict
    created_at: float


class JobQueue:
    """
    Manages pending/running jobs with two invariants:
    - At most one pending-or-running job per instance (instance guard).
    - Jobs within the same instance are processed in enqueue order.

    Payload convention for merged jobs:
      - "events": list[dict]     — present when on_message work is due
      - "heartbeat": True        — present when heartbeat work is due
      - "entry_points": list[str] — ordered list of entry points to run

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
        final_payload = payload or {}
        final_payload.setdefault("entry_points", [entry_point])
        data = {
            "instance_id": instance_id,
            "entry_point": entry_point,
            "payload": final_payload,
            "created_at": time.time(),
        }
        self._jobs.put(job_id, data)
        # Register the guard (unclaimed = pending)
        self._guards.put(instance_id, job_id)
        return job_id

    def merge_into_pending(
        self,
        instance_id: str,
        entry_point: str,
        payload_updates: dict | None = None,
    ) -> bool:
        """
        Merge work into an existing pending (unclaimed) job for instance_id.

        Adds entry_point to the job's entry_points list (if not already present)
        and merges payload_updates into the job's payload. For 'events', new
        events are appended to any existing list.

        Returns True if merge succeeded, False if no pending job exists
        (e.g. the job is already running or there is no job at all).
        """
        guard_items = self._guards.scan_items()
        guard = None
        for inst, jid, owner in guard_items:
            if inst == instance_id:
                guard = (jid, owner)
                break

        if guard is None:
            return False

        job_id, owner = guard
        if owner is not None:
            return False  # job is running, can't merge

        # Load the job data
        data = self._jobs.get(job_id)
        if data is None:
            return False

        # Merge entry_points
        existing_eps = data["payload"].get("entry_points", [data.get("entry_point", "on_message")])
        if entry_point not in existing_eps:
            existing_eps.append(entry_point)
        data["payload"]["entry_points"] = existing_eps

        # Merge payload updates
        if payload_updates:
            for key, value in payload_updates.items():
                if key == "entry_points":
                    continue  # handled above
                if key == "events" and isinstance(value, list):
                    existing_events = data["payload"].get("events", [])
                    existing_events.extend(value)
                    data["payload"]["events"] = existing_events
                else:
                    data["payload"][key] = value

        self._jobs.put(job_id, data)
        return True

    def has_active(self, instance_id: str) -> bool:
        """True if instance_id has a pending or running job."""
        return self._guards.get(instance_id) is not None

    def is_running(self, instance_id: str) -> bool:
        """True if instance_id has a running (claimed) job."""
        items = self._guards.scan_items()
        for inst, _, owner in items:
            if inst == instance_id:
                return owner is not None
        return False

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

    def cleanup_stale_guards(self, max_age_seconds: float = 3600) -> int:
        """Remove guards for jobs that have been running longer than max_age_seconds.

        This handles the case where a worker crashes without completing a job,
        leaving the guard stuck in "running" state forever.

        Returns the number of stale guards cleaned up.
        """
        now = time.time()
        cleaned = 0

        guards_items = list(self._guards.scan_items())  # [(instance_id, job_id, owner), ...]

        for instance_id, job_id, owner in guards_items:
            if owner is None:
                continue  # pending, not stale

            # Extract timestamp from job_id (first 20 digits are microseconds)
            try:
                ts_us = int(job_id.split(":")[0])
                created_at = ts_us / 1_000_000
            except (ValueError, IndexError):
                continue

            age = now - created_at
            if age > max_age_seconds:
                logger.warning(
                    "Cleaning up stale guard for %s (job %s, age=%.0fs, owner=%s)",
                    instance_id, job_id, age, owner,
                )
                # Force-release the guard and delete the job
                self._guards.delete(instance_id)
                self._jobs.delete(job_id)
                cleaned += 1

        return cleaned
