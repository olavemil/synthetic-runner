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
class Trigger:
    """A single work trigger for an instance - either message arrival or scheduled heartbeat."""
    timestamp: float       # when this work was triggered
    cause: str            # "on_message" or "heartbeat" or other entry point name
    payload: dict = field(default_factory=dict)  # events, etc.


@dataclass
class Job:
    job_id: str           # sortable: earliest trigger timestamp + uuid
    instance_id: str
    triggers: list[Trigger] = field(default_factory=list)
    created_at: float = 0.0  # timestamp of earliest trigger

    @property
    def entry_point(self) -> str:
        """Primary entry point for logging/compat - first trigger's cause."""
        return self.triggers[0].cause if self.triggers else "unknown"

    @property
    def payload(self) -> dict:
        """Legacy payload accessor - merged from all triggers."""
        merged = {}
        for trigger in self.triggers:
            merged.update(trigger.payload)
        return merged


class JobQueue:
    """
    Job queue with trigger accumulation.

    Each instance has at most one job entry (pending or running).
    Work accumulates as triggers in the job's triggers list.
    When claimed, worker receives all accumulated triggers.

    Storage layout (NamespacedStore):
      jobs namespace:   job_id -> {instance_id, triggers: [{timestamp, cause, payload}], created_at}
      guards namespace: instance_id -> job_id  (owner = worker_id while running)
    """

    _JOBS_NS = "jobqueue:jobs"
    _GUARDS_NS = "jobqueue:guards"

    def __init__(self, db: StoreDB):
        self._jobs = NamespacedStore(db, self._JOBS_NS)
        self._guards = NamespacedStore(db, self._GUARDS_NS)

    # ------------------------------------------------------------------
    # Migration helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _migrate_job_data(data: dict) -> list[Trigger]:
        """Convert old job format (entry_point, payload) to new format (triggers).
        
        Returns list of Trigger objects, handling both old and new storage formats.
        """
        # New format: triggers list already present
        if "triggers" in data:
            trigger_dicts = data["triggers"]
            return [
                Trigger(
                    timestamp=t["timestamp"],
                    cause=t["cause"],
                    payload=t.get("payload", {})
                )
                for t in trigger_dicts
            ]
        
        # Old format: migrate from entry_point + payload
        entry_point = data.get("entry_point", "on_message")
        payload = data.get("payload", {})
        created_at = data.get("created_at", time.time())
        
        # Extract entry_points list if present (from old merge logic)
        entry_points = payload.get("entry_points", [entry_point])
        if not isinstance(entry_points, list):
            entry_points = [entry_point]
        
        # Create one trigger per entry point
        triggers = []
        for ep in entry_points:
            triggers.append(Trigger(
                timestamp=created_at,
                cause=ep,
                payload=payload.copy()  # each gets a copy of the full payload
            ))
        
        return triggers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(self, instance_id: str, entry_point: str, payload: dict | None = None) -> str:
        """
        Add work for instance_id/entry_point.

        If instance already has a job (pending or running), adds a new trigger.
        Otherwise creates a new job with one trigger.

        Always returns a job_id (existing or new).
        """
        now = time.time()
        trigger = Trigger(timestamp=now, cause=entry_point, payload=payload or {})

        # Check if instance already has a job
        existing_job_id = self._guards.get(instance_id)
        if existing_job_id:
            # Add trigger to existing job
            data = self._jobs.get(existing_job_id)
            if data:
                triggers = data.get("triggers", [])
                triggers.append({
                    "timestamp": trigger.timestamp,
                    "cause": trigger.cause,
                    "payload": trigger.payload
                })
                data["triggers"] = triggers
                self._jobs.put(existing_job_id, data)
                logger.debug(
                    "Added trigger %s to existing job %s for %s",
                    entry_point, existing_job_id[:8], instance_id
                )
                return existing_job_id

        # Create new job
        job_id = self._make_job_id(now)
        data = {
            "instance_id": instance_id,
            "triggers": [{
                "timestamp": trigger.timestamp,
                "cause": trigger.cause,
                "payload": trigger.payload
            }],
            "created_at": now,
        }
        self._jobs.put(job_id, data)
        self._guards.put(instance_id, job_id)
        logger.debug(
            "Created new job %s for %s (trigger: %s)",
            job_id[:8], instance_id, entry_point
        )
        return job_id

    def merge_into_pending(
        self,
        instance_id: str,
        entry_point: str,
        payload_updates: dict | None = None,
    ) -> bool:
        """
        Legacy method - now just calls enqueue() which handles both cases.
        Merge work into an existing pending (unclaimed) job for instance_id.

        Adds entry_point to the job's entry_points list (if not already present)
        and merges payload_updates into the job's payload. For 'events', new
        events are appended to any existing list.

        Returns True if merge succeeded, False if no pending job exists
        (e.g. the job is already running or there is no job at all).
        """
        # Check if job exists and is pending
        guard_items = self._guards.scan_items()
        guard = None
        for inst, jid, owner in guard_items:
            if inst == instance_id:
                guard = (jid, owner)
                break

        if guard is None:
            return False  # no job exists

        _, owner = guard
        if owner is not None:
            return False  # job is running, can't merge

        # Job exists and is pending, enqueue will add the trigger
        self.enqueue(instance_id, entry_point, payload_updates or {})
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

        logger.debug("claim_next scanning %d jobs in queue", len(all_jobs))
        for idx, (job_id, data) in enumerate(all_jobs):
            instance_id = data["instance_id"]
            guard = guard_map.get(instance_id)
            if guard is None:
                logger.debug("  [%d] %s (job %s...) SKIP: no guard (stale)", idx, instance_id, job_id[:8])
                continue  # no guard — stale job, skip
            _, owner = guard
            if owner is not None:
                logger.debug("  [%d] %s (job %s...) SKIP: already running by %s", idx, instance_id, job_id[:8], owner)
                continue  # already running

            if can_run and not can_run(instance_id):
                logger.debug("  [%d] %s (job %s...) SKIP: can_run=False (provider slots full)", idx, instance_id, job_id[:8])
                continue  # provider slots full

            # Try to atomically claim the guard
            if self._guards.claim(instance_id, worker_id):
                logger.debug("  [%d] %s (job %s...) CLAIMED by %s", idx, instance_id, job_id[:8], worker_id)
                # Migrate old job data if needed
                triggers = self._migrate_job_data(data)
                return Job(
                    job_id=job_id,
                    instance_id=instance_id,
                    triggers=triggers,
                    created_at=data.get("created_at", time.time()),
                )
            else:
                logger.debug("  [%d] %s (job %s...) SKIP: atomic claim failed (race)", idx, instance_id, job_id[:8])

        logger.debug("claim_next found no claimable jobs")
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
                # Migrate old job data if needed
                triggers = self._migrate_job_data(data)
                jobs.append(Job(
                    job_id=job_id,
                    instance_id=data["instance_id"],
                    triggers=triggers,
                    created_at=data.get("created_at", time.time()),
                ))
        return jobs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_job_id(now: float | None = None) -> str:
        """Generate a sortable job ID from timestamp and UUID.
        
        Args:
            now: Optional timestamp (seconds since epoch). If None, uses current time.
        """
        if now is None:
            now = time.time()
        ts = int(now * 1_000_000)
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
