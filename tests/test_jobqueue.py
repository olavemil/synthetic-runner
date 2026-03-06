"""Tests for JobQueue."""

import time

import pytest

from symbiosis.harness.jobqueue import Job, JobQueue
from symbiosis.harness.store import open_store


@pytest.fixture
def queue():
    db = open_store()
    return JobQueue(db)


class TestEnqueue:
    def test_enqueue_returns_job_id(self, queue):
        job_id = queue.enqueue("inst-a", "on_message", {})
        assert job_id is not None
        assert isinstance(job_id, str)

    def test_enqueue_marks_active(self, queue):
        queue.enqueue("inst-a", "on_message", {})
        assert queue.has_active("inst-a")

    def test_enqueue_deduplicates(self, queue):
        first = queue.enqueue("inst-a", "on_message", {})
        second = queue.enqueue("inst-a", "heartbeat", {})
        assert first is not None
        assert second is None  # blocked by active job

    def test_enqueue_different_instances(self, queue):
        j1 = queue.enqueue("inst-a", "on_message", {})
        j2 = queue.enqueue("inst-b", "on_message", {})
        assert j1 is not None
        assert j2 is not None

    def test_no_active_when_empty(self, queue):
        assert not queue.has_active("inst-a")


class TestClaimNext:
    def test_claim_pending_job(self, queue):
        queue.enqueue("inst-a", "on_message", {})
        job = queue.claim_next("worker-1")
        assert job is not None
        assert job.instance_id == "inst-a"
        assert job.entry_point == "on_message"

    def test_claim_returns_none_when_empty(self, queue):
        job = queue.claim_next("worker-1")
        assert job is None

    def test_claim_respects_running_guard(self, queue):
        queue.enqueue("inst-a", "on_message", {})
        job1 = queue.claim_next("worker-1")
        # Job is now running; a second worker should not claim it
        job2 = queue.claim_next("worker-2")
        assert job1 is not None
        assert job2 is None

    def test_claim_with_can_run_false(self, queue):
        queue.enqueue("inst-a", "on_message", {})
        job = queue.claim_next("worker-1", can_run=lambda inst: False)
        assert job is None

    def test_claim_with_can_run_true(self, queue):
        queue.enqueue("inst-a", "on_message", {})
        job = queue.claim_next("worker-1", can_run=lambda inst: True)
        assert job is not None

    def test_claim_oldest_first(self, queue):
        queue.enqueue("inst-a", "heartbeat", {})
        time.sleep(0.001)
        queue.enqueue("inst-b", "heartbeat", {})
        job = queue.claim_next("worker-1")
        assert job is not None
        assert job.instance_id == "inst-a"


class TestComplete:
    def test_complete_clears_active(self, queue):
        queue.enqueue("inst-a", "on_message", {})
        job = queue.claim_next("worker-1")
        queue.complete(job, "worker-1")
        assert not queue.has_active("inst-a")

    def test_complete_allows_reenqueue(self, queue):
        queue.enqueue("inst-a", "on_message", {})
        job = queue.claim_next("worker-1")
        queue.complete(job, "worker-1")
        new_job_id = queue.enqueue("inst-a", "heartbeat", {})
        assert new_job_id is not None

    def test_complete_deletes_job(self, queue):
        queue.enqueue("inst-a", "on_message", {})
        job = queue.claim_next("worker-1")
        queue.complete(job, "worker-1")
        assert queue.pending_count() == 0


class TestCounts:
    def test_pending_count(self, queue):
        queue.enqueue("inst-a", "on_message", {})
        queue.enqueue("inst-b", "heartbeat", {})
        assert queue.pending_count() == 2

    def test_running_count(self, queue):
        queue.enqueue("inst-a", "on_message", {})
        queue.enqueue("inst-b", "heartbeat", {})
        queue.claim_next("worker-1")
        assert queue.running_count() == 1
        assert queue.pending_count() == 1

    def test_list_pending(self, queue):
        queue.enqueue("inst-a", "on_message", {})
        queue.enqueue("inst-b", "heartbeat", {})
        pending = queue.list_pending()
        assert len(pending) == 2
        instance_ids = {j.instance_id for j in pending}
        assert instance_ids == {"inst-a", "inst-b"}
