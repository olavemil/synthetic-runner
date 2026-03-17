"""Test backward compatibility for old job format migration."""

import pytest
from library.harness.jobqueue import JobQueue
from library.harness.store import open_store


@pytest.fixture
def queue():
    db = open_store()
    return JobQueue(db)


class TestOldJobFormatMigration:
    def test_migrate_simple_old_format(self, queue):
        """Test migration of old format: entry_point + payload."""
        # Manually insert old-format job data
        job_id = queue._make_job_id()
        old_data = {
            "instance_id": "test-inst",
            "entry_point": "on_message",
            "payload": {"events": [{"body": "hello"}]},
            "created_at": 12345.0,
        }
        queue._jobs.put(job_id, old_data)
        queue._guards.put("test-inst", job_id)
        
        # Claim the job - should migrate automatically
        job = queue.claim_next("worker-1")
        
        assert job is not None
        assert job.instance_id == "test-inst"
        assert len(job.triggers) == 1
        assert job.triggers[0].cause == "on_message"
        assert job.triggers[0].payload["events"][0]["body"] == "hello"
        assert job.created_at == 12345.0

    def test_migrate_old_format_with_entry_points_list(self, queue):
        """Test migration of old merged format with entry_points list."""
        # This was the old merge format where multiple entry points were stored
        job_id = queue._make_job_id()
        old_data = {
            "instance_id": "test-inst",
            "entry_point": "on_message",  # primary
            "payload": {
                "entry_points": ["on_message", "heartbeat"],
                "events": [{"body": "hello"}],
                "heartbeat": True,
            },
            "created_at": 12345.0,
        }
        queue._jobs.put(job_id, old_data)
        queue._guards.put("test-inst", job_id)
        
        # Claim the job - should create separate triggers
        job = queue.claim_next("worker-1")
        
        assert job is not None
        assert len(job.triggers) == 2
        assert job.triggers[0].cause == "on_message"
        assert job.triggers[1].cause == "heartbeat"
        # Both triggers get full payload (old behavior)
        assert "events" in job.triggers[0].payload
        assert "heartbeat" in job.triggers[1].payload

    def test_migrate_in_list_pending(self, queue):
        """Test that list_pending also migrates old format."""
        job_id = queue._make_job_id()
        old_data = {
            "instance_id": "test-inst",
            "entry_point": "heartbeat",
            "payload": {"heartbeat": True},
            "created_at": 12345.0,
        }
        queue._jobs.put(job_id, old_data)
        queue._guards.put("test-inst", job_id)
        
        # List pending should also migrate
        pending = queue.list_pending()
        
        assert len(pending) == 1
        job = pending[0]
        assert len(job.triggers) == 1
        assert job.triggers[0].cause == "heartbeat"

    def test_new_format_still_works(self, queue):
        """Test that new trigger format continues to work."""
        # Use normal enqueue which creates new format
        job_id = queue.enqueue("test-inst", "on_message", {"events": []})
        
        job = queue.claim_next("worker-1")
        
        assert job is not None
        assert len(job.triggers) == 1
        assert job.triggers[0].cause == "on_message"
