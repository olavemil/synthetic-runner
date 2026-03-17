"""Tests for sync token commit behavior after message consumption."""

from unittest.mock import MagicMock, patch

from library.harness.adapters import Event
from library.harness.checker import Checker
from library.harness.config import HarnessConfig, ProviderConfig, AdapterConfig
from library.harness.jobqueue import JobQueue
from library.harness.registry import Registry
from library.harness.store import open_store
from library.harness.worker import Worker
from library.species import Species, SpeciesManifest, EntryPoint


def _make_harness_config(adapter_type="matrix"):
    return HarnessConfig(
        providers=[ProviderConfig(id="test", type="anthropic")],
        adapters=[AdapterConfig(id="adapter-1", type=adapter_type)],
    )


def _make_species(schedule: str | None = None):
    h = MagicMock()
    manifest = SpeciesManifest(
        species_id="test-species",
        entry_points=[
            EntryPoint(name="on_message", handler=h, trigger="message", schedule=None),
            EntryPoint(name="heartbeat", handler=h, trigger=None, schedule=schedule),
        ],
        tools=[],
        default_files={},
        spawn=lambda ctx: None,
    )
    s = MagicMock(spec=Species)
    s.manifest.return_value = manifest
    return s


def _make_instance(instance_id="inst-1", with_messaging=True, entity_id="@bot:matrix.org"):
    from library.harness.config import InstanceConfig, MessagingConfig, SpaceMapping
    
    messaging = None
    if with_messaging:
        messaging = MessagingConfig(
            adapter="adapter-1",
            entity_id=entity_id,
            spaces=[SpaceMapping(name="main", handle="!room:matrix.org")],
        )
    
    return InstanceConfig(
        instance_id=instance_id,
        species="test-species",
        provider="test",
        model="test-model",
        messaging=messaging,
        schedule={"heartbeat": "* * * * *"},
    )


class TestSyncTokenCommit:
    """Test that sync tokens are committed only after successful message consumption."""
    
    def test_pending_token_saved_when_job_active(self):
        """When a job is already active, sync token should be saved as pending."""
        db = open_store()
        species = _make_species()
        instance = _make_instance()
        registry = Registry()
        registry.register_species(species)
        registry.register_instance(instance)
        
        checker = Checker(
            harness_config=_make_harness_config(),
            registry=registry,
            store_db=db,
            base_dir=None,
        )
        queue = JobQueue(db)
        
        # Create an active job first
        queue.enqueue("inst-1", "heartbeat", {})
        
        # Mock adapter that returns events
        evt = Event(
            event_id="$evt1",
            sender="@user:matrix.org",
            body="hello",
            timestamp=1,
            room="!room:matrix.org",
        )
        adapter = MagicMock()
        adapter.poll.return_value = ([evt], "tok2")
        
        with patch.object(checker, "_get_adapter", return_value=adapter):
            checker._poll_instance(instance)
        
        # Check that regular sync token was NOT updated
        token_key = "inst-1:main"
        actual_token = checker._store.get(token_key)
        assert actual_token is None  # Should be None since no previous poll
        
        # Check that pending token WAS saved
        pending_token_key = "pending_token:inst-1:main"
        pending_token = checker._store.get(pending_token_key)
        assert pending_token == "tok2"
    
    def test_sync_token_committed_immediately_when_no_job(self):
        """When no job is active, sync token should be committed immediately."""
        db = open_store()
        species = _make_species()
        instance = _make_instance()
        registry = Registry()
        registry.register_species(species)
        registry.register_instance(instance)
        
        checker = Checker(
            harness_config=_make_harness_config(),
            registry=registry,
            store_db=db,
            base_dir=None,
        )
        
        # Mock adapter for initial sync (no events)
        adapter = MagicMock()
        adapter.poll.return_value = ([], "tok1")
        
        with patch.object(checker, "_get_adapter", return_value=adapter):
            checker._poll_instance(instance)
        
        # Check that sync token was committed immediately
        token_key = "inst-1:main"
        actual_token = checker._store.get(token_key)
        assert actual_token == "tok1"
        
        # Check that no pending token exists
        pending_token_key = "pending_token:inst-1:main"
        pending_token = checker._store.get(pending_token_key)
        assert pending_token is None
    
    def test_worker_commits_pending_tokens_after_on_message(self):
        """Worker should commit pending tokens after successfully completing on_message."""
        db = open_store()
        species = _make_species()
        instance = _make_instance()
        registry = Registry()
        registry.register_species(species)
        registry.register_instance(instance)
        
        # Set up pending token
        from library.harness.store import NamespacedStore
        store = NamespacedStore(db, "checker")
        store.put("pending_token:inst-1:main", "tok_pending")
        
        # Set up job with on_message trigger
        queue = JobQueue(db)
        queue.enqueue("inst-1", "on_message", {"events": []})
        
        # Create worker with mocked context
        harness_config = _make_harness_config()
        worker = Worker(
            harness_config=harness_config,
            registry=registry,
            providers={},
            adapters={},
            store_db=db,
            base_dir="/tmp",
        )
        
        with patch.object(worker, "_build_context") as mock_build_ctx:
            mock_ctx = MagicMock()
            mock_ctx.sent_message_count = 0
            mock_build_ctx.return_value = mock_ctx
            
            worker.run()
        
        # Check that pending token was committed
        pending_token_key = "pending_token:inst-1:main"
        pending_token = store.get(pending_token_key)
        assert pending_token is None  # Should be deleted
        
        # Check that regular token was updated
        token_key = "inst-1:main"
        committed_token = store.get(token_key)
        assert committed_token == "tok_pending"
    
    def test_pending_tokens_not_committed_on_handler_error(self):
        """If on_message handler fails, pending tokens should NOT be committed."""
        db = open_store()
        
        # Create species with failing handler
        failing_handler = MagicMock(side_effect=RuntimeError("Handler failed"))
        manifest = SpeciesManifest(
            species_id="test-species",
            entry_points=[
                EntryPoint(name="on_message", handler=failing_handler, trigger="message", schedule=None),
                EntryPoint(name="heartbeat", handler=MagicMock(), trigger=None, schedule=None),
            ],
            tools=[],
            default_files={},
            spawn=lambda ctx: None,
        )
        failing_species = MagicMock(spec=Species)
        failing_species.manifest.return_value = manifest
        
        instance = _make_instance()
        registry = Registry()
        registry.register_species(failing_species)
        registry.register_instance(instance)
        
        # Set up pending token
        from library.harness.store import NamespacedStore
        store = NamespacedStore(db, "checker")
        store.put("pending_token:inst-1:main", "tok_pending")
        
        # Set up job with on_message trigger
        queue = JobQueue(db)
        queue.enqueue("inst-1", "on_message", {"events": []})
        
        # Create worker
        harness_config = _make_harness_config()
        worker = Worker(
            harness_config=harness_config,
            registry=registry,
            providers={},
            adapters={},
            store_db=db,
            base_dir="/tmp",
        )
        
        with patch.object(worker, "_build_context") as mock_build_ctx:
            mock_ctx = MagicMock()
            mock_ctx.sent_message_count = 0
            mock_build_ctx.return_value = mock_ctx
            
            worker.run()  # Should handle the error gracefully
        
        # Pending token should still exist (not committed due to failure)
        pending_token_key = "pending_token:inst-1:main"
        pending_token = store.get(pending_token_key)
        assert pending_token == "tok_pending"
        
        # Regular token should NOT be updated
        token_key = "inst-1:main"
        committed_token = store.get(token_key)
        assert committed_token is None
