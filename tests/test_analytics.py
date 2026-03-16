"""Tests for the analytics client and harness instrumentation."""

from __future__ import annotations

import json
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock, patch

import pytest

from library.harness.analytics import AnalyticsClient, _pseudonymize
from library.harness.config import AnalyticsConfig, HarnessConfig
from library.harness.context import InstanceContext
from library.harness.storage import NamespacedStorage
from library.harness.store import open_store
from library.harness.mailbox import Mailbox
from library.harness.config import InstanceConfig, MessagingConfig, SpaceMapping
from library.harness.providers import LLMResponse, ToolCall


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ctx(tmp_path, analytics=None, adapter=None):
    storage = NamespacedStorage(tmp_path / "instances", "test-1")
    provider = MagicMock()
    store_db = open_store()
    mailbox = Mailbox(tmp_path / "instances", "test-1")

    space_map = {"main": "!room:test"}
    msg_config = MessagingConfig(
        adapter="test-adapter",
        entity_id="@bot:test",
        access_token="test-token",
        spaces=[SpaceMapping(name="main", handle="!room:test")],
    )
    instance_config = InstanceConfig(
        instance_id="test-1",
        species="draum",
        provider="test-provider",
        model="test-model",
        messaging=msg_config,
    )
    return InstanceContext(
        instance_id="test-1",
        species_id="draum",
        storage=storage,
        provider=provider,
        default_model="test-model",
        adapter=adapter,
        space_map=space_map,
        store_db=store_db,
        mailbox=mailbox,
        instance_config=instance_config,
        analytics=analytics,
    )


# ---------------------------------------------------------------------------
# Unit tests: AnalyticsClient
# ---------------------------------------------------------------------------

class TestPseudonymize:
    def test_consistent(self):
        assert _pseudonymize("user-123") == _pseudonymize("user-123")

    def test_different_inputs_produce_different_outputs(self):
        assert _pseudonymize("user-123") != _pseudonymize("user-456")

    def test_prefix(self):
        assert _pseudonymize("x").startswith("anon_")

    def test_length(self):
        result = _pseudonymize("x")
        assert len(result) == len("anon_") + 12


class TestAnalyticsClientErrorSilencing:
    """The client must never raise even when the server is unreachable."""

    def test_track_does_not_raise_when_server_offline(self, tmp_path):
        client = AnalyticsClient(
            base_url="http://localhost:19999",  # nothing listening here
            instance_id="inst-1",
            session_id="sess-1",
            timeout=0.1,
        )
        # Should not raise
        client.track("test_event", {"foo": "bar"})
        # Give the daemon thread a moment to attempt the request
        time.sleep(0.2)

    def test_track_does_not_raise_on_invalid_url(self):
        client = AnalyticsClient(
            base_url="http://256.256.256.256",  # invalid
            instance_id="inst-1",
            session_id="sess-1",
            timeout=0.1,
        )
        client.track("test_event")
        time.sleep(0.2)


class TestAnalyticsClientSendsEvents:
    """Integration test using a local HTTP server."""

    def test_event_is_posted(self):
        received: list[dict] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                received.append(json.loads(body))
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args):
                pass  # suppress output

        server = HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        t = threading.Thread(target=server.handle_request, daemon=True)
        t.start()

        client = AnalyticsClient(
            base_url=f"http://127.0.0.1:{port}",
            instance_id="inst-1",
            session_id="sess-1",
        )
        client.track("test_event", {"key": "value"})

        t.join(timeout=3)
        server.server_close()

        assert len(received) == 1
        evt = received[0]
        assert evt["event_name"] == "test_event"
        assert evt["properties"] == {"key": "value"}
        assert evt["analytics_user_id"].startswith("anon_")
        assert evt["analytics_session_id"].startswith("anon_")
        assert "client_timestamp" in evt

    def test_user_and_session_are_pseudonymized(self):
        received: list[dict] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                received.append(json.loads(self.rfile.read(length)))
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        t = threading.Thread(target=server.handle_request, daemon=True)
        t.start()

        client = AnalyticsClient(
            base_url=f"http://127.0.0.1:{port}",
            instance_id="real-instance-id",
            session_id="real-session-id",
        )
        client.track("ping")
        t.join(timeout=3)
        server.server_close()

        assert len(received) == 1
        evt = received[0]
        assert "real-instance-id" not in evt["analytics_user_id"]
        assert "real-session-id" not in evt["analytics_session_id"]


# ---------------------------------------------------------------------------
# Unit tests: InstanceContext instrumentation
# ---------------------------------------------------------------------------

class TestContextTracksFileOps:
    def test_write_tracked(self, tmp_path):
        analytics = MagicMock()
        ctx = make_ctx(tmp_path, analytics=analytics)

        ctx.write("note.md", "hello world")

        analytics.track.assert_called_once_with(
            "file_written", {"path": "note.md", "length": 11}
        )

    def test_read_tracked(self, tmp_path):
        analytics = MagicMock()
        ctx = make_ctx(tmp_path, analytics=analytics)
        ctx._storage.write("note.md", "hi")
        analytics.track.reset_mock()

        ctx.read("note.md")

        analytics.track.assert_called_once_with("file_read", {"path": "note.md"})

    def test_write_binary_tracked(self, tmp_path):
        analytics = MagicMock()
        ctx = make_ctx(tmp_path, analytics=analytics)

        ctx.write_binary("model.pt", b"\x00\x01\x02")

        analytics.track.assert_called_once_with(
            "file_written", {"path": "model.pt", "size": 3, "binary": True}
        )

    def test_read_binary_tracked(self, tmp_path):
        analytics = MagicMock()
        ctx = make_ctx(tmp_path, analytics=analytics)
        ctx._storage.write_binary("model.pt", b"\x00")
        analytics.track.reset_mock()

        ctx.read_binary("model.pt")

        analytics.track.assert_called_once_with(
            "file_read", {"path": "model.pt", "binary": True}
        )

    def test_no_analytics_no_error(self, tmp_path):
        ctx = make_ctx(tmp_path, analytics=None)
        ctx.write("x.md", "y")
        ctx.read("x.md")
        ctx.write_binary("b.bin", b"\x00")
        ctx.read_binary("b.bin")


class TestContextTracksLLM:
    def _make_response(self, tool_calls=None, finish_reason="stop"):
        return LLMResponse(
            message="ok",
            tool_calls=tool_calls or [],
            finish_reason=finish_reason,
            usage={"input_tokens": 10, "output_tokens": 5},
        )

    def test_llm_call_tracked(self, tmp_path):
        analytics = MagicMock()
        ctx = make_ctx(tmp_path, analytics=analytics)
        ctx._provider.create.return_value = self._make_response()

        ctx.llm([{"role": "user", "content": "hi"}], caller="test_caller")

        analytics.track.assert_called_once()
        call_args = analytics.track.call_args
        assert call_args[0][0] == "llm_called"
        props = call_args[0][1]
        assert props["caller"] == "test_caller"
        assert props["finish_reason"] == "stop"
        assert props["prompt_tokens"] == 10
        assert props["completion_tokens"] == 5
        assert props["tool_call_count"] == 0

    def test_tool_calls_tracked(self, tmp_path):
        analytics = MagicMock()
        ctx = make_ctx(tmp_path, analytics=analytics)
        tool_calls = [
            ToolCall(id="1", name="read_file", arguments={"path": "x.md"}),
            ToolCall(id="2", name="write_file", arguments={"path": "y.md", "content": "z"}),
        ]
        ctx._provider.create.return_value = self._make_response(
            tool_calls=tool_calls, finish_reason="tool_calls"
        )

        ctx.llm([{"role": "user", "content": "hi"}], caller="tc")

        calls = analytics.track.call_args_list
        event_names = [c[0][0] for c in calls]
        assert "llm_called" in event_names
        assert event_names.count("tool_called") == 2

        tool_names = [c[0][1]["tool_name"] for c in calls if c[0][0] == "tool_called"]
        assert "read_file" in tool_names
        assert "write_file" in tool_names

    def test_no_analytics_no_error(self, tmp_path):
        ctx = make_ctx(tmp_path, analytics=None)
        ctx._provider.create.return_value = self._make_response()
        ctx.llm([{"role": "user", "content": "hi"}])


class TestContextTracksMessaging:
    def test_send_tracked(self, tmp_path):
        analytics = MagicMock()
        adapter = MagicMock()
        adapter.send.return_value = "$event1"
        ctx = make_ctx(tmp_path, analytics=analytics, adapter=adapter)

        ctx.send("main", "hello")

        analytics.track.assert_called_once_with(
            "message_sent", {"space": "main", "has_reply_to": False}
        )

    def test_send_with_reply_to_tracked(self, tmp_path):
        analytics = MagicMock()
        adapter = MagicMock()
        adapter.send.return_value = "$event2"
        ctx = make_ctx(tmp_path, analytics=analytics, adapter=adapter)

        ctx.send("main", "reply", reply_to="$orig")

        analytics.track.assert_called_once_with(
            "message_sent", {"space": "main", "has_reply_to": True}
        )

    def test_poll_tracked(self, tmp_path):
        from library.harness.adapters import Event as AdapterEvent
        analytics = MagicMock()
        adapter = MagicMock()
        events = [
            AdapterEvent(event_id="e1", sender="@a:test", body="hi", timestamp=0, room="!r:test"),
            AdapterEvent(event_id="e2", sender="@b:test", body="yo", timestamp=1, room="!r:test"),
        ]
        adapter.poll.return_value = (events, "next_token")
        ctx = make_ctx(tmp_path, analytics=analytics, adapter=adapter)

        ctx.poll("main")

        analytics.track.assert_called_once_with(
            "messages_polled", {"space": "main", "event_count": 2}
        )

    def test_blocked_send_not_tracked(self, tmp_path):
        analytics = MagicMock()
        adapter = MagicMock()
        ctx = make_ctx(tmp_path, analytics=analytics, adapter=adapter)
        ctx.configure_send_policy(allow_send=False, reason="test")

        ctx.send("main", "hello")

        analytics.track.assert_not_called()

    def test_no_analytics_no_error(self, tmp_path):
        adapter = MagicMock()
        adapter.send.return_value = "$e"
        adapter.poll.return_value = ([], "tok")
        ctx = make_ctx(tmp_path, analytics=None, adapter=adapter)
        ctx.send("main", "hi")
        ctx.poll("main")


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestAnalyticsConfig:
    def test_default_base_url(self):
        cfg = AnalyticsConfig()
        assert cfg.base_url == "http://localhost:4000"

    def test_harness_config_analytics_none_by_default(self):
        cfg = HarnessConfig()
        assert cfg.analytics is None

    def test_load_harness_config_with_analytics(self, tmp_path):
        from library.harness.config import load_harness_config

        harness_yaml = tmp_path / "config" / "harness.yaml"
        harness_yaml.parent.mkdir()
        harness_yaml.write_text(
            "providers: []\nadapters: []\nanalytics:\n  base_url: http://localhost:9000\n"
        )

        cfg = load_harness_config(harness_yaml)
        assert cfg.analytics is not None
        assert cfg.analytics.base_url == "http://localhost:9000"

    def test_load_harness_config_without_analytics(self, tmp_path):
        from library.harness.config import load_harness_config

        harness_yaml = tmp_path / "config" / "harness.yaml"
        harness_yaml.parent.mkdir()
        harness_yaml.write_text("providers: []\nadapters: []\n")

        cfg = load_harness_config(harness_yaml)
        assert cfg.analytics is None


# ---------------------------------------------------------------------------
# Tests: ctx.track() public method
# ---------------------------------------------------------------------------

class TestContextTrackMethod:
    def test_track_delegates_to_analytics(self, tmp_path):
        analytics = MagicMock()
        ctx = make_ctx(tmp_path, analytics=analytics)
        ctx.track("custom_event", {"x": 1})
        analytics.track.assert_called_once_with("custom_event", {"x": 1})

    def test_track_no_analytics_no_error(self, tmp_path):
        ctx = make_ctx(tmp_path, analytics=None)
        ctx.track("custom_event")  # Must not raise


# ---------------------------------------------------------------------------
# Tests: analytics debug logging
# ---------------------------------------------------------------------------

class TestAnalyticsLogging:
    def test_logs_debug_on_success(self, caplog):
        import logging

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        t = threading.Thread(target=server.handle_request, daemon=True)
        t.start()

        client = AnalyticsClient(
            base_url=f"http://127.0.0.1:{port}",
            instance_id="inst",
            session_id="sess",
        )
        with caplog.at_level(logging.DEBUG, logger="library.harness.analytics"):
            client.track("ping")
            t.join(timeout=3)
            time.sleep(0.1)

        server.server_close()
        assert any("ping" in r.message and "sent" in r.message for r in caplog.records)

    def test_logs_debug_on_failure(self, caplog):
        import logging

        client = AnalyticsClient(
            base_url="http://localhost:19998",
            instance_id="inst",
            session_id="sess",
            timeout=0.05,
        )
        with caplog.at_level(logging.DEBUG, logger="library.harness.analytics"):
            client.track("ping")
            time.sleep(0.3)

        assert any("not delivered" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Tests: worker phase tracking
# ---------------------------------------------------------------------------

def _make_species_for_analytics(handler, species_id="test"):
    """Build a minimal Species mock with heartbeat + on_message entry points."""
    from library.harness.registry import Registry
    from library.species import Species, SpeciesManifest, EntryPoint

    manifest = SpeciesManifest(
        species_id=species_id,
        entry_points=[
            EntryPoint(name="on_message", handler=handler, trigger="message", schedule=None),
            EntryPoint(name="heartbeat", handler=handler, trigger=None, schedule="0 * * * *"),
        ],
        tools=[],
        default_files={},
        spawn=lambda ctx: None,
    )
    species = MagicMock(spec=Species)
    species.manifest.return_value = manifest
    return species


def _build_analytics_worker(tmp_path, store_db, handler, species_id="test"):
    from library.harness.config import AnalyticsConfig, HarnessConfig
    from library.harness.worker import Worker
    from library.harness.registry import Registry

    analytics_config = AnalyticsConfig(base_url="http://localhost:1")
    harness_config = HarnessConfig(analytics=analytics_config)

    registry = Registry()
    registry.register_species(_make_species_for_analytics(handler, species_id))
    registry.register_instance(InstanceConfig(
        instance_id="test-inst",
        species=species_id,
        provider="fake",
        model="fake-model",
    ))

    provider_mock = MagicMock()
    provider_mock.create.return_value = MagicMock(
        message="ok", tool_calls=[], finish_reason="stop", usage={}
    )

    return Worker(
        harness_config=harness_config,
        registry=registry,
        providers={"fake": provider_mock},
        adapters={},
        store_db=store_db,
        base_dir=tmp_path,
    ), harness_config


class TestWorkerPhaseTracking:
    """Verify phase_started / phase_completed events via the worker."""

    def test_phase_events_emitted(self, tmp_path):
        from library.harness.store import open_store
        from library.harness.jobqueue import JobQueue
        from library.harness import analytics as analytics_mod

        tracked: list[tuple] = []

        class FakeAnalytics:
            def track(self, name, props=None):
                tracked.append((name, props or {}))

        store_db = open_store()
        handler = MagicMock()
        worker, _ = _build_analytics_worker(tmp_path, store_db, handler)

        with patch.object(analytics_mod, "AnalyticsClient", return_value=FakeAnalytics()):
            queue = JobQueue(store_db)
            queue.enqueue("test-inst", "heartbeat", {"heartbeat": True})
            worker.run()

        event_names = [name for name, _ in tracked]
        assert "phase_started" in event_names
        assert "phase_completed" in event_names

        started = next(p for n, p in tracked if n == "phase_started")
        completed = next(p for n, p in tracked if n == "phase_completed")

        assert started["phase"] == "heartbeat"
        assert "pending_messages" in started

        assert completed["phase"] == "heartbeat"
        assert completed["success"] is True
        assert "messages_sent" in completed

    def test_phase_completed_success_false_on_handler_error(self, tmp_path):
        from library.harness.store import open_store
        from library.harness.jobqueue import JobQueue
        from library.harness import analytics as analytics_mod

        tracked: list[tuple] = []

        class FakeAnalytics:
            def track(self, name, props=None):
                tracked.append((name, props or {}))

        def failing_handler(ctx, **kw):
            raise RuntimeError("boom")

        store_db = open_store()
        worker, _ = _build_analytics_worker(tmp_path, store_db, failing_handler)

        with patch.object(analytics_mod, "AnalyticsClient", return_value=FakeAnalytics()):
            queue = JobQueue(store_db)
            queue.enqueue("test-inst", "heartbeat", {"heartbeat": True})
            worker.run()  # Should not raise despite handler error

        completed = next((p for n, p in tracked if n == "phase_completed"), None)
        assert completed is not None
        assert completed["success"] is False
