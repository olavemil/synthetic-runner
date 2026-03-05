"""Tests for messaging adapters."""

import json

import pytest

from symbiosis.harness.adapters import Event
from symbiosis.harness.adapters.local_file import LocalFileAdapter


class TestLocalFileAdapter:
    def test_send_and_poll(self, tmp_path):
        adapter = LocalFileAdapter(tmp_path / "messages")

        event_id = adapter.send("test-room", "Hello, world!")
        assert event_id

        events, token = adapter.poll("test-room")
        assert len(events) == 1
        assert events[0].body == "Hello, world!"
        assert events[0].room == "test-room"

    def test_poll_with_since(self, tmp_path):
        adapter = LocalFileAdapter(tmp_path / "messages")

        adapter.send("room", "msg1")
        events1, token1 = adapter.poll("room")
        assert len(events1) == 1

        adapter.send("room", "msg2")
        events2, token2 = adapter.poll("room", since_token=token1)
        assert len(events2) == 1
        assert events2[0].body == "msg2"

    def test_poll_empty(self, tmp_path):
        adapter = LocalFileAdapter(tmp_path / "messages")
        events, token = adapter.poll("empty-room")
        assert events == []
        assert token == "0"

    def test_space_context(self, tmp_path):
        adapter = LocalFileAdapter(tmp_path / "messages")
        ctx = adapter.get_space_context("my-room")
        assert ctx["name"] == "my-room"
        assert ctx["room_id"] == "my-room"

    def test_multiple_spaces(self, tmp_path):
        adapter = LocalFileAdapter(tmp_path / "messages")

        adapter.send("room-a", "msg-a")
        adapter.send("room-b", "msg-b")

        events_a, _ = adapter.poll("room-a")
        events_b, _ = adapter.poll("room-b")

        assert len(events_a) == 1
        assert events_a[0].body == "msg-a"
        assert len(events_b) == 1
        assert events_b[0].body == "msg-b"

    def test_send_with_reply_to(self, tmp_path):
        adapter = LocalFileAdapter(tmp_path / "messages")
        event_id = adapter.send("room", "reply", reply_to="$original")
        assert event_id

        # The reply_to is stored but not exposed in Event
        events, _ = adapter.poll("room")
        assert len(events) == 1


class TestEvent:
    def test_event_dataclass(self):
        evt = Event(
            event_id="$123",
            sender="@alice:matrix.org",
            body="hello",
            timestamp=1000,
            room="!room:matrix.org",
        )
        assert evt.event_id == "$123"
        assert evt.sender == "@alice:matrix.org"
        assert evt.body == "hello"
        assert evt.room == "!room:matrix.org"

    def test_event_default_room(self):
        evt = Event(event_id="$1", sender="alice", body="hi", timestamp=0)
        assert evt.room is None
