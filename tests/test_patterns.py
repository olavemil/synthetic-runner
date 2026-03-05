"""Tests for pattern library with mock ctx."""

import json
from unittest.mock import MagicMock, patch, call

import pytest

from symbiosis.harness.adapters import Event
from symbiosis.harness.providers import LLMResponse, ToolCall


def make_mock_ctx(memory_files=None):
    """Create a mock InstanceContext."""
    ctx = MagicMock()
    ctx.instance_id = "test-1"
    ctx.species_id = "draum"

    files = memory_files or {}

    def mock_read(path):
        return files.get(path, "")

    def mock_write(path, content):
        files[path] = content

    ctx.read = MagicMock(side_effect=mock_read)
    ctx.write = MagicMock(side_effect=mock_write)
    ctx.list = MagicMock(return_value=list(files.keys()))
    ctx.exists = MagicMock(side_effect=lambda p: p in files)

    return ctx


class TestGutResponse:
    def test_returns_structured_guidance(self):
        from symbiosis.toolkit.patterns import gut_response

        ctx = make_mock_ctx({"thinking.md": "some thoughts", "intentions.md": "be helpful"})

        guidance = {"should_respond": True, "urgency": "medium", "brief": "greeting", "suggested_approach": "respond warmly", "rooms_to_respond": ["main"]}
        ctx.llm = MagicMock(return_value=LLMResponse(
            message=json.dumps(guidance)
        ))

        events = [Event(event_id="1", sender="alice", body="hello", timestamp=1000)]
        result = gut_response(ctx, events)

        assert result["should_respond"] is True
        assert result["urgency"] == "medium"
        ctx.llm.assert_called_once()

    def test_fallback_on_invalid_json(self):
        from symbiosis.toolkit.patterns import gut_response

        ctx = make_mock_ctx()
        ctx.llm = MagicMock(return_value=LLMResponse(message="not json"))

        events = [Event(event_id="1", sender="alice", body="hello", timestamp=1000)]
        result = gut_response(ctx, events)

        assert result["should_respond"] is True
        assert "not json" in result["suggested_approach"]


class TestComposeResponse:
    def test_returns_message(self):
        from symbiosis.toolkit.patterns import compose_response

        ctx = make_mock_ctx()
        ctx.llm = MagicMock(return_value=LLMResponse(message="Hi there!"))

        result = compose_response(ctx, {"guidance": "be friendly"})
        assert result == "Hi there!"

    def test_returns_none_for_null(self):
        from symbiosis.toolkit.patterns import compose_response

        ctx = make_mock_ctx()
        ctx.llm = MagicMock(return_value=LLMResponse(message="NULL"))

        result = compose_response(ctx, {"guidance": "nothing"})
        assert result is None


class TestSubconscious:
    def test_writes_subconscious(self):
        from symbiosis.toolkit.patterns import run_subconscious

        files = {"thinking.md": "thoughts"}
        ctx = make_mock_ctx(files)
        ctx.llm = MagicMock(return_value=LLMResponse(message="I feel reflective"))

        run_subconscious(ctx, "reactive")

        ctx.write.assert_called_with("subconscious.md", "I feel reflective")


class TestReact:
    def test_writes_intentions(self):
        from symbiosis.toolkit.patterns import run_react

        files = {"subconscious.md": "feeling good", "intentions.md": "old intentions"}
        ctx = make_mock_ctx(files)
        ctx.llm = MagicMock(return_value=LLMResponse(message="new intentions"))

        run_react(ctx, "reactive")

        ctx.write.assert_called_with("intentions.md", "new intentions")

    def test_skips_without_subconscious(self):
        from symbiosis.toolkit.patterns import run_react

        ctx = make_mock_ctx({})
        run_react(ctx, "reactive")
        ctx.llm.assert_not_called()


class TestUpdateRelationships:
    def test_updates_for_senders(self):
        from symbiosis.toolkit.patterns import update_relationships

        ctx = make_mock_ctx({})
        ctx.llm = MagicMock(return_value=LLMResponse(message="Alice is friendly"))

        events = [Event(event_id="1", sender="alice", body="hi", timestamp=1000)]
        update_relationships(ctx, "reactive", events)

        ctx.write.assert_called_with("relationships/alice.md", "Alice is friendly")

    def test_skips_without_events(self):
        from symbiosis.toolkit.patterns import update_relationships

        ctx = make_mock_ctx()
        update_relationships(ctx, "reactive", [])
        ctx.llm.assert_not_called()


class TestDistillMemory:
    def test_returns_digest(self):
        from symbiosis.toolkit.patterns import distill_memory

        files = {"thinking.md": "many thoughts", "project.md": "project info"}
        ctx = make_mock_ctx(files)
        ctx.llm = MagicMock(return_value=LLMResponse(message="compressed digest"))

        result = distill_memory(ctx)
        assert result == "compressed digest"

    def test_empty_memory(self):
        from symbiosis.toolkit.patterns import distill_memory

        ctx = make_mock_ctx({})
        result = distill_memory(ctx)
        assert result == ""


class TestRunSession:
    def test_session_with_tool_calls(self):
        from symbiosis.toolkit.patterns import run_session

        files = {"test.md": "content"}
        ctx = make_mock_ctx(files)

        # First call: tool call, second call: done
        ctx.llm = MagicMock(side_effect=[
            LLMResponse(
                message="",
                tool_calls=[ToolCall(id="1", name="read_file", arguments={"path": "test.md"})],
                finish_reason="tool_calls",
            ),
            LLMResponse(
                message="",
                tool_calls=[ToolCall(id="2", name="done", arguments={"summary": "done"})],
                finish_reason="tool_calls",
            ),
        ])

        result = run_session(ctx, "system prompt", "do something", tools=[])
        assert result is False  # no send_message called

    def test_session_ends_on_no_tools(self):
        from symbiosis.toolkit.patterns import run_session

        ctx = make_mock_ctx()
        ctx.llm = MagicMock(return_value=LLMResponse(message="thinking..."))

        result = run_session(ctx, "system", "go", tools=[])
        assert result is False
