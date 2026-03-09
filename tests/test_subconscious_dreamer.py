"""Tests for subconscious_dreamer species and new pipeline stages."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, call

import pytest

from library.harness.adapters import Event
from library.harness.providers import LLMResponse, ToolCall


def make_mock_ctx(files=None):
    """Create a minimal mock InstanceContext."""
    ctx = MagicMock()
    ctx.instance_id = "dreamer-1"
    ctx.species_id = "subconscious_dreamer"
    _files = files if files is not None else {}

    ctx.read = MagicMock(side_effect=lambda p: _files.get(p, ""))
    ctx.write = MagicMock(side_effect=lambda p, c: _files.update({p: c}))
    ctx.list_spaces = MagicMock(return_value=["main"])
    ctx.send = MagicMock(return_value="$event1")
    return ctx


# ---------------------------------------------------------------------------
# llm_generate
# ---------------------------------------------------------------------------


class TestLlmGenerate:
    def test_single_call_returns_stripped_text(self):
        from library.tools.patterns import llm_generate

        ctx = make_mock_ctx()
        ctx.llm = MagicMock(return_value=LLMResponse(message="  hello world  ", tool_calls=[]))
        result = llm_generate(ctx, system="be concise", content="say hi")
        assert result == "hello world"
        ctx.llm.assert_called_once()
        call_kwargs = ctx.llm.call_args[1]
        assert call_kwargs["caller"] == "llm_generate"

    def test_context_prepended_to_content(self):
        from library.tools.patterns import llm_generate

        ctx = make_mock_ctx()
        captured = {}

        def fake_llm(messages, **kwargs):
            captured["content"] = messages[0]["content"]
            return LLMResponse(message="ok", tool_calls=[])

        ctx.llm = MagicMock(side_effect=fake_llm)
        llm_generate(ctx, system="sys", content="content", context="extra context")
        assert captured["content"].startswith("extra context")
        assert "content" in captured["content"]

    def test_no_context_sends_content_only(self):
        from library.tools.patterns import llm_generate

        ctx = make_mock_ctx()
        captured = {}

        def fake_llm(messages, **kwargs):
            captured["content"] = messages[0]["content"]
            return LLMResponse(message="ok", tool_calls=[])

        ctx.llm = MagicMock(side_effect=fake_llm)
        llm_generate(ctx, system="sys", content="just content")
        assert captured["content"] == "just content"


# ---------------------------------------------------------------------------
# thinking_session
# ---------------------------------------------------------------------------


class TestThinkingSession:
    def test_append_thinking_writes_to_file(self):
        from library.tools.patterns import thinking_session

        files = {"thinking.md": "# Old thoughts"}
        ctx = make_mock_ctx(files)

        def fake_llm(messages, **kwargs):
            # First call: append + done
            if len(messages) == 1:
                return LLMResponse(
                    message="",
                    tool_calls=[
                        ToolCall(id="tc1", name="append_thinking", arguments={"content": "new thought"}),
                        ToolCall(id="tc2", name="done", arguments={"summary": "done"}),
                    ],
                )
            return LLMResponse(message="finished", tool_calls=[])

        ctx.llm = MagicMock(side_effect=fake_llm)
        thinking_session(ctx, system="think freely", initial_message="here is context")

        written = files.get("thinking.md", "")
        assert "Old thoughts" in written
        assert "new thought" in written

    def test_replace_thinking_overwrites_file(self):
        from library.tools.patterns import thinking_session

        files = {"thinking.md": "old stuff"}
        ctx = make_mock_ctx(files)

        def fake_llm(messages, **kwargs):
            if len(messages) == 1:
                return LLMResponse(
                    message="",
                    tool_calls=[
                        ToolCall(id="tc1", name="replace_thinking", arguments={"content": "fresh start"}),
                        ToolCall(id="tc2", name="done", arguments={}),
                    ],
                )
            return LLMResponse(message="done", tool_calls=[])

        ctx.llm = MagicMock(side_effect=fake_llm)
        thinking_session(ctx, system="sys", initial_message="msg")
        assert files["thinking.md"] == "fresh start"

    def test_done_stops_loop(self):
        from library.tools.patterns import thinking_session

        ctx = make_mock_ctx()
        call_count = 0

        def fake_llm(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            return LLMResponse(
                message="",
                tool_calls=[ToolCall(id="tc1", name="done", arguments={})],
            )

        ctx.llm = MagicMock(side_effect=fake_llm)
        thinking_session(ctx, system="sys", initial_message="msg")
        # Should stop after first done, not keep looping
        assert call_count == 1

    def test_no_tool_calls_stops_loop(self):
        from library.tools.patterns import thinking_session

        ctx = make_mock_ctx()
        ctx.llm = MagicMock(return_value=LLMResponse(message="no tools", tool_calls=[]))
        thinking_session(ctx, system="sys", initial_message="msg")
        ctx.llm.assert_called_once()


# ---------------------------------------------------------------------------
# format_context
# ---------------------------------------------------------------------------


class TestFormatContext:
    def test_builds_labeled_block(self):
        from library.tools.patterns import format_context

        files = {"thinking.md": "thoughts here", "dreams.md": "dream content"}
        ctx = make_mock_ctx(files)
        result = format_context(ctx, [["thinking.md", "Your Thoughts"], ["dreams.md", "Your Dreams"]])
        assert "## Your Thoughts" in result
        assert "thoughts here" in result
        assert "## Your Dreams" in result
        assert "dream content" in result

    def test_skips_empty_files(self):
        from library.tools.patterns import format_context

        files = {"thinking.md": "something", "dreams.md": ""}
        ctx = make_mock_ctx(files)
        result = format_context(ctx, [["thinking.md", "Thoughts"], ["dreams.md", "Dreams"]])
        assert "## Thoughts" in result
        assert "## Dreams" not in result

    def test_returns_empty_string_when_all_empty(self):
        from library.tools.patterns import format_context

        ctx = make_mock_ctx()
        result = format_context(ctx, [["thinking.md", "Thoughts"]])
        assert result == ""


# ---------------------------------------------------------------------------
# pipeline: file: source and initial_state
# ---------------------------------------------------------------------------


class TestPipelineFileSource:
    def test_file_source_loads_prompt_file(self, tmp_path):
        from library.tools.pipeline import resolve_input

        prompt_file = tmp_path / "prompts" / "test.md"
        prompt_file.parent.mkdir()
        prompt_file.write_text("Hello {instance_id}!")

        ctx = make_mock_ctx()
        state = {"_species_dir": str(tmp_path)}
        result = resolve_input(ctx, "file:prompts/test.md", state)
        assert result == "Hello dreamer-1!"

    def test_file_source_missing_file(self, tmp_path):
        from library.tools.pipeline import resolve_input

        ctx = make_mock_ctx()
        state = {"_species_dir": str(tmp_path)}
        result = resolve_input(ctx, "file:missing.md", state)
        assert "not found" in result

    def test_file_source_no_species_dir(self):
        from library.tools.pipeline import resolve_input

        ctx = make_mock_ctx()
        result = resolve_input(ctx, "file:prompts/test.md", {})
        assert "not found" in result


class TestPipelineInitialState:
    def test_initial_state_seeds_pipeline(self):
        from library.tools.pipeline import run_pipeline, load_pipeline

        pipeline = load_pipeline("""
steps:
  - stage: llm_generate
    inputs:
      system: pipeline.my_system
      content: pipeline.my_content
    outputs:
      result: pipeline.response
""")

        ctx = make_mock_ctx()
        ctx.llm = MagicMock(return_value=LLMResponse(message="result text", tool_calls=[]))

        state = run_pipeline(
            ctx,
            pipeline["steps"],
            initial_state={"my_system": "be helpful", "my_content": "say hi"},
        )
        assert state["response"] == "result text"


# ---------------------------------------------------------------------------
# subconscious_dreamer species
# ---------------------------------------------------------------------------


class TestSubconsciousDreamerManifest:
    def test_species_id(self):
        from library.__main__ import load_species

        s = load_species("subconscious_dreamer")
        assert s.manifest().species_id == "subconscious_dreamer"

    def test_entry_points(self):
        from library.__main__ import load_species

        s = load_species("subconscious_dreamer")
        names = [e.name for e in s.manifest().entry_points]
        assert "on_message" in names
        assert "heartbeat" in names

    def test_default_files(self):
        from library.__main__ import load_species

        s = load_species("subconscious_dreamer")
        files = s.manifest().default_files
        assert "thinking.md" in files
        assert "dreams.md" in files
        assert "concerns.md" in files


class TestSubconsciousDreamerHeartbeat:
    def test_heartbeat_runs_three_phases(self):
        from library.species import subconscious_dreamer as sd_mod

        ctx = make_mock_ctx({
            "thinking.md": "# Thinking\n\nsome thoughts",
            "dreams.md": "vague images",
            "concerns.md": "a few worries",
        })

        responses = iter([
            # thinking_session: append + done
            LLMResponse(message="", tool_calls=[
                ToolCall(id="t1", name="append_thinking", arguments={"content": "new thought"}),
                ToolCall(id="t2", name="done", arguments={"summary": "done"}),
            ]),
            # format_context subconscious_sections — no LLM call
            # llm_generate for subconscious → concerns
            LLMResponse(message="worry about X", tool_calls=[]),
            # format_context dreaming_sections — no LLM call
            # llm_generate for dreaming → dreams
            LLMResponse(message="dream of flying", tool_calls=[]),
        ])
        ctx.llm = MagicMock(side_effect=lambda **kwargs: next(responses))

        sd_mod.heartbeat(ctx)

        # concerns.md and dreams.md should have been updated
        write_calls = {call[0][0]: call[0][1] for call in ctx.write.call_args_list}
        assert "concerns.md" in write_calls
        assert write_calls["concerns.md"] == "worry about X"
        assert "dreams.md" in write_calls
        assert write_calls["dreams.md"] == "dream of flying"


class TestSubconsciousDreamerOnMessage:
    def test_on_message_sends_reply(self):
        from library.species import subconscious_dreamer as sd_mod

        ctx = make_mock_ctx({
            "thinking.md": "some thoughts",
            "dreams.md": "some dreams",
            "concerns.md": "some concerns",
        })

        responses = iter([
            LLMResponse(message="- feels urgent", tool_calls=[]),   # intuition
            LLMResponse(message="proceed carefully", tool_calls=[]), # worry
            LLMResponse(message="Here is my reply.", tool_calls=[]), # action
        ])
        ctx.llm = MagicMock(side_effect=lambda **kwargs: next(responses))

        events = [Event(event_id="e1", sender="@user:test", body="hello", timestamp=1, room="main")]
        sd_mod.on_message(ctx, events)

        ctx.send.assert_called_once()
        assert ctx.send.call_args[0][1] == "Here is my reply."

    def test_on_message_no_events_returns_early(self):
        from library.species import subconscious_dreamer as sd_mod

        ctx = make_mock_ctx()
        sd_mod.on_message(ctx, [])
        ctx.llm.assert_not_called()
        ctx.send.assert_not_called()

    def test_on_message_empty_response_not_sent(self):
        from library.species import subconscious_dreamer as sd_mod

        ctx = make_mock_ctx({
            "thinking.md": "thoughts",
            "dreams.md": "dreams",
            "concerns.md": "concerns",
        })

        responses = iter([
            LLMResponse(message="impressions", tool_calls=[]),
            LLMResponse(message="approach", tool_calls=[]),
            LLMResponse(message="   ", tool_calls=[]),  # blank response
        ])
        ctx.llm = MagicMock(side_effect=lambda **kwargs: next(responses))

        events = [Event(event_id="e1", sender="@u:test", body="hi", timestamp=1, room="main")]
        sd_mod.on_message(ctx, events)
        ctx.send.assert_not_called()
