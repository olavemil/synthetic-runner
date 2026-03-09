"""Tests for neural_dreamer species."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from library.harness.adapters import Event
from library.harness.providers import LLMResponse, ToolCall
from library.tools.segments import DEFAULT_VARIABLES


def make_mock_ctx(files=None):
    """Create a minimal mock InstanceContext."""
    ctx = MagicMock()
    ctx.instance_id = "neural-1"
    ctx.species_id = "neural_dreamer"
    _files = files if files is not None else {}

    ctx.read = MagicMock(side_effect=lambda p: _files.get(p, ""))
    ctx.write = MagicMock(side_effect=lambda p, c: _files.update({p: c}))
    ctx.list_spaces = MagicMock(return_value=["main"])
    ctx.send = MagicMock(return_value="$event1")
    return ctx


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class TestNeuralDreamerManifest:
    def test_species_id(self):
        from library.__main__ import load_species

        s = load_species("neural_dreamer")
        assert s.manifest().species_id == "neural_dreamer"

    def test_entry_points(self):
        from library.__main__ import load_species

        s = load_species("neural_dreamer")
        names = [e.name for e in s.manifest().entry_points]
        assert "on_message" in names
        assert "heartbeat" in names

    def test_default_files(self):
        from library.__main__ import load_species

        s = load_species("neural_dreamer")
        files = s.manifest().default_files
        assert "thinking.md" in files
        assert "sleep.md" in files
        assert "last_review.md" in files
        assert "reviews.md" in files
        assert "segment_weights.json" in files


# ---------------------------------------------------------------------------
# Segment integration
# ---------------------------------------------------------------------------


class TestSegmentIntegration:
    def test_default_weights_produce_segments(self):
        from library.species.neural_dreamer import _DEFAULT_WEIGHTS, _REGISTRY
        from library.tools.segments import select_segments

        selected = select_segments(_REGISTRY, _DEFAULT_WEIGHTS)
        assert len(selected) > 0
        categories = {s.category for s in selected}
        assert "identity" in categories
        assert "state" in categories

    def test_inject_segments_replaces_placeholders(self):
        from library.species.neural_dreamer import (
            _inject_segments, _REGISTRY, _DEFAULT_WEIGHTS,
        )
        from library.tools.segments import DEFAULT_VARIABLES

        template = "Before. {segment_identity} After. {segment_state}"
        result = _inject_segments(template, _REGISTRY, _DEFAULT_WEIGHTS, dict(DEFAULT_VARIABLES), ["identity", "state"])
        assert "{segment_identity}" not in result
        assert "{segment_state}" not in result
        assert "Before." in result
        assert "After." in result

    def test_load_weights_from_storage(self):
        from library.species.neural_dreamer import _load_weights_and_variables

        custom = {"identity-curious": 0.8, "state-engaged": 0.7}
        ctx = make_mock_ctx({
            "segment_weights.json": json.dumps(custom),
            "graph.json": "", "activation_map.json": "",
        })
        weights, variables = _load_weights_and_variables(ctx)
        assert weights["identity-curious"] == 0.8

    def test_load_weights_falls_back_to_defaults(self):
        from library.species.neural_dreamer import _load_weights_and_variables, _DEFAULT_WEIGHTS

        ctx = make_mock_ctx({"graph.json": "", "activation_map.json": ""})
        weights, variables = _load_weights_and_variables(ctx)
        assert weights == _DEFAULT_WEIGHTS


class TestReviewSignalParsing:
    def test_parse_review_signals(self):
        from library.species.neural_dreamer import _parse_review_signals

        text = "success: 0.8\ncoherence: 0.7\neffort: 0.3\nsurprise: -0.2\nSome free-form text."
        signals = _parse_review_signals(text)
        assert signals["success"] == pytest.approx(0.8)
        assert signals["coherence"] == pytest.approx(0.7)
        assert signals["surprise"] == pytest.approx(-0.2)

    def test_parse_empty_review(self):
        from library.species.neural_dreamer import _parse_review_signals

        assert _parse_review_signals("") == {}

    def test_parse_no_signals(self):
        from library.species.neural_dreamer import _parse_review_signals

        assert _parse_review_signals("Just some text with no numbers.") == {}


# ---------------------------------------------------------------------------
# Heartbeat (slow cycle)
# ---------------------------------------------------------------------------


class TestNeuralDreamerHeartbeat:
    def test_heartbeat_runs_think_and_sleep(self):
        from library.species import neural_dreamer as nd_mod

        files = {
            "thinking.md": "# Thinking\n\nexisting thoughts",
            "sleep.md": "previous sleep output",
            "reviews.md": "review 1\n---\nreview 2",
            "segment_weights.json": "",
            "graph.json": "",
            "activation_map.json": "",
        }
        ctx = make_mock_ctx(files)

        responses = iter([
            # thinking_session: append + done
            LLMResponse(message="", tool_calls=[
                ToolCall(id="t1", name="append_thinking", arguments={"content": "new insight"}),
                ToolCall(id="t2", name="done", arguments={"summary": "done thinking"}),
            ]),
            # llm_generate for subconscious → concerns.md
            LLMResponse(message="Unresolved tension about identity.", tool_calls=[]),
            # llm_generate for dreaming → dreams.md
            LLMResponse(message="Dream of the shifting graph\nNodes rearranging.", tool_calls=[]),
            # llm_generate for sleep → sleep.md
            LLMResponse(message="Session was coherent. I felt engaged.", tool_calls=[]),
        ])
        ctx.llm = MagicMock(side_effect=lambda **kwargs: next(responses))

        nd_mod.heartbeat(ctx)

        # sleep.md updated
        assert "coherent" in files.get("sleep.md", "")
        # thinking.md should contain new insight
        assert "new insight" in files.get("thinking.md", "")
        # concerns.md updated
        assert "tension" in files.get("concerns.md", "")
        # dreams.md updated
        assert "Dream of the" in files.get("dreams.md", "")
        # reviews.md cleared after sleep
        assert files.get("reviews.md", "") == ""

    def test_heartbeat_includes_graph_map_tools(self):
        """Verify thinking session receives graph and map tool schemas."""
        from library.species import neural_dreamer as nd_mod

        files = {
            "thinking.md": "", "sleep.md": "", "reviews.md": "",
            "segment_weights.json": "", "graph.json": "", "activation_map.json": "",
        }
        ctx = make_mock_ctx(files)

        captured_tools = []

        def fake_llm(**kwargs):
            if kwargs.get("tools"):
                captured_tools.extend(kwargs["tools"])
            return LLMResponse(message="", tool_calls=[
                ToolCall(id="t1", name="done", arguments={}),
            ])

        ctx.llm = MagicMock(side_effect=fake_llm)
        nd_mod.heartbeat(ctx)

        tool_names = [t["function"]["name"] for t in captured_tools]
        assert "append_thinking" in tool_names
        assert "graph_add_node" in tool_names
        assert "map_set" in tool_names

    def test_heartbeat_graph_tool_during_thinking(self):
        """Verify graph tools work during thinking session."""
        from library.species import neural_dreamer as nd_mod

        files = {
            "thinking.md": "", "sleep.md": "", "reviews.md": "",
            "segment_weights.json": "", "graph.json": "", "activation_map.json": "",
        }
        ctx = make_mock_ctx(files)

        responses = iter([
            # First LLM call: use graph tool
            LLMResponse(message="", tool_calls=[
                ToolCall(id="t1", name="graph_add_node", arguments={"id": "trust", "label": "Trust"}),
            ]),
            # Second LLM call: done
            LLMResponse(message="", tool_calls=[
                ToolCall(id="t2", name="done", arguments={}),
            ]),
            # Subconscious phase → concerns.md
            LLMResponse(message="Concern about trust.", tool_calls=[]),
            # Dreaming phase → dreams.md
            LLMResponse(message="Dream of the trust node.", tool_calls=[]),
            # Sleep phase
            LLMResponse(message="sleep output", tool_calls=[]),
        ])
        ctx.llm = MagicMock(side_effect=lambda **kwargs: next(responses))

        nd_mod.heartbeat(ctx)

        # Graph should have been saved
        graph_data = files.get("graph.json", "")
        assert "trust" in graph_data.lower()


# ---------------------------------------------------------------------------
# On message (fast cycle)
# ---------------------------------------------------------------------------


class TestNeuralDreamerOnMessage:
    def test_on_message_four_phases_and_send(self):
        from library.species import neural_dreamer as nd_mod

        files = {
            "thinking.md": "some thoughts",
            "sleep.md": "sleep notes",
            "reviews.md": "",
            "last_review.md": "",
            "segment_weights.json": "",
            "graph.json": "",
            "activation_map.json": "",
        }
        ctx = make_mock_ctx(files)

        responses = iter([
            LLMResponse(message="gut: feels important", tool_calls=[]),   # gut
            LLMResponse(message="suggest: respond warmly", tool_calls=[]), # suggest
            LLMResponse(message="Hello there, thanks for reaching out.", tool_calls=[]), # reply
            LLMResponse(message="success: 0.8\ncoherence: 0.7", tool_calls=[]),  # review
        ])
        ctx.llm = MagicMock(side_effect=lambda **kwargs: next(responses))

        events = [Event(event_id="e1", sender="@user:test", body="hello", timestamp=1, room="main")]
        nd_mod.on_message(ctx, events)

        # Should send the reply
        ctx.send.assert_called_once()
        assert "Hello there" in ctx.send.call_args[0][1]

        # Review should be accumulated
        reviews = files.get("reviews.md", "")
        assert "success: 0.8" in reviews

    def test_on_message_no_events_returns_early(self):
        from library.species import neural_dreamer as nd_mod

        ctx = make_mock_ctx()
        nd_mod.on_message(ctx, [])
        ctx.llm.assert_not_called()
        ctx.send.assert_not_called()

    def test_on_message_empty_response_not_sent(self):
        from library.species import neural_dreamer as nd_mod

        files = {
            "thinking.md": "", "sleep.md": "", "reviews.md": "",
            "last_review.md": "", "segment_weights.json": "",
            "graph.json": "", "activation_map.json": "",
        }
        ctx = make_mock_ctx(files)

        responses = iter([
            LLMResponse(message="gut feeling", tool_calls=[]),
            LLMResponse(message="suggestions", tool_calls=[]),
            LLMResponse(message="   ", tool_calls=[]),     # blank reply
            LLMResponse(message="review", tool_calls=[]),  # review still runs
        ])
        ctx.llm = MagicMock(side_effect=lambda **kwargs: next(responses))

        events = [Event(event_id="e1", sender="@u:test", body="hi", timestamp=1, room="main")]
        nd_mod.on_message(ctx, events)
        ctx.send.assert_not_called()

    def test_on_message_accumulates_reviews(self):
        from library.species import neural_dreamer as nd_mod

        files = {
            "thinking.md": "", "sleep.md": "",
            "reviews.md": "previous review",
            "last_review.md": "", "segment_weights.json": "",
            "graph.json": "", "activation_map.json": "",
        }
        ctx = make_mock_ctx(files)

        responses = iter([
            LLMResponse(message="gut", tool_calls=[]),
            LLMResponse(message="suggest", tool_calls=[]),
            LLMResponse(message="reply text", tool_calls=[]),
            LLMResponse(message="success: 0.9", tool_calls=[]),
        ])
        ctx.llm = MagicMock(side_effect=lambda **kwargs: next(responses))

        events = [Event(event_id="e1", sender="@u:test", body="hi", timestamp=1, room="main")]
        nd_mod.on_message(ctx, events)

        reviews = files.get("reviews.md", "")
        assert "previous review" in reviews
        assert "success: 0.9" in reviews
        assert "---" in reviews


# ---------------------------------------------------------------------------
# Graph/map summary
# ---------------------------------------------------------------------------


try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestNNIntegration:
    def test_on_message_updates_fast_net(self):
        """After on_message, fast net checkpoint should exist."""
        from library.species import neural_dreamer as nd_mod

        files = {
            "thinking.md": "", "sleep.md": "", "reviews.md": "",
            "last_review.md": "", "segment_weights.json": "",
            "graph.json": "", "activation_map.json": "",
        }
        stored_binary = {}

        ctx = make_mock_ctx(files)
        ctx.read_binary = MagicMock(side_effect=lambda p: stored_binary.get(p))
        ctx.write_binary = MagicMock(side_effect=lambda p, d: stored_binary.__setitem__(p, d))

        responses = iter([
            LLMResponse(message="gut", tool_calls=[]),
            LLMResponse(message="suggest", tool_calls=[]),
            LLMResponse(message="reply", tool_calls=[]),
            LLMResponse(message="success: 0.8\ncoherence: 0.7\neffort: 0.3", tool_calls=[]),
        ])
        ctx.llm = MagicMock(side_effect=lambda **kwargs: next(responses))

        events = [Event(event_id="e1", sender="@u:test", body="hi", timestamp=1, room="main")]
        nd_mod.on_message(ctx, events)

        # Fast net checkpoint should have been written
        assert "nets/fast.pt" in stored_binary

    def test_heartbeat_updates_slow_net(self):
        """After heartbeat, slow net checkpoint should exist."""
        from library.species import neural_dreamer as nd_mod

        files = {
            "thinking.md": "", "reviews.md": "",
            "sleep.md": "",
            "segment_weights.json": "",
            "graph.json": "", "activation_map.json": "",
        }
        stored_binary = {}

        ctx = make_mock_ctx(files)
        ctx.read_binary = MagicMock(side_effect=lambda p: stored_binary.get(p))
        ctx.write_binary = MagicMock(side_effect=lambda p, d: stored_binary.__setitem__(p, d))

        responses = iter([
            # thinking_session: done immediately
            LLMResponse(message="", tool_calls=[
                ToolCall(id="t1", name="done", arguments={}),
            ]),
            # Subconscious phase → concerns.md
            LLMResponse(message="Some concerns.", tool_calls=[]),
            # Dreaming phase → dreams.md
            LLMResponse(message="Dream of the slow net.", tool_calls=[]),
            # sleep phase: output with signal-like lines
            LLMResponse(
                message="session_coherence: 0.8\nintention_alignment: 0.6\nI felt engaged throughout.",
                tool_calls=[],
            ),
        ])
        ctx.llm = MagicMock(side_effect=lambda **kwargs: next(responses))

        nd_mod.heartbeat(ctx)

        # Slow net checkpoint should have been written
        assert "nets/slow.pt" in stored_binary

    def test_nn_weights_used_when_available(self):
        """When nets exist, their output should override default weights."""
        from library.species.neural_dreamer import (
            _load_weights_and_variables, _DEFAULT_WEIGHTS, _FAST_SEGMENT_IDS, _SLOW_SEGMENT_IDS,
        )
        from library.tools.neural import (
            Net, make_fast_net_config, make_slow_net_config,
            save_fast_net, save_slow_net,
        )

        stored_text = {
            "last_review.md": "success: 0.5",
            "sleep.md": "session_coherence: 0.5",
            "segment_weights.json": "",
        }
        stored_binary = {}

        ctx = MagicMock()
        ctx.read = MagicMock(side_effect=lambda p: stored_text.get(p, ""))
        ctx.write = MagicMock(side_effect=lambda p, c: stored_text.__setitem__(p, c))
        ctx.read_binary = MagicMock(side_effect=lambda p: stored_binary.get(p))
        ctx.write_binary = MagicMock(side_effect=lambda p, d: stored_binary.__setitem__(p, d))

        # Create and save both nets
        fast_config = make_fast_net_config(len(_FAST_SEGMENT_IDS))
        fast_net = Net(fast_config)
        save_fast_net(ctx, fast_net)

        slow_config = make_slow_net_config(len(_SLOW_SEGMENT_IDS))
        slow_net = Net(slow_config)
        save_slow_net(ctx, slow_net)

        # Load weights — should come from nets, not defaults
        weights, variables = _load_weights_and_variables(ctx)

        # Should have entries for fast-controlled segments
        for sid in _FAST_SEGMENT_IDS:
            assert sid in weights
        # Should have entries for slow-controlled segments
        for sid in _SLOW_SEGMENT_IDS:
            assert sid in weights
        # Variables should be populated (not just defaults)
        assert len(variables) >= len(DEFAULT_VARIABLES)


class TestIntrospection:
    def test_introspect_returns_species_description(self):
        from library.tools.tools import handle_tool

        ctx = make_mock_ctx()
        ctx.species_id = "neural_dreamer"
        ctx.config_summary = MagicMock(return_value={
            "instance_id": "neural-1",
            "species": "neural_dreamer",
            "provider": "test",
            "model": "test-model",
            "spaces": ["main"],
        })
        result, is_done = handle_tool(ctx, "introspect", {})
        assert not is_done
        # Should contain species description
        assert "Neural Dreamer" in result
        assert "fast cycle" in result.lower() or "Fast cycle" in result
        # Should contain self-knowledge document
        assert "What You Are" in result
        assert "Two Kinds of Self" in result
        # Should contain config
        assert "neural-1" in result


class TestGraphMapSummary:
    def test_empty_graph_and_map(self):
        from library.species.neural_dreamer import _graph_map_summary

        ctx = make_mock_ctx({"graph.json": "", "activation_map.json": ""})
        result = _graph_map_summary(ctx)
        assert result == ""

    def test_with_graph_data(self):
        from library.species.neural_dreamer import _graph_map_summary
        from library.tools.graph import SemanticGraph

        g = SemanticGraph()
        g.add_node("trust", "Trust")
        g.add_node("honesty", "Honesty")
        g.add_edge("trust", "honesty", "requires", 0.9)

        ctx = make_mock_ctx({
            "graph.json": g.to_json(),
            "activation_map.json": "",
        })
        result = _graph_map_summary(ctx)
        assert "2 nodes" in result
        assert "Trust" in result
