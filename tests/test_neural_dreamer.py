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
        weights, variables, shared = _load_weights_and_variables(ctx)
        assert weights["identity-curious"] == 0.8
        assert shared.get("_nn_available") is False

    def test_load_weights_falls_back_to_defaults(self):
        from library.species.neural_dreamer import _load_weights_and_variables, _DEFAULT_WEIGHTS

        ctx = make_mock_ctx({"graph.json": "", "activation_map.json": ""})
        weights, variables, shared = _load_weights_and_variables(ctx)
        assert weights == _DEFAULT_WEIGHTS
        assert shared.get("_nn_available") is False


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
            # distill_memory in think phase
            LLMResponse(message="Compressed memory for thinking", tool_calls=[]),
            # thinking_session: append + done
            LLMResponse(message="", tool_calls=[
                ToolCall(id="t1", name="append_thinking", arguments={"content": "new insight"}),
                ToolCall(id="t2", name="done", arguments={"summary": "done thinking"}),
            ]),
            # organize_session: done immediately
            LLMResponse(message="", tool_calls=[
                ToolCall(id="t3", name="done", arguments={}),
            ]),
            # llm_generate for subconscious → concerns_and_ideas.md
            LLMResponse(message="Unresolved tension about identity.", tool_calls=[]),
            # distill_memory in dream phase  
            LLMResponse(message="Compressed memory for dreaming", tool_calls=[]),
            # llm_generate for dreaming → dreams.md
            LLMResponse(message="Dream of the shifting graph\nNodes rearranging.", tool_calls=[]),
            # distill_memory in create phase
            LLMResponse(message="Compressed memory for creation", tool_calls=[]),
            # create phase: done immediately
            LLMResponse(message="", tool_calls=[
                ToolCall(id="t4", name="done", arguments={}),
            ]),
            # llm_generate for sleep → sleep.md
            LLMResponse(message="Session was coherent. I felt engaged.", tool_calls=[]),
        ])
        ctx.llm = MagicMock(side_effect=lambda **kwargs: next(responses))

        nd_mod.heartbeat(ctx)

        # sleep.md updated
        assert "coherent" in files.get("sleep.md", "")
        # thinking.md should contain new insight
        assert "new insight" in files.get("thinking.md", "")
        # concerns_and_ideas.md updated
        assert "tension" in files.get("concerns_and_ideas.md", "")
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
        call_count = [0]

        def fake_llm(**kwargs):
            call_count[0] += 1
            if kwargs.get("tools"):
                captured_tools.extend(kwargs["tools"])
            # tool-session phases return done; generate phases return text
            if kwargs.get("tools"):
                return LLMResponse(message="", tool_calls=[
                    ToolCall(id=f"t{call_count[0]}", name="done", arguments={}),
                ])
            return LLMResponse(message="phase output", tool_calls=[])

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
            # distill_memory in think phase
            LLMResponse(message="Compressed memory", tool_calls=[]),
            # First LLM call: use graph tool
            LLMResponse(message="", tool_calls=[
                ToolCall(id="t1", name="graph_add_node", arguments={"id": "trust", "label": "Trust"}),
            ]),
            # Second LLM call: done (think)
            LLMResponse(message="", tool_calls=[
                ToolCall(id="t2", name="done", arguments={}),
            ]),
            # Organize phase: done immediately
            LLMResponse(message="", tool_calls=[
                ToolCall(id="t3", name="done", arguments={}),
            ]),
            # Subconscious phase → concerns_and_ideas.md
            LLMResponse(message="Concern about trust.", tool_calls=[]),
            # distill_memory in dream phase
            LLMResponse(message="Compressed memory for dream", tool_calls=[]),
            # Dreaming phase → dreams.md
            LLMResponse(message="Dream of the trust node.", tool_calls=[]),
            # distill_memory in create phase
            LLMResponse(message="Compressed memory for create", tool_calls=[]),
            # Create phase: done immediately
            LLMResponse(message="", tool_calls=[
                ToolCall(id="t4", name="done", arguments={}),
            ]),
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
            LLMResponse(message="distilled messages summary", tool_calls=[]),  # distill_messages
            LLMResponse(message="distilled memory digest", tool_calls=[]),     # distill_memory
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
            LLMResponse(message="distilled messages", tool_calls=[]),  # distill_messages
            # No distill_memory call since thinking.md is empty
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
            LLMResponse(message="distilled messages", tool_calls=[]),  # distill_messages
            # No distill_memory call since thinking.md is empty
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

    def test_on_message_reply_suppressed(self):
        """When probabilistic returns False, reply should not be sent but review should still run."""
        from library.species import neural_dreamer as nd_mod
        from unittest.mock import patch

        files = {
            "thinking.md": "", "sleep.md": "", "reviews.md": "",
            "last_review.md": "", "segment_weights.json": "",
            "graph.json": "", "activation_map.json": "",
        }
        ctx = make_mock_ctx(files)

        responses = iter([
            LLMResponse(message="distilled messages", tool_calls=[]),  # distill_messages
            # No distill_memory call since thinking.md is empty
            LLMResponse(message="gut feeling\nreply_value: 0.3", tool_calls=[]),
            LLMResponse(message="suggestions", tool_calls=[]),
            LLMResponse(message="actual reply text", tool_calls=[]),
            LLMResponse(message="success: 0.7\ncoherence: 0.6", tool_calls=[]),
        ])
        ctx.llm = MagicMock(side_effect=lambda **kwargs: next(responses))

        events = [Event(event_id="e1", sender="@u:test", body="hi", timestamp=1, room="main")]

        # Force _nn_available=True and probabilistic to return False
        with patch.object(nd_mod, "_load_weights_and_variables") as mock_load, \
             patch.object(nd_mod, "probabilistic", return_value=False):
            from library.species.neural_dreamer import _DEFAULT_WEIGHTS
            mock_load.return_value = (
                dict(_DEFAULT_WEIGHTS),
                dict(DEFAULT_VARIABLES),
                {"_nn_available": True, "reply_willingness": 0.3},
            )
            nd_mod.on_message(ctx, events)

        # Reply should NOT be sent
        ctx.send.assert_not_called()
        # But review should still be accumulated
        assert "success: 0.7" in files.get("reviews.md", "")


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
            # organize phase: done immediately
            LLMResponse(message="", tool_calls=[
                ToolCall(id="t2", name="done", arguments={}),
            ]),
            # Subconscious phase → concerns_and_ideas.md
            LLMResponse(message="Some concerns.", tool_calls=[]),
            # Dreaming phase → dreams.md
            LLMResponse(message="Dream of the slow net.", tool_calls=[]),
            # Create phase: done immediately
            LLMResponse(message="", tool_calls=[
                ToolCall(id="t3", name="done", arguments={}),
            ]),
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
        weights, variables, shared = _load_weights_and_variables(ctx)

        # Should have entries for fast-controlled segments
        for sid in _FAST_SEGMENT_IDS:
            assert sid in weights
        # Should have entries for slow-controlled segments
        for sid in _SLOW_SEGMENT_IDS:
            assert sid in weights
        # Variables should be populated (not just defaults)
        assert len(variables) >= len(DEFAULT_VARIABLES)
        # Shared params should indicate NN is available
        assert shared.get("_nn_available") is True


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


class TestBuildHeartbeatPhases:
    def test_no_nn_returns_fixed_pipeline(self):
        from library.species.neural_dreamer import _build_heartbeat_phases

        phases = _build_heartbeat_phases({"_nn_available": False}, {})
        assert phases == ["think", "organize", "subconscious", "dream", "create", "sleep"]

    def test_nn_available_always_has_think_subconscious_sleep(self):
        from library.species.neural_dreamer import _build_heartbeat_phases
        from unittest.mock import patch

        # Force all probabilistic calls to return False
        with patch("library.species.neural_dreamer.probabilistic", return_value=False):
            phases = _build_heartbeat_phases({"_nn_available": True}, {})
        # Minimal order: [think, subconscious, dream, sleep]
        assert phases == ["think", "subconscious", "dream", "sleep"]
        assert "organize" not in phases

    def test_nn_available_all_phases_enabled(self):
        from library.species.neural_dreamer import _build_heartbeat_phases
        from unittest.mock import patch

        # Force all probabilistic calls to return True
        with patch("library.species.neural_dreamer.probabilistic", return_value=True):
            phases = _build_heartbeat_phases(
                {"_nn_available": True, "processing_depth": 0.9,
                 "organization_drive": 0.8, "creative_latitude": 0.8},
                {"max_think_iterations": 3},
            )
        # Should have multiple thinks, organize, dream
        assert phases.count("think") > 1
        assert "organize" in phases
        assert "dream" in phases
        assert phases[-1] == "sleep"

    def test_max_think_iterations_respected(self):
        from library.species.neural_dreamer import _build_heartbeat_phases
        from unittest.mock import patch

        with patch("library.species.neural_dreamer.probabilistic", return_value=True):
            phases = _build_heartbeat_phases(
                {"_nn_available": True, "processing_depth": 1.0},
                {"max_think_iterations": 2},
            )
        assert phases.count("think") == 2


class TestHeartbeatOrganizePhase:
    def test_heartbeat_with_organize(self):
        """When NN enables organize phase, it should run organize tools session."""
        from library.species import neural_dreamer as nd_mod
        from unittest.mock import patch

        files = {
            "thinking.md": "# Thinking\n\nsome thoughts",
            "sleep.md": "", "reviews.md": "",
            "segment_weights.json": "", "graph.json": "", "activation_map.json": "",
        }
        ctx = make_mock_ctx(files)

        responses = iter([
            # distill_memory for think phase
            LLMResponse(message="Compressed memory", tool_calls=[]),
            # think phase
            LLMResponse(message="", tool_calls=[
                ToolCall(id="t1", name="done", arguments={}),
            ]),
            # organize phase (thinking_session with organize tools)
            LLMResponse(message="", tool_calls=[
                ToolCall(id="t2", name="done", arguments={}),
            ]),
            # subconscious
            LLMResponse(message="concerns", tool_calls=[]),
            # sleep
            LLMResponse(message="consolidated", tool_calls=[]),
        ])
        ctx.llm = MagicMock(side_effect=lambda **kwargs: next(responses))
        ctx.exists = MagicMock(return_value=False)
        ctx.list = MagicMock(return_value=[])

        # Force phases: think, organize, subconscious, sleep (no dream)
        with patch.object(nd_mod, "_build_heartbeat_phases",
                          return_value=["think", "organize", "subconscious", "sleep"]):
            nd_mod.heartbeat(ctx)

        assert "consolidated" in files.get("sleep.md", "")
        assert "concerns" in files.get("concerns_and_ideas.md", "")


class TestPostReplyExtraThinking:
    def test_extra_thinking_runs_when_nn_enabled(self):
        """Post-reply extra thinking should run when probabilistic returns True."""
        from library.species import neural_dreamer as nd_mod
        from unittest.mock import patch, call

        files = {
            "thinking.md": "existing", "sleep.md": "", "reviews.md": "",
            "last_review.md": "", "segment_weights.json": "",
            "graph.json": "", "activation_map.json": "",
        }
        ctx = make_mock_ctx(files)

        responses = iter([
            # Distillation calls
            LLMResponse(message="distilled messages", tool_calls=[]),  # distill_messages
            LLMResponse(message="distilled thinking", tool_calls=[]),  # distill_memory
            # on_message pipeline: gut, suggest, reply, review
            LLMResponse(message="gut", tool_calls=[]),
            LLMResponse(message="suggest", tool_calls=[]),
            LLMResponse(message="reply text", tool_calls=[]),
            LLMResponse(message="success: 0.8", tool_calls=[]),
            # distill_memory for extra thinking session
            LLMResponse(message="distilled memory for thinking", tool_calls=[]),
            # extra thinking session
            LLMResponse(message="", tool_calls=[
                ToolCall(id="t1", name="done", arguments={}),
            ]),
        ])
        ctx.llm = MagicMock(side_effect=lambda **kwargs: next(responses))

        events = [Event(event_id="e1", sender="@u:test", body="hi", timestamp=1, room="main")]

        # Force NN available + probabilistic always True (reply + extra think)
        with patch.object(nd_mod, "_load_weights_and_variables") as mock_load, \
             patch.object(nd_mod, "probabilistic", return_value=True):
            from library.species.neural_dreamer import _DEFAULT_WEIGHTS
            mock_load.return_value = (
                dict(_DEFAULT_WEIGHTS),
                dict(DEFAULT_VARIABLES),
                {"_nn_available": True, "reply_willingness": 0.9, "processing_depth": 0.8},
            )
            nd_mod.on_message(ctx, events)

        # Should have made 8 LLM calls (2 distillation + 4 pipeline + 1 distillation + 1 extra think)
        assert ctx.llm.call_count == 8

    def test_no_extra_thinking_without_nn(self):
        """Post-reply extra thinking should NOT run when NN is not available."""
        from library.species import neural_dreamer as nd_mod

        files = {
            "thinking.md": "", "sleep.md": "", "reviews.md": "",
            "last_review.md": "", "segment_weights.json": "",
            "graph.json": "", "activation_map.json": "",
        }
        ctx = make_mock_ctx(files)

        responses = iter([
            LLMResponse(message="distilled messages", tool_calls=[]),  # distill_messages
            # No distill_memory since thinking.md is empty
            LLMResponse(message="gut", tool_calls=[]),
            LLMResponse(message="suggest", tool_calls=[]),
            LLMResponse(message="reply", tool_calls=[]),
            LLMResponse(message="success: 0.5", tool_calls=[]),
        ])
        ctx.llm = MagicMock(side_effect=lambda **kwargs: next(responses))

        events = [Event(event_id="e1", sender="@u:test", body="hi", timestamp=1, room="main")]
        nd_mod.on_message(ctx, events)

        # Should have made 5 LLM calls (1 distill + 4 pipeline, no extra think)
        assert ctx.llm.call_count == 5


class TestSessionMetrics:
    def test_snapshot_counts_empty(self):
        from library.species.neural_dreamer import _snapshot_counts

        ctx = make_mock_ctx({"graph.json": "", "activation_map.json": "", "thinking.md": ""})
        ctx.exists = MagicMock(return_value=False)
        ctx.list = MagicMock(return_value=[])
        snap = _snapshot_counts(ctx)
        assert snap["graph_nodes"] == 0
        assert snap["graph_edges"] == 0
        assert snap["topics"] == 0
        assert snap["thinking_chars"] == 0

    def test_snapshot_counts_with_data(self):
        from library.species.neural_dreamer import _snapshot_counts
        from library.tools.graph import SemanticGraph

        g = SemanticGraph()
        g.add_node("a", "A")
        g.add_node("b", "B")
        g.add_edge("a", "b", "rel", 1.0)

        files = {
            "graph.json": g.to_json(),
            "activation_map.json": "",
            "thinking.md": "some thinking text here",
            "knowledge/concepts/_meta.md": "# concepts\n",
            "knowledge/concepts/trust.md": "Trust stuff.",
        }
        ctx = make_mock_ctx(files)
        ctx.exists = MagicMock(side_effect=lambda p: p in files and bool(files[p]))
        ctx.list = MagicMock(side_effect=lambda prefix="": [
            k for k in sorted(files.keys())
            if k.startswith(prefix) and bool(files[k])
        ])

        snap = _snapshot_counts(ctx)
        assert snap["graph_nodes"] == 2
        assert snap["graph_edges"] == 1
        assert snap["topics"] == 1
        assert snap["thinking_chars"] == len("some thinking text here")

    def test_build_session_metrics_deltas(self):
        from library.species.neural_dreamer import _build_session_metrics

        files = {
            "graph.json": "",
            "activation_map.json": "",
            "thinking.md": "shorter",
        }
        ctx = make_mock_ctx(files)
        ctx.exists = MagicMock(return_value=False)
        ctx.list = MagicMock(return_value=[])

        pre = {
            "graph_nodes": 5, "graph_edges": 10,
            "topics": 3, "thinking_chars": 500,
        }
        metrics = _build_session_metrics(ctx, pre, think_count=2)

        assert metrics.think_iterations == 2
        assert metrics.graph_nodes_added == 0  # 0 - 5 is clamped to 0
        assert metrics.topics_added == 0
        # thinking shrunk: 500 -> len("shorter") = 7
        assert metrics.thoughts_archived_chars == 500 - len("shorter")

    def test_heartbeat_logs_metrics(self):
        """Heartbeat should compute and log session metrics."""
        from library.species import neural_dreamer as nd_mod
        import logging

        files = {
            "thinking.md": "# Thinking\n", "sleep.md": "", "reviews.md": "",
            "segment_weights.json": "", "graph.json": "", "activation_map.json": "",
        }
        ctx = make_mock_ctx(files)
        ctx.exists = MagicMock(return_value=False)
        ctx.list = MagicMock(return_value=[])

        responses = iter([
            # distill_memory for think phase
            LLMResponse(message="distilled memory for think", tool_calls=[]),
            LLMResponse(message="", tool_calls=[
                ToolCall(id="t1", name="done", arguments={}),
            ]),
            # organize phase
            LLMResponse(message="", tool_calls=[
                ToolCall(id="t2", name="done", arguments={}),
            ]),
            LLMResponse(message="concerns", tool_calls=[]),
            LLMResponse(message="distilled memory for dream", tool_calls=[]),  # dream's distill
            LLMResponse(message="dream", tool_calls=[]),
            # distill_memory for create phase
            LLMResponse(message="distilled memory for create", tool_calls=[]),
            # create phase
            LLMResponse(message="", tool_calls=[
                ToolCall(id="t3", name="done", arguments={}),
            ]),
            LLMResponse(message="sleep output", tool_calls=[]),
        ])
        ctx.llm = MagicMock(side_effect=lambda **kwargs: next(responses))

        # Just verify it doesn't crash — metrics are computed and logged
        nd_mod.heartbeat(ctx)
        # 9 LLM calls made (think distill + think + organize + subconscious + dream distill + dream + create distill + create + sleep)
        assert ctx.llm.call_count == 9


class TestRateLimitedPipeline:
    def test_rate_limited_runs_gut_and_review_only(self):
        """When rate-limited, on_message should run gut + review (2 LLM calls plus distillation)."""
        from library.species import neural_dreamer as nd_mod

        files = {
            "thinking.md": "some thoughts", "sleep.md": "", "reviews.md": "",
            "last_review.md": "", "segment_weights.json": "",
            "graph.json": "", "activation_map.json": "",
        }
        ctx = make_mock_ctx(files)
        ctx._reply_rate_limited = True  # Set the rate limit flag

        responses = iter([
            LLMResponse(message="distilled messages", tool_calls=[]),  # distill_messages
            LLMResponse(message="distilled thinking", tool_calls=[]),  # distill_memory
            LLMResponse(message="gut feeling about this message", tool_calls=[]),
            LLMResponse(message="success: 0.7\ncoherence: 0.6", tool_calls=[]),
        ])
        ctx.llm = MagicMock(side_effect=lambda **kwargs: next(responses))

        events = [Event(event_id="e1", sender="@u:test", body="hi", timestamp=1, room="main")]
        nd_mod.on_message(ctx, events)

        # Should have made 4 LLM calls (2 distillation + gut + review)
        assert ctx.llm.call_count == 4
        # No reply should be sent
        ctx.send.assert_not_called()
        # Review should still be accumulated
        assert "success: 0.7" in files.get("reviews.md", "")

    def test_rate_limited_flag_must_be_true(self):
        """MagicMock attributes should not trigger rate-limited path."""
        from library.species import neural_dreamer as nd_mod

        files = {
            "thinking.md": "", "sleep.md": "", "reviews.md": "",
            "last_review.md": "", "segment_weights.json": "",
            "graph.json": "", "activation_map.json": "",
        }
        ctx = make_mock_ctx(files)
        # MagicMock has all attributes — should NOT trigger rate limit

        responses = iter([
            LLMResponse(message="distilled messages", tool_calls=[]),  # distill_messages
            # No distill_memory since thinking.md is empty
            LLMResponse(message="gut", tool_calls=[]),
            LLMResponse(message="suggest", tool_calls=[]),
            LLMResponse(message="reply", tool_calls=[]),
            LLMResponse(message="success: 0.5", tool_calls=[]),
        ])
        ctx.llm = MagicMock(side_effect=lambda **kwargs: next(responses))

        events = [Event(event_id="e1", sender="@u:test", body="hi", timestamp=1, room="main")]
        nd_mod.on_message(ctx, events)

        # Should have made 5 LLM calls (1 distillation + 4 pipeline, not abbreviated)
        assert ctx.llm.call_count == 5


class TestOrganizeToolDispatch:
    def test_handle_tool_dispatches_organize(self):
        """handle_tool should dispatch organize_ prefixed tools."""
        from library.tools.tools import handle_tool

        files = {}
        ctx = make_mock_ctx(files)
        ctx.exists = MagicMock(return_value=False)
        ctx.list = MagicMock(return_value=[])

        result, is_done = handle_tool(ctx, "organize_list_categories", {})
        assert not is_done
        # Should have seeded defaults
        assert "concepts" in result

    def test_handle_tool_organize_write_topic(self):
        """handle_tool should dispatch organize_write_topic."""
        from library.tools.tools import handle_tool

        files = {}
        ctx = make_mock_ctx(files)
        ctx.exists = MagicMock(side_effect=lambda p: p in files and bool(files[p]))
        ctx.list = MagicMock(side_effect=lambda prefix="": [
            k for k in sorted(files.keys())
            if k.startswith(prefix) and bool(files[k])
        ])

        result, is_done = handle_tool(ctx, "organize_write_topic", {
            "category": "concepts",
            "topic": "trust",
            "content": "Trust is earned.",
        })
        assert not is_done
        assert "Created" in result


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
