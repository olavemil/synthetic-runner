"""Tests for symbiosis.toolkit.deliberate."""

from __future__ import annotations

import json
from types import SimpleNamespace

from symbiosis.toolkit.identity import Identity
from symbiosis.toolkit.deliberate import (
    generate_with_identity,
    multi_generate,
    multi_vote,
    deliberate,
    recompose,
    think_with_context,
)


def _voice(name: str, model: str = "m", personality: str = "p") -> Identity:
    return Identity(name=name, model=model, personality=personality)


def _colony_member(name: str, dims: dict | None = None) -> Identity:
    from symbiosis.toolkit.identity import AXIS_NAMES
    return Identity(
        name=name,
        dims=dims or {n: 0.1 for n in AXIS_NAMES},
    )


class DummyCtx:
    def __init__(self, llm_fn=None):
        self._llm_fn = llm_fn

    def llm(self, messages, **kwargs):
        if self._llm_fn:
            return self._llm_fn(messages, **kwargs)
        return SimpleNamespace(message="default response")


class TestGenerateWithIdentity:
    def test_returns_llm_message(self):
        ctx = DummyCtx(lambda msgs, **kw: SimpleNamespace(message="hello world"))
        identity = _voice("Aria")
        result = generate_with_identity(ctx, identity, "what do you think?")
        assert result == "hello world"

    def test_model_override_passed_to_llm(self):
        captured = {}

        def llm_fn(msgs, **kw):
            captured.update(kw)
            return SimpleNamespace(message="ok")

        ctx = DummyCtx(llm_fn)
        identity = _voice("Aria", model="default-model")
        generate_with_identity(ctx, identity, "prompt", model="anthropic/claude-haiku")
        assert captured.get("model") == "claude-haiku"
        assert captured.get("provider") == "anthropic"

    def test_identity_model_used_when_no_override(self):
        captured = {}

        def llm_fn(msgs, **kw):
            captured.update(kw)
            return SimpleNamespace(message="ok")

        ctx = DummyCtx(llm_fn)
        identity = Identity(name="Aria", model="my-model", provider="openai")
        generate_with_identity(ctx, identity, "prompt")
        assert captured.get("model") == "my-model"
        assert captured.get("provider") == "openai"

    def test_context_included_in_system(self):
        captured = {}

        def llm_fn(msgs, **kw):
            captured.update(kw)
            return SimpleNamespace(message="ok")

        ctx = DummyCtx(llm_fn)
        identity = _voice("Aria")
        generate_with_identity(ctx, identity, "prompt", context="some context")
        assert "some context" in captured.get("system", "")


class TestMultiGenerate:
    def test_returns_dict_keyed_by_name(self):
        call_count = [0]

        def llm_fn(msgs, **kw):
            call_count[0] += 1
            return SimpleNamespace(message=f"response {call_count[0]}")

        ctx = DummyCtx(llm_fn)
        voices = [_voice("Aria"), _voice("Sable"), _voice("Lune")]
        result = multi_generate(ctx, voices, "prompt")
        assert set(result.keys()) == {"Aria", "Sable", "Lune"}
        assert call_count[0] == 3


class TestMultiVote:
    def _voices(self):
        return [_voice("Aria"), _voice("Sable"), _voice("Lune")]

    def test_returns_ranking_per_voter(self):
        def llm_fn(msgs, **kw):
            return SimpleNamespace(message='{"ranking": ["Sable", "Lune"]}')

        ctx = DummyCtx(llm_fn)
        voices = self._voices()
        candidates = {"Aria": "text a", "Sable": "text b", "Lune": "text c"}
        votes = multi_vote(ctx, voices, candidates, "prompt")
        assert set(votes.keys()) == {"Aria", "Sable", "Lune"}

    def test_exclude_own_removes_self(self):
        """With exclude_own=True, no voter can rank their own candidate."""
        def llm_fn(msgs, **kw):
            # Try to vote for self by returning full list
            return SimpleNamespace(message='{"ranking": ["Aria", "Sable", "Lune"]}')

        ctx = DummyCtx(llm_fn)
        voices = self._voices()
        candidates = {"Aria": "text a", "Sable": "text b", "Lune": "text c"}
        votes = multi_vote(ctx, voices, candidates, "prompt", exclude_own=True)

        # Aria's ranking should not contain "Aria"
        assert "Aria" not in votes["Aria"]
        assert "Sable" not in votes["Sable"]
        assert "Lune" not in votes["Lune"]

    def test_top_n_truncates_ranking(self):
        def llm_fn(msgs, **kw):
            return SimpleNamespace(message='{"ranking": ["Sable", "Aria", "Lune"]}')

        ctx = DummyCtx(llm_fn)
        voices = self._voices()
        candidates = {"Aria": "text a", "Sable": "text b", "Lune": "text c"}
        votes = multi_vote(ctx, voices, candidates, "prompt", top_n=1)
        for ranking in votes.values():
            assert len(ranking) <= 1

    def test_invalid_json_falls_back_gracefully(self):
        def llm_fn(msgs, **kw):
            return SimpleNamespace(message="not json at all")

        ctx = DummyCtx(llm_fn)
        voices = self._voices()
        candidates = {"Aria": "text a", "Sable": "text b"}
        votes = multi_vote(ctx, voices, candidates, "prompt")
        # Should not raise; all candidates included in each ranking
        for ranking in votes.values():
            assert set(ranking) == {"Aria", "Sable"}


class TestDeliberate:
    def _voices(self):
        return [_voice("Aria"), _voice("Sable"), _voice("Lune")]

    def test_returns_expected_fields(self):
        def llm_fn(msgs, **kw):
            caller = kw.get("caller", "")
            if "vote" in caller:
                return SimpleNamespace(message='{"ranking": ["Aria"]}')
            return SimpleNamespace(message="generated text")

        ctx = DummyCtx(llm_fn)
        result = deliberate(ctx, self._voices(), "prompt")
        for field in ("winner_member", "winner_message", "scores", "candidate_count",
                      "vote_count", "is_tie", "has_consensus", "candidates", "votes"):
            assert field in result

    def test_subset_generates_only_subset(self):
        call_log = []

        def llm_fn(msgs, **kw):
            call_log.append(kw.get("caller", ""))
            if "vote" in kw.get("caller", ""):
                return SimpleNamespace(message='{"ranking": ["Aria"]}')
            return SimpleNamespace(message="text")

        ctx = DummyCtx(llm_fn)
        voices = self._voices()
        # Only Aria generates
        result = deliberate(ctx, voices, "prompt", subset=[voices[0]])
        assert "Aria" in result["candidates"]
        assert "Sable" not in result["candidates"]
        assert "Lune" not in result["candidates"]
        # All voices still vote
        assert len(result["votes"]) == 3

    def test_consensus_threshold_applied(self):
        def llm_fn(msgs, **kw):
            if "vote" in kw.get("caller", ""):
                return SimpleNamespace(message='{"ranking": ["Aria", "Sable"]}')
            return SimpleNamespace(message="text")

        ctx = DummyCtx(llm_fn)
        voices = self._voices()
        # threshold=0.0 → always consensus
        result = deliberate(ctx, voices, "prompt", consensus_threshold=0.0)
        assert result["has_consensus"] is True

    def test_empty_identities_returns_empty(self):
        ctx = DummyCtx()
        result = deliberate(ctx, [], "prompt")
        assert result["winner_member"] == ""
        assert result["candidates"] == {}


class TestRecompose:
    def test_with_all_candidates_includes_snippets(self):
        captured = []

        def llm_fn(msgs, **kw):
            captured.append(msgs[0]["content"])
            return SimpleNamespace(message="recomposed")

        ctx = DummyCtx(llm_fn)
        identity = _voice("Colony")
        result = recompose(
            ctx, identity, "winner text",
            all_candidates={"Aria": "aria text", "Sable": "sable text"},
        )
        assert result == "recomposed"
        assert "winner text" in captured[0]
        assert "aria text" in captured[0]

    def test_without_all_candidates_rewrites_voice(self):
        captured = []

        def llm_fn(msgs, **kw):
            captured.append(msgs[0]["content"])
            return SimpleNamespace(message="reworded")

        ctx = DummyCtx(llm_fn)
        identity = _voice("Aria")
        result = recompose(ctx, identity, "the winning text")
        assert result == "reworded"
        assert "the winning text" in captured[0]


class TestThinkWithContext:
    def test_returns_thought(self):
        ctx = DummyCtx(lambda msgs, **kw: SimpleNamespace(message="deep thought"))
        identity = _voice("Aria")
        result = think_with_context(ctx, identity)
        assert result == "deep thought"

    def test_others_thinking_included_in_prompt(self):
        captured = []

        def llm_fn(msgs, **kw):
            captured.append(msgs[0]["content"])
            return SimpleNamespace(message="thought")

        ctx = DummyCtx(llm_fn)
        identity = _voice("Aria")
        think_with_context(ctx, identity, others_thinking={"Sable": "sable thought"})
        assert "sable thought" in captured[0]

    def test_voice_memory_included_in_context(self):
        captured = []

        def llm_fn(msgs, **kw):
            captured.append(kw.get("system", ""))
            return SimpleNamespace(message="thought")

        ctx = DummyCtx(llm_fn)
        identity = _voice("Aria")
        think_with_context(
            ctx, identity,
            voice_memory={"subconscious": "hidden feeling"},
        )
        assert "hidden feeling" in captured[0]
