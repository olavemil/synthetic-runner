"""Tests for the Hecate toolkit."""

from __future__ import annotations

from types import SimpleNamespace

from symbiosis.toolkit.hecate import (
    Voice,
    HecateConfig,
    load_config,
    vote_response,
)


class DummyCtx:
    def __init__(self, hecate_cfg=None, vote_choice: int | None = None):
        self._hecate_cfg = hecate_cfg or {}
        self._vote_choice = vote_choice

    def config(self, key: str):
        if key == "hecate":
            return self._hecate_cfg
        return None

    def llm(self, messages, **kwargs):
        if self._vote_choice is not None:
            return SimpleNamespace(message=f'{{"choice": {self._vote_choice}}}')
        return SimpleNamespace(message="")


class TestLoadConfig:
    def test_three_voice_config_parsed(self):
        ctx = DummyCtx(
            hecate_cfg={
                "voices": [
                    {"name": "Aria", "model": "anthropic/claude-opus-4-6", "personality": "Analytical"},
                    {"name": "Sable", "model": "openai/gpt-4", "personality": "Bold"},
                    {"name": "Lune", "model": "local-model", "personality": "Poetic"},
                ],
                "thinking_iterations": 3,
                "voice_space": "chat",
            }
        )
        cfg = load_config(ctx)
        assert len(cfg.voices) == 3
        assert cfg.voices[0].name == "Aria"
        assert cfg.voices[0].provider == "anthropic"
        assert cfg.voices[0].model == "claude-opus-4-6"
        assert cfg.voices[1].name == "Sable"
        assert cfg.voices[1].provider == "openai"
        assert cfg.voices[2].provider is None  # no '/' in model string
        assert cfg.thinking_iterations == 3
        assert cfg.voice_space == "chat"

    def test_defaults_when_empty(self):
        ctx = DummyCtx()
        cfg = load_config(ctx)
        assert cfg.voices == []
        assert cfg.thinking_iterations == 2
        assert cfg.voice_space == "main"

    def test_thinking_iterations_minimum_one(self):
        ctx = DummyCtx(hecate_cfg={"thinking_iterations": 0})
        cfg = load_config(ctx)
        assert cfg.thinking_iterations == 1


class TestVoteResponse:
    def _voices(self) -> list[Voice]:
        return [
            Voice(name="Aria", model="m", personality="p"),
            Voice(name="Sable", model="m", personality="p"),
            Voice(name="Lune", model="m", personality="p"),
        ]

    def _suggestions(self) -> list[dict]:
        return [
            {"text": "reply A", "argument": "arg A"},
            {"text": "reply B", "argument": "arg B"},
            {"text": "reply C", "argument": "arg C"},
        ]

    def test_never_returns_own_index(self):
        suggestions = self._suggestions()
        voices = self._voices()
        # Try all three voices as voters
        for my_idx in range(3):
            ctx = DummyCtx(vote_choice=my_idx)  # LLM "tries" to vote for own index
            result = vote_response(ctx, voices[my_idx], suggestions, my_idx)
            assert result != my_idx, f"Voice {my_idx} voted for itself"

    def test_returns_valid_other_index(self):
        suggestions = self._suggestions()
        voices = self._voices()
        for my_idx in range(3):
            ctx = DummyCtx(vote_choice=(my_idx + 1) % 3)
            result = vote_response(ctx, voices[my_idx], suggestions, my_idx)
            assert result != my_idx
            assert 0 <= result < 3

    def test_falls_back_on_invalid_llm_response(self):
        """If LLM returns garbage, falls back to first other index."""
        ctx = DummyCtx()
        ctx._vote_choice = None

        def broken_llm(messages, **kwargs):
            return SimpleNamespace(message="not json at all")

        ctx.llm = broken_llm
        voice = Voice(name="Aria", model="m", personality="p")
        suggestions = self._suggestions()
        result = vote_response(ctx, voice, suggestions, my_idx=0)
        assert result != 0  # should not be own index
        assert result in [1, 2]


class TestVoteTally:
    """Test the vote tally logic (inline, as the tally is done in the species)."""

    def test_two_vote_winner(self):
        vote_results = [1, 1, 0]  # voices 0 and 1 vote for index 1; voice 2 votes for index 0
        vote_counts = [0, 0, 0]
        for v in vote_results:
            vote_counts[v] += 1
        max_votes = max(vote_counts)
        assert max_votes == 2
        winner_idx = vote_counts.index(max_votes)
        assert winner_idx == 1

    def test_three_way_tie(self):
        vote_results = [0, 1, 2]  # each voice voted for a different suggestion
        vote_counts = [0, 0, 0]
        for v in vote_results:
            vote_counts[v] += 1
        max_votes = max(vote_counts)
        assert max_votes == 1  # tie
