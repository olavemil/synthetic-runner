"""Tests for the Hecate toolkit."""

from __future__ import annotations

from symbiosis.toolkit.identity import Identity
from symbiosis.toolkit.hecate import (
    HecateConfig,
    load_config,
)


class DummyCtx:
    def __init__(self, hecate_cfg=None):
        self._hecate_cfg = hecate_cfg or {}

    def config(self, key: str):
        if key == "hecate":
            return self._hecate_cfg
        return None


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

    def test_voices_are_identity_instances(self):
        ctx = DummyCtx(
            hecate_cfg={
                "voices": [
                    {"name": "Aria", "model": "m", "personality": "p"},
                ],
            }
        )
        cfg = load_config(ctx)
        assert isinstance(cfg.voices[0], Identity)

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


class TestVoteTally:
    """Test the vote tally logic (inline, as the tally is now in voting.py)."""

    def test_two_vote_winner(self):
        # Simulating: voices 0 and 1 vote for Sable; voice 2 votes for Aria
        from symbiosis.toolkit.voting import borda_tally
        candidates = {"Aria": "text a", "Sable": "text b", "Lune": "text c"}
        votes = {
            "Aria": ["Sable"],   # Aria votes for Sable
            "Sable": ["Aria"],   # Sable votes for Aria (exclude_own)
            "Lune": ["Sable"],   # Lune votes for Sable
        }
        tally = borda_tally(candidates, votes)
        assert tally["winner_member"] == "Sable"

    def test_three_way_tie(self):
        from symbiosis.toolkit.voting import borda_tally
        candidates = {"Aria": "text a", "Sable": "text b", "Lune": "text c"}
        # Each voice votes for a different candidate → all same score
        votes = {
            "Aria": ["Sable"],
            "Sable": ["Lune"],
            "Lune": ["Aria"],
        }
        tally = borda_tally(candidates, votes)
        assert tally["is_tie"] is True
