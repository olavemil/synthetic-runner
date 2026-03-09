"""Tests for library.tools.identity."""

from __future__ import annotations

from library.tools.identity import (
    AXES,
    AXIS_NAMES,
    Identity,
    format_persona,
    parse_model,
    load_identity,
)


class TestFormatPersonaWithDims:
    def _identity(self, dims: dict) -> Identity:
        return Identity(name="test", dims=dims)

    def test_extreme_trait(self):
        dims = {name: 0.0 for name in AXIS_NAMES}
        dims["conservative_liberal"] = 0.9
        result = format_persona(self._identity(dims))
        assert "extremely" in result
        assert "conservative" in result

    def test_negative_pole(self):
        dims = {name: 0.0 for name in AXIS_NAMES}
        dims["optimistic_pessimistic"] = -0.7
        result = format_persona(self._identity(dims))
        assert "very" in result
        assert "pessimistic" in result

    def test_midrange_label(self):
        dims = {name: 0.0 for name in AXIS_NAMES}
        dims["cautious_bold"] = 0.5
        result = format_persona(self._identity(dims))
        assert "fairly" in result
        assert "cautious" in result

    def test_all_near_zero_fallback(self):
        dims = {name: 0.0 for name in AXIS_NAMES}
        dims["analytical_emotional"] = 0.1
        result = format_persona(self._identity(dims))
        assert "barely" in result
        assert "analytical" in result

    def test_traits_ordered_by_magnitude(self):
        dims = {name: 0.0 for name in AXIS_NAMES}
        dims["conservative_liberal"] = 0.3
        dims["analytical_emotional"] = 0.85
        result = format_persona(self._identity(dims))
        assert result.index("analytical") < result.index("conservative")


class TestFormatPersonaWithoutDims:
    def test_returns_personality_when_no_dims(self):
        identity = Identity(name="Aria", personality="Bold and analytical")
        assert format_persona(identity) == "Bold and analytical"

    def test_falls_back_to_name_when_no_personality(self):
        identity = Identity(name="Aria")
        assert format_persona(identity) == "Aria"

    def test_dims_none_uses_personality(self):
        identity = Identity(name="Sable", dims=None, personality="Poetic and dreamy")
        assert format_persona(identity) == "Poetic and dreamy"


class TestParseModel:
    def test_provider_model_split(self):
        provider, model = parse_model("anthropic/claude-sonnet-4-6")
        assert provider == "anthropic"
        assert model == "claude-sonnet-4-6"

    def test_no_slash_returns_none_provider(self):
        provider, model = parse_model("local-model")
        assert provider is None
        assert model == "local-model"

    def test_empty_string(self):
        provider, model = parse_model("")
        assert provider is None
        assert model == ""

    def test_leading_slash_gives_none_provider(self):
        provider, model = parse_model("/model-only")
        assert provider is None
        assert model == "model-only"


class TestLoadIdentity:
    def test_loads_name_and_personality(self):
        raw = {"name": "Aria", "personality": "Analytical", "model": "m"}
        identity = load_identity(raw)
        assert identity.name == "Aria"
        assert identity.personality == "Analytical"
        assert identity.model == "m"
        assert identity.provider is None

    def test_supports_id_field_as_name(self):
        raw = {"id": "uuid-123", "dims": {"conservative_liberal": 0.5}}
        identity = load_identity(raw)
        assert identity.name == "uuid-123"

    def test_parses_provider_from_model(self):
        raw = {"name": "V", "model": "openai/gpt-4"}
        identity = load_identity(raw)
        assert identity.provider == "openai"
        assert identity.model == "gpt-4"

    def test_loads_dims(self):
        raw = {"name": "x", "dims": {"conservative_liberal": 0.3, "simple_complex": -0.5}}
        identity = load_identity(raw)
        assert identity.dims is not None
        assert identity.dims["conservative_liberal"] == 0.3

    def test_no_dims_gives_none(self):
        raw = {"name": "v", "personality": "bold"}
        identity = load_identity(raw)
        assert identity.dims is None

    def test_loads_approval_and_created_at(self):
        raw = {"name": "x", "approval": 5, "created_at": 1000}
        identity = load_identity(raw)
        assert identity.approval == 5
        assert identity.created_at == 1000
