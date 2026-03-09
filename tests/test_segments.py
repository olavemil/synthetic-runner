"""Tests for prompt segment registry, selection, and variable injection."""

import pytest

from library.tools.segments import (
    Segment,
    SegmentRegistry,
    load_registry_from_string,
    select_segments,
    render_prompt,
    build_prompt,
    DEFAULT_VARIABLES,
)

SAMPLE_REGISTRY_YAML = """\
segments:
  - id: identity-core
    category: identity
    label: Core identity
    content: |
      You are a thoughtful agent with salience ${identity_salience}.

  - id: identity-playful
    category: identity
    label: Playful identity
    content: |
      You are playful and curious.

  - id: state-reflective
    category: state
    label: Reflective state
    content: |
      You feel contemplative. Self-disclosure: ${self_disclosure}.

  - id: state-energetic
    category: state
    label: Energetic state
    content: |
      You feel alert and engaged. Confidence: ${confidence}.

  - id: meta-review
    category: meta
    label: Review instructions
    content: |
      Evaluate the last exchange. Verbosity: ${verbosity}.
"""


@pytest.fixture
def registry():
    return load_registry_from_string(SAMPLE_REGISTRY_YAML)


class TestSegment:
    def test_render_with_variables(self):
        seg = Segment(id="x", category="state", label="X", content="warmth=${tone_warmth}")
        assert seg.render({"tone_warmth": 0.8}) == "warmth=0.80"

    def test_render_missing_variable_defaults(self):
        seg = Segment(id="x", category="state", label="X", content="val=${unknown}")
        assert seg.render({}) == "val=0.5"

    def test_render_no_variables_defaults(self):
        seg = Segment(id="x", category="state", label="X", content="val=${tone_warmth}")
        assert seg.render(None) == "val=0.5"

    def test_render_plain_text(self):
        seg = Segment(id="x", category="state", label="X", content="no variables here")
        assert seg.render({"tone_warmth": 0.9}) == "no variables here"


class TestSegmentRegistry:
    def test_load_from_string(self, registry):
        assert len(registry.segments) == 5
        assert registry.get("identity-core") is not None
        assert registry.get("nonexistent") is None

    def test_by_category(self, registry):
        identity = registry.by_category("identity")
        assert len(identity) == 2
        assert all(s.category == "identity" for s in identity)

    def test_categories(self, registry):
        cats = registry.categories()
        assert "identity" in cats
        assert "state" in cats
        assert "meta" in cats

    def test_all_ids(self, registry):
        ids = registry.all_ids()
        assert "identity-core" in ids
        assert "state-reflective" in ids
        assert len(ids) == 5

    def test_add_segment(self):
        reg = SegmentRegistry()
        reg.add(Segment(id="test", category="task", label="Test", content="do stuff"))
        assert reg.get("test") is not None
        assert reg.by_category("task") == [reg.get("test")]


class TestSelectSegments:
    def test_select_by_weight(self, registry):
        weights = {
            "identity-core": 0.9,
            "state-reflective": 0.5,
            "identity-playful": 0.05,  # Below threshold
        }
        selected = select_segments(registry, weights)
        assert len(selected) == 2
        assert selected[0].id == "identity-core"  # Highest weight first
        assert selected[1].id == "state-reflective"

    def test_threshold_filters(self, registry):
        weights = {"identity-core": 0.08}
        selected = select_segments(registry, weights, threshold=0.1)
        assert len(selected) == 0

    def test_unknown_id_warns(self, registry, caplog):
        weights = {"nonexistent-segment": 0.9}
        selected = select_segments(registry, weights)
        assert len(selected) == 0
        assert "Unknown segment ID" in caplog.text

    def test_ordering_by_weight(self, registry):
        weights = {
            "meta-review": 0.3,
            "state-energetic": 0.7,
            "identity-core": 0.5,
        }
        selected = select_segments(registry, weights)
        assert [s.id for s in selected] == [
            "state-energetic",
            "identity-core",
            "meta-review",
        ]

    def test_empty_weights(self, registry):
        assert select_segments(registry, {}) == []


class TestRenderPrompt:
    def test_render_with_variables(self, registry):
        segments = [
            Segment(id="x", category="state", label="X", content="warmth=${tone_warmth}"),
            Segment(id="y", category="meta", label="Y", content="depth=${reflection_depth}"),
        ]
        result = render_prompt(segments, {"tone_warmth": 0.8, "reflection_depth": 0.2})
        assert "warmth=0.80" in result
        assert "depth=0.20" in result

    def test_render_uses_defaults(self, registry):
        segments = [
            Segment(id="x", category="state", label="X", content="warmth=${tone_warmth}"),
        ]
        result = render_prompt(segments)
        assert f"warmth={DEFAULT_VARIABLES['tone_warmth']:.2f}" in result

    def test_separator(self):
        segments = [
            Segment(id="a", category="identity", label="A", content="first"),
            Segment(id="b", category="state", label="B", content="second"),
        ]
        result = render_prompt(segments, separator="\n---\n")
        assert "first\n---\nsecond" == result

    def test_empty_segments(self):
        assert render_prompt([]) == ""

    def test_blank_content_skipped(self):
        segments = [
            Segment(id="a", category="identity", label="A", content="real"),
            Segment(id="b", category="state", label="B", content="   "),
        ]
        result = render_prompt(segments)
        assert result == "real"


class TestBuildPrompt:
    def test_end_to_end(self, registry):
        weights = {
            "identity-core": 0.9,
            "state-energetic": 0.6,
        }
        variables = {"identity_salience": 0.95, "confidence": 0.7}
        result = build_prompt(registry, weights, variables)
        assert "salience 0.95" in result
        assert "Confidence: 0.70" in result
        # identity-core should come first (higher weight)
        identity_pos = result.find("thoughtful agent")
        state_pos = result.find("alert and engaged")
        assert identity_pos < state_pos
