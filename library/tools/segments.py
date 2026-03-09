"""Prompt segment registry, selection, and variable injection.

A segment is a pre-written prompt fragment belonging to a category. The registry
holds all available segments. At prompt-build time, a selection vector (weights
per segment) determines which segments are active and their order. Named variables
are injected into active segments via ``${variable_name}`` placeholders.

This module is usable standalone with manual weights. When the neural net layer
is available, the nets produce both the selection weights and variable values.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from library.harness.context import InstanceContext

logger = logging.getLogger(__name__)

# Categories and which net primarily controls them (informational — enforcement
# happens in the neural net layer, not here).
CATEGORY_NET_AFFINITY = {
    "identity": "slow",
    "state": "fast",
    "relational": "fast",
    "task": "slow",
    "temporal": "slow",
    "meta": "fast",
}

# Default variable definitions with their net source and neutral default.
DEFAULT_VARIABLES: dict[str, float] = {
    # Fast net variables
    "tone_warmth": 0.5,
    "verbosity": 0.5,
    "risk_tolerance": 0.5,
    "self_disclosure": 0.5,
    "confidence": 0.5,
    # Slow net variables
    "identity_salience": 0.5,
    "temporal_weight": 0.5,
    "relational_depth": 0.5,
    "reflection_depth": 0.5,
}

_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


@dataclass
class Segment:
    """A single prompt segment."""

    id: str
    category: str
    label: str
    content: str
    weight: float = 0.0  # Selection weight (0 = inactive)

    def render(self, variables: dict[str, float] | None = None) -> str:
        """Return content with variables substituted."""
        if not variables:
            return _VAR_PATTERN.sub("0.5", self.content)

        def _replace(m: re.Match) -> str:
            name = m.group(1)
            if name in variables:
                return f"{variables[name]:.2f}"
            return "0.5"

        return _VAR_PATTERN.sub(_replace, self.content)


@dataclass
class SegmentRegistry:
    """Collection of prompt segments organised by category."""

    segments: dict[str, Segment] = field(default_factory=dict)

    def add(self, segment: Segment) -> None:
        self.segments[segment.id] = segment

    def get(self, segment_id: str) -> Segment | None:
        return self.segments.get(segment_id)

    def by_category(self, category: str) -> list[Segment]:
        return [s for s in self.segments.values() if s.category == category]

    def categories(self) -> list[str]:
        return sorted({s.category for s in self.segments.values()})

    def all_ids(self) -> list[str]:
        return sorted(self.segments.keys())


def load_registry(source: str | Path) -> SegmentRegistry:
    """Load a segment registry from a YAML file.

    Expected format::

        segments:
          - id: identity-core
            category: identity
            label: Core identity
            content: |
              You are ... ${identity_salience} ...
          - id: state-reflective
            category: state
            label: Reflective state
            content: |
              Current internal state ... ${self_disclosure} ...
    """
    path = Path(source)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    registry = SegmentRegistry()
    for entry in data.get("segments", []):
        registry.add(
            Segment(
                id=entry["id"],
                category=entry["category"],
                label=entry.get("label", entry["id"]),
                content=entry.get("content", ""),
            )
        )
    return registry


def load_registry_from_string(text: str) -> SegmentRegistry:
    """Load a segment registry from a YAML string."""
    data = yaml.safe_load(text)
    registry = SegmentRegistry()
    for entry in data.get("segments", []):
        registry.add(
            Segment(
                id=entry["id"],
                category=entry["category"],
                label=entry.get("label", entry["id"]),
                content=entry.get("content", ""),
            )
        )
    return registry


def select_segments(
    registry: SegmentRegistry,
    weights: dict[str, float],
    *,
    threshold: float = 0.1,
) -> list[Segment]:
    """Select and order segments by weight.

    Args:
        registry: The full segment registry.
        weights: Mapping of segment ID → activation weight (0.0–1.0).
        threshold: Minimum weight for a segment to be included.

    Returns:
        Active segments ordered by descending weight (earlier = higher influence).
    """
    active = []
    for seg_id, weight in weights.items():
        seg = registry.get(seg_id)
        if seg is None:
            logger.warning("Unknown segment ID in selection: %s", seg_id)
            continue
        if weight >= threshold:
            active.append(Segment(
                id=seg.id,
                category=seg.category,
                label=seg.label,
                content=seg.content,
                weight=weight,
            ))
    active.sort(key=lambda s: s.weight, reverse=True)
    return active


def render_prompt(
    segments: list[Segment],
    variables: dict[str, float] | None = None,
    *,
    separator: str = "\n\n",
) -> str:
    """Render selected segments into a single prompt string.

    Args:
        segments: Ordered list of active segments (from ``select_segments``).
        variables: Named float values to inject via ``${name}`` placeholders.
        separator: String between rendered segments.

    Returns:
        Assembled prompt text.
    """
    vars_with_defaults = dict(DEFAULT_VARIABLES)
    if variables:
        vars_with_defaults.update(variables)

    parts = []
    for seg in segments:
        rendered = seg.render(vars_with_defaults)
        if rendered.strip():
            parts.append(rendered.strip())
    return separator.join(parts)


def build_prompt(
    registry: SegmentRegistry,
    weights: dict[str, float],
    variables: dict[str, float] | None = None,
    *,
    threshold: float = 0.1,
    separator: str = "\n\n",
) -> str:
    """Convenience: select segments and render in one call."""
    active = select_segments(registry, weights, threshold=threshold)
    return render_prompt(active, variables, separator=separator)
