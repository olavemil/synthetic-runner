"""Shared identity representation for multi-voice and colony species."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

# Six personality axes — (positive_pole, negative_pole)
AXES: list[tuple[str, str]] = [
    ("conservative", "liberal"),
    ("simple", "complex"),
    ("optimistic", "pessimistic"),
    ("extrovert", "introvert"),
    ("cautious", "bold"),
    ("analytical", "emotional"),
]

# Axis names used as dict keys
AXIS_NAMES: list[str] = [f"{a}_{b}" for a, b in AXES]


@dataclass
class Identity:
    """Unified representation for colony individuals and named voices.

    Colony individuals: name=UUID, dims set, personality derived at format time.
    Named voices: name=display name, personality=free text, dims=None.
    """

    name: str
    model: str = ""
    provider: str | None = None
    personality: str = ""
    dims: dict[str, float] | None = None
    cohesion: float = 0.0  # for colonies, how closely aligned to the original spawn prompt
    approval: int = 0
    created_at: int = field(default_factory=lambda: int(time.time()))
    age: int = 0  # number of thinking sessions since spawn
    parents: list[str] = field(default_factory=list)  # parent names for colony individuals


def format_persona(identity: Identity) -> str:
    """Return a descriptive persona string for the identity.

    If dims are present, use magnitude-label axis formatting.
    Otherwise return personality text (or name as fallback).
    """
    if identity.dims is None:
        return identity.personality or identity.name

    def magnitude_label(v: float) -> str:
        a = abs(v)
        if a >= 0.8:
            return "extremely"
        if a >= 0.6:
            return "very"
        if a >= 0.4:
            return "fairly"
        if a >= 0.2:
            return "somewhat"
        return ""

    entries = []
    for (pos_pole, neg_pole), name in zip(AXES, AXIS_NAMES):
        v = identity.dims.get(name, 0.0)
        pole = pos_pole if v >= 0 else neg_pole
        label = magnitude_label(v)
        entries.append((abs(v), label, pole))

    entries.sort(key=lambda x: -x[0])

    significant = [(mag, label, pole) for mag, label, pole in entries if mag >= 0.2]

    if not significant:
        mag, _label, pole = entries[0]
        return f"barely {pole}"

    parts = [f"{label} {pole}" for _mag, label, pole in significant]
    return ", ".join(parts)


def parse_model(model_str: str) -> tuple[str | None, str]:
    """Split 'provider/model' → (provider, model). Returns (None, model) if no '/'."""
    if "/" in model_str:
        provider, _, model = model_str.partition("/")
        return provider or None, model
    return None, model_str


def load_identity(raw: dict) -> Identity:
    """Create an Identity from a raw dict. Supports 'id'/'name' and 'dims'/'personality'."""
    model_str = str(raw.get("model", ""))
    provider, model = parse_model(model_str)
    name = str(raw.get("name") or raw.get("id", ""))
    dims_raw = raw.get("dims")
    dims = {k: float(v) for k, v in dims_raw.items()} if dims_raw else None
    parents_raw = raw.get("parents", [])
    parents = list(parents_raw) if isinstance(parents_raw, list) else []
    return Identity(
        name=name,
        model=model,
        provider=provider,
        personality=str(raw.get("personality", "")),
        dims=dims,
        cohesion=float(raw.get("cohesion", 0.0)),
        approval=int(raw.get("approval", 0)),
        created_at=int(raw.get("created_at", int(time.time()))),
        age=int(raw.get("age", 0)),
        parents=parents,
    )
