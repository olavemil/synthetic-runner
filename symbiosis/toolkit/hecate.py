"""Hecate toolkit — voice config and memory helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from symbiosis.toolkit.identity import Identity, parse_model, load_identity
from symbiosis.toolkit.deliberate import generate_with_identity

if TYPE_CHECKING:
    from symbiosis.harness.context import InstanceContext

# Backward-compat alias: voices ARE Identity objects
Voice = Identity


@dataclass
class HecateConfig:
    voices: list[Identity] = field(default_factory=list)
    thinking_iterations: int = 2
    voice_space: str = "main"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(ctx: InstanceContext) -> HecateConfig:
    raw = ctx.config("hecate") or {}
    if not isinstance(raw, dict):
        raw = {}

    voices = []
    for v in raw.get("voices", []):
        if not isinstance(v, dict):
            continue
        model_str = str(v.get("model", ""))
        provider, model = parse_model(model_str)
        voices.append(
            Identity(
                name=str(v.get("name", "Voice")),
                model=model,
                provider=provider,
                personality=str(v.get("personality", "")),
            )
        )

    try:
        thinking_iterations = int(raw.get("thinking_iterations", 2))
    except (TypeError, ValueError):
        thinking_iterations = 2

    return HecateConfig(
        voices=voices,
        thinking_iterations=max(1, thinking_iterations),
        voice_space=str(raw.get("voice_space", "main")),
    )


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------


def load_shared_memory(ctx: InstanceContext) -> dict[str, str]:
    """Load shared files: memory.md, constitution.md."""
    result = {}
    for fname in ("memory.md", "constitution.md"):
        content = ctx.read(fname)
        if content:
            result[fname] = content
    return result


def load_voice_memory(ctx: InstanceContext, identity: Identity) -> dict[str, str]:
    """Load per-voice files: {name}_thinking.md, {name}_subconscious.md, {name}_motivation.md."""
    name = identity.name.lower()
    result = {}
    for suffix in ("thinking", "subconscious", "motivation"):
        fname = f"{name}_{suffix}.md"
        content = ctx.read(fname)
        if content:
            result[suffix] = content
    return result


def _build_memory_context(shared_memory: dict[str, str], voice_memory: dict[str, str] | None = None) -> str:
    """Build a context string from shared and per-voice memory dicts."""
    parts = []
    if shared_memory.get("memory.md"):
        parts.append(f"## Shared Memory\n{shared_memory['memory.md']}")
    if shared_memory.get("constitution.md"):
        parts.append(f"## Constitution\n{shared_memory['constitution.md']}")
    if voice_memory:
        if voice_memory.get("thinking"):
            parts.append(f"## Your Thoughts\n{voice_memory['thinking']}")
        if voice_memory.get("subconscious"):
            parts.append(f"## Subconscious\n{voice_memory['subconscious']}")
        if voice_memory.get("motivation"):
            parts.append(f"## Motivation\n{voice_memory['motivation']}")
    return "\n\n".join(parts)


def build_memory_snapshot_context(ctx: InstanceContext, voices: list[Identity]) -> str:
    """Render a pre-iteration snapshot of memory directory files."""
    paths: list[str] = []
    if hasattr(ctx, "list"):
        try:
            paths = sorted(p for p in ctx.list("") if p.endswith(".md"))
        except Exception:
            paths = []

    if not paths:
        fallback = ["memory.md", "constitution.md"]
        for voice in voices:
            base = voice.name.lower()
            fallback.extend(
                [f"{base}_thinking.md", f"{base}_subconscious.md", f"{base}_motivation.md"]
            )
        # Keep order, remove duplicates.
        seen = set()
        for path in fallback:
            if path not in seen:
                seen.add(path)
                paths.append(path)

    parts = ["## Memory Directory Snapshot (before iteration)"]
    for path in paths:
        content = ctx.read(path)
        if not content:
            continue
        parts.append(f"### {path}\n{content}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Voice operations
# ---------------------------------------------------------------------------


def update_voice_subconscious(
    ctx: InstanceContext,
    identity: Identity,
    prompt: str,
    shared_memory: dict[str, str],
    voice_memory: dict[str, str],
    user_prompt: str = "",
) -> str:
    """Generate updated subconscious text. Caller writes {name}_subconscious.md."""
    context = _build_memory_context(shared_memory, {"subconscious": voice_memory.get("subconscious", "")})
    if not user_prompt:
        user_prompt = (
            f"After this conversation:\n{prompt}\n\n"
            "Write a brief subconscious note to yourself about how you feel and what lingers."
        )
    return generate_with_identity(ctx, identity, user_prompt, context=context)
