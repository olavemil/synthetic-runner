"""Hecate toolkit — three-voice deliberation system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from symbiosis.harness.context import InstanceContext


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Voice:
    name: str
    model: str
    personality: str
    provider: str | None = None


@dataclass
class HecateConfig:
    voices: list[Voice] = field(default_factory=list)
    thinking_iterations: int = 2
    voice_space: str = "main"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _split_model(model_str: str) -> tuple[str | None, str]:
    """Split 'provider/model' → (provider, model). Returns (None, model) if no '/'."""
    if "/" in model_str:
        provider, _, model = model_str.partition("/")
        return provider or None, model
    return None, model_str


def load_config(ctx: InstanceContext) -> HecateConfig:
    raw = ctx.config("hecate") or {}
    if not isinstance(raw, dict):
        raw = {}

    voices = []
    for v in raw.get("voices", []):
        if not isinstance(v, dict):
            continue
        model_str = str(v.get("model", ""))
        provider, model = _split_model(model_str)
        voices.append(
            Voice(
                name=str(v.get("name", "Voice")),
                model=model,
                personality=str(v.get("personality", "")),
                provider=provider,
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


def load_voice_memory(ctx: InstanceContext, voice: Voice) -> dict[str, str]:
    """Load per-voice files: {name}_thinking.md, {name}_subconscious.md, {name}_motivation.md."""
    name = voice.name.lower()
    result = {}
    for suffix in ("thinking", "subconscious", "motivation"):
        fname = f"{name}_{suffix}.md"
        content = ctx.read(fname)
        if content:
            result[suffix] = content
    return result


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------


def _call_voice(ctx: InstanceContext, voice: Voice, messages: list[dict], **kwargs):
    """Call ctx.llm with voice's model/provider."""
    kwargs["model"] = voice.model
    if voice.provider:
        kwargs["provider"] = voice.provider
    return ctx.llm(messages, **kwargs)


# ---------------------------------------------------------------------------
# Voice operations
# ---------------------------------------------------------------------------


def think(
    ctx: InstanceContext,
    voice: Voice,
    shared_memory: dict[str, str],
    others_thinking: dict[str, str],
) -> str:
    """Generate thinking for this voice, optionally informed by others' thoughts."""
    memory_parts = []
    if shared_memory.get("memory.md"):
        memory_parts.append(f"## Shared Memory\n{shared_memory['memory.md']}")
    if shared_memory.get("constitution.md"):
        memory_parts.append(f"## Constitution\n{shared_memory['constitution.md']}")
    memory_ctx = "\n\n".join(memory_parts)

    others_block = ""
    if others_thinking:
        lines = [f"**{name}**: {thought}" for name, thought in others_thinking.items()]
        others_block = "\n\n## Other voices' thoughts\n" + "\n\n".join(lines)

    system = (
        f"You are {voice.name}. Personality: {voice.personality}\n\n"
        f"{memory_ctx}"
    )
    user_msg = (
        f"Reflect and think freely.{others_block}\n\n"
        "Write your current thoughts."
    )
    response = _call_voice(
        ctx,
        voice,
        [{"role": "user", "content": user_msg}],
        system=system,
        max_tokens=512,
        caller=f"hecate_think_{voice.name}",
    )
    return response.message.strip()


def suggest_response(
    ctx: InstanceContext,
    voice: Voice,
    prompt: str,
    shared_memory: dict[str, str],
    voice_memory: dict[str, str],
) -> dict:
    """Generate a candidate response. Returns {"text": str, "argument": str}."""
    memory_parts = []
    if shared_memory.get("memory.md"):
        memory_parts.append(f"## Shared Memory\n{shared_memory['memory.md']}")
    if shared_memory.get("constitution.md"):
        memory_parts.append(f"## Constitution\n{shared_memory['constitution.md']}")
    if voice_memory.get("thinking"):
        memory_parts.append(f"## Your Thoughts\n{voice_memory['thinking']}")
    if voice_memory.get("subconscious"):
        memory_parts.append(f"## Subconscious\n{voice_memory['subconscious']}")
    if voice_memory.get("motivation"):
        memory_parts.append(f"## Motivation\n{voice_memory['motivation']}")
    memory_ctx = "\n\n".join(memory_parts)

    system = (
        f"You are {voice.name}. Personality: {voice.personality}\n\n"
        f"{memory_ctx}"
    )
    user_msg = (
        f"Conversation:\n{prompt}\n\n"
        "Write a candidate reply and a brief argument for why it's the right response.\n"
        'Return JSON: {"text": "...", "argument": "..."}'
    )
    response = _call_voice(
        ctx,
        voice,
        [{"role": "user", "content": user_msg}],
        system=system,
        max_tokens=512,
        caller=f"hecate_suggest_{voice.name}",
    )
    # Parse JSON
    try:
        import json
        raw = response.message.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(raw)
        return {
            "text": str(data.get("text", response.message.strip())),
            "argument": str(data.get("argument", "")),
        }
    except Exception:
        return {"text": response.message.strip(), "argument": ""}


def vote_response(
    ctx: InstanceContext,
    voice: Voice,
    suggestions: list[dict],
    my_idx: int,
) -> int:
    """Vote on which of the OTHER voices' suggestions is best. Returns 0-based index."""
    other_indices = [i for i in range(len(suggestions)) if i != my_idx]
    if not other_indices:
        return my_idx  # fallback (shouldn't happen with 3 voices)
    if len(other_indices) == 1:
        return other_indices[0]

    options = "\n".join(
        f"{i}: {suggestions[i]['text']} (argument: {suggestions[i]['argument']})"
        for i in other_indices
    )
    system = f"You are {voice.name}. Personality: {voice.personality}"
    user_msg = (
        f"Choose the best response from the other voices:\n{options}\n\n"
        f'Return JSON: {{"choice": <index>}}'
    )
    response = _call_voice(
        ctx,
        voice,
        [{"role": "user", "content": user_msg}],
        system=system,
        max_tokens=64,
        caller=f"hecate_vote_{voice.name}",
    )
    try:
        import json
        raw = response.message.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(raw)
        choice = int(data.get("choice", other_indices[0]))
        if choice not in other_indices:
            return other_indices[0]
        return choice
    except Exception:
        return other_indices[0]


def reword_response(
    ctx: InstanceContext,
    voice: Voice,
    winning_text: str,
    shared_memory: dict[str, str],
    voice_memory: dict[str, str],
) -> str:
    """Rewrite the winning text through this voice's perspective."""
    memory_parts = []
    if shared_memory.get("memory.md"):
        memory_parts.append(f"## Shared Memory\n{shared_memory['memory.md']}")
    if voice_memory.get("thinking"):
        memory_parts.append(f"## Your Thoughts\n{voice_memory['thinking']}")
    memory_ctx = "\n\n".join(memory_parts)

    system = (
        f"You are {voice.name}. Personality: {voice.personality}\n\n"
        f"{memory_ctx}"
    )
    user_msg = (
        f"Rewrite this response in your own voice and style:\n\n{winning_text}"
    )
    response = _call_voice(
        ctx,
        voice,
        [{"role": "user", "content": user_msg}],
        system=system,
        max_tokens=512,
        caller=f"hecate_reword_{voice.name}",
    )
    return response.message.strip()


def update_voice_subconscious(
    ctx: InstanceContext,
    voice: Voice,
    prompt: str,
    shared_memory: dict[str, str],
    voice_memory: dict[str, str],
) -> str:
    """Generate updated subconscious text. Caller writes {name}_subconscious.md."""
    memory_parts = []
    if shared_memory.get("memory.md"):
        memory_parts.append(f"## Shared Memory\n{shared_memory['memory.md']}")
    if voice_memory.get("subconscious"):
        memory_parts.append(f"## Current Subconscious\n{voice_memory['subconscious']}")
    memory_ctx = "\n\n".join(memory_parts)

    system = (
        f"You are {voice.name}. Personality: {voice.personality}\n\n"
        f"{memory_ctx}"
    )
    user_msg = (
        f"After this conversation:\n{prompt}\n\n"
        "Write a brief subconscious note to yourself about how you feel and what lingers."
    )
    response = _call_voice(
        ctx,
        voice,
        [{"role": "user", "content": user_msg}],
        system=system,
        max_tokens=256,
        caller=f"hecate_subconscious_{voice.name}",
    )
    return response.message.strip()
