"""Hecate toolkit — voice config and memory helpers."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from library.tools.identity import Identity, parse_model, load_identity
from library.tools.deliberate import generate_with_identity

if TYPE_CHECKING:
    from library.harness.context import InstanceContext

logger = logging.getLogger(__name__)

_SPECIES_DIR = Path(__file__).parent.parent / "species" / "hecate"
_PROMPT_VOICE_MESSAGING = (_SPECIES_DIR / "prompts/voice_messaging.md").read_text()

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


# ---------------------------------------------------------------------------
# Private voice-to-voice messaging
# ---------------------------------------------------------------------------


def _voice_message_path(sender_name: str, recipient_name: str) -> str:
    """Stable path for a private message from sender to recipient."""
    return f"voice_messages/{sender_name.lower()}_{recipient_name.lower()}.md"


def load_voice_inbox(ctx: "InstanceContext", voice: Identity, all_voices: list[Identity]) -> dict[str, str]:
    """Load private messages for a voice — only messages where voice is sender or receiver.

    Returns a dict with keys like 'To Aria' or 'From Sable' and message content as values.
    The third voice in a trio cannot see messages between the other two.
    """
    messages: dict[str, str] = {}
    for other in all_voices:
        if other.name == voice.name:
            continue
        sent_path = _voice_message_path(voice.name, other.name)
        sent = ctx.read(sent_path)
        if sent and sent.strip():
            messages[f"To {other.name}"] = sent.strip()
        recv_path = _voice_message_path(other.name, voice.name)
        recv = ctx.read(recv_path)
        if recv and recv.strip():
            messages[f"From {other.name}"] = recv.strip()
    return messages


def format_voice_inbox(inbox: dict[str, str]) -> str:
    """Format a voice inbox dict as a readable block for prompt injection."""
    if not inbox:
        return ""
    lines = ["Your private messages:"]
    for label, msg in inbox.items():
        lines.append(f"  [{label}]: {msg}")
    return "\n".join(lines)


def send_voice_message(
    ctx: "InstanceContext",
    sender: Identity,
    recipient: Identity,
    message: str,
) -> None:
    """Overwrite the private message from sender to recipient."""
    path = _voice_message_path(sender.name, recipient.name)
    ctx.write(path, message.strip())
    logger.info(
        "Hecate voice message from %s to %s (%d chars)",
        sender.name, recipient.name, len(message),
    )


def run_voice_messaging_phase(
    ctx: "InstanceContext",
    voices: list[Identity],
    cfg: HecateConfig,
) -> int:
    """Each voice may send one private message to another. Returns count sent."""
    sent = 0
    voice_names = [v.name for v in voices]
    for voice in voices:
        others = [v for v in voices if v.name != voice.name]
        inbox = load_voice_inbox(ctx, voice, voices)
        inbox_text = format_voice_inbox(inbox)
        inbox_block = f"\n{inbox_text}\n" if inbox_text else ""
        prompt = (
            _PROMPT_VOICE_MESSAGING
            .replace("{voice_name}", voice.name)
            .replace("{others}", ", ".join(v.name for v in others))
            .replace("{inbox_block}", inbox_block)
        )
        raw = generate_with_identity(
            ctx, voice, prompt,
            context="Hecate private voice messaging phase.",
            max_tokens=256,
        )
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(cleaned)
            target_name = data.get("to")
            msg = str(data.get("message", "")).strip()
            if target_name and msg and target_name in voice_names and target_name != voice.name:
                recipient = next((v for v in voices if v.name == target_name), None)
                if recipient:
                    send_voice_message(ctx, voice, recipient, msg[:300])
                    sent += 1
        except Exception:
            pass

    logger.info("Hecate voice messaging phase: %d messages sent", sent)
    return sent
