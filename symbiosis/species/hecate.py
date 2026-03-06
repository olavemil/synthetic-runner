"""Hecate species — three-voice deliberation with thinking and memory."""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from symbiosis.species import Species, SpeciesManifest, EntryPoint
from symbiosis.toolkit.hecate import (
    HecateConfig,
    load_config,
    load_shared_memory,
    load_voice_memory,
    update_voice_subconscious,
    _build_memory_context,
)
from symbiosis.toolkit.deliberate import deliberate, recompose, think_with_context
from symbiosis.toolkit.prompts import format_events

if TYPE_CHECKING:
    from symbiosis.harness.adapters import Event
    from symbiosis.harness.context import InstanceContext

logger = logging.getLogger(__name__)

_DEFAULT_VOICE_NAMES = ["Aria", "Sable", "Lune"]

DEFAULT_SHARED_FILES = {
    "memory.md": "# Memory\n",
    "constitution.md": "# Constitution\n",
}


def _default_voice_files(voice_names: list[str]) -> dict[str, str]:
    files = {}
    for name in voice_names:
        n = name.lower()
        files[f"{n}_thinking.md"] = f"# {name} Thinking\n"
        files[f"{n}_subconscious.md"] = f"# {name} Subconscious\n"
        files[f"{n}_motivation.md"] = f"# {name} Motivation\n"
    return files


def on_message(ctx: InstanceContext, events: list[Event]) -> None:
    """Three voices deliberate and produce a reply."""
    if not events:
        return

    cfg = load_config(ctx)
    if len(cfg.voices) != 3:
        logger.warning("Hecate requires exactly 3 voices; got %d", len(cfg.voices))
        return

    shared_memory = load_shared_memory(ctx)
    context = _build_memory_context(shared_memory)
    prompt = format_events(events)

    result = deliberate(
        ctx,
        cfg.voices,
        prompt,
        context=context,
        exclude_own=True,
        top_n=1,
    )

    if not result["candidates"]:
        return

    shuffled_voices = list(cfg.voices)
    random.shuffle(shuffled_voices)

    if result["is_tie"]:
        rewrites = [
            recompose(ctx, v, result["candidates"][v.name], context=context)
            for v in shuffled_voices
        ]
        ctx.send(cfg.voice_space, "\n\n---\n\n".join(rewrites))
    else:
        winner_voice = next(v for v in cfg.voices if v.name == result["winner_member"])
        ctx.send(
            cfg.voice_space,
            recompose(ctx, winner_voice, result["winner_message"], context=context),
        )

    # Update each voice's subconscious
    for v in cfg.voices:
        voice_memory = load_voice_memory(ctx, v)
        new_sub = update_voice_subconscious(ctx, v, prompt, shared_memory, voice_memory)
        ctx.write(f"{v.name.lower()}_subconscious.md", new_sub)


def heartbeat(ctx: InstanceContext) -> None:
    """Thinking iterations: each voice reflects, informed by others."""
    cfg = load_config(ctx)
    if len(cfg.voices) != 3:
        logger.warning("Hecate requires exactly 3 voices; got %d", len(cfg.voices))
        return

    shared_memory = load_shared_memory(ctx)
    context = _build_memory_context(shared_memory)
    voice_memories = {v.name: load_voice_memory(ctx, v) for v in cfg.voices}

    # Iteration 0: think without others' context
    current_thoughts: dict[str, str] = {}
    for voice in cfg.voices:
        thought = think_with_context(
            ctx, voice, context=context, voice_memory=voice_memories[voice.name]
        )
        current_thoughts[voice.name] = thought

    # Iterations 1..thinking_iterations: each voice sees others' previous thoughts
    for _iteration in range(1, cfg.thinking_iterations):
        new_thoughts: dict[str, str] = {}
        for voice in cfg.voices:
            others = {name: t for name, t in current_thoughts.items() if name != voice.name}
            thought = think_with_context(
                ctx,
                voice,
                context=context,
                others_thinking=others,
                voice_memory=voice_memories[voice.name],
            )
            new_thoughts[voice.name] = thought
        current_thoughts = new_thoughts

    # Write final thoughts
    for voice in cfg.voices:
        ctx.write(f"{voice.name.lower()}_thinking.md", current_thoughts[voice.name])


class HecateSpecies(Species):
    def manifest(self) -> SpeciesManifest:
        voice_names = _DEFAULT_VOICE_NAMES
        default_files = {**DEFAULT_SHARED_FILES, **_default_voice_files(voice_names)}

        return SpeciesManifest(
            species_id="hecate",
            entry_points=[
                EntryPoint(
                    name="on_message",
                    handler=on_message,
                    trigger="message",
                ),
                EntryPoint(
                    name="heartbeat",
                    handler=heartbeat,
                    schedule="0 * * * *",
                ),
            ],
            tools=[],
            default_files=default_files,
            spawn=self._spawn,
        )

    def _spawn(self, ctx: InstanceContext) -> None:
        for path, content in DEFAULT_SHARED_FILES.items():
            if not ctx.exists(path):
                ctx.write(path, content)

        cfg = load_config(ctx)
        voices_names = [v.name for v in cfg.voices] if cfg.voices else _DEFAULT_VOICE_NAMES
        for path, content in _default_voice_files(voices_names).items():
            if not ctx.exists(path):
                ctx.write(path, content)
