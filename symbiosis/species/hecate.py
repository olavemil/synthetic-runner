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
    think,
    suggest_response,
    vote_response,
    reword_response,
    update_voice_subconscious,
)
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
    voice_memories = [load_voice_memory(ctx, v) for v in cfg.voices]
    prompt = format_events(events)

    # Each voice suggests a response
    suggestions = [
        suggest_response(ctx, v, prompt, shared_memory, voice_memories[i])
        for i, v in enumerate(cfg.voices)
    ]

    # Each voice votes on the other voices' suggestions
    vote_results = [
        vote_response(ctx, v, suggestions, i)
        for i, v in enumerate(cfg.voices)
    ]

    # Tally votes per suggestion index
    vote_counts = [0, 0, 0]
    for voted_idx in vote_results:
        if 0 <= voted_idx < 3:
            vote_counts[voted_idx] += 1

    # Determine winner: one suggestion with 2 votes, or 3-way tie (1-1-1)
    max_votes = max(vote_counts)
    shuffled_voices = list(enumerate(cfg.voices))
    random.shuffle(shuffled_voices)

    if max_votes >= 2:
        # Single winner
        winner_idx = vote_counts.index(max_votes)
        winning_text = suggestions[winner_idx]["text"]
        # Winner's voice rewrites
        winner_voice = cfg.voices[winner_idx]
        winner_vm = voice_memories[winner_idx]
        final = reword_response(ctx, winner_voice, winning_text, shared_memory, winner_vm)
        ctx.send(cfg.voice_space, final)
    else:
        # 3-way tie — all three voices reword their own suggestion, joined
        rewrites = []
        for i, v in shuffled_voices:
            reworded = reword_response(ctx, v, suggestions[i]["text"], shared_memory, voice_memories[i])
            rewrites.append(reworded)
        ctx.send(cfg.voice_space, "\n\n---\n\n".join(rewrites))

    # Update each voice's subconscious
    for i, v in enumerate(cfg.voices):
        new_sub = update_voice_subconscious(ctx, v, prompt, shared_memory, voice_memories[i])
        ctx.write(f"{v.name.lower()}_subconscious.md", new_sub)


def heartbeat(ctx: InstanceContext) -> None:
    """Thinking iterations: each voice reflects, informed by others."""
    cfg = load_config(ctx)
    if len(cfg.voices) != 3:
        logger.warning("Hecate requires exactly 3 voices; got %d", len(cfg.voices))
        return

    shared_memory = load_shared_memory(ctx)

    # Iteration 0: think without others' context
    current_thoughts: dict[str, str] = {}
    for voice in cfg.voices:
        thought = think(ctx, voice, shared_memory, others_thinking={})
        current_thoughts[voice.name] = thought

    # Iterations 1..thinking_iterations: each voice sees others' previous thoughts
    for _iteration in range(1, cfg.thinking_iterations):
        new_thoughts: dict[str, str] = {}
        for voice in cfg.voices:
            others = {name: t for name, t in current_thoughts.items() if name != voice.name}
            thought = think(ctx, voice, shared_memory, others_thinking=others)
            new_thoughts[voice.name] = thought
        current_thoughts = new_thoughts

    # Write final thoughts
    for voice in cfg.voices:
        ctx.write(f"{voice.name.lower()}_thinking.md", current_thoughts[voice.name])


class HecateSpecies(Species):
    def manifest(self) -> SpeciesManifest:
        # Build default files using configured voice names if available
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
        # Write shared files
        for path, content in DEFAULT_SHARED_FILES.items():
            if not ctx.exists(path):
                ctx.write(path, content)

        # Write per-voice files using configured voices if available
        cfg = load_config(ctx)
        voices_names = [v.name for v in cfg.voices] if cfg.voices else _DEFAULT_VOICE_NAMES
        for path, content in _default_voice_files(voices_names).items():
            if not ctx.exists(path):
                ctx.write(path, content)
