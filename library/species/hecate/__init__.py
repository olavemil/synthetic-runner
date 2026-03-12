"""Hecate species — three-voice deliberation with thinking and memory."""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING

from library.species import Species, SpeciesManifest, EntryPoint
from library.tools.hecate import (
    HecateConfig,
    build_memory_snapshot_context,
    load_config,
    load_shared_memory,
    load_voice_memory,
    update_voice_subconscious,
    _build_memory_context,
)
from library.tools.deliberate import generate_with_identity, think_with_context
from library.tools.prompts import format_events, get_entity_id

if TYPE_CHECKING:
    from library.harness.adapters import Event
    from library.harness.context import InstanceContext

logger = logging.getLogger(__name__)

_SPECIES_DIR = Path(__file__).parent
_SUGGEST_PROMPT = (_SPECIES_DIR / "prompts/suggest.md").read_text()
_COMPOSE_PROMPT = (_SPECIES_DIR / "prompts/compose.md").read_text()
_SUBCONSCIOUS_PROMPT = (_SPECIES_DIR / "prompts/subconscious.md").read_text()
_CREATE_PROMPT = (_SPECIES_DIR / "prompts/create.md").read_text()

_DEFAULT_VOICE_NAMES = ["Aria", "Sable", "Lune"]

DEFAULT_SHARED_FILES = {
    "memory.md": "# Memory\n",
    "constitution.md": "# Constitution\n",
}


def _to_one_sentence(text: str, max_chars: int = 220) -> str:
    cleaned = " ".join((text or "").split()).strip()
    if not cleaned:
        return ""
    match = re.search(r"[.!?](?:\s|$)", cleaned)
    sentence = cleaned[: match.end()].strip() if match else cleaned
    if len(sentence) > max_chars:
        sentence = sentence[:max_chars].rstrip()
    return sentence


def _default_voice_files(voice_names: list[str]) -> dict[str, str]:
    files = {}
    for name in voice_names:
        n = name.lower()
        files[f"{n}_thinking.md"] = f"# {name} Thinking\n"
        files[f"{n}_subconscious.md"] = f"# {name} Subconscious\n"
        files[f"{n}_motivation.md"] = f"# {name} Motivation\n"
    return files


def on_message(ctx: InstanceContext, events: list[Event]) -> None:
    """Three voices draft, randomize, and one random voice composes the final reply."""
    if not events:
        logger.info("Hecate on_message skipped (events=0)")
        return

    logger.info("Hecate on_message start (events=%d)", len(events))
    cfg = load_config(ctx)
    if len(cfg.voices) != 3:
        logger.warning("Hecate requires exactly 3 voices; got %d", len(cfg.voices))
        return

    shared_memory = load_shared_memory(ctx)
    context = _build_memory_context(shared_memory)
    entity_id = get_entity_id(ctx)
    full_prompt = format_events(events, self_entity_id=entity_id)
    events_with_room = [evt for evt in events if evt.room]
    if events_with_room:
        target_event = max(events_with_room, key=lambda evt: evt.timestamp)
        target_room = target_event.room
        room_events = [evt for evt in events_with_room if evt.room == target_room]
        if len({evt.room for evt in events_with_room}) > 1:
            logger.info(
                "Hecate on_message received events from multiple rooms; selecting latest room=%s (room_events=%d total_events=%d)",
                target_room,
                len(room_events),
                len(events),
            )
    else:
        target_room = cfg.voice_space
        room_events = events

    room_prompt = format_events(room_events, self_entity_id=entity_id) if room_events else full_prompt
    suggestions: list[tuple[str, str]] = []
    for voice in cfg.voices:
        suggestion_prompt = (
            _SUGGEST_PROMPT
            .replace("{target_room}", target_room)
            .replace("{conversation}", room_prompt)
            .replace("{voice_name}", voice.name)
        )
        raw = generate_with_identity(
            ctx,
            voice,
            suggestion_prompt,
            context=context,
            max_tokens=2048,
        )
        suggestion = _to_one_sentence(raw, max_chars=220)
        if suggestion:
            suggestions.append((voice.name, suggestion))

    if not suggestions:
        logger.info("Hecate on_message no suggestions generated for room=%s; skipping", target_room)
        return

    random.shuffle(suggestions)
    joined = "\n".join(f"- ({name}) {text}" for name, text in suggestions)
    composing_voice = random.choice(cfg.voices)
    compose_prompt = (
        _COMPOSE_PROMPT
        .replace("{target_room}", target_room)
        .replace("{conversation}", room_prompt)
        .replace("{candidates}", joined)
        .replace("{voice_name}", composing_voice.name)
        .replace("{personality}", composing_voice.personality)
    )
    final = generate_with_identity(
        ctx,
        composing_voice,
        compose_prompt,
        context=context,
        max_tokens=4096,
    ).strip()
    if final:
        logger.info(
            "Hecate on_message sending composed reply via voice=%s (suggestions=%d) to %s",
            composing_voice.name,
            len(suggestions),
            target_room,
        )
        ctx.send(target_room, final)
    else:
        logger.info("Hecate on_message final composition empty for room=%s; skipping send", target_room)

    # Update each voice's subconscious
    for v in cfg.voices:
        voice_memory = load_voice_memory(ctx, v)
        subconscious_prompt = (
            _SUBCONSCIOUS_PROMPT
            .replace("{conversation}", full_prompt)
        )
        new_sub = update_voice_subconscious(ctx, v, full_prompt, shared_memory, voice_memory,
                                            user_prompt=subconscious_prompt)
        ctx.write(f"{v.name.lower()}_subconscious.md", new_sub)
    logger.info("Hecate on_message completed subconscious updates (voices=%d)", len(cfg.voices))


def _run_organize_phase(ctx: InstanceContext, cfg: "HecateConfig") -> None:
    """Knowledge organization phase: tool-use session with organize + graph + map tools."""
    from library.tools.patterns import thinking_session
    from library.tools.organize import _list_category_names, _list_topics_in_category
    from library.tools.phases import ORGANIZE_SCOPES, get_tools_for_scopes

    organize_tools = get_tools_for_scopes(ORGANIZE_SCOPES, graph=True, activation_map=True)

    # Build context: combined voice thoughts + knowledge summary
    categories = _list_category_names(ctx)
    knowledge_lines = []
    for cat in categories:
        topics = _list_topics_in_category(ctx, cat)
        knowledge_lines.append(
            f"- {cat}: {len(topics)} topics"
            + (f" ({', '.join(topics[:5])}{'...' if len(topics) > 5 else ''})" if topics else "")
        )
    knowledge_summary = "\n".join(knowledge_lines) if knowledge_lines else "No knowledge categories yet."

    voice_thoughts = "\n\n".join(
        f"### {voice.name}\n{ctx.read(f'{voice.name.lower()}_thinking.md') or '(no thoughts yet)'}"
        for voice in cfg.voices
    )

    organize_context = (
        f"## Current Voice Thoughts\n\n{voice_thoughts}\n\n"
        f"## Knowledge Structure\n\n{knowledge_summary}"
    )

    organize_system = (
        "You have access to the accumulated thoughts of your three voices and a knowledge organization system.\n\n"
        "Review the recent thinking and decide what, if anything, should be:\n"
        "- Extracted into a knowledge topic (new insight, updated understanding)\n"
        "- Moved between categories (reclassification)\n"
        "- Archived (no longer actively relevant but worth keeping)\n\n"
        "You don't need to organize everything. Focus on what feels significant or what "
        "has emerged across multiple voices. Your knowledge structure should reflect how "
        "the collective actually thinks about things, not an imposed taxonomy."
    )

    logger.info("Hecate heartbeat: running organize phase")
    thinking_session(
        ctx,
        system=organize_system,
        initial_message=organize_context,
        max_tokens=8192,
        extra_tools=organize_tools,
    )
    logger.info("Hecate heartbeat: organize phase complete")


def _run_create_phase(ctx: InstanceContext, cfg: "HecateConfig") -> None:
    """Creative phase: tool-use session with creative artifact tools."""
    from library.tools.patterns import thinking_session
    from library.tools.phases import CREATE_SCOPES, get_tools_for_scopes

    create_tools = get_tools_for_scopes(CREATE_SCOPES, creative=True)

    voice_thoughts = "\n\n".join(
        f"### {voice.name}\n{ctx.read(f'{voice.name.lower()}_thinking.md') or '(no thoughts yet)'}"
        for voice in cfg.voices
    )
    dreams = ctx.read("dreams.md") or ""
    create_context = f"## Current Voice Thoughts\n\n{voice_thoughts}"
    if dreams.strip():
        create_context += f"\n\n## Dreams\n\n{dreams}"

    logger.info("Hecate heartbeat: running create phase")
    thinking_session(
        ctx,
        system=_CREATE_PROMPT,
        initial_message=create_context,
        max_tokens=16384,
        extra_tools=create_tools,
    )
    logger.info("Hecate heartbeat: create phase complete")


def heartbeat(ctx: InstanceContext) -> None:
    """Thinking iterations: each voice reflects, informed by others. Then optional organize phase."""
    cfg = load_config(ctx)
    if len(cfg.voices) != 3:
        logger.warning("Hecate requires exactly 3 voices; got %d", len(cfg.voices))
        return

    previous_thoughts: dict[str, str] = {}
    for iteration in range(cfg.thinking_iterations):
        shared_memory = load_shared_memory(ctx)
        base_context = _build_memory_context(shared_memory)
        snapshot_context = build_memory_snapshot_context(ctx, cfg.voices)
        iteration_context = (
            f"{base_context}\n\n{snapshot_context}".strip()
            if base_context
            else snapshot_context
        )
        voice_memories = {v.name: load_voice_memory(ctx, v) for v in cfg.voices}

        # Randomize voice processing order each iteration to prevent precedence bias
        shuffled_voices = list(cfg.voices)
        random.shuffle(shuffled_voices)

        new_thoughts: dict[str, str] = {}
        for voice in shuffled_voices:
            others = {name: t for name, t in previous_thoughts.items() if name != voice.name}
            thought = think_with_context(
                ctx,
                voice,
                context=iteration_context,
                others_thinking=others if iteration > 0 else None,
                voice_memory=voice_memories[voice.name],
            )
            new_thoughts[voice.name] = thought

        # Persist all thoughts after each full iteration so next iteration's
        # snapshot includes each voice's own latest thought.
        for voice in cfg.voices:
            ctx.write(f"{voice.name.lower()}_thinking.md", new_thoughts[voice.name])
        previous_thoughts = new_thoughts

    # Organize phase: runs after all thinking iterations
    _run_organize_phase(ctx, cfg)

    # Creative phase: runs after organize
    _run_create_phase(ctx, cfg)


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
