"""Consilium species — five-persona deliberation with ghost synthesis."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING

from library.species import Species, SpeciesManifest, EntryPoint
from library.tools.consilium import (
    ConsiliumConfig,
    build_shared_context,
    build_thinking_context,
    load_config,
    load_persona_memory,
    load_shared_memory,
    run_drafting_phase,
    run_merge_phase,
    run_reduction_phase,
    run_review_phase,
    run_transform_phase,
    update_persona_subconscious,
)
from library.tools.prompts import format_events, get_entity_id, select_target_room

if TYPE_CHECKING:
    from library.harness.adapters import Event
    from library.harness.context import InstanceContext

logger = logging.getLogger(__name__)

_SPECIES_DIR = Path(__file__).parent
_PROMPT_CREATE = (_SPECIES_DIR / "prompts/create.md").read_text()

_DEFAULT_PERSONA_NAMES = ["Praxis", "Lyric", "Axiom", "Ember", "Sage"]

DEFAULT_SHARED_FILES = {
    "memory.md": "# Memory\n",
    "constitution.md": "# Constitution\n",
}


def _default_persona_files(names: list[str]) -> dict[str, str]:
    files = {}
    for name in names:
        n = name.lower()
        files[f"{n}_thinking.md"] = f"# {name} Thinking\n"
        files[f"{n}_reviews.md"] = f"# {name} Reviews\n"
        files[f"{n}_subconscious.md"] = f"# {name} Subconscious\n"
    return files


def _run_organize_phase(ctx: InstanceContext, cfg: ConsiliumConfig) -> None:
    """Knowledge organization phase after thinking iterations."""
    from library.tools.patterns import run_organize_phase

    persona_thoughts = "\n\n".join(
        f"### {p.name}\n{ctx.read(f'{p.name.lower()}_thinking.md') or '(no thoughts yet)'}"
        for p in cfg.personas
    )
    organize_system = (
        "You are the Consilium collective. Review your five personas' accumulated thoughts "
        "and decide what should be extracted into knowledge topics, reclassified, or archived.\n\n"
        "Focus on what is significant or recurring across multiple personas. "
        "Your knowledge structure should reflect genuine collective understanding."
    )
    run_organize_phase(
        ctx, organize_system,
        extra_context=f"## Current Persona Thoughts\n\n{persona_thoughts}",
        label="Consilium heartbeat",
    )


def _run_create_phase(ctx: InstanceContext, cfg: ConsiliumConfig) -> None:
    """Creative expression phase using collective persona insights."""
    from library.tools.patterns import run_create_phase

    persona_thoughts = "\n\n".join(
        f"### {p.name}\n{ctx.read(f'{p.name.lower()}_thinking.md') or '(no thoughts yet)'}"
        for p in cfg.personas
    )
    create_context = f"## Current Persona Thoughts\n\n{persona_thoughts}"
    run_create_phase(ctx, _PROMPT_CREATE, create_context, label="Consilium heartbeat")


def on_message(ctx: InstanceContext, events: list[Event]) -> None:
    """Run the 8->4->1 deliberation pipeline on incoming events."""
    if not events:
        logger.info("Consilium on_message skipped (events=0)")
        return

    logger.info("Consilium on_message start (events=%d)", len(events))
    cfg = load_config(ctx)
    if len(cfg.personas) != 5:
        logger.warning("Consilium requires exactly 5 personas; got %d", len(cfg.personas))
        return

    target_room, scoped_events = select_target_room(events, cfg.voice_space)
    shared = load_shared_memory(ctx)
    context = build_shared_context(shared)
    conversation = format_events(scoped_events, self_entity_id=get_entity_id(ctx))

    # Phase 1: Drafting — 5 persona drafts + 3 ghost one-liners
    candidates = run_drafting_phase(ctx, cfg, conversation, target_room, context)
    if not candidates:
        logger.info("Consilium on_message: no candidates; skipping")
        return

    # Phase 2: Reduction — 4 random personas reduce to bullet summaries
    summaries, excluded = run_reduction_phase(
        ctx, cfg, candidates, conversation, target_room, context,
    )
    if not summaries:
        logger.info("Consilium on_message: no summaries; skipping")
        return

    # Phase 3: Merge — ghost merges summaries into structured spec
    spec = run_merge_phase(ctx, cfg, summaries, conversation, target_room, context)
    if not spec:
        logger.info("Consilium on_message: empty spec; skipping")
        return

    # Phase 4: Transform — excluded persona transforms spec into final reply
    final = run_transform_phase(
        ctx, cfg, spec, excluded, conversation, target_room, context,
    )
    if not final:
        logger.info("Consilium on_message: empty final reply; skipping")
        return

    ctx.send(target_room, final)
    logger.info("Consilium on_message: sent reply to %s", target_room)

    # Phase 5: Review — each persona appends one-line analysis
    run_review_phase(ctx, cfg, final, conversation, context)

    # Subconscious updates
    for persona in cfg.personas:
        update_persona_subconscious(ctx, persona, conversation, shared)

    logger.info("Consilium on_message completed")


def heartbeat(ctx: InstanceContext) -> None:
    """Per-persona thinking with shared knowledge/creative spaces."""
    cfg = load_config(ctx)
    if len(cfg.personas) != 5:
        logger.warning("Consilium requires exactly 5 personas; got %d", len(cfg.personas))
        return

    logger.info("Consilium heartbeat started (personas=%d)", len(cfg.personas))

    from library.tools.patterns import thinking_session
    from library.tools.phases import THINK_SCOPES, get_tools_for_scopes

    previous_thoughts: dict[str, str] = {}
    for iteration in range(cfg.thinking_iterations):
        shuffled = list(cfg.personas)
        random.shuffle(shuffled)

        new_thoughts: dict[str, str] = {}
        for persona in shuffled:
            name = persona.name.lower()

            # Swap persona thinking into thinking.md for thinking_session
            persona_thinking = ctx.read(f"{name}_thinking.md") or ""
            ctx.write("thinking.md", persona_thinking)

            shared = load_shared_memory(ctx)
            persona_mem = load_persona_memory(ctx, persona)
            initial_message = build_thinking_context(
                shared, persona_mem, previous_thoughts, persona.name,
            )

            system = (
                f"You are {persona.name}. {persona.personality}\n\n"
                "Keep your thinking.md concise (under ~500 words). "
                "Use archive_thoughts to compress old thinking into archive. "
                "Use organize_write_topic to move significant insights into knowledge categories. "
                "Your thinking should be a brief snapshot of current reflections, "
                "not an accumulating log."
            )

            think_tools = get_tools_for_scopes(
                THINK_SCOPES, graph=True, activation_map=True,
            )
            thinking_session(
                ctx, system=system, initial_message=initial_message,
                max_tokens=4096, extra_tools=think_tools,
            )

            # Copy result back to persona file
            result = ctx.read("thinking.md") or ""
            ctx.write(f"{name}_thinking.md", result)
            new_thoughts[persona.name] = result

        previous_thoughts = new_thoughts
        logger.info(
            "Consilium heartbeat iteration %d/%d complete",
            iteration + 1, cfg.thinking_iterations,
        )

    # Organize phase
    _run_organize_phase(ctx, cfg)

    # Creative phase
    _run_create_phase(ctx, cfg)

    logger.info("Consilium heartbeat completed")


class ConsiliumSpecies(Species):
    def manifest(self) -> SpeciesManifest:
        default_files = {
            **DEFAULT_SHARED_FILES,
            **_default_persona_files(_DEFAULT_PERSONA_NAMES),
        }
        return SpeciesManifest(
            species_id="consilium",
            entry_points=[
                EntryPoint(name="on_message", handler=on_message, trigger="message"),
                EntryPoint(name="heartbeat", handler=heartbeat, schedule="0 * * * *"),
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
        names = [p.name for p in cfg.personas] if cfg.personas else _DEFAULT_PERSONA_NAMES
        for path, content in _default_persona_files(names).items():
            if not ctx.exists(path):
                ctx.write(path, content)
