"""Consilium toolkit — 5-persona + ghost config, memory helpers, pipeline phases."""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from library.tools.identity import Identity, parse_model
from library.tools.deliberate import generate_with_identity

if TYPE_CHECKING:
    from library.harness.context import InstanceContext

logger = logging.getLogger(__name__)

_SPECIES_DIR = Path(__file__).parent.parent / "species" / "consilium"
_PROMPT_DRAFT = (_SPECIES_DIR / "prompts/draft.md").read_text()
_PROMPT_GHOST_DRAFT = (_SPECIES_DIR / "prompts/ghost_draft.md").read_text()
_PROMPT_REDUCE = (_SPECIES_DIR / "prompts/reduce.md").read_text()
_PROMPT_MERGE = (_SPECIES_DIR / "prompts/merge.md").read_text()
_PROMPT_TRANSFORM = (_SPECIES_DIR / "prompts/transform.md").read_text()
_PROMPT_REVIEW = (_SPECIES_DIR / "prompts/review.md").read_text()
_PROMPT_SUBCONSCIOUS = (_SPECIES_DIR / "prompts/subconscious.md").read_text()
_PROMPT_GHOST_REVIEW = (_SPECIES_DIR / "prompts/ghost_review.md").read_text()


@dataclass
class ConsiliumConfig:
    personas: list[Identity] = field(default_factory=list)
    ghost: Identity = field(default_factory=lambda: Identity(name="Ghost", model="devstral"))
    thinking_iterations: int = 2
    voice_space: str = "main"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(ctx: "InstanceContext") -> ConsiliumConfig:
    raw = ctx.config("consilium") or {}
    if not isinstance(raw, dict):
        raw = {}

    personas = []
    for p in raw.get("personas", []):
        if not isinstance(p, dict):
            continue
        model_str = str(p.get("model", "devstral"))
        provider, model = parse_model(model_str)
        personas.append(Identity(
            name=str(p.get("name", "Persona")),
            model=model or "devstral",
            provider=provider,
            personality=str(p.get("personality", "")),
        ))

    ghost_raw = raw.get("ghost", {})
    if isinstance(ghost_raw, dict):
        ghost_model_str = str(ghost_raw.get("model", "devstral"))
        ghost_provider, ghost_model = parse_model(ghost_model_str)
        ghost = Identity(name="Ghost", model=ghost_model or "devstral", provider=ghost_provider)
    else:
        ghost = Identity(name="Ghost", model="devstral")

    try:
        thinking_iterations = int(raw.get("thinking_iterations", 2))
    except (TypeError, ValueError):
        thinking_iterations = 2

    return ConsiliumConfig(
        personas=personas,
        ghost=ghost,
        thinking_iterations=max(1, thinking_iterations),
        voice_space=str(raw.get("voice_space", "main")),
    )


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------


def load_persona_memory(ctx: "InstanceContext", persona: Identity) -> dict[str, str]:
    """Read per-persona files: {name}_thinking.md, {name}_reviews.md, {name}_subconscious.md."""
    name = persona.name.lower()
    result = {}
    for suffix in ("thinking", "reviews", "subconscious"):
        fname = f"{name}_{suffix}.md"
        content = ctx.read(fname)
        if content:
            result[suffix] = content
    return result


def load_shared_memory(ctx: "InstanceContext") -> dict[str, str]:
    """Load shared files: memory.md, constitution.md."""
    result = {}
    for fname in ("memory.md", "constitution.md"):
        content = ctx.read(fname)
        if content:
            result[fname] = content
    return result


def build_shared_context(shared: dict[str, str]) -> str:
    """Build context string from shared memory files."""
    parts = []
    if shared.get("memory.md"):
        parts.append(f"## Shared Memory\n{shared['memory.md']}")
    if shared.get("constitution.md"):
        parts.append(f"## Constitution\n{shared['constitution.md']}")
    return "\n\n".join(parts)


def build_thinking_context(
    shared: dict[str, str],
    persona_mem: dict[str, str],
    others_thoughts: dict[str, str],
    own_name: str,
    recent_msgs: str = "",
) -> str:
    """Build initial message for a persona's thinking session."""
    parts = []
    ctx_str = build_shared_context(shared)
    if ctx_str:
        parts.append(ctx_str)
    if recent_msgs.strip():
        parts.append(f"## Messages Received Since Last Thinking\n{recent_msgs.strip()}")
    if persona_mem.get("subconscious"):
        parts.append(f"## Your Subconscious\n{persona_mem['subconscious']}")
    if persona_mem.get("reviews"):
        parts.append(f"## Your Recent Reviews\n{persona_mem['reviews']}")
    if others_thoughts:
        names = list(others_thoughts.keys())
        random.shuffle(names)
        lines = [
            f"**{n}**: {others_thoughts[n]}"
            for n in names
            if n != own_name and others_thoughts[n]
        ]
        if lines:
            parts.append(
                "## Other Personas' Thoughts (read-only)\n" + "\n\n".join(lines)
            )
    return "\n\n".join(parts) or "Begin your thinking session."


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _parse_ghost_lines(raw: str) -> list[str]:
    """Parse numbered or newline-separated lines from ghost output."""
    lines = []
    for line in raw.strip().splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        # Strip leading numbering like "1." or "1)" or "- "
        cleaned = re.sub(r"^\d+[.)]\s*", "", cleaned)
        cleaned = re.sub(r"^[-*]\s+", "", cleaned)
        cleaned = cleaned.strip()
        if cleaned:
            lines.append(cleaned)
    return lines


# ---------------------------------------------------------------------------
# Pipeline phases
# ---------------------------------------------------------------------------


def run_drafting_phase(
    ctx: "InstanceContext",
    cfg: ConsiliumConfig,
    conversation: str,
    target_room: str,
    context: str,
    persona_extra_context: str = "",
) -> list[tuple[str, str]]:
    """Phase 1: 5 persona drafts + 3 ghost one-liners = up to 8 candidates.

    Returns list of (author_id, text) tuples.
    persona_extra_context: additional context injected for persona drafts only (not ghost).
    """
    candidates: list[tuple[str, str]] = []

    persona_context = (context + "\n\n" + persona_extra_context).strip() if persona_extra_context else context

    for persona in cfg.personas:
        prompt = (
            _PROMPT_DRAFT
            .replace("{conversation}", conversation)
            .replace("{target_room}", target_room)
            .replace("{persona_name}", persona.name)
        )
        draft = generate_with_identity(
            ctx, persona, prompt, context=persona_context, max_tokens=2048,
        )
        if draft.strip():
            candidates.append((persona.name, draft.strip()))
        else:
            logger.warning("Consilium drafting: empty draft from persona %s", persona.name)

    ghost_prompt = (
        _PROMPT_GHOST_DRAFT
        .replace("{conversation}", conversation)
        .replace("{target_room}", target_room)
    )
    ghost_raw = generate_with_identity(
        ctx, cfg.ghost, ghost_prompt, context=context, max_tokens=1024,
    )
    ghost_lines = _parse_ghost_lines(ghost_raw)[:3]
    if not ghost_lines:
        logger.warning("Consilium drafting: ghost produced no lines (raw=%d chars)", len(ghost_raw))
    for i, line in enumerate(ghost_lines):
        candidates.append((f"Ghost-{i + 1}", line))

    logger.info("Consilium drafting: %d candidates (personas=%d ghost=%d)", len(candidates), len(cfg.personas), len(ghost_lines))
    return candidates


def run_reduction_phase(
    ctx: "InstanceContext",
    cfg: ConsiliumConfig,
    candidates: list[tuple[str, str]],
    conversation: str,
    target_room: str,
    context: str,
) -> tuple[list[str], Identity]:
    """Phase 2: 4 random personas each reduce a subset of candidates to bullet summaries.

    Returns (list of bullet summaries, excluded 5th persona).
    """
    shuffled = list(cfg.personas)
    random.shuffle(shuffled)
    reducers = shuffled[:4]
    excluded = shuffled[4]

    shuffled_candidates = list(candidates)
    random.shuffle(shuffled_candidates)

    # Distribute candidates round-robin across reducers
    per_reducer: list[list[tuple[str, str]]] = [[] for _ in range(len(reducers))]
    for i, c in enumerate(shuffled_candidates):
        per_reducer[i % len(reducers)].append(c)

    summaries: list[str] = []
    for reducer, chunk in zip(reducers, per_reducer):
        if not chunk:
            continue
        pair_text = "\n\n".join(f"**{author}**: {text}" for author, text in chunk)
        prompt = (
            _PROMPT_REDUCE
            .replace("{conversation}", conversation)
            .replace("{target_room}", target_room)
            .replace("{candidates}", pair_text)
            .replace("{persona_name}", reducer.name)
        )
        summary = generate_with_identity(
            ctx, reducer, prompt, context=context, max_tokens=1024,
        )
        if summary.strip():
            summaries.append(summary.strip())

    logger.info(
        "Consilium reduction: %d summaries, excluded=%s",
        len(summaries), excluded.name,
    )
    return summaries, excluded


def run_merge_phase(
    ctx: "InstanceContext",
    cfg: ConsiliumConfig,
    summaries: list[str],
    conversation: str,
    target_room: str,
    context: str,
) -> str:
    """Phase 3: ghost merges summaries into a structured spec."""
    summaries_text = "\n\n".join(
        f"Summary {i + 1}:\n{s}" for i, s in enumerate(summaries)
    )
    prompt = (
        _PROMPT_MERGE
        .replace("{conversation}", conversation)
        .replace("{target_room}", target_room)
        .replace("{summaries}", summaries_text)
    )
    spec = generate_with_identity(
        ctx, cfg.ghost, prompt, context=context, max_tokens=2048,
    )
    logger.info("Consilium merge: spec=%d chars", len(spec))
    return spec.strip()


def run_transform_phase(
    ctx: "InstanceContext",
    cfg: ConsiliumConfig,
    spec: str,
    transformer: Identity,
    conversation: str,
    target_room: str,
    context: str,
) -> str:
    """Phase 4: excluded persona transforms spec into final reply."""
    prompt = (
        _PROMPT_TRANSFORM
        .replace("{conversation}", conversation)
        .replace("{target_room}", target_room)
        .replace("{spec}", spec)
        .replace("{persona_name}", transformer.name)
    )
    final = generate_with_identity(
        ctx, transformer, prompt, context=context, max_tokens=4096,
    )
    logger.info(
        "Consilium transform: transformer=%s final=%d chars",
        transformer.name, len(final),
    )
    return final.strip()


def run_review_phase(
    ctx: "InstanceContext",
    cfg: ConsiliumConfig,
    sent_response: str,
    conversation: str,
    context: str,
) -> None:
    """Phase 5: each persona appends one-line analysis to {name}_reviews.md."""
    for persona in cfg.personas:
        prompt = (
            _PROMPT_REVIEW
            .replace("{conversation}", conversation)
            .replace("{sent_response}", sent_response)
            .replace("{persona_name}", persona.name)
        )
        analysis = generate_with_identity(
            ctx, persona, prompt, context=context, max_tokens=256,
        )
        one_line = analysis.strip().split("\n")[0].strip()
        if one_line:
            fname = f"{persona.name.lower()}_reviews.md"
            existing = ctx.read(fname) or ""
            lines = [l for l in existing.strip().splitlines() if l.strip()]
            lines.append(one_line)
            lines = lines[-10:]  # Keep reviews brief
            ctx.write(fname, "\n".join(lines) + "\n")

    logger.info("Consilium review: %d personas reviewed", len(cfg.personas))


def load_ghost_context(ctx: "InstanceContext", cfg: ConsiliumConfig) -> str:
    """Load ideas.md and recommendations.md for injection into persona drafts."""
    parts = []
    ideas = ctx.read("ideas.md")
    if ideas and ideas.strip():
        parts.append(f"## Ideas\n{ideas.strip()}")
    recs = ctx.read("recommendations.md")
    if recs and recs.strip():
        parts.append(f"## Recommendations\n{recs.strip()}")
    return "\n\n".join(parts)


def run_ghost_review_phase(
    ctx: "InstanceContext",
    cfg: ConsiliumConfig,
    meta_thinking: str,
) -> None:
    """Ghost subconscious review: reads persona thinking + meta-thinking + memory,
    then updates ideas.md and recommendations.md with short bullet lists.
    """
    import json as _json

    shared = load_shared_memory(ctx)
    context_parts = []
    if shared.get("memory.md"):
        context_parts.append(f"## Memory\n{shared['memory.md']}")
    if shared.get("constitution.md"):
        context_parts.append(f"## Constitution\n{shared['constitution.md']}")
    if meta_thinking.strip():
        context_parts.append(f"## Persona Thinking\n{meta_thinking.strip()}")
    context = "\n\n".join(context_parts)

    raw = generate_with_identity(
        ctx, cfg.ghost, _PROMPT_GHOST_REVIEW,
        context=context, max_tokens=1024,
    )

    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
        data = _json.loads(cleaned)
        ideas = str(data.get("ideas", "")).strip()
        recs = str(data.get("recommendations", "")).strip()
        if ideas:
            ctx.write("ideas.md", ideas + "\n")
        if recs:
            ctx.write("recommendations.md", recs + "\n")
        logger.info(
            "Consilium ghost review: updated ideas.md (%d chars) recommendations.md (%d chars)",
            len(ideas), len(recs),
        )
    except Exception as exc:
        logger.warning("Consilium ghost review: failed to parse output (%s)", exc)


def update_persona_subconscious(
    ctx: "InstanceContext",
    persona: Identity,
    conversation: str,
    shared: dict[str, str],
) -> None:
    """Update {name}_subconscious.md with brief snapshot."""
    context = build_shared_context(shared)
    prompt = _PROMPT_SUBCONSCIOUS.replace("{conversation}", conversation)
    result = generate_with_identity(
        ctx, persona, prompt, context=context, max_tokens=512,
    )
    if result.strip():
        ctx.write(f"{persona.name.lower()}_subconscious.md", result.strip())
