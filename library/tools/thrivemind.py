"""Single-instance colony toolkit for Thrivemind — colony biology only."""

from __future__ import annotations

import json
import logging
import random
import re
import time
import uuid
from urllib.parse import quote
from typing import TYPE_CHECKING

from library.harness.sanitize import strip_think_blocks
from library.tools.identity import (
    AXES,
    AXIS_NAMES,
    Identity,
    format_persona,
    parse_model,
)
from library.tools.voting import approval_weights, weighted_sample
from library.tools.deliberate import generate_with_identity
from library.tools.prompts import format_events

if TYPE_CHECKING:
    from library.harness.context import InstanceContext

# Backward-compat alias: colony individuals ARE Identity objects
Individual = Identity

COLONY_NAMESPACE = "colony"
logger = logging.getLogger(__name__)
_CODE_FENCE_RE = re.compile(r"(?is)```(?:[a-zA-Z0-9_-]+)?\n(.*?)```")
_WORD_RE = re.compile(r"\b\w+\b")


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def _strip_reasoning_blocks(text: str) -> str:
    cleaned = strip_think_blocks(text)
    # Keep fenced content when it's not explicitly a thinking block.
    cleaned = _CODE_FENCE_RE.sub(lambda m: m.group(1), cleaned)
    return cleaned


def _extract_first_sentence(text: str) -> str:
    match = re.search(r"[.!?](?:\s|$)", text)
    if match:
        return text[: match.end()].strip()
    return text.strip()


def _sanitize_contribution(text: str, max_chars: int = 220) -> str:
    cleaned = _strip_reasoning_blocks(text).replace("\r\n", "\n")
    lines = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line)
        if re.match(r"^(analysis|reasoning|thoughts?|thinking)\s*:", line, flags=re.IGNORECASE):
            continue
        lines.append(line)

    merged = " ".join(lines)
    merged = re.sub(r"(?i)^(?:principle|refinement|proposal)\s*:\s*", "", merged).strip()
    merged = re.sub(r"\s+", " ", merged)
    if not merged:
        return ""

    one_sentence = _extract_first_sentence(merged)
    if len(one_sentence) > max_chars:
        one_sentence = one_sentence[:max_chars].rstrip()
    return one_sentence


def _truncate_to_words(text: str, max_words: int) -> str:
    if _word_count(text) <= max_words:
        return text.strip()
    return " ".join((text or "").split()[:max_words]).strip()


def _truncate_for_prompt(text: str, max_chars: int) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    clipped = cleaned[:max_chars].rsplit(" ", 1)[0].rstrip()
    if not clipped:
        clipped = cleaned[:max_chars].rstrip()
    return f"{clipped}\n...[truncated]"


def _sanitize_constitution_candidate(text: str, max_words: int) -> str:
    cleaned = _strip_reasoning_blocks(text).replace("\r\n", "\n").strip()
    if not cleaned:
        return ""

    cleaned = re.sub(
        r"(?is)^\s*(?:here(?:'s| is)\s+(?:the|a)\s+(?:rewritten|revised)\s+constitution\s*:?)\s*",
        "",
        cleaned,
    )

    lower = cleaned.lower()
    for marker in ("# constitution", "constitution:"):
        idx = lower.find(marker)
        if idx > 0 and idx <= max(240, len(cleaned) // 2):
            cleaned = cleaned[idx:].strip()
            break

    cleaned = _truncate_to_words(cleaned, max_words=max_words)
    return cleaned


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ThrivemindConfig:
    def __init__(
        self,
        colony_size: int = 12,
        suggestion_fraction: float = 0.5,
        approval_threshold: int = 3,
        consensus_threshold: float = 0.6,
        round_timeout_s: int = 30,
        suggestion_model: str = "",
        writer_model: str = "",
        voice_space: str = "main",
    ):
        self.colony_size = colony_size
        self.suggestion_fraction = suggestion_fraction
        self.approval_threshold = approval_threshold
        self.consensus_threshold = consensus_threshold
        self.round_timeout_s = round_timeout_s
        self.suggestion_model = suggestion_model
        self.writer_model = writer_model
        self.voice_space = voice_space


# ---------------------------------------------------------------------------
# Config / colony persistence
# ---------------------------------------------------------------------------


def _coerce_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_config(ctx: InstanceContext) -> ThrivemindConfig:
    raw = ctx.config("thrivemind") or {}
    if not isinstance(raw, dict):
        raw = {}
    return ThrivemindConfig(
        colony_size=max(1, _coerce_int(raw.get("colony_size"), 12)),
        suggestion_fraction=max(0.1, _coerce_float(raw.get("suggestion_fraction"), 0.5)),
        approval_threshold=_coerce_int(raw.get("approval_threshold"), 3),
        consensus_threshold=max(0.0, _coerce_float(raw.get("consensus_threshold"), 0.6)),
        round_timeout_s=_coerce_int(raw.get("round_timeout_s"), 30),
        suggestion_model=str(raw.get("suggestion_model", "")),
        writer_model=str(raw.get("writer_model", "")),
        voice_space=str(raw.get("voice_space", "main")),
    )


def _identity_from_dict(d: dict) -> Identity:
    return Identity(
        name=str(d.get("name") or d.get("id", "")),
        dims={k: float(v) for k, v in d.get("dims", {}).items()},
        approval=int(d.get("approval", 0)),
        created_at=int(d.get("created_at", 0)),
    )


def _identity_to_dict(ind: Identity) -> dict:
    return {
        "name": ind.name,
        "dims": ind.dims or {},
        "approval": ind.approval,
        "created_at": ind.created_at,
    }


def load_colony(ctx: InstanceContext) -> list[Identity]:
    store = ctx.store(COLONY_NAMESPACE)
    result = []
    for _key, value in store.scan():
        if isinstance(value, dict) and ("name" in value or "id" in value):
            result.append(_identity_from_dict(value))
    result.sort(key=lambda i: i.created_at)
    return result


def save_colony(ctx: InstanceContext, colony: list[Identity]) -> None:
    store = ctx.store(COLONY_NAMESPACE)
    for key, _ in store.scan():
        store.delete(key)
    for ind in colony:
        store.put(ind.name, _identity_to_dict(ind))


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def build_colony_snapshot(colony: list[Identity]) -> str:
    """Render a human-readable colony snapshot markdown document."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    lines = [
        "# Colony",
        "",
        f"Generated: {timestamp}",
        f"Individuals: {len(colony)}",
        "",
        "| Individual | Personality | Approval |",
        "| --- | --- | ---: |",
    ]

    if not colony:
        lines.append("| - | - | 0 |")
        return "\n".join(lines) + "\n"

    for ind in sorted(colony, key=lambda i: (-i.approval, i.created_at, i.name)):
        personality = _md_escape(format_persona(ind))
        lines.append(f"| `{ind.name}` | {personality} | {ind.approval} |")

    return "\n".join(lines) + "\n"


def save_colony_snapshot(ctx: InstanceContext, colony: list[Identity]) -> None:
    """Write colony.md snapshot to instance memory storage."""
    content = build_colony_snapshot(colony)
    logger.info(
        "Writing thrivemind colony.md (individuals=%d, chars=%d)",
        len(colony),
        len(content),
    )
    ctx.write("colony.md", content)


def _reflection_path(individual: Identity | str) -> str:
    name = individual.name if isinstance(individual, Identity) else str(individual)
    return f"reflections/{quote(name, safe='-_.,')}.md"


def load_reflection(ctx: InstanceContext, individual: Identity) -> str:
    return ctx.read(_reflection_path(individual))


def save_reflection(ctx: InstanceContext, individual: Identity, text: str) -> None:
    path = _reflection_path(individual)
    logger.info(
        "Writing thrivemind %s (individual=%s, chars=%d)",
        path,
        individual.name,
        len(text or ""),
    )
    ctx.write(path, text)


def summarize_message_history(
    ctx: InstanceContext,
    events: list,
    cfg: ThrivemindConfig,
) -> str:
    """Summarize current message history once for a shared colony context."""
    if not events:
        return ""

    provider, model = parse_model(cfg.suggestion_model) if cfg.suggestion_model else (None, "")
    summarizer = Identity(
        name="ColonySummarizer",
        model=model,
        provider=provider,
        personality="Concise colony historian focused on decision-relevant context.",
    )
    prompt = (
        "Summarize this message history for colony decision-making.\n"
        "Keep it brief (4-8 bullet points), factual, and focused on stakes and requests.\n\n"
        f"{format_events(events)}"
    )
    summary = generate_with_identity(ctx, summarizer, prompt, model=cfg.suggestion_model).strip()
    logger.info("Generated thrivemind message summary (chars=%d)", len(summary))
    return summary


_REFLECTION_CONTEXT_LIMITS = {
    "colony_md": 1000,
    "constitution": 2000,
    "prior_reflection": 800,
    "message_summary": 600,
}


def build_reflection_context(
    colony_md: str,
    constitution: str,
    prior_reflection: str,
    message_summary: str,
) -> str:
    parts = []
    if colony_md:
        cm = (colony_md or "").strip()
        max_cm = _REFLECTION_CONTEXT_LIMITS["colony_md"]
        if len(cm) > max_cm:
            cm = cm[:max_cm] + "\n...[truncated]"
        parts.append(f"## Colony\n{cm}")
    if constitution:
        cs = (constitution or "").strip()
        max_cs = _REFLECTION_CONTEXT_LIMITS["constitution"]
        if len(cs) > max_cs:
            cs = cs[:max_cs] + "\n...[truncated]"
        parts.append(f"## Constitution\n{cs}")
    if prior_reflection:
        pr = (prior_reflection or "").strip()
        max_pr = _REFLECTION_CONTEXT_LIMITS["prior_reflection"]
        if len(pr) > max_pr:
            pr = pr[-max_pr:]  # tail: keep most recent content
        parts.append(f"## Prior Reflection\n{pr}")
    if message_summary:
        ms = (message_summary or "").strip()
        max_ms = _REFLECTION_CONTEXT_LIMITS["message_summary"]
        if len(ms) > max_ms:
            ms = ms[:max_ms] + "\n...[truncated]"
        parts.append(f"## Message Summary\n{ms}")
    return "\n\n".join(parts)


def reflect_on_colony(
    ctx: InstanceContext,
    cfg: ThrivemindConfig,
    individual: Identity,
    colony: list[Identity],
    constitution: str,
    prior_reflection: str,
    message_summary: str,
) -> str:
    colony_md = ctx.read("colony.md") or build_colony_snapshot(colony)
    context = build_reflection_context(
        colony_md=colony_md,
        constitution=constitution,
        prior_reflection=prior_reflection,
        message_summary=message_summary,
    )
    reflection = generate_with_identity(
        ctx,
        individual,
        "This is your moment to reflect on the state of your colony.",
        context=context,
        model=cfg.suggestion_model,
    ).strip()
    logger.info(
        "Generated thrivemind reflection (individual=%s, chars=%d)",
        individual.name,
        len(reflection),
    )
    return reflection


def spawn_initial_colony(
    cfg: ThrivemindConfig, rng: random.Random | None = None
) -> list[Identity]:
    rng = rng or random.Random()
    colony = []
    now = int(time.time())
    for _ in range(cfg.colony_size):
        dims = {name: round(rng.uniform(-1.0, 1.0), 4) for name in AXIS_NAMES}
        colony.append(Identity(name=str(uuid.uuid4()), dims=dims, approval=0, created_at=now))
    return colony


# ---------------------------------------------------------------------------
# Colony selection
# ---------------------------------------------------------------------------


def select_suggesters(
    colony: list[Identity],
    n: int,
    rng: random.Random | None = None,
) -> list[Identity]:
    """Select n individuals weighted by approval (approval+1, min 0)."""
    rng = rng or random.Random()
    weights = approval_weights(colony)
    return weighted_sample(colony, weights, n, rng)


# ---------------------------------------------------------------------------
# Constitution management
# ---------------------------------------------------------------------------


def contribute_constitution_line(
    ctx: InstanceContext,
    individual: Identity,
    current_constitution: str,
    cfg: ThrivemindConfig,
    reflections: str = "",
) -> str:
    persona = format_persona(individual)
    constitution_excerpt = _truncate_for_prompt(current_constitution, max_chars=2400)
    reflections_excerpt = _truncate_for_prompt(reflections, max_chars=900)
    reflections_block = f"\n\nReflections:\n{reflections_excerpt}" if reflections_excerpt else ""
    prompt = (
        f"You are {individual.name}, one individual in a colony that is collectively drafting a constitution to guide its future decisions and values.\n"
        f"Your personality: {persona}\n\n"
        f"Current constitution:\n{constitution_excerpt}\n\n"
        f"{reflections_block}"
        "Propose exactly one new principle or refinement.\n"
        "Constraints:\n"
        "- Keep this limited to 1 line.\n"
        "- Maximum 18 words.\n"
        "- One sentence only.\n"
        "- Start directly with the principle text.\n"
        "- No analysis or meta commentary.\n"
        "Return only the principle sentence."
    )
    raw = generate_with_identity(
        ctx,
        individual,
        prompt,
        context=(
            "You are contributing one principle or value to your colony's shared constitution.\n"
            "Keep it concise and practical.\n"
            "Your first character should be the principle sentence."
        ),
        model=cfg.suggestion_model,
        max_tokens=2048,
    )
    line = _sanitize_contribution(raw, max_chars=220)
    if not line:
        retry_prompt = (
            f"You are {individual.name}, one individual in a colony that is collectively drafting a constitution to guide its future decisions and values.\n"
            f"Your personality: {persona}\n\n"
            "Your previous response was invalid.\n"
            "Return exactly one plain sentence proposing one principle.\n"
            "Constraints:\n"
            "- Maximum 18 words.\n"
            "- One sentence only.\n"
            "- Start directly with the sentence.\n"
            "- No labels or commentary.\n"
            "Return only the sentence."
        )
        retry_raw = generate_with_identity(
            ctx,
            individual,
            retry_prompt,
            context="Final-answer mode.",
            model=cfg.suggestion_model,
            max_tokens=2048,
        )
        retry_line = _sanitize_contribution(retry_raw, max_chars=220)
        if retry_line:
            logger.info(
                "Recovered constitution contribution after empty sanitized output (individual=%s, retry_chars=%d)",
                individual.name,
                len(retry_line),
            )
            line = retry_line
    if line != (raw or "").strip():
        logger.info(
            "Sanitized constitution contribution output (individual=%s, raw_chars=%d, final_chars=%d)",
            individual.name,
            len(raw or ""),
            len(line),
        )
    return line


def rewrite_constitution(
    ctx: InstanceContext,
    lines: list[str],
    current_constitution: str,
    cfg: ThrivemindConfig,
) -> str:
    combined = "\n".join(f"- {line}" for line in lines if line.strip())
    constitution_excerpt = _truncate_for_prompt(current_constitution, max_chars=5000)
    combined_excerpt = _truncate_for_prompt(combined, max_chars=2600)
    input_word_count = _word_count(current_constitution) + _word_count(combined)
    target_words = max(20, min(500, input_word_count))
    min_words = max(12, int(target_words * 0.7))
    max_words = max(min_words + 8, int(target_words * 1.3))
    rewrite_max_tokens = max(2048, min(8192, max_words * 4))

    # Use a writer identity with the writer model
    provider, model = parse_model(cfg.writer_model) if cfg.writer_model else (None, "")
    writer = Identity(
        name="ColonyWriter",
        model=model,
        provider=provider,
        personality="You are a scribe immortalizing your colony's principles into a concise constitution.",
    )
    prompt = (
        f"Current constitution:\n{constitution_excerpt}\n\n"
        f"Proposed principles:\n{combined_excerpt}\n\n"
        "Refine the principles into a new constitution.\n"
        "Constraints:\n"
        "- Output only the final constitution text.\n"
        "- Start directly with the constitution, no preface.\n"
        "- Do not include thinking, analysis, or meta commentary.\n"
        f"- Keep length similar to input (target ~{target_words} words, acceptable range {min_words}-{max_words}).\n"
        "- Keep concise headings/bullets if useful, but avoid repetition."
    )
    raw = generate_with_identity(
        ctx,
        writer,
        prompt,
        model=cfg.writer_model,
        max_tokens=rewrite_max_tokens,
    )
    rewritten = _sanitize_constitution_candidate(raw, max_words=max_words)
    if not rewritten:
        rewritten = combined
    if rewritten != (raw or "").strip():
        logger.info(
            "Sanitized constitution candidate output (raw_chars=%d, final_chars=%d)",
            len(raw or ""),
            len(rewritten),
        )
    if rewritten:
        return rewritten
    fallback = current_constitution.strip() or "# Constitution\n"
    logger.warning("Constitution rewrite became empty after sanitization; falling back to current constitution")
    return fallback


def vote_constitution(
    ctx: InstanceContext,
    individual: Identity,
    current: str,
    proposed: str,
    cfg: ThrivemindConfig,
    reflections: str = "",
    round_context: str = "",
) -> bool:
    persona = format_persona(individual)
    current_excerpt = _truncate_for_prompt(current, max_chars=3200)
    proposed_excerpt = _truncate_for_prompt(proposed, max_chars=3200)
    reflections_excerpt = _truncate_for_prompt(reflections, max_chars=1000)
    reflections_block = f"\n\nReflections:\n{reflections_excerpt}" if reflections_excerpt else ""
    prompt_parts = [
        f"You are {individual.name}, one individual in a colony that is collectively voting on whether to adopt a proposed constitution that will guide its future decisions and values.\n"
        f"Personality: {persona}\n\n"
        f"Current:\n{current_excerpt}\n\n"
        f"Proposed:\n{proposed_excerpt}\n\n"
        f"{reflections_block}"
        "Stakes: individuals who vote with the ultimately winning direction gain approval; "
        "individuals who back the losing direction lose approval.\n"
        "Vote strategically for the colony's long-term survival.\n\n"
        'Reply with JSON: {"accept": true} or {"accept": false}'
    ]
    if round_context.strip():
        round_context_excerpt = _truncate_for_prompt(round_context.strip(), max_chars=1400)
        prompt_parts.append(f"\n\nAdditional context:\n{round_context_excerpt}")
    prompt = "".join(prompt_parts)
    raw = generate_with_identity(
        ctx,
        individual,
        prompt,
        context="You are voting on whether to adopt a new constitution for your colony.",
        model=cfg.suggestion_model,
    )
    try:
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(raw)
        return bool(data.get("accept", False))
    except Exception:
        return False


def save_constitution(ctx: InstanceContext, text: str) -> None:
    logger.info(
        "Writing thrivemind constitution.md (chars=%d)",
        len(text or ""),
    )
    ctx.write("constitution.md", text)


def build_contributions_snapshot(lines: list[str]) -> str:
    """Render the raw constitution contributions for visibility."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    out = [
        "# Contributions",
        "",
        f"Generated: {timestamp}",
        f"Count: {len(lines)}",
        "",
    ]
    if not lines:
        out.append("(no contributions)")
    else:
        for idx, line in enumerate(lines, start=1):
            out.append(f"{idx}. {line}")
    out.append("")
    return "\n".join(out)


def save_contributions(ctx: InstanceContext, lines: list[str]) -> None:
    content = build_contributions_snapshot(lines)
    logger.info(
        "Writing thrivemind contributions.md (count=%d, chars=%d)",
        len(lines),
        len(content),
    )
    ctx.write("contributions.md", content)


def save_candidate(ctx: InstanceContext, candidate: str) -> None:
    logger.info("Writing thrivemind candidate.md (chars=%d)", len(candidate or ""))
    ctx.write("candidate.md", candidate)


def load_constitution(ctx: InstanceContext) -> str:
    existing = ctx.read("constitution.md")
    if existing:
        return existing
    logger.info("No thrivemind constitution.md found; using default scaffold")
    return "# Constitution\n"


# ---------------------------------------------------------------------------
# Spawn cycle
# ---------------------------------------------------------------------------


def run_spawn_cycle(
    colony: list[Identity],
    cfg: ThrivemindConfig,
    rng: random.Random | None = None,
) -> list[Identity]:
    """Eligible individuals spawn offspring and are replaced."""
    rng = rng or random.Random()
    if not colony:
        return colony

    eligible = [ind for ind in colony if ind.approval >= cfg.approval_threshold]
    if not eligible:
        return colony

    total_approval = sum(ind.approval for ind in eligible)
    if total_approval == 0:
        return colony

    to_remove_names: set[str] = set()
    new_individuals: list[Identity] = []
    now = int(time.time())

    for primary in eligible:
        weights = [ind.approval / total_approval for ind in eligible]
        (other,) = rng.choices(eligible, weights=weights, k=1)

        offspring_dims: dict[str, float] = {}
        for name in AXIS_NAMES:
            roll = rng.random()
            if roll < 0.15:
                offspring_dims[name] = round(rng.uniform(-1.0, 1.0), 4)
            elif roll < 0.40:
                offspring_dims[name] = other.dims.get(name, 0.0) if other.dims else 0.0
            else:
                offspring_dims[name] = primary.dims.get(name, 0.0) if primary.dims else 0.0

        new_individuals.append(
            Identity(name=str(uuid.uuid4()), dims=offspring_dims, approval=0, created_at=now)
        )
        to_remove_names.add(primary.name)

    remaining = [ind for ind in colony if ind.name not in to_remove_names]
    projected = len(remaining) + len(new_individuals)
    while projected < cfg.colony_size:
        primary = rng.choice(eligible)
        other = rng.choice(eligible)
        dims: dict[str, float] = {}
        for name in AXIS_NAMES:
            roll = rng.random()
            if roll < 0.15:
                dims[name] = round(rng.uniform(-1.0, 1.0), 4)
            elif roll < 0.40:
                dims[name] = other.dims.get(name, 0.0) if other.dims else 0.0
            else:
                dims[name] = primary.dims.get(name, 0.0) if primary.dims else 0.0
        new_individuals.append(Identity(name=str(uuid.uuid4()), dims=dims, approval=0, created_at=now))
        projected += 1

    new_colony = remaining + new_individuals
    if len(new_colony) > cfg.colony_size:
        new_colony.sort(key=lambda i: i.approval)
        new_colony = new_colony[len(new_colony) - cfg.colony_size :]

    return new_colony


# ---------------------------------------------------------------------------
# Approval updates
# ---------------------------------------------------------------------------


def update_approvals(
    colony: list[Identity],
    votes: dict[str, list[str]],
    winner_id: str,
    cfg: ThrivemindConfig,  # noqa: ARG001 kept for API consistency
) -> list[Identity]:
    """Update approval scores based on vote results."""
    id_map = {ind.name: ind for ind in colony}

    for voter_name, ranking in votes.items():
        if not ranking:
            continue
        top_pick = ranking[0]
        if top_pick == winner_id:
            if winner_id in id_map:
                id_map[winner_id].approval += 1
        else:
            if voter_name in id_map:
                id_map[voter_name].approval -= 1

    return list(id_map.values())
