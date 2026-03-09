"""Single-instance colony toolkit for Thrivemind — colony biology only."""

from __future__ import annotations

import json
import logging
import random
import re
import time
import uuid
from pathlib import Path
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
from library.tools.prompts import format_events, get_entity_id

if TYPE_CHECKING:
    from library.harness.context import InstanceContext

# Backward-compat alias: colony individuals ARE Identity objects
Individual = Identity

COLONY_NAMESPACE = "colony"
logger = logging.getLogger(__name__)
_CODE_FENCE_RE = re.compile(r"(?is)```(?:[a-zA-Z0-9_-]+)?\n(.*?)```")
_WORD_RE = re.compile(r"\b\w+\b")

# Load prompts (thrivemind is in species/ but we're in tools/, so find species path)
_SPECIES_DIR = Path(__file__).parent.parent / "species" / "thrivemind"
_PROMPTS_DIR = _SPECIES_DIR / "prompts"
_PROMPT_REFLECT = (_PROMPTS_DIR / "reflect.md").read_text()
_PROMPT_SUMMARIZE = (_PROMPTS_DIR / "summarize.md").read_text()
_PROMPT_MESSAGING = (_PROMPTS_DIR / "messaging.md").read_text()
_PROMPT_CONTRIBUTE = (_PROMPTS_DIR / "contribute_constitution.md").read_text()
_PROMPT_CONTRIBUTE_RETRY = (_PROMPTS_DIR / "contribute_constitution_retry.md").read_text()
_PROMPT_REWRITE = (_PROMPTS_DIR / "rewrite_constitution.md").read_text()
_PROMPT_VOTE = (_PROMPTS_DIR / "vote_constitution.md").read_text()


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
        colony_size: int = 12,  # deprecated, used as fallback
        min_colony_size: int = 8,
        max_colony_size: int = 16,
        suggestion_fraction: float = 0.5,
        approval_threshold: int = 3,
        consensus_threshold: float = 0.6,
        round_timeout_s: int = 30,
        suggestion_model: str = "",
        writer_model: str = "",
        voice_space: str = "main",
    ):
        # Support legacy colony_size as fallback for min/max
        self.min_colony_size = min_colony_size
        self.max_colony_size = max_colony_size
        # If only colony_size was set (legacy), use it for both bounds
        self._legacy_colony_size = colony_size
        self.suggestion_fraction = suggestion_fraction
        self.approval_threshold = approval_threshold
        self.consensus_threshold = consensus_threshold
        self.round_timeout_s = round_timeout_s
        self.suggestion_model = suggestion_model
        self.writer_model = writer_model
        self.voice_space = voice_space

    @property
    def colony_size(self) -> int:
        """Backward-compat: return max_colony_size as the target."""
        return self.max_colony_size


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
    legacy_size = max(1, _coerce_int(raw.get("colony_size"), 0))
    min_size = max(1, _coerce_int(raw.get("min_colony_size"), legacy_size or 8))
    max_size = max(min_size, _coerce_int(raw.get("max_colony_size"), legacy_size or 16))
    return ThrivemindConfig(
        colony_size=legacy_size,
        min_colony_size=min_size,
        max_colony_size=max_size,
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
        age=int(d.get("age", 0)),
    )


def _identity_to_dict(ind: Identity) -> dict:
    return {
        "name": ind.name,
        "dims": ind.dims or {},
        "approval": ind.approval,
        "created_at": ind.created_at,
        "age": ind.age,
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


def build_colony_snapshot(colony: list[Identity], events: list[dict] | None = None) -> str:
    """Render a human-readable colony snapshot markdown document."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    lines = [
        "# Colony",
        "",
        f"Generated: {timestamp}",
        f"Individuals: {len(colony)}",
        "",
        "| Individual | Personality | Approval | Age |",
        "| --- | --- | ---: | ---: |",
    ]

    if not colony:
        lines.append("| - | - | 0 | 0 |")
    else:
        for ind in sorted(colony, key=lambda i: (-i.approval, i.created_at, i.name)):
            personality = _md_escape(format_persona(ind))
            lines.append(f"| `{ind.name}` | {personality} | {ind.approval} | {ind.age} |")

    events_section = format_events_for_context(events or [])
    if events_section:
        lines.append("")
        lines.append(events_section)

    return "\n".join(lines) + "\n"


def save_colony_snapshot(ctx: InstanceContext, colony: list[Identity]) -> None:
    """Write colony.md snapshot to instance memory storage."""
    events = load_events(ctx)
    content = build_colony_snapshot(colony, events=events)
    logger.info(
        "Writing thrivemind colony.md (individuals=%d, events=%d, chars=%d)",
        len(colony),
        len(events),
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
    prompt = _PROMPT_SUMMARIZE.replace(
        "{events}", format_events(events, self_entity_id=get_entity_id(ctx))
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
    events_text: str = "",
    inbox_text: str = "",
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
    if events_text:
        parts.append(events_text)
    if inbox_text:
        parts.append(f"## Messages Received\n{inbox_text}")
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


def increment_ages(colony: list[Identity]) -> None:
    """Increment age counter for all colony members (call once per thinking session)."""
    for ind in colony:
        ind.age += 1


def _relative_age_label(individual: Identity, colony: list[Identity]) -> str:
    """Return a relative age label for context injection."""
    if not colony:
        return ""
    avg_age = sum(ind.age for ind in colony) / len(colony)
    if avg_age == 0:
        return "newborn"
    ratio = individual.age / avg_age
    if ratio < 0.3:
        return "very young (much younger than average)"
    if ratio < 0.7:
        return "young (younger than average)"
    if ratio < 1.3:
        return "average age"
    if ratio < 2.0:
        return "experienced (older than average)"
    return "elder (much older than average)"


def reflect_on_colony(
    ctx: InstanceContext,
    cfg: ThrivemindConfig,
    individual: Identity,
    colony: list[Identity],
    constitution: str,
    prior_reflection: str,
    message_summary: str,
    events_text: str = "",
) -> str:
    colony_md = ctx.read("colony.md") or build_colony_snapshot(colony)
    inbox_text = read_inbox(ctx, individual)
    context = build_reflection_context(
        colony_md=colony_md,
        constitution=constitution,
        prior_reflection=prior_reflection,
        message_summary=message_summary,
        events_text=events_text,
        inbox_text=inbox_text,
    )
    age_label = _relative_age_label(individual, colony)
    age_block = f"\nYou are {age_label} in the colony (age={individual.age}, colony average={sum(i.age for i in colony)/len(colony):.1f}).\n" if colony else ""
    prompt = _PROMPT_REFLECT.replace("{age_block}", age_block)
    reflection = generate_with_identity(
        ctx,
        individual,
        prompt,
        context=context,
        model=cfg.suggestion_model,
    ).strip()
    logger.info(
        "Generated thrivemind reflection (individual=%s, chars=%d)",
        individual.name,
        len(reflection),
    )
    return reflection


# ---------------------------------------------------------------------------
# Colony events
# ---------------------------------------------------------------------------

_EVENTS_FILE = "colony_events.json"


def load_events(ctx: InstanceContext) -> list[dict]:
    """Load pending colony events."""
    raw = ctx.read(_EVENTS_FILE)
    if not raw:
        return []
    try:
        events = json.loads(raw)
        return events if isinstance(events, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _append_event(ctx: InstanceContext, event: dict) -> None:
    """Append a single event to the event log."""
    events = load_events(ctx)
    events.append(event)
    ctx.write(_EVENTS_FILE, json.dumps(events, ensure_ascii=False))


def clear_events(ctx: InstanceContext) -> None:
    """Clear the event log (call after events have been consumed)."""
    ctx.write(_EVENTS_FILE, "[]")


def record_reply_event(
    ctx: InstanceContext,
    winner_name: str,
    candidate_count: int,
    has_consensus: bool,
    approval_ratio: float,
) -> None:
    _append_event(ctx, {
        "type": "reply",
        "winner": winner_name,
        "candidates": candidate_count,
        "consensus": has_consensus,
        "approval_ratio": round(approval_ratio, 3),
        "time": int(time.time()),
    })


def record_constitution_event(
    ctx: InstanceContext,
    accepted: bool,
    acceptance_ratio: float,
    rounds: int,
) -> None:
    _append_event(ctx, {
        "type": "constitution_vote",
        "accepted": accepted,
        "ratio": round(acceptance_ratio, 3),
        "rounds": rounds,
        "time": int(time.time()),
    })


def record_spawn_event(
    ctx: InstanceContext,
    removed: list[str],
    spawned: int,
    colony_size: int,
) -> None:
    _append_event(ctx, {
        "type": "spawn",
        "removed": removed,
        "spawned": spawned,
        "colony_size": colony_size,
        "time": int(time.time()),
    })


def format_events_for_context(events: list[dict]) -> str:
    """Format colony events as human-readable text for injection into prompts."""
    if not events:
        return ""
    lines = ["## Recent Colony Events"]
    for evt in events:
        etype = evt.get("type", "unknown")
        if etype == "reply":
            consensus = "with consensus" if evt.get("consensus") else "without consensus"
            lines.append(
                f"- Reply sent ({consensus}, winner: {evt.get('winner', '?')}, "
                f"{evt.get('candidates', '?')} candidates, "
                f"{evt.get('approval_ratio', 0)*100:.0f}% approval)"
            )
        elif etype == "constitution_vote":
            result = "accepted" if evt.get("accepted") else "rejected"
            lines.append(
                f"- Constitution {result} "
                f"({evt.get('ratio', 0)*100:.0f}% approval, "
                f"{evt.get('rounds', 1)} round(s))"
            )
        elif etype == "spawn":
            removed = evt.get("removed", [])
            if removed:
                lines.append(
                    f"- Spawn cycle: {len(removed)} removed ({', '.join(removed)}), "
                    f"{evt.get('spawned', 0)} new, colony now {evt.get('colony_size', '?')}"
                )
            else:
                lines.append(
                    f"- Spawn cycle: {evt.get('spawned', 0)} new members added, "
                    f"colony now {evt.get('colony_size', '?')}"
                )
        elif etype == "message":
            lines.append(
                f"- Message from {evt.get('from', '?')} to {evt.get('to', '?')}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual inboxes
# ---------------------------------------------------------------------------


def _inbox_path(individual: Identity | str) -> str:
    name = individual.name if isinstance(individual, Identity) else str(individual)
    return f"inbox/{quote(name, safe='-_.,')}.md"


def read_inbox(ctx: InstanceContext, individual: Identity) -> str:
    """Read and return inbox contents for an individual."""
    return ctx.read(_inbox_path(individual)) or ""


def clear_inbox(ctx: InstanceContext, individual: Identity) -> None:
    """Clear an individual's inbox after messages have been consumed."""
    ctx.write(_inbox_path(individual), "")


def deliver_message(
    ctx: InstanceContext,
    sender: Identity,
    recipient_name: str,
    message: str,
) -> None:
    """Deliver a short message to another individual's inbox."""
    path = _inbox_path(recipient_name)
    existing = ctx.read(path) or ""
    entry = f"**{sender.name}**: {message}\n"
    ctx.write(path, existing + entry)
    logger.info(
        "Delivered colony message from %s to %s (%d chars)",
        sender.name, recipient_name, len(message),
    )


def run_messaging_phase(
    ctx: InstanceContext,
    colony: list[Identity],
    cfg: ThrivemindConfig,
) -> int:
    """Give each individual a chance to send one message to another.

    Returns the number of messages sent.
    """
    colony_names = [ind.name for ind in colony]
    sent = 0

    for individual in colony:
        others = [n for n in colony_names if n != individual.name]
        if not others:
            continue

        persona = format_persona(individual)
        inbox = read_inbox(ctx, individual)
        inbox_block = f"\nMessages in your inbox:\n{inbox}\n" if inbox.strip() else ""
        prompt = (
            _PROMPT_MESSAGING
            .replace("{individual_name}", individual.name)
            .replace("{colony_size}", str(len(colony)))
            .replace("{personality}", persona)
            .replace("{inbox_block}", inbox_block)
            .replace("{others}", ", ".join(others))
        )
        raw = generate_with_identity(
            ctx, individual, prompt,
            context="Colony internal messaging phase. Keep messages brief and meaningful.",
            model=cfg.suggestion_model,
            max_tokens=256,
        )

        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(cleaned)
            target = data.get("to")
            msg = str(data.get("message", "")).strip()
            if target and msg and target in colony_names and target != individual.name:
                deliver_message(ctx, individual, target, msg[:200])
                sent += 1
        except Exception:
            pass

    logger.info("Colony messaging phase: %d messages sent", sent)
    return sent


_NAME_ADJECTIVES = [
    "amber", "ashen", "bright", "calm", "cold", "crimson", "dark", "deep",
    "dim", "dusk", "fading", "fierce", "frozen", "gentle", "gilded", "grey",
    "hollow", "iron", "keen", "last", "lone", "lucid", "muted", "pale",
    "quiet", "restless", "risen", "sharp", "silent", "silver", "slow",
    "soft", "stark", "still", "swift", "thin", "true", "vast", "warm", "wild",
]
_NAME_NOUNS = [
    "ash", "bell", "bone", "brook", "cairn", "clay", "coal", "crow",
    "dawn", "dust", "elm", "ember", "fern", "flame", "fox", "frost",
    "gate", "glass", "grove", "hawk", "haze", "hedge", "lark", "leaf",
    "marsh", "moss", "moth", "oak", "pearl", "pine", "reed", "root",
    "sage", "salt", "shade", "shard", "smoke", "spark", "spine", "stone",
    "storm", "thorn", "tide", "vale", "vine", "wren",
]


def _generate_name(rng: random.Random, existing_names: set[str]) -> str:
    """Generate a unique adjective-adjective-noun name."""
    for _ in range(50):
        a1 = rng.choice(_NAME_ADJECTIVES)
        a2 = rng.choice(_NAME_ADJECTIVES)
        while a2 == a1:
            a2 = rng.choice(_NAME_ADJECTIVES)
        noun = rng.choice(_NAME_NOUNS)
        name = f"{a1}-{a2}-{noun}"
        if name not in existing_names:
            return name
    # Fallback: append UUID fragment for uniqueness
    return f"{rng.choice(_NAME_ADJECTIVES)}-{rng.choice(_NAME_ADJECTIVES)}-{uuid.uuid4().hex[:6]}"


def spawn_initial_colony(
    cfg: ThrivemindConfig, rng: random.Random | None = None
) -> list[Identity]:
    rng = rng or random.Random()
    colony = []
    now = int(time.time())
    existing_names: set[str] = set()
    target = cfg.min_colony_size
    for _ in range(target):
        dims = {name: round(rng.uniform(-1.0, 1.0), 4) for name in AXIS_NAMES}
        ind_name = _generate_name(rng, existing_names)
        existing_names.add(ind_name)
        colony.append(Identity(name=ind_name, dims=dims, approval=0, created_at=now))
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
        _PROMPT_CONTRIBUTE
        .replace("{individual_name}", individual.name)
        .replace("{personality}", persona)
        .replace("{constitution}", constitution_excerpt)
        .replace("{reflections_block}", reflections_block)
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
            _PROMPT_CONTRIBUTE_RETRY
            .replace("{individual_name}", individual.name)
            .replace("{personality}", persona)
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
        _PROMPT_REWRITE
        .replace("{current_constitution}", constitution_excerpt)
        .replace("{proposed_principles}", combined_excerpt)
        .replace("{target_words}", str(target_words))
        .replace("{min_words}", str(min_words))
        .replace("{max_words}", str(max_words))
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
    round_context_block = ""
    if round_context.strip():
        round_context_excerpt = _truncate_for_prompt(round_context.strip(), max_chars=1400)
        round_context_block = f"\n\nAdditional context:\n{round_context_excerpt}"
    
    prompt = (
        _PROMPT_VOTE
        .replace("{individual_name}", individual.name)
        .replace("{personality}", persona)
        .replace("{current_constitution}", current_excerpt)
        .replace("{proposed_constitution}", proposed_excerpt)
        .replace("{reflections_block}", reflections_block)
        .replace("{round_context_block}", round_context_block)
    )
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


def _make_offspring(
    primary: Identity,
    other: Identity,
    rng: random.Random,
    existing_names: set[str],
    now: int,
) -> Identity:
    """Create a single offspring from primary × other with dim inheritance."""
    offspring_dims: dict[str, float] = {}
    for name in AXIS_NAMES:
        roll = rng.random()
        if roll < 0.15:
            offspring_dims[name] = round(rng.uniform(-1.0, 1.0), 4)
        elif roll < 0.40:
            offspring_dims[name] = other.dims.get(name, 0.0) if other.dims else 0.0
        else:
            offspring_dims[name] = primary.dims.get(name, 0.0) if primary.dims else 0.0
    child_name = _generate_name(rng, existing_names)
    existing_names.add(child_name)
    return Identity(name=child_name, dims=offspring_dims, approval=0, created_at=now)


def run_spawn_cycle(
    colony: list[Identity],
    cfg: ThrivemindConfig,
    rng: random.Random | None = None,
) -> list[Identity]:
    """Eligible individuals spawn offspring; colony stays within size bounds.

    - If colony is below max, eligible members spawn offspring (parents stay).
    - If colony is at max, eligible members are replaced by offspring.
    - Colony is trimmed to max_colony_size if over.
    """
    rng = rng or random.Random()
    if not colony:
        return colony

    eligible = [ind for ind in colony if ind.approval >= cfg.approval_threshold]
    if not eligible:
        return colony

    total_approval = sum(ind.approval for ind in eligible)
    if total_approval == 0:
        return colony

    existing_names = {ind.name for ind in colony}
    new_individuals: list[Identity] = []
    now = int(time.time())
    room = cfg.max_colony_size - len(colony)

    if room > 0:
        # Under max: parents survive, just add offspring
        for primary in eligible:
            if len(new_individuals) >= room:
                break
            weights = [ind.approval / total_approval for ind in eligible]
            (other,) = rng.choices(eligible, weights=weights, k=1)
            new_individuals.append(_make_offspring(primary, other, rng, existing_names, now))

        new_colony = colony + new_individuals
    else:
        # At or over max: replace eligible parents with offspring
        to_remove_names: set[str] = set()
        for primary in eligible:
            weights = [ind.approval / total_approval for ind in eligible]
            (other,) = rng.choices(eligible, weights=weights, k=1)
            new_individuals.append(_make_offspring(primary, other, rng, existing_names, now))
            to_remove_names.add(primary.name)

        remaining = [ind for ind in colony if ind.name not in to_remove_names]
        new_colony = remaining + new_individuals

    # Trim to max
    if len(new_colony) > cfg.max_colony_size:
        new_colony.sort(key=lambda i: i.approval)
        new_colony = new_colony[len(new_colony) - cfg.max_colony_size:]

    # Pad to min
    while len(new_colony) < cfg.min_colony_size:
        primary = rng.choice(eligible)
        other = rng.choice(eligible)
        new_colony.append(_make_offspring(primary, other, rng, existing_names, now))

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
    """Update approval scores based on vote results.

    Voters who ranked the winner in their top 2 picks get +1 approval for
    the winner. Voters whose top pick was not the winner lose -1 on
    themselves. This dual-pick system makes positive approval more
    achievable since voters have two chances to back the winner.
    """
    id_map = {ind.name: ind for ind in colony}

    for voter_name, ranking in votes.items():
        if not ranking:
            continue
        top_two = ranking[:2]
        if winner_id in top_two:
            if winner_id in id_map:
                id_map[winner_id].approval += 1
        else:
            if voter_name in id_map:
                id_map[voter_name].approval -= 1

    return list(id_map.values())


# ---------------------------------------------------------------------------
# Deliberation / consensus helpers
# ---------------------------------------------------------------------------


def pick_fallback_candidate_ids(
    scores: dict[str, int], threshold: float
) -> tuple[list[str], float]:
    """Pick top candidates until cumulative score crosses threshold ratio."""
    if not scores:
        return [], 0.0

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    total = sum(max(0, score) for _, score in ranked)
    if total <= 0:
        return [ranked[0][0]], 1.0

    chosen: list[str] = []
    running = 0
    coverage = 0.0
    for candidate_id, score in ranked:
        chosen.append(candidate_id)
        running += max(0, score)
        coverage = running / total
        if coverage > threshold:
            break
    return chosen, coverage


def join_fallback_messages(candidates: dict[str, str], candidate_ids: list[str]) -> str:
    """Join selected candidate messages in ranked order, dropping empties/duplicates."""
    parts: list[str] = []
    seen: set[str] = set()
    for candidate_id in candidate_ids:
        text = (candidates.get(candidate_id) or "").strip()
        if not text:
            continue
        normalized = " ".join(text.split())
        if normalized in seen:
            continue
        seen.add(normalized)
        parts.append(text)
    return "\n\n".join(parts).strip()


def winner_approval_ratio(result: dict) -> float:
    """Calculate the approval ratio of the winning candidate."""
    scores = result.get("scores", {})
    if not isinstance(scores, dict) or not scores:
        return 0.0

    total_score = sum(max(0, int(score)) for score in scores.values())
    if total_score <= 0:
        return 1.0 if int(result.get("candidate_count", 0) or 0) == 1 else 0.0

    winner_id = str(result.get("winner_member", ""))
    winner_score = max(0, int(scores.get(winner_id, 0)))
    return winner_score / total_score


def format_consensus_status(
    result: dict, fallback_coverage: float | None = None
) -> str:
    """Format a human-readable consensus status line."""
    winner_pct = max(0, min(100, int(round(winner_approval_ratio(result) * 100))))
    has_consensus = (
        bool(result.get("has_consensus"))
        or int(result.get("candidate_count", 0) or 0) == 1
    )

    if has_consensus:
        return f"Consensus: {winner_pct}% approval."

    if fallback_coverage is not None and fallback_coverage > 0:
        fallback_pct = max(0, min(100, int(round(fallback_coverage * 100))))
        return (
            "Consensus: No consensus "
            f"(top candidate {winner_pct}% approval; fallback coalition {fallback_pct}% approval)."
        )

    return f"Consensus: No consensus (top candidate {winner_pct}% approval)."


def with_consensus_status(status: str, message: str) -> str:
    """Prepend consensus status to a message body."""
    body = (message or "").strip()
    return f"{status}\n\n{body}".strip() if body else status
