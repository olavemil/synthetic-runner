"""Single-instance colony toolkit for Thrivemind."""

from __future__ import annotations

import json
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from symbiosis.harness.context import InstanceContext

COLONY_NAMESPACE = "colony"

# Six personality axes — (positive_pole, negative_pole)
AXES: list[tuple[str, str]] = [
    ("conservative", "liberal"),
    ("simple", "complex"),
    ("optimistic", "pessimistic"),
    ("extrovert", "introvert"),
    ("cautious", "bold"),
    ("analytical", "emotional"),
]

# Axis names used as keys
AXIS_NAMES = [f"{a}_{b}" for a, b in AXES]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ThrivemindConfig:
    colony_size: int = 12
    suggestion_fraction: float = 0.5
    approval_threshold: int = 3
    consensus_threshold: float = 0.6
    round_timeout_s: int = 30
    suggestion_model: str = ""
    writer_model: str = ""
    voice_space: str = "main"


@dataclass
class Individual:
    id: str
    dims: dict[str, float]
    approval: int = 0
    created_at: int = field(default_factory=lambda: int(time.time()))


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


def _split_model(model_str: str) -> tuple[str | None, str]:
    """Split 'provider/model' → (provider, model). Returns (None, model) if no '/'."""
    if "/" in model_str:
        provider, _, model = model_str.partition("/")
        return provider or None, model
    return None, model_str


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


def _individual_from_dict(d: dict) -> Individual:
    return Individual(
        id=str(d["id"]),
        dims={k: float(v) for k, v in d.get("dims", {}).items()},
        approval=int(d.get("approval", 0)),
        created_at=int(d.get("created_at", 0)),
    )


def _individual_to_dict(ind: Individual) -> dict:
    return {
        "id": ind.id,
        "dims": ind.dims,
        "approval": ind.approval,
        "created_at": ind.created_at,
    }


def load_colony(ctx: InstanceContext) -> list[Individual]:
    store = ctx.store(COLONY_NAMESPACE)
    result = []
    for _key, value in store.scan():
        if isinstance(value, dict) and "id" in value:
            result.append(_individual_from_dict(value))
    result.sort(key=lambda i: i.created_at)
    return result


def save_colony(ctx: InstanceContext, colony: list[Individual]) -> None:
    store = ctx.store(COLONY_NAMESPACE)
    # Delete all existing entries
    for key, _ in store.scan():
        store.delete(key)
    for ind in colony:
        store.put(ind.id, _individual_to_dict(ind))


def spawn_initial_colony(cfg: ThrivemindConfig, rng: random.Random | None = None) -> list[Individual]:
    rng = rng or random.Random()
    colony = []
    now = int(time.time())
    for _ in range(cfg.colony_size):
        dims = {name: round(rng.uniform(-1.0, 1.0), 4) for name in AXIS_NAMES}
        colony.append(Individual(id=str(uuid.uuid4()), dims=dims, approval=0, created_at=now))
    return colony


# ---------------------------------------------------------------------------
# Persona formatting
# ---------------------------------------------------------------------------


def format_persona(individual: Individual) -> str:
    """Map individual dimensions to a descriptive persona string."""

    def magnitude_label(v: float) -> str:
        a = abs(v)
        if a >= 0.8:
            return "extremely"
        if a >= 0.6:
            return "very"
        if a >= 0.4:
            return "fairly"
        if a >= 0.2:
            return "somewhat"
        return ""

    # Build (magnitude, label, trait_name) for non-zero axes
    entries = []
    for (pos_pole, neg_pole), name in zip(AXES, AXIS_NAMES):
        v = individual.dims.get(name, 0.0)
        pole = pos_pole if v >= 0 else neg_pole
        label = magnitude_label(v)
        entries.append((abs(v), label, pole))

    # Sort descending by magnitude
    entries.sort(key=lambda x: -x[0])

    # Filter out entries below 0.2 threshold
    significant = [(mag, label, pole) for mag, label, pole in entries if mag >= 0.2]

    if not significant:
        # Fallback: include highest-magnitude as "barely <axis>"
        mag, _label, pole = entries[0]
        return f"barely {pole}"

    parts = [f"{label} {pole}" for _mag, label, pole in significant]
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Colony selection
# ---------------------------------------------------------------------------


def select_suggesters(
    colony: list[Individual],
    n: int,
    rng: random.Random | None = None,
) -> list[Individual]:
    """Select n individuals weighted by approval (approval+1, min 0)."""
    rng = rng or random.Random()
    if not colony:
        return []
    n = min(n, len(colony))

    weights = [max(0, ind.approval + 1) for ind in colony]
    total = sum(weights)

    if total == 0:
        # Fall back to uniform random
        return rng.sample(colony, n)

    # Weighted sample without replacement
    selected = []
    remaining = list(zip(weights, colony))
    for _ in range(n):
        if not remaining:
            break
        r_weights = [w for w, _ in remaining]
        r_total = sum(r_weights)
        pick = rng.uniform(0, r_total)
        cumulative = 0.0
        chosen_idx = 0
        for idx, (w, _) in enumerate(remaining):
            cumulative += w
            if pick <= cumulative:
                chosen_idx = idx
                break
        _, chosen = remaining.pop(chosen_idx)
        selected.append(chosen)

    return selected


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def _llm_with_model(ctx: InstanceContext, model_str: str, messages: list[dict], **kwargs):
    """Call ctx.llm, optionally routing to a specific provider/model."""
    if model_str:
        provider, model = _split_model(model_str)
        kwargs["model"] = model
        if provider:
            kwargs["provider"] = provider
    return ctx.llm(messages, **kwargs)


def generate_suggestion(
    ctx: InstanceContext,
    individual: Individual,
    prompt: str,
    cfg: ThrivemindConfig,
) -> str:
    persona = format_persona(individual)
    system = (
        "You are one voice in a colony deliberating on how to respond. "
        "Speak from your personality. Be concise and direct."
    )
    user_msg = f"Personality: {persona}\n\nConversation:\n{prompt}\n\nWrite a candidate reply."
    response = _llm_with_model(
        ctx,
        cfg.suggestion_model,
        [{"role": "user", "content": user_msg}],
        system=system,
        max_tokens=512,
        caller="thrivemind_suggestion",
    )
    return response.message.strip()


def generate_vote(
    ctx: InstanceContext,
    individual: Individual,
    candidates: dict[str, str],
    prompt: str,
    cfg: ThrivemindConfig,
) -> list[str]:
    """Return ranked list of candidate individual IDs (best first)."""
    if not candidates:
        return []
    if len(candidates) == 1:
        return list(candidates.keys())

    persona = format_persona(individual)
    options = "\n".join(f"- {cid}: {text}" for cid, text in candidates.items())
    system = "You are voting on candidate replies. Rank them by how well they address the conversation."
    user_msg = (
        f"Personality: {persona}\n\nConversation:\n{prompt}\n\n"
        f"Candidates:\n{options}\n\n"
        f"Return JSON: {{\"ranking\": [\"id1\", \"id2\", ...]}}"
    )
    response = _llm_with_model(
        ctx,
        cfg.suggestion_model,
        [{"role": "user", "content": user_msg}],
        system=system,
        max_tokens=256,
        caller="thrivemind_vote",
    )
    # Parse JSON ranking
    candidate_ids = list(candidates.keys())
    try:
        raw = response.message.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(raw)
        ranking = [str(r) for r in data.get("ranking", [])]
    except Exception:
        ranking = []

    # Normalize: ensure all ids present
    normalized = [cid for cid in ranking if cid in candidate_ids]
    for cid in candidate_ids:
        if cid not in normalized:
            normalized.append(cid)
    return normalized


# ---------------------------------------------------------------------------
# Borda tally
# ---------------------------------------------------------------------------


def tally_borda(
    candidates: dict[str, str],
    votes: dict[str, list[str]],
) -> dict:
    """Borda count over candidate id→text, votes as voter_id→ranked id list."""
    scores: dict[str, int] = {cid: 0 for cid in candidates}
    n = len(candidates)

    if votes:
        for ranking in votes.values():
            for idx, cid in enumerate(ranking):
                if cid in scores:
                    scores[cid] += n - idx
    else:
        for cid in scores:
            scores[cid] = 1

    winner_id = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[0][0]

    return {
        "winner_member": winner_id,
        "winner_message": candidates[winner_id],
        "scores": scores,
        "candidate_count": len(candidates),
        "vote_count": len(votes),
    }


# ---------------------------------------------------------------------------
# Output composition
# ---------------------------------------------------------------------------


def write_message(
    ctx: InstanceContext,
    prompt: str,
    winner_text: str,
    all_candidates: dict[str, str],
    constitution: str,
    cfg: ThrivemindConfig,
) -> str:
    snippets = "\n".join(f"- {text}" for text in all_candidates.values())
    system = (
        "You are the unified voice of a colony. "
        "Write one coherent, considered reply that reflects the colony's deliberation.\n\n"
        f"Colony constitution:\n{constitution}"
        if constitution.strip()
        else "You are the unified voice of a colony. Write one coherent, considered reply."
    )
    user_msg = (
        f"Conversation:\n{prompt}\n\n"
        f"Winning candidate:\n{winner_text}\n\n"
        f"All candidates:\n{snippets}\n\n"
        "Write the final reply."
    )
    response = _llm_with_model(
        ctx,
        cfg.writer_model,
        [{"role": "user", "content": user_msg}],
        system=system,
        max_tokens=1024,
        caller="thrivemind_write",
    )
    return response.message.strip()


# ---------------------------------------------------------------------------
# Constitution management
# ---------------------------------------------------------------------------


def contribute_constitution_line(
    ctx: InstanceContext,
    individual: Individual,
    current_constitution: str,
    cfg: ThrivemindConfig,
) -> str:
    persona = format_persona(individual)
    system = "You are contributing one principle or value to your colony's shared constitution."
    user_msg = (
        f"Personality: {persona}\n\n"
        f"Current constitution:\n{current_constitution}\n\n"
        "Propose one new principle or refinement (one sentence)."
    )
    response = _llm_with_model(
        ctx,
        cfg.suggestion_model,
        [{"role": "user", "content": user_msg}],
        system=system,
        max_tokens=128,
        caller="thrivemind_constitution_line",
    )
    return response.message.strip()


def rewrite_constitution(
    ctx: InstanceContext,
    lines: list[str],
    current_constitution: str,
    cfg: ThrivemindConfig,
) -> str:
    combined = "\n".join(f"- {line}" for line in lines if line.strip())
    system = "You are synthesizing colony principles into a coherent constitution."
    user_msg = (
        f"Current constitution:\n{current_constitution}\n\n"
        f"Proposed principles:\n{combined}\n\n"
        "Rewrite the constitution incorporating the best principles. Keep it concise."
    )
    response = _llm_with_model(
        ctx,
        cfg.writer_model,
        [{"role": "user", "content": user_msg}],
        system=system,
        max_tokens=512,
        caller="thrivemind_constitution_rewrite",
    )
    return response.message.strip()


def vote_constitution(
    ctx: InstanceContext,
    individual: Individual,
    current: str,
    proposed: str,
    cfg: ThrivemindConfig,
) -> bool:
    persona = format_persona(individual)
    system = "You are voting on whether to adopt a new constitution for your colony."
    user_msg = (
        f"Personality: {persona}\n\n"
        f"Current:\n{current}\n\n"
        f"Proposed:\n{proposed}\n\n"
        "Reply with JSON: {\"accept\": true} or {\"accept\": false}"
    )
    response = _llm_with_model(
        ctx,
        cfg.suggestion_model,
        [{"role": "user", "content": user_msg}],
        system=system,
        max_tokens=64,
        caller="thrivemind_constitution_vote",
    )
    try:
        raw = response.message.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(raw)
        return bool(data.get("accept", False))
    except Exception:
        return False


def save_constitution(ctx: InstanceContext, text: str) -> None:
    ctx.write("constitution.md", text)


def load_constitution(ctx: InstanceContext) -> str:
    return ctx.read("constitution.md") or "# Constitution\n"


# ---------------------------------------------------------------------------
# Spawn cycle
# ---------------------------------------------------------------------------


def run_spawn_cycle(
    colony: list[Individual],
    cfg: ThrivemindConfig,
    rng: random.Random | None = None,
) -> list[Individual]:
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

    to_remove_ids: set[str] = set()
    new_individuals: list[Individual] = []
    now = int(time.time())

    for primary in eligible:
        # Select other parent weighted by approval
        weights = [ind.approval / total_approval for ind in eligible]
        (other,) = rng.choices(eligible, weights=weights, k=1)

        # Build offspring dims
        offspring_dims: dict[str, float] = {}
        for name in AXIS_NAMES:
            roll = rng.random()
            if roll < 0.15:
                # 15% randomized entirely
                offspring_dims[name] = round(rng.uniform(-1.0, 1.0), 4)
            elif roll < 0.40:
                # 25% from other parent
                offspring_dims[name] = other.dims.get(name, 0.0)
            else:
                # Rest from primary
                offspring_dims[name] = primary.dims.get(name, 0.0)

        new_individuals.append(
            Individual(id=str(uuid.uuid4()), dims=offspring_dims, approval=0, created_at=now)
        )
        to_remove_ids.add(primary.id)

    # Ensure colony stays at target size
    remaining = [ind for ind in colony if ind.id not in to_remove_ids]
    projected = len(remaining) + len(new_individuals)
    while projected < cfg.colony_size:
        # Spawn extra offspring from random eligible pair
        primary = rng.choice(eligible)
        other = rng.choice(eligible)
        dims: dict[str, float] = {}
        for name in AXIS_NAMES:
            roll = rng.random()
            if roll < 0.15:
                dims[name] = round(rng.uniform(-1.0, 1.0), 4)
            elif roll < 0.40:
                dims[name] = other.dims.get(name, 0.0)
            else:
                dims[name] = primary.dims.get(name, 0.0)
        new_individuals.append(Individual(id=str(uuid.uuid4()), dims=dims, approval=0, created_at=now))
        projected += 1

    new_colony = remaining + new_individuals
    # If over colony_size, remove lowest-approval members
    if len(new_colony) > cfg.colony_size:
        new_colony.sort(key=lambda i: i.approval)
        new_colony = new_colony[len(new_colony) - cfg.colony_size :]

    return new_colony


# ---------------------------------------------------------------------------
# Approval updates
# ---------------------------------------------------------------------------


def update_approvals(
    colony: list[Individual],
    votes: dict[str, list[str]],
    winner_id: str,
    cfg: ThrivemindConfig,  # noqa: ARG001 kept for API consistency
) -> list[Individual]:
    """Update approval scores based on vote results."""
    id_map = {ind.id: ind for ind in colony}

    for voter_id, ranking in votes.items():
        if not ranking:
            continue
        top_pick = ranking[0]
        if top_pick == winner_id:
            # Winner gets +1 per voter who ranked them first
            if winner_id in id_map:
                id_map[winner_id].approval += 1
        else:
            # Non-winner voter whose top vote ≠ winner gets -1
            if voter_id in id_map:
                id_map[voter_id].approval -= 1

    return list(id_map.values())
