"""Single-instance colony toolkit for Thrivemind — colony biology only."""

from __future__ import annotations

import json
import random
import time
import uuid
from typing import TYPE_CHECKING

from symbiosis.toolkit.identity import (
    AXES,
    AXIS_NAMES,
    Identity,
    format_persona,
    parse_model,
)
from symbiosis.toolkit.voting import approval_weights, weighted_sample
from symbiosis.toolkit.deliberate import generate_with_identity

if TYPE_CHECKING:
    from symbiosis.harness.context import InstanceContext

# Backward-compat alias: colony individuals ARE Identity objects
Individual = Identity

COLONY_NAMESPACE = "colony"


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
) -> str:
    persona = format_persona(individual)
    prompt = (
        f"Personality: {persona}\n\n"
        f"Current constitution:\n{current_constitution}\n\n"
        "Propose one new principle or refinement (one sentence)."
    )
    return generate_with_identity(
        ctx,
        individual,
        prompt,
        context="You are contributing one principle or value to your colony's shared constitution.",
        model=cfg.suggestion_model,
    )


def rewrite_constitution(
    ctx: InstanceContext,
    lines: list[str],
    current_constitution: str,
    cfg: ThrivemindConfig,
) -> str:
    combined = "\n".join(f"- {line}" for line in lines if line.strip())
    # Use a writer identity with the writer model
    provider, model = parse_model(cfg.writer_model) if cfg.writer_model else (None, "")
    writer = Identity(
        name="ColonyWriter",
        model=model,
        provider=provider,
        personality="You are synthesizing colony principles into a coherent constitution.",
    )
    prompt = (
        f"Current constitution:\n{current_constitution}\n\n"
        f"Proposed principles:\n{combined}\n\n"
        "Rewrite the constitution incorporating the best principles. Keep it concise."
    )
    return generate_with_identity(ctx, writer, prompt, model=cfg.writer_model)


def vote_constitution(
    ctx: InstanceContext,
    individual: Identity,
    current: str,
    proposed: str,
    cfg: ThrivemindConfig,
) -> bool:
    persona = format_persona(individual)
    prompt = (
        f"Personality: {persona}\n\n"
        f"Current:\n{current}\n\n"
        f"Proposed:\n{proposed}\n\n"
        'Reply with JSON: {"accept": true} or {"accept": false}'
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
    ctx.write("constitution.md", text)


def load_constitution(ctx: InstanceContext) -> str:
    return ctx.read("constitution.md") or "# Constitution\n"


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
