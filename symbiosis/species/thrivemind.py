"""Thrivemind species — single-instance colony deliberation."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from symbiosis.species import Species, SpeciesManifest, EntryPoint
from symbiosis.toolkit.hivemind import (
    ThrivemindConfig,
    load_config,
    load_colony,
    save_colony,
    spawn_initial_colony,
    load_constitution,
    save_constitution,
    select_suggesters,
    contribute_constitution_line,
    rewrite_constitution,
    vote_constitution,
    run_spawn_cycle,
    update_approvals,
)
from symbiosis.toolkit.identity import Identity, parse_model
from symbiosis.toolkit.deliberate import deliberate, recompose
from symbiosis.toolkit.prompts import format_events

if TYPE_CHECKING:
    from symbiosis.harness.adapters import Event
    from symbiosis.harness.context import InstanceContext

logger = logging.getLogger(__name__)

DEFAULT_FILES = {
    "constitution.md": "# Constitution\n",
    "sessions.md": "# Sessions\n",
}


def on_message(ctx: InstanceContext, events: list[Event]) -> None:
    """Run colony deliberation round on incoming events."""
    if not events:
        return

    cfg = load_config(ctx)
    colony = load_colony(ctx)
    if not colony:
        colony = spawn_initial_colony(cfg)
        save_colony(ctx, colony)

    constitution = load_constitution(ctx)
    prompt = format_events(events)

    n_suggesters = max(1, math.ceil(cfg.colony_size * cfg.suggestion_fraction))
    suggesters = select_suggesters(colony, n_suggesters)

    result = deliberate(
        ctx,
        colony,
        prompt,
        context=constitution,
        subset=suggesters,
        model=cfg.suggestion_model,
        consensus_threshold=cfg.consensus_threshold,
    )

    if not result["candidates"]:
        return

    if result["has_consensus"] or result["candidate_count"] == 1:
        provider, model = parse_model(cfg.writer_model) if cfg.writer_model else (None, "")
        writer = Identity(
            name="Colony",
            model=model,
            provider=provider,
            personality="unified colony voice",
        )
        final = recompose(
            ctx,
            writer,
            result["winner_message"],
            result["candidates"],
            constitution,
            cfg.writer_model,
        )
        ctx.send(cfg.voice_space, final)

    colony = update_approvals(colony, result["votes"], result["winner_member"], cfg)
    save_colony(ctx, colony)


def heartbeat(ctx: InstanceContext) -> None:
    """Constitution update and spawn cycle."""
    cfg = load_config(ctx)
    colony = load_colony(ctx)
    if not colony:
        colony = spawn_initial_colony(cfg)
        save_colony(ctx, colony)

    constitution = load_constitution(ctx)

    lines = []
    for individual in colony:
        line = contribute_constitution_line(ctx, individual, constitution, cfg)
        if line:
            lines.append(line)

    proposed = rewrite_constitution(ctx, lines, constitution, cfg)

    votes_accept = sum(
        1 for individual in colony if vote_constitution(ctx, individual, constitution, proposed, cfg)
    )
    total = len(colony)
    if total > 0 and (votes_accept / total) > cfg.consensus_threshold:
        save_constitution(ctx, proposed)

    colony = run_spawn_cycle(colony, cfg)
    save_colony(ctx, colony)


class ThrivemindSpecies(Species):
    def manifest(self) -> SpeciesManifest:
        return SpeciesManifest(
            species_id="thrivemind",
            entry_points=[
                EntryPoint(
                    name="on_message",
                    handler=on_message,
                    trigger="message",
                ),
                EntryPoint(
                    name="heartbeat",
                    handler=heartbeat,
                    schedule="*/15 * * * *",
                ),
            ],
            tools=[],
            default_files=DEFAULT_FILES,
            spawn=self._spawn,
        )

    def _spawn(self, ctx: InstanceContext) -> None:
        for path, content in DEFAULT_FILES.items():
            if not ctx.exists(path):
                ctx.write(path, content)
