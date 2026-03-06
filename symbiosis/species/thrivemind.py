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
    generate_suggestion,
    generate_vote,
    tally_borda,
    write_message,
    contribute_constitution_line,
    rewrite_constitution,
    vote_constitution,
    run_spawn_cycle,
    update_approvals,
)
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

    # Select suggesters (approval-weighted)
    n_suggesters = max(1, math.ceil(cfg.colony_size * cfg.suggestion_fraction))
    suggesters = select_suggesters(colony, n_suggesters)

    # Generate candidate suggestions
    candidates: dict[str, str] = {}
    for individual in suggesters:
        text = generate_suggestion(ctx, individual, prompt, cfg)
        if text:
            candidates[individual.id] = text

    if not candidates:
        return

    # All colony members vote
    votes: dict[str, list[str]] = {}
    for individual in colony:
        ranking = generate_vote(ctx, individual, candidates, prompt, cfg)
        votes[individual.id] = ranking

    tally = tally_borda(candidates, votes)
    winner_id = tally["winner_member"]
    winner_text = tally["winner_message"]

    # Check consensus
    total_score = sum(tally["scores"].values())
    winner_score = tally["scores"].get(winner_id, 0)
    has_consensus = total_score == 0 or (winner_score / total_score) > cfg.consensus_threshold

    if has_consensus or len(candidates) == 1:
        final = write_message(ctx, prompt, winner_text, candidates, constitution, cfg)
        ctx.send(cfg.voice_space, final)

    # Update approvals and persist
    colony = update_approvals(colony, votes, winner_id, cfg)
    save_colony(ctx, colony)


def heartbeat(ctx: InstanceContext) -> None:
    """Constitution update and spawn cycle."""
    cfg = load_config(ctx)
    colony = load_colony(ctx)
    if not colony:
        colony = spawn_initial_colony(cfg)
        save_colony(ctx, colony)

    constitution = load_constitution(ctx)

    # Each individual contributes a constitution line
    lines = []
    for individual in colony:
        line = contribute_constitution_line(ctx, individual, constitution, cfg)
        if line:
            lines.append(line)

    # Synthesize proposed constitution
    proposed = rewrite_constitution(ctx, lines, constitution, cfg)

    # Each individual votes on the proposed constitution
    votes_accept = sum(
        1 for individual in colony if vote_constitution(ctx, individual, constitution, proposed, cfg)
    )
    total = len(colony)
    if total > 0 and (votes_accept / total) > cfg.consensus_threshold:
        save_constitution(ctx, proposed)

    # Run spawn cycle
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
