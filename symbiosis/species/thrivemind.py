"""Thrivemind species — distributed hivemind coordination on shared toolkit primitives."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from symbiosis.species import Species, SpeciesManifest, EntryPoint
from symbiosis.toolkit.hivemind import (
    HivemindConfig,
    load_hivemind_config,
    register_member,
    claim_fresh_events,
    create_round_from_events,
    list_open_rounds,
    get_candidates,
    get_votes,
    submit_candidate,
    submit_vote,
    generate_candidate_message,
    generate_vote,
    tally_borda,
    compose_consensus_output,
    round_ready,
    mark_round_finalized,
)

if TYPE_CHECKING:
    from symbiosis.harness.adapters import Event
    from symbiosis.harness.context import InstanceContext

logger = logging.getLogger(__name__)


DEFAULT_FILES = {
    "hivemind.md": "# Hivemind\n",
    "voice.md": "# Voice\n",
    "sessions.md": "# Sessions\n",
}


def _contribute_round(ctx: InstanceContext, cfg: HivemindConfig, round_data: dict) -> None:
    round_id = str(round_data["id"])
    candidates = get_candidates(ctx, round_id)

    if ctx.instance_id not in candidates:
        candidate_text = generate_candidate_message(ctx, round_data, cfg)
        if candidate_text:
            submit_candidate(
                ctx,
                round_id,
                ctx.instance_id,
                candidate_text,
                persona=cfg.persona,
                max_chars=cfg.max_internal_chars,
            )
            candidates = get_candidates(ctx, round_id)

    if not candidates:
        return

    votes = get_votes(ctx, round_id)
    if ctx.instance_id not in votes:
        ranking, rationale = generate_vote(ctx, round_data, cfg, candidates)
        submit_vote(ctx, round_id, ctx.instance_id, ranking, rationale)


def _emit_message(ctx: InstanceContext, cfg: HivemindConfig, message: str) -> None:
    if not message:
        return

    if cfg.can_speak:
        ctx.send(cfg.voice_space, message)
        return

    if cfg.speaker_instance:
        payload = json.dumps(
            {
                "kind": "hivemind_output",
                "space": cfg.voice_space,
                "message": message,
            }
        )
        ctx.send_to(cfg.speaker_instance, payload)


def _finalize_ready_rounds(ctx: InstanceContext, cfg: HivemindConfig) -> None:
    if not cfg.can_coordinate:
        return

    for round_data in list_open_rounds(ctx):
        round_id = str(round_data["id"])
        candidates = get_candidates(ctx, round_id)
        votes = get_votes(ctx, round_id)

        if not round_ready(
            round_data,
            candidate_count=len(candidates),
            vote_count=len(votes),
            quorum=cfg.quorum,
            timeout_s=cfg.round_timeout_s,
        ):
            continue
        if not candidates:
            continue

        tally = tally_borda(candidates, votes)
        final_message = compose_consensus_output(ctx, round_data, cfg, tally, candidates)
        mark_round_finalized(
            ctx,
            round_data,
            final_message=final_message,
            winner_member=str(tally["winner_member"]),
        )
        _emit_message(ctx, cfg, final_message)


def _drain_speaker_inbox(ctx: InstanceContext, cfg: HivemindConfig) -> None:
    if not cfg.can_speak:
        return

    for msg in ctx.read_inbox():
        body = str(msg.get("body", ""))
        if not body:
            continue
        try:
            payload = json.loads(body)
            if isinstance(payload, dict):
                space = str(payload.get("space", cfg.voice_space))
                message = str(payload.get("message", ""))
                if message:
                    ctx.send(space, message)
                    continue
        except json.JSONDecodeError:
            pass
        ctx.send(cfg.voice_space, body)


def on_message(ctx: InstanceContext, events: list[Event]) -> None:
    """Create hivemind round from external events, then contribute/finalize if role allows."""
    if not events:
        return

    cfg = load_hivemind_config(ctx)
    register_member(ctx, cfg)
    if not cfg.can_coordinate:
        return

    fresh_events = claim_fresh_events(ctx, events)
    if not fresh_events:
        return

    source_space = next((evt.room for evt in fresh_events if evt.room), cfg.voice_space)
    round_data = create_round_from_events(ctx, fresh_events, source_space=source_space)

    if cfg.can_work:
        _contribute_round(ctx, cfg, round_data)
    _finalize_ready_rounds(ctx, cfg)


def heartbeat(ctx: InstanceContext) -> None:
    """Periodic worker contribution and coordinator finalization cycle."""
    cfg = load_hivemind_config(ctx)
    register_member(ctx, cfg)

    for round_data in list_open_rounds(ctx):
        if cfg.can_work:
            _contribute_round(ctx, cfg, round_data)

    _finalize_ready_rounds(ctx, cfg)
    _drain_speaker_inbox(ctx, cfg)


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
