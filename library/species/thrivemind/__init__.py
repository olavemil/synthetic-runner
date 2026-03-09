"""Thrivemind species — single-instance colony deliberation.

Two cycles:
  on_message: Colony deliberates on incoming events, votes, and replies.
  heartbeat:  Constitution update and spawn cycle.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

from library.species import Species, SpeciesManifest, EntryPoint
from library.tools.thrivemind import (
    ThrivemindConfig,
    build_colony_snapshot,
    build_reflection_context,
    clear_events,
    clear_inbox,
    format_consensus_status,
    format_events_for_context,
    increment_ages,
    join_fallback_messages,
    load_colony,
    load_config,
    load_constitution,
    load_events,
    load_reflection,
    pick_fallback_candidate_ids,
    record_constitution_event,
    record_reply_event,
    record_spawn_event,
    reflect_on_colony,
    run_messaging_phase,
    run_spawn_cycle,
    save_colony,
    save_colony_snapshot,
    save_constitution,
    save_reflection,
    spawn_initial_colony,
    update_approvals,
    winner_approval_ratio,
    with_consensus_status,
    # Constitution voting
    contribute_constitution_line,
    rewrite_constitution,
    save_candidate,
    save_contributions,
    select_suggesters,
    summarize_message_history,
    vote_constitution,
)
from library.tools.identity import Identity, parse_model
from library.tools.deliberate import deliberate, recompose
from library.tools.prompts import format_events, get_entity_id

if TYPE_CHECKING:
    from library.harness.adapters import Event
    from library.harness.context import InstanceContext

logger = logging.getLogger(__name__)

_SPECIES_DIR = Path(__file__).parent
_PROMPT_CANDIDATE = (_SPECIES_DIR / "prompts/candidate.md").read_text()

DEFAULT_FILES = {
    "constitution.md": "# Constitution\n",
    "sessions.md": "# Sessions\n",
}

_WRITER_CONSTITUTION_MAX = 2000
_IDENTITY_REQUIREMENT = """Output style requirement:
Speak as a unified hivemind voice.
You are the Thrivemind.
Do not force a rigid introductory phrase.
Draft a complete reply up to about 100 words when context warrants it."""


def on_message(ctx: InstanceContext, events: list[Event]) -> None:
    """Run colony deliberation round on incoming events."""
    if not events:
        logger.info("Thrivemind on_message skipped (events=0)")
        return

    logger.info("Thrivemind on_message start (events=%d)", len(events))
    cfg = load_config(ctx)

    # Select target room from events
    events_with_room = [evt for evt in events if evt.room]
    if events_with_room:
        target_event = max(events_with_room, key=lambda evt: evt.timestamp)
        target_room = target_event.room
        room_events = [evt for evt in events_with_room if evt.room == target_room]
        if len({evt.room for evt in events_with_room}) > 1:
            logger.info(
                "Thrivemind on_message received events from multiple rooms; "
                "selecting latest room=%s (room_events=%d total_events=%d)",
                target_room, len(room_events), len(events),
            )
    else:
        target_room = cfg.voice_space
        room_events = events

    scoped_events = room_events if room_events else events

    # Load or initialize colony
    colony = load_colony(ctx)
    if not colony:
        colony = spawn_initial_colony(cfg)
        save_colony(ctx, colony)
        logger.info("Thrivemind on_message initialized colony (size=%d)", len(colony))

    increment_ages(colony)

    # Build context for deliberation
    constitution = load_constitution(ctx)
    conversation = format_events(scoped_events, self_entity_id=get_entity_id(ctx))
    prompt = _PROMPT_CANDIDATE.replace("{conversation}", conversation)
    message_summary = summarize_message_history(ctx, scoped_events, cfg)
    colony_md = ctx.read("colony.md") or build_colony_snapshot(colony)
    colony_events = load_events(ctx)
    events_text = format_events_for_context(colony_events)

    shared_context = build_reflection_context(
        colony_md=colony_md,
        constitution=constitution,
        prior_reflection="",
        message_summary=message_summary,
        events_text=events_text,
    )

    # Generate per-individual reflections
    individual_contexts: dict[str, str] = {}
    for individual in colony:
        prior_reflection = load_reflection(ctx, individual)
        reflection = reflect_on_colony(
            ctx, cfg, individual, colony, constitution,
            prior_reflection, message_summary, events_text=events_text,
        )
        save_reflection(ctx, individual, reflection)
        clear_inbox(ctx, individual)
        individual_contexts[individual.name] = build_reflection_context(
            colony_md=colony_md,
            constitution=constitution,
            prior_reflection=reflection,
            message_summary=message_summary,
            events_text=events_text,
        )
    clear_events(ctx)
    logger.info("Thrivemind on_message generated per-individual reflections=%d", len(individual_contexts))

    run_messaging_phase(ctx, colony, cfg)

    # Select suggesters and deliberate
    n_suggesters = max(1, math.ceil(len(colony) * cfg.suggestion_fraction))
    suggesters = select_suggesters(colony, n_suggesters)
    logger.info("Thrivemind on_message selected suggesters=%d/%d", len(suggesters), len(colony))

    result = deliberate(
        ctx, colony, prompt,
        context=shared_context,
        context_by_identity=individual_contexts,
        subset=suggesters,
        model=cfg.suggestion_model,
        consensus_threshold=cfg.consensus_threshold,
        vote_context=shared_context,
        vote_context_by_identity=individual_contexts,
    )
    logger.info(
        "Thrivemind deliberation candidate_count=%d vote_count=%d tie=%s consensus=%s winner=%s",
        result.get("candidate_count", 0), result.get("vote_count", 0),
        result.get("is_tie", False), result.get("has_consensus", False),
        result.get("winner_member", ""),
    )

    if not result["candidates"]:
        logger.info("Thrivemind on_message no candidates; skipping send")
        return

    # Create writer identity
    provider, model = parse_model(cfg.writer_model) if cfg.writer_model else (None, "")
    writer = Identity(name="Colony", model=model, provider=provider, personality="unified colony voice")
    constitution_for_writer = (constitution or "")[:_WRITER_CONSTITUTION_MAX]
    writer_context = f"{constitution_for_writer}\n\n{_IDENTITY_REQUIREMENT}"

    # Send response (consensus or fallback)
    if result["has_consensus"] or result["candidate_count"] == 1:
        final = recompose(
            ctx, writer, result["winner_message"], result["candidates"],
            writer_context, cfg.writer_model, max_tokens=4096,
        )
        outbound = with_consensus_status(format_consensus_status(result), final)
        logger.info("Thrivemind on_message sending response to %s", target_room)
        ctx.send(target_room, outbound)
    else:
        selected_ids, covered_ratio = pick_fallback_candidate_ids(
            result.get("scores", {}), cfg.consensus_threshold,
        )
        fallback = join_fallback_messages(result["candidates"], selected_ids)
        if not fallback:
            fallback = (result.get("winner_message") or "").strip()

        if fallback:
            selected_candidates = {
                cid: result["candidates"].get(cid, "")
                for cid in selected_ids
                if result["candidates"].get(cid, "").strip()
            }
            logger.info(
                "Thrivemind on_message fallback reply selected=%d/%d coverage=%.3f threshold=%.3f",
                len(selected_ids), result["candidate_count"], covered_ratio, cfg.consensus_threshold,
            )
            final = recompose(
                ctx, writer, fallback, selected_candidates or None,
                writer_context, cfg.writer_model, max_tokens=4096,
            )
            outbound = with_consensus_status(format_consensus_status(result, covered_ratio), final)
            logger.info("Thrivemind on_message sending fallback writer response to %s", target_room)
            ctx.send(target_room, outbound)
        else:
            logger.info(
                "Thrivemind on_message no consensus and empty fallback (candidate_count=%d); skipping send",
                result["candidate_count"],
            )

    record_reply_event(
        ctx,
        winner_name=str(result.get("winner_member", "")),
        candidate_count=int(result.get("candidate_count", 0)),
        has_consensus=bool(result.get("has_consensus")),
        approval_ratio=winner_approval_ratio(result),
    )

    # Update approvals and save colony
    colony = update_approvals(colony, result["votes"], result["winner_member"], cfg)
    save_colony(ctx, colony)
    approvals = [ind.approval for ind in colony]
    logger.info(
        "Thrivemind on_message updated approvals (min=%d max=%d size=%d)",
        min(approvals) if approvals else 0, max(approvals) if approvals else 0, len(colony),
    )


def heartbeat(ctx: InstanceContext) -> None:
    """Constitution update and spawn cycle."""
    cfg = load_config(ctx)

    # Load or initialize colony
    colony = load_colony(ctx)
    if not colony:
        colony = spawn_initial_colony(cfg)
        save_colony(ctx, colony)

    increment_ages(colony)
    logger.info("Thrivemind heartbeat started (colony_size=%d)", len(colony))

    constitution = load_constitution(ctx)
    colony_events = load_events(ctx)
    events_text = format_events_for_context(colony_events)

    # Generate per-individual reflections
    individual_reflections: dict[str, str] = {}
    for individual in colony:
        prior_reflection = load_reflection(ctx, individual)
        reflection = reflect_on_colony(
            ctx, cfg, individual, colony, constitution,
            prior_reflection, "", events_text=events_text,
        )
        save_reflection(ctx, individual, reflection)
        clear_inbox(ctx, individual)
        individual_reflections[individual.name] = reflection
    clear_events(ctx)
    logger.info("Thrivemind heartbeat generated per-individual reflections=%d", len(individual_reflections))

    run_messaging_phase(ctx, colony, cfg)

    # Constitution contribution phase
    lines = []
    for individual in colony:
        line = contribute_constitution_line(
            ctx, individual, constitution, cfg,
            reflections=individual_reflections.get(individual.name, ""),
        )
        if line:
            lines.append(line)
    logger.info("Thrivemind heartbeat proposed constitution lines=%d", len(lines))
    save_contributions(ctx, lines)

    # Rewrite and vote on constitution
    proposed = rewrite_constitution(ctx, lines, constitution, cfg)
    logger.info("Thrivemind heartbeat generated candidate constitution (chars=%d)", len(proposed))
    save_candidate(ctx, proposed)

    total = len(colony)

    # Round 1 voting
    round1_votes: dict[str, bool] = {}
    for individual in colony:
        accepted = vote_constitution(
            ctx, individual, constitution, proposed, cfg,
            reflections=individual_reflections.get(individual.name, ""),
        )
        round1_votes[individual.name] = accepted

    votes_accept = sum(1 for accepted in round1_votes.values() if accepted)
    acceptance_ratio = (votes_accept / total) if total > 0 else 0.0
    logger.info(
        "Thrivemind constitution vote round=1 accept=%d total=%d ratio=%.3f threshold=%.3f",
        votes_accept, total, acceptance_ratio, cfg.consensus_threshold,
    )

    adopted = total > 0 and acceptance_ratio > cfg.consensus_threshold
    acceptance_ratio_round2 = acceptance_ratio

    # Round 2 if needed
    if not adopted and total > 0:
        round1_lines = [
            f"- {ind.name}: {'accept' if round1_votes.get(ind.name, False) else 'reject'} (approval={ind.approval})"
            for ind in colony
        ]
        round2_context = (
            f"Round 1 results: accept={votes_accept}/{total} "
            f"(ratio={acceptance_ratio:.3f}, required>{cfg.consensus_threshold:.3f}).\n"
            "Per-individual votes and current approvals:\n" + "\n".join(round1_lines)
        )
        logger.info("Thrivemind constitution not approved in round 1; starting round 2")

        votes_accept_round2 = sum(
            1 for individual in colony
            if vote_constitution(
                ctx, individual, constitution, proposed, cfg,
                reflections=individual_reflections.get(individual.name, ""),
                round_context=round2_context,
            )
        )
        acceptance_ratio_round2 = (votes_accept_round2 / total) if total > 0 else 0.0
        logger.info(
            "Thrivemind constitution vote round=2 accept=%d total=%d ratio=%.3f threshold=%.3f",
            votes_accept_round2, total, acceptance_ratio_round2, cfg.consensus_threshold,
        )
        adopted = acceptance_ratio_round2 > cfg.consensus_threshold

    if adopted:
        logger.info("Thrivemind constitution adopted; writing constitution.md")
        save_constitution(ctx, proposed)
    else:
        logger.info("Thrivemind constitution not adopted; skipping constitution.md write")

    round1_passed = total > 0 and acceptance_ratio > cfg.consensus_threshold
    record_constitution_event(
        ctx,
        accepted=adopted,
        acceptance_ratio=acceptance_ratio if round1_passed else acceptance_ratio_round2,
        rounds=1 if round1_passed else 2,
    )

    # Spawn cycle
    old_names = [ind.name for ind in colony]
    colony = run_spawn_cycle(colony, cfg)
    new_names = {ind.name for ind in colony}
    removed = [n for n in old_names if n not in new_names]
    spawned = len(new_names - set(old_names))
    if removed or spawned:
        record_spawn_event(ctx, removed=removed, spawned=spawned, colony_size=len(colony))

    save_colony(ctx, colony)
    save_colony_snapshot(ctx, colony)


class ThrivemindSpecies(Species):
    def manifest(self) -> SpeciesManifest:
        return SpeciesManifest(
            species_id="thrivemind",
            entry_points=[
                EntryPoint(name="on_message", handler=on_message, trigger="message"),
                EntryPoint(name="heartbeat", handler=heartbeat, schedule="*/15 * * * *"),
            ],
            tools=[],
            default_files=DEFAULT_FILES,
            spawn=self._spawn,
        )

    def _spawn(self, ctx: InstanceContext) -> None:
        for path, content in DEFAULT_FILES.items():
            if not ctx.exists(path):
                ctx.write(path, content)
