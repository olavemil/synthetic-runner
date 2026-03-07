"""Thrivemind species — single-instance colony deliberation."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from symbiosis.species import Species, SpeciesManifest, EntryPoint
from symbiosis.toolkit.thrivemind import (
    build_colony_snapshot,
    ThrivemindConfig,
    build_reflection_context,
    load_config,
    load_colony,
    load_reflection,
    save_colony,
    save_colony_snapshot,
    save_reflection,
    spawn_initial_colony,
    load_constitution,
    save_constitution,
    save_contributions,
    save_candidate,
    select_suggesters,
    summarize_message_history,
    reflect_on_colony,
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


def _pick_fallback_candidate_ids(scores: dict[str, int], threshold: float) -> tuple[list[str], float]:
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


def _join_fallback_messages(candidates: dict[str, str], candidate_ids: list[str]) -> str:
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


def _winner_approval_ratio(result: dict) -> float:
    scores = result.get("scores", {})
    if not isinstance(scores, dict) or not scores:
        return 0.0

    total_score = sum(max(0, int(score)) for score in scores.values())
    if total_score <= 0:
        return 1.0 if int(result.get("candidate_count", 0) or 0) == 1 else 0.0

    winner_id = str(result.get("winner_member", ""))
    winner_score = max(0, int(scores.get(winner_id, 0)))
    return winner_score / total_score


def _format_consensus_status(result: dict, fallback_coverage: float | None = None) -> str:
    winner_pct = max(0, min(100, int(round(_winner_approval_ratio(result) * 100))))
    has_consensus = bool(result.get("has_consensus")) or int(result.get("candidate_count", 0) or 0) == 1

    if has_consensus:
        return f"Consensus: {winner_pct}% approval."

    if fallback_coverage is not None and fallback_coverage > 0:
        fallback_pct = max(0, min(100, int(round(fallback_coverage * 100))))
        return (
            "Consensus: No consensus "
            f"(top candidate {winner_pct}% approval; fallback coalition {fallback_pct}% approval)."
        )

    return f"Consensus: No consensus (top candidate {winner_pct}% approval)."


def _with_consensus_status(status: str, message: str) -> str:
    body = (message or "").strip()
    return f"{status}\n\n{body}".strip() if body else status


def on_message(ctx: InstanceContext, events: list[Event]) -> None:
    """Run colony deliberation round on incoming events."""
    if not events:
        logger.info("Thrivemind on_message skipped (events=0)")
        return

    logger.info("Thrivemind on_message start (events=%d)", len(events))
    cfg = load_config(ctx)
    events_with_room = [evt for evt in events if evt.room]
    if events_with_room:
        target_event = max(events_with_room, key=lambda evt: evt.timestamp)
        target_room = target_event.room
        room_events = [evt for evt in events_with_room if evt.room == target_room]
        if len({evt.room for evt in events_with_room}) > 1:
            logger.info(
                "Thrivemind on_message received events from multiple rooms; selecting latest room=%s (room_events=%d total_events=%d)",
                target_room,
                len(room_events),
                len(events),
            )
    else:
        target_room = cfg.voice_space
        room_events = events

    scoped_events = room_events if room_events else events
    colony = load_colony(ctx)
    if not colony:
        colony = spawn_initial_colony(cfg)
        save_colony(ctx, colony)
        logger.info("Thrivemind on_message initialized colony (size=%d)", len(colony))

    constitution = load_constitution(ctx)
    prompt = format_events(scoped_events)
    message_summary = summarize_message_history(ctx, scoped_events, cfg)
    colony_md = ctx.read("colony.md") or build_colony_snapshot(colony)
    shared_context = build_reflection_context(
        colony_md=colony_md,
        constitution=constitution,
        prior_reflection="",
        message_summary=message_summary,
    )
    individual_contexts: dict[str, str] = {}
    for individual in colony:
        prior_reflection = load_reflection(ctx, individual)
        reflection = reflect_on_colony(
            ctx,
            cfg,
            individual,
            colony,
            constitution,
            prior_reflection,
            message_summary,
        )
        save_reflection(ctx, individual, reflection)
        individual_contexts[individual.name] = build_reflection_context(
            colony_md=colony_md,
            constitution=constitution,
            prior_reflection=reflection,
            message_summary=message_summary,
        )
    logger.info(
        "Thrivemind on_message generated per-individual reflections=%d",
        len(individual_contexts),
    )

    n_suggesters = max(1, math.ceil(cfg.colony_size * cfg.suggestion_fraction))
    suggesters = select_suggesters(colony, n_suggesters)
    logger.info(
        "Thrivemind on_message selected suggesters=%d/%d",
        len(suggesters),
        len(colony),
    )

    result = deliberate(
        ctx,
        colony,
        prompt,
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
        result.get("candidate_count", 0),
        result.get("vote_count", 0),
        result.get("is_tie", False),
        result.get("has_consensus", False),
        result.get("winner_member", ""),
    )

    if not result["candidates"]:
        logger.info("Thrivemind on_message no candidates; skipping send")
        return

    provider, model = parse_model(cfg.writer_model) if cfg.writer_model else (None, "")
    writer = Identity(
        name="Colony",
        model=model,
        provider=provider,
        personality="unified colony voice",
    )

    identity_requirement = (
        "Output style requirement:\n"
        "Speak as a unified hivemind voice.\n"
        "You are Thrivemind and also answer to the name subconscious-entity.\n"
        "Do not force a rigid introductory phrase.\n"
        "Draft a complete reply up to about 100 words when context warrants it.\n"
        "Do not include a separate consensus-status header."
    )

    _WRITER_CONSTITUTION_MAX = 2000
    constitution_for_writer = (constitution or "")[:_WRITER_CONSTITUTION_MAX]

    if result["has_consensus"] or result["candidate_count"] == 1:
        writer_context = (
            f"{constitution_for_writer}\n\n"
            f"{identity_requirement}"
        )
        final = recompose(
            ctx,
            writer,
            result["winner_message"],
            result["candidates"],
            writer_context,
            cfg.writer_model,
            max_tokens=320,
        )
        outbound = _with_consensus_status(_format_consensus_status(result), final)
        logger.info("Thrivemind on_message sending response to %s", target_room)
        ctx.send(target_room, outbound)
    else:
        selected_ids, covered_ratio = _pick_fallback_candidate_ids(
            result.get("scores", {}),
            cfg.consensus_threshold,
        )
        fallback = _join_fallback_messages(result["candidates"], selected_ids)
        if not fallback:
            fallback = (result.get("winner_message") or "").strip()
        if fallback:
            selected_candidates = {
                candidate_id: result["candidates"].get(candidate_id, "")
                for candidate_id in selected_ids
                if result["candidates"].get(candidate_id, "").strip()
            }
            logger.info(
                "Thrivemind on_message fallback reply selected=%d/%d coverage=%.3f threshold=%.3f",
                len(selected_ids),
                result["candidate_count"],
                covered_ratio,
                cfg.consensus_threshold,
            )
            final = recompose(
                ctx,
                writer,
                fallback,
                selected_candidates or None,
                (
                    f"{constitution_for_writer}\n\n"
                    f"{identity_requirement}"
                ),
                cfg.writer_model,
                max_tokens=320,
            )
            outbound = _with_consensus_status(
                _format_consensus_status(result, covered_ratio),
                final,
            )
            logger.info("Thrivemind on_message sending fallback writer response to %s", target_room)
            ctx.send(target_room, outbound)
        else:
            logger.info(
                "Thrivemind on_message no consensus and empty fallback (candidate_count=%d); skipping send",
                result["candidate_count"],
            )

    colony = update_approvals(colony, result["votes"], result["winner_member"], cfg)
    save_colony(ctx, colony)
    approvals = [ind.approval for ind in colony]
    logger.info(
        "Thrivemind on_message updated approvals (min=%d max=%d size=%d)",
        min(approvals) if approvals else 0,
        max(approvals) if approvals else 0,
        len(colony),
    )


def heartbeat(ctx: InstanceContext) -> None:
    """Constitution update and spawn cycle."""
    cfg = load_config(ctx)
    colony = load_colony(ctx)
    if not colony:
        colony = spawn_initial_colony(cfg)
        save_colony(ctx, colony)
    logger.info("Thrivemind heartbeat started (colony_size=%d)", len(colony))

    constitution = load_constitution(ctx)
    individual_reflections: dict[str, str] = {}
    for individual in colony:
        prior_reflection = load_reflection(ctx, individual)
        reflection = reflect_on_colony(
            ctx,
            cfg,
            individual,
            colony,
            constitution,
            prior_reflection,
            "",
        )
        save_reflection(ctx, individual, reflection)
        individual_reflections[individual.name] = reflection
    logger.info(
        "Thrivemind heartbeat generated per-individual reflections=%d",
        len(individual_reflections),
    )

    lines = []
    for individual in colony:
        line = contribute_constitution_line(
            ctx,
            individual,
            constitution,
            cfg,
            reflections=individual_reflections.get(individual.name, ""),
        )
        if line:
            lines.append(line)
    logger.info("Thrivemind heartbeat proposed constitution lines=%d", len(lines))
    save_contributions(ctx, lines)

    proposed = rewrite_constitution(ctx, lines, constitution, cfg)
    logger.info("Thrivemind heartbeat generated candidate constitution (chars=%d)", len(proposed))
    save_candidate(ctx, proposed)

    total = len(colony)

    round1_votes: dict[str, bool] = {}
    for individual in colony:
        accepted = vote_constitution(
            ctx,
            individual,
            constitution,
            proposed,
            cfg,
            reflections=individual_reflections.get(individual.name, ""),
        )
        round1_votes[individual.name] = accepted

    votes_accept = sum(1 for accepted in round1_votes.values() if accepted)
    acceptance_ratio = (votes_accept / total) if total > 0 else 0.0
    logger.info(
        "Thrivemind constitution vote round=1 accept=%d total=%d ratio=%.3f threshold=%.3f",
        votes_accept,
        total,
        acceptance_ratio,
        cfg.consensus_threshold,
    )

    adopted = total > 0 and acceptance_ratio > cfg.consensus_threshold
    if not adopted and total > 0:
        round1_lines = []
        for individual in colony:
            vote = "accept" if round1_votes.get(individual.name, False) else "reject"
            round1_lines.append(
                f"- {individual.name}: {vote} (approval={individual.approval})"
            )
        round2_context = (
            f"Round 1 results: accept={votes_accept}/{total} "
            f"(ratio={acceptance_ratio:.3f}, required>{cfg.consensus_threshold:.3f}).\n"
            "Per-individual votes and current approvals:\n"
            + "\n".join(round1_lines)
        )
        logger.info("Thrivemind constitution not approved in round 1; starting round 2")

        votes_accept_round2 = sum(
            1
            for individual in colony
            if vote_constitution(
                ctx,
                individual,
                constitution,
                proposed,
                cfg,
                reflections=individual_reflections.get(individual.name, ""),
                round_context=round2_context,
            )
        )
        acceptance_ratio_round2 = (votes_accept_round2 / total) if total > 0 else 0.0
        logger.info(
            "Thrivemind constitution vote round=2 accept=%d total=%d ratio=%.3f threshold=%.3f",
            votes_accept_round2,
            total,
            acceptance_ratio_round2,
            cfg.consensus_threshold,
        )
        adopted = acceptance_ratio_round2 > cfg.consensus_threshold

    if adopted:
        logger.info("Thrivemind constitution adopted; writing constitution.md")
        save_constitution(ctx, proposed)
    else:
        logger.info("Thrivemind constitution not adopted; skipping constitution.md write")

    colony = run_spawn_cycle(colony, cfg)
    save_colony(ctx, colony)
    save_colony_snapshot(ctx, colony)


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
