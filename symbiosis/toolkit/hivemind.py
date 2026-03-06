"""Reusable hivemind orchestration toolkit."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from symbiosis.toolkit.prompts import format_events

if TYPE_CHECKING:
    from symbiosis.harness.adapters import Event
    from symbiosis.harness.context import InstanceContext


HIVEMIND_NAMESPACE = "hivemind"


@dataclass
class HivemindConfig:
    role: str = "speaker_coordinator"
    persona: str = "balanced"
    voice_space: str = "main"
    quorum: int = 3
    round_timeout_s: int = 45
    max_internal_chars: int = 280
    max_output_chars: int = 1200
    speaker_instance: str = ""

    @property
    def can_coordinate(self) -> bool:
        return self.role in {"coordinator", "speaker_coordinator"}

    @property
    def can_speak(self) -> bool:
        return self.role in {"speaker", "speaker_coordinator"}

    @property
    def can_work(self) -> bool:
        return self.role in {"worker", "coordinator", "speaker_coordinator", "speaker"}


def _coerce_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _short_text(value: str, max_chars: int) -> str:
    text = (value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def load_hivemind_config(ctx: InstanceContext) -> HivemindConfig:
    raw = ctx.config("hivemind") or {}
    if not isinstance(raw, dict):
        raw = {}

    return HivemindConfig(
        role=str(raw.get("role", "speaker_coordinator")),
        persona=str(raw.get("persona", "balanced")),
        voice_space=str(raw.get("voice_space", "main")),
        quorum=max(1, _coerce_int(raw.get("quorum"), 3)),
        round_timeout_s=max(5, _coerce_int(raw.get("round_timeout_s"), 45)),
        max_internal_chars=max(80, _coerce_int(raw.get("max_internal_chars"), 280)),
        max_output_chars=max(200, _coerce_int(raw.get("max_output_chars"), 1200)),
        speaker_instance=str(raw.get("speaker_instance", "")),
    )


def _store(ctx: InstanceContext):
    return ctx.shared_store(HIVEMIND_NAMESPACE)


def register_member(ctx: InstanceContext, cfg: HivemindConfig) -> None:
    _store(ctx).put(
        f"member:{ctx.instance_id}",
        {
            "instance_id": ctx.instance_id,
            "role": cfg.role,
            "persona": cfg.persona,
            "updated_at": int(time.time() * 1000),
        },
    )


def claim_fresh_events(ctx: InstanceContext, events: list[Event]) -> list[Event]:
    shared = _store(ctx)
    fresh: list[Event] = []
    for evt in events:
        if shared.claim(f"event:{evt.event_id}", ctx.instance_id):
            fresh.append(evt)
    return fresh


def create_round_from_events(
    ctx: InstanceContext,
    events: list[Event],
    source_space: str,
) -> dict:
    now = int(time.time() * 1000)
    round_id = f"r-{now}-{uuid.uuid4().hex[:8]}"
    prompt = format_events(events)
    round_data = {
        "id": round_id,
        "status": "collecting",
        "created_at": now,
        "created_by": ctx.instance_id,
        "source_space": source_space,
        "event_ids": [evt.event_id for evt in events],
        "prompt": prompt,
        "final_message": "",
    }
    _store(ctx).put(f"round:{round_id}", round_data)
    return round_data


def list_open_rounds(ctx: InstanceContext) -> list[dict]:
    rounds = []
    for _, value in _store(ctx).scan("round:"):
        if isinstance(value, dict) and value.get("status") == "collecting":
            rounds.append(value)
    rounds.sort(key=lambda item: item.get("created_at", 0))
    return rounds


def get_round(ctx: InstanceContext, round_id: str) -> dict | None:
    data = _store(ctx).get(f"round:{round_id}")
    return data if isinstance(data, dict) else None


def get_candidates(ctx: InstanceContext, round_id: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    prefix = f"candidate:{round_id}:"
    for key, value in _store(ctx).scan(prefix):
        member_id = key.removeprefix(prefix)
        if isinstance(value, dict):
            out[member_id] = value
    return out


def get_votes(ctx: InstanceContext, round_id: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    prefix = f"vote:{round_id}:"
    for key, value in _store(ctx).scan(prefix):
        voter_id = key.removeprefix(prefix)
        if isinstance(value, dict):
            out[voter_id] = value
    return out


def submit_candidate(
    ctx: InstanceContext,
    round_id: str,
    member_id: str,
    message: str,
    *,
    persona: str,
    max_chars: int,
) -> None:
    _store(ctx).put(
        f"candidate:{round_id}:{member_id}",
        {
            "member_id": member_id,
            "persona": persona,
            "message": _short_text(message, max_chars),
            "created_at": int(time.time() * 1000),
        },
    )


def submit_vote(
    ctx: InstanceContext,
    round_id: str,
    voter_id: str,
    ranking: list[str],
    rationale: str,
) -> None:
    _store(ctx).put(
        f"vote:{round_id}:{voter_id}",
        {
            "voter_id": voter_id,
            "ranking": ranking,
            "rationale": rationale,
            "created_at": int(time.time() * 1000),
        },
    )


def _parse_vote_response(text: str, candidate_ids: list[str]) -> tuple[list[str], str]:
    ranking = []
    rationale = ""
    try:
        raw = text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(raw)
        if isinstance(data, dict):
            raw_ranking = data.get("ranking", [])
            if isinstance(raw_ranking, list):
                ranking = [str(item) for item in raw_ranking]
            rationale = str(data.get("rationale", ""))
    except Exception:
        pass

    normalized = [cid for cid in ranking if cid in candidate_ids]
    for cid in candidate_ids:
        if cid not in normalized:
            normalized.append(cid)

    return normalized, rationale


def generate_candidate_message(
    ctx: InstanceContext,
    round_data: dict,
    cfg: HivemindConfig,
) -> str:
    system = (
        "You are one shard in a hivemind. Produce a concise candidate reply "
        "to the external conversation. Keep it short and concrete."
    )
    user_msg = f"""Persona: {cfg.persona}
Round prompt:
{round_data.get("prompt", "")}

Return only the candidate reply text. Max {cfg.max_internal_chars} chars."""
    response = ctx.llm(
        messages=[{"role": "user", "content": user_msg}],
        system=system,
        max_tokens=512,
        caller="hivemind_candidate",
    )
    return _short_text(response.message, cfg.max_internal_chars)


def generate_vote(
    ctx: InstanceContext,
    round_data: dict,
    cfg: HivemindConfig,
    candidates: dict[str, dict],
) -> tuple[list[str], str]:
    candidate_ids = sorted(candidates.keys())
    if len(candidate_ids) <= 1:
        return candidate_ids, "Single candidate."

    options = []
    for cid in candidate_ids:
        msg = str(candidates[cid].get("message", ""))
        options.append(f"- {cid}: {msg}")

    system = (
        "You are a hivemind shard voting on candidate replies. "
        "Rank candidates by relevance, clarity, and usefulness."
    )
    user_msg = f"""Persona: {cfg.persona}
Round prompt:
{round_data.get("prompt", "")}

Candidates:
{chr(10).join(options)}

Return JSON:
{{
  "ranking": ["best_id", "next_id"],
  "rationale": "brief reason"
}}"""
    response = ctx.llm(
        messages=[{"role": "user", "content": user_msg}],
        system=system,
        max_tokens=512,
        caller="hivemind_vote",
    )
    return _parse_vote_response(response.message, candidate_ids)


def tally_borda(candidates: dict[str, dict], votes: dict[str, dict]) -> dict:
    scores = {cid: 0 for cid in candidates}

    if votes:
        for vote in votes.values():
            ranking = vote.get("ranking", [])
            if not isinstance(ranking, list):
                continue
            n = len(ranking)
            for idx, cid in enumerate(ranking):
                if cid in scores:
                    scores[cid] += n - idx
    else:
        for cid in scores:
            scores[cid] = 1

    winner_member = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[0][0]
    winner = candidates[winner_member]

    return {
        "winner_member": winner_member,
        "winner_message": str(winner.get("message", "")),
        "scores": scores,
        "candidate_count": len(candidates),
        "vote_count": len(votes),
    }


def compose_consensus_output(
    ctx: InstanceContext,
    round_data: dict,
    cfg: HivemindConfig,
    tally: dict,
    candidates: dict[str, dict],
) -> str:
    winner_id = tally["winner_member"]
    winner_text = tally["winner_message"]
    snippets = []
    for cid, data in sorted(candidates.items()):
        snippets.append(f"- {cid}: {data.get('message', '')}")

    system = (
        "You are the unified external voice of a hivemind. "
        "Write one coherent reply for the conversation."
    )
    user_msg = f"""Round prompt:
{round_data.get("prompt", "")}

Winning candidate ({winner_id}):
{winner_text}

All candidates:
{chr(10).join(snippets)}

Write one coherent external reply. Max {cfg.max_output_chars} chars."""
    response = ctx.llm(
        messages=[{"role": "user", "content": user_msg}],
        system=system,
        max_tokens=1024,
        caller="hivemind_consensus",
    )
    return _short_text(response.message, cfg.max_output_chars)


def round_ready(
    round_data: dict,
    *,
    candidate_count: int,
    vote_count: int,
    quorum: int,
    timeout_s: int,
) -> bool:
    if candidate_count >= quorum and vote_count >= quorum:
        return True

    age_ms = int(time.time() * 1000) - int(round_data.get("created_at", 0))
    if age_ms >= timeout_s * 1000 and candidate_count >= 1:
        return True
    return False


def mark_round_finalized(
    ctx: InstanceContext,
    round_data: dict,
    *,
    final_message: str,
    winner_member: str,
) -> None:
    updated = dict(round_data)
    updated["status"] = "finalized"
    updated["winner_member"] = winner_member
    updated["final_message"] = final_message
    updated["finalized_at"] = int(time.time() * 1000)
    _store(ctx).put(f"round:{round_data['id']}", updated)
