"""Multi-identity LLM deliberation patterns."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from symbiosis.toolkit.identity import Identity, format_persona, parse_model
from symbiosis.toolkit.voting import borda_tally

if TYPE_CHECKING:
    from symbiosis.harness.context import InstanceContext


def generate_with_identity(
    ctx: InstanceContext,
    identity: Identity,
    prompt: str,
    context: str = "",
    model: str = "",
) -> str:
    """Generate text via the identity's model/provider, with persona in system prompt.

    model="" → use identity.model/provider. A non-empty model string may be
    "provider/model" format, which overrides both.
    """
    persona = format_persona(identity)
    system_parts = []
    if identity.name:
        system_parts.append(f"You are {identity.name}.")
    if persona and persona != identity.name:
        system_parts.append(f"Personality: {persona}")
    if context:
        system_parts.append(f"\nContext:\n{context}")
    system = " ".join(system_parts) if system_parts else None

    # Resolve model/provider
    if model:
        use_provider, use_model = parse_model(model)
    else:
        use_model = identity.model
        use_provider = identity.provider

    kwargs: dict = {
        "max_tokens": 512,
        "caller": f"generate_{identity.name}",
    }
    if system:
        kwargs["system"] = system
    if use_model:
        kwargs["model"] = use_model
    if use_provider:
        kwargs["provider"] = use_provider

    response = ctx.llm([{"role": "user", "content": prompt}], **kwargs)
    return response.message.strip()


def multi_generate(
    ctx: InstanceContext,
    identities: list[Identity],
    prompt: str,
    context: str = "",
    model: str = "",
) -> dict[str, str]:
    """Generate text for each identity. Returns {identity.name → text}.

    model="" → each identity uses its own model field.
    """
    return {
        identity.name: generate_with_identity(ctx, identity, prompt, context, model)
        for identity in identities
    }


def _generate_ranking(
    ctx: InstanceContext,
    voter: Identity,
    candidates: dict[str, str],
    prompt: str,
    model: str = "",
) -> list[str]:
    """Ask a voter to rank candidates. Returns ordered list of candidate names."""
    candidate_ids = list(candidates.keys())
    if len(candidates) == 1:
        return candidate_ids

    persona = format_persona(voter)
    options = "\n".join(f"- {cid}: {text}" for cid, text in candidates.items())
    system = (
        f"You are {voter.name}. Personality: {persona}\n"
        "Rank the candidate replies by how well they address the conversation."
    )
    user_msg = (
        f"Conversation:\n{prompt}\n\n"
        f"Candidates:\n{options}\n\n"
        f'Return JSON: {{"ranking": ["id1", "id2", ...]}}'
    )

    # Resolve model/provider
    if model:
        use_provider, use_model = parse_model(model)
    else:
        use_model = voter.model
        use_provider = voter.provider

    kwargs: dict = {
        "max_tokens": 256,
        "caller": f"vote_{voter.name}",
        "system": system,
    }
    if use_model:
        kwargs["model"] = use_model
    if use_provider:
        kwargs["provider"] = use_provider

    response = ctx.llm([{"role": "user", "content": user_msg}], **kwargs)

    try:
        raw = response.message.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(raw)
        ranking = [str(r) for r in data.get("ranking", [])]
    except Exception:
        ranking = []

    # Normalize: ensure all candidate ids present
    normalized = [cid for cid in ranking if cid in candidate_ids]
    for cid in candidate_ids:
        if cid not in normalized:
            normalized.append(cid)
    return normalized


def multi_vote(
    ctx: InstanceContext,
    identities: list[Identity],
    candidates: dict[str, str],
    prompt: str,
    model: str = "",
    exclude_own: bool = False,
    top_n: int = 0,
) -> dict[str, list[str]]:
    """Have each identity rank candidates. Returns {voter.name → ranked candidate names}.

    exclude_own=True: voter's own name is excluded from candidates they rank.
    top_n>0: truncate each ranking to top_n items.
    """
    votes: dict[str, list[str]] = {}
    for voter in identities:
        if exclude_own:
            voter_candidates = {k: v for k, v in candidates.items() if k != voter.name}
        else:
            voter_candidates = candidates

        if not voter_candidates:
            votes[voter.name] = []
            continue

        if len(voter_candidates) == 1:
            ranking = list(voter_candidates.keys())
        else:
            ranking = _generate_ranking(ctx, voter, voter_candidates, prompt, model)

        if top_n > 0:
            ranking = ranking[:top_n]
        votes[voter.name] = ranking

    return votes


def deliberate(
    ctx: InstanceContext,
    identities: list[Identity],
    prompt: str,
    context: str = "",
    subset: list[Identity] | None = None,
    model: str = "",
    exclude_own: bool = False,
    top_n: int = 0,
    consensus_threshold: float = 0.6,
) -> dict:
    """Full deliberation: generate candidates, vote, tally.

    subset: identities who generate candidates (None = all identities).
    Returns: winner_member, winner_message, scores, candidate_count, vote_count,
             is_tie, has_consensus, candidates dict, votes dict.
    """
    generators = subset if subset is not None else identities
    candidates = multi_generate(ctx, generators, prompt, context, model)

    if not candidates:
        return {
            "winner_member": "",
            "winner_message": "",
            "scores": {},
            "candidate_count": 0,
            "vote_count": 0,
            "is_tie": False,
            "has_consensus": False,
            "candidates": {},
            "votes": {},
        }

    votes = multi_vote(ctx, identities, candidates, prompt, model, exclude_own, top_n)
    tally = borda_tally(candidates, votes)

    winner_id = tally["winner_member"]
    total_score = sum(tally["scores"].values())
    winner_score = tally["scores"].get(winner_id, 0)
    has_consensus = total_score == 0 or (winner_score / total_score) > consensus_threshold

    return {
        "winner_member": winner_id,
        "winner_message": tally["winner_message"],
        "scores": tally["scores"],
        "candidate_count": tally["candidate_count"],
        "vote_count": tally["vote_count"],
        "is_tie": tally["is_tie"],
        "has_consensus": has_consensus,
        "candidates": candidates,
        "votes": votes,
    }


def recompose(
    ctx: InstanceContext,
    identity: Identity,
    winning_text: str,
    all_candidates: dict[str, str] | None = None,
    context: str = "",
    model: str = "",
) -> str:
    """Rewrite winning text through identity's voice.

    all_candidates provided → unified composition citing all candidates (Thrivemind style).
    all_candidates absent  → direct rewrite through identity's voice (Hecate style).
    """
    if all_candidates:
        snippets = "\n".join(f"- {text}" for text in all_candidates.values())
        prompt = (
            f"Winning candidate:\n{winning_text}\n\n"
            f"All candidates:\n{snippets}\n\n"
            "Write the final, unified reply."
        )
    else:
        prompt = f"Rewrite this response in your own voice and style:\n\n{winning_text}"

    return generate_with_identity(ctx, identity, prompt, context, model)


def think_with_context(
    ctx: InstanceContext,
    identity: Identity,
    context: str = "",
    others_thinking: dict[str, str] | None = None,
    voice_memory: dict[str, str] | None = None,
) -> str:
    """Reflective thinking, optionally informed by other identities' prior thoughts.

    context: shared context string (memory, constitution, etc.)
    others_thinking: {name: thought} from other identities
    voice_memory: per-identity memory dict with keys like 'thinking', 'subconscious', 'motivation'
    """
    parts = []
    if context:
        parts.append(context)

    if voice_memory:
        if voice_memory.get("subconscious"):
            parts.append(f"## Subconscious\n{voice_memory['subconscious']}")
        if voice_memory.get("motivation"):
            parts.append(f"## Motivation\n{voice_memory['motivation']}")
        if voice_memory.get("thinking"):
            parts.append(f"## Previous Thoughts\n{voice_memory['thinking']}")

    full_context = "\n\n".join(parts)

    others_block = ""
    if others_thinking:
        lines = [f"**{name}**: {thought}" for name, thought in others_thinking.items()]
        others_block = "\n\n## Other voices' thoughts\n" + "\n\n".join(lines)

    prompt = f"Reflect and think freely.{others_block}\n\nWrite your current thoughts."
    return generate_with_identity(ctx, identity, prompt, full_context, "")
