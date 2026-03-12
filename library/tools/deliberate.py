"""Multi-identity LLM deliberation patterns."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from library.harness.sanitize import strip_think_blocks
from library.tools.identity import Identity, format_persona, parse_model
from library.tools.voting import borda_tally, approval_tally

if TYPE_CHECKING:
    from library.harness.context import InstanceContext

logger = logging.getLogger(__name__)


def generate_with_identity(
    ctx: InstanceContext,
    identity: Identity,
    prompt: str,
    context: str = "",
    model: str = "",
    max_tokens: int = 4096,
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
        "max_tokens": max_tokens,
        "caller": f"generate_{identity.name}",
    }
    if system:
        kwargs["system"] = system
    if use_model:
        kwargs["model"] = use_model
    if use_provider:
        kwargs["provider"] = use_provider

    response = ctx.llm([{"role": "user", "content": prompt}], **kwargs)
    tool_calls = getattr(response, "tool_calls", []) or []
    if tool_calls:
        tool_names = ",".join(
            str(getattr(tc, "name", "")) for tc in tool_calls if getattr(tc, "name", "")
        ) or "-"
        logger.info(
            "LLM caller=%s model=%s finish_reason=%s tool_calls=%d tools=%s",
            kwargs.get("caller", "?"),
            kwargs.get("model") or "(default)",
            getattr(response, "finish_reason", "unknown"),
            len(tool_calls),
            tool_names,
        )
    else:
        logger.debug(
            "LLM caller=%s model=%s finish_reason=%s tool_calls=0",
            kwargs.get("caller", "?"),
            kwargs.get("model") or "(default)",
            getattr(response, "finish_reason", "unknown"),
        )
    return strip_think_blocks(response.message or "")


def multi_generate(
    ctx: InstanceContext,
    identities: list[Identity],
    prompt: str,
    context: str = "",
    model: str = "",
    context_by_identity: dict[str, str] | None = None,
) -> dict[str, str]:
    """Generate text for each identity. Returns {identity.name → text}.

    model="" → each identity uses its own model field.
    """
    outputs: dict[str, str] = {}
    for identity in identities:
        use_context = context_by_identity.get(identity.name, context) if context_by_identity else context
        if use_context and len(use_context) > _VOTE_CONTEXT_MAX_CHARS:
            use_context = use_context[-_VOTE_CONTEXT_MAX_CHARS:]
        outputs[identity.name] = generate_with_identity(ctx, identity, prompt, use_context, model)
    return outputs


def _generate_approval(
    ctx: InstanceContext,
    voter: Identity,
    candidates: dict[str, str],
    prompt: str,
    model: str = "",
    vote_context: str = "",
) -> list[str]:
    """Ask a voter to approve exactly 2 candidates. Returns list of approved candidate ids."""
    candidate_ids = list(candidates.keys())
    if len(candidates) <= 2:
        # If 2 or fewer candidates, approve all
        return candidate_ids

    persona = format_persona(voter)
    options = "\n".join(f"- {cid}: {text}" for cid, text in candidates.items())
    system = (
        f"You are {voter.name}. Personality: {persona}\n"
        "Select exactly 2 candidates whose replies best address the conversation."
    )
    truncated_context = vote_context[-_VOTE_CONTEXT_MAX_CHARS:] if vote_context else ""
    context_block = f"\nShared context:\n{truncated_context}\n\n" if truncated_context else ""
    user_msg = (
        f"Conversation:\n{prompt}\n\n"
        f"{context_block}"
        f"Candidates:\n{options}\n\n"
        f'Return JSON: {{"approved": ["id1", "id2"]}}'
    )

    # Resolve model/provider
    if model:
        use_provider, use_model = parse_model(model)
    else:
        use_model = voter.model
        use_provider = voter.provider

    kwargs: dict = {
        "max_tokens": 2048,
        "caller": f"vote_{voter.name}",
        "system": system,
    }
    if use_model:
        kwargs["model"] = use_model
    if use_provider:
        kwargs["provider"] = use_provider

    response = ctx.llm([{"role": "user", "content": user_msg}], **kwargs)
    tool_calls = getattr(response, "tool_calls", []) or []
    if tool_calls:
        tool_names = ",".join(
            str(getattr(tc, "name", "")) for tc in tool_calls if getattr(tc, "name", "")
        ) or "-"
        logger.info(
            "LLM caller=%s model=%s finish_reason=%s tool_calls=%d tools=%s",
            kwargs.get("caller", "?"),
            kwargs.get("model") or "(default)",
            getattr(response, "finish_reason", "unknown"),
            len(tool_calls),
            tool_names,
        )
    else:
        logger.debug(
            "LLM caller=%s model=%s finish_reason=%s tool_calls=0",
            kwargs.get("caller", "?"),
            kwargs.get("model") or "(default)",
            getattr(response, "finish_reason", "unknown"),
        )

    try:
        raw = response.message.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(raw)
        approved = [str(a) for a in data.get("approved", [])]
    except Exception:
        approved = []

    # Normalize: ensure exactly 2 valid candidates, pick first 2 if more, fill with remaining if fewer
    normalized = [cid for cid in approved if cid in candidate_ids][:2]
    if len(normalized) < 2:
        remaining = [cid for cid in candidate_ids if cid not in normalized]
        normalized.extend(remaining[: 2 - len(normalized)])
    
    return normalized


def multi_vote(
    ctx: InstanceContext,
    identities: list[Identity],
    candidates: dict[str, str],
    prompt: str,
    model: str = "",
    exclude_own: bool = False,
    top_n: int = 0,
    vote_context: str = "",
    vote_context_by_identity: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    """Have each identity approve exactly 2 candidates. Returns {voter.name → approved candidate ids}.

    exclude_own=True: voter's own name is excluded from candidates they can approve.
    top_n: (deprecated, ignored - always returns exactly 2 approvals)
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

        if len(voter_candidates) <= 2:
            approved = list(voter_candidates.keys())
        else:
            use_vote_context = (
                vote_context_by_identity.get(voter.name, vote_context)
                if vote_context_by_identity
                else vote_context
            )
            approved = _generate_approval(
                ctx,
                voter,
                voter_candidates,
                prompt,
                model,
                use_vote_context,
            )

        votes[voter.name] = approved

    return votes


def deliberate(
    ctx: InstanceContext,
    identities: list[Identity],
    prompt: str,
    context: str = "",
    context_by_identity: dict[str, str] | None = None,
    subset: list[Identity] | None = None,
    model: str = "",
    exclude_own: bool = False,
    top_n: int = 0,
    consensus_threshold: float = 0.6,
    vote_context: str = "",
    vote_context_by_identity: dict[str, str] | None = None,
) -> dict:
    """Full deliberation: generate candidates, vote, tally using approval voting.

    Each voter approves exactly 2 candidates (or all if fewer than 3).
    Winner needs consensus (votes/voters) > consensus_threshold.
    If winner alone insufficient but top 2 combined > threshold, enters fallback combining mode.

    subset: identities who generate candidates (None = all identities).
    Returns: winner_member, winner_message, scores, candidate_count, vote_count,
             is_tie, has_consensus, has_combined_consensus, candidates dict, votes dict.
    """
    generators = subset if subset is not None else identities
    candidates = multi_generate(ctx, generators, prompt, context, model, context_by_identity)

    if not candidates:
        return {
            "winner_member": "",
            "winner_message": "",
            "scores": {},
            "candidate_count": 0,
            "vote_count": 0,
            "is_tie": False,
            "has_consensus": False,
            "has_combined_consensus": False,
            "candidates": {},
            "votes": {},
        }

    votes = multi_vote(
        ctx,
        identities,
        candidates,
        prompt,
        model,
        exclude_own,
        top_n,
        vote_context,
        vote_context_by_identity,
    )
    tally = approval_tally(candidates, votes, consensus_threshold)

    return {
        "winner_member": tally["winner_member"],
        "winner_message": tally["winner_message"],
        "second_member": tally.get("second_member", ""),
        "second_message": tally.get("second_message", ""),
        "scores": tally["scores"],
        "candidate_count": tally["candidate_count"],
        "vote_count": tally["vote_count"],
        "is_tie": tally["is_tie"],
        "consensus": tally["consensus"],
        "combined_consensus": tally.get("combined_consensus", 0.0),
        "has_consensus": tally["has_consensus"],
        "has_combined_consensus": tally.get("has_combined_consensus", False),
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
    max_tokens: int = 4096,
    prompt_template: str = "",
) -> str:
    """Rewrite winning text through identity's voice.

    all_candidates provided → unified composition citing all candidates (Thrivemind style).
    all_candidates absent  → direct rewrite through identity's voice (Hecate style).
    prompt_template: optional template with {winning_text} and {all_candidates} placeholders.
    """
    if prompt_template:
        # Use provided template
        if all_candidates:
            snippets = "\n".join(f"- {text}" for text in all_candidates.values())
            prompt = prompt_template.replace("{winning_text}", winning_text).replace("{all_candidates}", snippets)
        else:
            prompt = prompt_template.replace("{winning_text}", winning_text)
    elif all_candidates:
        # Default Thrivemind-style prompt
        snippets = "\n".join(f"- {text}" for text in all_candidates.values())
        prompt = (
            f"Winning candidate:\n{winning_text}\n\n"
            f"All candidates:\n{snippets}\n\n"
            "Write the final, unified reply."
        )
    else:
        # Default Hecate-style prompt
        prompt = f"Rewrite this response in your own voice and style:\n\n{winning_text}"

    return generate_with_identity(
        ctx,
        identity,
        prompt,
        context,
        model,
        max_tokens=max_tokens,
    )


_VOICE_MEMORY_MAX_CHARS = 1500   # per-field cap for think_with_context
_VOTE_CONTEXT_MAX_CHARS = 1500   # per-voter context cap for _generate_ranking


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
            parts.append(f"## Your Subconscious\n{voice_memory['subconscious'][-_VOICE_MEMORY_MAX_CHARS:]}")
        if voice_memory.get("motivation"):
            parts.append(f"## Your Motivation\n{voice_memory['motivation'][-_VOICE_MEMORY_MAX_CHARS:]}")
        if voice_memory.get("thinking"):
            parts.append(f"## Your Previous Thoughts\n{voice_memory['thinking'][-_VOICE_MEMORY_MAX_CHARS:]}")

    full_context = "\n\n".join(parts)

    others_block = ""
    if others_thinking:
        # Randomize order to prevent precedence bias
        import random
        names = list(others_thinking.keys())
        random.shuffle(names)
        lines = [f"**{name}**: {others_thinking[name]}" for name in names]
        others_block = (
            "\n\n## Other voices' thoughts (not yours)\n"
            "Treat these as advisory input from the other voices.\n\n"
            + "\n\n".join(lines)
        )

    prompt = (
        f"You are {identity.name}, one voice in a group guiding the same entity.\n"
        "Guidance:\n"
        "- Treat sections labeled 'Your ...' as your own memory and commitments.\n"
        "- Treat 'Other voices' content as someone else's perspective, not your own memory.\n"
        "- Think from your own perspective while coordinating with the other voices.\n"
        f"{others_block}\n\n"
        f"Write your current thoughts for this iteration. Start with '**{identity.name}'s Current Thoughts**' "
        f"and end with '*— {identity.name}*' so others know who is speaking."
    )
    return generate_with_identity(ctx, identity, prompt, full_context, "")
