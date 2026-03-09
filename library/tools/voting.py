"""Pure voting functions — no LLM, no ctx."""

from __future__ import annotations

import random
from typing import Any


def borda_tally(candidates: dict[str, str], votes: dict[str, list[str]]) -> dict:
    """Borda count over candidate id→text, votes as voter_id→ranked id list.

    Returns a dict with winner_member, winner_message, scores, candidate_count,
    vote_count, and is_tie (True when all scores equal and >1 candidate).
    """
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
    winner_score = scores[winner_id]
    is_tie = len(candidates) > 1 and all(v == winner_score for v in scores.values())

    return {
        "winner_member": winner_id,
        "winner_message": candidates[winner_id],
        "scores": scores,
        "candidate_count": len(candidates),
        "vote_count": len(votes),
        "is_tie": is_tie,
    }


def approval_weights(identities: list, floor: int = 1) -> list[float]:
    """Return max(0, identity.approval + floor) for each identity."""
    return [float(max(0, getattr(i, "approval", 0) + floor)) for i in identities]


def weighted_sample(
    population: list,
    weights: list[float],
    n: int,
    rng: random.Random | None = None,
) -> list:
    """Sample n items without replacement using weights. Falls back to uniform if all weights 0."""
    rng = rng or random.Random()
    if not population:
        return []
    n = min(n, len(population))
    total = sum(weights)

    if total == 0:
        return rng.sample(population, n)

    selected = []
    remaining = list(zip(weights, population))
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
