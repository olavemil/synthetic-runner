"""Validate LLM responses to detect and reject pathological outputs."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


def _get_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs (separated by blank lines)."""
    return [p.strip() for p in re.split(r'\n\s*\n', text.strip()) if p.strip()]


def _get_sentences(text: str) -> list[str]:
    """Extract sentences from text."""
    # Split on periods, question marks, exclamation marks, but be lenient
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]


def _normalize_for_comparison(text: str) -> str:
    """Normalize text for repetition detection (lowercase, collapse whitespace)."""
    return ' '.join(text.lower().split())


def _check_repeating_paragraphs(text: str) -> tuple[bool, str]:
    """Detect if paragraphs repeat. Return (is_bad, reason)."""
    paragraphs = _get_paragraphs(text)
    if len(paragraphs) < 3:
        return False, ""

    normalized = [_normalize_for_comparison(p) for p in paragraphs]

    # Check for exact duplicates
    seen = {}
    for i, para in enumerate(normalized):
        if para in seen:
            return True, f"paragraph {i} duplicates paragraph {seen[para]}"
        seen[para] = i

    return False, ""


def _check_repeating_sentences(text: str) -> tuple[bool, str]:
    """Detect if sentences repeat frequently. Return (is_bad, reason)."""
    sentences = _get_sentences(text)
    if len(sentences) < 5:
        return False, ""

    normalized = [_normalize_for_comparison(s) for s in sentences]

    # Count sentence frequency
    counts = {}
    for s in normalized:
        counts[s] = counts.get(s, 0) + 1

    # If any sentence appears more than twice, it's suspicious (in typical varied text), unless it's a header of some kind
    for s, count in counts.items():
        if count > 2 and len(s) > 20:  # Ignore very short sentences which might be common phrases
            return True, f"sentence appears {count} times: '{s[:50]}...'"

    return False, ""


def _check_excessive_length_without_breaks(text: str) -> tuple[bool, str]:
    """Detect massive single blocks without natural paragraph breaks. Return (is_bad, reason)."""
    paragraphs = _get_paragraphs(text)

    # Should have reasonable number of paragraphs for the length
    word_count = len(text.split())

    # Rule: more than 1000 words should have at least 5 paragraphs
    if word_count > 1000 and len(paragraphs) < 5:
        return True, f"{word_count} words in only {len(paragraphs)} paragraphs (expected at least 5)"

    # Rule: more than 500 words should have at least 3 paragraphs
    if word_count > 500 and len(paragraphs) < 3:
        return True, f"{word_count} words in only {len(paragraphs)} paragraphs (expected at least 3)"

    return False, ""


def is_response_pathological(text: str) -> tuple[bool, str]:
    """Check if LLM response is pathological (repetitive, excessively long without breaks, etc).

    Returns:
        (is_bad, reason) - True if response should be rejected, along with human-readable reason.
    """
    if not text or not text.strip():
        return False, ""

    # Check for repeating paragraphs
    is_bad, reason = _check_repeating_paragraphs(text)
    if is_bad:
        logger.warning("Response rejected: %s", reason)
        return True, reason

    # Check for repeating sentences
    is_bad, reason = _check_repeating_sentences(text)
    if is_bad:
        logger.warning("Response rejected: %s", reason)
        return True, reason

    # Check for excessive length without paragraph breaks
    is_bad, reason = _check_excessive_length_without_breaks(text)
    if is_bad:
        logger.warning("Response rejected: %s", reason)
        return True, reason

    return False, ""

def normalize_pathological_response(text: str) -> str:
    """Attempt to salvage a pathological response by normalizing it (e.g. removing repeated paragraphs).

    This is a best-effort attempt and may not always produce a good result, but can be useful in cases where the LLM output is mostly fine except for some repetition.

    Returns:
        Normalized text with obvious pathologies removed.
    """
    paragraphs = _get_paragraphs(text)
    seen = set()
    unique_paragraphs = []
    for p in paragraphs:
        norm = _normalize_for_comparison(p)
        if norm not in seen:
            seen.add(norm)
            unique_paragraphs.append(p)
    return "\n\n".join(unique_paragraphs)
