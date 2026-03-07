"""Canonical think-block sanitization for LLM responses."""

from __future__ import annotations

import re

THINK_PATTERN = re.compile(
    r"(?is)<\s*(think|thinking|analysis|reasoning)\b[^>]*>.*?<\s*/\s*\1\s*>"
)
THINK_FENCE_PATTERN = re.compile(
    r"(?is)```(?:\s*(?:think|thinking|analysis|reasoning)[^\n]*)\n.*?```"
)
THINK_LINE_PATTERN = re.compile(
    r"(?im)^\s*<\s*(?:think|thinking|analysis|reasoning)\b[^>]*>.*$"
)
THINK_SINGLE_TAG_PATTERN = re.compile(
    r"(?is)</?\s*(?:think|thinking|analysis|reasoning)\b[^>]*>"
)


def strip_think_blocks(text: str) -> str:
    """Remove all think/reasoning blocks from text."""
    cleaned = THINK_PATTERN.sub("", text or "")
    cleaned = THINK_FENCE_PATTERN.sub("", cleaned)
    cleaned = THINK_LINE_PATTERN.sub("", cleaned)
    cleaned = THINK_SINGLE_TAG_PATTERN.sub("", cleaned)
    return cleaned.strip()
