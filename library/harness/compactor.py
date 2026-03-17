"""Compactor — harness-driven file compaction using a dedicated LLM call.

Triggered automatically when written file content exceeds a configured threshold.
Uses a separate provider/model (e.g. devstral) to reduce redundancy while
preserving tone, voice, and significant insights.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from library.harness.config import CompactConfig
    from library.harness.providers import LLMProvider

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a careful editor compacting a memory or thinking file that has grown too long.

Rules:
- Remove redundancy and repetition
- Preserve all significant insights, decisions, conclusions, and unique observations
- Maintain the author's voice and tone exactly
- Keep chronological markers and structural headers where meaningful
- Aim for 40–60% of the original length

Return only the compacted content. No preamble, no explanation, no meta-commentary.\
"""


class Compactor:
    """Compacts file content via LLM when it exceeds the configured threshold."""

    def __init__(self, provider: LLMProvider, config: CompactConfig):
        self._provider = provider
        self._config = config

    def maybe_compact(self, content: str, path: str | None = None) -> str | None:
        """Return compacted content if threshold exceeded, else None.

        Args:
            content: The full file content to potentially compact.
            path: Optional file path hint for logging.

        Returns:
            Compacted string if content exceeded threshold, None otherwise.
        """
        if len(content) <= self._config.threshold_chars:
            return None

        label = path or "file"
        original_chars = len(content)
        logger.info(
            "Compacting %s (%d chars > threshold %d) using %s/%s",
            label,
            original_chars,
            self._config.threshold_chars,
            self._config.provider,
            self._config.model,
        )

        try:
            response = self._provider.create(
                model=self._config.model,
                messages=[{"role": "user", "content": content}],
                system=_SYSTEM_PROMPT,
                max_tokens=4096,
                caller="compactor",
            )
        except Exception:
            logger.exception("Compaction LLM call failed for %s — keeping original", label)
            return None

        compacted = (response.message or "").strip()
        if not compacted:
            logger.warning("Compaction returned empty content for %s — keeping original", label)
            return None

        # Safety: reject if the result is longer than the original (shouldn't happen but guard it)
        if len(compacted) >= original_chars:
            logger.warning(
                "Compaction did not reduce %s (%d → %d chars) — keeping original",
                label,
                original_chars,
                len(compacted),
            )
            return None

        logger.info(
            "Compacted %s: %d → %d chars (%.0f%%)",
            label,
            original_chars,
            len(compacted),
            100 * len(compacted) / original_chars,
        )
        return compacted
