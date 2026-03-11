"""Probabilistic decision utilities for neural net-driven behavior.

Provides the core mechanism for converting continuous NN output values (0.0-1.0)
into yes/no decisions via `random() < value`, making every small change in NN
output statistically meaningful.
"""

from __future__ import annotations

import logging
import random

logger = logging.getLogger(__name__)


def probabilistic(value: float, *, label: str = "") -> bool:
    """Return True with probability `value` (clamped to 0-1).

    Args:
        value: Probability threshold, clamped to [0.0, 1.0].
        label: Optional label for logging (e.g. "organization_drive").

    Returns:
        True with probability `value`.
    """
    clamped = max(0.0, min(1.0, value))
    result = random.random() < clamped
    if label:
        logger.info("Decision '%s': value=%.3f outcome=%s", label, clamped, result)
    return result
