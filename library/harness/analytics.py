"""Analytics client — fire-and-forget event tracking.

Sends structured events to a local HTTP analytics endpoint.
All errors are silently swallowed: the service may not be running.
Events are dispatched in daemon threads to avoid blocking callers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import urllib.request
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class AnalyticsClient:
    """Post analytics events to a local HTTP endpoint.

    The service is assumed to be optional (e.g. running only during local
    development/testing). Any network or HTTP error is silently ignored.
    """

    def __init__(
        self,
        base_url: str,
        instance_id: str,
        session_id: str,
        timeout: float = 2.0,
    ):
        self._endpoint = base_url.rstrip("/") + "/api/analytics/events"
        self._analytics_user_id = _pseudonymize(instance_id)
        self._analytics_session_id = _pseudonymize(session_id)
        self._timeout = timeout

    def track(self, event_name: str, properties: dict | None = None) -> None:
        """Fire-and-forget: send an event in a background daemon thread."""
        event = {
            "event_name": event_name,
            "analytics_user_id": self._analytics_user_id,
            "analytics_session_id": self._analytics_session_id,
            "properties": properties or {},
            "client_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        threading.Thread(target=self._send, args=(event,), daemon=True).start()

    def _send(self, event: dict) -> None:
        try:
            data = json.dumps(event).encode()
            req = urllib.request.Request(
                self._endpoint,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self._timeout):
                pass
            logger.debug("Analytics event sent: %s", event["event_name"])
        except Exception as exc:
            logger.debug(
                "Analytics event not delivered (%s): %s",
                type(exc).__name__,
                event["event_name"],
            )


def _pseudonymize(value: str) -> str:
    """One-way hash: consistent but irreversible identifier."""
    return "anon_" + hashlib.sha256(value.encode()).hexdigest()[:12]
