"""Local file messaging adapter — read/write messages as JSON files in a directory."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from . import Event, MessagingAdapter


class LocalFileAdapter(MessagingAdapter):
    def __init__(self, base_dir: str | Path):
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    def _space_dir(self, space_handle: str) -> Path:
        d = self._base / space_handle
        d.mkdir(parents=True, exist_ok=True)
        return d

    def send(self, space_handle: str, message: str, reply_to: str | None = None) -> str:
        event_id = str(uuid.uuid4())
        ts = int(time.time() * 1000)
        event = {
            "event_id": event_id,
            "sender": "local",
            "body": message,
            "timestamp": ts,
            "reply_to": reply_to,
        }
        target = self._space_dir(space_handle) / f"{ts}_{event_id}.json"
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(event), encoding="utf-8")
        tmp.rename(target)
        return event_id

    def poll(self, space_handle: str, since_token: str | None = None) -> tuple[list[Event], str]:
        space_dir = self._space_dir(space_handle)
        since_ts = int(since_token) if since_token else 0

        events = []
        max_ts = since_ts

        for f in sorted(space_dir.glob("*.json")):
            data = json.loads(f.read_text(encoding="utf-8"))
            ts = data.get("timestamp", 0)
            if ts > since_ts:
                events.append(Event(
                    event_id=data["event_id"],
                    sender=data.get("sender", "unknown"),
                    body=data.get("body", ""),
                    timestamp=ts,
                    room=space_handle,
                ))
                max_ts = max(max_ts, ts)

        next_token = str(max_ts) if max_ts > since_ts else (since_token or "0")
        return events, next_token

    def get_space_context(self, space_handle: str) -> dict:
        return {
            "room_id": space_handle,
            "name": space_handle,
            "topic": "",
            "members": [],
        }
