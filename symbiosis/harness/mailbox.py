"""Inter-instance mailbox — harness-mediated message passing between instances."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path


class Mailbox:
    def __init__(self, base_dir: str | Path, instance_id: str):
        self._base = Path(base_dir)
        self._instance_id = instance_id
        self._inbox_dir = self._base / instance_id / "inbox"
        self._inbox_dir.mkdir(parents=True, exist_ok=True)

    def send_to(self, target_id: str, message: str) -> None:
        """Write a message to another instance's inbox."""
        target_dir = self._base / target_id / "inbox"
        target_dir.mkdir(parents=True, exist_ok=True)

        msg = {
            "sender": self._instance_id,
            "body": message,
            "timestamp": int(time.time() * 1000),
        }
        msg_id = str(uuid.uuid4())
        target = target_dir / f"{msg_id}.json"
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(msg), encoding="utf-8")
        tmp.rename(target)

    def read_inbox(self) -> list[dict]:
        """Read and clear all inbox messages."""
        messages = []
        for f in sorted(self._inbox_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                messages.append(data)
                f.unlink()
            except (json.JSONDecodeError, OSError):
                continue
        return messages
