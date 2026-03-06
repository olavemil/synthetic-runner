"""Tests for thrivemind species behavior."""

from __future__ import annotations

import json
from types import SimpleNamespace

from symbiosis.harness.adapters import Event
from symbiosis.harness.store import open_store, NamespacedStore
from symbiosis.species.thrivemind import on_message, heartbeat


class DummyCtx:
    def __init__(self, instance_id: str, store_db, hivemind_cfg: dict):
        self.instance_id = instance_id
        self._store_db = store_db
        self._hivemind_cfg = hivemind_cfg
        self.sent: list[tuple[str, str]] = []
        self.sent_to: list[tuple[str, str]] = []
        self.inbox: list[dict] = []

    def config(self, key: str):
        if key == "hivemind":
            return self._hivemind_cfg
        return None

    def shared_store(self, namespace: str):
        return NamespacedStore(self._store_db, f"species:test:{namespace}")

    def llm(self, messages, **kwargs):  # noqa: ARG002
        caller = kwargs.get("caller", "")
        if caller == "hivemind_candidate":
            return SimpleNamespace(message="Short candidate.")
        if caller == "hivemind_consensus":
            return SimpleNamespace(message="Unified external reply.")
        return SimpleNamespace(message='{"ranking":["coordinator-1"],"rationale":"single"}')

    def send(self, space: str, message: str, reply_to=None):  # noqa: ARG002
        self.sent.append((space, message))
        return "$event"

    def send_to(self, target_id: str, message: str):
        self.sent_to.append((target_id, message))

    def read_inbox(self) -> list[dict]:
        items = list(self.inbox)
        self.inbox = []
        return items


class TestThrivemindSpecies:
    def test_on_message_coordinator_creates_and_emits_consensus(self):
        db = open_store()
        ctx = DummyCtx(
            "coordinator-1",
            db,
            {
                "role": "speaker_coordinator",
                "persona": "strategic",
                "quorum": 1,
                "voice_space": "main",
                "round_timeout_s": 30,
            },
        )
        events = [Event(event_id="$1", sender="@u:matrix.org", body="status?", timestamp=1, room="main")]

        on_message(ctx, events)

        assert len(ctx.sent) == 1
        assert ctx.sent[0][0] == "main"
        assert "Unified external reply." in ctx.sent[0][1]

    def test_worker_does_not_claim_external_events(self):
        db = open_store()
        worker = DummyCtx(
            "worker-1",
            db,
            {"role": "worker", "persona": "skeptical", "quorum": 1, "voice_space": "main"},
        )
        coordinator = DummyCtx(
            "coordinator-1",
            db,
            {"role": "speaker_coordinator", "persona": "lead", "quorum": 1, "voice_space": "main"},
        )
        events = [Event(event_id="$event", sender="@u:matrix.org", body="hello", timestamp=1, room="main")]

        on_message(worker, events)
        on_message(coordinator, events)

        assert worker.sent == []
        assert len(coordinator.sent) == 1

    def test_speaker_forwards_coordinator_outputs_from_inbox(self):
        db = open_store()
        speaker = DummyCtx(
            "speaker-1",
            db,
            {"role": "speaker", "persona": "voice", "quorum": 2, "voice_space": "main"},
        )
        speaker.inbox.append(
            {"body": json.dumps({"kind": "hivemind_output", "space": "main", "message": "Ready."})}
        )

        heartbeat(speaker)
        assert speaker.sent == [("main", "Ready.")]
