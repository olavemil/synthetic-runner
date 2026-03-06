"""Tests for reusable hivemind toolkit primitives."""

from types import SimpleNamespace

from symbiosis.harness.adapters import Event
from symbiosis.harness.store import open_store, NamespacedStore
from symbiosis.toolkit.hivemind import (
    load_hivemind_config,
    claim_fresh_events,
    create_round_from_events,
    get_round,
    submit_candidate,
    submit_vote,
    get_candidates,
    get_votes,
    tally_borda,
    round_ready,
)


class DummyCtx:
    def __init__(self, instance_id: str, store_db, hivemind_cfg=None):
        self.instance_id = instance_id
        self._store_db = store_db
        self._hivemind_cfg = hivemind_cfg or {}

    def config(self, key: str):
        if key == "hivemind":
            return self._hivemind_cfg
        return None

    def shared_store(self, namespace: str):
        return NamespacedStore(self._store_db, f"species:test:{namespace}")

    def llm(self, messages, **kwargs):  # noqa: ARG002
        return SimpleNamespace(message="")


class TestHivemindToolkit:
    def test_load_hivemind_config_defaults_and_custom(self):
        db = open_store()
        default_ctx = DummyCtx("a", db)
        cfg = load_hivemind_config(default_ctx)
        assert cfg.role == "speaker_coordinator"
        assert cfg.quorum == 3
        assert cfg.voice_space == "main"

        custom_ctx = DummyCtx(
            "b",
            db,
            hivemind_cfg={"role": "worker", "quorum": 5, "voice_space": "lobby"},
        )
        custom = load_hivemind_config(custom_ctx)
        assert custom.role == "worker"
        assert custom.quorum == 5
        assert custom.voice_space == "lobby"

    def test_claim_fresh_events_is_shared_and_deduped(self):
        db = open_store()
        ctx_a = DummyCtx("agent-a", db)
        ctx_b = DummyCtx("agent-b", db)
        evt = Event(event_id="$1", sender="@u:matrix.org", body="hi", timestamp=1, room="main")

        first = claim_fresh_events(ctx_a, [evt])
        second = claim_fresh_events(ctx_b, [evt])

        assert len(first) == 1
        assert second == []

    def test_round_and_voting_lifecycle(self):
        db = open_store()
        ctx = DummyCtx("agent-a", db)
        evt = Event(event_id="$1", sender="@u:matrix.org", body="hi", timestamp=1, room="main")
        round_data = create_round_from_events(ctx, [evt], source_space="main")
        round_id = round_data["id"]

        submit_candidate(
            ctx,
            round_id,
            "worker-1",
            "option one",
            persona="bold",
            max_chars=280,
        )
        submit_candidate(
            ctx,
            round_id,
            "worker-2",
            "option two",
            persona="calm",
            max_chars=280,
        )
        submit_vote(ctx, round_id, "worker-1", ["worker-2", "worker-1"], "prefer two")
        submit_vote(ctx, round_id, "worker-2", ["worker-2", "worker-1"], "agree")

        candidates = get_candidates(ctx, round_id)
        votes = get_votes(ctx, round_id)
        tally = tally_borda(candidates, votes)

        assert tally["winner_member"] == "worker-2"
        assert tally["candidate_count"] == 2
        assert tally["vote_count"] == 2

        stored_round = get_round(ctx, round_id)
        assert stored_round is not None
        assert round_ready(
            stored_round,
            candidate_count=2,
            vote_count=2,
            quorum=2,
            timeout_s=45,
        )
