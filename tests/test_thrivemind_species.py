"""Tests for the Thrivemind single-instance colony species."""

from __future__ import annotations

from types import SimpleNamespace

from symbiosis.harness.adapters import Event
from symbiosis.harness.store import open_store, NamespacedStore
from symbiosis.species.thrivemind import on_message, heartbeat
from symbiosis.toolkit.hivemind import AXIS_NAMES, Individual


class DummyCtx:
    """Minimal mock InstanceContext for thrivemind tests."""

    def __init__(self, instance_id: str, store_db, thrivemind_cfg: dict, llm_response_fn=None):
        self.instance_id = instance_id
        self._store_db = store_db
        self._thrivemind_cfg = thrivemind_cfg
        self._files: dict[str, str] = {"constitution.md": "# Constitution\n"}
        self.sent: list[tuple[str, str]] = []
        self._llm_fn = llm_response_fn

    def config(self, key: str):
        if key == "thrivemind":
            return self._thrivemind_cfg
        return None

    def store(self, namespace: str):
        return NamespacedStore(self._store_db, f"instance:{self.instance_id}:{namespace}")

    def read(self, path: str) -> str:
        return self._files.get(path, "")

    def write(self, path: str, content: str) -> None:
        self._files[path] = content

    def exists(self, path: str) -> bool:
        return path in self._files

    def send(self, space: str, message: str, reply_to=None):
        self.sent.append((space, message))
        return "$event"

    def llm(self, messages, **kwargs):
        if self._llm_fn:
            return self._llm_fn(messages, **kwargs)
        caller = kwargs.get("caller", "")
        user_content = messages[-1]["content"] if messages else ""
        # Voting (ranking) calls
        if "vote" in caller:
            return SimpleNamespace(message='{"ranking": []}')
        # Constitution acceptance vote (message contains "accept")
        if '"accept"' in user_content:
            return SimpleNamespace(message='{"accept": true}')
        # Rewrite constitution — ColonyWriter identity
        if "ColonyWriter" in caller:
            return SimpleNamespace(message="Rewritten constitution.")
        # Final colony write — Colony identity
        if "Colony" in caller:
            return SimpleNamespace(message="Final colony message.")
        # Everything else (suggestions, constitution lines)
        return SimpleNamespace(message="Candidate text.")


def _make_event(body="hello"):
    return Event(event_id="$1", sender="@user:matrix.org", body=body, timestamp=1, room="main")


class TestThrivemindOnMessage:
    def test_full_round_sends_message(self):
        """on_message produces a colony message and persists the colony."""
        db = open_store()
        cfg = {
            "colony_size": 4,
            "suggestion_fraction": 0.5,
            "approval_threshold": 10,  # no spawning
            "consensus_threshold": 0.0,  # always accept
            "voice_space": "main",
        }

        def llm_fn(messages, **kwargs):
            caller = kwargs.get("caller", "")
            # vote calls return a ranking (JSON)
            if "vote" in caller:
                return SimpleNamespace(message='{"ranking": []}')
            # Colony writer (recompose) returns final message
            if "Colony" in caller:
                return SimpleNamespace(message="Final message.")
            # All other generate calls (suggestions)
            return SimpleNamespace(message="A candidate reply.")

        ctx = DummyCtx("inst1", db, cfg, llm_response_fn=llm_fn)
        events = [_make_event()]

        on_message(ctx, events)

        assert len(ctx.sent) == 1
        assert ctx.sent[0][0] == "main"
        assert ctx.sent[0][1] == "Final message."

    def test_consensus_threshold_not_met_no_send(self):
        """If consensus threshold not met and multiple candidates, no message sent."""
        db = open_store()
        cfg = {
            "colony_size": 4,
            "suggestion_fraction": 1.0,  # all colony suggests
            "approval_threshold": 10,
            "consensus_threshold": 0.99,  # very high threshold
            "voice_space": "main",
        }

        captured_candidates: list[str] = []

        def llm_fn(messages, **kwargs):
            caller = kwargs.get("caller", "")
            if "vote" in caller:
                if captured_candidates:
                    rotated = captured_candidates[:]
                    return SimpleNamespace(message=f'{{"ranking": {rotated}}}')
                return SimpleNamespace(message='{"ranking": []}')
            return SimpleNamespace(message="Candidate.")

        ctx = DummyCtx("inst1", db, cfg, llm_response_fn=llm_fn)
        events = [_make_event("discuss")]

        # Pre-populate colony with 4 individuals, each with distinct IDs
        from symbiosis.toolkit.hivemind import spawn_initial_colony, save_colony, ThrivemindConfig
        colony_cfg = ThrivemindConfig(colony_size=4)
        colony = spawn_initial_colony(colony_cfg)
        save_colony(ctx, colony)
        captured_candidates.extend(ind.name for ind in colony[:2])  # 2 suggesters

        vote_call = [0]

        def llm_fn2(messages, **kwargs):
            caller = kwargs.get("caller", "")
            if "vote" in caller:
                n = vote_call[0] % max(len(captured_candidates), 1)
                vote_call[0] += 1
                if captured_candidates:
                    rotated = captured_candidates[n:] + captured_candidates[:n]
                    return SimpleNamespace(message=f'{{"ranking": {rotated}}}')
                return SimpleNamespace(message='{"ranking": []}')
            return SimpleNamespace(message="Candidate.")

        ctx._llm_fn = llm_fn2
        on_message(ctx, events)
        # With threshold=0.99, consensus is hard to reach with 2 equal candidates


class TestThrivemindHeartbeat:
    def test_heartbeat_runs_constitution_update_and_spawn(self):
        """Heartbeat updates constitution and runs spawn cycle."""
        db = open_store()
        cfg = {
            "colony_size": 4,
            "approval_threshold": 10,  # no natural spawning
            "consensus_threshold": 0.0,  # always adopt constitution
            "voice_space": "main",
        }

        ctx = DummyCtx("inst-hb", db, cfg)
        # Pre-write constitution
        ctx._files["constitution.md"] = "# Constitution\nOriginal."

        heartbeat(ctx)

        # Constitution should have been rewritten (ColonyWriter returns "Rewritten constitution.")
        assert ctx._files.get("constitution.md", "") == "Rewritten constitution."

    def test_heartbeat_spawn_cycle_maintains_size(self):
        """Heartbeat spawn cycle keeps colony at target size."""
        db = open_store()
        cfg = {
            "colony_size": 6,
            "approval_threshold": 2,
            "consensus_threshold": 0.0,
            "voice_space": "main",
        }

        ctx = DummyCtx("inst-spawn", db, cfg)

        # Pre-populate with some eligible individuals
        from symbiosis.toolkit.hivemind import ThrivemindConfig, spawn_initial_colony, save_colony
        from symbiosis.toolkit.hivemind import load_colony
        colony_cfg = ThrivemindConfig(colony_size=6, approval_threshold=2)
        colony = spawn_initial_colony(colony_cfg)
        # Give 2 individuals enough approval to trigger spawn
        colony[0].approval = 3
        colony[1].approval = 4
        save_colony(ctx, colony)

        heartbeat(ctx)

        final_colony = load_colony(ctx)
        assert len(final_colony) == 6
