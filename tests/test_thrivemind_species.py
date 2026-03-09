"""Tests for the Thrivemind single-instance colony species."""

from __future__ import annotations

import logging
from types import SimpleNamespace

from library.harness.adapters import Event
from library.harness.store import open_store, NamespacedStore
from library.species.thrivemind import on_message, heartbeat
from library.tools.thrivemind import AXIS_NAMES, Individual


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


def _make_event(body="hello", room="main", timestamp=1):
    return Event(event_id="$1", sender="@user:matrix.org", body=body, timestamp=timestamp, room=room)


class TestThrivemindOnMessage:
    def test_full_round_sends_message(self):
        """on_message produces a colony message and persists the colony."""
        db = open_store()
        cfg = {
            "min_colony_size": 4, "max_colony_size": 4,
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
        assert ctx.sent[0][1].startswith("Consensus:")
        assert ctx.sent[0][1].endswith("Final message.")

    def test_on_message_sends_reply_to_event_room(self):
        db = open_store()
        cfg = {
            "min_colony_size": 4, "max_colony_size": 4,
            "suggestion_fraction": 0.5,
            "approval_threshold": 10,
            "consensus_threshold": 0.0,
            "voice_space": "main",
        }

        def llm_fn(messages, **kwargs):
            caller = kwargs.get("caller", "")
            if "vote" in caller:
                return SimpleNamespace(message='{"ranking": []}')
            if "Colony" in caller:
                return SimpleNamespace(message="Reply in incoming room.")
            return SimpleNamespace(message="Candidate text.")

        ctx = DummyCtx("inst-room", db, cfg, llm_response_fn=llm_fn)
        on_message(ctx, [_make_event("hello from friend-room", room="friend-room", timestamp=2)])

        assert len(ctx.sent) == 1
        assert ctx.sent[0][0] == "friend-room"
        assert ctx.sent[0][1].startswith("Consensus:")
        assert ctx.sent[0][1].endswith("Reply in incoming room.")

    def test_consensus_threshold_not_met_uses_writer_on_joined_fallback_reply(self):
        """If consensus is not met, top-rated candidates are joined then rewritten by writer."""
        db = open_store()
        cfg = {
            "min_colony_size": 4, "max_colony_size": 4,
            "suggestion_fraction": 1.0,  # all colony suggests
            "approval_threshold": 10,
            "consensus_threshold": 0.99,  # very high threshold
            "voice_space": "main",
        }

        captured_candidates: list[str] = []

        writer_calls = 0
        writer_prompt = ""
        writer_system = ""

        def llm_fn(messages, **kwargs):
            nonlocal writer_calls
            caller = kwargs.get("caller", "")
            if "vote" in caller:
                if captured_candidates:
                    rotated = captured_candidates[:]
                    return SimpleNamespace(message=f'{{"ranking": {rotated}}}')
                return SimpleNamespace(message='{"ranking": []}')
            if caller == "generate_Colony":
                writer_calls += 1
                return SimpleNamespace(message="Fallback final message.")
            if caller.startswith("generate_"):
                return SimpleNamespace(message=f"Candidate from {caller.removeprefix('generate_')}.")
            return SimpleNamespace(message="Candidate text.")

        ctx = DummyCtx("inst1", db, cfg, llm_response_fn=llm_fn)
        events = [_make_event("discuss")]

        # Pre-populate colony with 4 individuals, each with distinct IDs
        from library.tools.thrivemind import spawn_initial_colony, save_colony, ThrivemindConfig
        colony_cfg = ThrivemindConfig(min_colony_size=4, max_colony_size=4)
        colony = spawn_initial_colony(colony_cfg)
        save_colony(ctx, colony)
        captured_candidates.extend(ind.name for ind in colony[:2])  # 2 suggesters

        vote_call = [0]

        def llm_fn2(messages, **kwargs):
            nonlocal writer_calls, writer_prompt, writer_system
            caller = kwargs.get("caller", "")
            if "vote" in caller:
                n = vote_call[0] % max(len(captured_candidates), 1)
                vote_call[0] += 1
                if captured_candidates:
                    rotated = captured_candidates[n:] + captured_candidates[:n]
                    return SimpleNamespace(message=f'{{"ranking": {rotated}}}')
                return SimpleNamespace(message='{"ranking": []}')
            if caller == "generate_Colony":
                writer_calls += 1
                writer_prompt = messages[-1]["content"] if messages else ""
                writer_system = kwargs.get("system", "")
                return SimpleNamespace(message="Fallback final message.")
            if caller.startswith("generate_"):
                return SimpleNamespace(message=f"Candidate from {caller.removeprefix('generate_')}.")
            return SimpleNamespace(message="Candidate text.")

        ctx._llm_fn = llm_fn2
        on_message(ctx, events)
        # With threshold=0.99 and split votes, fallback should join top candidates.
        assert len(ctx.sent) == 1
        sent_body = ctx.sent[0][1]
        assert sent_body.startswith("Consensus: No consensus")
        assert sent_body.endswith("Fallback final message.")
        assert writer_calls == 1
        assert f"Candidate from {captured_candidates[0]}." in writer_prompt
        assert f"Candidate from {captured_candidates[1]}." in writer_prompt
        assert "Speak as a unified hivemind voice." in writer_system
        assert "up to about 100 words" in writer_system

    def test_on_message_writes_reflection_and_uses_it_in_vote_context(self):
        db = open_store()
        cfg = {
            "min_colony_size": 4, "max_colony_size": 4,
            "suggestion_fraction": 0.5,
            "approval_threshold": 10,
            "consensus_threshold": 0.99,
            "voice_space": "main",
        }
        calls: list[tuple[str, str, dict]] = []

        def llm_fn(messages, **kwargs):
            caller = kwargs.get("caller", "")
            user_content = messages[-1]["content"] if messages else ""
            calls.append((caller, user_content, kwargs))
            if caller == "generate_ColonySummarizer":
                return SimpleNamespace(message="shared-summary")
            if "This is your moment to reflect on the state of your colony." in user_content:
                individual = caller.removeprefix("generate_")
                return SimpleNamespace(message=f"reflection-for-{individual}")
            if "vote" in caller:
                return SimpleNamespace(message='{"ranking": []}')
            return SimpleNamespace(message="Candidate.")

        ctx = DummyCtx("inst-reflect", db, cfg, llm_response_fn=llm_fn)
        on_message(ctx, [_make_event("incoming request")])

        reflection_files = sorted(path for path in ctx._files if path.startswith("reflections/"))
        assert len(reflection_files) == 4
        assert "reflections.md" not in ctx._files
        summary_calls = [c for c in calls if c[0] == "generate_ColonySummarizer"]
        assert len(summary_calls) == 1
        vote_calls = [(caller, content) for caller, content, _ in calls if "vote_" in caller]
        assert vote_calls
        for caller, content in vote_calls:
            voter = caller.removeprefix("vote_")
            assert "Shared context:" in content
            assert "Prior Reflection" in content
            assert f"reflection-for-{voter}" in content


class TestThrivemindHeartbeat:
    def test_heartbeat_runs_constitution_update_and_spawn(self, caplog):
        """Heartbeat updates constitution and runs spawn cycle."""
        db = open_store()
        cfg = {
            "min_colony_size": 4, "max_colony_size": 4,
            "approval_threshold": 10,  # no natural spawning
            "consensus_threshold": 0.0,  # always adopt constitution
            "voice_space": "main",
        }

        ctx = DummyCtx("inst-hb", db, cfg)
        # Pre-write constitution
        ctx._files["constitution.md"] = "# Constitution\nOriginal."

        with caplog.at_level(logging.INFO):
            heartbeat(ctx)

        # Constitution should have been rewritten (ColonyWriter returns "Rewritten constitution.")
        assert ctx._files.get("constitution.md", "") == "Rewritten constitution."
        assert "colony.md" in ctx._files
        reflection_files = [path for path in ctx._files if path.startswith("reflections/")]
        assert len(reflection_files) == 4
        assert "contributions.md" in ctx._files
        assert "candidate.md" in ctx._files
        assert "Contributions" in ctx._files["contributions.md"]
        assert ctx._files["candidate.md"] == "Rewritten constitution."
        assert "| Individual | Personality | Approval | Age |" in ctx._files["colony.md"]
        messages = [r.getMessage() for r in caplog.records]
        assert any("Thrivemind constitution adopted; writing constitution.md" in m for m in messages)
        assert any("Writing thrivemind constitution.md" in m for m in messages)

    def test_heartbeat_logs_when_constitution_not_adopted(self, caplog):
        db = open_store()
        cfg = {
            "min_colony_size": 3, "max_colony_size": 3,
            "approval_threshold": 10,
            "consensus_threshold": 1.0,  # impossible with strict >
            "voice_space": "main",
        }

        def llm_fn(messages, **kwargs):
            caller = kwargs.get("caller", "")
            user_content = messages[-1]["content"] if messages else ""
            if "vote" in caller:
                return SimpleNamespace(message='{"ranking": []}')
            if '"accept"' in user_content:
                return SimpleNamespace(message='{"accept": false}')
            if "ColonyWriter" in caller:
                return SimpleNamespace(message="Should not be adopted.")
            return SimpleNamespace(message="Candidate text.")

        ctx = DummyCtx("inst-hb-no-adopt", db, cfg, llm_response_fn=llm_fn)
        ctx._files["constitution.md"] = "# Constitution\nOriginal."

        with caplog.at_level(logging.INFO):
            heartbeat(ctx)

        assert ctx._files.get("constitution.md", "") == "# Constitution\nOriginal."
        assert "colony.md" in ctx._files
        assert "contributions.md" in ctx._files
        assert "candidate.md" in ctx._files
        messages = [r.getMessage() for r in caplog.records]
        assert any("Thrivemind constitution not adopted; skipping constitution.md write" in m for m in messages)

    def test_heartbeat_spawn_cycle_maintains_size(self):
        """Heartbeat spawn cycle keeps colony at target size."""
        db = open_store()
        cfg = {
            "min_colony_size": 6, "max_colony_size": 6,
            "approval_threshold": 2,
            "consensus_threshold": 0.0,
            "voice_space": "main",
        }

        ctx = DummyCtx("inst-spawn", db, cfg)

        # Pre-populate with some eligible individuals
        from library.tools.thrivemind import ThrivemindConfig, spawn_initial_colony, save_colony
        from library.tools.thrivemind import load_colony
        colony_cfg = ThrivemindConfig(min_colony_size=6, max_colony_size=6, approval_threshold=2)
        colony = spawn_initial_colony(colony_cfg)
        # Give 2 individuals enough approval to trigger spawn
        colony[0].approval = 3
        colony[1].approval = 4
        save_colony(ctx, colony)

        heartbeat(ctx)

        final_colony = load_colony(ctx)
        assert len(final_colony) == 6

    def test_heartbeat_sanitizes_contributions_and_candidate_outputs(self):
        db = open_store()
        cfg = {
            "min_colony_size": 1, "max_colony_size": 1,
            "approval_threshold": 10,
            "consensus_threshold": 0.0,
            "voice_space": "main",
        }

        def llm_fn(messages, **kwargs):
            caller = kwargs.get("caller", "")
            user_content = messages[-1]["content"] if messages else ""
            if '"accept"' in user_content:
                return SimpleNamespace(message='{"accept": true}')
            if "Propose exactly one new principle or refinement." in user_content:
                return SimpleNamespace(
                    message="<think>private reasoning without closing tag\n"
                    "Principle: Keep order and purpose. Add extra implementation details."
                )
            if "ColonyWriter" in caller:
                return SimpleNamespace(
                    message="<think>rewrite draft</think>\n"
                    "Here is the rewritten constitution:\n"
                    "# Constitution\n"
                    "- Keep order and purpose.\n"
                    "- Maintain a shared direction.\n"
                )
            return SimpleNamespace(message="brief reflection")

        ctx = DummyCtx("inst-hb-sanitize", db, cfg, llm_response_fn=llm_fn)
        ctx._files["constitution.md"] = "# Constitution\nOriginal."

        heartbeat(ctx)

        contributions = ctx._files.get("contributions.md", "")
        candidate = ctx._files.get("candidate.md", "")
        assert "<think>" not in contributions
        assert "<think>" not in candidate
        assert "Add extra implementation details" not in contributions
        assert "Here is the rewritten constitution" not in candidate
        assert "1. Keep order and purpose." in contributions
        assert "# Constitution" in candidate

    def test_second_vote_round_uses_round_one_context(self, caplog):
        db = open_store()
        cfg = {
            "min_colony_size": 3, "max_colony_size": 3,
            "approval_threshold": 10,
            "consensus_threshold": 0.6,
            "voice_space": "main",
        }
        vote_prompts: list[str] = []

        def llm_fn(messages, **kwargs):
            caller = kwargs.get("caller", "")
            user_content = messages[-1]["content"] if messages else ""
            if "vote" in caller:
                return SimpleNamespace(message='{"ranking": []}')
            if '"accept"' in user_content:
                vote_prompts.append(user_content)
                if "Round 1 results" in user_content:
                    return SimpleNamespace(message='{"accept": true}')
                return SimpleNamespace(message='{"accept": false}')
            if "ColonyWriter" in caller:
                return SimpleNamespace(message="Rewritten constitution.")
            return SimpleNamespace(message="Candidate text.")

        ctx = DummyCtx("inst-hb-round2", db, cfg, llm_response_fn=llm_fn)
        ctx._files["constitution.md"] = "# Constitution\nOriginal."

        with caplog.at_level(logging.INFO):
            heartbeat(ctx)

        assert len(vote_prompts) == 6  # 3 individuals x 2 rounds
        assert any("Stakes:" in prompt for prompt in vote_prompts)
        assert any("Reflections:" in prompt for prompt in vote_prompts)
        assert any("Round 1 results" in prompt for prompt in vote_prompts)
        assert ctx._files.get("constitution.md", "") == "Rewritten constitution."
        messages = [r.getMessage() for r in caplog.records]
        assert any("starting round 2" in m for m in messages)

    def test_heartbeat_retries_empty_contribution_after_think_truncation(self):
        db = open_store()
        cfg = {
            "min_colony_size": 1, "max_colony_size": 1,
            "approval_threshold": 10,
            "consensus_threshold": 0.0,
            "voice_space": "main",
        }
        contribution_calls = 0

        def llm_fn(messages, **kwargs):
            nonlocal contribution_calls
            caller = kwargs.get("caller", "")
            user_content = messages[-1]["content"] if messages else ""
            if '"accept"' in user_content:
                return SimpleNamespace(message='{"accept": true}')
            if "ColonyWriter" in caller:
                return SimpleNamespace(message="Rewritten constitution.")
            if (
                "Propose exactly one new principle or refinement." in user_content
                or "Your previous response was invalid." in user_content
            ):
                contribution_calls += 1
                if contribution_calls == 1:
                    return SimpleNamespace(message="<think>internal planning without close tag")
                return SimpleNamespace(message="Protect coherence while adapting fast.")
            return SimpleNamespace(message="brief reflection")

        ctx = DummyCtx("inst-hb-retry", db, cfg, llm_response_fn=llm_fn)
        ctx._files["constitution.md"] = "# Constitution\nOriginal."

        heartbeat(ctx)

        contributions = ctx._files.get("contributions.md", "")
        assert contribution_calls == 2
        assert "1. Protect coherence while adapting fast." in contributions
