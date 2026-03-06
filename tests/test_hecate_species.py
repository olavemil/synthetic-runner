"""Tests for the Hecate species."""

from __future__ import annotations

from types import SimpleNamespace

from symbiosis.harness.adapters import Event
from symbiosis.species.hecate import on_message, heartbeat


THREE_VOICES_CFG = {
    "voices": [
        {"name": "Aria", "model": "m", "personality": "Analytical"},
        {"name": "Sable", "model": "m", "personality": "Bold"},
        {"name": "Lune", "model": "m", "personality": "Poetic"},
    ],
    "thinking_iterations": 2,
    "voice_space": "main",
}


class DummyCtx:
    def __init__(self, hecate_cfg: dict, llm_fn=None):
        self._hecate_cfg = hecate_cfg
        self._files: dict[str, str] = {
            "memory.md": "# Memory\n",
            "constitution.md": "# Constitution\n",
        }
        self.sent: list[tuple[str, str]] = []
        self._llm_fn = llm_fn

    def config(self, key: str):
        if key == "hecate":
            return self._hecate_cfg
        return None

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
        return SimpleNamespace(message="LLM response.")


def _event(body="hello"):
    return Event(event_id="$1", sender="@u:h.org", body=body, timestamp=1, room="main")


class TestHecateOnMessage:
    def test_two_vote_winner_single_reworded_reply(self):
        """When one suggestion gets 2 votes, a single reworded reply is sent."""
        reword_count = [0]
        suggest_count = [0]

        def llm_fn(messages, **kwargs):
            caller = kwargs.get("caller", "")
            user_content = messages[-1]["content"] if messages else ""

            # Vote calls return a ranking (new format)
            if "vote" in caller:
                # Aria and Lune vote for Sable; Sable votes for Aria (can't vote for itself)
                voter_name = caller.split("_")[-1]
                if voter_name in ("Aria", "Lune"):
                    return SimpleNamespace(message='{"ranking": ["Sable"]}')
                else:  # Sable votes for Aria
                    return SimpleNamespace(message='{"ranking": ["Aria"]}')

            # Recompose / reword: message starts with "Rewrite"
            if user_content.startswith("Rewrite"):
                reword_count[0] += 1
                return SimpleNamespace(message="Reworded message.")

            # Subconscious update: message contains "subconscious"
            if "subconscious" in user_content.lower():
                return SimpleNamespace(message="Subconscious update.")

            # Suggestion generation
            suggest_count[0] += 1
            return SimpleNamespace(message=f"suggestion {suggest_count[0]}")

        ctx = DummyCtx(THREE_VOICES_CFG, llm_fn=llm_fn)
        on_message(ctx, [_event()])

        assert len(ctx.sent) == 1
        assert ctx.sent[0][0] == "main"
        assert ctx.sent[0][1] == "Reworded message."
        assert reword_count[0] == 1

    def test_three_way_tie_joined_reply(self):
        """When all 3 get 1 vote each, all 3 rewrites are joined with separator."""
        reword_count = [0]

        def llm_fn(messages, **kwargs):
            caller = kwargs.get("caller", "")
            user_content = messages[-1]["content"] if messages else ""

            if "vote" in caller:
                # Each voice votes for a different candidate → tie
                voter_name = caller.split("_")[-1]
                if voter_name == "Aria":
                    return SimpleNamespace(message='{"ranking": ["Sable"]}')
                elif voter_name == "Sable":
                    return SimpleNamespace(message='{"ranking": ["Lune"]}')
                else:
                    return SimpleNamespace(message='{"ranking": ["Aria"]}')

            if user_content.startswith("Rewrite"):
                reword_count[0] += 1
                return SimpleNamespace(message=f"Reword {reword_count[0]}")

            if "subconscious" in user_content.lower():
                return SimpleNamespace(message="sub.")

            return SimpleNamespace(message="a suggestion")

        ctx = DummyCtx(THREE_VOICES_CFG, llm_fn=llm_fn)
        on_message(ctx, [_event()])

        assert len(ctx.sent) == 1
        assert "---" in ctx.sent[0][1]
        assert reword_count[0] == 3

    def test_subconscious_files_written(self):
        """After on_message, each voice's subconscious file is updated."""
        def llm_fn(messages, **kwargs):
            caller = kwargs.get("caller", "")
            user_content = messages[-1]["content"] if messages else ""
            if "vote" in caller:
                return SimpleNamespace(message='{"ranking": ["Sable"]}')
            if user_content.startswith("Rewrite"):
                return SimpleNamespace(message="reworded")
            if "subconscious" in user_content.lower():
                voice_name = caller.split("_")[-1]
                return SimpleNamespace(message=f"{voice_name} subconscious")
            return SimpleNamespace(message="suggestion")

        ctx = DummyCtx(THREE_VOICES_CFG, llm_fn=llm_fn)
        on_message(ctx, [_event()])

        for name in ("aria", "sable", "lune"):
            assert f"{name}_subconscious.md" in ctx._files

    def test_wrong_voice_count_does_not_send(self):
        """If not exactly 3 voices configured, no message sent."""
        bad_cfg = {
            "voices": [
                {"name": "Solo", "model": "m", "personality": "p"},
            ],
            "thinking_iterations": 1,
            "voice_space": "main",
        }
        ctx = DummyCtx(bad_cfg)
        on_message(ctx, [_event()])
        assert ctx.sent == []


class TestHecateHeartbeat:
    def test_heartbeat_writes_thinking_for_all_voices(self):
        """heartbeat writes {name}_thinking.md for each of the 3 voices."""
        think_count = [0]

        def llm_fn(messages, **kwargs):
            think_count[0] += 1
            caller = kwargs.get("caller", "")
            voice_name = caller.split("_")[-1] if "_" in caller else "Voice"
            return SimpleNamespace(message=f"Thoughts from {voice_name}")

        ctx = DummyCtx(THREE_VOICES_CFG, llm_fn=llm_fn)
        heartbeat(ctx)

        for name in ("aria", "sable", "lune"):
            key = f"{name}_thinking.md"
            assert key in ctx._files
            assert ctx._files[key]  # non-empty

    def test_heartbeat_two_iterations(self):
        """With thinking_iterations=2, LLM is called at least 6 times (3 voices × 2 rounds)."""
        calls = [0]

        def llm_fn(messages, **kwargs):
            calls[0] += 1
            return SimpleNamespace(message="thought")

        cfg = dict(THREE_VOICES_CFG)
        cfg["thinking_iterations"] = 2
        ctx = DummyCtx(cfg, llm_fn=llm_fn)
        heartbeat(ctx)

        # Iteration 0: 3 calls; Iteration 1: 3 calls → total 6
        assert calls[0] >= 6
