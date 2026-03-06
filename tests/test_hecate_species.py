"""Tests for the Hecate species."""

from __future__ import annotations

import json
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
        call_counts = {"suggest": 0, "vote": 0, "reword": 0}

        def llm_fn(messages, **kwargs):
            caller = kwargs.get("caller", "")
            if "suggest" in caller:
                call_counts["suggest"] += 1
                return SimpleNamespace(
                    message=json.dumps({"text": f"suggestion {call_counts['suggest']}", "argument": "arg"})
                )
            if "vote" in caller:
                call_counts["vote"] += 1
                # Voices 0 and 1 vote for index 1 (Sable's suggestion); voice 2 votes for 0
                voice_name = caller.split("_")[-1]
                if voice_name in ("Aria", "Sable"):
                    return SimpleNamespace(message='{"choice": 1}')
                else:
                    return SimpleNamespace(message='{"choice": 0}')
            if "reword" in caller:
                call_counts["reword"] += 1
                return SimpleNamespace(message="Reworded message.")
            if "subconscious" in caller:
                return SimpleNamespace(message="Subconscious update.")
            return SimpleNamespace(message="")

        ctx = DummyCtx(THREE_VOICES_CFG, llm_fn=llm_fn)
        on_message(ctx, [_event()])

        assert len(ctx.sent) == 1
        assert ctx.sent[0][0] == "main"
        assert ctx.sent[0][1] == "Reworded message."
        # Exactly one reword called (for winner)
        assert call_counts["reword"] == 1

    def test_three_way_tie_joined_reply(self):
        """When all 3 get 1 vote each, all 3 rewrites are joined with separator."""
        reword_count = [0]

        def llm_fn(messages, **kwargs):
            caller = kwargs.get("caller", "")
            if "suggest" in caller:
                return SimpleNamespace(
                    message=json.dumps({"text": "a suggestion", "argument": "arg"})
                )
            if "vote" in caller:
                # Each voice votes for a different index (tie)
                voice_name = caller.split("_")[-1]
                if voice_name == "Aria":
                    return SimpleNamespace(message='{"choice": 1}')
                elif voice_name == "Sable":
                    return SimpleNamespace(message='{"choice": 2}')
                else:
                    return SimpleNamespace(message='{"choice": 0}')
            if "reword" in caller:
                reword_count[0] += 1
                return SimpleNamespace(message=f"Reword {reword_count[0]}")
            if "subconscious" in caller:
                return SimpleNamespace(message="sub.")
            return SimpleNamespace(message="")

        ctx = DummyCtx(THREE_VOICES_CFG, llm_fn=llm_fn)
        on_message(ctx, [_event()])

        assert len(ctx.sent) == 1
        assert "---" in ctx.sent[0][1]
        # All 3 rewrites joined
        assert reword_count[0] == 3

    def test_subconscious_files_written(self):
        """After on_message, each voice's subconscious file is updated."""
        def llm_fn(messages, **kwargs):
            caller = kwargs.get("caller", "")
            if "suggest" in caller:
                return SimpleNamespace(message=json.dumps({"text": "t", "argument": "a"}))
            if "vote" in caller:
                return SimpleNamespace(message='{"choice": 1}')
            if "reword" in caller:
                return SimpleNamespace(message="reworded")
            if "subconscious" in caller:
                voice_name = caller.split("_")[-1]
                return SimpleNamespace(message=f"{voice_name} subconscious")
            return SimpleNamespace(message="")

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
