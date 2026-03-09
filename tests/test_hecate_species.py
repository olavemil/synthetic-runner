"""Tests for the Hecate species."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from library.harness.adapters import Event
from library.species.hecate import on_message, heartbeat


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

    def list(self, prefix: str = "") -> list[str]:
        return sorted(path for path in self._files if path.startswith(prefix))

    def send(self, space: str, message: str, reply_to=None):
        self.sent.append((space, message))
        return "$event"

    def llm(self, messages, **kwargs):
        if self._llm_fn:
            return self._llm_fn(messages, **kwargs)
        return SimpleNamespace(message="LLM response.")


def _event(body="hello", room="main", event_id="$1", timestamp=1):
    return Event(event_id=event_id, sender="@u:h.org", body=body, timestamp=timestamp, room=room)


class TestHecateOnMessage:
    def test_random_voice_composes_joined_suggestions(self):
        """All 3 voices suggest; random voice composes the joined draft."""
        composed_calls = []

        def llm_fn(messages, **kwargs):
            caller = kwargs.get("caller", "")
            user_content = messages[-1]["content"] if messages else ""
            if "Write a brief subconscious note" in user_content:
                return SimpleNamespace(message="Subconscious update.")
            if "Write the final reply in your own voice." in user_content:
                composed_calls.append((caller, user_content))
                return SimpleNamespace(message="Composed final message.")
            if "propose exactly one brief way to answer." in user_content:
                voice_name = caller.split("_")[-1]
                return SimpleNamespace(message=f"{voice_name} suggestion.")
            return SimpleNamespace(message="fallback")

        ctx = DummyCtx(THREE_VOICES_CFG, llm_fn=llm_fn)
        with patch("library.species.hecate.random.shuffle", side_effect=lambda seq: seq.reverse()):
            with patch("library.species.hecate.random.choice", side_effect=lambda seq: seq[1]):
                on_message(ctx, [_event()])

        assert len(ctx.sent) == 1
        assert ctx.sent[0][0] == "main"
        assert ctx.sent[0][1] == "Composed final message."
        assert len(composed_calls) == 1
        compose_caller, compose_prompt = composed_calls[0]
        assert compose_caller == "generate_Sable"
        assert "(Aria) Aria suggestion." in compose_prompt
        assert "(Sable) Sable suggestion." in compose_prompt
        assert "(Lune) Lune suggestion." in compose_prompt
        assert "and you are Bold" in compose_prompt
        assert "1 to 4 paragraphs" in compose_prompt

    def test_on_message_routes_reply_to_latest_room_and_uses_room_scoped_context(self):
        composed_calls = []

        def llm_fn(messages, **kwargs):
            user_content = messages[-1]["content"] if messages else ""
            if "Write a brief subconscious note" in user_content:
                return SimpleNamespace(message="sub.")
            if "Write the final reply in your own voice." in user_content:
                composed_calls.append(user_content)
                return SimpleNamespace(message="Composed for ops.")
            if "propose exactly one brief way to answer." in user_content:
                return SimpleNamespace(message="One suggestion.")
            return SimpleNamespace(message="fallback")

        ctx = DummyCtx(THREE_VOICES_CFG, llm_fn=llm_fn)
        events = [
            _event(body="main message", room="main", event_id="$1", timestamp=1),
            _event(body="ops message", room="ops", event_id="$2", timestamp=2),
        ]
        with patch("library.species.hecate.random.choice", side_effect=lambda seq: seq[0]):
            on_message(ctx, events)

        assert len(ctx.sent) == 1
        assert ctx.sent[0] == ("ops", "Composed for ops.")
        assert len(composed_calls) == 1
        prompt = composed_calls[0]
        assert "Conversation in room 'ops'" in prompt
        assert "ops message" in prompt
        assert "main message" not in prompt

    def test_suggestions_are_clipped_to_one_sentence_in_joined_draft(self):
        composed_prompts = []

        def llm_fn(messages, **kwargs):
            user_content = messages[-1]["content"] if messages else ""
            if "Write a brief subconscious note" in user_content:
                return SimpleNamespace(message="sub.")
            if "Write the final reply in your own voice." in user_content:
                composed_prompts.append(user_content)
                return SimpleNamespace(message="Composed final message.")
            if "propose exactly one brief way to answer." in user_content:
                return SimpleNamespace(message="First sentence. Second sentence should be removed.")
            return SimpleNamespace(message="fallback")

        ctx = DummyCtx(THREE_VOICES_CFG, llm_fn=llm_fn)
        with patch("library.species.hecate.random.choice", side_effect=lambda seq: seq[0]):
            on_message(ctx, [_event()])
        assert len(ctx.sent) == 1
        assert ctx.sent[0][1] == "Composed final message."
        assert composed_prompts
        assert "First sentence." in composed_prompts[0]
        assert "Second sentence should be removed." not in composed_prompts[0]

    def test_subconscious_files_written(self):
        """After on_message, each voice's subconscious file is updated."""
        def llm_fn(messages, **kwargs):
            caller = kwargs.get("caller", "")
            user_content = messages[-1]["content"] if messages else ""
            if "Write a brief subconscious note" in user_content:
                voice_name = caller.split("_")[-1]
                return SimpleNamespace(message=f"{voice_name} subconscious")
            if "Write the final reply in your own voice." in user_content:
                return SimpleNamespace(message="Composed final message.")
            if "propose exactly one brief way to answer." in user_content:
                return SimpleNamespace(message="suggestion")
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

    def test_heartbeat_second_iteration_uses_own_latest_thought_and_snapshot(self):
        cfg = dict(THREE_VOICES_CFG)
        cfg["thinking_iterations"] = 2
        captured: list[tuple[str, str, str]] = []
        per_voice_count: dict[str, int] = {"Aria": 0, "Sable": 0, "Lune": 0}

        def llm_fn(messages, **kwargs):
            caller = kwargs.get("caller", "")
            voice_name = caller.split("_")[-1] if "_" in caller else "Voice"
            system = kwargs.get("system", "")
            user_content = messages[-1]["content"] if messages else ""
            captured.append((voice_name, system, user_content))
            idx = per_voice_count.get(voice_name, 0)
            per_voice_count[voice_name] = idx + 1
            return SimpleNamespace(message=f"{voice_name} thought round {idx}")

        ctx = DummyCtx(cfg, llm_fn=llm_fn)
        heartbeat(ctx)

        aria_calls = [entry for entry in captured if entry[0] == "Aria"]
        assert len(aria_calls) >= 2
        _voice, second_system, _second_user = aria_calls[1]
        assert "## Your Previous Thoughts\nAria thought round 0" in second_system
        assert "## Memory Directory Snapshot (before iteration)" in second_system
        assert "### aria_thinking.md" in second_system
