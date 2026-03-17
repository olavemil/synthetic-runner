"""Tests for the Consilium species."""

from types import SimpleNamespace

import pytest

from library.harness.adapters import Event
from library.tools.consilium import ConsiliumConfig, _parse_ghost_lines, load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIVE_PERSONAS_CFG = {
    "personas": [
        {"name": "Praxis", "model": "devstral", "personality": "Pragmatic"},
        {"name": "Lyric", "model": "devstral", "personality": "Poetic"},
        {"name": "Axiom", "model": "devstral", "personality": "Logical"},
        {"name": "Ember", "model": "devstral", "personality": "Passionate"},
        {"name": "Sage", "model": "devstral", "personality": "Reflective"},
    ],
    "ghost": {"model": "devstral"},
    "thinking_iterations": 2,
    "voice_space": "main",
}


class DummyCtx:
    def __init__(self, consilium_cfg: dict, llm_fn=None):
        self._consilium_cfg = consilium_cfg
        self._files: dict[str, str] = {
            "memory.md": "# Memory\n",
            "constitution.md": "# Constitution\n",
        }
        self.sent: list[tuple[str, str]] = []
        self._llm_fn = llm_fn
        self.instance_id = "test-consilium"

    def config(self, key: str):
        if key == "consilium":
            return self._consilium_cfg
        return None

    def read(self, path: str) -> str:
        return self._files.get(path, "")

    def write(self, path: str, content: str) -> None:
        self._files[path] = content

    def exists(self, path: str) -> bool:
        return path in self._files

    def list(self, prefix: str = "") -> list[str]:
        return sorted(p for p in self._files if p.startswith(prefix))

    def send(self, space: str, message: str, reply_to=None):
        self.sent.append((space, message))
        return "$event"

    def llm(self, messages, **kwargs):
        if self._llm_fn:
            raw = self._llm_fn(messages, **kwargs)
            if not hasattr(raw, "tool_calls"):
                raw.tool_calls = []
            return raw
        return SimpleNamespace(message="LLM response.", tool_calls=[])


def _event(body="hello", room="main", event_id="$1", timestamp=1):
    return Event(
        event_id=event_id, sender="@user:matrix.org",
        body=body, timestamp=timestamp, room=room,
    )


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConsiliumConfig:
    def test_load_config_parses_personas_and_ghost(self):
        ctx = DummyCtx(_FIVE_PERSONAS_CFG)
        cfg = load_config(ctx)
        assert len(cfg.personas) == 5
        assert cfg.personas[0].name == "Praxis"
        assert cfg.ghost.name == "Ghost"
        assert cfg.ghost.model == "devstral"
        assert cfg.thinking_iterations == 2
        assert cfg.voice_space == "main"

    def test_load_config_defaults_model_to_devstral(self):
        minimal = {
            "personas": [
                {"name": n} for n in ["A", "B", "C", "D", "E"]
            ],
        }
        ctx = DummyCtx(minimal)
        cfg = load_config(ctx)
        for p in cfg.personas:
            assert p.model == "devstral"
        assert cfg.ghost.model == "devstral"

    def test_load_config_empty_returns_defaults(self):
        ctx = DummyCtx({})
        cfg = load_config(ctx)
        assert cfg.personas == []
        assert cfg.ghost.model == "devstral"
        assert cfg.thinking_iterations == 2


# ---------------------------------------------------------------------------
# Ghost line parsing
# ---------------------------------------------------------------------------


class TestGhostLineParsing:
    def test_numbered_lines(self):
        raw = "1. First angle\n2. Second angle\n3. Third angle"
        assert _parse_ghost_lines(raw) == [
            "First angle", "Second angle", "Third angle",
        ]

    def test_unnumbered_lines(self):
        raw = "First angle\nSecond angle\nThird angle"
        assert _parse_ghost_lines(raw) == [
            "First angle", "Second angle", "Third angle",
        ]

    def test_bulleted_lines(self):
        raw = "- First angle\n- Second angle\n- Third angle"
        assert _parse_ghost_lines(raw) == [
            "First angle", "Second angle", "Third angle",
        ]

    def test_empty_input(self):
        assert _parse_ghost_lines("") == []
        assert _parse_ghost_lines("   \n  \n  ") == []

    def test_partial_output(self):
        raw = "1. Only one line"
        assert _parse_ghost_lines(raw) == ["Only one line"]


# ---------------------------------------------------------------------------
# on_message tests
# ---------------------------------------------------------------------------


class TestConsiliumOnMessage:
    def test_full_pipeline_sends_reply(self):
        """Full 8->4->1 pipeline sends exactly one reply and writes review files."""
        from library.species.consilium import on_message

        def llm_fn(messages, **kwargs):
            caller = kwargs.get("caller", "")
            if "Ghost" in caller:
                return SimpleNamespace(
                    message="1. angle one\n2. angle two\n3. angle three",
                )
            return SimpleNamespace(message="Draft response.")

        ctx = DummyCtx(_FIVE_PERSONAS_CFG, llm_fn=llm_fn)
        on_message(ctx, [_event()])

        assert len(ctx.sent) == 1
        assert ctx.sent[0][0] == "main"
        # Reviews should be written for all 5 personas
        for name in ["praxis", "lyric", "axiom", "ember", "sage"]:
            assert f"{name}_reviews.md" in ctx._files

    def test_empty_events_no_send(self):
        from library.species.consilium import on_message

        ctx = DummyCtx(_FIVE_PERSONAS_CFG)
        on_message(ctx, [])
        assert ctx.sent == []

    def test_wrong_persona_count_no_send(self):
        from library.species.consilium import on_message

        bad_cfg = {"personas": [{"name": "A"}], "ghost": {"model": "devstral"}}
        ctx = DummyCtx(bad_cfg)
        on_message(ctx, [_event()])
        assert ctx.sent == []


# ---------------------------------------------------------------------------
# heartbeat tests
# ---------------------------------------------------------------------------


class TestConsiliumHeartbeat:
    def test_heartbeat_writes_thinking_files(self):
        """Heartbeat should write all 5 {name}_thinking.md files."""
        from library.species.consilium import heartbeat

        def llm_fn(messages, **kwargs):
            if kwargs.get("tools"):
                return SimpleNamespace(
                    message="",
                    tool_calls=[
                        SimpleNamespace(
                            id="t1", name="done",
                            arguments={"summary": "done"},
                        ),
                    ],
                )
            return SimpleNamespace(message="thought content", tool_calls=[])

        ctx = DummyCtx(_FIVE_PERSONAS_CFG, llm_fn=llm_fn)
        heartbeat(ctx)

        for name in ["praxis", "lyric", "axiom", "ember", "sage"]:
            assert f"{name}_thinking.md" in ctx._files
