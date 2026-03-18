"""Tests for Draum species behavior."""

from library.harness.adapters import Event
from library.species import draum as draum_mod


class DummyCtx:
    def __init__(self, instance_id: str, send_impl):
        self.instance_id = instance_id
        self._send_impl = send_impl

    def send(self, space: str, message: str, reply_to=None):  # noqa: ANN001
        return self._send_impl(space, message)

    def get_space_context(self, space: str) -> dict:  # noqa: ARG002
        return {}

    def list_spaces(self) -> list[str]:
        return ["main"]


def test_on_message_falls_back_to_main_for_unmapped_space(monkeypatch):
    monkeypatch.setattr(draum_mod, "read_memory", lambda ctx: {})
    monkeypatch.setattr(draum_mod, "format_relationships_block", lambda ctx, sender_ids: "")
    monkeypatch.setattr(
        draum_mod,
        "gut_response",
        lambda ctx, events, memory=None, relationships_block="": {  # noqa: ARG005
            "should_respond": True,
            "rooms_to_respond": [],
        },
    )
    monkeypatch.setattr(
        draum_mod,
        "plan_response",
        lambda ctx, gut, messages_by_room=None, room_contexts=None, memory=None: {  # noqa: ARG005
            "rooms": [{"space": "matrix.org", "guidance": "respond"}]
        },
    )
    monkeypatch.setattr(draum_mod, "compose_response", lambda ctx, guidance, **kwargs: "hello")
    monkeypatch.setattr(draum_mod, "run_subconscious", lambda ctx, session_type="reactive": None)
    monkeypatch.setattr(draum_mod, "run_react", lambda ctx, session_type="reactive": None)
    monkeypatch.setattr(draum_mod, "update_relationships", lambda ctx, session_type, events=None: None)
    monkeypatch.setattr(draum_mod, "run_entity_mapping_phase", lambda ctx, events=None: None)

    calls: list[str] = []

    def send_impl(space: str, message: str) -> str:
        calls.append(space)
        if space == "matrix.org":
            raise KeyError("unmapped")
        return "$event"

    ctx = DummyCtx("thrivemind", send_impl)
    events = [Event(event_id="1", sender="@u:matrix.org", body="hi", timestamp=1, room="main")]

    draum_mod.on_message(ctx, events)
    assert calls == ["main"]
