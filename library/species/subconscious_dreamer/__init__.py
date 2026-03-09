"""Subconscious Dreamer species — three-phase thinking and three-phase response."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from library.species import Species, SpeciesManifest, EntryPoint
from library.tools.pipeline import run_pipeline, load_pipeline
from library.tools.prompts import format_events

if TYPE_CHECKING:
    from library.harness.adapters import Event
    from library.harness.context import InstanceContext

logger = logging.getLogger(__name__)

_SPECIES_DIR = Path(__file__).parent
_HEARTBEAT_STEPS = load_pipeline((_SPECIES_DIR / "heartbeat.yaml").read_text())["steps"]
_ON_MESSAGE_STEPS = load_pipeline((_SPECIES_DIR / "on_message.yaml").read_text())["steps"]

DEFAULT_FILES = {
    "thinking.md": "# Thinking\n",
    "dreams.md": "",
    "concerns.md": "",
}


def _build_context(*sections: tuple[str, str]) -> str:
    """Assemble labeled sections into a context block."""
    parts = []
    for label, text in sections:
        if text and text.strip():
            parts.append(f"## {label}\n\n{text.strip()}")
    return "\n\n".join(parts)


def _extract_dream_name(dreams: str) -> str:
    """Return the dream title from the first line of dreams.md, or empty string."""
    first_line = dreams.strip().split("\n")[0].strip() if dreams.strip() else ""
    if first_line.lower().startswith("dream of"):
        return first_line
    return ""


def _target_room(events: list[Event], ctx: InstanceContext) -> str:
    """Determine the logical space name to reply to."""
    spaces = ctx.list_spaces()
    if spaces:
        for evt in events:
            if evt.room in spaces:
                return evt.room
        return spaces[0]
    return "main"


def heartbeat(ctx: InstanceContext) -> None:
    """Three-phase thinking: active session → subconscious → dreaming."""
    thinking = ctx.read("thinking.md") or ""
    dreams = ctx.read("dreams.md") or ""
    concerns = ctx.read("concerns.md") or ""

    thinking_context = _build_context(
        ("Your Concerns", concerns),
        ("Your Dreams", dreams),
        ("Your Current Thoughts", thinking),
    ) or "This is your first thinking session."

    initial_state = {
        "thinking_context": thinking_context,
        "subconscious_sections": [
            ["thinking.md", "Your Thoughts"],
            ["dreams.md", "Your Dreams"],
        ],
        "dreaming_sections": [
            ["thinking.md", "Your Thoughts"],
            ["concerns.md", "Your Concerns"],
        ],
        "_species_dir": str(_SPECIES_DIR),
    }

    run_pipeline(ctx, _HEARTBEAT_STEPS, initial_state=initial_state)


def on_message(ctx: InstanceContext, events: list[Event]) -> None:
    """Three-phase response: intuition → worry → action → send."""
    if not events:
        return

    events_text = format_events(events)
    dreams = ctx.read("dreams.md") or ""
    concerns = ctx.read("concerns.md") or ""
    thinking = ctx.read("thinking.md") or ""
    dream_name = _extract_dream_name(dreams)

    initial_state = {
        "intuition_context": _build_context(
            ("Incoming Messages", events_text),
            ("Your Dreams", dreams),
        ),
        "worry_context": _build_context(
            ("Incoming Messages", events_text),
            ("Your Concerns", concerns),
        ),
        "action_context": _build_context(
            ("Incoming Messages", events_text),
            ("Your Current Dream", dream_name),
            ("Your Thoughts", thinking),
        ),
        "_species_dir": str(_SPECIES_DIR),
    }

    state = run_pipeline(ctx, _ON_MESSAGE_STEPS, events=events, initial_state=initial_state)
    response = state.get("response", "")
    if response and response.strip():
        ctx.send(_target_room(events, ctx), response.strip())


class SubconsciousDreamer(Species):
    def manifest(self) -> SpeciesManifest:
        return SpeciesManifest(
            species_id="subconscious_dreamer",
            entry_points=[
                EntryPoint(name="on_message", handler=on_message, trigger="message"),
                EntryPoint(name="heartbeat", handler=heartbeat, schedule="heartbeat"),
            ],
            default_files=DEFAULT_FILES,
        )
