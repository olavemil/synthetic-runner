"""Draum species — persistent memory agent with gut→plan→compose pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from symbiosis.species import Species, SpeciesManifest, EntryPoint
from symbiosis.toolkit.patterns import (
    gut_response,
    plan_response,
    compose_response,
    run_subconscious,
    run_react,
    update_relationships,
    distill_memory,
    run_session,
)
from symbiosis.toolkit.prompts import read_memory, format_relationships_block, format_events
from symbiosis.toolkit.tools import make_tools

if TYPE_CHECKING:
    from symbiosis.harness.adapters import Event
    from symbiosis.harness.context import InstanceContext


DEFAULT_FILES = {
    "thinking.md": "# Thinking\n",
    "project.md": "# Project\n",
    "sessions.md": "# Sessions\n",
    "scratchpad.md": "# Scratchpad\n",
    "sensitivity.md": "# Sensitivity\n",
    "intentions.md": "",
    "subconscious.md": "",
}


def on_message(ctx: InstanceContext, events: list[Event]) -> None:
    """Reactive pipeline: gut → plan → compose → send, then post-session processes."""
    if not events:
        return

    memory = read_memory(ctx)
    sender_ids = list({evt.sender for evt in events})
    relationships_block = format_relationships_block(ctx, sender_ids)

    # 1. Gut response
    gut = gut_response(ctx, events, memory=memory, relationships_block=relationships_block)

    if not gut.get("should_respond", True):
        # Still run post-session processes
        run_subconscious(ctx, "reactive")
        run_react(ctx, "reactive")
        update_relationships(ctx, "reactive", events)
        return

    # 2. Plan response
    rooms_to_respond = gut.get("rooms_to_respond", [])
    messages_by_room: dict[str, list[Event]] = {}
    room_contexts: dict[str, dict] = {}

    for room in rooms_to_respond:
        room_events = [e for e in events if e.room == room or not e.room]
        if room_events:
            messages_by_room[room] = room_events
        try:
            room_contexts[room] = ctx.get_space_context(room)
        except (KeyError, RuntimeError):
            pass

    plan = plan_response(
        ctx, gut,
        messages_by_room=messages_by_room,
        room_contexts=room_contexts,
        memory=memory,
    )

    # 3. Compose and send for each room
    for room_plan in plan.get("rooms", []):
        space = room_plan.get("space", "main")
        room_context = room_contexts.get(space)

        message = compose_response(
            ctx, room_plan,
            room_context=room_context,
            relationships_block=relationships_block,
            memory=memory,
        )
        if message:
            ctx.send(space, message)

    # 4. Post-session processes
    run_subconscious(ctx, "reactive")
    run_react(ctx, "reactive")
    update_relationships(ctx, "reactive", events)


def heartbeat(ctx: InstanceContext) -> None:
    """Scheduled heartbeat: memory distillation and spontaneous thinking session."""
    memory = read_memory(ctx)

    # Distill memory
    digest = distill_memory(ctx, exclude=["sessions.md"])

    # Run a thinking session
    tools = make_tools(ctx, {"messaging": True, "inter_instance": False})
    system = (
        "You are in a spontaneous thinking session. Reflect on your recent experiences, "
        "review your memory, and think about what matters to you. You may write to your "
        "memory files or send a message if you have something meaningful to share."
    )

    initial_message = f"""## Memory Digest
{digest}

## Current Thinking
{memory.get('thinking.md', '')}

## Intentions
{memory.get('intentions.md', '')}

## Subconscious
{memory.get('subconscious.md', '')}

Take some time to think. Use tools to read/write memory or send messages if appropriate."""

    run_session(ctx, system, initial_message, tools=tools)

    # Post-session processes
    run_subconscious(ctx, "heartbeat")
    run_react(ctx, "heartbeat")


class DraumSpecies(Species):
    def manifest(self) -> SpeciesManifest:
        return SpeciesManifest(
            species_id="draum",
            entry_points=[
                EntryPoint(
                    name="on_message",
                    handler=on_message,
                    trigger="message",
                ),
                EntryPoint(
                    name="heartbeat",
                    handler=heartbeat,
                    schedule="0 * * * *",
                ),
            ],
            tools=[],
            default_files=DEFAULT_FILES,
            spawn=self._spawn,
        )

    def _spawn(self, ctx: InstanceContext) -> None:
        """Initialize a new Draum instance with default files."""
        for path, content in DEFAULT_FILES.items():
            if not ctx.exists(path):
                ctx.write(path, content)
