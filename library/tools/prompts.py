"""Prompt formatting helpers — prepare context blocks for LLM calls."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from library.harness.adapters import Event
    from library.harness.context import InstanceContext

logger = logging.getLogger(__name__)


STANDARD_MEMORY_FILES = [
    "thinking.md",
    "project.md",
    "sessions.md",
    "scratchpad.md",
    "sensitivity.md",
    "intentions.md",
    "subconscious.md",
]


def read_memory(ctx: InstanceContext) -> dict[str, str]:
    """Load standard memory files into a dict."""
    memory = {}
    for fname in STANDARD_MEMORY_FILES:
        content = ctx.read(fname)
        if content:
            memory[fname] = content
    return memory


def get_entity_id(ctx: InstanceContext) -> str:
    """Return the entity_id for the instance, or empty string if not configured."""
    try:
        return ctx.config("entity_id") or ""
    except (KeyError, AttributeError):
        return ""


def format_events(events: list[Event], self_entity_id: str = "") -> str:
    """Format events for prompt injection.

    When *self_entity_id* is provided, messages from that sender are
    labelled ``(you)`` so the LLM can distinguish its own prior replies
    from external messages.
    """
    if not events:
        return "(no new events)"
    lines = []
    for evt in events:
        if self_entity_id and evt.sender == self_entity_id:
            lines.append(f"[{evt.sender} (you)] {evt.body}")
        else:
            lines.append(f"[{evt.sender}] {evt.body}")
    return "\n".join(lines)


def format_relationships_block(ctx: InstanceContext, sender_ids: list[str] | None = None) -> str:
    """Load and format relationship files for relevant senders."""
    rel_files = ctx.list("relationships")
    if not rel_files:
        return ""

    blocks = []
    for path in rel_files:
        if sender_ids:
            if not any(sid in path for sid in sender_ids):
                continue
        content = ctx.read(path)
        if content:
            blocks.append(f"--- {path} ---\n{content}")

    return "\n\n".join(blocks)


def format_agent_context(ctx: InstanceContext, limit: int = 50) -> str:
    """Format the agents room context if available."""
    try:
        context = ctx.get_space_context("agents")
    except (KeyError, RuntimeError):
        return ""

    if not context:
        return ""

    parts = []
    if context.get("name"):
        parts.append(f"Room: {context['name']}")
    if context.get("topic"):
        parts.append(f"Topic: {context['topic']}")
    if context.get("members"):
        parts.append(f"Members: {', '.join(context['members'][:limit])}")
    return "\n".join(parts)


def format_intentions_block(memory: dict[str, str]) -> str:
    """Format the intentions block from memory."""
    content = memory.get("intentions.md", "")
    if not content:
        return ""
    return f"## Current Intentions\n{content}"


def format_subconscious_block(memory: dict[str, str]) -> str:
    """Format the subconscious block from memory."""
    content = memory.get("subconscious.md", "")
    if not content:
        return ""
    return f"## Subconscious Assessment\n{content}"


def format_rooms_context(ctx: InstanceContext) -> str:
    """Format all rooms into a prompt context block."""
    contexts = ctx.get_all_space_contexts()
    if not contexts:
        return ""
    parts = ["## Rooms"]
    for space_name, info in sorted(contexts.items()):
        room_name = info.get("name") or space_name
        topic = info.get("topic", "")
        members = info.get("members", [])
        line = f"- **{room_name}**"
        if topic:
            line += f": {topic}"
        line += f" ({len(members)} members)"
        parts.append(line)
    return "\n".join(parts)


def format_memory_context(memory: dict[str, str], exclude: list[str] | None = None) -> str:
    """Format all memory files into a single context block."""
    exclude = exclude or []
    parts = []
    for fname, content in sorted(memory.items()):
        if fname in exclude or not content:
            continue
        parts.append(f"## {fname}\n{content}")
    return "\n\n".join(parts)


def select_target_room(
    events: list[Event], default_space: str
) -> tuple[str, list[Event]]:
    """Select target room from events, preferring the latest room with events.

    Returns (target_room, scoped_events) where scoped_events are filtered to
    the target room when multiple rooms are present, or all events if none
    have a room set.
    """
    events_with_room = [evt for evt in events if evt.room]
    if events_with_room:
        target_event = max(events_with_room, key=lambda evt: evt.timestamp)
        target_room = target_event.room
        room_events = [evt for evt in events_with_room if evt.room == target_room]
        if len({evt.room for evt in events_with_room}) > 1:
            logger.info(
                "select_target_room: multiple rooms; selecting latest room=%s "
                "(room_events=%d total_events=%d)",
                target_room,
                len(room_events),
                len(events),
            )
        return target_room, room_events
    return default_space, events


# ---------------------------------------------------------------------------
# Recent received messages — persistent log cleared after heartbeat
# ---------------------------------------------------------------------------

_RECEIVED_MESSAGES_FILE = "received_messages.md"
_RECEIVED_MESSAGES_MAX_LINES = 100


def append_received_events(ctx: "InstanceContext", events: "list[Event]") -> None:
    """Append formatted incoming events to received_messages.md for later recall.
    
    Uses event_id for deduplication to prevent the same event from being logged multiple times.
    """
    if not events:
        return
    existing = ctx.read(_RECEIVED_MESSAGES_FILE) or ""
    lines = [l for l in existing.splitlines() if l.strip()]
    
    # Extract seen event IDs from existing lines (format: <!-- event_id:xyz -->)
    seen_event_ids = set()
    for line in lines:
        if line.startswith("<!-- event_id:") and line.endswith(" -->"):
            event_id = line[14:-4]  # Extract ID between markers
            seen_event_ids.add(event_id)
    
    new_lines = []
    added_count = 0
    for evt in events:
        event_id = getattr(evt, "event_id", None)
        if not event_id:
            # No event_id available - skip deduplication for this event
            pass
        elif event_id in seen_event_ids:
            # Already logged - skip
            continue
        else:
            # Mark as seen for subsequent events in this batch
            seen_event_ids.add(event_id)
        
        sender = getattr(evt, "sender", "?")
        room = getattr(evt, "room", "")
        body = getattr(evt, "body", "") or ""
        loc = f" [{room}]" if room else ""
        
        # Store event_id as HTML comment (invisible in markdown rendering)
        if event_id:
            new_lines.append(f"<!-- event_id:{event_id} -->")
        new_lines.append(f"- **{sender}**{loc}: {body.strip()[:300]}")
        added_count += 1
    
    if not new_lines:
        # All events were duplicates
        return
    
    lines.extend(new_lines)
    # Keep rolling window (count actual message lines, not comment lines)
    message_lines = [l for l in lines if not l.startswith("<!--")]
    if len(message_lines) > _RECEIVED_MESSAGES_MAX_LINES:
        # Need to trim - rebuild from tail keeping comments with their messages
        lines_to_keep = []
        message_count = 0
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i]
            if line.startswith("<!--"):
                # Keep comment if we're keeping the next message
                if message_count < _RECEIVED_MESSAGES_MAX_LINES:
                    lines_to_keep.insert(0, line)
            else:
                message_count += 1
                if message_count <= _RECEIVED_MESSAGES_MAX_LINES:
                    lines_to_keep.insert(0, line)
        lines = lines_to_keep
    
    ctx.write(_RECEIVED_MESSAGES_FILE, "\n".join(lines) + "\n")


def load_received_messages(ctx: "InstanceContext") -> str:
    """Return the accumulated received_messages.md content."""
    return ctx.read(_RECEIVED_MESSAGES_FILE) or ""


def clear_received_messages(ctx: "InstanceContext") -> None:
    """Clear received_messages.md (call at start of heartbeat after consuming)."""
    ctx.write(_RECEIVED_MESSAGES_FILE, "")
