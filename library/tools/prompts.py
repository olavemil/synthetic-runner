"""Prompt formatting helpers — prepare context blocks for LLM calls."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from library.harness.adapters import Event
    from library.harness.context import InstanceContext


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
