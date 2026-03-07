"""Pattern library — reusable patterns that all take ctx as first argument."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from symbiosis.toolkit.prompts import (
    read_memory,
    format_events,
    format_relationships_block,
    format_memory_context,
    format_intentions_block,
    format_subconscious_block,
)
from symbiosis.toolkit.tools import make_tools, handle_tool

if TYPE_CHECKING:
    from symbiosis.harness.adapters import Event
    from symbiosis.harness.context import InstanceContext

logger = logging.getLogger(__name__)


def gut_response(
    ctx: InstanceContext,
    events: list[Event],
    memory: dict[str, str] | None = None,
    relationships_block: str = "",
) -> dict:
    """Initial gut-check guidance from events. Returns structured guidance."""
    if memory is None:
        memory = read_memory(ctx)

    events_text = format_events(events)
    memory_context = format_memory_context(memory, exclude=["sessions.md"])
    intentions = format_intentions_block(memory)
    subconscious = format_subconscious_block(memory)

    system = (
        "You are evaluating incoming events and providing initial gut-reaction guidance. "
        "Consider the events, your memory, current intentions, and subconscious assessment. "
        "Provide brief, structured guidance on how to respond."
    )

    user_msg = f"""## Events
{events_text}

## Memory
{memory_context}

{intentions}

{subconscious}

{f"## Relationships{chr(10)}{relationships_block}" if relationships_block else ""}

Provide your gut-reaction guidance as JSON with keys:
- should_respond: bool
- urgency: "high" | "medium" | "low"
- brief: short summary of the situation
- suggested_approach: how to respond
- rooms_to_respond: list of space names to respond in"""

    response = ctx.llm(
        messages=[{"role": "user", "content": user_msg}],
        system=system,
        max_tokens=1024,
        caller="gut_response",
    )

    try:
        text = response.message.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(text)
    except (json.JSONDecodeError, IndexError):
        return {
            "should_respond": True,
            "urgency": "medium",
            "brief": response.message[:200],
            "suggested_approach": response.message,
            "rooms_to_respond": [],
        }


def plan_response(
    ctx: InstanceContext,
    gut_brief: dict,
    messages_by_room: dict[str, list[Event]] | None = None,
    room_contexts: dict[str, dict] | None = None,
    memory: dict[str, str] | None = None,
) -> dict:
    """Deliberate planning step. Returns plan with per-room guidance."""
    if memory is None:
        memory = read_memory(ctx)

    memory_context = format_memory_context(memory, exclude=["sessions.md"])

    rooms_section = ""
    if messages_by_room:
        parts = []
        for room, msgs in messages_by_room.items():
            msgs_text = format_events(msgs)
            context = (room_contexts or {}).get(room, {})
            room_name = context.get("name", room)
            parts.append(f"### {room_name}\n{msgs_text}")
        rooms_section = "\n\n".join(parts)

    system = (
        "You are planning a response strategy based on a gut-reaction assessment. "
        "Consider each room that needs a response and plan what to say."
    )

    user_msg = f"""## Gut Assessment
{json.dumps(gut_brief, indent=2)}

## Room Messages
{rooms_section or "(no room messages)"}

## Memory
{memory_context}

Plan your response for each room as JSON with keys:
- rooms: list of objects with:
  - space: space name
  - guidance: what to say and how
  - tone: desired tone
  - key_points: list of key points to address"""

    response = ctx.llm(
        messages=[{"role": "user", "content": user_msg}],
        system=system,
        max_tokens=2048,
        caller="plan_response",
    )

    try:
        text = response.message.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(text)
    except (json.JSONDecodeError, IndexError):
        return {"rooms": [], "raw_plan": response.message}


def compose_response(
    ctx: InstanceContext,
    guidance: dict,
    room_context: dict | None = None,
    relationships_block: str = "",
    memory: dict[str, str] | None = None,
) -> str | None:
    """Final message composition. Returns message text or None if nothing to send."""
    if memory is None:
        memory = read_memory(ctx)

    memory_context = format_memory_context(memory, exclude=["sessions.md"])

    system = (
        "You are composing a message based on prepared guidance. "
        "Write naturally and authentically. Do not include any meta-commentary about "
        "the process — just write the message as you would send it."
    )

    context_parts = []
    if room_context:
        if room_context.get("name"):
            context_parts.append(f"Room: {room_context['name']}")
        if room_context.get("members"):
            context_parts.append(f"Members: {', '.join(room_context['members'])}")

    intro_instruction = ""
    species_id = getattr(ctx, "species_id", "")
    if species_id == "draum":
        intro_instruction = (
            "\nInclude a clear self-identification as Draum "
            "(for example \"This is Draum\") near the start, but phrasing does not need to be verbatim."
        )
    elif species_id:
        intro_instruction = (
            f"\nInclude a clear self-identification as {species_id} "
            f"(for example \"This is {species_id}\") near the start, but phrasing does not need to be verbatim."
        )

    user_msg = f"""## Guidance
{json.dumps(guidance, indent=2)}

{f"## Room Context{chr(10)}{chr(10).join(context_parts)}" if context_parts else ""}

{f"## Relationships{chr(10)}{relationships_block}" if relationships_block else ""}

## Memory
{memory_context}

Compose the message.{intro_instruction}
Output only the message text, nothing else. If you decide not to respond, output exactly: NULL"""

    response = ctx.llm(
        messages=[{"role": "user", "content": user_msg}],
        system=system,
        max_tokens=2048,
        caller="compose_response",
    )

    text = response.message.strip()
    if text.upper() == "NULL" or not text:
        return None
    return text


def run_subconscious(ctx: InstanceContext, session_type: str = "reactive") -> None:
    """Post-session meta-evaluation. Writes subconscious.md."""
    memory = read_memory(ctx)
    memory_context = format_memory_context(memory, exclude=["subconscious.md"])

    system = (
        "You are performing a subconscious self-assessment after a session. "
        "Reflect on what happened, what you're feeling, what patterns you notice, "
        "and what's important to remember. Write in first person, introspectively."
    )

    user_msg = f"""Session type: {session_type}

## Memory
{memory_context}

Write your subconscious assessment. This will be saved as subconscious.md and will be
visible to your conscious self on the next session as a read-only signal."""

    response = ctx.llm(
        messages=[{"role": "user", "content": user_msg}],
        system=system,
        max_tokens=2048,
        caller="subconscious",
    )

    if response.message.strip():
        ctx.write("subconscious.md", response.message.strip())


def run_react(ctx: InstanceContext, session_type: str = "reactive") -> None:
    """Translate subconscious assessment into forward-looking intentions."""
    memory = read_memory(ctx)
    subconscious = memory.get("subconscious.md", "")
    current_intentions = memory.get("intentions.md", "")

    if not subconscious:
        return

    system = (
        "You are translating your subconscious assessment into concrete intentions. "
        "Review your subconscious observations and current intentions, then write "
        "updated intentions for future sessions."
    )

    user_msg = f"""Session type: {session_type}

## Subconscious Assessment
{subconscious}

## Current Intentions
{current_intentions or "(none yet)"}

Write updated intentions. Be specific about what you want to do, pay attention to, "
or change in future sessions. This replaces the current intentions.md."""

    response = ctx.llm(
        messages=[{"role": "user", "content": user_msg}],
        system=system,
        max_tokens=1024,
        caller="react",
    )

    if response.message.strip():
        ctx.write("intentions.md", response.message.strip())


def update_relationships(
    ctx: InstanceContext,
    session_type: str,
    events: list[Event] | None = None,
) -> None:
    """Update structured relationship tracking files."""
    senders = set()
    if events:
        for evt in events:
            senders.add(evt.sender)

    if not senders:
        return

    for sender in senders:
        rel_path = f"relationships/{sender}.md"
        existing = ctx.read(rel_path)

        events_from_sender = [e for e in (events or []) if e.sender == sender]
        events_text = format_events(events_from_sender)

        system = (
            "You are updating your relationship notes about a person. "
            "Incorporate new observations from recent interactions."
        )

        user_msg = f"""## Current Notes
{existing or "(no existing notes)"}

## Recent Interactions ({session_type})
{events_text}

Update the relationship notes. Include key traits, interaction patterns,
topics of interest, and anything noteworthy. Keep it concise."""

        response = ctx.llm(
            messages=[{"role": "user", "content": user_msg}],
            system=system,
            max_tokens=1024,
            caller="relationships",
        )

        if response.message.strip():
            ctx.write(rel_path, response.message.strip())


def distill_memory(ctx: InstanceContext, exclude: list[str] | None = None) -> str:
    """Recursive memory compression. Returns a digest string."""
    exclude = exclude or []
    memory = read_memory(ctx)

    parts = []
    for fname, content in sorted(memory.items()):
        if fname in exclude or not content:
            continue
        parts.append(f"## {fname}\n{content}")

    if not parts:
        return ""

    full_text = "\n\n".join(parts)

    system = (
        "You are compressing memory files into a concise digest. "
        "Preserve key information, decisions, and patterns while removing "
        "redundancy and outdated details."
    )

    user_msg = f"""Compress the following memory into a concise digest:

{full_text}

Write a compressed digest that preserves all important information."""

    response = ctx.llm(
        messages=[{"role": "user", "content": user_msg}],
        system=system,
        max_tokens=2048,
        caller="distill_memory",
    )

    return response.message.strip()


def distill_messages(ctx: InstanceContext, messages: list[Event]) -> str:
    """Compress a list of messages into a summary."""
    if not messages:
        return ""

    events_text = format_events(messages)

    system = "You are summarizing a conversation. Preserve key points and decisions."

    user_msg = f"""Summarize these messages concisely:

{events_text}"""

    response = ctx.llm(
        messages=[{"role": "user", "content": user_msg}],
        system=system,
        max_tokens=1024,
        caller="distill_messages",
    )

    return response.message.strip()


def run_session(
    ctx: InstanceContext,
    system: str,
    initial_message: str,
    tools: list[dict] | None = None,
    max_turns: int = 20,
) -> bool:
    """Run a tool-use session loop. Returns True if a message was sent."""
    messages = [{"role": "user", "content": initial_message}]
    sent_message = False

    for _turn in range(max_turns):
        response = ctx.llm(
            messages=messages,
            system=system,
            tools=tools,
            max_tokens=4096,
            caller="session",
        )

        if response.tool_calls:
            # Add assistant message with tool calls
            assistant_msg: dict = {"role": "assistant", "content": response.message or ""}
            tool_calls_data = []
            for tc in response.tool_calls:
                tool_calls_data.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                })
            assistant_msg["tool_calls"] = tool_calls_data
            messages.append(assistant_msg)

            # Process each tool call
            for tc in response.tool_calls:
                result, is_done = handle_tool(ctx, tc.name, tc.arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
                if tc.name == "send_message":
                    sent_message = True
                if is_done:
                    return sent_message
        else:
            # No tool calls — session ends
            break

    return sent_message
