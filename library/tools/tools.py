"""Tool definitions and dispatch — all tools operate through ctx."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from library.harness.context import InstanceContext

# Species description files live alongside the species Python modules.
_SPECIES_DIR = Path(__file__).parent.parent / "species"


def _load_species_description(species_id: str) -> str:
    parts = []

    # Check about.md inside package directory (e.g. species/neural_dreamer/about.md)
    path = _SPECIES_DIR / species_id / "about.md"
    if path.exists():
        parts.append(path.read_text())

    # Load any additional docs (e.g. species/neural_dreamer/docs/*.md)
    docs_dir = _SPECIES_DIR / species_id / "docs"
    if docs_dir.is_dir():
        for doc in sorted(docs_dir.glob("*.md")):
            parts.append(doc.read_text())

    return "\n\n---\n\n".join(parts) if parts else "(no species description available)"


def make_tools(ctx: InstanceContext, options: dict | None = None) -> list[dict]:
    """Build tool schemas in OpenAI function-calling format."""
    opts = options or {}
    tools = []

    tools.append({
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a memory file by path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to memory directory"},
                },
                "required": ["path"],
            },
        },
    })

    tools.append({
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a memory file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to memory directory"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    })

    tools.append({
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in the memory directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prefix": {"type": "string", "description": "Path prefix filter", "default": ""},
                },
            },
        },
    })

    if opts.get("messaging", True):
        tools.append({
            "type": "function",
            "function": {
                "name": "send_message",
                "description": "Send a message to a space.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "space": {"type": "string", "description": "Logical space name"},
                        "message": {"type": "string", "description": "Message content"},
                        "reply_to": {"type": "string", "description": "Event ID to reply to"},
                    },
                    "required": ["space", "message"],
                },
            },
        })

    if opts.get("rooms", True):
        tools.append({
            "type": "function",
            "function": {
                "name": "list_rooms",
                "description": "List all configured rooms with name, topic, and member count.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        })

    if opts.get("inter_instance", False):
        tools.append({
            "type": "function",
            "function": {
                "name": "send_to_instance",
                "description": "Send a message to another instance's mailbox.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_id": {"type": "string", "description": "Target instance ID"},
                        "message": {"type": "string", "description": "Message content"},
                    },
                    "required": ["target_id", "message"],
                },
            },
        })

    if opts.get("introspect", True):
        tools.append({
            "type": "function",
            "function": {
                "name": "introspect",
                "description": (
                    "Learn how this instance is configured: species description, "
                    "provider, model, spaces, and species-specific settings."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        })

    if opts.get("graph", False):
        from library.tools.graph import GRAPH_TOOL_SCHEMAS
        tools.extend(GRAPH_TOOL_SCHEMAS)

    if opts.get("activation_map", False):
        from library.tools.activation_map import MAP_TOOL_SCHEMAS
        tools.extend(MAP_TOOL_SCHEMAS)

    tools.append({
        "type": "function",
        "function": {
            "name": "done",
            "description": "Signal that the session is complete.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Brief summary of what was done"},
                },
            },
        },
    })

    return tools


def handle_tool(ctx: InstanceContext, name: str, arguments: dict) -> tuple[str, bool]:
    """Execute a tool call. Returns (result_text, is_done)."""
    if name == "read_file":
        content = ctx.read(arguments["path"])
        return content or "(empty file)", False

    if name == "write_file":
        ctx.write(arguments["path"], arguments["content"])
        return "File written.", False

    if name == "list_files":
        prefix = arguments.get("prefix", "")
        files = ctx.list(prefix)
        return "\n".join(files) if files else "(no files)", False

    if name == "send_message":
        event_id = ctx.send(
            arguments["space"],
            arguments["message"],
            reply_to=arguments.get("reply_to"),
        )
        return f"Sent (event_id: {event_id})", False

    if name == "list_rooms":
        contexts = ctx.get_all_space_contexts()
        if not contexts:
            return "(no rooms configured)", False
        lines = []
        for space_name, info in sorted(contexts.items()):
            room_name = info.get("name") or space_name
            topic = info.get("topic", "")
            members = info.get("members", [])
            parts = [f"- {room_name}"]
            if topic:
                parts.append(f"  Topic: {topic}")
            parts.append(f"  Members: {len(members)}")
            lines.append("\n".join(parts))
        return "\n".join(lines), False

    if name == "send_to_instance":
        ctx.send_to(arguments["target_id"], arguments["message"])
        return "Message sent to inbox.", False

    if name == "introspect":
        desc = _load_species_description(ctx.species_id)
        summary = ctx.config_summary()
        config_lines = []
        for k, v in summary.items():
            if isinstance(v, (dict, list)):
                config_lines.append(f"- **{k}**: {json.dumps(v, ensure_ascii=False)}")
            else:
                config_lines.append(f"- **{k}**: {v}")
        config_block = "\n".join(config_lines)
        result = f"{desc}\n\n---\n\n## Instance Config\n\n{config_block}"
        return result, False

    if name == "done":
        summary = arguments.get("summary", "Done.")
        return summary, True

    if name.startswith("graph_"):
        from library.tools.graph import handle_graph_tool
        return handle_graph_tool(ctx, name, arguments), False

    if name.startswith("map_"):
        from library.tools.activation_map import handle_map_tool
        return handle_map_tool(ctx, name, arguments), False

    return f"Unknown tool: {name}", False
