"""Tool definitions and dispatch — all tools operate through ctx."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from symbiosis.harness.context import InstanceContext


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

    if name == "send_to_instance":
        ctx.send_to(arguments["target_id"], arguments["message"])
        return "Message sent to inbox.", False

    if name == "done":
        summary = arguments.get("summary", "Done.")
        return summary, True

    return f"Unknown tool: {name}", False
