"""Knowledge organization tools for structured memory management.

Provides tools for managing a category/topic hierarchy under knowledge/
in instance storage, plus archiving old thinking entries.

Tools are exposed via ORGANIZE_TOOL_SCHEMAS and dispatched via handle_organize_tool().
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from library.harness.context import InstanceContext

logger = logging.getLogger(__name__)

# Default categories seeded on first organize run

DEFAULT_CATEGORIES = ["spaces", "entities", "concepts", "events"]

ORGANIZE_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "organize_list_categories",
            "description": "List all knowledge categories with topic counts.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "organize_create_category",
            "description": "Create a new knowledge category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Category name (slug format, e.g. 'concepts')"},
                    "description": {"type": "string", "description": "What this category is for"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "organize_remove_category",
            "description": "Remove a category. Optionally merge its topics into another category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Category to remove"},
                    "merge_into": {"type": "string", "description": "If set, move all topics to this category instead of deleting"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "organize_list_topics",
            "description": "List topics within a category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Category name"},
                },
                "required": ["category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "organize_read_topic",
            "description": "Read the content of a topic file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Category name"},
                    "topic": {"type": "string", "description": "Topic name (without .md extension)"},
                },
                "required": ["category", "topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "organize_write_topic",
            "description": "Create or update a topic. Creates the category if it doesn't exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Category name"},
                    "topic": {"type": "string", "description": "Topic name (without .md extension)"},
                    "content": {"type": "string", "description": "Topic content"},
                },
                "required": ["category", "topic", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "organize_remove_topic",
            "description": "Remove a topic file from a category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Category name"},
                    "topic": {"type": "string", "description": "Topic name (without .md extension)"},
                },
                "required": ["category", "topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "organize_archive_thoughts",
            "description": "Move content from thinking.md to the archive.",
            "parameters": {
                "type": "object",
                "properties": {
                    "before_marker": {"type": "string", "description": "Archive everything above this text marker in thinking.md"},
                    "label": {"type": "string", "description": "Label for the archive file (default: timestamp)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "organize_list_archives",
            "description": "List archive entries (previously archived thinking sessions).",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "organize_read_archive",
            "description": "Read the content of an archive entry.",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Archive label (from organize_list_archives)"},
                },
                "required": ["label"],
            },
        },
    },
]


def _ensure_category(ctx: InstanceContext, name: str, description: str = "") -> None:
    """Ensure a category directory exists with a _meta.md file."""
    meta_path = f"knowledge/{name}/_meta.md"
    if not ctx.exists(meta_path):
        ctx.write(meta_path, f"# {name}\n\n{description}\n" if description else f"# {name}\n")


def _category_exists(ctx: InstanceContext, name: str) -> bool:
    """Check if a category exists."""
    return ctx.exists(f"knowledge/{name}/_meta.md")


def _list_category_names(ctx: InstanceContext) -> list[str]:
    """List all category names by scanning knowledge/ directory."""
    all_files = ctx.list("knowledge/")
    categories = set()
    for f in all_files:
        # Paths are relative to instance root: "knowledge/concepts/topic.md"
        parts = f.split("/")
        if len(parts) >= 2:
            categories.add(parts[1])
    return sorted(categories)


def _list_topics_in_category(ctx: InstanceContext, category: str) -> list[str]:
    """List topic names (without .md) in a category."""
    prefix = f"knowledge/{category}/"
    all_files = ctx.list(prefix)
    topics = []
    for f in all_files:
        name = f.split("/")[-1]
        if name.endswith(".md") and name != "_meta.md":
            topics.append(name[:-3])
    return sorted(topics)


def count_all_topics(ctx: InstanceContext) -> int:
    """Count total topics across all categories."""
    total = 0
    for cat in _list_category_names(ctx):
        total += len(_list_topics_in_category(ctx, cat))
    return total


def _list_archive_labels(ctx: InstanceContext) -> list[str]:
    """List archive entry labels (filenames without .md)."""
    all_files = ctx.list("archive/")
    labels = []
    for f in all_files:
        name = f.split("/")[-1]
        if name.endswith(".md"):
            labels.append(name[:-3])
    return sorted(labels)


def handle_organize_tool(ctx: InstanceContext, name: str, arguments: dict) -> str:
    """Dispatch an organize tool call. Returns result text."""
    logger.info("organize tool: %s %s", name, {k: v[:40] + "..." if isinstance(v, str) and len(v) > 40 else v for k, v in arguments.items()})

    if name == "organize_list_categories":
        categories = _list_category_names(ctx)
        if not categories:
            for cat in DEFAULT_CATEGORIES:
                _ensure_category(ctx, cat)
            categories = DEFAULT_CATEGORIES

        lines = []
        for cat in categories:
            topics = _list_topics_in_category(ctx, cat)
            lines.append(f"- {cat} ({len(topics)} topics)")
        result = "\n".join(lines) if lines else "No categories found."
        logger.info("organize_list_categories: %d categories", len(categories))
        return result

    elif name == "organize_create_category":
        cat_name = arguments["name"]
        desc = arguments.get("description", "")
        if _category_exists(ctx, cat_name):
            logger.info("organize_create_category: '%s' already exists", cat_name)
            return f"Category '{cat_name}' already exists."
        _ensure_category(ctx, cat_name, desc)
        logger.info("organize_create_category: created '%s'", cat_name)
        return f"Created category '{cat_name}'."

    elif name == "organize_remove_category":
        cat_name = arguments["name"]
        merge_into = arguments.get("merge_into")

        if not _category_exists(ctx, cat_name):
            return f"Category '{cat_name}' does not exist."

        topics = _list_topics_in_category(ctx, cat_name)

        if merge_into:
            _ensure_category(ctx, merge_into)
            for topic in topics:
                content = ctx.read(f"knowledge/{cat_name}/{topic}.md") or ""
                ctx.write(f"knowledge/{merge_into}/{topic}.md", content)

        # Clear all files in the category (no ctx.delete, so write empty)
        prefix = f"knowledge/{cat_name}/"
        for f in ctx.list(prefix):
            ctx.write(f, "")

        if merge_into:
            logger.info("organize_remove_category: merged %d topics from '%s' into '%s'", len(topics), cat_name, merge_into)
            return f"Merged {len(topics)} topics from '{cat_name}' into '{merge_into}' and removed '{cat_name}'."
        logger.info("organize_remove_category: removed '%s' (%d topics)", cat_name, len(topics))
        return f"Removed category '{cat_name}' ({len(topics)} topics deleted)."

    elif name == "organize_list_topics":
        category = arguments["category"]
        if not _category_exists(ctx, category):
            return f"Category '{category}' does not exist."
        topics = _list_topics_in_category(ctx, category)
        if not topics:
            return f"No topics in '{category}'."
        lines = []
        for t in topics:
            content = ctx.read(f"knowledge/{category}/{t}.md") or ""
            preview = content[:100].replace("\n", " ").strip()
            lines.append(f"- {t}: {preview}{'...' if len(content) > 100 else ''}")
        logger.info("organize_list_topics: %d topics in '%s'", len(topics), category)
        return "\n".join(lines)

    elif name == "organize_read_topic":
        category = arguments["category"]
        topic = arguments["topic"]
        content = ctx.read(f"knowledge/{category}/{topic}.md")
        if not content:
            logger.info("organize_read_topic: '%s/%s' not found", category, topic)
            return f"Topic '{topic}' not found in '{category}'."
        logger.info("organize_read_topic: '%s/%s' (%d chars)", category, topic, len(content))
        return content

    elif name == "organize_write_topic":
        category = arguments["category"]
        topic = arguments["topic"]
        content = arguments["content"]
        _ensure_category(ctx, category)
        path = f"knowledge/{category}/{topic}.md"
        is_new = not ctx.exists(path)
        ctx.write(path, content)
        action = "Created" if is_new else "Updated"
        logger.info("organize_write_topic: %s '%s/%s' (%d chars)", action.lower(), category, topic, len(content))
        return f"{action} topic '{topic}' in '{category}'."

    elif name == "organize_remove_topic":
        category = arguments["category"]
        topic = arguments["topic"]
        path = f"knowledge/{category}/{topic}.md"
        if not ctx.exists(path):
            return f"Topic '{topic}' not found in '{category}'."
        ctx.write(path, "")
        logger.info("organize_remove_topic: removed '%s/%s'", category, topic)
        return f"Removed topic '{topic}' from '{category}'."

    elif name == "organize_archive_thoughts":
        before_marker = arguments.get("before_marker")
        label = arguments.get("label")

        thinking = ctx.read("thinking.md") or ""
        if not thinking.strip():
            return "thinking.md is empty, nothing to archive."

        if before_marker and before_marker in thinking:
            idx = thinking.index(before_marker)
            to_archive = thinking[:idx].strip()
            remaining = thinking[idx:].strip()
        else:
            to_archive = thinking.strip()
            remaining = ""

        if not to_archive:
            return "No content to archive."

        if not label:
            label = time.strftime("%Y-%m-%d_%H%M%S")

        archive_path = f"archive/{label}.md"
        ctx.write(archive_path, to_archive)
        ctx.write("thinking.md", remaining if remaining else "# Thinking\n")

        logger.info("organize_archive_thoughts: archived %d chars to %s", len(to_archive), archive_path)
        return f"Archived {len(to_archive)} chars to {archive_path}."

    elif name == "organize_list_archives":
        labels = _list_archive_labels(ctx)
        if not labels:
            return "No archive entries found."
        logger.info("organize_list_archives: %d entries", len(labels))
        return "\n".join(f"- {label}" for label in labels)

    elif name == "organize_read_archive":
        label = arguments["label"]
        archive_path = f"archive/{label}.md"
        content = ctx.read(archive_path)
        if not content:
            logger.info("organize_read_archive: '%s' not found", label)
            return f"Archive entry '{label}' not found."
        logger.info("organize_read_archive: '%s' (%d chars)", label, len(content))
        return content

    else:
        logger.warning("organize tool unknown: %s", name)
        return f"Unknown organize tool: {name}"
