"""Creative artifact tools for persistent creative expression.

Provides tools for creating, editing, reading, and managing creative artifacts
stored in creations/ within instance memory. Supports multiple content types:
art (SVG), narrative (markdown), poetry (markdown), music (ABC notation),
and game (self-contained HTML).

Tools are exposed via CREATIVE_TOOL_SCHEMAS and dispatched via handle_creative_tool().
"""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from library.harness.context import InstanceContext

logger = logging.getLogger(__name__)

# Valid creative types and their file extensions
CREATIVE_TYPES = {
    "art": ".svg",
    "narrative": ".md",
    "poetry": ".md",
    "music": ".abc",
    "game": ".html",
}

# Size limits per type (bytes)
_SIZE_LIMITS = {
    "art": 100_000,
    "narrative": 50_000,
    "poetry": 50_000,
    "music": 20_000,
    "game": 200_000,
}

_CREATIONS_DIR = "creations"


# ---------------------------------------------------------------------------
# Phase map
# ---------------------------------------------------------------------------

CREATIVE_SCOPE_MAP: dict[str, frozenset[str]] = {
    "creative_list":   frozenset(["creative-read"]),
    "creative_read":   frozenset(["creative-read"]),
    "creative_new":    frozenset(["creative-write"]),
    "creative_edit":   frozenset(["creative-write"]),
    "creative_delete": frozenset(["creative-write"]),
}


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

CREATIVE_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "creative_list",
            "description": "List all creative works with title, type, and date.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "creative_read",
            "description": "Read a creative work by its slug identifier.",
            "parameters": {
                "type": "object",
                "properties": {
                    "slug": {"type": "string", "description": "Slug identifier (from creative_list)"},
                },
                "required": ["slug"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "creative_new",
            "description": "Create a new creative artifact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["art", "narrative", "poetry", "music", "game"],
                        "description": "Type of creative work",
                    },
                    "title": {
                        "type": "string",
                        "description": "Title of the work",
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "The creative content: SVG markup for art, markdown for narrative/poetry, "
                            "ABC notation for music, or self-contained HTML for game"
                        ),
                    },
                },
                "required": ["type", "title", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "creative_edit",
            "description": "Update an existing creation's content. Replaces the full content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "slug": {"type": "string", "description": "Slug identifier (from creative_list)"},
                    "content": {"type": "string", "description": "Updated content"},
                },
                "required": ["slug", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "creative_delete",
            "description": "Remove a creative work.",
            "parameters": {
                "type": "object",
                "properties": {
                    "slug": {"type": "string", "description": "Slug identifier (from creative_list)"},
                },
                "required": ["slug"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Slug generation
# ---------------------------------------------------------------------------

def _slugify(title: str) -> str:
    """Convert a title to a filesystem-safe slug."""
    slug = title.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    slug = slug.strip("-")
    return slug[:60] or "untitled"


# ---------------------------------------------------------------------------
# Frontmatter handling
# ---------------------------------------------------------------------------

def _make_frontmatter(title: str, creative_type: str, created: int, updated: int) -> str:
    """Build YAML frontmatter string."""
    return (
        f"---\n"
        f"title: \"{title}\"\n"
        f"type: {creative_type}\n"
        f"created: {created}\n"
        f"updated: {updated}\n"
        f"---\n"
    )


def _make_comment_header(title: str, creative_type: str, created: int, updated: int) -> str:
    """Build an HTML/XML comment header for non-markdown types."""
    return f"<!-- title: {title} | type: {creative_type} | created: {created} | updated: {updated} -->\n"


def _parse_metadata(raw: str) -> dict:
    """Extract metadata from frontmatter or comment header."""
    meta: dict = {}

    # Try YAML frontmatter
    if raw.startswith("---\n"):
        end = raw.find("\n---\n", 4)
        if end != -1:
            header = raw[4:end]
            for line in header.split("\n"):
                if ":" in line:
                    key, _, val = line.partition(":")
                    val = val.strip().strip('"')
                    meta[key.strip()] = val
            meta["_body_start"] = end + 5
            return meta

    # Try comment header
    m = re.match(r"^<!--\s*(.+?)\s*-->\n?", raw)
    if m:
        for part in m.group(1).split("|"):
            if ":" in part:
                key, _, val = part.partition(":")
                meta[key.strip()] = val.strip()
        meta["_body_start"] = m.end()
        return meta

    return meta


def _get_body(raw: str) -> str:
    """Extract body content after frontmatter/header."""
    meta = _parse_metadata(raw)
    start = meta.get("_body_start", 0)
    return raw[start:] if isinstance(start, int) else raw


def _wrap_content(title: str, creative_type: str, content: str, created: int, updated: int) -> str:
    """Wrap content with appropriate frontmatter/header for the type."""
    ext = CREATIVE_TYPES[creative_type]
    if ext == ".md":
        return _make_frontmatter(title, creative_type, created, updated) + "\n" + content
    else:
        return _make_comment_header(title, creative_type, created, updated) + content


# ---------------------------------------------------------------------------
# SVG sanitization
# ---------------------------------------------------------------------------

def _sanitize_svg(content: str) -> str:
    """Remove potentially dangerous elements from SVG content."""
    # Strip <script> elements
    content = re.sub(r"<script[\s>].*?</script>", "", content, flags=re.DOTALL | re.IGNORECASE)
    # Strip on* event attributes
    content = re.sub(r'\s+on\w+\s*=\s*"[^"]*"', "", content, flags=re.IGNORECASE)
    content = re.sub(r"\s+on\w+\s*=\s*'[^']*'", "", content, flags=re.IGNORECASE)
    # Strip <foreignObject> elements
    content = re.sub(
        r"<foreignObject[\s>].*?</foreignObject>", "", content, flags=re.DOTALL | re.IGNORECASE
    )
    return content


# ---------------------------------------------------------------------------
# File path helpers
# ---------------------------------------------------------------------------

def _file_path(slug: str, creative_type: str) -> str:
    """Build the storage path for a creation."""
    ext = CREATIVE_TYPES[creative_type]
    return f"{_CREATIONS_DIR}/{slug}{ext}"


def _find_creation(ctx: "InstanceContext", slug: str) -> tuple[str, str, dict] | None:
    """Find a creation by slug. Returns (path, raw_content, metadata) or None."""
    files = ctx.list(f"{_CREATIONS_DIR}/")
    for path in files:
        # Extract slug from path: creations/my-slug.ext → my-slug
        filename = path.split("/")[-1] if "/" in path else path
        file_slug = filename.rsplit(".", 1)[0] if "." in filename else filename
        if file_slug == slug:
            raw = ctx.read(path) or ""
            meta = _parse_metadata(raw)
            return path, raw, meta
    return None


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def _handle_list(ctx: "InstanceContext") -> str:
    """List all creative works."""
    files = ctx.list(f"{_CREATIONS_DIR}/")
    if not files:
        return "(no creations yet)"

    lines = []
    for path in sorted(files):
        raw = ctx.read(path) or ""
        meta = _parse_metadata(raw)
        filename = path.split("/")[-1] if "/" in path else path
        slug = filename.rsplit(".", 1)[0] if "." in filename else filename
        title = meta.get("title", slug)
        creative_type = meta.get("type", "unknown")
        updated = meta.get("updated", "")
        lines.append(f"- `{slug}` ({creative_type}) — {title} [updated: {updated}]")

    return "\n".join(lines)


def _handle_read(ctx: "InstanceContext", slug: str) -> str:
    """Read a creation's content."""
    result = _find_creation(ctx, slug)
    if result is None:
        return f"Creation not found: {slug}"
    _path, raw, meta = result
    title = meta.get("title", slug)
    creative_type = meta.get("type", "unknown")
    body = _get_body(raw)
    return f"**{title}** ({creative_type})\n\n{body}"


def _handle_new(ctx: "InstanceContext", creative_type: str, title: str, content: str) -> str:
    """Create a new creative artifact."""
    if creative_type not in CREATIVE_TYPES:
        return f"Unknown type: {creative_type}. Valid types: {', '.join(CREATIVE_TYPES)}"

    # Size check
    limit = _SIZE_LIMITS[creative_type]
    if len(content.encode()) > limit:
        return f"Content too large ({len(content.encode())} bytes). Limit for {creative_type}: {limit} bytes."

    # Sanitize SVG
    if creative_type == "art":
        content = _sanitize_svg(content)

    slug = _slugify(title)

    # Check for existing creation with same slug
    if _find_creation(ctx, slug) is not None:
        # Append a short suffix
        slug = f"{slug}-{int(time.time()) % 10000}"

    now = int(time.time())
    wrapped = _wrap_content(title, creative_type, content, now, now)
    path = _file_path(slug, creative_type)
    ctx.write(path, wrapped)
    logger.info("Created %s (%s): %s", creative_type, slug, title)
    return f"Created {creative_type} '{title}' → {slug}"


def _handle_edit(ctx: "InstanceContext", slug: str, content: str) -> str:
    """Update an existing creation."""
    result = _find_creation(ctx, slug)
    if result is None:
        return f"Creation not found: {slug}"
    path, _raw, meta = result

    creative_type = meta.get("type", "narrative")
    title = meta.get("title", slug)
    created = int(meta.get("created", 0)) or int(time.time())

    # Size check
    limit = _SIZE_LIMITS.get(creative_type, 50_000)
    if len(content.encode()) > limit:
        return f"Content too large ({len(content.encode())} bytes). Limit for {creative_type}: {limit} bytes."

    # Sanitize SVG
    if creative_type == "art":
        content = _sanitize_svg(content)

    now = int(time.time())
    wrapped = _wrap_content(title, creative_type, content, created, now)
    ctx.write(path, wrapped)
    logger.info("Updated %s (%s): %s", creative_type, slug, title)
    return f"Updated '{title}' ({slug})"


def _handle_delete(ctx: "InstanceContext", slug: str) -> str:
    """Delete a creation."""
    result = _find_creation(ctx, slug)
    if result is None:
        return f"Creation not found: {slug}"
    path, _raw, _meta = result
    ctx.write(path, "")
    logger.info("Deleted creation: %s", slug)
    return f"Deleted: {slug}"


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def handle_creative_tool(ctx: "InstanceContext", name: str, arguments: dict) -> str:
    """Dispatch a creative_* tool call. Returns result text."""
    if name == "creative_list":
        return _handle_list(ctx)
    if name == "creative_read":
        return _handle_read(ctx, arguments.get("slug", ""))
    if name == "creative_new":
        return _handle_new(
            ctx,
            arguments.get("type", "narrative"),
            arguments.get("title", "Untitled"),
            arguments.get("content", ""),
        )
    if name == "creative_edit":
        return _handle_edit(ctx, arguments.get("slug", ""), arguments.get("content", ""))
    if name == "creative_delete":
        return _handle_delete(ctx, arguments.get("slug", ""))
    return f"Unknown creative tool: {name}"
