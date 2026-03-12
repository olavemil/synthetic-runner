"""Scope-based tool filtering for species tool-use sessions.

Tools declare which scope they belong to (what kind of operation they perform).
Agent phases declare which scopes they need, and receive only the matching tools.

Scopes:
  archive-write   — archive/compress accumulated thoughts
  knowledge-read  — list and read knowledge topics and archives
  knowledge-write — create, update, or remove knowledge topics
  graph-read      — query and inspect the semantic graph
  graph-write     — add, remove, or snapshot graph nodes/edges
  map-read        — read and describe the activation map
  map-write       — define, set, clear, or snapshot the activation map
  creative-read   — list and read creative artifacts
  creative-write  — create, edit, or delete creative artifacts

Predefined scope sets per agent phase:
  THINK_SCOPES        — active introspection: full read/write (no creative)
  ORGANIZE_SCOPES     — knowledge/graph/map management (no archive or creative)
  SUBCONSCIOUS_SCOPES — surface tensions: read-only across all structured state
  DREAM_SCOPES        — associative: read graph and map only (no structured knowledge)
  CREATE_SCOPES       — creative: write only creative output, read everything else
  SLEEP_SCOPES        — consolidation: archive + update knowledge, read graph/map
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Scope constants
# ---------------------------------------------------------------------------

ARCHIVE_WRITE = "archive-write"
KNOWLEDGE_READ = "knowledge-read"
KNOWLEDGE_WRITE = "knowledge-write"
GRAPH_READ = "graph-read"
GRAPH_WRITE = "graph-write"
MAP_READ = "map-read"
MAP_WRITE = "map-write"
CREATIVE_READ = "creative-read"
CREATIVE_WRITE = "creative-write"

# ---------------------------------------------------------------------------
# Named scope sets for agent phases
# ---------------------------------------------------------------------------

# Active introspection: full read/write access to all structured state.
# No creative-write — think phases are not for producing creative artifacts.
THINK_SCOPES = frozenset([
    ARCHIVE_WRITE,
    KNOWLEDGE_READ, KNOWLEDGE_WRITE,
    GRAPH_READ, GRAPH_WRITE,
    MAP_READ, MAP_WRITE,
])

# Knowledge and graph/map management.
# No archive-write (doesn't compress thoughts) and no creative access.
ORGANIZE_SCOPES = frozenset([
    KNOWLEDGE_READ, KNOWLEDGE_WRITE,
    GRAPH_READ, GRAPH_WRITE,
    MAP_READ, MAP_WRITE,
])

# Surfaces tensions, curiosities, and unresolved ideas from existing state.
# Read-only — subconscious observes, it doesn't modify.
# Includes structured knowledge (tensions can reference known concepts).
SUBCONSCIOUS_SCOPES = frozenset([
    KNOWLEDGE_READ,
    GRAPH_READ,
    MAP_READ,
])

# Associative, pre-linguistic processing.
# Read-only and narrower than subconscious — dreams draw from graph/map
# patterns rather than structured knowledge categories.
# Creative output is included: existing works can seed associative imagery.
DREAM_SCOPES = frozenset([
    GRAPH_READ,
    MAP_READ,
    CREATIVE_READ,
])

# Creative expression: write creative artifacts, read broadly for inspiration.
# No write access to anything other than creative output.
CREATE_SCOPES = frozenset([
    KNOWLEDGE_READ,
    GRAPH_READ,
    MAP_READ,
    CREATIVE_READ, CREATIVE_WRITE,
])

# Consolidation: archive old thoughts, update knowledge from the session.
# Reads graph/map for context but does not write them —
# graph/map state is shaped during active phases (think, organize).
SLEEP_SCOPES = frozenset([
    ARCHIVE_WRITE,
    KNOWLEDGE_READ, KNOWLEDGE_WRITE,
    GRAPH_READ,
    MAP_READ,
])


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_for_scopes(
    schemas: list[dict],
    scope_map: dict[str, frozenset[str]],
    scopes: frozenset[str],
) -> list[dict]:
    """Return schemas whose tool scope intersects with the requested scopes.

    Tools not present in scope_map are excluded (no default inclusion).
    """
    result = []
    for schema in schemas:
        name = schema["function"]["name"]
        tool_scopes = scope_map.get(name)
        if tool_scopes is not None and tool_scopes & scopes:
            result.append(schema)
    return result


def get_tools_for_scopes(
    scopes: frozenset[str],
    *,
    graph: bool = False,
    activation_map: bool = False,
    creative: bool = False,
) -> list[dict]:
    """Get organize tools plus optional tool sets, filtered by scopes.

    Args:
        scopes: Set of scope strings declaring which operations are permitted.
                Use the predefined sets (THINK_SCOPES, ORGANIZE_SCOPES,
                CREATE_SCOPES) or compose a custom frozenset.
        graph: Include graph tools (default False).
        activation_map: Include activation map tools (default False).
        creative: Include creative artifact tools (default False).

    Returns:
        List of tool schema dicts applicable for the given scopes.
    """
    from library.tools.organize import ORGANIZE_TOOL_SCHEMAS, ORGANIZE_SCOPE_MAP
    tools = filter_for_scopes(ORGANIZE_TOOL_SCHEMAS, ORGANIZE_SCOPE_MAP, scopes)

    if graph:
        from library.tools.graph import GRAPH_TOOL_SCHEMAS, GRAPH_SCOPE_MAP
        tools = tools + filter_for_scopes(GRAPH_TOOL_SCHEMAS, GRAPH_SCOPE_MAP, scopes)

    if activation_map:
        from library.tools.activation_map import MAP_TOOL_SCHEMAS, MAP_SCOPE_MAP
        tools = tools + filter_for_scopes(MAP_TOOL_SCHEMAS, MAP_SCOPE_MAP, scopes)

    if creative:
        from library.tools.creative import CREATIVE_TOOL_SCHEMAS, CREATIVE_SCOPE_MAP
        tools = tools + filter_for_scopes(CREATIVE_TOOL_SCHEMAS, CREATIVE_SCOPE_MAP, scopes)

    return tools
