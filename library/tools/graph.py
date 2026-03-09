"""Semantic graph — persistent directed weighted graph for relational memory.

The agent manipulates the graph via tool calls. The graph's topology (clustering,
centrality, bridge nodes) carries meaning beyond individual nodes or edges.

Storage is JSON via ctx.write/read, keeping it human-readable and diff-friendly.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from library.harness.context import InstanceContext

logger = logging.getLogger(__name__)

GRAPH_FILE = "graph.json"


@dataclass
class Node:
    id: str
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    source: str
    target: str
    relation: str
    weight: float = 0.5
    directed: bool = True


@dataclass
class Snapshot:
    label: str
    node_count: int
    edge_count: int


class SemanticGraph:
    """In-memory semantic graph with serialisation to/from JSON."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.snapshots: list[Snapshot] = []

    # --- Mutation ---

    def add_node(self, id: str, label: str, metadata: dict | None = None) -> None:
        self.nodes[id] = Node(id=id, label=label, metadata=metadata or {})

    def add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        weight: float = 0.5,
        directed: bool = True,
    ) -> None:
        # Update existing edge if same source/target/relation
        for i, e in enumerate(self.edges):
            if e.source == source and e.target == target and e.relation == relation:
                self.edges[i] = Edge(source, target, relation, weight, directed)
                return
        self.edges.append(Edge(source, target, relation, weight, directed))

    def remove_node(self, id: str) -> bool:
        if id not in self.nodes:
            return False
        del self.nodes[id]
        self.edges = [e for e in self.edges if e.source != id and e.target != id]
        return True

    def remove_edge(self, source: str, target: str, relation: str) -> bool:
        before = len(self.edges)
        self.edges = [
            e
            for e in self.edges
            if not (e.source == source and e.target == target and e.relation == relation)
        ]
        return len(self.edges) < before

    def snapshot(self, label: str) -> Snapshot:
        snap = Snapshot(
            label=label,
            node_count=len(self.nodes),
            edge_count=len(self.edges),
        )
        self.snapshots.append(snap)
        return snap

    # --- Query ---

    def query(
        self,
        *,
        node_ids: list[str] | None = None,
        relation_types: list[str] | None = None,
        min_weight: float | None = None,
        central_node: str | None = None,
        max_depth: int = 1,
    ) -> tuple[list[Node], list[Edge]]:
        """Return a subgraph matching filter criteria."""
        # Start with all edges, narrow down
        edges = list(self.edges)

        if relation_types is not None:
            edges = [e for e in edges if e.relation in relation_types]
        if min_weight is not None:
            edges = [e for e in edges if e.weight >= min_weight]

        if central_node is not None:
            # BFS from central node up to max_depth
            reachable: set[str] = {central_node}
            frontier = {central_node}
            for _ in range(max_depth):
                next_frontier: set[str] = set()
                for e in edges:
                    if e.source in frontier:
                        next_frontier.add(e.target)
                    if e.target in frontier:
                        next_frontier.add(e.source)
                next_frontier -= reachable
                reachable |= next_frontier
                frontier = next_frontier
                if not frontier:
                    break
            edges = [e for e in edges if e.source in reachable and e.target in reachable]
            node_set = reachable
        elif node_ids is not None:
            id_set = set(node_ids)
            edges = [e for e in edges if e.source in id_set or e.target in id_set]
            node_set = id_set | {e.source for e in edges} | {e.target for e in edges}
        else:
            node_set = {e.source for e in edges} | {e.target for e in edges}

        nodes = [self.nodes[nid] for nid in node_set if nid in self.nodes]
        return nodes, edges

    def degree(self, node_id: str) -> int:
        """Count edges connected to a node."""
        return sum(1 for e in self.edges if e.source == node_id or e.target == node_id)

    def describe(self) -> dict[str, Any]:
        """Summary statistics for agent consumption."""
        if not self.nodes:
            return {
                "node_count": 0,
                "edge_count": 0,
                "top_nodes": [],
                "isolated_nodes": [],
                "relation_types": [],
            }

        degrees: dict[str, int] = defaultdict(int)
        for e in self.edges:
            degrees[e.source] += 1
            degrees[e.target] += 1

        connected = set(degrees.keys())
        isolated = [nid for nid in self.nodes if nid not in connected]

        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        relation_types = sorted({e.relation for e in self.edges})

        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "top_nodes": [{"id": nid, "label": self.nodes[nid].label, "degree": deg}
                          for nid, deg in top_nodes if nid in self.nodes],
            "isolated_nodes": isolated,
            "relation_types": relation_types,
        }

    # --- Serialisation ---

    def to_dict(self) -> dict:
        return {
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for e in self.edges],
            "snapshots": [asdict(s) for s in self.snapshots],
        }

    @classmethod
    def from_dict(cls, data: dict) -> SemanticGraph:
        g = cls()
        for n in data.get("nodes", []):
            g.nodes[n["id"]] = Node(**n)
        for e in data.get("edges", []):
            g.edges.append(Edge(**e))
        for s in data.get("snapshots", []):
            g.snapshots.append(Snapshot(**s))
        return g

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, text: str) -> SemanticGraph:
        return cls.from_dict(json.loads(text))


# --- Context integration ---

def load_graph(ctx: InstanceContext) -> SemanticGraph:
    """Load the graph from instance storage, or return empty."""
    raw = ctx.read(GRAPH_FILE)
    if not raw:
        return SemanticGraph()
    try:
        return SemanticGraph.from_json(raw)
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.warning("Failed to parse graph.json, starting fresh")
        return SemanticGraph()


def save_graph(ctx: InstanceContext, graph: SemanticGraph) -> None:
    """Persist the graph to instance storage."""
    ctx.write(GRAPH_FILE, graph.to_json())


# --- Tool schemas ---

GRAPH_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "graph_add_node",
            "description": "Add or update a node in the semantic graph.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Stable identifier (slug format)"},
                    "label": {"type": "string", "description": "Human-readable name"},
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata (type, salience, notes)",
                    },
                },
                "required": ["id", "label"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_add_edge",
            "description": "Add or update an edge in the semantic graph.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string", "description": "Source node ID"},
                    "target_id": {"type": "string", "description": "Target node ID"},
                    "relation": {"type": "string", "description": "Relationship type"},
                    "weight": {
                        "type": "number",
                        "description": "Strength/confidence 0.0-1.0 (default 0.5)",
                    },
                    "directed": {
                        "type": "boolean",
                        "description": "Whether the edge is directed (default true)",
                    },
                },
                "required": ["source_id", "target_id", "relation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_remove_node",
            "description": "Remove a node and all its connected edges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Node ID to remove"},
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_remove_edge",
            "description": "Remove a specific edge.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string", "description": "Source node ID"},
                    "target_id": {"type": "string", "description": "Target node ID"},
                    "relation": {"type": "string", "description": "Relationship type"},
                },
                "required": ["source_id", "target_id", "relation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_query",
            "description": "Query a subgraph by filter criteria.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter to these node IDs",
                    },
                    "relation_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter to these relation types",
                    },
                    "min_weight": {
                        "type": "number",
                        "description": "Minimum edge weight",
                    },
                    "central_node": {
                        "type": "string",
                        "description": "BFS from this node",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Max BFS depth (default 1)",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_describe",
            "description": "Get summary statistics of the graph: node/edge counts, top nodes by centrality, isolated nodes.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_snapshot",
            "description": "Save a named snapshot of the current graph state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Snapshot label (e.g. session_042_end)"},
                },
                "required": ["label"],
            },
        },
    },
]


def handle_graph_tool(
    ctx: InstanceContext,
    name: str,
    arguments: dict,
) -> str:
    """Handle a graph tool call. Loads graph, mutates, saves, returns result text."""
    graph = load_graph(ctx)
    result: str

    if name == "graph_add_node":
        graph.add_node(
            id=arguments["id"],
            label=arguments["label"],
            metadata=arguments.get("metadata", {}),
        )
        save_graph(ctx, graph)
        result = f"Node '{arguments['id']}' added/updated."

    elif name == "graph_add_edge":
        graph.add_edge(
            source=arguments["source_id"],
            target=arguments["target_id"],
            relation=arguments["relation"],
            weight=arguments.get("weight", 0.5),
            directed=arguments.get("directed", True),
        )
        save_graph(ctx, graph)
        result = (
            f"Edge {arguments['source_id']} --[{arguments['relation']}]--> "
            f"{arguments['target_id']} added/updated."
        )

    elif name == "graph_remove_node":
        removed = graph.remove_node(arguments["id"])
        if removed:
            save_graph(ctx, graph)
            result = f"Node '{arguments['id']}' and connected edges removed."
        else:
            result = f"Node '{arguments['id']}' not found."

    elif name == "graph_remove_edge":
        removed = graph.remove_edge(
            arguments["source_id"],
            arguments["target_id"],
            arguments["relation"],
        )
        if removed:
            save_graph(ctx, graph)
            result = "Edge removed."
        else:
            result = "Edge not found."

    elif name == "graph_query":
        nodes, edges = graph.query(
            node_ids=arguments.get("node_ids"),
            relation_types=arguments.get("relation_types"),
            min_weight=arguments.get("min_weight"),
            central_node=arguments.get("central_node"),
            max_depth=arguments.get("max_depth", 1),
        )
        lines = [f"Subgraph: {len(nodes)} nodes, {len(edges)} edges"]
        for n in nodes:
            lines.append(f"  [{n.id}] {n.label}")
        for e in edges:
            arrow = "-->" if e.directed else "<-->"
            lines.append(f"  {e.source} {arrow} {e.target} ({e.relation}, w={e.weight:.2f})")
        result = "\n".join(lines)

    elif name == "graph_describe":
        desc = graph.describe()
        lines = [
            f"Graph: {desc['node_count']} nodes, {desc['edge_count']} edges",
            f"Relation types: {', '.join(desc['relation_types']) or 'none'}",
        ]
        if desc["top_nodes"]:
            lines.append("Top nodes by degree:")
            for n in desc["top_nodes"]:
                lines.append(f"  {n['label']} ({n['id']}): degree {n['degree']}")
        if desc["isolated_nodes"]:
            lines.append(f"Isolated: {', '.join(desc['isolated_nodes'])}")
        result = "\n".join(lines)

    elif name == "graph_snapshot":
        snap = graph.snapshot(arguments["label"])
        save_graph(ctx, graph)
        result = f"Snapshot '{snap.label}' saved ({snap.node_count} nodes, {snap.edge_count} edges)."

    else:
        result = f"Unknown graph tool: {name}"

    return result
