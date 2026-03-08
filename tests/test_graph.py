"""Tests for semantic graph toolkit."""

import json

import pytest
from unittest.mock import MagicMock

from symbiosis.toolkit.graph import (
    SemanticGraph,
    Node,
    Edge,
    load_graph,
    save_graph,
    handle_graph_tool,
    GRAPH_TOOL_SCHEMAS,
)


class TestSemanticGraph:
    def test_add_and_get_node(self):
        g = SemanticGraph()
        g.add_node("trust", "Trust", {"salience": 0.8})
        assert "trust" in g.nodes
        assert g.nodes["trust"].label == "Trust"
        assert g.nodes["trust"].metadata["salience"] == 0.8

    def test_add_node_overwrites(self):
        g = SemanticGraph()
        g.add_node("x", "Original")
        g.add_node("x", "Updated")
        assert g.nodes["x"].label == "Updated"

    def test_add_and_get_edge(self):
        g = SemanticGraph()
        g.add_node("a", "A")
        g.add_node("b", "B")
        g.add_edge("a", "b", "requires", weight=0.9)
        assert len(g.edges) == 1
        assert g.edges[0].relation == "requires"
        assert g.edges[0].weight == 0.9

    def test_add_edge_updates_existing(self):
        g = SemanticGraph()
        g.add_edge("a", "b", "requires", weight=0.5)
        g.add_edge("a", "b", "requires", weight=0.9)
        assert len(g.edges) == 1
        assert g.edges[0].weight == 0.9

    def test_add_edge_different_relation(self):
        g = SemanticGraph()
        g.add_edge("a", "b", "requires", weight=0.5)
        g.add_edge("a", "b", "opposes", weight=0.3)
        assert len(g.edges) == 2

    def test_remove_node(self):
        g = SemanticGraph()
        g.add_node("a", "A")
        g.add_node("b", "B")
        g.add_edge("a", "b", "links")
        assert g.remove_node("a")
        assert "a" not in g.nodes
        assert len(g.edges) == 0

    def test_remove_nonexistent_node(self):
        g = SemanticGraph()
        assert not g.remove_node("nope")

    def test_remove_edge(self):
        g = SemanticGraph()
        g.add_edge("a", "b", "links")
        assert g.remove_edge("a", "b", "links")
        assert len(g.edges) == 0

    def test_remove_nonexistent_edge(self):
        g = SemanticGraph()
        assert not g.remove_edge("a", "b", "nope")

    def test_degree(self):
        g = SemanticGraph()
        g.add_node("a", "A")
        g.add_node("b", "B")
        g.add_node("c", "C")
        g.add_edge("a", "b", "x")
        g.add_edge("a", "c", "y")
        assert g.degree("a") == 2
        assert g.degree("b") == 1

    def test_snapshot(self):
        g = SemanticGraph()
        g.add_node("a", "A")
        g.add_edge("a", "a", "self")
        snap = g.snapshot("test_snap")
        assert snap.label == "test_snap"
        assert snap.node_count == 1
        assert snap.edge_count == 1
        assert len(g.snapshots) == 1

    def test_describe_empty(self):
        g = SemanticGraph()
        desc = g.describe()
        assert desc["node_count"] == 0
        assert desc["edge_count"] == 0

    def test_describe_with_data(self):
        g = SemanticGraph()
        g.add_node("a", "A")
        g.add_node("b", "B")
        g.add_node("c", "C")  # isolated
        g.add_edge("a", "b", "links")
        desc = g.describe()
        assert desc["node_count"] == 3
        assert desc["edge_count"] == 1
        assert "c" in desc["isolated_nodes"]
        assert len(desc["top_nodes"]) == 2
        assert desc["relation_types"] == ["links"]


class TestGraphQuery:
    def _sample_graph(self):
        g = SemanticGraph()
        for nid in ["a", "b", "c", "d"]:
            g.add_node(nid, nid.upper())
        g.add_edge("a", "b", "knows", 0.9)
        g.add_edge("b", "c", "knows", 0.5)
        g.add_edge("c", "d", "trusts", 0.3)
        return g

    def test_query_by_node_ids(self):
        g = self._sample_graph()
        nodes, edges = g.query(node_ids=["a", "b"])
        ids = {n.id for n in nodes}
        assert "a" in ids
        assert "b" in ids

    def test_query_by_relation(self):
        g = self._sample_graph()
        nodes, edges = g.query(relation_types=["trusts"])
        assert len(edges) == 1
        assert edges[0].relation == "trusts"

    def test_query_by_min_weight(self):
        g = self._sample_graph()
        nodes, edges = g.query(min_weight=0.8)
        assert len(edges) == 1
        assert edges[0].weight == 0.9

    def test_query_central_node_depth_1(self):
        g = self._sample_graph()
        nodes, edges = g.query(central_node="b", max_depth=1)
        ids = {n.id for n in nodes}
        assert "b" in ids
        assert "a" in ids
        assert "c" in ids
        assert "d" not in ids

    def test_query_central_node_depth_2(self):
        g = self._sample_graph()
        nodes, edges = g.query(central_node="a", max_depth=2)
        ids = {n.id for n in nodes}
        assert ids == {"a", "b", "c"}

    def test_query_no_filters(self):
        g = self._sample_graph()
        nodes, edges = g.query()
        assert len(edges) == 3


class TestGraphSerialisation:
    def test_roundtrip(self):
        g = SemanticGraph()
        g.add_node("trust", "Trust", {"salience": 0.8})
        g.add_node("consistency", "Consistency")
        g.add_edge("trust", "consistency", "requires", 0.9)
        g.snapshot("test")

        text = g.to_json()
        g2 = SemanticGraph.from_json(text)
        assert len(g2.nodes) == 2
        assert len(g2.edges) == 1
        assert len(g2.snapshots) == 1
        assert g2.nodes["trust"].metadata["salience"] == 0.8
        assert g2.edges[0].weight == 0.9

    def test_from_empty_json(self):
        g = SemanticGraph.from_json('{"nodes": [], "edges": []}')
        assert len(g.nodes) == 0


class TestGraphContextIntegration:
    def test_load_empty(self):
        ctx = MagicMock()
        ctx.read.return_value = ""
        g = load_graph(ctx)
        assert len(g.nodes) == 0

    def test_load_and_save(self):
        stored = {}

        def mock_write(path, content):
            stored[path] = content

        def mock_read(path):
            return stored.get(path, "")

        ctx = MagicMock()
        ctx.read.side_effect = mock_read
        ctx.write.side_effect = mock_write

        g = SemanticGraph()
        g.add_node("x", "X")
        save_graph(ctx, g)

        g2 = load_graph(ctx)
        assert "x" in g2.nodes

    def test_load_corrupt_json(self):
        ctx = MagicMock()
        ctx.read.return_value = "not json"
        g = load_graph(ctx)
        assert len(g.nodes) == 0


class TestHandleGraphTool:
    def _mock_ctx(self):
        stored = {}

        def mock_write(path, content):
            stored[path] = content

        def mock_read(path):
            return stored.get(path, "")

        ctx = MagicMock()
        ctx.read.side_effect = mock_read
        ctx.write.side_effect = mock_write
        ctx._stored = stored
        return ctx

    def test_add_node(self):
        ctx = self._mock_ctx()
        result = handle_graph_tool(ctx, "graph_add_node", {"id": "x", "label": "X"})
        assert "added" in result
        assert ctx._stored.get("graph.json")

    def test_add_edge(self):
        ctx = self._mock_ctx()
        handle_graph_tool(ctx, "graph_add_node", {"id": "a", "label": "A"})
        handle_graph_tool(ctx, "graph_add_node", {"id": "b", "label": "B"})
        result = handle_graph_tool(ctx, "graph_add_edge", {
            "source_id": "a", "target_id": "b", "relation": "knows"
        })
        assert "added" in result

    def test_remove_node(self):
        ctx = self._mock_ctx()
        handle_graph_tool(ctx, "graph_add_node", {"id": "x", "label": "X"})
        result = handle_graph_tool(ctx, "graph_remove_node", {"id": "x"})
        assert "removed" in result

    def test_remove_nonexistent_node(self):
        ctx = self._mock_ctx()
        result = handle_graph_tool(ctx, "graph_remove_node", {"id": "nope"})
        assert "not found" in result

    def test_query(self):
        ctx = self._mock_ctx()
        handle_graph_tool(ctx, "graph_add_node", {"id": "a", "label": "A"})
        handle_graph_tool(ctx, "graph_add_node", {"id": "b", "label": "B"})
        handle_graph_tool(ctx, "graph_add_edge", {
            "source_id": "a", "target_id": "b", "relation": "links"
        })
        result = handle_graph_tool(ctx, "graph_query", {"central_node": "a"})
        assert "2 nodes" in result
        assert "links" in result

    def test_describe(self):
        ctx = self._mock_ctx()
        handle_graph_tool(ctx, "graph_add_node", {"id": "a", "label": "A"})
        result = handle_graph_tool(ctx, "graph_describe", {})
        assert "1 nodes" in result

    def test_snapshot(self):
        ctx = self._mock_ctx()
        handle_graph_tool(ctx, "graph_add_node", {"id": "a", "label": "A"})
        result = handle_graph_tool(ctx, "graph_snapshot", {"label": "session_1"})
        assert "session_1" in result
        assert "saved" in result

    def test_unknown_tool(self):
        ctx = self._mock_ctx()
        result = handle_graph_tool(ctx, "graph_unknown", {})
        assert "Unknown" in result

    def test_tool_schemas_valid(self):
        for schema in GRAPH_TOOL_SCHEMAS:
            assert schema["type"] == "function"
            assert "name" in schema["function"]
            assert schema["function"]["name"].startswith("graph_")
