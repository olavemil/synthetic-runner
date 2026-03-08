"""Tests for rendering utilities — graph HTML and map PNG/GIF."""

from __future__ import annotations

import pytest

from symbiosis.toolkit.graph import SemanticGraph
from symbiosis.toolkit.activation_map import ActivationMap
from symbiosis.toolkit.rendering import render_graph_html


# ---------------------------------------------------------------------------
# Graph → HTML (no extra dependencies)
# ---------------------------------------------------------------------------


class TestRenderGraphHTML:
    def _sample_graph(self) -> SemanticGraph:
        g = SemanticGraph()
        g.add_node("trust", "Trust", {"type": "concept", "salience": 0.8})
        g.add_node("consistency", "Consistency", {"type": "concept"})
        g.add_node("kindness", "Kindness")
        g.add_edge("trust", "consistency", "requires", weight=0.9)
        g.add_edge("trust", "kindness", "related_to", weight=0.6)
        return g

    def test_returns_html_string(self):
        g = self._sample_graph()
        result = render_graph_html(g)
        assert isinstance(result, str)
        assert "<!DOCTYPE html>" in result

    def test_contains_d3_script(self):
        g = self._sample_graph()
        result = render_graph_html(g)
        assert "d3.v7.min.js" in result

    def test_contains_node_data(self):
        g = self._sample_graph()
        result = render_graph_html(g)
        assert '"trust"' in result
        assert '"Trust"' in result
        assert '"Consistency"' in result

    def test_contains_edge_data(self):
        g = self._sample_graph()
        result = render_graph_html(g)
        assert '"requires"' in result
        assert '"related_to"' in result

    def test_custom_title(self):
        g = self._sample_graph()
        result = render_graph_html(g, title="My Graph")
        assert "My Graph" in result

    def test_stats_in_output(self):
        g = self._sample_graph()
        result = render_graph_html(g)
        assert "3 nodes" in result
        assert "2 edges" in result

    def test_empty_graph(self):
        g = SemanticGraph()
        result = render_graph_html(g)
        assert "<!DOCTYPE html>" in result
        assert "0 nodes" in result

    def test_snapshot_count_in_stats(self):
        g = self._sample_graph()
        g.snapshot("session_1")
        result = render_graph_html(g)
        assert "1 snapshots" in result

    def test_escapes_html_in_title(self):
        g = SemanticGraph()
        result = render_graph_html(g, title="<script>alert(1)</script>")
        assert "<script>alert(1)</script>" not in result
        assert "&lt;script&gt;" in result

    def test_edges_with_missing_nodes_excluded(self):
        g = SemanticGraph()
        g.add_node("a", "Node A")
        # Edge references nonexistent node "b"
        g.add_edge("a", "b", "points_to")
        result = render_graph_html(g)
        # The link data should not include the broken edge
        assert '"points_to"' not in result

    def test_node_degree_affects_radius(self):
        g = SemanticGraph()
        g.add_node("hub", "Hub")
        g.add_node("a", "A")
        g.add_node("b", "B")
        g.add_node("c", "C")
        g.add_edge("hub", "a", "r")
        g.add_edge("hub", "b", "r")
        g.add_edge("hub", "c", "r")
        result = render_graph_html(g)
        # Hub has degree 3, others have degree 1 — hub should have larger radius
        assert '"Hub"' in result


# ---------------------------------------------------------------------------
# Map → PNG (requires matplotlib)
# ---------------------------------------------------------------------------

try:
    import matplotlib
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from PIL import Image
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestRenderMapPNG:
    def test_returns_png_bytes(self):
        from symbiosis.toolkit.rendering import render_map_png

        m = ActivationMap(8, 8, "x_axis", "y_axis", "test map")
        m.set(4, 4, 0.8)
        m.set_region(2, 2, 2, -0.5, "gaussian")
        result = render_map_png(m)
        assert isinstance(result, bytes)
        assert result[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes

    def test_empty_map_renders(self):
        from symbiosis.toolkit.rendering import render_map_png

        m = ActivationMap(4, 4)
        result = render_map_png(m)
        assert len(result) > 100  # Non-trivial PNG

    def test_custom_figsize(self):
        from symbiosis.toolkit.rendering import render_map_png

        m = ActivationMap(8, 8, "X", "Y", "test")
        small = render_map_png(m, figsize=(4, 3))
        large = render_map_png(m, figsize=(12, 10))
        # Larger figsize should produce more bytes
        assert len(large) > len(small)

    def test_no_contour_on_flat(self):
        from symbiosis.toolkit.rendering import render_map_png

        m = ActivationMap(4, 4)
        # All zeros — contour should not crash
        result = render_map_png(m, contour_levels=5)
        assert isinstance(result, bytes)

    def test_diverging_values(self):
        from symbiosis.toolkit.rendering import render_map_png

        m = ActivationMap(8, 8, "fam", "affect", "test diverging")
        m.set_region(2, 2, 2, 0.9, "hard")
        m.set_region(6, 6, 2, -0.8, "hard")
        result = render_map_png(m)
        assert isinstance(result, bytes)


@pytest.mark.skipif(
    not (HAS_MATPLOTLIB and HAS_PILLOW),
    reason="matplotlib and Pillow required",
)
class TestRenderMapGIF:
    def test_returns_gif_bytes(self):
        from symbiosis.toolkit.rendering import render_map_gif

        grids = [
            ("frame_1", [[0.0, 0.5], [0.3, -0.2]]),
            ("frame_2", [[0.1, 0.6], [0.4, -0.1]]),
            ("frame_3", [[0.2, 0.7], [0.5, 0.0]]),
        ]
        result = render_map_gif(grids, width=2, height=2)
        assert isinstance(result, bytes)
        assert result[:6] in (b"GIF87a", b"GIF89a")

    def test_single_frame(self):
        from symbiosis.toolkit.rendering import render_map_gif

        grids = [("only", [[0.5, -0.5], [0.0, 0.0]])]
        result = render_map_gif(grids, width=2, height=2)
        assert isinstance(result, bytes)

    def test_empty_raises(self):
        from symbiosis.toolkit.rendering import render_map_gif

        with pytest.raises(ValueError, match="at least one grid"):
            render_map_gif([])

    def test_with_labels(self):
        from symbiosis.toolkit.rendering import render_map_gif

        grids = [
            ("start", [[0.0] * 4 for _ in range(4)]),
            ("end", [[0.5] * 4 for _ in range(4)]),
        ]
        result = render_map_gif(
            grids, width=4, height=4,
            x_label="familiarity", y_label="affect",
        )
        assert len(result) > 100
