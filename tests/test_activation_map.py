"""Tests for activation map toolkit."""

import json
import math

import pytest
from unittest.mock import MagicMock

from library.tools.activation_map import (
    ActivationMap,
    load_map,
    save_map,
    handle_map_tool,
    MAP_TOOL_SCHEMAS,
)


class TestActivationMap:
    def test_default_dimensions(self):
        m = ActivationMap()
        assert m.width == 16
        assert m.height == 16
        assert all(v == 0.0 for row in m.grid for v in row)

    def test_define(self):
        m = ActivationMap()
        m.define(8, 8, "familiarity", "affect", "test map")
        assert m.width == 8
        assert m.height == 8
        assert m.x_label == "familiarity"
        assert len(m.grid) == 8
        assert len(m.grid[0]) == 8

    def test_set_and_get(self):
        m = ActivationMap(4, 4)
        m.set(2, 3, 0.7)
        assert m.get_cell(2, 3) == pytest.approx(0.7)

    def test_set_clamps(self):
        m = ActivationMap(4, 4)
        m.set(0, 0, 2.0)
        assert m.get_cell(0, 0) == 1.0
        m.set(0, 0, -2.0)
        assert m.get_cell(0, 0) == -1.0

    def test_set_out_of_bounds_ignored(self):
        m = ActivationMap(4, 4)
        m.set(10, 10, 0.5)  # Should not raise
        assert m.get_cell(10, 10) == 0.0

    def test_set_region_hard(self):
        m = ActivationMap(8, 8)
        m.set_region(4, 4, 1, 0.9, falloff="hard")
        assert m.get_cell(4, 4) == pytest.approx(0.9)
        assert m.get_cell(4, 3) == pytest.approx(0.9)
        assert m.get_cell(3, 4) == pytest.approx(0.9)
        # Corner at distance sqrt(2) > 1, should not be set
        assert m.get_cell(3, 3) == 0.0

    def test_set_region_linear(self):
        m = ActivationMap(8, 8)
        m.set_region(4, 4, 2, 1.0, falloff="linear")
        assert m.get_cell(4, 4) == pytest.approx(1.0)
        # Distance 1 from centre → factor 0.5
        assert m.get_cell(5, 4) == pytest.approx(0.5)
        # Distance 2 from centre → factor 0.0
        assert m.get_cell(6, 4) == pytest.approx(0.0)

    def test_set_region_gaussian(self):
        m = ActivationMap(8, 8)
        m.set_region(4, 4, 2, 1.0, falloff="gaussian")
        assert m.get_cell(4, 4) == pytest.approx(1.0)
        # Distance 1, sigma=1 → exp(-0.5) ≈ 0.607
        assert m.get_cell(5, 4) == pytest.approx(math.exp(-0.5), abs=0.01)

    def test_clear(self):
        m = ActivationMap(4, 4, x_label="X", y_label="Y", description="test")
        m.set(1, 1, 0.5)
        m.clear()
        assert m.get_cell(1, 1) == 0.0
        assert m.x_label == "X"  # Metadata preserved

    def test_snapshot(self):
        m = ActivationMap(8, 8, x_label="X", y_label="Y")
        snap = m.snapshot("session_1")
        assert snap.label == "session_1"
        assert snap.width == 8
        assert len(m.snapshots) == 1

    def test_describe_empty(self):
        m = ActivationMap(4, 4)
        desc = m.describe()
        assert desc["active_cells"] == 0
        assert desc["mean"] == 0.0

    def test_describe_with_data(self):
        m = ActivationMap(4, 4, x_label="X", y_label="Y", description="test")
        m.set(1, 1, 0.8)
        m.set(2, 2, -0.5)
        desc = m.describe()
        assert desc["positive_cells"] == 1
        assert desc["negative_cells"] == 1
        assert desc["peak"]["value"] == 0.8
        assert desc["trough"]["value"] == -0.5


class TestMapSerialisation:
    def test_roundtrip(self):
        m = ActivationMap(8, 8, "fam", "aff", "testing")
        m.set(3, 5, 0.7)
        m.snapshot("s1")

        text = m.to_json()
        m2 = ActivationMap.from_json(text)
        assert m2.width == 8
        assert m2.x_label == "fam"
        assert m2.get_cell(3, 5) == pytest.approx(0.7)
        assert len(m2.snapshots) == 1
        assert m2.snapshots[0].label == "s1"

    def test_from_empty_json(self):
        m = ActivationMap.from_json('{"width": 4, "height": 4}')
        assert m.width == 4


class TestMapContextIntegration:
    def test_load_empty(self):
        ctx = MagicMock()
        ctx.read.return_value = ""
        m = load_map(ctx)
        assert m.width == 16

    def test_load_and_save(self):
        stored = {}
        ctx = MagicMock()
        ctx.read.side_effect = lambda p: stored.get(p, "")
        ctx.write.side_effect = lambda p, c: stored.__setitem__(p, c)

        m = ActivationMap(8, 8)
        m.set(1, 1, 0.5)
        save_map(ctx, m)

        m2 = load_map(ctx)
        assert m2.get_cell(1, 1) == pytest.approx(0.5)

    def test_load_corrupt_json(self):
        ctx = MagicMock()
        ctx.read.return_value = "not json"
        m = load_map(ctx)
        assert m.width == 16


class TestHandleMapTool:
    def _mock_ctx(self):
        stored = {}
        ctx = MagicMock()
        ctx.read.side_effect = lambda p: stored.get(p, "")
        ctx.write.side_effect = lambda p, c: stored.__setitem__(p, c)
        ctx._stored = stored
        return ctx

    def test_define(self):
        ctx = self._mock_ctx()
        result = handle_map_tool(ctx, "map_define", {
            "width": 8, "height": 8,
            "x_label": "familiarity", "y_label": "affect",
            "description": "test map",
        })
        assert "8x8" in result
        assert "familiarity" in result

    def test_set(self):
        ctx = self._mock_ctx()
        result = handle_map_tool(ctx, "map_set", {"x": 3, "y": 5, "value": 0.7})
        assert "0.70" in result

    def test_set_region(self):
        ctx = self._mock_ctx()
        result = handle_map_tool(ctx, "map_set_region", {
            "x": 4, "y": 4, "radius": 2, "value": 0.8,
        })
        assert "region" in result.lower()

    def test_get(self):
        ctx = self._mock_ctx()
        handle_map_tool(ctx, "map_set", {"x": 0, "y": 0, "value": 0.5})
        result = handle_map_tool(ctx, "map_get", {})
        data = json.loads(result)
        assert "grid" in data

    def test_describe(self):
        ctx = self._mock_ctx()
        handle_map_tool(ctx, "map_set", {"x": 1, "y": 1, "value": 0.8})
        result = handle_map_tool(ctx, "map_describe", {})
        assert "Map:" in result

    def test_clear(self):
        ctx = self._mock_ctx()
        handle_map_tool(ctx, "map_set", {"x": 0, "y": 0, "value": 0.9})
        result = handle_map_tool(ctx, "map_clear", {})
        assert "cleared" in result.lower()

    def test_snapshot(self):
        ctx = self._mock_ctx()
        result = handle_map_tool(ctx, "map_snapshot", {"label": "s1"})
        assert "s1" in result
        assert "saved" in result.lower()

    def test_unknown_tool(self):
        ctx = self._mock_ctx()
        result = handle_map_tool(ctx, "map_unknown", {})
        assert "Unknown" in result

    def test_tool_schemas_valid(self):
        for schema in MAP_TOOL_SCHEMAS:
            assert schema["type"] == "function"
            assert schema["function"]["name"].startswith("map_")
