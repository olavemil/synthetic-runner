"""Activation map — 2D grid for attentional/affective state.

The agent defines what the axes mean and places float values (-1.0 to 1.0)
across a conceptual space. The map is not a fixed schema — different sessions
may use different framings. The definition itself is stored as metadata.

Storage is JSON via ctx.write/read. Feeds the fast net as a natural tensor input.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from symbiosis.harness.context import InstanceContext

logger = logging.getLogger(__name__)

MAP_FILE = "activation_map.json"


@dataclass
class MapSnapshot:
    label: str
    width: int
    height: int
    x_label: str
    y_label: str


class ActivationMap:
    """2D float grid with axis metadata and snapshot history."""

    def __init__(
        self,
        width: int = 16,
        height: int = 16,
        x_label: str = "",
        y_label: str = "",
        description: str = "",
    ) -> None:
        self.width = width
        self.height = height
        self.x_label = x_label
        self.y_label = y_label
        self.description = description
        self.grid: list[list[float]] = [[0.0] * width for _ in range(height)]
        self.snapshots: list[MapSnapshot] = []

    def define(
        self,
        width: int,
        height: int,
        x_label: str,
        y_label: str,
        description: str,
    ) -> None:
        """Reinitialise map dimensions and semantics."""
        self.width = width
        self.height = height
        self.x_label = x_label
        self.y_label = y_label
        self.description = description
        self.grid = [[0.0] * width for _ in range(height)]

    def _clamp(self, value: float) -> float:
        return max(-1.0, min(1.0, value))

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def set(self, x: int, y: int, value: float) -> None:
        if not self._in_bounds(x, y):
            return
        self.grid[y][x] = self._clamp(value)

    def get_cell(self, x: int, y: int) -> float:
        if not self._in_bounds(x, y):
            return 0.0
        return self.grid[y][x]

    def set_region(
        self,
        x: int,
        y: int,
        radius: int,
        value: float,
        falloff: str = "linear",
    ) -> None:
        """Set a circular region with optional falloff."""
        value = self._clamp(value)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                px, py = x + dx, y + dy
                if not self._in_bounds(px, py):
                    continue
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > radius:
                    continue

                if falloff == "hard":
                    self.grid[py][px] = value
                elif falloff == "gaussian":
                    sigma = radius / 2.0 if radius > 0 else 1.0
                    factor = math.exp(-(dist * dist) / (2 * sigma * sigma))
                    self.grid[py][px] = self._clamp(value * factor)
                else:  # linear
                    factor = 1.0 - (dist / radius) if radius > 0 else 1.0
                    self.grid[py][px] = self._clamp(value * factor)

    def clear(self) -> None:
        """Reset all values to 0.0. Metadata preserved."""
        self.grid = [[0.0] * self.width for _ in range(self.height)]

    def snapshot(self, label: str) -> MapSnapshot:
        snap = MapSnapshot(
            label=label,
            width=self.width,
            height=self.height,
            x_label=self.x_label,
            y_label=self.y_label,
        )
        self.snapshots.append(snap)
        return snap

    def describe(self) -> dict[str, Any]:
        """Natural language summary for agent consumption."""
        flat = [v for row in self.grid for v in row]
        if not flat:
            return {"summary": "Empty map.", "x_label": self.x_label, "y_label": self.y_label}

        total = len(flat)
        nonzero = [v for v in flat if abs(v) > 0.01]
        positive = [v for v in flat if v > 0.01]
        negative = [v for v in flat if v < -0.01]

        mean_val = sum(flat) / total

        # Find peak
        peak_val = 0.0
        peak_x, peak_y = 0, 0
        trough_val = 0.0
        trough_x, trough_y = 0, 0
        for y in range(self.height):
            for x in range(self.width):
                v = self.grid[y][x]
                if v > peak_val:
                    peak_val = v
                    peak_x, peak_y = x, y
                if v < trough_val:
                    trough_val = v
                    trough_x, trough_y = x, y

        return {
            "dimensions": f"{self.width}x{self.height}",
            "x_label": self.x_label,
            "y_label": self.y_label,
            "description": self.description,
            "mean": round(mean_val, 3),
            "active_cells": len(nonzero),
            "positive_cells": len(positive),
            "negative_cells": len(negative),
            "peak": {"x": peak_x, "y": peak_y, "value": round(peak_val, 3)},
            "trough": {"x": trough_x, "y": trough_y, "value": round(trough_val, 3)},
        }

    # --- Serialisation ---

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "description": self.description,
            "grid": self.grid,
            "snapshots": [asdict(s) for s in self.snapshots],
        }

    @classmethod
    def from_dict(cls, data: dict) -> ActivationMap:
        m = cls(
            width=data.get("width", 16),
            height=data.get("height", 16),
            x_label=data.get("x_label", ""),
            y_label=data.get("y_label", ""),
            description=data.get("description", ""),
        )
        if "grid" in data:
            m.grid = data["grid"]
        for s in data.get("snapshots", []):
            m.snapshots.append(MapSnapshot(**s))
        return m

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, text: str) -> ActivationMap:
        return cls.from_dict(json.loads(text))


# --- Context integration ---

def load_map(ctx: InstanceContext) -> ActivationMap:
    """Load activation map from instance storage, or return default."""
    raw = ctx.read(MAP_FILE)
    if not raw:
        return ActivationMap()
    try:
        return ActivationMap.from_json(raw)
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.warning("Failed to parse activation_map.json, starting fresh")
        return ActivationMap()


def save_map(ctx: InstanceContext, m: ActivationMap) -> None:
    ctx.write(MAP_FILE, m.to_json())


# --- Tool schemas ---

MAP_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "map_define",
            "description": "Initialise or redefine the activation map dimensions and axis semantics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "width": {"type": "integer", "description": "Grid width (suggest 16-32)"},
                    "height": {"type": "integer", "description": "Grid height (suggest 16-32)"},
                    "x_label": {"type": "string", "description": "What the X axis represents"},
                    "y_label": {"type": "string", "description": "What the Y axis represents"},
                    "description": {"type": "string", "description": "What this map captures"},
                },
                "required": ["width", "height", "x_label", "y_label", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "map_set",
            "description": "Set a single cell value (-1.0 to 1.0).",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "X coordinate"},
                    "y": {"type": "integer", "description": "Y coordinate"},
                    "value": {"type": "number", "description": "Float -1.0 to 1.0"},
                },
                "required": ["x", "y", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "map_set_region",
            "description": "Set a circular region with optional falloff.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "Centre X"},
                    "y": {"type": "integer", "description": "Centre Y"},
                    "radius": {"type": "integer", "description": "Radius in cells"},
                    "value": {"type": "number", "description": "Float -1.0 to 1.0"},
                    "falloff": {
                        "type": "string",
                        "enum": ["linear", "gaussian", "hard"],
                        "description": "Falloff type (default linear)",
                    },
                },
                "required": ["x", "y", "radius", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "map_get",
            "description": "Return the full grid with metadata.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "map_describe",
            "description": "Return a natural language summary of the map state.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "map_clear",
            "description": "Reset all values to 0.0. Metadata and axis definitions preserved.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "map_snapshot",
            "description": "Save a named snapshot of the current map state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Snapshot label"},
                },
                "required": ["label"],
            },
        },
    },
]


def handle_map_tool(
    ctx: InstanceContext,
    name: str,
    arguments: dict,
) -> str:
    """Handle a map tool call. Loads map, mutates, saves, returns result text."""
    m = load_map(ctx)
    result: str

    if name == "map_define":
        m.define(
            width=arguments["width"],
            height=arguments["height"],
            x_label=arguments["x_label"],
            y_label=arguments["y_label"],
            description=arguments["description"],
        )
        save_map(ctx, m)
        result = (
            f"Map defined: {m.width}x{m.height}, "
            f"X={m.x_label}, Y={m.y_label}. {m.description}"
        )

    elif name == "map_set":
        m.set(arguments["x"], arguments["y"], arguments["value"])
        save_map(ctx, m)
        result = f"Cell ({arguments['x']}, {arguments['y']}) set to {arguments['value']:.2f}."

    elif name == "map_set_region":
        m.set_region(
            x=arguments["x"],
            y=arguments["y"],
            radius=arguments["radius"],
            value=arguments["value"],
            falloff=arguments.get("falloff", "linear"),
        )
        save_map(ctx, m)
        result = (
            f"Region at ({arguments['x']}, {arguments['y']}) r={arguments['radius']} "
            f"set to {arguments['value']:.2f} ({arguments.get('falloff', 'linear')} falloff)."
        )

    elif name == "map_get":
        data = m.to_dict()
        result = json.dumps(data, indent=2)

    elif name == "map_describe":
        desc = m.describe()
        lines = [
            f"Map: {desc['dimensions']}, X={desc['x_label']}, Y={desc['y_label']}",
        ]
        if desc.get("description"):
            lines.append(f"Purpose: {desc['description']}")
        lines.append(f"Mean: {desc['mean']}, Active: {desc['active_cells']}")
        lines.append(f"Positive: {desc['positive_cells']}, Negative: {desc['negative_cells']}")
        if desc["peak"]["value"] > 0:
            p = desc["peak"]
            lines.append(f"Peak: ({p['x']}, {p['y']}) = {p['value']}")
        if desc["trough"]["value"] < 0:
            t = desc["trough"]
            lines.append(f"Trough: ({t['x']}, {t['y']}) = {t['value']}")
        result = "\n".join(lines)

    elif name == "map_clear":
        m.clear()
        save_map(ctx, m)
        result = "Map cleared. Metadata preserved."

    elif name == "map_snapshot":
        snap = m.snapshot(arguments["label"])
        save_map(ctx, m)
        result = f"Snapshot '{snap.label}' saved ({snap.width}x{snap.height})."

    else:
        result = f"Unknown map tool: {name}"

    return result
