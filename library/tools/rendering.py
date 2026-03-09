"""Rendering utilities for semantic graph and activation map.

Graph → self-contained HTML with D3.js force-directed layout.
Map → PNG heatmap or animated GIF (requires matplotlib).

Both renderers accept data objects and return bytes/str — no ctx dependency.
"""

from __future__ import annotations

import html
import io
import json
import math
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from library.tools.graph import SemanticGraph
    from library.tools.activation_map import ActivationMap


# ---------------------------------------------------------------------------
# Graph → HTML
# ---------------------------------------------------------------------------

_D3_CDN = "https://d3js.org/d3.v7.min.js"

_GRAPH_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="{d3_src}"></script>
<style>
  body {{ margin: 0; background: #1a1a2e; color: #e0e0e0; font-family: sans-serif; }}
  svg {{ width: 100vw; height: 100vh; }}
  .node circle {{ stroke: #fff; stroke-width: 1.5px; cursor: pointer; }}
  .node text {{ font-size: 11px; fill: #ccc; pointer-events: none; }}
  .link {{ stroke-opacity: 0.6; }}
  .link-label {{ font-size: 9px; fill: #888; pointer-events: none; }}
  .tooltip {{
    position: absolute; background: #16213e; border: 1px solid #0f3460;
    padding: 6px 10px; border-radius: 4px; font-size: 12px;
    pointer-events: none; opacity: 0; transition: opacity 0.15s;
  }}
  h1 {{ position: absolute; top: 10px; left: 16px; font-size: 16px; margin: 0; color: #e94560; }}
  .stats {{ position: absolute; top: 36px; left: 16px; font-size: 12px; color: #888; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="stats">{stats}</div>
<div class="tooltip" id="tooltip"></div>
<svg></svg>
<script>
const nodes = {nodes_json};
const links = {links_json};

const svg = d3.select("svg");
const width = window.innerWidth;
const height = window.innerHeight;
svg.attr("viewBox", [0, 0, width, height]);

const simulation = d3.forceSimulation(nodes)
  .force("link", d3.forceLink(links).id(d => d.id).distance(100))
  .force("charge", d3.forceManyBody().strength(-200))
  .force("center", d3.forceCenter(width / 2, height / 2))
  .force("collision", d3.forceCollide().radius(d => d.radius + 4));

const link = svg.append("g")
  .selectAll("line")
  .data(links)
  .join("line")
  .attr("class", "link")
  .attr("stroke", "#0f3460")
  .attr("stroke-width", d => Math.max(1, d.weight * 4));

const linkLabel = svg.append("g")
  .selectAll("text")
  .data(links)
  .join("text")
  .attr("class", "link-label")
  .text(d => d.relation);

const tooltip = d3.select("#tooltip");

const node = svg.append("g")
  .selectAll("g")
  .data(nodes)
  .join("g")
  .attr("class", "node")
  .call(d3.drag()
    .on("start", (e, d) => {{ if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }})
    .on("drag", (e, d) => {{ d.fx = e.x; d.fy = e.y; }})
    .on("end", (e, d) => {{ if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }}))
  .on("mouseover", (e, d) => {{
    tooltip.style("opacity", 1)
      .html("<strong>" + d.label + "</strong><br>ID: " + d.id + "<br>Degree: " + d.degree
        + (d.meta ? "<br>" + d.meta : ""));
  }})
  .on("mousemove", (e) => {{
    tooltip.style("left", (e.pageX + 12) + "px").style("top", (e.pageY - 20) + "px");
  }})
  .on("mouseout", () => {{ tooltip.style("opacity", 0); }});

node.append("circle")
  .attr("r", d => d.radius)
  .attr("fill", d => d.color);

node.append("text")
  .attr("dx", d => d.radius + 4)
  .attr("dy", "0.35em")
  .text(d => d.label);

simulation.on("tick", () => {{
  link
    .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
    .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
  linkLabel
    .attr("x", d => (d.source.x + d.target.x) / 2)
    .attr("y", d => (d.source.y + d.target.y) / 2);
  node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
}});
</script>
</body>
</html>
"""

# Color palette for node types (hue-shifted warm tones).
_NODE_COLORS = [
    "#e94560", "#0f3460", "#533483", "#16c79a",
    "#f5a623", "#50d890", "#6c5ce7", "#fd79a8",
]


def _node_color(metadata: dict, index: int) -> str:
    """Pick a color from metadata type or fallback to palette rotation."""
    node_type = metadata.get("type", "")
    if node_type:
        return _NODE_COLORS[hash(node_type) % len(_NODE_COLORS)]
    return _NODE_COLORS[index % len(_NODE_COLORS)]


def render_graph_html(
    graph: SemanticGraph,
    title: str = "Semantic Graph",
) -> str:
    """Render a SemanticGraph as self-contained HTML with D3.js force layout.

    Returns the HTML string. Write to a file for viewing.
    """
    # Compute degrees for sizing
    degrees: dict[str, int] = defaultdict(int)
    for e in graph.edges:
        degrees[e.source] += 1
        degrees[e.target] += 1

    max_degree = max(degrees.values()) if degrees else 1

    # Build D3 node data
    d3_nodes = []
    for i, node in enumerate(graph.nodes.values()):
        deg = degrees.get(node.id, 0)
        radius = 6 + (deg / max_degree) * 18 if max_degree > 0 else 8
        meta_parts = []
        for k, v in node.metadata.items():
            meta_parts.append(f"{k}: {v}")
        d3_nodes.append({
            "id": node.id,
            "label": node.label,
            "degree": deg,
            "radius": round(radius, 1),
            "color": _node_color(node.metadata, i),
            "meta": ", ".join(meta_parts) if meta_parts else "",
        })

    # Build D3 link data — only include edges where both endpoints exist
    node_ids = set(graph.nodes.keys())
    d3_links = []
    for e in graph.edges:
        if e.source in node_ids and e.target in node_ids:
            d3_links.append({
                "source": e.source,
                "target": e.target,
                "relation": e.relation,
                "weight": e.weight,
            })

    stats = f"{len(graph.nodes)} nodes, {len(graph.edges)} edges"
    if graph.snapshots:
        stats += f", {len(graph.snapshots)} snapshots"

    return _GRAPH_HTML_TEMPLATE.format(
        title=html.escape(title),
        d3_src=_D3_CDN,
        stats=html.escape(stats),
        nodes_json=json.dumps(d3_nodes),
        links_json=json.dumps(d3_links),
    )


# ---------------------------------------------------------------------------
# Map → PNG / GIF (requires matplotlib)
# ---------------------------------------------------------------------------


def _check_matplotlib():
    """Import and return matplotlib modules, raising ImportError if missing."""
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    return plt, mcolors


def render_map_png(
    m: ActivationMap,
    *,
    contour_levels: int = 5,
    figsize: tuple[float, float] = (8, 6),
) -> bytes:
    """Render an ActivationMap as a PNG heatmap. Returns PNG bytes.

    Requires matplotlib. Raises ImportError if not installed.
    """
    plt, mcolors = _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    # Diverging colormap: cool (negative) → neutral (zero) → warm (positive)
    cmap = plt.cm.RdBu_r  # Red=positive, Blue=negative
    norm = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    im = ax.imshow(
        m.grid,
        cmap=cmap,
        norm=norm,
        origin="lower",
        aspect="auto",
        interpolation="bilinear",
    )

    # Contour lines
    if contour_levels > 0 and any(v != 0.0 for row in m.grid for v in row):
        try:
            ax.contour(
                m.grid,
                levels=contour_levels,
                colors="black",
                alpha=0.3,
                linewidths=0.5,
                origin="lower",
            )
        except ValueError:
            pass  # All-zero or flat grid — no contours possible

    ax.set_xlabel(m.x_label or "X")
    ax.set_ylabel(m.y_label or "Y")

    title = "Activation Map"
    if m.description:
        title += f"\n{m.description}"
    ax.set_title(title, fontsize=10)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Activation")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def render_map_gif(
    grids: list[tuple[str, list[list[float]]]],
    *,
    width: int = 16,
    height: int = 16,
    x_label: str = "",
    y_label: str = "",
    frame_duration_ms: int = 500,
    figsize: tuple[float, float] = (6, 5),
) -> bytes:
    """Render a sequence of grids as an animated GIF. Returns GIF bytes.

    Args:
        grids: List of (label, grid_2d_array) tuples — one per frame.
        width, height: Grid dimensions (for axis sizing).
        x_label, y_label: Axis labels.
        frame_duration_ms: Duration per frame in milliseconds.

    Requires matplotlib and Pillow. Raises ImportError if not installed.
    """
    plt, mcolors = _check_matplotlib()
    from PIL import Image

    if not grids:
        raise ValueError("Need at least one grid to render")

    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    frames: list[Image.Image] = []
    for label, grid in grids:
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(
            grid, cmap=cmap, norm=norm,
            origin="lower", aspect="auto", interpolation="bilinear",
        )
        ax.set_xlabel(x_label or "X")
        ax.set_ylabel(y_label or "Y")
        ax.set_title(label, fontsize=10)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())

    # Assemble GIF
    out = io.BytesIO()
    frames[0].save(
        out,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
    )
    out.seek(0)
    return out.read()
