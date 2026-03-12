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
# Map → HTML (no external dependencies)
# ---------------------------------------------------------------------------

_MAP_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ margin: 0; background: #1a1a2e; color: #e0e0e0; font-family: sans-serif;
         display: flex; flex-direction: column; align-items: center; padding: 20px; }}
  h1 {{ font-size: 16px; margin: 0 0 4px; color: #e94560; }}
  .description {{ font-size: 12px; color: #888; margin-bottom: 12px; }}
  .map-container {{ position: relative; display: inline-block; }}
  canvas {{ border: 1px solid #0f3460; cursor: crosshair; }}
  .axis-label {{ font-size: 12px; color: #aaa; }}
  .x-label {{ text-align: center; margin-top: 6px; }}
  .y-label {{ writing-mode: vertical-rl; transform: rotate(180deg);
              position: absolute; left: -24px; top: 50%; transform-origin: center;
              transform: rotate(180deg) translateX(50%); }}
  .tooltip {{
    position: fixed; background: #16213e; border: 1px solid #0f3460;
    padding: 4px 8px; border-radius: 4px; font-size: 11px;
    pointer-events: none; opacity: 0; transition: opacity 0.1s;
    white-space: nowrap;
  }}
  .legend {{ display: flex; align-items: center; margin-top: 10px; font-size: 11px; gap: 6px; }}
  .legend canvas {{ border: none; cursor: default; }}
  .snapshots {{ margin-top: 16px; display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; }}
  .snapshots button {{
    background: #16213e; color: #ccc; border: 1px solid #0f3460;
    padding: 4px 10px; border-radius: 3px; cursor: pointer; font-size: 11px;
  }}
  .snapshots button:hover {{ background: #0f3460; }}
  .snapshots button.active {{ background: #e94560; border-color: #e94560; color: #fff; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="description">{description}</div>
<div class="map-container">
  <div class="y-label axis-label">{y_label}</div>
  <canvas id="map" width="{canvas_w}" height="{canvas_h}"></canvas>
  <div class="x-label axis-label">{x_label}</div>
</div>
<div class="legend">
  <span>-1.0</span>
  <canvas id="legend" width="200" height="14"></canvas>
  <span>+1.0</span>
</div>
<div class="tooltip" id="tooltip"></div>
<div class="snapshots" id="snapshots"></div>
<script>
const gridData = {grid_json};
const snapshots = {snapshots_json};
const W = {grid_w}, H = {grid_h};
const cellSize = {cell_size};
const canvas = document.getElementById("map");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");

function valueToColor(v) {{
  // Diverging: blue (-1) → white (0) → red (+1)
  v = Math.max(-1, Math.min(1, v));
  let r, g, b;
  if (v < 0) {{
    const t = 1 + v; // 0..1
    r = Math.round(59 + t * 196);
    g = Math.round(76 + t * 179);
    b = Math.round(192 + t * 63);
  }} else {{
    const t = v;
    r = 255;
    g = Math.round(255 - t * 162);
    b = Math.round(255 - t * 168);
  }}
  return `rgb(${{r}},${{g}},${{b}})`;
}}

function drawGrid(grid) {{
  for (let y = 0; y < H; y++) {{
    for (let x = 0; x < W; x++) {{
      const val = grid[H - 1 - y][x]; // flip Y so 0 is bottom
      ctx.fillStyle = valueToColor(val);
      ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
    }}
  }}
  // Grid lines
  ctx.strokeStyle = "rgba(255,255,255,0.06)";
  ctx.lineWidth = 0.5;
  for (let x = 0; x <= W; x++) {{
    ctx.beginPath(); ctx.moveTo(x * cellSize, 0); ctx.lineTo(x * cellSize, H * cellSize); ctx.stroke();
  }}
  for (let y = 0; y <= H; y++) {{
    ctx.beginPath(); ctx.moveTo(0, y * cellSize); ctx.lineTo(W * cellSize, y * cellSize); ctx.stroke();
  }}
}}

drawGrid(gridData);

// Legend bar
const lctx = document.getElementById("legend").getContext("2d");
for (let i = 0; i < 200; i++) {{
  lctx.fillStyle = valueToColor((i / 199) * 2 - 1);
  lctx.fillRect(i, 0, 1, 14);
}}

// Tooltip on hover
canvas.addEventListener("mousemove", (e) => {{
  const rect = canvas.getBoundingClientRect();
  const cx = Math.floor((e.clientX - rect.left) / cellSize);
  const cy = Math.floor((e.clientY - rect.top) / cellSize);
  if (cx >= 0 && cx < W && cy >= 0 && cy < H) {{
    const gy = H - 1 - cy;
    const val = gridData[gy][cx];
    tooltip.style.opacity = 1;
    tooltip.innerHTML = `(${{cx}}, ${{gy}}) = ${{val.toFixed(3)}}`;
    tooltip.style.left = (e.clientX + 14) + "px";
    tooltip.style.top = (e.clientY - 10) + "px";
  }} else {{
    tooltip.style.opacity = 0;
  }}
}});
canvas.addEventListener("mouseleave", () => {{ tooltip.style.opacity = 0; }});

// Snapshot buttons
if (snapshots.length > 0) {{
  const container = document.getElementById("snapshots");
  const currentBtn = document.createElement("button");
  currentBtn.textContent = "Current";
  currentBtn.className = "active";
  currentBtn.onclick = () => {{
    drawGrid(gridData);
    container.querySelectorAll("button").forEach(b => b.className = "");
    currentBtn.className = "active";
  }};
  container.appendChild(currentBtn);
  snapshots.forEach((snap, i) => {{
    const btn = document.createElement("button");
    btn.textContent = snap.label || ("Snapshot " + (i + 1));
    btn.onclick = () => {{
      drawGrid(snap.grid);
      container.querySelectorAll("button").forEach(b => b.className = "");
      btn.className = "active";
    }};
    container.appendChild(btn);
  }});
}}
</script>
</body>
</html>
"""


def render_map_html(
    m: ActivationMap,
    title: str = "Activation Map",
    cell_size: int = 0,
) -> str:
    """Render an ActivationMap as self-contained HTML with canvas heatmap.

    No external dependencies required. Returns the HTML string.
    """
    # Auto-size cells: aim for ~400-600px canvas
    if cell_size <= 0:
        cell_size = max(4, min(32, 512 // max(m.width, m.height)))

    canvas_w = m.width * cell_size
    canvas_h = m.height * cell_size

    # Prepare snapshot data (only include grid + label)
    snap_data = []
    for snap in (m.snapshots or []):
        if isinstance(snap, dict) and "grid" in snap:
            snap_data.append({"label": snap.get("label", ""), "grid": snap["grid"]})

    description = m.description or f"{m.width}x{m.height} grid"

    return _MAP_HTML_TEMPLATE.format(
        title=html.escape(title),
        description=html.escape(description),
        x_label=html.escape(m.x_label or "X"),
        y_label=html.escape(m.y_label or "Y"),
        grid_json=json.dumps(m.grid),
        snapshots_json=json.dumps(snap_data),
        grid_w=m.width,
        grid_h=m.height,
        cell_size=cell_size,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
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


# ---------------------------------------------------------------------------
# Creations → Gallery HTML
# ---------------------------------------------------------------------------

_ABCJS_CDN = "https://cdn.jsdelivr.net/npm/abcjs@6/dist/abcjs-basic-min.js"

_GALLERY_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="{abcjs_src}"></script>
<style>
  body {{ margin: 0; background: #1a1a2e; color: #e0e0e0; font-family: sans-serif; padding: 24px; }}
  h1 {{ font-size: 20px; color: #e94560; margin: 0 0 8px; }}
  .subtitle {{ font-size: 13px; color: #888; margin-bottom: 24px; }}
  .creation {{ background: #16213e; border: 1px solid #0f3460; border-radius: 6px;
               padding: 16px; margin-bottom: 16px; }}
  .creation h2 {{ font-size: 16px; color: #e94560; margin: 0 0 4px; }}
  .creation .meta {{ font-size: 11px; color: #666; margin-bottom: 12px; }}
  .creation .body {{ font-size: 14px; line-height: 1.6; }}
  .creation .body p {{ margin: 0.5em 0; }}
  .creation svg {{ max-width: 100%; height: auto; background: #0d1117; border-radius: 4px; padding: 8px; }}
  .abc-render {{ background: #f8f8f8; border-radius: 4px; padding: 8px; }}
  .game-link {{ display: inline-block; background: #0f3460; color: #e0e0e0; padding: 8px 16px;
                border-radius: 4px; text-decoration: none; margin-top: 8px; }}
  .game-link:hover {{ background: #e94560; }}
  .type-badge {{ display: inline-block; background: #0f3460; color: #aaa; font-size: 10px;
                 padding: 2px 6px; border-radius: 3px; margin-left: 8px; text-transform: uppercase; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="subtitle">{subtitle}</div>
{entries}
<script>
document.querySelectorAll('.abc-source').forEach(function(el) {{
  var target = el.nextElementSibling;
  if (typeof ABCJS !== 'undefined' && target) {{
    ABCJS.renderAbc(target, el.textContent.trim());
  }}
}});
</script>
</body>
</html>
"""


def _simple_md_to_html(text: str) -> str:
    """Minimal markdown-to-HTML for rendering narrative/poetry in the gallery."""
    lines = text.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            result.append("")
            continue
        # Headers
        if stripped.startswith("### "):
            result.append(f"<h4>{html.escape(stripped[4:])}</h4>")
        elif stripped.startswith("## "):
            result.append(f"<h3>{html.escape(stripped[3:])}</h3>")
        elif stripped.startswith("# "):
            result.append(f"<h2>{html.escape(stripped[2:])}</h2>")
        else:
            result.append(html.escape(stripped))

    # Join and wrap non-heading lines in paragraphs
    output = []
    paragraph: list[str] = []
    for line in result:
        if line.startswith("<h"):
            if paragraph:
                output.append("<p>" + "<br>\n".join(paragraph) + "</p>")
                paragraph = []
            output.append(line)
        elif line == "":
            if paragraph:
                output.append("<p>" + "<br>\n".join(paragraph) + "</p>")
                paragraph = []
        else:
            paragraph.append(line)
    if paragraph:
        output.append("<p>" + "<br>\n".join(paragraph) + "</p>")

    return "\n".join(output)


def render_creations_gallery_html(
    creations: list[dict],
    title: str = "Creations",
) -> str:
    """Render a list of creations as a self-contained HTML gallery.

    Args:
        creations: List of dicts with keys: title, type, body, slug, updated.
                   body is the raw content (SVG, markdown, ABC, or HTML).
        title: Gallery page title.

    Returns:
        Self-contained HTML string.
    """
    entries_html = []

    for c in creations:
        c_type = c.get("type", "unknown")
        c_title = html.escape(c.get("title", "Untitled"))
        c_slug = html.escape(c.get("slug", ""))
        c_updated = c.get("updated", "")
        body = c.get("body", "")

        meta_parts = [f'<span class="type-badge">{html.escape(c_type)}</span>']
        if c_updated:
            meta_parts.append(f'updated: {html.escape(str(c_updated))}')
        meta_line = ' '.join(meta_parts)

        if c_type == "art":
            # SVG — inline directly (already sanitized on creation), plus link to raw file
            body_html = (
                body
                + f'\n<p><a class="game-link" href="creations/{c_slug}.svg" target="_blank">'
                f'View SVG</a></p>'
            )
        elif c_type in ("narrative", "poetry"):
            body_html = _simple_md_to_html(body)
        elif c_type == "music":
            # ABC — hidden source + render target for abcjs
            escaped_abc = html.escape(body)
            body_html = (
                f'<pre class="abc-source" style="display:none">{escaped_abc}</pre>\n'
                f'<div class="abc-render"></div>'
            )
        elif c_type == "game":
            # HTML game — published as individual file, link to it
            body_html = (
                f'<a class="game-link" href="creations/{c_slug}.html" target="_blank">'
                f'Open interactive experience</a>'
            )
        else:
            body_html = f"<pre>{html.escape(body[:2000])}</pre>"

        entry = (
            f'<div class="creation">\n'
            f'  <h2>{c_title}</h2>\n'
            f'  <div class="meta">{meta_line}</div>\n'
            f'  <div class="body">{body_html}</div>\n'
            f'</div>'
        )
        entries_html.append(entry)

    subtitle = f"{len(creations)} work{'s' if len(creations) != 1 else ''}"

    return _GALLERY_HTML_TEMPLATE.format(
        title=html.escape(title),
        subtitle=html.escape(subtitle),
        abcjs_src=_ABCJS_CDN,
        entries="\n".join(entries_html),
    )
