"""Publish rendered artefacts to the data repository.

Handles both agent-initiated publishes (via tool calls) and automatic
post-heartbeat rendering (graph HTML, map PNG/GIF).

Published files go to a separate `_published/` directory within the
instance's data repo space, keeping them distinct from memory files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from library.harness.context import InstanceContext

logger = logging.getLogger(__name__)


def _get_publish_dir(ctx: InstanceContext) -> Path | None:
    """Resolve the publish directory in the data repo for this instance.

    Returns None if sync is not configured.
    """
    from library.harness.config import load_harness_config

    # Access harness config via ctx internals
    sync_config = getattr(ctx, "_sync_config", None)
    if sync_config is None:
        return None

    repo = sync_config.repo
    if not isinstance(repo, str) or not repo:
        return None
    repo_path = Path(repo)

    prefix = sync_config.prefix or ""
    base = repo_path / prefix if prefix else repo_path
    return base / ctx.instance_id / "_published"


def publish_file(ctx: InstanceContext, path: str, content: str | bytes) -> str:
    """Publish a file to the data repo under _published/.

    Args:
        path: Relative path within the published directory (e.g. "report.md")
        content: String or bytes content

    Returns a status message.
    """
    pub_dir = _get_publish_dir(ctx)
    if pub_dir is None:
        # Fallback: write to instance memory under _published/
        pub_path = f"_published/{path}"
        if isinstance(content, bytes):
            ctx.write_binary(pub_path, content)
        else:
            ctx.write(pub_path, content)
        return f"Published to memory: {pub_path}"

    dest = pub_dir / path
    dest.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(content, bytes):
        dest.write_bytes(content)
    else:
        dest.write_text(content)

    logger.info("Published %s (%d bytes)", dest, len(content))
    return f"Published: {path}"


def render_and_publish(ctx: InstanceContext) -> None:
    """Render graph and activation map, publish to data repo.

    Called after heartbeat completes. Renders whatever representations
    exist (graph, map) and writes them to the publish directory.
    """
    graph_json = ctx.read("graph.json")
    map_json = ctx.read("activation_map.json")

    if not graph_json and not map_json:
        return

    published = []

    # Render graph
    if graph_json:
        try:
            from library.tools.graph import SemanticGraph
            from library.tools.rendering import render_graph_html

            graph = SemanticGraph.from_json(graph_json)
            if graph.nodes:
                html = render_graph_html(graph, title=f"{ctx.instance_id} — Semantic Graph")
                publish_file(ctx, "graph.html", html)
                published.append("graph.html")
        except Exception as exc:
            logger.warning("Failed to render graph: %s", exc)

    # Render activation map
    if map_json:
        try:
            from library.tools.activation_map import ActivationMap
            from library.tools.rendering import render_map_png

            m = ActivationMap.from_json(map_json)
            if m.width > 0:
                png = render_map_png(m)
                publish_file(ctx, "map.png", png)
                published.append("map.png")

                # Render GIF from snapshots if available
                if m.snapshots:
                    from library.tools.rendering import render_map_gif
                    grids = [(s.get("label", ""), s["grid"]) for s in m.snapshots if "grid" in s]
                    if grids:
                        gif = render_map_gif(grids)
                        publish_file(ctx, "map_session.gif", gif)
                        published.append("map_session.gif")
        except ImportError:
            logger.debug("Skipping map render (matplotlib/Pillow not installed)")
        except Exception as exc:
            logger.warning("Failed to render map: %s", exc)

    if published:
        logger.info("Rendered and published: %s", ", ".join(published))
