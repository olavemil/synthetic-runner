"""Neural Dreamer species — NN-driven extension of Subconscious Dreamer.

Two cycles:
  Slow (heartbeat): think → [organize] → subconscious → [dream] → sleep
    - Think can iterate (controlled by processing_depth)
    - Organize is probabilistic (controlled by organization_drive)
    - Dream is probabilistic (controlled by creative_latitude)
  Fast (on_message): gut → suggest → [reply] → review → [extra think]
    - Reply is probabilistic (controlled by reply_willingness × reply_value)
    - Extra think is probabilistic (controlled by processing_depth)

When neural nets are available, they produce segment selection weights,
variable values, and shared behavioral parameters. Falls back to manual
defaults when PyTorch is not installed or nets haven't been initialised yet.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, TYPE_CHECKING

from library.species import Species, SpeciesManifest, EntryPoint
from library.tools.decisions import probabilistic
from library.tools.pipeline import run_pipeline, load_pipeline
from library.tools.prompts import format_events, get_entity_id
from library.tools.segments import (
    SegmentRegistry,
    load_registry,
    build_prompt,
    render_prompt,
    select_segments,
    DEFAULT_VARIABLES,
)

if TYPE_CHECKING:
    from library.harness.adapters import Event
    from library.harness.context import InstanceContext

logger = logging.getLogger(__name__)

_SPECIES_DIR = Path(__file__).parent
_HEARTBEAT_STEPS = load_pipeline((_SPECIES_DIR / "heartbeat.yaml").read_text())["steps"]
_ON_MESSAGE_STEPS = load_pipeline((_SPECIES_DIR / "on_message.yaml").read_text())["steps"]
_REGISTRY = load_registry(_SPECIES_DIR / "segments" / "registry.yaml")

DEFAULT_FILES = {
    "thinking.md": "# Thinking\n",
    "dreams.md": "",
    "concerns.md": "",
    "sleep.md": "",
    "last_review.md": "",
    "reviews.md": "",
    "segment_weights.json": "",
}

# Default segment weights — used when NN is not available.
_DEFAULT_WEIGHTS: dict[str, float] = {
    "identity-core": 0.9,
    "state-reflective": 0.6,
    "relational-warm": 0.5,
    "task-responsive": 0.7,
    "temporal-present": 0.6,
    "meta-concise": 0.5,
}

# Segment IDs controlled by each net
_FAST_SEGMENT_IDS = [
    sid for sid in sorted(_REGISTRY.all_ids())
    if _REGISTRY.get(sid) and _REGISTRY.get(sid).category in ("state", "relational", "meta")
]
_SLOW_SEGMENT_IDS = [
    sid for sid in sorted(_REGISTRY.all_ids())
    if _REGISTRY.get(sid) and _REGISTRY.get(sid).category in ("identity", "temporal", "task")
]


def _try_nn_weights_and_variables(
    ctx: InstanceContext,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]] | None:
    """Try to get weights and variables from neural nets. Returns None if unavailable.

    Returns (weights, variables, shared_params) or None.

    When graph/map data is available, their summary features are appended to
    the signal vectors — but only if the loaded net's input_dim matches.
    """
    try:
        from library.tools.neural import (
            load_fast_net, load_slow_net,
            make_fast_net_config, make_slow_net_config,
            encode_fast_signals, encode_slow_signals,
            encode_graph_features, encode_map_features,
            decode_fast_output, decode_slow_output,
            blend_shared_parameters,
            FAST_SIGNAL_NAMES, MAP_FEATURE_NAMES,
            SLOW_SIGNAL_NAMES, GRAPH_FEATURE_NAMES,
            SLOW_ACTIVITY_SIGNAL_NAMES,
        )
    except ImportError:
        return None

    try:
        fast_net, fast_meta = load_fast_net(ctx)
        slow_net, slow_meta = load_slow_net(ctx)
    except ImportError:
        return None

    if fast_net is None and slow_net is None:
        logger.info("NN forward: no nets available, falling back to defaults")
        return None

    weights: dict[str, float] = {}
    variables: dict[str, float] = {}
    fast_shared: dict[str, float] = {}
    slow_shared: dict[str, float] = {}

    if fast_net is not None:
        last_review = ctx.read("last_review.md") or ""
        signals = _parse_review_signals(last_review)
        input_vals = encode_fast_signals(signals)

        # Append map features if the net expects them
        has_map = False
        expected_with_map = len(FAST_SIGNAL_NAMES) + len(MAP_FEATURE_NAMES)
        if fast_net.config.input_dim == expected_with_map:
            from library.tools.activation_map import load_map
            m = load_map(ctx)
            input_vals += encode_map_features(m)
            has_map = True

        raw = fast_net.forward(input_vals)
        decoded = decode_fast_output(raw, _FAST_SEGMENT_IDS)
        weights.update(decoded["weights"])
        variables.update(decoded["variables"])
        fast_shared = decoded.get("shared", {})

        top_weights = sorted(decoded["weights"].items(), key=lambda x: x[1], reverse=True)[:3]
        top_str = ", ".join(f"{k}={v:.2f}" for k, v in top_weights)
        var_str = ", ".join(f"{k}={v:.2f}" for k, v in decoded["variables"].items())
        logger.info(
            "Fast net forward (updates=%d, signals=%d, map=%s): top_weights=[%s] vars=[%s]",
            fast_meta.update_count, len(signals), has_map, top_str, var_str,
        )
    else:
        logger.info("Fast net not available, using default state/relational/meta weights")

    if slow_net is not None:
        sleep = ctx.read("sleep.md") or ""
        signals = _parse_sleep_signals(sleep)
        input_vals = encode_slow_signals(signals)

        # Append graph and/or activity features if the net expects them
        has_graph = False
        has_activity = False
        base_plus_graph = len(SLOW_SIGNAL_NAMES) + len(GRAPH_FEATURE_NAMES)
        base_plus_graph_plus_activity = base_plus_graph + len(SLOW_ACTIVITY_SIGNAL_NAMES)
        if slow_net.config.input_dim in (base_plus_graph, base_plus_graph_plus_activity):
            from library.tools.graph import load_graph
            graph = load_graph(ctx)
            input_vals += encode_graph_features(graph)
            has_graph = True
        if slow_net.config.input_dim == base_plus_graph_plus_activity:
            # Pad with zeros — no session metrics during forward inference
            input_vals += [0.0] * len(SLOW_ACTIVITY_SIGNAL_NAMES)
            has_activity = True

        raw = slow_net.forward(input_vals)
        decoded = decode_slow_output(raw, _SLOW_SEGMENT_IDS)
        weights.update(decoded["weights"])
        variables.update(decoded["variables"])
        slow_shared = decoded.get("shared", {})

        top_weights = sorted(decoded["weights"].items(), key=lambda x: x[1], reverse=True)[:3]
        top_str = ", ".join(f"{k}={v:.2f}" for k, v in top_weights)
        var_str = ", ".join(f"{k}={v:.2f}" for k, v in decoded["variables"].items())
        logger.info(
            "Slow net forward (updates=%d, signals=%d, graph=%s, activity=%s): "
            "top_weights=[%s] vars=[%s]",
            slow_meta.update_count, len(signals), has_graph, has_activity,
            top_str, var_str,
        )
    else:
        logger.info("Slow net not available, using default identity/temporal/task weights")

    shared_params = blend_shared_parameters(fast_shared, slow_shared)
    shared_params["_nn_available"] = True
    return weights, variables, shared_params


def _load_weights_and_variables(
    ctx: InstanceContext,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Load weights, variables, and shared params. Prefers NN output, falls back to defaults.

    Returns (weights, variables, shared_params). When NN is not available,
    shared_params will have ``_nn_available`` set to False.
    """
    default_shared: dict[str, float] = {"_nn_available": False}
    nn_result = _try_nn_weights_and_variables(ctx)
    if nn_result is not None:
        weights, variables, shared_params = nn_result
        merged_weights = dict(_DEFAULT_WEIGHTS)
        merged_weights.update(weights)
        merged_variables = dict(DEFAULT_VARIABLES)
        merged_variables.update(variables)
        logger.info("Weights source: neural nets (%d weights, %d variables)", len(merged_weights), len(merged_variables))
        return merged_weights, merged_variables, shared_params

    # Fall back to manual weights from storage or defaults
    raw = ctx.read("segment_weights.json")
    if raw:
        try:
            weights = json.loads(raw)
            logger.info("Weights source: segment_weights.json (%d weights)", len(weights))
            return weights, dict(DEFAULT_VARIABLES), dict(default_shared)
        except (json.JSONDecodeError, TypeError):
            pass
    logger.info("Weights source: hardcoded defaults")
    return dict(_DEFAULT_WEIGHTS), dict(DEFAULT_VARIABLES), dict(default_shared)


def _parse_review_signals(review_text: str) -> dict[str, float]:
    """Parse structured review output into signal dict."""
    signals: dict[str, float] = {}
    for line in review_text.split("\n"):
        match = re.match(r"(\w+):\s*([-\d.]+)", line.strip())
        if match:
            name, value = match.group(1), match.group(2)
            try:
                signals[name] = float(value)
            except ValueError:
                pass
    return signals


def _parse_sleep_signals(sleep_text: str) -> dict[str, float]:
    """Extract signals from sleep output. Currently uses simple heuristics."""
    # Sleep output is free-form text. We extract what we can.
    signals: dict[str, float] = {}
    # Look for explicit signal lines (same format as review)
    for line in sleep_text.split("\n"):
        match = re.match(r"(\w+):\s*([-\d.]+)", line.strip())
        if match:
            name, value = match.group(1), match.group(2)
            try:
                signals[name] = float(value)
            except ValueError:
                pass
    return signals


def _update_fast_net(ctx: InstanceContext, review_text: str) -> None:
    """Parse review signals and update fast net.

    New nets are created with map features enabled. Existing nets keep
    their original input_dim — map features are appended only when dims match.
    """
    try:
        from library.tools.neural import (
            load_fast_net, save_fast_net, make_fast_net_config,
            encode_fast_signals, encode_map_features,
            FAST_SIGNAL_NAMES, MAP_FEATURE_NAMES,
            CheckpointMeta, record_checkpoint,
        )
    except ImportError:
        return

    signals = _parse_review_signals(review_text)
    if not signals:
        logger.debug("Fast net update skipped: no signals parsed from review")
        return

    sig_str = ", ".join(f"{k}={v:.2f}" for k, v in signals.items())
    logger.info("Fast net update: parsed signals [%s]", sig_str)

    config = make_fast_net_config(len(_FAST_SEGMENT_IDS), include_map_features=True)
    try:
        fast_net, meta = load_fast_net(ctx, fallback_config=config)
    except ImportError:
        return
    created_new = False
    if fast_net is None:
        fast_net = _create_net_with_config(config)
        if fast_net is None:
            logger.warning("Fast net update: PyTorch unavailable, cannot create net")
            return
        created_new = True
        logger.info("Fast net created (input_dim=%d, hidden=%d, output_dim=%d)",
                     config.input_dim, config.hidden_dim, config.output_dim)

    input_vals = encode_fast_signals(signals)

    # Append map features if the net expects them
    expected_with_map = len(FAST_SIGNAL_NAMES) + len(MAP_FEATURE_NAMES)
    if fast_net.config.input_dim == expected_with_map:
        from library.tools.activation_map import load_map
        input_vals += encode_map_features(load_map(ctx))

    current_output = fast_net.forward(input_vals)
    target = _nudge_output(current_output, signals, _FAST_SEGMENT_IDS)

    loss = fast_net.train_step(input_vals, target)

    meta.update_count += 1
    save_fast_net(ctx, fast_net, meta)
    record_checkpoint(ctx, f"fast_update_{meta.update_count}", "fast")
    logger.info("Fast net trained: update_count=%d, loss=%.6f%s",
                meta.update_count, loss, " (new net)" if created_new else "")


def _update_slow_net(
    ctx: InstanceContext, sleep_text: str, session_label: str,
    metrics: Any = None,
) -> None:
    """Parse sleep signals and update slow net.

    New nets are created with graph + activity features enabled. Existing nets
    keep their original input_dim — extra features are appended only when dims match.
    """
    try:
        from library.tools.neural import (
            load_slow_net, save_slow_net, make_slow_net_config,
            encode_slow_signals, encode_graph_features, encode_activity_signals,
            SLOW_SIGNAL_NAMES, GRAPH_FEATURE_NAMES, SLOW_ACTIVITY_SIGNAL_NAMES,
            CheckpointMeta, record_checkpoint,
        )
    except ImportError:
        return

    signals = _parse_sleep_signals(sleep_text)
    if not signals:
        logger.debug("Slow net update skipped: no signals parsed from sleep output")
        return

    sig_str = ", ".join(f"{k}={v:.2f}" for k, v in signals.items())
    logger.info("Slow net update: parsed signals [%s] (session=%s)", sig_str, session_label)

    config = make_slow_net_config(
        len(_SLOW_SEGMENT_IDS),
        include_graph_features=True,
        include_activity_signals=metrics is not None,
    )
    try:
        slow_net, meta = load_slow_net(ctx, fallback_config=config)
    except ImportError:
        return
    created_new = False
    if slow_net is None:
        slow_net = _create_net_with_config(config)
        if slow_net is None:
            logger.warning("Slow net update: PyTorch unavailable, cannot create net")
            return
        created_new = True
        logger.info("Slow net created (input_dim=%d, hidden=%d, output_dim=%d)",
                     config.input_dim, config.hidden_dim, config.output_dim)

    input_vals = encode_slow_signals(signals)

    # Append graph features if the net expects them
    base_plus_graph = len(SLOW_SIGNAL_NAMES) + len(GRAPH_FEATURE_NAMES)
    base_plus_graph_plus_activity = base_plus_graph + len(SLOW_ACTIVITY_SIGNAL_NAMES)
    if slow_net.config.input_dim in (base_plus_graph, base_plus_graph_plus_activity):
        from library.tools.graph import load_graph
        input_vals += encode_graph_features(load_graph(ctx))

    # Append activity signals if the net expects them
    if slow_net.config.input_dim == base_plus_graph_plus_activity and metrics is not None:
        input_vals += encode_activity_signals(metrics)

    current_output = slow_net.forward(input_vals)
    target = _nudge_output(current_output, signals, _SLOW_SEGMENT_IDS)

    loss = slow_net.train_step(input_vals, target)

    meta.update_count += 1
    meta.session_label = session_label
    save_slow_net(ctx, slow_net, meta)
    record_checkpoint(ctx, session_label, "slow")
    logger.info("Slow net trained: update_count=%d, loss=%.6f, session=%s%s",
                meta.update_count, loss, session_label, " (new net)" if created_new else "")


def _create_net_with_config(config: Any) -> Any:
    """Create a Net instance, returning None if PyTorch unavailable."""
    try:
        from library.tools.neural import Net
        return Net(config)
    except ImportError:
        return None


def _nudge_output(
    current: list[float],
    signals: dict[str, float],
    segment_ids: list[str],
) -> list[float]:
    """Create a training target by nudging current output toward signal-derived targets.

    Positive signals (success, coherence) reinforce the current output.
    Negative signals (unresolved, surprise) push away from it.
    """
    target = list(current)
    n_segments = len(segment_ids)

    # Extract key signals
    success = signals.get("success", 0.5)
    coherence = signals.get("coherence", 0.5)
    # For slow net signals
    session_coherence = signals.get("session_coherence", 0.5)
    intention_alignment = signals.get("intention_alignment", 0.5)

    # Reinforcement factor: high success/coherence → keep current output.
    # Low success/coherence → push toward 0.5 (neutral).
    reinforce = max(success, coherence, session_coherence, intention_alignment, 0.5)
    drift_toward_neutral = 1.0 - reinforce

    for i in range(len(target)):
        # Blend between current and neutral (0.0 in raw space)
        target[i] = current[i] * (1.0 - drift_toward_neutral * 0.3)

    return target


def _build_context(*sections: tuple[str, str]) -> str:
    """Assemble labeled sections into a context block."""
    parts = []
    for label, text in sections:
        if text and text.strip():
            parts.append(f"## {label}\n\n{text.strip()}")
    return "\n\n".join(parts)


def _inject_segments(template: str, registry: SegmentRegistry, weights: dict[str, float],
                     variables: dict[str, float], categories: list[str]) -> str:
    """Inject segment content into prompt template placeholders.

    Replaces {segment_identity}, {segment_state}, etc. with rendered segments
    from the matching categories.
    """
    result = template
    for category in categories:
        placeholder = f"{{segment_{category}}}"
        if placeholder in result:
            cat_weights = {
                sid: w for sid, w in weights.items()
                if registry.get(sid) and registry.get(sid).category == category
            }
            selected = select_segments(registry, cat_weights)
            rendered = render_prompt(selected, variables) if selected else ""
            result = result.replace(placeholder, rendered)
    return result


def _target_room(events: list[Event], ctx: InstanceContext) -> str:
    """Determine the logical space name to reply to."""
    spaces = ctx.list_spaces()
    if spaces:
        for evt in events:
            if evt.room in spaces:
                return evt.room
        return spaces[0]
    return "main"


def _graph_map_summary(ctx: InstanceContext) -> str:
    """Build a summary of graph and map state for context injection."""
    parts = []
    from library.tools.graph import load_graph
    graph = load_graph(ctx)
    desc = graph.describe()
    if desc["node_count"] > 0:
        top = ", ".join(f"{n['label']} ({n['degree']})" for n in desc["top_nodes"][:3])
        parts.append(
            f"Graph: {desc['node_count']} nodes, {desc['edge_count']} edges. "
            f"Top: {top}."
        )

    from library.tools.activation_map import load_map
    m = load_map(ctx)
    if m.x_label:
        md = m.describe()
        parts.append(
            f"Map ({m.x_label} x {m.y_label}): "
            f"{md['active_cells']} active cells, mean {md['mean']}."
        )

    return "\n".join(parts) if parts else ""


# --- Metrics helpers ---

def _snapshot_counts(ctx: InstanceContext) -> dict[str, int]:
    """Snapshot current graph/organize/thinking counts for delta computation."""
    from library.tools.graph import load_graph
    from library.tools.organize import count_all_topics

    graph = load_graph(ctx)
    desc = graph.describe()
    thinking = ctx.read("thinking.md") or ""

    return {
        "graph_nodes": desc["node_count"],
        "graph_edges": desc["edge_count"],
        "topics": count_all_topics(ctx),
        "thinking_chars": len(thinking),
    }


def _build_session_metrics(
    ctx: InstanceContext,
    pre: dict[str, int],
    think_count: int,
) -> "SessionMetrics":
    """Build SessionMetrics by comparing pre-session snapshot to current state."""
    from library.tools.neural import SessionMetrics

    post = _snapshot_counts(ctx)
    return SessionMetrics(
        think_iterations=think_count,
        think_tokens=0,  # not tracked mechanically (would need LLM response accumulation)
        graph_nodes_added=max(0, post["graph_nodes"] - pre["graph_nodes"]),
        graph_edges_added=max(0, post["graph_edges"] - pre["graph_edges"]),
        map_cells_changed=0,  # would need map diff; left as 0 for now
        topics_added=max(0, post["topics"] - pre["topics"]),
        topics_modified=0,  # would need per-topic diff tracking
        thoughts_archived_chars=max(0, pre["thinking_chars"] - post["thinking_chars"]),
        thinking_chars_pre_session=pre["thinking_chars"],
        tool_calls=0,  # not tracked mechanically
        graph_nodes_total=post["graph_nodes"],
        graph_edges_total=post["graph_edges"],
        topics_total=post["topics"],
    )


# --- Phase helpers ---

def _build_heartbeat_phases(shared_params: dict[str, float], config: dict) -> list[str]:
    """Determine which heartbeat phases to run and in what order.

    Uses shared behavioral parameters for probabilistic phase decisions.
    When NN is not available, runs the fixed pipeline (think → subconscious → dream → sleep).
    """
    nn_available = shared_params.get("_nn_available", False)
    if not nn_available:
        return ["think", "subconscious", "dream", "sleep"]

    max_thinks = int(config.get("max_think_iterations", 3))
    phases = ["think"]

    # Additional think iterations with diminishing probability
    processing_depth = shared_params.get("processing_depth", 0.5)
    for i in range(1, max_thinks):
        if probabilistic(processing_depth - 0.3 * i, label=f"extra_think_{i}"):
            phases.append("think")
        else:
            break

    # Optional organize phase
    organization_drive = shared_params.get("organization_drive", 0.5)
    if probabilistic(organization_drive, label="organize_phase"):
        phases.append("organize")

    phases.append("subconscious")

    # Optional dream phase
    creative_latitude = shared_params.get("creative_latitude", 0.5)
    if probabilistic(creative_latitude, label="dream_phase"):
        phases.append("dream")

    phases.append("sleep")
    return phases


def _get_thinking_tools() -> list[dict]:
    """Assemble tool schemas for thinking session (graph + map + organize + publish)."""
    from library.tools.graph import GRAPH_TOOL_SCHEMAS
    from library.tools.activation_map import MAP_TOOL_SCHEMAS
    from library.tools.organize import ORGANIZE_TOOL_SCHEMAS
    publish_schema = {
        "type": "function",
        "function": {
            "name": "publish",
            "description": (
                "Publish a file to the shared data repository, visible externally. "
                "Use for reports, summaries, or creative output you want to share."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path (e.g. 'report.md')"},
                    "content": {"type": "string", "description": "File content"},
                },
                "required": ["path", "content"],
            },
        },
    }
    return GRAPH_TOOL_SCHEMAS + MAP_TOOL_SCHEMAS + ORGANIZE_TOOL_SCHEMAS + [publish_schema]


def _get_organize_tools() -> list[dict]:
    """Assemble tool schemas for organize session (organize + graph)."""
    from library.tools.graph import GRAPH_TOOL_SCHEMAS
    from library.tools.organize import ORGANIZE_TOOL_SCHEMAS
    return ORGANIZE_TOOL_SCHEMAS + GRAPH_TOOL_SCHEMAS


def _run_think_phase(ctx: InstanceContext, weights: dict, variables: dict) -> None:
    """Run one think phase iteration."""
    from library.tools.patterns import thinking_session

    thinking = ctx.read("thinking.md") or ""
    concerns = ctx.read("concerns.md") or ""
    dreams = ctx.read("dreams.md") or ""
    sleep_output = ctx.read("sleep.md") or ""
    reviews = ctx.read("reviews.md") or ""
    rep_summary = _graph_map_summary(ctx)

    think_template = (_SPECIES_DIR / "prompts" / "think.md").read_text()
    think_system = _inject_segments(
        think_template, _REGISTRY, weights, variables,
        ["identity", "state", "meta"],
    )

    thinking_context = _build_context(
        ("Your Concerns", concerns),
        ("Your Dreams", dreams),
        ("Your Sleep Consolidation", sleep_output),
        ("Recent Reviews", reviews),
        ("Representation State", rep_summary),
        ("Your Current Thoughts", thinking),
    ) or "This is your first thinking session."

    thinking_session(
        ctx,
        system=think_system,
        initial_message=thinking_context,
        max_tokens=16384,
        extra_tools=_get_thinking_tools(),
    )


def _run_organize_phase(ctx: InstanceContext, weights: dict, variables: dict) -> None:
    """Run the organize phase — a tool-use session for knowledge management."""
    from library.tools.patterns import thinking_session

    organize_template = (_SPECIES_DIR / "prompts" / "organize.md").read_text()
    organize_system = _inject_segments(
        organize_template, _REGISTRY, weights, variables,
        ["identity"],
    )

    # Build context: current thinking + knowledge summary
    from library.tools.organize import _list_category_names, _list_topics_in_category
    categories = _list_category_names(ctx)
    knowledge_lines = []
    for cat in categories:
        topics = _list_topics_in_category(ctx, cat)
        knowledge_lines.append(f"- {cat}: {len(topics)} topics ({', '.join(topics[:5])}{'...' if len(topics) > 5 else ''})")
    knowledge_summary = "\n".join(knowledge_lines) if knowledge_lines else "No knowledge categories yet."

    thinking = ctx.read("thinking.md") or ""
    organize_context = _build_context(
        ("Your Current Thoughts", thinking),
        ("Knowledge Structure", knowledge_summary),
    )

    thinking_session(
        ctx,
        system=organize_system,
        initial_message=organize_context,
        max_tokens=8192,
        extra_tools=_get_organize_tools(),
    )


def _run_subconscious_phase(ctx: InstanceContext, weights: dict, variables: dict) -> None:
    """Run the subconscious phase — surfaces concerns from thinking + dreams."""
    from library.tools.patterns import llm_generate

    subconscious_template = (_SPECIES_DIR / "prompts" / "subconscious.md").read_text()
    subconscious_system = _inject_segments(
        subconscious_template, _REGISTRY, weights, variables,
        ["state"],
    )

    thinking = ctx.read("thinking.md") or ""
    dreams = ctx.read("dreams.md") or ""
    content = _build_context(
        ("Your Thoughts", thinking),
        ("Your Dreams", dreams),
    )

    result = llm_generate(ctx, system=subconscious_system, content=content, max_tokens=4096)
    ctx.write("concerns.md", result)


def _run_dream_phase(ctx: InstanceContext, weights: dict, variables: dict) -> None:
    """Run the dream phase — associative processing from thinking + concerns."""
    from library.tools.patterns import llm_generate

    dreaming_template = (_SPECIES_DIR / "prompts" / "dreaming.md").read_text()
    dreaming_system = _inject_segments(
        dreaming_template, _REGISTRY, weights, variables,
        ["identity"],
    )

    thinking = ctx.read("thinking.md") or ""
    concerns = ctx.read("concerns.md") or ""
    content = _build_context(
        ("Your Thoughts", thinking),
        ("Your Concerns", concerns),
    )

    result = llm_generate(ctx, system=dreaming_system, content=content, max_tokens=4096)
    ctx.write("dreams.md", result)


def _run_sleep_phase(ctx: InstanceContext) -> None:
    """Run the sleep phase — consolidation and self-description."""
    from library.tools.patterns import llm_generate

    sleep_system = (_SPECIES_DIR / "prompts" / "sleep.md").read_text()

    thinking = ctx.read("thinking.md") or ""
    concerns = ctx.read("concerns.md") or ""
    dreams = ctx.read("dreams.md") or ""
    reviews = ctx.read("reviews.md") or ""
    prev_sleep = ctx.read("sleep.md") or ""
    content = _build_context(
        ("Your Thoughts", thinking),
        ("Your Concerns", concerns),
        ("Your Dreams", dreams),
        ("Session Reviews", reviews),
        ("Previous Sleep Output", prev_sleep),
    )

    result = llm_generate(ctx, system=sleep_system, content=content, max_tokens=8192)
    ctx.write("sleep.md", result)


# --- Entry points ---

def heartbeat(ctx: InstanceContext) -> None:
    """Slow cycle with variable iteration.

    Phase sequence is determined by shared behavioral parameters:
    - Think can repeat (controlled by processing_depth, max 3 iterations)
    - Organize is probabilistic (controlled by organization_drive)
    - Dream is probabilistic (controlled by creative_latitude)
    - Subconscious and sleep always run
    """
    logger.info("Heartbeat starting (instance=%s)", ctx.instance_id)

    weights, variables, shared_params = _load_weights_and_variables(ctx)

    schedule_config = {}
    try:
        raw = getattr(ctx, "instance_config", None)
        if raw is not None:
            sched = getattr(raw, "schedule", None)
            if isinstance(sched, dict):
                schedule_config = sched
    except AttributeError:
        pass

    phases = _build_heartbeat_phases(shared_params, schedule_config)
    logger.info("Heartbeat phases: %s", " → ".join(phases))

    # Snapshot state before phases for metrics delta computation
    pre_snapshot = _snapshot_counts(ctx)

    phase_count = 0
    think_count = 0
    max_phases = int(schedule_config.get("max_phases_per_heartbeat", 8))

    for phase in phases:
        if phase_count >= max_phases:
            logger.info("Heartbeat phase cap reached (%d), skipping remaining", max_phases)
            break
        phase_count += 1

        logger.info("Heartbeat running phase: %s (%d/%d)", phase, phase_count, len(phases))

        if phase == "think":
            _run_think_phase(ctx, weights, variables)
            think_count += 1
        elif phase == "organize":
            _run_organize_phase(ctx, weights, variables)
        elif phase == "subconscious":
            _run_subconscious_phase(ctx, weights, variables)
        elif phase == "dream":
            _run_dream_phase(ctx, weights, variables)
        elif phase == "sleep":
            _run_sleep_phase(ctx)
        else:
            logger.warning("Unknown heartbeat phase: %s", phase)

    # Build session metrics from pre/post snapshots
    metrics = _build_session_metrics(ctx, pre_snapshot, think_count)
    logger.info(
        "Session metrics: thinks=%d, graph +%d/%d nodes +%d/%d edges, "
        "topics +%d/%d, archived=%d chars",
        metrics.think_iterations,
        metrics.graph_nodes_added, metrics.graph_nodes_total,
        metrics.graph_edges_added, metrics.graph_edges_total,
        metrics.topics_added, metrics.topics_total,
        metrics.thoughts_archived_chars,
    )

    # Update slow net with sleep output + activity signals
    new_sleep = ctx.read("sleep.md") or ""
    if new_sleep:
        logger.info("Heartbeat sleep output: %d chars, updating slow net", len(new_sleep))
        _update_slow_net(ctx, new_sleep, f"heartbeat_{ctx.instance_id}", metrics)
    else:
        logger.info("Heartbeat: no sleep output produced")

    # Clear accumulated reviews after sleep consolidation
    ctx.write("reviews.md", "")

    # Render and publish graph/map visualizations
    try:
        from library.publish import render_and_publish
        render_and_publish(ctx)
    except Exception as exc:
        logger.warning("Post-heartbeat render failed: %s", exc)

    logger.info("Heartbeat complete (phases=%d, reviews cleared)", phase_count)


def _on_message_rate_limited(
    ctx: InstanceContext,
    events_text: str,
    thinking: str,
    weights: dict,
    variables: dict,
    shared_params: dict,
) -> None:
    """Abbreviated on_message for rate-limited situations.

    Runs gut + review only (2 LLM calls instead of 4). No reply is generated
    or sent, but the fast net still learns from the observation.
    """
    from library.tools.patterns import llm_generate

    # Gut phase
    gut_template = (_SPECIES_DIR / "prompts" / "gut.md").read_text()
    gut_system = _inject_segments(
        gut_template, _REGISTRY, weights, variables,
        ["state", "relational"],
    )
    gut_context = _build_context(
        ("Incoming Messages", events_text),
        ("Your Thoughts", thinking),
    )
    gut_output = llm_generate(ctx, system=gut_system, content=gut_context, max_tokens=4096)

    # Review phase (no reply to review, but we still assess the conversation)
    review_system = (_SPECIES_DIR / "prompts" / "review.md").read_text()
    review_context = _build_context(
        ("Incoming Messages", events_text),
        ("Your Thoughts", thinking),
        ("Gut Assessment", gut_output),
    )
    review = llm_generate(ctx, system=review_system, content=review_context, max_tokens=4096)

    # Update fast net with review signals
    if review:
        logger.info("Rate-limited review: %d chars, updating fast net", len(review))
        ctx.write("last_review.md", review)
        existing = ctx.read("reviews.md") or ""
        ctx.write("reviews.md", f"{existing}\n---\n{review}".strip())

        # Mechanical signals: no reply was generated
        review_with_mechanical = f"{review}\nreply_length: 0.000\nreply_sent: 0.0"
        _update_fast_net(ctx, review_with_mechanical)

    logger.info("on_message rate-limited complete (gut + review only)")


def on_message(ctx: InstanceContext, events: list[Event]) -> None:
    """Fast cycle: gut → suggest → reply → review → send."""
    if not events:
        return

    senders = {e.sender for e in events}
    logger.info("on_message starting (instance=%s, events=%d, senders=%s)",
                ctx.instance_id, len(events), ", ".join(senders))

    events_text = format_events(events, self_entity_id=get_entity_id(ctx))
    thinking = ctx.read("thinking.md") or ""
    sleep_output = ctx.read("sleep.md") or ""
    rep_summary = _graph_map_summary(ctx)

    weights, variables, shared_params = _load_weights_and_variables(ctx)

    # --- Rate-limited abbreviated pipeline: gut + review only ---
    rate_limited = getattr(ctx, "_reply_rate_limited", False) is True
    if rate_limited:
        logger.info("on_message rate-limited: running gut + review only (silent learning)")
        return _on_message_rate_limited(ctx, events_text, thinking, weights, variables, shared_params)

    # Build system prompts with segment injection
    gut_template = (_SPECIES_DIR / "prompts" / "gut.md").read_text()
    gut_system = _inject_segments(
        gut_template, _REGISTRY, weights, variables,
        ["state", "relational"],
    )

    suggest_template = (_SPECIES_DIR / "prompts" / "suggest.md").read_text()
    suggest_system = _inject_segments(
        suggest_template, _REGISTRY, weights, variables,
        ["state", "task"],
    )

    reply_template = (_SPECIES_DIR / "prompts" / "reply.md").read_text()
    reply_system = _inject_segments(
        reply_template, _REGISTRY, weights, variables,
        ["identity", "state"],
    )

    initial_state = {
        "gut_system": gut_system,
        "gut_context": _build_context(
            ("Incoming Messages", events_text),
            ("Your Thoughts", thinking),
        ),
        "suggest_system": suggest_system,
        "suggest_context": _build_context(
            ("Incoming Messages", events_text),
            ("Representation State", rep_summary),
        ),
        "reply_system": reply_system,
        "reply_context": _build_context(
            ("Incoming Messages", events_text),
            ("Your Thoughts", thinking),
        ),
        # Review context is assembled after reply is generated — the pipeline
        # sets pipeline.response, so we can reference it. We pre-build the
        # static parts and the review stage reads the dynamic ones.
        "review_context": _build_context(
            ("Incoming Messages", events_text),
            ("Your Thoughts", thinking),
        ),
        "_species_dir": str(_SPECIES_DIR),
    }

    logger.info("on_message running pipeline: gut → suggest → reply → review")
    state = run_pipeline(ctx, _ON_MESSAGE_STEPS, events=events, initial_state=initial_state)

    # --- Reply gating ---
    nn_available = shared_params.get("_nn_available", False)
    response = state.get("response", "")
    reply_sent = False

    if nn_available:
        # Extract reply_value from gut output
        gut_output = state.get("gut_feeling", "")
        gut_signals = _parse_review_signals(gut_output)
        reply_value = gut_signals.get("reply_value", 1.0)

        reply_willingness = shared_params.get("reply_willingness", 0.5)

        should_reply = probabilistic(
            reply_value * reply_willingness,
            label="reply_decision",
        )
        logger.info(
            "Reply gating: reply_value=%.2f, reply_willingness=%.2f, "
            "probability=%.3f, decision=%s",
            reply_value, reply_willingness,
            reply_value * reply_willingness, should_reply,
        )
    else:
        # No NN — always reply (deterministic old behavior)
        should_reply = True

    if should_reply and response and response.strip():
        target = _target_room(events, ctx)
        logger.info("on_message sending reply (%d chars) to %s", len(response.strip()), target)
        ctx.send(target, response.strip())
        reply_sent = True
    elif not should_reply:
        logger.info("on_message: reply suppressed by gating")
    else:
        logger.info("on_message: no reply generated")

    # Accumulate review into reviews.md for sleep phase
    review = state.get("last_review", "")
    if review:
        logger.info("on_message review output: %d chars, updating fast net", len(review))
        ctx.write("last_review.md", review)
        existing = ctx.read("reviews.md") or ""
        ctx.write("reviews.md", f"{existing}\n---\n{review}".strip())

        # Inject mechanical signal: reply_length
        from library.tools.neural import normalize_reply_length
        reply_length_val = normalize_reply_length(response) if response else 0.0
        review_with_mechanical = f"{review}\nreply_length: {reply_length_val:.3f}"

        # Inject reply gating signals
        if not reply_sent:
            review_with_mechanical += "\nreply_sent: 0.0"
            # Heuristic silence_confidence: higher when reply_value is low
            # and reply_willingness is low
            gut_output = state.get("gut_feeling", "")
            gut_signals = _parse_review_signals(gut_output)
            rv = gut_signals.get("reply_value", 1.0)
            rw = shared_params.get("reply_willingness", 0.5)
            silence_confidence = rv * (1.0 - rw)
            review_with_mechanical += f"\nsilence_confidence: {silence_confidence:.3f}"

        # Update fast net with review signals (including mechanical)
        _update_fast_net(ctx, review_with_mechanical)
    else:
        logger.info("on_message: no review output")

    # --- Optional post-reply extra thinking (step 10) ---
    nn_available = shared_params.get("_nn_available", False)
    if nn_available:
        processing_depth = shared_params.get("processing_depth", 0.5)
        if probabilistic(processing_depth * 0.3, label="post_reply_think"):
            logger.info("on_message: running post-reply extra thinking")
            _run_think_phase(ctx, weights, variables)


class NeuralDreamer(Species):
    def manifest(self) -> SpeciesManifest:
        return SpeciesManifest(
            species_id="neural_dreamer",
            entry_points=[
                EntryPoint(name="on_message", handler=on_message, trigger="message"),
                EntryPoint(name="heartbeat", handler=heartbeat, schedule="heartbeat"),
            ],
            default_files=DEFAULT_FILES,
        )
