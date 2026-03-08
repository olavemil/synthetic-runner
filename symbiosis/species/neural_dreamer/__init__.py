"""Neural Dreamer species — NN-driven extension of Subconscious Dreamer.

Two cycles:
  Slow (heartbeat): think → sleep (updates slow net)
  Fast (on_message): gut → suggest → reply → review (updates fast net)

When neural nets are available, they produce segment selection weights and
variable values. Falls back to manual defaults when PyTorch is not installed
or nets haven't been initialised yet.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, TYPE_CHECKING

from symbiosis.species import Species, SpeciesManifest, EntryPoint
from symbiosis.toolkit.pipeline import run_pipeline, load_pipeline
from symbiosis.toolkit.prompts import format_events
from symbiosis.toolkit.segments import (
    SegmentRegistry,
    load_registry,
    build_prompt,
    render_prompt,
    select_segments,
    DEFAULT_VARIABLES,
)

if TYPE_CHECKING:
    from symbiosis.harness.adapters import Event
    from symbiosis.harness.context import InstanceContext

logger = logging.getLogger(__name__)

_SPECIES_DIR = Path(__file__).parent
_HEARTBEAT_STEPS = load_pipeline((_SPECIES_DIR / "heartbeat.yaml").read_text())["steps"]
_ON_MESSAGE_STEPS = load_pipeline((_SPECIES_DIR / "on_message.yaml").read_text())["steps"]
_REGISTRY = load_registry(_SPECIES_DIR / "segments" / "registry.yaml")

DEFAULT_FILES = {
    "thinking.md": "# Thinking\n",
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


def _try_nn_weights_and_variables(ctx: InstanceContext) -> tuple[dict[str, float], dict[str, float]] | None:
    """Try to get weights and variables from neural nets. Returns None if unavailable.

    When graph/map data is available, their summary features are appended to
    the signal vectors — but only if the loaded net's input_dim matches.
    """
    try:
        from symbiosis.toolkit.neural import (
            load_fast_net, load_slow_net,
            make_fast_net_config, make_slow_net_config,
            encode_fast_signals, encode_slow_signals,
            encode_graph_features, encode_map_features,
            decode_fast_output, decode_slow_output,
            FAST_SIGNAL_NAMES, MAP_FEATURE_NAMES,
            SLOW_SIGNAL_NAMES, GRAPH_FEATURE_NAMES,
        )
    except ImportError:
        return None

    fast_net, _ = load_fast_net(ctx)
    slow_net, _ = load_slow_net(ctx)

    if fast_net is None and slow_net is None:
        return None

    weights: dict[str, float] = {}
    variables: dict[str, float] = {}

    if fast_net is not None:
        # Use last review signals as input, or zeros
        last_review = ctx.read("last_review.md") or ""
        signals = _parse_review_signals(last_review)
        input_vals = encode_fast_signals(signals)

        # Append map features if the net expects them
        expected_with_map = len(FAST_SIGNAL_NAMES) + len(MAP_FEATURE_NAMES)
        if fast_net.config.input_dim == expected_with_map:
            from symbiosis.toolkit.activation_map import load_map
            m = load_map(ctx)
            input_vals += encode_map_features(m)

        raw = fast_net.forward(input_vals)
        decoded = decode_fast_output(raw, _FAST_SEGMENT_IDS)
        weights.update(decoded["weights"])
        variables.update(decoded["variables"])

    if slow_net is not None:
        # Use sleep signals as input, or zeros
        sleep = ctx.read("sleep.md") or ""
        signals = _parse_sleep_signals(sleep)
        input_vals = encode_slow_signals(signals)

        # Append graph features if the net expects them
        expected_with_graph = len(SLOW_SIGNAL_NAMES) + len(GRAPH_FEATURE_NAMES)
        if slow_net.config.input_dim == expected_with_graph:
            from symbiosis.toolkit.graph import load_graph
            graph = load_graph(ctx)
            input_vals += encode_graph_features(graph)

        raw = slow_net.forward(input_vals)
        decoded = decode_slow_output(raw, _SLOW_SEGMENT_IDS)
        weights.update(decoded["weights"])
        variables.update(decoded["variables"])

    return weights, variables


def _load_weights_and_variables(ctx: InstanceContext) -> tuple[dict[str, float], dict[str, float]]:
    """Load weights and variables, preferring NN output, falling back to defaults."""
    nn_result = _try_nn_weights_and_variables(ctx)
    if nn_result is not None:
        weights, variables = nn_result
        # Merge with defaults for any missing segments
        merged_weights = dict(_DEFAULT_WEIGHTS)
        merged_weights.update(weights)
        merged_variables = dict(DEFAULT_VARIABLES)
        merged_variables.update(variables)
        return merged_weights, merged_variables

    # Fall back to manual weights from storage or defaults
    raw = ctx.read("segment_weights.json")
    if raw:
        try:
            weights = json.loads(raw)
            return weights, dict(DEFAULT_VARIABLES)
        except (json.JSONDecodeError, TypeError):
            pass
    return dict(_DEFAULT_WEIGHTS), dict(DEFAULT_VARIABLES)


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
        from symbiosis.toolkit.neural import (
            load_fast_net, save_fast_net, make_fast_net_config,
            encode_fast_signals, encode_map_features,
            FAST_SIGNAL_NAMES, MAP_FEATURE_NAMES,
            CheckpointMeta, record_checkpoint,
        )
    except ImportError:
        return

    signals = _parse_review_signals(review_text)
    if not signals:
        return

    config = make_fast_net_config(len(_FAST_SEGMENT_IDS), include_map_features=True)
    fast_net, meta = load_fast_net(ctx, fallback_config=config)
    if fast_net is None:
        fast_net = _create_net_with_config(config)
        if fast_net is None:
            return

    input_vals = encode_fast_signals(signals)

    # Append map features if the net expects them
    expected_with_map = len(FAST_SIGNAL_NAMES) + len(MAP_FEATURE_NAMES)
    if fast_net.config.input_dim == expected_with_map:
        from symbiosis.toolkit.activation_map import load_map
        input_vals += encode_map_features(load_map(ctx))

    current_output = fast_net.forward(input_vals)
    target = _nudge_output(current_output, signals, _FAST_SEGMENT_IDS)

    fast_net.train_step(input_vals, target)

    meta.update_count += 1
    save_fast_net(ctx, fast_net, meta)
    record_checkpoint(ctx, f"fast_update_{meta.update_count}", "fast")


def _update_slow_net(ctx: InstanceContext, sleep_text: str, session_label: str) -> None:
    """Parse sleep signals and update slow net.

    New nets are created with graph features enabled. Existing nets keep
    their original input_dim — graph features are appended only when dims match.
    """
    try:
        from symbiosis.toolkit.neural import (
            load_slow_net, save_slow_net, make_slow_net_config,
            encode_slow_signals, encode_graph_features,
            SLOW_SIGNAL_NAMES, GRAPH_FEATURE_NAMES,
            CheckpointMeta, record_checkpoint,
        )
    except ImportError:
        return

    signals = _parse_sleep_signals(sleep_text)
    if not signals:
        return

    config = make_slow_net_config(len(_SLOW_SEGMENT_IDS), include_graph_features=True)
    slow_net, meta = load_slow_net(ctx, fallback_config=config)
    if slow_net is None:
        slow_net = _create_net_with_config(config)
        if slow_net is None:
            return

    input_vals = encode_slow_signals(signals)

    # Append graph features if the net expects them
    expected_with_graph = len(SLOW_SIGNAL_NAMES) + len(GRAPH_FEATURE_NAMES)
    if slow_net.config.input_dim == expected_with_graph:
        from symbiosis.toolkit.graph import load_graph
        input_vals += encode_graph_features(load_graph(ctx))

    current_output = slow_net.forward(input_vals)
    target = _nudge_output(current_output, signals, _SLOW_SEGMENT_IDS)

    slow_net.train_step(input_vals, target)

    meta.update_count += 1
    meta.session_label = session_label
    save_slow_net(ctx, slow_net, meta)
    record_checkpoint(ctx, session_label, "slow")


def _create_net_with_config(config: Any) -> Any:
    """Create a Net instance, returning None if PyTorch unavailable."""
    try:
        from symbiosis.toolkit.neural import Net
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
    from symbiosis.toolkit.graph import load_graph
    graph = load_graph(ctx)
    desc = graph.describe()
    if desc["node_count"] > 0:
        top = ", ".join(f"{n['label']} ({n['degree']})" for n in desc["top_nodes"][:3])
        parts.append(
            f"Graph: {desc['node_count']} nodes, {desc['edge_count']} edges. "
            f"Top: {top}."
        )

    from symbiosis.toolkit.activation_map import load_map
    m = load_map(ctx)
    if m.x_label:
        md = m.describe()
        parts.append(
            f"Map ({m.x_label} x {m.y_label}): "
            f"{md['active_cells']} active cells, mean {md['mean']}."
        )

    return "\n".join(parts) if parts else ""


# --- Entry points ---

def heartbeat(ctx: InstanceContext) -> None:
    """Slow cycle: think → sleep → update slow net."""
    thinking = ctx.read("thinking.md") or ""
    sleep_output = ctx.read("sleep.md") or ""
    reviews = ctx.read("reviews.md") or ""
    rep_summary = _graph_map_summary(ctx)

    weights, variables = _load_weights_and_variables(ctx)

    # Build thinking system prompt with segment injection
    think_template = (_SPECIES_DIR / "prompts" / "think.md").read_text()
    think_system = _inject_segments(
        think_template, _REGISTRY, weights, variables,
        ["identity", "state", "meta"],
    )

    thinking_context = _build_context(
        ("Your Sleep Consolidation", sleep_output),
        ("Recent Reviews", reviews),
        ("Representation State", rep_summary),
        ("Your Current Thoughts", thinking),
    ) or "This is your first thinking session."

    # Assemble extra tools for thinking session (graph + map)
    from symbiosis.toolkit.graph import GRAPH_TOOL_SCHEMAS
    from symbiosis.toolkit.activation_map import MAP_TOOL_SCHEMAS
    thinking_tools = GRAPH_TOOL_SCHEMAS + MAP_TOOL_SCHEMAS

    initial_state = {
        "think_system": think_system,
        "thinking_context": thinking_context,
        "thinking_tools": thinking_tools,
        "sleep_sections": [
            ["thinking.md", "Your Thoughts"],
            ["reviews.md", "Session Reviews"],
            ["sleep.md", "Previous Sleep Output"],
        ],
        "_species_dir": str(_SPECIES_DIR),
    }

    run_pipeline(ctx, _HEARTBEAT_STEPS, initial_state=initial_state)

    # Update slow net with sleep output
    new_sleep = ctx.read("sleep.md") or ""
    if new_sleep:
        _update_slow_net(ctx, new_sleep, f"heartbeat_{ctx.instance_id}")

    # Clear accumulated reviews after sleep consolidation
    ctx.write("reviews.md", "")


def on_message(ctx: InstanceContext, events: list[Event]) -> None:
    """Fast cycle: gut → suggest → reply → review → send."""
    if not events:
        return

    events_text = format_events(events)
    thinking = ctx.read("thinking.md") or ""
    sleep_output = ctx.read("sleep.md") or ""
    rep_summary = _graph_map_summary(ctx)

    weights, variables = _load_weights_and_variables(ctx)

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

    state = run_pipeline(ctx, _ON_MESSAGE_STEPS, events=events, initial_state=initial_state)

    response = state.get("response", "")
    if response and response.strip():
        ctx.send(_target_room(events, ctx), response.strip())

    # Accumulate review into reviews.md for sleep phase
    review = state.get("last_review", "")
    if review:
        ctx.write("last_review.md", review)
        existing = ctx.read("reviews.md") or ""
        ctx.write("reviews.md", f"{existing}\n---\n{review}".strip())

        # Update fast net with review signals
        _update_fast_net(ctx, review)


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
