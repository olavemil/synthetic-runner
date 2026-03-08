"""Fast/slow neural network management for the Neural Dreamer species.

Provides two small feedforward nets that modulate LLM prompt configuration:
  - FastNet: shallow, updates per message review, encodes session affect
  - SlowNet: deeper, updates during sleep, encodes accumulated disposition

Both nets output segment selection weights and continuous variable values.
Checkpoints are stored as binary files via ctx.read_binary/write_binary.

PyTorch is lazily imported — the module loads without it, failing only when
nets are actually instantiated. Tests can exercise the interface layer
(signal encoding, config, checkpoint metadata) without PyTorch.
"""

from __future__ import annotations

import io
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from symbiosis.harness.context import InstanceContext

logger = logging.getLogger(__name__)

# Checkpoint paths
FAST_NET_PATH = "nets/fast.pt"
SLOW_NET_PATH = "nets/slow.pt"
FAST_NET_META_PATH = "nets/fast_meta.json"
SLOW_NET_META_PATH = "nets/slow_meta.json"
CHECKPOINT_HISTORY_PATH = "nets/history.json"

# How many session checkpoints to keep
MAX_CHECKPOINT_HISTORY = 10


def _import_torch():
    """Lazy import of torch with clear error message."""
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for neural net features. "
            "Install with: pip install symbiosis[neural]"
        )


# ---------------------------------------------------------------------------
# Signal definitions
# ---------------------------------------------------------------------------

# Fast net input signals (from review phase)
FAST_SIGNAL_NAMES = [
    "success",          # 0-1
    "coherence",        # 0-1
    "effort",           # 0-1
    "surprise",         # -1 to 1
    "unresolved",       # 0-1
    "external_valence", # -1 to 1
    "novelty",          # 0-1
]

# Slow net input signals (from sleep phase)
SLOW_SIGNAL_NAMES = [
    "session_coherence",    # 0-1
    "identity_drift",       # -1 to 1
    "accumulated_effort",   # 0-1
    "intention_alignment",  # 0-1
]
# Plus embedding vectors: emotional_characterisation, consolidation_items
# These are handled separately as variable-dim inputs.

# Fast net output: variable names it controls
FAST_VARIABLE_NAMES = [
    "tone_warmth",
    "verbosity",
    "risk_tolerance",
    "self_disclosure",
    "confidence",
]

# Slow net output: variable names it controls
SLOW_VARIABLE_NAMES = [
    "identity_salience",
    "temporal_weight",
    "relational_depth",
    "reflection_depth",
]


# Graph summary features for slow net input
GRAPH_FEATURE_NAMES = [
    "graph_node_count",      # normalized 0-1
    "graph_edge_count",      # normalized 0-1
    "graph_density",         # edges / max_possible_edges
    "graph_mean_degree",     # normalized
    "graph_max_degree",      # normalized
    "graph_mean_weight",     # average edge weight
    "graph_num_isolates",    # fraction of isolated nodes
    "graph_mean_salience",   # average salience from metadata
]

# Map summary features for fast net input
MAP_FEATURE_NAMES = [
    "map_mean",           # mean activation
    "map_variance",       # variance
    "map_peak_value",     # max absolute value
    "map_peak_x_norm",    # peak x position (0-1)
    "map_peak_y_norm",    # peak y position (0-1)
    "map_trough_value",   # min value
    "map_active_fraction",# fraction of non-zero cells
    "map_asymmetry",      # difference between positive and negative mass
]


@dataclass
class NetConfig:
    """Configuration for a single neural net."""
    input_dim: int
    hidden_dim: int
    num_layers: int
    output_dim: int
    learning_rate: float
    weight_decay: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> NetConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class CheckpointMeta:
    """Metadata stored alongside a checkpoint."""
    created_at: float = 0.0
    update_count: int = 0
    session_label: str = ""
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> CheckpointMeta:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Signal encoding/decoding (no PyTorch dependency)
# ---------------------------------------------------------------------------

def encode_fast_signals(signals: dict[str, float]) -> list[float]:
    """Encode review signals into a flat list for fast net input."""
    return [signals.get(name, 0.0) for name in FAST_SIGNAL_NAMES]


def encode_slow_signals(signals: dict[str, float]) -> list[float]:
    """Encode sleep signals into a flat list for slow net input."""
    return [signals.get(name, 0.0) for name in SLOW_SIGNAL_NAMES]


def encode_graph_features(graph: Any) -> list[float]:
    """Encode a SemanticGraph into summary statistics for slow net input.

    Returns a fixed-length vector matching GRAPH_FEATURE_NAMES.
    Uses simple statistics rather than GNN — appropriate for small nets.
    """
    from collections import defaultdict

    n_nodes = len(graph.nodes)
    n_edges = len(graph.edges)

    if n_nodes == 0:
        return [0.0] * len(GRAPH_FEATURE_NAMES)

    # Degree computation
    degrees: dict[str, int] = defaultdict(int)
    for e in graph.edges:
        degrees[e.source] += 1
        degrees[e.target] += 1

    max_possible = n_nodes * (n_nodes - 1) if n_nodes > 1 else 1
    density = n_edges / max_possible

    deg_values = list(degrees.values()) if degrees else [0]
    mean_degree = sum(deg_values) / n_nodes
    max_degree = max(deg_values) if deg_values else 0

    # Normalize counts with soft cap (sigmoid-like squashing)
    node_norm = min(n_nodes / 100.0, 1.0)
    edge_norm = min(n_edges / 200.0, 1.0)
    mean_deg_norm = min(mean_degree / 10.0, 1.0)
    max_deg_norm = min(max_degree / 20.0, 1.0)

    mean_weight = (sum(e.weight for e in graph.edges) / n_edges) if n_edges > 0 else 0.0

    connected = set(degrees.keys())
    n_isolates = sum(1 for nid in graph.nodes if nid not in connected)
    isolate_frac = n_isolates / n_nodes

    saliences = [
        n.metadata.get("salience", 0.5)
        for n in graph.nodes.values()
        if isinstance(n.metadata.get("salience"), (int, float))
    ]
    mean_salience = sum(saliences) / len(saliences) if saliences else 0.5

    return [
        node_norm,
        edge_norm,
        density,
        mean_deg_norm,
        max_deg_norm,
        mean_weight,
        isolate_frac,
        mean_salience,
    ]


def encode_map_features(m: Any) -> list[float]:
    """Encode an ActivationMap into summary statistics for fast net input.

    Returns a fixed-length vector matching MAP_FEATURE_NAMES.
    Uses summary statistics rather than CNN — appropriate for small nets.
    """
    flat = [v for row in m.grid for v in row]
    total = len(flat)

    if total == 0:
        return [0.0] * len(MAP_FEATURE_NAMES)

    mean_val = sum(flat) / total
    variance = sum((v - mean_val) ** 2 for v in flat) / total

    peak_val = 0.0
    peak_x, peak_y = 0, 0
    trough_val = 0.0

    for y in range(m.height):
        for x in range(m.width):
            v = m.grid[y][x]
            if v > peak_val:
                peak_val = v
                peak_x, peak_y = x, y
            if v < trough_val:
                trough_val = v

    peak_x_norm = peak_x / max(m.width - 1, 1)
    peak_y_norm = peak_y / max(m.height - 1, 1)

    nonzero = sum(1 for v in flat if abs(v) > 0.01)
    active_fraction = nonzero / total

    pos_mass = sum(v for v in flat if v > 0)
    neg_mass = sum(-v for v in flat if v < 0)
    total_mass = pos_mass + neg_mass
    asymmetry = (pos_mass - neg_mass) / total_mass if total_mass > 0 else 0.0

    return [
        mean_val,
        variance,
        peak_val,
        peak_x_norm,
        peak_y_norm,
        trough_val,
        active_fraction,
        asymmetry,
    ]


def decode_fast_output(values: list[float], segment_ids: list[str]) -> dict:
    """Decode fast net output into segment weights and variables.

    Output layout: [segment_weights..., variables...]
    """
    n_segments = len(segment_ids)
    n_vars = len(FAST_VARIABLE_NAMES)
    expected = n_segments + n_vars

    # Pad or truncate
    if len(values) < expected:
        values = values + [0.5] * (expected - len(values))

    weights = {sid: _sigmoid(values[i]) for i, sid in enumerate(segment_ids)}
    variables = {
        name: _sigmoid(values[n_segments + i])
        for i, name in enumerate(FAST_VARIABLE_NAMES)
    }
    return {"weights": weights, "variables": variables}


def decode_slow_output(values: list[float], segment_ids: list[str]) -> dict:
    """Decode slow net output into segment weights and variables."""
    n_segments = len(segment_ids)
    n_vars = len(SLOW_VARIABLE_NAMES)
    expected = n_segments + n_vars

    if len(values) < expected:
        values = values + [0.5] * (expected - len(values))

    weights = {sid: _sigmoid(values[i]) for i, sid in enumerate(segment_ids)}
    variables = {
        name: _sigmoid(values[n_segments + i])
        for i, name in enumerate(SLOW_VARIABLE_NAMES)
    }
    return {"weights": weights, "variables": variables}


def _sigmoid(x: float) -> float:
    """Sigmoid function clamped to avoid overflow."""
    import math
    x = max(-10.0, min(10.0, x))
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# Net construction (requires PyTorch)
# ---------------------------------------------------------------------------

def _build_net(config: NetConfig):
    """Build a feedforward net from config. Returns a torch.nn.Sequential."""
    torch = _import_torch()
    layers = []
    in_dim = config.input_dim
    for i in range(config.num_layers):
        out_dim = config.hidden_dim if i < config.num_layers - 1 else config.output_dim
        layers.append(torch.nn.Linear(in_dim, out_dim))
        if i < config.num_layers - 1:
            layers.append(torch.nn.ReLU())
        in_dim = out_dim
    return torch.nn.Sequential(*layers)


def make_fast_net_config(
    num_segments: int,
    *,
    hidden_dim: int = 32,
    num_layers: int = 3,
    learning_rate: float = 0.01,
    include_map_features: bool = False,
) -> NetConfig:
    """Create config for a fast net.

    Input: review signals (7) + optional map features (8)
    Output: segment weights + fast variables
    """
    input_dim = len(FAST_SIGNAL_NAMES)
    if include_map_features:
        input_dim += len(MAP_FEATURE_NAMES)
    return NetConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=num_segments + len(FAST_VARIABLE_NAMES),
        learning_rate=learning_rate,
    )


def make_slow_net_config(
    num_segments: int,
    *,
    hidden_dim: int = 64,
    num_layers: int = 5,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    include_graph_features: bool = False,
) -> NetConfig:
    """Create config for a slow net.

    Input: sleep signals (4) + optional graph features (8)
    Output: segment weights + slow variables
    """
    input_dim = len(SLOW_SIGNAL_NAMES)
    if include_graph_features:
        input_dim += len(GRAPH_FEATURE_NAMES)
    return NetConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=num_segments + len(SLOW_VARIABLE_NAMES),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )


# ---------------------------------------------------------------------------
# Net wrapper (requires PyTorch)
# ---------------------------------------------------------------------------

class Net:
    """Wrapper around a PyTorch net with forward pass and training step."""

    def __init__(self, config: NetConfig, state_dict: dict | None = None):
        torch = _import_torch()
        self.config = config
        self.model = _build_net(config)
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def forward(self, input_values: list[float]) -> list[float]:
        """Run forward pass. Returns raw output values (not yet decoded)."""
        torch = _import_torch()
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor([input_values], dtype=torch.float32)
            y = self.model(x)
            return y[0].tolist()

    def train_step(self, input_values: list[float], target_values: list[float]) -> float:
        """Single training step. Returns loss value."""
        torch = _import_torch()
        self.model.train()
        self.optimizer.zero_grad()
        x = torch.tensor([input_values], dtype=torch.float32)
        t = torch.tensor([target_values], dtype=torch.float32)
        y = self.model(x)
        loss = torch.nn.functional.mse_loss(y, t)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def state_dict(self) -> dict:
        return self.model.state_dict()


# ---------------------------------------------------------------------------
# Checkpoint management (requires PyTorch for save/load)
# ---------------------------------------------------------------------------

def save_net(ctx: InstanceContext, net: Net, path: str, meta_path: str,
             meta: CheckpointMeta | None = None) -> None:
    """Save net checkpoint and metadata to instance storage."""
    torch = _import_torch()
    buf = io.BytesIO()
    torch.save({
        "state_dict": net.state_dict(),
        "config": net.config.to_dict(),
    }, buf)
    ctx.write_binary(path, buf.getvalue())

    if meta is None:
        meta = CheckpointMeta()
    meta.config = net.config.to_dict()
    meta.created_at = time.time()
    ctx.write(meta_path, json.dumps(meta.to_dict(), indent=2))


def load_net(ctx: InstanceContext, path: str, meta_path: str,
             fallback_config: NetConfig | None = None) -> tuple[Net | None, CheckpointMeta]:
    """Load net from checkpoint. Returns (net, meta) or (None, empty_meta) if not found."""
    torch = _import_torch()

    data = ctx.read_binary(path)
    if data is None:
        return None, CheckpointMeta()

    try:
        buf = io.BytesIO(data)
        checkpoint = torch.load(buf, weights_only=False)
        config = NetConfig.from_dict(checkpoint["config"])
        net = Net(config, state_dict=checkpoint["state_dict"])
    except Exception:
        logger.warning("Failed to load checkpoint from %s", path)
        if fallback_config:
            return Net(fallback_config), CheckpointMeta()
        return None, CheckpointMeta()

    meta_raw = ctx.read(meta_path)
    meta = CheckpointMeta()
    if meta_raw:
        try:
            meta = CheckpointMeta.from_dict(json.loads(meta_raw))
        except (json.JSONDecodeError, TypeError):
            pass

    return net, meta


def save_fast_net(ctx: InstanceContext, net: Net, meta: CheckpointMeta | None = None) -> None:
    save_net(ctx, net, FAST_NET_PATH, FAST_NET_META_PATH, meta)


def load_fast_net(ctx: InstanceContext, fallback_config: NetConfig | None = None) -> tuple[Net | None, CheckpointMeta]:
    return load_net(ctx, FAST_NET_PATH, FAST_NET_META_PATH, fallback_config)


def save_slow_net(ctx: InstanceContext, net: Net, meta: CheckpointMeta | None = None) -> None:
    save_net(ctx, net, SLOW_NET_PATH, SLOW_NET_META_PATH, meta)


def load_slow_net(ctx: InstanceContext, fallback_config: NetConfig | None = None) -> tuple[Net | None, CheckpointMeta]:
    return load_net(ctx, SLOW_NET_PATH, SLOW_NET_META_PATH, fallback_config)


# ---------------------------------------------------------------------------
# Checkpoint history (rolling, no PyTorch dependency)
# ---------------------------------------------------------------------------

def record_checkpoint(ctx: InstanceContext, session_label: str, net_type: str) -> None:
    """Record a checkpoint event in the rolling history."""
    raw = ctx.read(CHECKPOINT_HISTORY_PATH)
    history: list[dict] = []
    if raw:
        try:
            history = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass

    history.append({
        "session_label": session_label,
        "net_type": net_type,
        "timestamp": time.time(),
    })

    # Keep only the last N entries
    if len(history) > MAX_CHECKPOINT_HISTORY * 2:
        history = history[-MAX_CHECKPOINT_HISTORY * 2:]

    ctx.write(CHECKPOINT_HISTORY_PATH, json.dumps(history, indent=2))


def get_checkpoint_history(ctx: InstanceContext) -> list[dict]:
    """Read the checkpoint history."""
    raw = ctx.read(CHECKPOINT_HISTORY_PATH)
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
