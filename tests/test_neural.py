"""Tests for neural net toolkit.

Tests are split: interface tests run without PyTorch, net tests require it.
"""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock

import pytest

from symbiosis.toolkit.neural import (
    FAST_SIGNAL_NAMES,
    SLOW_SIGNAL_NAMES,
    FAST_VARIABLE_NAMES,
    SLOW_VARIABLE_NAMES,
    GRAPH_FEATURE_NAMES,
    MAP_FEATURE_NAMES,
    NetConfig,
    CheckpointMeta,
    encode_fast_signals,
    encode_slow_signals,
    encode_graph_features,
    encode_map_features,
    decode_fast_output,
    decode_slow_output,
    make_fast_net_config,
    make_slow_net_config,
    record_checkpoint,
    get_checkpoint_history,
    _sigmoid,
)


# ---------------------------------------------------------------------------
# Signal encoding (no PyTorch)
# ---------------------------------------------------------------------------


class TestSignalEncoding:
    def test_encode_fast_signals(self):
        signals = {"success": 0.8, "coherence": 0.6, "effort": 0.3}
        result = encode_fast_signals(signals)
        assert len(result) == len(FAST_SIGNAL_NAMES)
        assert result[0] == 0.8  # success
        assert result[1] == 0.6  # coherence
        assert result[2] == 0.3  # effort
        # Missing signals default to 0.0
        assert result[5] == 0.0  # external_valence

    def test_encode_slow_signals(self):
        signals = {"session_coherence": 0.9, "identity_drift": -0.2}
        result = encode_slow_signals(signals)
        assert len(result) == len(SLOW_SIGNAL_NAMES)
        assert result[0] == 0.9
        assert result[1] == -0.2

    def test_encode_empty_signals(self):
        result = encode_fast_signals({})
        assert all(v == 0.0 for v in result)


class TestSignalDecoding:
    def test_decode_fast_output(self):
        segment_ids = ["state-reflective", "relational-warm", "meta-concise"]
        # 3 segments + 5 variables = 8 values
        values = [2.0, -1.0, 0.0, 1.0, -0.5, 0.5, 0.0, -1.0]
        result = decode_fast_output(values, segment_ids)

        assert "weights" in result
        assert "variables" in result
        assert len(result["weights"]) == 3
        assert len(result["variables"]) == 5
        # Sigmoid of 2.0 ≈ 0.88
        assert result["weights"]["state-reflective"] == pytest.approx(_sigmoid(2.0), abs=0.01)
        # Variables should be sigmoid-transformed
        assert all(0.0 <= v <= 1.0 for v in result["variables"].values())

    def test_decode_slow_output(self):
        segment_ids = ["identity-core", "temporal-present", "task-open"]
        values = [1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5]
        result = decode_slow_output(values, segment_ids)

        assert len(result["weights"]) == 3
        assert len(result["variables"]) == 4
        assert all(name in result["variables"] for name in SLOW_VARIABLE_NAMES)

    def test_decode_pads_short_output(self):
        segment_ids = ["a", "b"]
        values = [1.0]  # Too short — should pad
        result = decode_fast_output(values, segment_ids)
        assert len(result["weights"]) == 2
        assert len(result["variables"]) == 5

    def test_sigmoid_bounds(self):
        assert _sigmoid(0.0) == pytest.approx(0.5)
        assert _sigmoid(10.0) > 0.99
        assert _sigmoid(-10.0) < 0.01
        # Extreme values clamped
        assert _sigmoid(100.0) > 0.99
        assert _sigmoid(-100.0) < 0.01


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestNetConfig:
    def test_fast_net_config(self):
        config = make_fast_net_config(6, hidden_dim=32)
        assert config.input_dim == len(FAST_SIGNAL_NAMES)
        assert config.output_dim == 6 + len(FAST_VARIABLE_NAMES)
        assert config.num_layers == 3
        assert config.learning_rate == 0.01

    def test_slow_net_config(self):
        config = make_slow_net_config(5, hidden_dim=64)
        assert config.input_dim == len(SLOW_SIGNAL_NAMES)
        assert config.output_dim == 5 + len(SLOW_VARIABLE_NAMES)
        assert config.num_layers == 5
        assert config.weight_decay == 0.01

    def test_config_roundtrip(self):
        config = NetConfig(input_dim=7, hidden_dim=32, num_layers=3,
                           output_dim=11, learning_rate=0.01)
        d = config.to_dict()
        config2 = NetConfig.from_dict(d)
        assert config == config2


class TestCheckpointMeta:
    def test_roundtrip(self):
        meta = CheckpointMeta(
            created_at=1234567890.0,
            update_count=42,
            session_label="session_5",
            config={"input_dim": 7},
        )
        d = meta.to_dict()
        meta2 = CheckpointMeta.from_dict(d)
        assert meta == meta2

    def test_defaults(self):
        meta = CheckpointMeta()
        assert meta.update_count == 0
        assert meta.session_label == ""


# ---------------------------------------------------------------------------
# Checkpoint history (no PyTorch)
# ---------------------------------------------------------------------------


class TestCheckpointHistory:
    def _mock_ctx(self):
        stored = {}
        ctx = MagicMock()
        ctx.read.side_effect = lambda p: stored.get(p, "")
        ctx.write.side_effect = lambda p, c: stored.__setitem__(p, c)
        ctx._stored = stored
        return ctx

    def test_record_and_get(self):
        ctx = self._mock_ctx()
        record_checkpoint(ctx, "session_1", "fast")
        record_checkpoint(ctx, "session_1", "slow")
        history = get_checkpoint_history(ctx)
        assert len(history) == 2
        assert history[0]["net_type"] == "fast"
        assert history[1]["net_type"] == "slow"

    def test_empty_history(self):
        ctx = self._mock_ctx()
        assert get_checkpoint_history(ctx) == []

    def test_corrupt_history(self):
        ctx = self._mock_ctx()
        ctx._stored["nets/history.json"] = "not json"
        assert get_checkpoint_history(ctx) == []


# ---------------------------------------------------------------------------
# Net construction and training (requires PyTorch)
# ---------------------------------------------------------------------------

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestNet:
    def test_forward_returns_correct_dim(self):
        from symbiosis.toolkit.neural import Net

        config = make_fast_net_config(6)
        net = Net(config)
        result = net.forward(encode_fast_signals({"success": 0.8}))
        assert len(result) == config.output_dim

    def test_forward_deterministic(self):
        from symbiosis.toolkit.neural import Net

        config = make_fast_net_config(4)
        net = Net(config)
        signals = encode_fast_signals({"success": 0.5, "coherence": 0.7})
        r1 = net.forward(signals)
        r2 = net.forward(signals)
        assert r1 == r2

    def test_train_step_reduces_loss(self):
        from symbiosis.toolkit.neural import Net

        config = make_fast_net_config(3, hidden_dim=16)
        net = Net(config)
        input_vals = encode_fast_signals({"success": 0.8, "coherence": 0.6})
        target = [0.5] * config.output_dim

        loss1 = net.train_step(input_vals, target)
        for _ in range(50):
            net.train_step(input_vals, target)
        loss2 = net.train_step(input_vals, target)

        assert loss2 < loss1

    def test_slow_net_forward(self):
        from symbiosis.toolkit.neural import Net

        config = make_slow_net_config(5)
        net = Net(config)
        signals = encode_slow_signals({"session_coherence": 0.9})
        result = net.forward(signals)
        assert len(result) == config.output_dim

    def test_state_dict_restorable(self):
        from symbiosis.toolkit.neural import Net

        config = make_fast_net_config(4)
        net1 = Net(config)
        signals = encode_fast_signals({"success": 0.5})
        r1 = net1.forward(signals)

        # Save and restore
        sd = net1.state_dict()
        net2 = Net(config, state_dict=sd)
        r2 = net2.forward(signals)
        assert r1 == r2


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestNetCheckpoints:
    def _mock_ctx(self):
        stored_text = {}
        stored_binary = {}
        ctx = MagicMock()
        ctx.read.side_effect = lambda p: stored_text.get(p, "")
        ctx.write.side_effect = lambda p, c: stored_text.__setitem__(p, c)
        ctx.read_binary.side_effect = lambda p: stored_binary.get(p)
        ctx.write_binary.side_effect = lambda p, d: stored_binary.__setitem__(p, d)
        return ctx

    def test_save_and_load_fast_net(self):
        from symbiosis.toolkit.neural import (
            Net, save_fast_net, load_fast_net, CheckpointMeta,
        )

        ctx = self._mock_ctx()
        config = make_fast_net_config(4)
        net = Net(config)
        signals = encode_fast_signals({"success": 0.7})
        expected = net.forward(signals)

        meta = CheckpointMeta(update_count=5, session_label="s3")
        save_fast_net(ctx, net, meta)

        loaded_net, loaded_meta = load_fast_net(ctx)
        assert loaded_net is not None
        assert loaded_net.forward(signals) == expected
        assert loaded_meta.update_count == 5
        assert loaded_meta.session_label == "s3"

    def test_save_and_load_slow_net(self):
        from symbiosis.toolkit.neural import (
            Net, save_slow_net, load_slow_net,
        )

        ctx = self._mock_ctx()
        config = make_slow_net_config(3)
        net = Net(config)
        signals = encode_slow_signals({"session_coherence": 0.8})
        expected = net.forward(signals)

        save_slow_net(ctx, net)
        loaded_net, _ = load_slow_net(ctx)
        assert loaded_net is not None
        assert loaded_net.forward(signals) == expected

    def test_load_missing_returns_none(self):
        from symbiosis.toolkit.neural import load_fast_net

        ctx = self._mock_ctx()
        net, meta = load_fast_net(ctx)
        assert net is None
        assert meta.update_count == 0

    def test_load_missing_with_fallback(self):
        from symbiosis.toolkit.neural import load_fast_net

        ctx = self._mock_ctx()
        fallback = make_fast_net_config(4)
        net, meta = load_fast_net(ctx, fallback_config=fallback)
        assert net is None  # No checkpoint data, returns None

    def test_end_to_end_fast_cycle(self):
        """Simulate: forward → decode → train → save → load → verify."""
        from symbiosis.toolkit.neural import (
            Net, save_fast_net, load_fast_net, CheckpointMeta,
        )

        ctx = self._mock_ctx()
        segment_ids = ["state-reflective", "relational-warm", "meta-concise"]
        config = make_fast_net_config(len(segment_ids))
        net = Net(config)

        # Forward pass
        signals = encode_fast_signals({"success": 0.8, "coherence": 0.7, "novelty": 0.3})
        raw_output = net.forward(signals)
        decoded = decode_fast_output(raw_output, segment_ids)

        assert all(0.0 <= v <= 1.0 for v in decoded["weights"].values())
        assert all(0.0 <= v <= 1.0 for v in decoded["variables"].values())

        # Train step (target: all 0.5)
        target = [0.0] * config.output_dim
        net.train_step(signals, target)

        # Save and reload
        meta = CheckpointMeta(update_count=1)
        save_fast_net(ctx, net, meta)
        loaded_net, loaded_meta = load_fast_net(ctx)
        assert loaded_net is not None
        assert loaded_meta.update_count == 1

        # Loaded net produces same output
        assert loaded_net.forward(signals) == net.forward(signals)


# ---------------------------------------------------------------------------
# Representation feature encoding (no PyTorch)
# ---------------------------------------------------------------------------


class TestGraphFeatureEncoding:
    def _make_graph(self):
        from symbiosis.toolkit.graph import SemanticGraph
        g = SemanticGraph()
        g.add_node("a", "Node A", {"salience": 0.8})
        g.add_node("b", "Node B", {"salience": 0.6})
        g.add_node("c", "Node C")
        g.add_node("d", "Node D")  # isolated
        g.add_edge("a", "b", "connects", weight=0.9)
        g.add_edge("a", "c", "relates", weight=0.5)
        g.add_edge("b", "c", "links", weight=0.7)
        return g

    def test_returns_correct_length(self):
        g = self._make_graph()
        result = encode_graph_features(g)
        assert len(result) == len(GRAPH_FEATURE_NAMES)

    def test_empty_graph(self):
        from symbiosis.toolkit.graph import SemanticGraph
        g = SemanticGraph()
        result = encode_graph_features(g)
        assert result == [0.0] * len(GRAPH_FEATURE_NAMES)

    def test_density_correct(self):
        g = self._make_graph()
        result = encode_graph_features(g)
        # 4 nodes, 3 edges, max_possible = 4*3 = 12, density = 3/12 = 0.25
        assert result[2] == pytest.approx(0.25)

    def test_mean_weight(self):
        g = self._make_graph()
        result = encode_graph_features(g)
        # mean weight = (0.9 + 0.5 + 0.7) / 3 = 0.7
        assert result[5] == pytest.approx(0.7)

    def test_isolate_fraction(self):
        g = self._make_graph()
        result = encode_graph_features(g)
        # 1 isolated node out of 4 = 0.25
        assert result[6] == pytest.approx(0.25)

    def test_mean_salience(self):
        g = self._make_graph()
        result = encode_graph_features(g)
        # salience values: 0.8, 0.6 (two nodes with explicit salience)
        assert result[7] == pytest.approx(0.7)

    def test_all_values_bounded(self):
        g = self._make_graph()
        result = encode_graph_features(g)
        assert all(-1.0 <= v <= 1.0 for v in result)


class TestMapFeatureEncoding:
    def _make_map(self):
        from symbiosis.toolkit.activation_map import ActivationMap
        m = ActivationMap(8, 8, "x", "y", "test")
        m.set_region(4, 4, 2, 0.8, "gaussian")
        m.set(1, 1, -0.5)
        return m

    def test_returns_correct_length(self):
        m = self._make_map()
        result = encode_map_features(m)
        assert len(result) == len(MAP_FEATURE_NAMES)

    def test_empty_map(self):
        from symbiosis.toolkit.activation_map import ActivationMap
        m = ActivationMap(4, 4)
        result = encode_map_features(m)
        # All zeros → mean=0, variance=0, peak=0, etc.
        assert result[0] == 0.0  # mean
        assert result[1] == 0.0  # variance
        assert result[2] == 0.0  # peak value
        assert result[6] == 0.0  # active fraction

    def test_peak_value_captured(self):
        m = self._make_map()
        result = encode_map_features(m)
        assert result[2] > 0.0  # peak value > 0

    def test_trough_value_captured(self):
        m = self._make_map()
        result = encode_map_features(m)
        assert result[5] < 0.0  # trough value < 0

    def test_active_fraction(self):
        m = self._make_map()
        result = encode_map_features(m)
        assert 0.0 < result[6] <= 1.0  # some cells active

    def test_asymmetry_positive_biased(self):
        from symbiosis.toolkit.activation_map import ActivationMap
        m = ActivationMap(4, 4)
        m.set_region(2, 2, 1, 0.9, "hard")
        result = encode_map_features(m)
        assert result[7] > 0.0  # more positive mass → positive asymmetry

    def test_asymmetry_negative_biased(self):
        from symbiosis.toolkit.activation_map import ActivationMap
        m = ActivationMap(4, 4)
        m.set_region(2, 2, 2, -0.8, "hard")
        result = encode_map_features(m)
        assert result[7] < 0.0  # more negative mass → negative asymmetry


class TestConfigWithRepFeatures:
    def test_fast_config_with_map(self):
        config = make_fast_net_config(6, include_map_features=True)
        assert config.input_dim == len(FAST_SIGNAL_NAMES) + len(MAP_FEATURE_NAMES)

    def test_fast_config_without_map(self):
        config = make_fast_net_config(6, include_map_features=False)
        assert config.input_dim == len(FAST_SIGNAL_NAMES)

    def test_slow_config_with_graph(self):
        config = make_slow_net_config(5, include_graph_features=True)
        assert config.input_dim == len(SLOW_SIGNAL_NAMES) + len(GRAPH_FEATURE_NAMES)

    def test_slow_config_without_graph(self):
        config = make_slow_net_config(5, include_graph_features=False)
        assert config.input_dim == len(SLOW_SIGNAL_NAMES)
