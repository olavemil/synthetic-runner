"""Tests for config loading and env var resolution."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from library.harness.config import (
    HarnessConfig,
    InstanceConfig,
    load_harness_config,
    load_instance_config,
    _resolve_env_vars,
    _resolve_recursive,
)


class TestEnvResolution:
    def test_simple_var(self):
        result = _resolve_env_vars("${FOO}", {"FOO": "bar"})
        assert result == "bar"

    def test_multiple_vars(self):
        result = _resolve_env_vars("${A}/${B}", {"A": "x", "B": "y"})
        assert result == "x/y"

    def test_no_vars(self):
        result = _resolve_env_vars("plain text", {})
        assert result == "plain text"

    def test_missing_var_raises(self):
        with pytest.raises(ValueError, match="MISSING"):
            _resolve_env_vars("${MISSING}", {})

    def test_recursive_dict(self):
        obj = {"key": "${VAR}", "nested": {"inner": "${VAR}"}}
        result = _resolve_recursive(obj, {"VAR": "val"})
        assert result == {"key": "val", "nested": {"inner": "val"}}

    def test_recursive_list(self):
        obj = ["${A}", "${B}"]
        result = _resolve_recursive(obj, {"A": "1", "B": "2"})
        assert result == ["1", "2"]


class TestLoadHarnessConfig:
    def test_loads_yaml(self, tmp_path):
        config = {
            "providers": [
                {"id": "test", "type": "openai_compat", "base_url": "http://localhost:1234", "api_key": "key"},
            ],
            "adapters": [],
            "storage_dir": "data",
            "poll_interval": 10,
        }
        config_path = tmp_path / "config" / "harness.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(yaml.dump(config))

        result = load_harness_config(config_path)
        assert isinstance(result, HarnessConfig)
        assert len(result.providers) == 1
        assert result.providers[0].id == "test"
        assert result.providers[0].type == "openai_compat"
        assert result.storage_dir == "data"
        assert result.poll_interval == 10

    def test_env_resolution(self, tmp_path):
        config = {
            "providers": [
                {"id": "p1", "type": "anthropic", "api_key": "${TEST_KEY}"},
            ],
        }
        config_path = tmp_path / "config" / "harness.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(yaml.dump(config))

        env_path = tmp_path / ".env"
        env_path.write_text("TEST_KEY=secret123\n")

        result = load_harness_config(config_path, env_path=env_path)
        assert result.providers[0].api_key == "secret123"

    def test_get_provider(self):
        config = HarnessConfig(
            providers=[
                type("P", (), {"id": "a", "type": "x", "base_url": None, "api_key": None})(),  # type: ignore
            ]
        )
        # Use the dataclass properly
        from library.harness.config import ProviderConfig
        config = HarnessConfig(providers=[ProviderConfig(id="a", type="x")])
        assert config.get_provider("a").id == "a"
        with pytest.raises(KeyError):
            config.get_provider("missing")

    def test_defaults(self, tmp_path):
        config_path = tmp_path / "config" / "harness.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text("{}")

        result = load_harness_config(config_path)
        assert result.storage_dir == "instances"
        assert result.poll_interval == 30


class TestLoadInstanceConfig:
    def test_loads_instance(self, tmp_path):
        config = {
            "species": "draum",
            "provider": "anthropic",
            "model": "claude-opus-4-6",
            "messaging": {
                "adapter": "matrix-main",
                "entity_id": "@bot:matrix.org",
                "access_token": "secret-token-123",
                "spaces": [
                    {"name": "main", "handle": "!room:matrix.org"},
                ],
            },
            "schedule": {"heartbeat": "0 * * * *"},
        }
        config_dir = tmp_path / "config" / "instances"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "test-1.yaml"
        config_path.write_text(yaml.dump(config))

        result = load_instance_config(config_path)
        assert isinstance(result, InstanceConfig)
        assert result.instance_id == "test-1"
        assert result.species == "draum"
        assert result.provider == "anthropic"
        assert result.model == "claude-opus-4-6"
        assert result.messaging is not None
        assert result.messaging.adapter == "matrix-main"
        assert result.messaging.access_token == "secret-token-123"
        assert len(result.messaging.spaces) == 1
        assert result.messaging.spaces[0].name == "main"
        assert result.schedule == {"heartbeat": "0 * * * *"}

    def test_no_messaging(self, tmp_path):
        config = {"species": "worker", "provider": "lmstudio", "model": "local"}
        config_dir = tmp_path / "config" / "instances"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "worker-1.yaml"
        config_path.write_text(yaml.dump(config))

        result = load_instance_config(config_path)
        assert result.messaging is None

    def test_schedule_with_mixed_types(self, tmp_path):
        config = {
            "species": "draum",
            "provider": "anthropic",
            "model": "claude-opus-4-6",
            "schedule": {"heartbeat": "0 * * * *", "max_idle_heartbeats": 3},
        }
        config_dir = tmp_path / "config" / "instances"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "test.yaml"
        config_path.write_text(yaml.dump(config))

        result = load_instance_config(config_path)
        assert result.schedule["heartbeat"] == "0 * * * *"
        assert result.schedule["max_idle_heartbeats"] == 3


class TestSchedulerConfig:
    def test_defaults(self, tmp_path):
        config_path = tmp_path / "config" / "harness.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text("{}")

        from library.harness.config import load_harness_config, SchedulerConfig
        result = load_harness_config(config_path)
        assert isinstance(result.scheduler, SchedulerConfig)
        assert result.scheduler.check_interval == 300
        assert result.scheduler.work_interval == 60
        assert result.scheduler.log_file is None

    def test_custom_values(self, tmp_path):
        config = {
            "scheduler": {
                "check_interval": 120,
                "work_interval": 30,
                "log_file": "logs/run.log",
            }
        }
        config_path = tmp_path / "config" / "harness.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(yaml.dump(config))

        from library.harness.config import load_harness_config
        result = load_harness_config(config_path)
        assert result.scheduler.check_interval == 120
        assert result.scheduler.work_interval == 30
        assert result.scheduler.log_file == "logs/run.log"


class TestProviderMaxConcurrency:
    def test_max_concurrency_field(self):
        from library.harness.config import ProviderConfig
        pc = ProviderConfig(id="lmstudio", type="openai_compat", max_concurrency=2)
        assert pc.max_concurrency == 2

    def test_max_concurrency_default_none(self):
        from library.harness.config import ProviderConfig
        pc = ProviderConfig(id="anthropic", type="anthropic")
        assert pc.max_concurrency is None

    def test_max_concurrency_parsed_from_yaml(self, tmp_path):
        config = {
            "providers": [
                {"id": "lms", "type": "openai_compat", "max_concurrency": 2},
                {"id": "anth", "type": "anthropic"},
            ]
        }
        config_path = tmp_path / "config" / "harness.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(yaml.dump(config))

        from library.harness.config import load_harness_config
        result = load_harness_config(config_path)
        assert result.providers[0].max_concurrency == 2
        assert result.providers[1].max_concurrency is None


class TestSchedulingConstraints:
    """Tests for SchedulingConstraints parsing in instance config."""

    def test_scheduling_constraints_defaults(self):
        from library.harness.config import SchedulingConstraints
        sc = SchedulingConstraints()
        assert sc.guaranteed_thinking_interval == 14400
        assert sc.max_replies_per_window == 3
        assert sc.reactive_thinking_max_sessions == 2
        assert sc.reactive_thinking_cooldown == 900
        assert sc.on_message_thinking_phases is None

    def test_scheduling_constraints_custom(self):
        from library.harness.config import SchedulingConstraints
        sc = SchedulingConstraints(
            guaranteed_thinking_interval=7200,
            max_replies_per_window=5,
            reactive_thinking_max_sessions=3,
            reactive_thinking_cooldown=600,
            on_message_thinking_phases={"THINKING", "COMPOSING"},
        )
        assert sc.guaranteed_thinking_interval == 7200
        assert sc.max_replies_per_window == 5
        assert sc.reactive_thinking_max_sessions == 3
        assert sc.reactive_thinking_cooldown == 600
        assert sc.on_message_thinking_phases == {"THINKING", "COMPOSING"}

    def test_scheduling_constraints_parsed_from_yaml(self, tmp_path):
        config = {
            "species": "draum",
            "provider": "anthropic",
            "model": "claude-opus-4-6",
            "scheduling_constraints": {
                "guaranteed_thinking_interval": 7200,
                "max_replies_per_window": 5,
                "reactive_thinking_max_sessions": 3,
                "reactive_thinking_cooldown": 600,
                "on_message_thinking_phases": ["THINKING"],
            },
        }
        config_dir = tmp_path / "config" / "instances"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "test.yaml"
        config_path.write_text(yaml.dump(config))

        result = load_instance_config(config_path)
        assert result.scheduling_constraints is not None
        sc = result.scheduling_constraints
        assert sc.guaranteed_thinking_interval == 7200
        assert sc.max_replies_per_window == 5
        assert sc.reactive_thinking_max_sessions == 3
        assert sc.reactive_thinking_cooldown == 600
        assert sc.on_message_thinking_phases == {"THINKING"}

    def test_scheduling_constraints_not_in_extra(self, tmp_path):
        """Ensure scheduling_constraints is parsed and not left in extra dict."""
        config = {
            "species": "draum",
            "provider": "anthropic",
            "model": "claude",
            "scheduling_constraints": {
                "max_replies_per_window": 2,
            },
            "custom_key": "should be in extra",
        }
        config_dir = tmp_path / "config" / "instances"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "test.yaml"
        config_path.write_text(yaml.dump(config))

        result = load_instance_config(config_path)
        assert result.scheduling_constraints is not None
        assert "scheduling_constraints" not in result.extra
        assert result.extra.get("custom_key") == "should be in extra"

    def test_no_scheduling_constraints(self, tmp_path):
        """Instance without scheduling_constraints has None."""
        config = {
            "species": "draum",
            "provider": "anthropic",
            "model": "claude",
        }
        config_dir = tmp_path / "config" / "instances"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "test.yaml"
        config_path.write_text(yaml.dump(config))

        result = load_instance_config(config_path)
        assert result.scheduling_constraints is None
