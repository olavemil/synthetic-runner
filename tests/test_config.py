"""Tests for config loading and env var resolution."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from symbiosis.harness.config import (
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
        from symbiosis.harness.config import ProviderConfig
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
