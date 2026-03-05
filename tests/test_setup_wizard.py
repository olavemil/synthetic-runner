"""Tests for interactive setup wizard."""

from pathlib import Path

import yaml

from symbiosis.setup_wizard import run_setup


def _input_from(values: list[str]):
    items = iter(values)
    return lambda _prompt="": next(items)


class TestSetupWizard:
    def test_bootstrap_creates_harness_instance_and_pipeline(self, tmp_path: Path):
        outputs: list[str] = []
        run_setup(
            tmp_path,
            input_fn=_input_from(
                [
                    "sk-ant-test",  # ANTHROPIC_API_KEY
                    "",             # LMSTUDIO_BASE_URL (accept default)
                    "",             # MATRIX_HOMESERVER (empty)
                    "1",            # Provider setup: both
                    "1",            # Matrix adapter in harness: no
                    "1",            # Create first instance: yes
                    "alpha",        # Instance name
                    "1",            # Intelligence type: persistent
                    "2",            # Operating mode: hybrid
                    "2",            # Instance provider: anthropic
                    "",             # Model: accept default
                    "1",            # Configure messaging now: no
                ]
            ),
            output_fn=outputs.append,
        )

        env_text = (tmp_path / ".env").read_text()
        assert "ANTHROPIC_API_KEY=sk-ant-test" in env_text
        assert "LMSTUDIO_BASE_URL=http://localhost:1234/v1" in env_text
        assert "MATRIX_HOMESERVER=" in env_text

        harness = yaml.safe_load((tmp_path / "config" / "harness.yaml").read_text())
        provider_ids = [p["id"] for p in harness["providers"]]
        assert provider_ids == ["lmstudio", "anthropic"]
        assert harness["adapters"] == []

        instance = yaml.safe_load((tmp_path / "config" / "instances" / "alpha.yaml").read_text())
        assert instance["species"] == "draum"
        assert instance["provider"] == "anthropic"
        assert instance["model"] == "claude-sonnet-4-6"
        assert instance["intelligence"]["type"] == "persistent"
        assert instance["schedule"]["heartbeat"] == "0 * * * *"
        assert instance["pipeline"]["file"] == "config/pipelines/alpha.yaml"

        pipeline = yaml.safe_load((tmp_path / "config" / "pipelines" / "alpha.yaml").read_text())
        assert pipeline["intelligence_type"] == "persistent"
        assert "on_inbox" in pipeline["pipeline"]
        assert "on_schedule" in pipeline["pipeline"]

    def test_skips_first_species_when_instance_exists(self, tmp_path: Path):
        config_dir = tmp_path / "config"
        instances_dir = config_dir / "instances"
        instances_dir.mkdir(parents=True)

        (config_dir / "harness.yaml").write_text(
            yaml.safe_dump(
                {
                    "providers": [
                        {
                            "id": "lmstudio",
                            "type": "openai_compat",
                            "base_url": "${LMSTUDIO_BASE_URL}",
                            "api_key": "lm-studio",
                        }
                    ],
                    "adapters": [],
                    "storage_dir": "instances",
                    "store_path": "harness.db",
                    "poll_interval": 30,
                },
                sort_keys=False,
            )
        )
        (instances_dir / "existing.yaml").write_text(
            yaml.safe_dump(
                {"species": "draum", "provider": "lmstudio", "model": "local-model"},
                sort_keys=False,
            )
        )

        outputs: list[str] = []
        run_setup(
            tmp_path,
            input_fn=_input_from(
                [
                    "",   # ANTHROPIC_API_KEY
                    "",   # LMSTUDIO_BASE_URL
                    "",   # MATRIX_HOMESERVER
                    "4",  # Keep existing providers
                    "1",  # Matrix adapter: no
                ]
            ),
            output_fn=outputs.append,
        )

        instance_files = list((tmp_path / "config" / "instances").glob("*.yaml"))
        assert len(instance_files) == 1
        assert instance_files[0].name == "existing.yaml"

        pipeline_files = list((tmp_path / "config" / "pipelines").glob("*.yaml"))
        assert pipeline_files == []
        assert any("skipped first-species bootstrap" in line.lower() for line in outputs)

    def test_ignores_example_instance_when_checking_first_species(self, tmp_path: Path):
        config_dir = tmp_path / "config"
        instances_dir = config_dir / "instances"
        instances_dir.mkdir(parents=True)
        (instances_dir / "example.yaml").write_text(
            yaml.safe_dump(
                {
                    "species": "draum",
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-6",
                    "messaging": {
                        "adapter": "matrix-main",
                        "entity_id": "@example:matrix.org",
                        "access_token": "${EXAMPLE_MATRIX_TOKEN}",
                        "spaces": [{"name": "main", "handle": "!room:matrix.org"}],
                    },
                },
                sort_keys=False,
            )
        )

        outputs: list[str] = []
        run_setup(
            tmp_path,
            input_fn=_input_from(
                [
                    "",   # ANTHROPIC_API_KEY
                    "",   # LMSTUDIO_BASE_URL
                    "",   # MATRIX_HOMESERVER
                    "1",  # Provider setup: both
                    "1",  # Matrix adapter in harness: no
                    "1",  # Create first instance: yes
                    "beta",
                    "1",  # Intelligence type: persistent
                    "2",  # Operating mode: hybrid
                    "1",  # Provider: lmstudio
                    "",   # Model default
                    "1",  # Configure messaging: no
                ]
            ),
            output_fn=outputs.append,
        )

        assert (instances_dir / "beta.yaml").exists()
