"""Tests for CLI helper behavior."""

from pathlib import Path

from symbiosis.__main__ import _is_template_instance_file


class TestMainHelpers:
    def test_template_instance_names_are_ignored(self):
        assert _is_template_instance_file(Path("config/instances/example.yaml"))
        assert _is_template_instance_file(Path("config/instances/sample.yaml"))
        assert _is_template_instance_file(Path("config/instances/template.yaml"))
        assert _is_template_instance_file(Path("config/instances/my-bot.example.yaml"))
        assert not _is_template_instance_file(Path("config/instances/agent-1.yaml"))
