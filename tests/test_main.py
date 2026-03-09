"""Tests for CLI helper behavior."""

from pathlib import Path

import pytest

from library.__main__ import _is_template_instance_file, load_species


class TestMainHelpers:
    def test_template_instance_names_are_ignored(self):
        assert _is_template_instance_file(Path("config/instances/example.yaml"))
        assert _is_template_instance_file(Path("config/instances/sample.yaml"))
        assert _is_template_instance_file(Path("config/instances/template.yaml"))
        assert _is_template_instance_file(Path("config/instances/my-bot.example.yaml"))
        assert not _is_template_instance_file(Path("config/instances/agent-1.yaml"))

    def test_dynamic_species_loader(self):
        species = load_species("draum")
        assert species.manifest().species_id == "draum"

        thrivemind = load_species("thrivemind")
        assert thrivemind.manifest().species_id == "thrivemind"

        with pytest.raises(ValueError):
            load_species("missing-species")
