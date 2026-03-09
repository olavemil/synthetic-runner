"""Tests for YAML pipeline parsing and execution."""

from unittest.mock import MagicMock, patch

import pytest

from library.tools.pipeline import (
    resolve_input,
    write_output,
    consume_input,
    apply_preprocessor,
    run_stage,
    run_pipeline,
    load_pipeline,
    register_stage,
    STAGE_REGISTRY,
)


def make_mock_ctx(files=None):
    """Create a mock InstanceContext for pipeline tests."""
    ctx = MagicMock()
    ctx.instance_id = "test-1"

    storage = files or {}

    def mock_read(path):
        return storage.get(path, "")

    def mock_write(path, content):
        storage[path] = content

    ctx.read = MagicMock(side_effect=mock_read)
    ctx.write = MagicMock(side_effect=mock_write)
    ctx.read_inbox = MagicMock(return_value=[{"sender": "a", "body": "hi"}])

    mock_store = MagicMock()
    ctx.store = MagicMock(return_value=mock_store)
    ctx.shared_store = MagicMock(return_value=mock_store)
    ctx.config = MagicMock(return_value="config_value")

    return ctx


class TestResolveInput:
    def test_memory_file(self):
        ctx = make_mock_ctx({"thinking.md": "thoughts"})
        result = resolve_input(ctx, "memory.thinking", {})
        assert result == "thoughts"

    def test_pipeline_state(self):
        ctx = make_mock_ctx()
        state = {"guidance": "be helpful"}
        result = resolve_input(ctx, "pipeline.guidance", state)
        assert result == "be helpful"

    def test_inbox(self):
        ctx = make_mock_ctx()
        result = resolve_input(ctx, "inbox.messages", {})
        assert isinstance(result, list)
        assert result[0]["body"] == "hi"

    def test_config(self):
        ctx = make_mock_ctx()
        result = resolve_input(ctx, "config.entity_id", {})
        assert result == "config_value"

    def test_literal_fallback(self):
        ctx = make_mock_ctx()
        result = resolve_input(ctx, "just a string", {})
        assert result == "just a string"


class TestWriteOutput:
    def test_memory(self):
        ctx = make_mock_ctx()
        state = {}
        write_output(ctx, "memory.output", "result", state)
        ctx.write.assert_called_once_with("output.md", "result")

    def test_pipeline_state(self):
        ctx = make_mock_ctx()
        state = {}
        write_output(ctx, "pipeline.result", {"key": "value"}, state)
        assert state["result"] == {"key": "value"}


class TestApplyPreprocessor:
    def test_truncate(self):
        ctx = make_mock_ctx()
        result = apply_preprocessor(ctx, "a" * 5000, {"type": "truncate", "max_chars": 100})
        assert len(result) == 100

    def test_truncate_short(self):
        ctx = make_mock_ctx()
        result = apply_preprocessor(ctx, "short", {"type": "truncate", "max_chars": 100})
        assert result == "short"

    def test_unknown_preprocessor(self):
        ctx = make_mock_ctx()
        result = apply_preprocessor(ctx, "value", {"type": "unknown"})
        assert result == "value"


class TestRunStage:
    def test_runs_registered_stage(self):
        ctx = make_mock_ctx()

        def my_stage(ctx, **kwargs):
            return "stage result"

        register_stage("test_stage", my_stage)

        stage_def = {
            "stage": "test_stage",
            "outputs": {"result": "pipeline.result"},
        }

        state = {}
        run_stage(ctx, stage_def, state)
        assert state["result"] == "stage result"

        # Cleanup
        del STAGE_REGISTRY["test_stage"]

    def test_unknown_stage_raises(self):
        ctx = make_mock_ctx()
        with pytest.raises(Exception, match="Unknown stage"):
            run_stage(ctx, {"stage": "nonexistent"}, {})


class TestRunPipeline:
    def test_linear_pipeline(self):
        ctx = make_mock_ctx()

        def stage_a(ctx, **kwargs):
            return "output_a"

        def stage_b(ctx, **kwargs):
            return "output_b"

        register_stage("stage_a", stage_a)
        register_stage("stage_b", stage_b)

        steps = [
            {"stage": "stage_a", "outputs": {"result": "pipeline.a"}},
            {"stage": "stage_b", "outputs": {"result": "pipeline.b"}},
        ]

        state = run_pipeline(ctx, steps)
        assert state["a"] == "output_a"
        assert state["b"] == "output_b"

        del STAGE_REGISTRY["stage_a"]
        del STAGE_REGISTRY["stage_b"]


class TestLoadPipeline:
    def test_parses_yaml(self):
        yaml_text = """
species_id: test
pipeline:
  on_inbox:
    steps:
      - stage: gut_response
        inputs:
          events: inbox.messages
"""
        result = load_pipeline(yaml_text)
        assert result["species_id"] == "test"
        assert "pipeline" in result

    def test_invalid_yaml(self):
        with pytest.raises(Exception):
            load_pipeline("not: [valid: yaml: {")
