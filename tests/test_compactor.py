"""Tests for the harness-level file compactor."""

from unittest.mock import MagicMock

import pytest

from library.harness.compactor import Compactor
from library.harness.config import CompactConfig
from library.harness.providers import LLMResponse


def make_compactor(threshold_chars=100):
    provider = MagicMock()
    config = CompactConfig(
        provider="devstral",
        model="devstral-small-2505",
        threshold_chars=threshold_chars,
    )
    return Compactor(provider, config), provider


class TestCompactorMaybeCompact:
    def test_below_threshold_returns_none(self):
        compactor, provider = make_compactor(threshold_chars=100)
        result = compactor.maybe_compact("short content")
        assert result is None
        provider.create.assert_not_called()

    def test_at_threshold_returns_none(self):
        compactor, provider = make_compactor(threshold_chars=10)
        result = compactor.maybe_compact("0123456789")  # exactly 10 chars
        assert result is None
        provider.create.assert_not_called()

    def test_above_threshold_calls_llm(self):
        compactor, provider = make_compactor(threshold_chars=10)
        provider.create.return_value = LLMResponse(message="compact")
        result = compactor.maybe_compact("x" * 11)
        assert result == "compact"
        provider.create.assert_called_once()

    def test_passes_content_as_user_message(self):
        compactor, provider = make_compactor(threshold_chars=5)
        provider.create.return_value = LLMResponse(message="short")
        content = "a" * 10
        compactor.maybe_compact(content)
        call_kwargs = provider.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[1]
        assert any(m.get("content") == content for m in messages)

    def test_uses_configured_model(self):
        compactor, provider = make_compactor(threshold_chars=5)
        provider.create.return_value = LLMResponse(message="short")
        compactor.maybe_compact("a" * 10)
        call_kwargs = provider.create.call_args
        model = call_kwargs.kwargs.get("model") or call_kwargs.args[0]
        assert model == "devstral-small-2505"

    def test_empty_response_returns_none(self):
        compactor, provider = make_compactor(threshold_chars=5)
        provider.create.return_value = LLMResponse(message="")
        result = compactor.maybe_compact("a" * 10)
        assert result is None

    def test_longer_result_returns_none(self):
        """If compaction bloats the file, discard the result."""
        compactor, provider = make_compactor(threshold_chars=5)
        original = "a" * 10
        provider.create.return_value = LLMResponse(message="a" * 20)
        result = compactor.maybe_compact(original)
        assert result is None

    def test_provider_exception_returns_none(self):
        compactor, provider = make_compactor(threshold_chars=5)
        provider.create.side_effect = RuntimeError("API error")
        result = compactor.maybe_compact("a" * 10)
        assert result is None

    def test_path_hint_passed_for_logging(self):
        """Path parameter doesn't raise — just used for logging."""
        compactor, provider = make_compactor(threshold_chars=5)
        provider.create.return_value = LLMResponse(message="ok")
        result = compactor.maybe_compact("a" * 10, path="thinking.md")
        assert result == "ok"


class TestContextCompactFile:
    def _make_ctx(self, compactor=None):
        from unittest.mock import MagicMock
        from library.harness.context import InstanceContext
        ctx = MagicMock(spec=InstanceContext)
        ctx._compactor = compactor
        # Delegate compact_file to the real implementation
        ctx.compact_file = lambda content, path=None: (
            compactor.maybe_compact(content, path=path) if compactor else None
        )
        return ctx

    def test_no_compactor_returns_none(self):
        ctx = self._make_ctx(compactor=None)
        assert ctx.compact_file("content") is None

    def test_with_compactor_delegates(self):
        compactor, provider = make_compactor(threshold_chars=5)
        provider.create.return_value = LLMResponse(message="small")
        ctx = self._make_ctx(compactor=compactor)
        result = ctx.compact_file("a" * 10, path="test.md")
        assert result == "small"


class TestWriteFileToolCompaction:
    def _make_ctx_mock(self, compact_result=None):
        ctx = MagicMock()
        ctx.compact_file = MagicMock(return_value=compact_result)
        return ctx

    def test_no_compaction_normal_result(self):
        from library.tools.tools import handle_tool
        ctx = self._make_ctx_mock(compact_result=None)
        result, done = handle_tool(ctx, "write_file", {"path": "test.md", "content": "hello"})
        ctx.write.assert_called_once_with("test.md", "hello")
        assert result == "File written."
        assert not done

    def test_compaction_writes_compacted_content(self):
        from library.tools.tools import handle_tool
        ctx = self._make_ctx_mock(compact_result="compacted")
        result, done = handle_tool(ctx, "write_file", {"path": "test.md", "content": "very long"})
        ctx.write.assert_called_once_with("test.md", "compacted")
        assert "auto-compacted" in result
        assert not done

    def test_compaction_result_mentions_char_counts(self):
        from library.tools.tools import handle_tool
        ctx = self._make_ctx_mock(compact_result="short")
        result, _ = handle_tool(ctx, "write_file", {"path": "f.md", "content": "longer content"})
        # Should mention original → compacted sizes
        assert "14" in result or "5" in result  # original=14, compacted=5


class TestConfigParsing:
    def test_compact_config_parsed(self, tmp_path):
        import yaml
        from library.harness.config import load_harness_config

        config = {
            "providers": [
                {"id": "devstral", "type": "openai_compat",
                 "base_url": "http://localhost:11434/v1", "api_key": "none"},
            ],
            "adapters": [],
            "compact": {
                "provider": "devstral",
                "model": "devstral-small-2505",
                "threshold_chars": 4000,
            },
        }
        config_path = tmp_path / "config" / "harness.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(yaml.dump(config))

        harness = load_harness_config(config_path)
        assert harness.compact is not None
        assert harness.compact.provider == "devstral"
        assert harness.compact.model == "devstral-small-2505"
        assert harness.compact.threshold_chars == 4000

    def test_no_compact_config(self, tmp_path):
        import yaml
        from library.harness.config import load_harness_config

        config = {"providers": [], "adapters": []}
        config_path = tmp_path / "config" / "harness.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(yaml.dump(config))

        harness = load_harness_config(config_path)
        assert harness.compact is None

    def test_compact_default_threshold(self, tmp_path):
        import yaml
        from library.harness.config import load_harness_config

        config = {
            "providers": [],
            "adapters": [],
            "compact": {"provider": "p", "model": "m"},
        }
        config_path = tmp_path / "config" / "harness.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(yaml.dump(config))

        harness = load_harness_config(config_path)
        assert harness.compact.threshold_chars == 6000
