"""Tests for LLM provider abstraction."""

import json
from unittest.mock import MagicMock, patch

import pytest

from library.harness.providers import LLMResponse, ToolCall


class TestLLMResponse:
    def test_defaults(self):
        r = LLMResponse(message="hello")
        assert r.message == "hello"
        assert r.tool_calls == []
        assert r.finish_reason == "stop"
        assert r.usage == {}

    def test_with_tool_calls(self):
        tc = ToolCall(id="1", name="read_file", arguments={"path": "test.md"})
        r = LLMResponse(message="", tool_calls=[tc], finish_reason="tool_calls")
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].name == "read_file"


class TestOpenAICompatProvider:
    def test_normalize_response(self):
        from library.harness.providers.openai_compat import OpenAICompatProvider

        provider = OpenAICompatProvider.__new__(OpenAICompatProvider)

        # Mock response
        mock_msg = MagicMock()
        mock_msg.content = "Hello world"
        mock_msg.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=20, total_tokens=30
        )

        result = provider._normalize(mock_response)
        assert isinstance(result, LLMResponse)
        assert result.message == "Hello world"
        assert result.finish_reason == "stop"
        assert result.usage["total_tokens"] == 30

    def test_think_block_stripping(self):
        from library.harness.providers.openai_compat import OpenAICompatProvider

        provider = OpenAICompatProvider.__new__(OpenAICompatProvider)

        mock_msg = MagicMock()
        mock_msg.content = "<think>internal thoughts</think>\nActual response"
        mock_msg.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        result = provider._normalize(mock_response)
        assert result.message == "Actual response"
        assert "<think>" not in result.message

    def test_tool_call_parsing(self):
        from library.harness.providers.openai_compat import OpenAICompatProvider

        provider = OpenAICompatProvider.__new__(OpenAICompatProvider)

        mock_tc = MagicMock()
        mock_tc.id = "call_123"
        mock_tc.function.name = "read_file"
        mock_tc.function.arguments = '{"path": "test.md"}'

        mock_msg = MagicMock()
        mock_msg.content = ""
        mock_msg.tool_calls = [mock_tc]

        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_choice.finish_reason = "tool_calls"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        result = provider._normalize(mock_response)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "read_file"
        assert result.tool_calls[0].arguments == {"path": "test.md"}


class TestAnthropicProvider:
    def test_convert_tools(self):
        from library.harness.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider.__new__(AnthropicProvider)

        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            }
        ]
        result = provider._convert_tools(openai_tools)
        assert len(result) == 1
        assert result[0]["name"] == "read_file"
        assert "input_schema" in result[0]

    def test_convert_tool_choice(self):
        from library.harness.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider.__new__(AnthropicProvider)

        assert provider._convert_tool_choice("auto") == {"type": "auto"}
        assert provider._convert_tool_choice("none") == {"type": "none"}
        assert provider._convert_tool_choice("required") == {"type": "any"}
        assert provider._convert_tool_choice({"function": {"name": "foo"}}) == {
            "type": "tool", "name": "foo"
        }

    def test_normalize_response(self):
        from library.harness.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider.__new__(AnthropicProvider)

        mock_text = MagicMock()
        mock_text.type = "text"
        mock_text.text = "Hello"

        mock_response = MagicMock()
        mock_response.content = [mock_text]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)

        result = provider._normalize(mock_response)
        assert result.message == "Hello"
        assert result.finish_reason == "stop"
        assert result.usage["prompt_tokens"] == 10

    def test_normalize_tool_use(self):
        from library.harness.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider.__new__(AnthropicProvider)

        mock_tool = MagicMock()
        mock_tool.type = "tool_use"
        mock_tool.id = "tu_123"
        mock_tool.name = "read_file"
        mock_tool.input = {"path": "test.md"}

        mock_response = MagicMock()
        mock_response.content = [mock_tool]
        mock_response.stop_reason = "tool_use"
        mock_response.usage = MagicMock(input_tokens=5, output_tokens=10)

        result = provider._normalize(mock_response)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "read_file"
        assert result.finish_reason == "tool_calls"
