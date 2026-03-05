"""Anthropic LLM provider — wraps the Anthropic SDK with normalized responses."""

from __future__ import annotations

import json
import logging
import re
import time

import anthropic

from . import LLMProvider, LLMResponse, ToolCall

logger = logging.getLogger(__name__)

THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

RETRY_DELAYS = [10, 20, 40, 80]


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str | None = None):
        self._client = anthropic.Anthropic(api_key=api_key)

    def create(
        self,
        model: str,
        messages: list[dict],
        system: str | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        max_tokens: int = 1024,
        caller: str = "?",
    ) -> LLMResponse:
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._convert_tools(tools)
        if tool_choice is not None:
            kwargs["tool_choice"] = self._convert_tool_choice(tool_choice)

        response = self._call_with_retry(kwargs, caller)
        return self._normalize(response)

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert OpenAI-format tools to Anthropic format."""
        converted = []
        for tool in tools:
            if "function" in tool:
                fn = tool["function"]
                converted.append({
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                })
            else:
                converted.append(tool)
        return converted

    def _convert_tool_choice(self, choice: str | dict) -> dict:
        """Convert OpenAI tool_choice format to Anthropic format."""
        if isinstance(choice, str):
            if choice == "auto":
                return {"type": "auto"}
            if choice == "none":
                return {"type": "none"}
            if choice == "required":
                return {"type": "any"}
        if isinstance(choice, dict) and "function" in choice:
            return {"type": "tool", "name": choice["function"]["name"]}
        return {"type": "auto"}

    def _call_with_retry(self, kwargs: dict, caller: str):
        last_error = None
        for attempt, delay in enumerate(RETRY_DELAYS):
            try:
                return self._client.messages.create(**kwargs)
            except (anthropic.APIError, anthropic.APIConnectionError, anthropic.RateLimitError) as e:
                last_error = e
                logger.warning(
                    "[%s] Anthropic API error (attempt %d/%d): %s — retrying in %ds",
                    caller, attempt + 1, len(RETRY_DELAYS), e, delay,
                )
                time.sleep(delay)

        try:
            return self._client.messages.create(**kwargs)
        except (anthropic.APIError, anthropic.APIConnectionError, anthropic.RateLimitError):
            raise last_error  # type: ignore[misc]

    def _normalize(self, response) -> LLMResponse:
        content_parts = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else json.loads(block.input),
                    )
                )

        content = "\n".join(content_parts)
        content = THINK_PATTERN.sub("", content).strip()

        finish_reason_map = {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
            "max_tokens": "length",
        }

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }

        return LLMResponse(
            message=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason_map.get(response.stop_reason, response.stop_reason or "stop"),
            usage=usage,
        )
