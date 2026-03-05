"""OpenAI-compatible LLM provider — covers OpenAI, LMStudio, and similar endpoints."""

from __future__ import annotations

import json
import logging
import re
import time

from openai import OpenAI, APIError, APIConnectionError, RateLimitError

from . import LLMProvider, LLMResponse, ToolCall

logger = logging.getLogger(__name__)

THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

RETRY_DELAYS = [10, 20, 40, 80]


class OpenAICompatProvider(LLMProvider):
    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key or "not-set",
        )

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
        full_messages = list(messages)
        if system:
            full_messages.insert(0, {"role": "system", "content": system})

        kwargs: dict = {
            "model": model,
            "messages": full_messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        response = self._call_with_retry(kwargs, caller)
        return self._normalize(response)

    def _call_with_retry(self, kwargs: dict, caller: str):
        last_error = None
        for attempt, delay in enumerate(RETRY_DELAYS):
            try:
                return self._client.chat.completions.create(**kwargs)
            except (APIError, APIConnectionError, RateLimitError) as e:
                last_error = e
                logger.warning(
                    "[%s] API error (attempt %d/%d): %s — retrying in %ds",
                    caller, attempt + 1, len(RETRY_DELAYS), e, delay,
                )
                time.sleep(delay)

        # Final attempt
        try:
            return self._client.chat.completions.create(**kwargs)
        except (APIError, APIConnectionError, RateLimitError):
            raise last_error  # type: ignore[misc]

    def _normalize(self, response) -> LLMResponse:
        choice = response.choices[0]
        message = choice.message

        content = message.content or ""
        content = THINK_PATTERN.sub("", content).strip()

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    args = json.loads(args)
                tool_calls.append(
                    ToolCall(id=tc.id, name=tc.function.name, arguments=args)
                )

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            message=content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
        )
