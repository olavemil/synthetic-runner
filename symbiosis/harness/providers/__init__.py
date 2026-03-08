"""LLM provider abstraction — normalized interface across OpenAI-compat and Anthropic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    message: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict = field(default_factory=dict)


class LLMProvider(ABC):
    @abstractmethod
    def create(
        self,
        model: str,
        messages: list[dict],
        system: str | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        max_tokens: int = 4096,
        caller: str = "?",
    ) -> LLMResponse:
        ...
