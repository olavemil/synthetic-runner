"""Messaging adapter abstraction — platform-agnostic send/receive."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Event:
    event_id: str
    sender: str
    body: str
    timestamp: int
    room: str | None = None


class MessagingAdapter(ABC):
    @abstractmethod
    def send(self, space_handle: str, message: str, reply_to: str | None = None) -> str:
        """Send a message. Returns event/message ID."""
        ...

    @abstractmethod
    def poll(self, space_handle: str, since_token: str | None = None) -> tuple[list[Event], str]:
        """Return (new_events, next_token)."""
        ...

    @abstractmethod
    def get_space_context(self, space_handle: str) -> dict:
        """Return name, topic, members for composing context blocks."""
        ...
