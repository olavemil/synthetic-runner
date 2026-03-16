"""InstanceContext — the core abstraction through which Species code interacts with infrastructure."""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from library.harness.storage import NamespacedStorage
from library.harness.store import NamespacedStore, StoreDB
from library.harness.mailbox import Mailbox

if TYPE_CHECKING:
    from library.harness.analytics import AnalyticsClient
    from library.harness.providers import LLMProvider, LLMResponse
    from library.harness.adapters import Event, MessagingAdapter
    from library.harness.config import InstanceConfig

logger = logging.getLogger(__name__)


class InstanceContext:
    def __init__(
        self,
        instance_id: str,
        species_id: str,
        storage: NamespacedStorage,
        provider: LLMProvider,
        default_model: str,
        adapter: MessagingAdapter | None,
        space_map: dict[str, str],
        store_db: StoreDB,
        mailbox: Mailbox,
        instance_config: InstanceConfig,
        analytics: AnalyticsClient | None = None,
    ):
        self._instance_id = instance_id
        self._species_id = species_id
        self._storage = storage
        self._provider = provider
        self._default_model = default_model
        self._adapter = adapter
        self._space_map = space_map
        self._store_db = store_db
        self._mailbox = mailbox
        self._instance_config = instance_config
        self._analytics = analytics
        self._send_allowed = True
        self._send_max: int | None = None
        self._send_reason = ""
        self._send_count = 0

    @property
    def instance_id(self) -> str:
        return self._instance_id

    @property
    def species_id(self) -> str:
        return self._species_id

    # --- Storage ---

    def read(self, path: str) -> str:
        result = self._storage.read(path)
        if self._analytics is not None:
            self._analytics.track("file_read", {"path": path})
        return result

    def write(self, path: str, content: str) -> None:
        self._storage.write(path, content)
        if self._analytics is not None:
            self._analytics.track("file_written", {"path": path, "length": len(content)})

    def list(self, prefix: str = "") -> list[str]:
        return self._storage.list(prefix)

    def exists(self, path: str) -> bool:
        return self._storage.exists(path)

    def read_binary(self, path: str) -> bytes | None:
        result = self._storage.read_binary(path)
        if self._analytics is not None:
            self._analytics.track("file_read", {"path": path, "binary": True})
        return result

    def write_binary(self, path: str, data: bytes) -> None:
        self._storage.write_binary(path, data)
        if self._analytics is not None:
            self._analytics.track("file_written", {"path": path, "size": len(data), "binary": True})

    # --- Config (read-only) ---

    def config(self, key: str) -> Any:
        if key == "instance_id":
            return self._instance_id
        if key == "species":
            return self._species_id
        if key == "provider":
            return self._instance_config.provider
        if key == "model":
            return self._instance_config.model
        if key == "entity_id" and self._instance_config.messaging:
            return self._instance_config.messaging.entity_id
        if key in self._instance_config.extra:
            return self._instance_config.extra[key]
        return None

    # --- LLM ---

    def llm(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        provider: str | None = None,
        system: str | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        max_tokens: int = 4096,
        caller: str = "?",
    ) -> LLMResponse:
        from library.harness.response_validator import is_response_pathological

        # Compatibility: toolkit helpers may pass a provider hint.
        # InstanceContext currently routes to the instance's configured provider.
        configured_provider = self._instance_config.provider
        if provider and provider != configured_provider:
            logger.warning(
                "Provider override '%s' requested by %s for instance '%s'; "
                "using configured provider '%s'",
                provider,
                caller,
                self._instance_id,
                configured_provider,
            )

        # Retry up to 3 times if response is pathological
        for attempt in range(1, 4):
            response = self._provider.create(
                model=model or self._default_model,
                messages=messages,
                system=system,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                caller=caller,
            )

            # Validate response message if no tool calls (tool outputs are trusted)
            if not response.tool_calls:
                is_bad, reason = is_response_pathological(response.message)
                if is_bad:
                    rejection_note = f" (attempt {attempt}/3: last output rejected — {reason})"
                    if attempt < 3:
                        # Add rejection context for retry
                        messages = [*messages, {
                            "role": "assistant",
                            "content": response.message,
                        }, {
                            "role": "user",
                            "content": f"That response was rejected due to: {reason}. Please regenerate.{rejection_note}",
                        }]
                        logger.info(
                            "[%s] Response rejected (attempt %d/3): %s — retrying",
                            caller, attempt, reason,
                        )
                        continue
                    else:
                        logger.warning(
                            "[%s] Response rejected after 3 attempts: %s — returning last attempt",
                            caller, reason,
                        )

            # Response is good or we're out of retries
            if self._analytics is not None:
                self._analytics.track("llm_called", {
                    "model": model or self._default_model,
                    "caller": caller,
                    "finish_reason": response.finish_reason,
                    "prompt_tokens": response.usage.get("input_tokens") or response.usage.get("prompt_tokens"),
                    "completion_tokens": response.usage.get("output_tokens") or response.usage.get("completion_tokens"),
                    "tool_call_count": len(response.tool_calls),
                })
                for tc in response.tool_calls:
                    self._analytics.track("tool_called", {"tool_name": tc.name, "caller": caller})
            return response

        # Should not reach here
        return response

    # --- Messaging ---

    def _resolve_space(self, space: str) -> str:
        # Preferred path: logical space name configured on the instance.
        handle = self._space_map.get(space)
        if handle is None:
            # Accept direct adapter handle (e.g. "!room:matrix.org") to allow
            # species/tool outputs to pass through concrete room IDs.
            if space in self._space_map.values():
                return space
            known_spaces = ", ".join(sorted(self._space_map)) or "(none)"
            raise KeyError(
                f"Space '{space}' not mapped for instance '{self._instance_id}'. "
                f"Known spaces: {known_spaces}"
            )
        return handle

    def can_send_reply(self) -> bool:
        """Check if enough thinks have occurred since last reply to allow sending.

        Returns True if min_thinks_per_reply is satisfied or not configured.
        """
        min_thinks = self._instance_config.schedule.get("min_thinks_per_reply")
        if min_thinks is None:
            return True  # No throttling configured
        
        try:
            min_thinks = int(min_thinks)
        except (ValueError, TypeError):
            return True  # Invalid config, allow sending
        
        from library.harness.store import NamespacedStore
        store = NamespacedStore(self._store_db, "checker")
        thinks_key = f"thinks_since_reply:{self._instance_id}"
        thinks_count = store.get(thinks_key) or 0
        
        if thinks_count < min_thinks:
            logger.info(
                "Instance '%s' throttled (thinks=%d < min=%d); skipping send",
                self._instance_id, thinks_count, min_thinks,
            )
            return False
        
        return True

    def _reset_thinks_counter(self) -> None:
        """Reset thinks_since_reply counter after sending a message."""
        from library.harness.store import NamespacedStore
        store = NamespacedStore(self._store_db, "checker")
        thinks_key = f"thinks_since_reply:{self._instance_id}"
        store.put(thinks_key, 0)
        logger.debug("Reset thinks_since_reply for %s", self._instance_id)

    def send(self, space: str, message: str, reply_to: str | None = None) -> str:
        if self._adapter is None:
            raise RuntimeError(f"No messaging adapter configured for instance '{self._instance_id}'")
        if not self._send_allowed:
            logger.info(
                "Blocked send for instance '%s' (reason=%s)",
                self._instance_id,
                self._send_reason or "send policy disallows messaging in this job",
            )
            return ""
        if self._send_max is not None and self._send_count >= self._send_max:
            logger.info(
                "Blocked send for instance '%s' (max_sends=%d reached)",
                self._instance_id,
                self._send_max,
            )
            return ""
        try:
            resolved_space = self._resolve_space(space)
        except KeyError:
            fallback_handle = self._space_map.get("main")
            if fallback_handle is None and len(self._space_map) == 1:
                fallback_handle = next(iter(self._space_map.values()))
            if fallback_handle is None:
                raise
            logger.warning(
                "Space '%s' not mapped for instance '%s'; falling back to '%s'",
                space,
                self._instance_id,
                fallback_handle,
            )
            resolved_space = fallback_handle
        event_id = self._adapter.send(resolved_space, message, reply_to)
        self._send_count += 1

        # Reset thinks counter after successful send
        self._reset_thinks_counter()

        if self._analytics is not None:
            self._analytics.track("message_sent", {
                "space": space,
                "has_reply_to": reply_to is not None,
            })

        return event_id

    def poll(self, space: str, since_token: str | None = None) -> tuple[list[Event], str]:
        if self._adapter is None:
            raise RuntimeError(f"No messaging adapter configured for instance '{self._instance_id}'")
        events, next_token = self._adapter.poll(self._resolve_space(space), since_token)
        if self._analytics is not None:
            self._analytics.track("messages_polled", {
                "space": space,
                "event_count": len(events),
            })
        return events, next_token

    def get_space_context(self, space: str) -> dict:
        if self._adapter is None:
            return {}
        return self._adapter.get_space_context(self._resolve_space(space))

    def get_all_space_contexts(self) -> dict[str, dict]:
        """Return context for all configured logical spaces."""
        if self._adapter is None:
            return {}
        result = {}
        for space_name, handle in self._space_map.items():
            try:
                result[space_name] = self._adapter.get_space_context(handle)
            except Exception:
                result[space_name] = {"room_id": handle}
        return result

    def list_spaces(self) -> list[str]:
        """Return configured logical messaging space names."""
        return sorted(self._space_map.keys())

    def configure_send_policy(
        self,
        *,
        allow_send: bool,
        max_sends: int | None = None,
        reason: str = "",
    ) -> None:
        self._send_allowed = bool(allow_send)
        self._send_max = max_sends if max_sends is None else max(0, int(max_sends))
        self._send_reason = reason
        self._send_count = 0

    @property
    def sent_message_count(self) -> int:
        return self._send_count

    # --- Inter-instance ---

    def send_to(self, target_instance_id: str, message: str) -> None:
        self._mailbox.send_to(target_instance_id, message)

    def read_inbox(self) -> list[dict]:
        return self._mailbox.read_inbox()

    # --- Structured store ---

    def store(self, namespace: str) -> NamespacedStore:
        return NamespacedStore(self._store_db, f"instance:{self._instance_id}:{namespace}")

    def shared_store(self, namespace: str) -> NamespacedStore:
        return NamespacedStore(self._store_db, f"species:{self._species_id}:{namespace}")

    # --- Config summary ---

    # --- Analytics ---

    def track(self, event_name: str, properties: dict | None = None) -> None:
        """Track an analytics event if an analytics client is configured."""
        if self._analytics is not None:
            self._analytics.track(event_name, properties)

    # --- Config summary ---

    def config_summary(self) -> dict:
        """Return all non-secret instance config as a plain dict."""
        d: dict = {
            "instance_id": self._instance_id,
            "species": self._species_id,
            "provider": self._instance_config.provider,
            "model": self._instance_config.model,
            "spaces": self.list_spaces(),
        }
        if self._instance_config.messaging and self._instance_config.messaging.entity_id:
            d["entity_id"] = self._instance_config.messaging.entity_id
        if self._instance_config.schedule:
            d["schedule"] = dict(self._instance_config.schedule)
        d.update(self._instance_config.extra)
        return d
