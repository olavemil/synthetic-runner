"""Matrix messaging adapter — send/receive via Matrix client-server API."""

from __future__ import annotations

import logging
import time

import httpx

from . import Event, MessagingAdapter
from library.harness.sanitize import strip_think_blocks

logger = logging.getLogger(__name__)


class MatrixAdapter(MessagingAdapter):
    def __init__(self, homeserver: str, access_token: str):
        self._homeserver = homeserver.rstrip("/")
        self._token = access_token
        self._client = httpx.Client(timeout=30)
        self._entity_id_cache: str | None = None

    @classmethod
    def login(
        cls,
        homeserver: str,
        user: str,
        password: str,
        device_name: str = "symbiosis",
    ) -> str:
        """Log in to a Matrix homeserver with username/password and return the access token."""
        url = f"{homeserver.rstrip('/')}/_matrix/client/v3/login"
        resp = httpx.post(
            url,
            json={
                "type": "m.login.password",
                "identifier": {"type": "m.id.user", "user": user},
                "password": password,
                "initial_device_display_name": device_name,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if "access_token" not in data:
            raise ValueError(f"Login response missing access_token: {data}")
        return data["access_token"]

    def _url(self, path: str) -> str:
        return f"{self._homeserver}/_matrix/client/v3{path}"

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._token}"}

    def get_entity_id(self) -> str | None:
        """Resolve and cache the authenticated Matrix user ID."""
        if self._entity_id_cache:
            return self._entity_id_cache

        resp = self._client.get(
            self._url("/account/whoami"),
            headers=self._headers(),
        )
        resp.raise_for_status()
        data = resp.json()
        user_id = str(data.get("user_id", "")).strip()
        if not user_id:
            raise ValueError(f"/account/whoami missing user_id: {data}")
        self._entity_id_cache = user_id
        return user_id

    def send(self, space_handle: str, message: str, reply_to: str | None = None) -> str:
        """Send a text message to a Matrix room. Returns event ID."""
        clean = strip_think_blocks(message)
        if not clean:
            return ""

        txn_id = f"m{int(time.time() * 1000)}"
        url = self._url(f"/rooms/{space_handle}/send/m.room.message/{txn_id}")

        body: dict = {
            "msgtype": "m.text",
            "body": clean,
        }
        if reply_to:
            body["m.relates_to"] = {
                "m.in_reply_to": {"event_id": reply_to}
            }

        resp = self._client.put(url, json=body, headers=self._headers())
        resp.raise_for_status()
        return resp.json().get("event_id", "")

    def poll(self, space_handle: str, since_token: str | None = None) -> tuple[list[Event], str]:
        """Poll for new messages using /sync with a room filter."""
        filter_json = {
            "room": {
                "rooms": [space_handle],
                "timeline": {"limit": 50},
            },
            "presence": {"types": []},
            "account_data": {"types": []},
        }

        params: dict = {
            "filter": str(filter_json).replace("'", '"'),
            "timeout": "0",
        }
        if since_token:
            params["since"] = since_token

        resp = self._client.get(
            self._url("/sync"),
            params=params,
            headers=self._headers(),
        )
        resp.raise_for_status()
        data = resp.json()

        next_token = data.get("next_batch", since_token or "")
        events = []

        rooms = data.get("rooms", {}).get("join", {})
        if space_handle in rooms:
            timeline = rooms[space_handle].get("timeline", {})
            for evt in timeline.get("events", []):
                if evt.get("type") != "m.room.message":
                    continue
                content = evt.get("content", {})
                if content.get("msgtype") != "m.text":
                    continue
                events.append(Event(
                    event_id=evt["event_id"],
                    sender=evt["sender"],
                    body=content.get("body", ""),
                    timestamp=evt.get("origin_server_ts", 0),
                    room=space_handle,
                ))

        return events, next_token

    def get_space_context(self, space_handle: str) -> dict:
        """Get room name, topic, and member list."""
        context: dict = {"room_id": space_handle, "name": "", "topic": "", "members": []}

        # Room state
        try:
            resp = self._client.get(
                self._url(f"/rooms/{space_handle}/state"),
                headers=self._headers(),
            )
            resp.raise_for_status()
            for event in resp.json():
                if event.get("type") == "m.room.name":
                    context["name"] = event.get("content", {}).get("name", "")
                elif event.get("type") == "m.room.topic":
                    context["topic"] = event.get("content", {}).get("topic", "")
                elif event.get("type") == "m.room.member":
                    if event.get("content", {}).get("membership") == "join":
                        context["members"].append(event.get("state_key", ""))
        except httpx.HTTPError as e:
            logger.warning("Failed to get room context for %s: %s", space_handle, e)

        return context

    def fetch_recent_messages(self, space_handle: str, limit: int = 50) -> list[Event]:
        """Fetch recent messages from a room using /messages endpoint."""
        resp = self._client.get(
            self._url(f"/rooms/{space_handle}/messages"),
            params={"dir": "b", "limit": str(limit)},
            headers=self._headers(),
        )
        resp.raise_for_status()
        data = resp.json()

        events = []
        for evt in reversed(data.get("chunk", [])):
            if evt.get("type") != "m.room.message":
                continue
            content = evt.get("content", {})
            if content.get("msgtype") != "m.text":
                continue
            events.append(Event(
                event_id=evt["event_id"],
                sender=evt["sender"],
                body=content.get("body", ""),
                timestamp=evt.get("origin_server_ts", 0),
                room=space_handle,
            ))

        return events
