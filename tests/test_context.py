"""Tests for InstanceContext delegation."""

from unittest.mock import MagicMock

import pytest

from symbiosis.harness.context import InstanceContext
from symbiosis.harness.storage import NamespacedStorage
from symbiosis.harness.store import open_store
from symbiosis.harness.mailbox import Mailbox
from symbiosis.harness.config import InstanceConfig, MessagingConfig, SpaceMapping
from symbiosis.harness.adapters import Event


def make_ctx(tmp_path, adapter=None, messaging=True):
    """Build a test InstanceContext."""
    storage = NamespacedStorage(tmp_path / "instances", "test-1")
    provider = MagicMock()
    store_db = open_store()
    mailbox = Mailbox(tmp_path / "instances", "test-1")

    space_map = {"main": "!room:test"} if messaging else {}
    msg_config = MessagingConfig(
        adapter="test-adapter",
        entity_id="@bot:test",
        access_token="test-token",
        spaces=[SpaceMapping(name="main", handle="!room:test")],
    ) if messaging else None

    instance_config = InstanceConfig(
        instance_id="test-1",
        species="draum",
        provider="test-provider",
        model="test-model",
        messaging=msg_config,
    )

    return InstanceContext(
        instance_id="test-1",
        species_id="draum",
        storage=storage,
        provider=provider,
        default_model="test-model",
        adapter=adapter,
        space_map=space_map,
        store_db=store_db,
        mailbox=mailbox,
        instance_config=instance_config,
    )


class TestInstanceContext:
    def test_properties(self, tmp_path):
        ctx = make_ctx(tmp_path)
        assert ctx.instance_id == "test-1"
        assert ctx.species_id == "draum"

    def test_storage_delegates(self, tmp_path):
        ctx = make_ctx(tmp_path)
        ctx.write("test.md", "hello")
        assert ctx.read("test.md") == "hello"
        assert ctx.exists("test.md")
        assert "test.md" in ctx.list()

    def test_config_access(self, tmp_path):
        ctx = make_ctx(tmp_path)
        assert ctx.config("instance_id") == "test-1"
        assert ctx.config("species") == "draum"
        assert ctx.config("model") == "test-model"
        assert ctx.config("entity_id") == "@bot:test"
        assert ctx.config("nonexistent") is None

    def test_llm_delegates(self, tmp_path):
        ctx = make_ctx(tmp_path)
        ctx.llm([{"role": "user", "content": "hi"}])
        ctx._provider.create.assert_called_once()
        call_kwargs = ctx._provider.create.call_args
        assert call_kwargs[1]["model"] == "test-model"

    def test_llm_accepts_provider_keyword(self, tmp_path):
        ctx = make_ctx(tmp_path)
        ctx.llm(
            [{"role": "user", "content": "hi"}],
            provider="test-provider",
            model="override-model",
        )
        call_kwargs = ctx._provider.create.call_args
        assert call_kwargs[1]["model"] == "override-model"

    def test_send_delegates(self, tmp_path):
        adapter = MagicMock()
        adapter.send.return_value = "$event1"
        ctx = make_ctx(tmp_path, adapter=adapter)
        result = ctx.send("main", "hello")
        adapter.send.assert_called_once_with("!room:test", "hello", None)
        assert result == "$event1"

    def test_send_unknown_space_falls_back_to_main(self, tmp_path):
        adapter = MagicMock()
        adapter.send.return_value = "$event-fallback"
        ctx = make_ctx(tmp_path, adapter=adapter)
        result = ctx.send("unknown", "hi")
        adapter.send.assert_called_once_with("!room:test", "hi", None)
        assert result == "$event-fallback"

    def test_send_direct_handle(self, tmp_path):
        adapter = MagicMock()
        adapter.send.return_value = "$event2"
        ctx = make_ctx(tmp_path, adapter=adapter)
        result = ctx.send("!room:test", "hello")
        adapter.send.assert_called_once_with("!room:test", "hello", None)
        assert result == "$event2"

    def test_send_unknown_space_raises_without_any_fallback(self, tmp_path):
        adapter = MagicMock()
        storage = NamespacedStorage(tmp_path / "instances", "test-2")
        provider = MagicMock()
        store_db = open_store()
        mailbox = Mailbox(tmp_path / "instances", "test-2")
        instance_config = InstanceConfig(
            instance_id="test-2",
            species="draum",
            provider="test-provider",
            model="test-model",
            messaging=MessagingConfig(
                adapter="test-adapter",
                entity_id="@bot:test",
                access_token="test-token",
                spaces=[],
            ),
        )
        ctx = InstanceContext(
            instance_id="test-2",
            species_id="draum",
            storage=storage,
            provider=provider,
            default_model="test-model",
            adapter=adapter,
            space_map={},
            store_db=store_db,
            mailbox=mailbox,
            instance_config=instance_config,
        )
        with pytest.raises(KeyError, match="unknown"):
            ctx.send("unknown", "hi")

    def test_send_no_adapter(self, tmp_path):
        ctx = make_ctx(tmp_path, adapter=None, messaging=False)
        with pytest.raises(RuntimeError, match="No messaging adapter"):
            ctx.send("main", "hi")

    def test_send_policy_blocks_sends(self, tmp_path):
        adapter = MagicMock()
        adapter.send.return_value = "$event-blocked"
        ctx = make_ctx(tmp_path, adapter=adapter)
        ctx.configure_send_policy(allow_send=False, max_sends=0, reason="test")
        result = ctx.send("main", "hello")
        assert result == ""
        adapter.send.assert_not_called()

    def test_send_policy_limits_to_one_send(self, tmp_path):
        adapter = MagicMock()
        adapter.send.side_effect = ["$event-1", "$event-2"]
        ctx = make_ctx(tmp_path, adapter=adapter)
        ctx.configure_send_policy(allow_send=True, max_sends=1, reason="test")
        first = ctx.send("main", "hello")
        second = ctx.send("main", "again")
        assert first == "$event-1"
        assert second == ""
        assert ctx.sent_message_count == 1
        adapter.send.assert_called_once_with("!room:test", "hello", None)

    def test_get_all_space_contexts_no_adapter(self, tmp_path):
        ctx = make_ctx(tmp_path, adapter=None, messaging=False)
        assert ctx.get_all_space_contexts() == {}

    def test_get_all_space_contexts_delegates(self, tmp_path):
        adapter = MagicMock()
        adapter.get_space_context.return_value = {
            "room_id": "!room:test",
            "name": "Main Room",
            "topic": "General",
            "members": ["@a:test", "@b:test"],
        }
        ctx = make_ctx(tmp_path, adapter=adapter)
        result = ctx.get_all_space_contexts()
        assert "main" in result
        assert result["main"]["name"] == "Main Room"
        assert len(result["main"]["members"]) == 2

    def test_get_all_space_contexts_handles_exception(self, tmp_path):
        adapter = MagicMock()
        adapter.get_space_context.side_effect = Exception("network error")
        ctx = make_ctx(tmp_path, adapter=adapter)
        result = ctx.get_all_space_contexts()
        assert "main" in result
        assert result["main"] == {"room_id": "!room:test"}

    def test_poll_delegates(self, tmp_path):
        adapter = MagicMock()
        adapter.poll.return_value = ([], "token1")
        ctx = make_ctx(tmp_path, adapter=adapter)
        events, token = ctx.poll("main", since_token="t0")
        adapter.poll.assert_called_once_with("!room:test", "t0")

    def test_store_namespaced(self, tmp_path):
        ctx = make_ctx(tmp_path)
        store = ctx.store("data")
        store.put("key", "value")
        assert store.get("key") == "value"

    def test_shared_store(self, tmp_path):
        ctx = make_ctx(tmp_path)
        shared = ctx.shared_store("election")
        shared.put("vote:1", {"rank": 1})
        assert shared.get("vote:1") == {"rank": 1}


class TestListRoomsTool:
    def test_list_rooms_returns_room_info(self, tmp_path):
        adapter = MagicMock()
        adapter.get_space_context.return_value = {
            "room_id": "!room:test",
            "name": "Main Room",
            "topic": "General chat",
            "members": ["@a:test", "@b:test"],
        }
        ctx = make_ctx(tmp_path, adapter=adapter)
        from symbiosis.toolkit.tools import handle_tool
        result, is_done = handle_tool(ctx, "list_rooms", {})
        assert not is_done
        assert "Main Room" in result
        assert "General chat" in result
        assert "Members: 2" in result

    def test_list_rooms_no_adapter(self, tmp_path):
        ctx = make_ctx(tmp_path, adapter=None, messaging=False)
        from symbiosis.toolkit.tools import handle_tool
        result, is_done = handle_tool(ctx, "list_rooms", {})
        assert "no rooms" in result

    def test_list_rooms_tool_in_make_tools(self, tmp_path):
        ctx = make_ctx(tmp_path)
        from symbiosis.toolkit.tools import make_tools
        tools = make_tools(ctx)
        tool_names = [t["function"]["name"] for t in tools]
        assert "list_rooms" in tool_names

    def test_list_rooms_tool_gated_by_option(self, tmp_path):
        ctx = make_ctx(tmp_path)
        from symbiosis.toolkit.tools import make_tools
        tools = make_tools(ctx, options={"rooms": False})
        tool_names = [t["function"]["name"] for t in tools]
        assert "list_rooms" not in tool_names


class TestIntrospectTool:
    def test_introspect_returns_species_description_and_config(self, tmp_path):
        ctx = make_ctx(tmp_path)
        from symbiosis.toolkit.tools import handle_tool
        result, is_done = handle_tool(ctx, "introspect", {})
        assert not is_done
        assert "## Instance Config" in result
        assert "test-1" in result
        assert "draum" in result

    def test_introspect_includes_spaces(self, tmp_path):
        ctx = make_ctx(tmp_path)
        from symbiosis.toolkit.tools import handle_tool
        result, _ = handle_tool(ctx, "introspect", {})
        assert "main" in result

    def test_introspect_tool_in_make_tools(self, tmp_path):
        ctx = make_ctx(tmp_path)
        from symbiosis.toolkit.tools import make_tools
        tools = make_tools(ctx)
        tool_names = [t["function"]["name"] for t in tools]
        assert "introspect" in tool_names

    def test_introspect_tool_gated_by_option(self, tmp_path):
        ctx = make_ctx(tmp_path)
        from symbiosis.toolkit.tools import make_tools
        tools = make_tools(ctx, options={"introspect": False})
        tool_names = [t["function"]["name"] for t in tools]
        assert "introspect" not in tool_names

    def test_config_summary_returns_expected_keys(self, tmp_path):
        ctx = make_ctx(tmp_path)
        summary = ctx.config_summary()
        assert summary["instance_id"] == "test-1"
        assert summary["species"] == "draum"
        assert summary["provider"] == "test-provider"
        assert summary["model"] == "test-model"
        assert "main" in summary["spaces"]
        assert summary["entity_id"] == "@bot:test"


class TestFormatRoomsContext:
    def test_format_rooms_context(self, tmp_path):
        adapter = MagicMock()
        adapter.get_space_context.return_value = {
            "room_id": "!room:test",
            "name": "Main Room",
            "topic": "General chat",
            "members": ["@a:test"],
        }
        ctx = make_ctx(tmp_path, adapter=adapter)
        from symbiosis.toolkit.prompts import format_rooms_context
        result = format_rooms_context(ctx)
        assert "## Rooms" in result
        assert "**Main Room**" in result
        assert "General chat" in result
        assert "1 members" in result

    def test_format_rooms_context_no_adapter(self, tmp_path):
        ctx = make_ctx(tmp_path, adapter=None, messaging=False)
        from symbiosis.toolkit.prompts import format_rooms_context
        assert format_rooms_context(ctx) == ""


class TestMailboxIntegration:
    def test_send_and_read(self, tmp_path):
        base = tmp_path / "instances"
        ctx_a = InstanceContext(
            instance_id="a",
            species_id="test",
            storage=NamespacedStorage(base, "a"),
            provider=MagicMock(),
            default_model="m",
            adapter=None,
            space_map={},
            store_db=open_store(),
            mailbox=Mailbox(base, "a"),
            instance_config=InstanceConfig(
                instance_id="a", species="test", provider="p", model="m"
            ),
        )
        ctx_b = InstanceContext(
            instance_id="b",
            species_id="test",
            storage=NamespacedStorage(base, "b"),
            provider=MagicMock(),
            default_model="m",
            adapter=None,
            space_map={},
            store_db=open_store(),
            mailbox=Mailbox(base, "b"),
            instance_config=InstanceConfig(
                instance_id="b", species="test", provider="p", model="m"
            ),
        )

        ctx_a.send_to("b", "hello from a")
        messages = ctx_b.read_inbox()
        assert len(messages) == 1
        assert messages[0]["sender"] == "a"
        assert messages[0]["body"] == "hello from a"

        # Inbox is cleared after read
        assert ctx_b.read_inbox() == []
