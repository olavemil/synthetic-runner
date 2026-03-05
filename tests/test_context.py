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

    def test_send_delegates(self, tmp_path):
        adapter = MagicMock()
        adapter.send.return_value = "$event1"
        ctx = make_ctx(tmp_path, adapter=adapter)
        result = ctx.send("main", "hello")
        adapter.send.assert_called_once_with("!room:test", "hello", None)
        assert result == "$event1"

    def test_send_unknown_space(self, tmp_path):
        adapter = MagicMock()
        ctx = make_ctx(tmp_path, adapter=adapter)
        with pytest.raises(KeyError, match="unknown"):
            ctx.send("unknown", "hi")

    def test_send_no_adapter(self, tmp_path):
        ctx = make_ctx(tmp_path, adapter=None, messaging=False)
        with pytest.raises(RuntimeError, match="No messaging adapter"):
            ctx.send("main", "hi")

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
