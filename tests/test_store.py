"""Tests for SQLite structured store."""

import pytest

from symbiosis.harness.store import NamespacedStore, StoreDB, open_store


class TestStoreDB:
    def test_put_get(self):
        db = open_store()
        db.put("key1", {"a": 1})
        assert db.get("key1") == {"a": 1}

    def test_get_missing(self):
        db = open_store()
        assert db.get("missing") is None

    def test_put_overwrite(self):
        db = open_store()
        db.put("k", "first")
        db.put("k", "second")
        assert db.get("k") == "second"

    def test_delete(self):
        db = open_store()
        db.put("k", "val")
        db.delete("k")
        assert db.get("k") is None

    def test_scan(self):
        db = open_store()
        db.put("prefix:a", 1)
        db.put("prefix:b", 2)
        db.put("other:c", 3)
        results = db.scan("prefix:")
        assert len(results) == 2
        assert results[0] == ("prefix:a", 1)
        assert results[1] == ("prefix:b", 2)

    def test_count(self):
        db = open_store()
        db.put("x:1", "a")
        db.put("x:2", "b")
        db.put("y:1", "c")
        assert db.count("x:") == 2
        assert db.count("y:") == 1
        assert db.count("") == 3

    def test_claim_release(self):
        db = open_store()
        assert db.claim("task:1", "worker-a")
        assert not db.claim("task:1", "worker-b")
        assert db.claim("task:1", "worker-a")  # same owner OK
        assert db.release("task:1", "worker-a")
        assert db.claim("task:1", "worker-b")  # now available

    def test_release_wrong_owner(self):
        db = open_store()
        db.claim("task:1", "worker-a")
        assert not db.release("task:1", "worker-b")

    def test_release_nonexistent(self):
        db = open_store()
        assert not db.release("nope", "worker-a")


class TestNamespacedStore:
    def test_namespacing(self):
        db = open_store()
        s1 = NamespacedStore(db, "ns1")
        s2 = NamespacedStore(db, "ns2")

        s1.put("key", "from ns1")
        s2.put("key", "from ns2")

        assert s1.get("key") == "from ns1"
        assert s2.get("key") == "from ns2"

    def test_scan_scoped(self):
        db = open_store()
        s = NamespacedStore(db, "test")
        s.put("a", 1)
        s.put("b", 2)
        results = s.scan()
        assert len(results) == 2
        assert results[0][0] == "a"

    def test_count(self):
        db = open_store()
        s = NamespacedStore(db, "test")
        s.put("x:1", "a")
        s.put("x:2", "b")
        assert s.count("x:") == 2

    def test_claim_release(self):
        db = open_store()
        s = NamespacedStore(db, "test")
        assert s.claim("task", "owner1")
        assert not s.claim("task", "owner2")
        assert s.release("task", "owner1")
        assert s.claim("task", "owner2")

    def test_delete(self):
        db = open_store()
        s = NamespacedStore(db, "test")
        s.put("k", "v")
        s.delete("k")
        assert s.get("k") is None
