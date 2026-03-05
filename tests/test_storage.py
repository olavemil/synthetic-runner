"""Tests for namespaced file storage."""

import pytest

from symbiosis.harness.storage import NamespacedStorage


class TestNamespacedStorage:
    def test_read_write(self, tmp_path):
        storage = NamespacedStorage(tmp_path, "test-instance")
        storage.write("hello.md", "# Hello")
        assert storage.read("hello.md") == "# Hello"

    def test_read_nonexistent(self, tmp_path):
        storage = NamespacedStorage(tmp_path, "test-instance")
        assert storage.read("missing.md") == ""

    def test_write_creates_dirs(self, tmp_path):
        storage = NamespacedStorage(tmp_path, "test-instance")
        storage.write("sub/dir/file.md", "content")
        assert storage.read("sub/dir/file.md") == "content"

    def test_list_files(self, tmp_path):
        storage = NamespacedStorage(tmp_path, "test-instance")
        storage.write("a.md", "a")
        storage.write("b.md", "b")
        storage.write("sub/c.md", "c")
        files = storage.list()
        assert "a.md" in files
        assert "b.md" in files
        assert "sub/c.md" in files

    def test_list_with_prefix(self, tmp_path):
        storage = NamespacedStorage(tmp_path, "test-instance")
        storage.write("rel/alice.md", "alice")
        storage.write("rel/bob.md", "bob")
        storage.write("other.md", "other")
        files = storage.list("rel")
        assert len(files) == 2
        assert all("rel/" in f for f in files)

    def test_exists(self, tmp_path):
        storage = NamespacedStorage(tmp_path, "test-instance")
        assert not storage.exists("nope.md")
        storage.write("yes.md", "y")
        assert storage.exists("yes.md")

    def test_delete(self, tmp_path):
        storage = NamespacedStorage(tmp_path, "test-instance")
        storage.write("temp.md", "temp")
        assert storage.exists("temp.md")
        assert storage.delete("temp.md")
        assert not storage.exists("temp.md")
        assert not storage.delete("temp.md")

    def test_path_escape_blocked(self, tmp_path):
        storage = NamespacedStorage(tmp_path, "test-instance")
        with pytest.raises(ValueError, match="escapes namespace"):
            storage.read("../../etc/passwd")

    def test_namespace_isolation(self, tmp_path):
        s1 = NamespacedStorage(tmp_path, "instance-1")
        s2 = NamespacedStorage(tmp_path, "instance-2")
        s1.write("data.md", "from instance 1")
        assert s2.read("data.md") == ""
