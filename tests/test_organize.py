"""Tests for knowledge organization tools."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from library.tools.organize import (
    handle_organize_tool,
    ORGANIZE_TOOL_SCHEMAS,
    DEFAULT_CATEGORIES,
    _list_category_names,
    _list_topics_in_category,
    count_all_topics,
)


def make_mock_ctx(files=None):
    """Create a mock InstanceContext with file storage.

    The mock ctx.list() returns paths relative to instance root (matching
    NamespacedStorage behavior), and ctx.exists() checks both presence and
    non-empty content.
    """
    _files = files if files is not None else {}
    ctx = MagicMock()
    ctx.read = MagicMock(side_effect=lambda p: _files.get(p, ""))
    ctx.write = MagicMock(side_effect=lambda p, c: _files.update({p: c}))
    ctx.exists = MagicMock(side_effect=lambda p: p in _files and bool(_files[p]))
    ctx.list = MagicMock(side_effect=lambda prefix="": [
        k for k in sorted(_files.keys())
        if k.startswith(prefix) and bool(_files[k])
    ])
    ctx._files = _files
    return ctx


class TestOrganizeToolSchemas:
    def test_schema_count(self):
        assert len(ORGANIZE_TOOL_SCHEMAS) == 10

    def test_all_have_names(self):
        names = [s["function"]["name"] for s in ORGANIZE_TOOL_SCHEMAS]
        assert "organize_list_categories" in names
        assert "organize_write_topic" in names
        assert "organize_archive_thoughts" in names
        assert "organize_list_archives" in names
        assert "organize_read_archive" in names


class TestListCategories:
    def test_empty_seeds_defaults(self):
        ctx = make_mock_ctx()
        result = handle_organize_tool(ctx, "organize_list_categories", {})
        for cat in DEFAULT_CATEGORIES:
            assert cat in result

    def test_with_existing_categories(self):
        ctx = make_mock_ctx({
            "knowledge/concepts/_meta.md": "# concepts\n",
            "knowledge/concepts/trust.md": "Trust is important.",
        })
        result = handle_organize_tool(ctx, "organize_list_categories", {})
        assert "concepts" in result
        assert "1 topics" in result


class TestCreateCategory:
    def test_create_new(self):
        ctx = make_mock_ctx()
        result = handle_organize_tool(ctx, "organize_create_category",
                                      {"name": "beliefs", "description": "Core beliefs"})
        assert "Created" in result

    def test_create_existing(self):
        ctx = make_mock_ctx({"knowledge/concepts/_meta.md": "# concepts\n"})
        result = handle_organize_tool(ctx, "organize_create_category", {"name": "concepts"})
        assert "already exists" in result


class TestWriteTopic:
    def test_create_topic(self):
        ctx = make_mock_ctx()
        result = handle_organize_tool(ctx, "organize_write_topic", {
            "category": "concepts",
            "topic": "trust",
            "content": "Trust is earned.",
        })
        assert "Created" in result
        ctx.write.assert_any_call("knowledge/concepts/trust.md", "Trust is earned.")

    def test_update_topic(self):
        ctx = make_mock_ctx({
            "knowledge/concepts/_meta.md": "# concepts\n",
            "knowledge/concepts/trust.md": "Old content.",
        })
        result = handle_organize_tool(ctx, "organize_write_topic", {
            "category": "concepts",
            "topic": "trust",
            "content": "Updated content.",
        })
        assert "Updated" in result

    def test_auto_creates_category(self):
        ctx = make_mock_ctx()
        handle_organize_tool(ctx, "organize_write_topic", {
            "category": "new_cat",
            "topic": "topic1",
            "content": "Content.",
        })
        # Should have created _meta.md for the new category
        meta_calls = [c for c in ctx.write.call_args_list
                      if c[0][0] == "knowledge/new_cat/_meta.md"]
        assert len(meta_calls) > 0


class TestReadTopic:
    def test_read_existing(self):
        ctx = make_mock_ctx({
            "knowledge/concepts/_meta.md": "# concepts\n",
            "knowledge/concepts/trust.md": "Trust content.",
        })
        result = handle_organize_tool(ctx, "organize_read_topic",
                                      {"category": "concepts", "topic": "trust"})
        assert result == "Trust content."

    def test_read_missing(self):
        ctx = make_mock_ctx()
        result = handle_organize_tool(ctx, "organize_read_topic",
                                      {"category": "concepts", "topic": "missing"})
        assert "not found" in result


class TestRemoveTopic:
    def test_remove_existing(self):
        ctx = make_mock_ctx({
            "knowledge/concepts/_meta.md": "# concepts\n",
            "knowledge/concepts/trust.md": "Content.",
        })
        result = handle_organize_tool(ctx, "organize_remove_topic",
                                      {"category": "concepts", "topic": "trust"})
        assert "Removed" in result

    def test_remove_missing(self):
        ctx = make_mock_ctx()
        result = handle_organize_tool(ctx, "organize_remove_topic",
                                      {"category": "concepts", "topic": "missing"})
        assert "not found" in result


class TestListTopics:
    def test_list_existing(self):
        ctx = make_mock_ctx({
            "knowledge/concepts/_meta.md": "# concepts\n",
            "knowledge/concepts/trust.md": "Trust is important.",
            "knowledge/concepts/honesty.md": "Honesty matters.",
        })
        result = handle_organize_tool(ctx, "organize_list_topics", {"category": "concepts"})
        assert "trust" in result
        assert "honesty" in result

    def test_list_empty_category(self):
        ctx = make_mock_ctx({"knowledge/concepts/_meta.md": "# concepts\n"})
        result = handle_organize_tool(ctx, "organize_list_topics", {"category": "concepts"})
        assert "No topics" in result

    def test_list_nonexistent_category(self):
        ctx = make_mock_ctx()
        result = handle_organize_tool(ctx, "organize_list_topics", {"category": "missing"})
        assert "does not exist" in result


class TestRemoveCategory:
    def test_remove_with_merge(self):
        ctx = make_mock_ctx({
            "knowledge/old/_meta.md": "# old\n",
            "knowledge/old/topic1.md": "Content 1.",
            "knowledge/new/_meta.md": "# new\n",
        })
        result = handle_organize_tool(ctx, "organize_remove_category",
                                      {"name": "old", "merge_into": "new"})
        assert "Merged" in result
        ctx.write.assert_any_call("knowledge/new/topic1.md", "Content 1.")

    def test_remove_without_merge(self):
        ctx = make_mock_ctx({
            "knowledge/old/_meta.md": "# old\n",
            "knowledge/old/topic1.md": "Content.",
        })
        result = handle_organize_tool(ctx, "organize_remove_category", {"name": "old"})
        assert "Removed" in result
        assert "1 topics deleted" in result

    def test_remove_nonexistent(self):
        ctx = make_mock_ctx()
        result = handle_organize_tool(ctx, "organize_remove_category", {"name": "missing"})
        assert "does not exist" in result


class TestArchiveThoughts:
    def test_archive_all(self):
        ctx = make_mock_ctx({"thinking.md": "# Thinking\n\nSome deep thoughts here."})
        result = handle_organize_tool(ctx, "organize_archive_thoughts", {})
        assert "Archived" in result
        assert "chars" in result

    def test_archive_before_marker(self):
        thinking = "Old thoughts.\n\n---MARKER---\n\nNew thoughts."
        ctx = make_mock_ctx({"thinking.md": thinking})
        result = handle_organize_tool(ctx, "organize_archive_thoughts",
                                      {"before_marker": "---MARKER---"})
        assert "Archived" in result
        # Remaining should start from marker
        remaining_calls = [c for c in ctx.write.call_args_list if c[0][0] == "thinking.md"]
        assert remaining_calls
        assert "New thoughts" in remaining_calls[-1][0][1]

    def test_archive_empty(self):
        ctx = make_mock_ctx({"thinking.md": ""})
        result = handle_organize_tool(ctx, "organize_archive_thoughts", {})
        assert "empty" in result

    def test_archive_with_label(self):
        ctx = make_mock_ctx({"thinking.md": "Some thoughts."})
        result = handle_organize_tool(ctx, "organize_archive_thoughts", {"label": "session_42"})
        assert "session_42" in result
        ctx.write.assert_any_call("archive/session_42.md", "Some thoughts.")


class TestCountTopics:
    def test_empty(self):
        ctx = make_mock_ctx()
        assert count_all_topics(ctx) == 0

    def test_counts_across_categories(self):
        ctx = make_mock_ctx({
            "knowledge/concepts/_meta.md": "# concepts\n",
            "knowledge/concepts/a.md": "A",
            "knowledge/concepts/b.md": "B",
            "knowledge/entities/_meta.md": "# entities\n",
            "knowledge/entities/alice.md": "Alice",
        })
        assert count_all_topics(ctx) == 3


class TestListArchives:
    def test_empty(self):
        ctx = make_mock_ctx()
        result = handle_organize_tool(ctx, "organize_list_archives", {})
        assert "No archive" in result

    def test_with_entries(self):
        ctx = make_mock_ctx({
            "archive/2024-01-01_120000.md": "Old thoughts.",
            "archive/2024-01-02_090000.md": "More thoughts.",
        })
        result = handle_organize_tool(ctx, "organize_list_archives", {})
        assert "2024-01-01_120000" in result
        assert "2024-01-02_090000" in result


class TestReadArchive:
    def test_read_existing(self):
        ctx = make_mock_ctx({"archive/session_42.md": "Archived content here."})
        result = handle_organize_tool(ctx, "organize_read_archive", {"label": "session_42"})
        assert result == "Archived content here."

    def test_read_missing(self):
        ctx = make_mock_ctx()
        result = handle_organize_tool(ctx, "organize_read_archive", {"label": "nonexistent"})
        assert "not found" in result


class TestOrganizeLogging:
    def test_tool_calls_are_logged(self, caplog):
        import logging
        ctx = make_mock_ctx()
        with caplog.at_level(logging.INFO, logger="library.tools.organize"):
            handle_organize_tool(ctx, "organize_list_categories", {})
        assert any("organize_list_categories" in r.message for r in caplog.records)

    def test_write_topic_logged(self, caplog):
        import logging
        ctx = make_mock_ctx()
        with caplog.at_level(logging.INFO, logger="library.tools.organize"):
            handle_organize_tool(ctx, "organize_write_topic", {
                "category": "concepts", "topic": "trust", "content": "Trust is earned.",
            })
        assert any("created" in r.message for r in caplog.records)


class TestUnknownTool:
    def test_unknown_tool(self):
        ctx = make_mock_ctx()
        result = handle_organize_tool(ctx, "organize_foobar", {})
        assert "Unknown" in result
