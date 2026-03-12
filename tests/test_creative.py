"""Tests for creative artifact tools."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from library.tools.creative import (
    CREATIVE_TYPES,
    _slugify,
    _sanitize_svg,
    _parse_metadata,
    _get_body,
    _wrap_content,
    handle_creative_tool,
)


def make_mock_ctx(files=None):
    _files = files if files is not None else {}
    ctx = MagicMock()
    ctx.read = MagicMock(side_effect=lambda p: _files.get(p, ""))
    ctx.write = MagicMock(side_effect=lambda p, c: _files.update({p: c}))
    ctx.list = MagicMock(side_effect=lambda prefix="": [
        k for k in sorted(_files.keys())
        if k.startswith(prefix) and _files[k].strip()
    ])
    return ctx


class TestSlugify:
    def test_basic(self):
        assert _slugify("Hello World") == "hello-world"

    def test_special_chars(self):
        assert _slugify("My Poem! (v2)") == "my-poem-v2"

    def test_long_title_truncated(self):
        slug = _slugify("a" * 100)
        assert len(slug) <= 60

    def test_empty_title(self):
        assert _slugify("!!!") == "untitled"


class TestSanitizeSvg:
    def test_strips_script_tags(self):
        svg = '<svg><script>alert("xss")</script><circle r="10"/></svg>'
        result = _sanitize_svg(svg)
        assert "<script" not in result
        assert "<circle" in result

    def test_strips_on_attributes(self):
        svg = '<svg><rect onclick="alert(1)" width="10"/></svg>'
        result = _sanitize_svg(svg)
        assert "onclick" not in result
        assert "width" in result

    def test_strips_foreign_object(self):
        svg = '<svg><foreignObject><div>bad</div></foreignObject><circle/></svg>'
        result = _sanitize_svg(svg)
        assert "<foreignObject" not in result
        assert "<circle" in result

    def test_clean_svg_unchanged(self):
        svg = '<svg><circle cx="50" cy="50" r="40" fill="red"/></svg>'
        assert _sanitize_svg(svg) == svg


class TestMetadata:
    def test_parse_yaml_frontmatter(self):
        raw = '---\ntitle: "My Poem"\ntype: poetry\ncreated: 1000\nupdated: 2000\n---\n\nContent here.'
        meta = _parse_metadata(raw)
        assert meta["title"] == "My Poem"
        assert meta["type"] == "poetry"
        assert meta["created"] == "1000"

    def test_parse_comment_header(self):
        raw = "<!-- title: My Art | type: art | created: 1000 | updated: 2000 -->\n<svg></svg>"
        meta = _parse_metadata(raw)
        assert meta["title"] == "My Art"
        assert meta["type"] == "art"

    def test_get_body_strips_frontmatter(self):
        raw = '---\ntitle: "Test"\ntype: narrative\ncreated: 1\nupdated: 1\n---\n\nThe story begins.'
        body = _get_body(raw)
        assert body.strip() == "The story begins."
        assert "---" not in body

    def test_get_body_strips_comment_header(self):
        raw = "<!-- title: X | type: art -->\n<svg>content</svg>"
        body = _get_body(raw)
        assert body.strip() == "<svg>content</svg>"


class TestWrapContent:
    def test_markdown_type_uses_frontmatter(self):
        result = _wrap_content("My Poem", "poetry", "Roses are red.", 100, 200)
        assert result.startswith("---\n")
        assert 'title: "My Poem"' in result
        assert "type: poetry" in result
        assert "Roses are red." in result

    def test_svg_type_uses_comment_header(self):
        result = _wrap_content("My Art", "art", "<svg></svg>", 100, 200)
        assert result.startswith("<!-- ")
        assert "title: My Art" in result
        assert "<svg></svg>" in result

    def test_abc_type_uses_comment_header(self):
        result = _wrap_content("My Song", "music", "X:1\nT:Test\nK:C\nCDEF|", 100, 200)
        assert result.startswith("<!-- ")
        assert "type: music" in result


class TestHandleCreativeTool:
    def test_list_empty(self):
        ctx = make_mock_ctx()
        result = handle_creative_tool(ctx, "creative_list", {})
        assert "no creations" in result.lower()

    def test_new_and_list(self):
        files = {}
        ctx = make_mock_ctx(files)
        result = handle_creative_tool(ctx, "creative_new", {
            "type": "poetry",
            "title": "First Poem",
            "content": "Roses are red.",
        })
        assert "Created" in result
        assert "first-poem" in result

        # File should exist
        assert any("first-poem" in k for k in files)

        # List should show it
        result = handle_creative_tool(ctx, "creative_list", {})
        assert "first-poem" in result
        assert "poetry" in result

    def test_read(self):
        files = {}
        ctx = make_mock_ctx(files)
        handle_creative_tool(ctx, "creative_new", {
            "type": "narrative",
            "title": "A Story",
            "content": "Once upon a time.",
        })
        result = handle_creative_tool(ctx, "creative_read", {"slug": "a-story"})
        assert "A Story" in result
        assert "Once upon a time." in result

    def test_read_not_found(self):
        ctx = make_mock_ctx()
        result = handle_creative_tool(ctx, "creative_read", {"slug": "nonexistent"})
        assert "not found" in result.lower()

    def test_edit(self):
        files = {}
        ctx = make_mock_ctx(files)
        handle_creative_tool(ctx, "creative_new", {
            "type": "poetry",
            "title": "Draft",
            "content": "First draft.",
        })
        result = handle_creative_tool(ctx, "creative_edit", {
            "slug": "draft",
            "content": "Revised version.",
        })
        assert "Updated" in result

        # Verify content changed
        read_result = handle_creative_tool(ctx, "creative_read", {"slug": "draft"})
        assert "Revised version." in read_result

    def test_delete(self):
        files = {}
        ctx = make_mock_ctx(files)
        handle_creative_tool(ctx, "creative_new", {
            "type": "poetry",
            "title": "To Delete",
            "content": "Temporary.",
        })
        result = handle_creative_tool(ctx, "creative_delete", {"slug": "to-delete"})
        assert "Deleted" in result

        # Should no longer list
        result = handle_creative_tool(ctx, "creative_list", {})
        assert "no creations" in result.lower()

    def test_new_invalid_type(self):
        ctx = make_mock_ctx()
        result = handle_creative_tool(ctx, "creative_new", {
            "type": "invalid",
            "title": "Bad",
            "content": "x",
        })
        assert "Unknown type" in result

    def test_new_size_limit(self):
        ctx = make_mock_ctx()
        result = handle_creative_tool(ctx, "creative_new", {
            "type": "music",
            "title": "Too Big",
            "content": "x" * 30_000,
        })
        assert "too large" in result.lower()

    def test_new_art_sanitized(self):
        files = {}
        ctx = make_mock_ctx(files)
        handle_creative_tool(ctx, "creative_new", {
            "type": "art",
            "title": "Safe Art",
            "content": '<svg><script>alert("xss")</script><circle r="10"/></svg>',
        })
        # The stored content should not have script tags
        stored = files.get("creations/safe-art.svg", "")
        assert "<script" not in stored
        assert "<circle" in stored

    def test_duplicate_slug_gets_suffix(self):
        files = {}
        ctx = make_mock_ctx(files)
        handle_creative_tool(ctx, "creative_new", {
            "type": "poetry",
            "title": "Same Name",
            "content": "First.",
        })
        result = handle_creative_tool(ctx, "creative_new", {
            "type": "poetry",
            "title": "Same Name",
            "content": "Second.",
        })
        assert "Created" in result
        # Should have two files
        creation_files = [k for k in files if k.startswith("creations/")]
        assert len(creation_files) == 2

    def test_unknown_tool(self):
        ctx = make_mock_ctx()
        result = handle_creative_tool(ctx, "creative_unknown", {})
        assert "Unknown" in result


class TestToolDispatch:
    def test_handle_tool_dispatches_creative(self):
        from library.tools.tools import handle_tool

        files = {}
        ctx = make_mock_ctx(files)
        result, is_done = handle_tool(ctx, "creative_list", {})
        assert not is_done
        assert "no creations" in result.lower()


class TestScopeFiltering:
    def test_creative_write_tools_in_create_scopes(self):
        from library.tools.phases import CREATE_SCOPES, get_tools_for_scopes

        tools = get_tools_for_scopes(CREATE_SCOPES, creative=True)
        names = [t["function"]["name"] for t in tools]
        assert "creative_new" in names
        assert "creative_list" in names
        assert "creative_edit" in names

    def test_creative_write_tools_not_in_think_scopes(self):
        from library.tools.phases import THINK_SCOPES, get_tools_for_scopes

        tools = get_tools_for_scopes(THINK_SCOPES, creative=True)
        names = [t["function"]["name"] for t in tools]
        assert "creative_new" not in names
        assert "creative_edit" not in names
        assert "creative_list" not in names

    def test_creative_write_tools_not_in_organize_scopes(self):
        from library.tools.phases import ORGANIZE_SCOPES, get_tools_for_scopes

        tools = get_tools_for_scopes(ORGANIZE_SCOPES, creative=True)
        names = [t["function"]["name"] for t in tools]
        assert "creative_new" not in names
        assert "creative_list" not in names


class TestGalleryRendering:
    def test_render_empty_gallery(self):
        from library.tools.rendering import render_creations_gallery_html

        html = render_creations_gallery_html([], title="Test Gallery")
        assert "Test Gallery" in html
        assert "0 works" in html

    def test_render_gallery_with_entries(self):
        from library.tools.rendering import render_creations_gallery_html

        creations = [
            {"title": "My Poem", "type": "poetry", "slug": "my-poem",
             "updated": "1000", "body": "Roses are red."},
            {"title": "My Art", "type": "art", "slug": "my-art",
             "updated": "2000", "body": '<svg><circle r="10"/></svg>'},
        ]
        html = render_creations_gallery_html(creations, title="Gallery")
        assert "My Poem" in html
        assert "My Art" in html
        assert "2 works" in html
        assert "Roses are red." in html
        # meta in its own div, not inside h2
        assert 'class="meta"' in html

    def test_render_art_includes_svg_link(self):
        from library.tools.rendering import render_creations_gallery_html

        creations = [
            {"title": "My Art", "type": "art", "slug": "my-art",
             "updated": "", "body": '<svg><circle r="10"/></svg>'},
        ]
        html = render_creations_gallery_html(creations)
        assert "<svg>" in html  # inlined
        assert "creations/my-art.svg" in html  # also linked

    def test_render_music_entry(self):
        from library.tools.rendering import render_creations_gallery_html

        creations = [
            {"title": "A Tune", "type": "music", "slug": "a-tune",
             "updated": "", "body": "X:1\nT:Test\nK:C\nCDEF|"},
        ]
        html = render_creations_gallery_html(creations)
        assert "abc-source" in html
        assert "abc-render" in html

    def test_render_game_entry(self):
        from library.tools.rendering import render_creations_gallery_html

        creations = [
            {"title": "My Game", "type": "game", "slug": "my-game",
             "updated": "", "body": "<html>game</html>"},
        ]
        html = render_creations_gallery_html(creations)
        assert "game-link" in html
        assert "creations/my-game.html" in html
