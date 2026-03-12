"""Tests for library.publish — render_and_publish."""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Minimal ctx stub (no data repo — publish_file falls back to ctx.write)
# ---------------------------------------------------------------------------

class _FakeCtx:
    def __init__(self, files: dict[str, str] | None = None):
        self.instance_id = "test-instance"
        self._files: dict[str, str | bytes] = dict(files or {})
        self._sync_config = None  # no data repo → fallback to ctx.write

    def read(self, path: str) -> str | None:
        v = self._files.get(path)
        return v if isinstance(v, str) else None

    def write(self, path: str, content: str) -> None:
        self._files[path] = content

    def write_binary(self, path: str, content: bytes) -> None:
        self._files[path] = content

    def list(self, prefix: str) -> list[str]:
        return [k for k in self._files if k.startswith(prefix) and self._files[k]]


# ---------------------------------------------------------------------------
# Helper to build creation file content
# ---------------------------------------------------------------------------

def _md_creation(title: str, c_type: str, body: str) -> str:
    return f"---\ntitle: {title}\ntype: {c_type}\ncreated: 1000\nupdated: 2000\n---\n{body}"


def _comment_creation(title: str, c_type: str, body: str) -> str:
    return f"<!-- title: {title} | type: {c_type} | created: 1000 | updated: 2000 -->\n{body}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRenderAndPublishSkipsEmpty:
    def test_no_artifacts_returns_early(self):
        from library.publish import render_and_publish

        ctx = _FakeCtx()
        render_and_publish(ctx)
        # only the original empty files dict
        assert ctx._files == {}


class TestRenderAndPublishCreations:
    def test_game_html_published_individually(self):
        from library.publish import render_and_publish

        game_html = "<html><body>game</body></html>"
        ctx = _FakeCtx({
            "creations/my-game.html": _comment_creation("My Game", "game", game_html),
        })
        render_and_publish(ctx)

        assert "_published/creations/my-game.html" in ctx._files
        assert ctx._files["_published/creations/my-game.html"] == game_html

    def test_art_svg_published_individually(self):
        from library.publish import render_and_publish

        svg = '<svg><circle cx="50" cy="50" r="40" fill="red"/></svg>'
        ctx = _FakeCtx({
            "creations/my-art.svg": _comment_creation("My Art", "art", svg),
        })
        render_and_publish(ctx)

        assert "_published/creations/my-art.svg" in ctx._files
        assert ctx._files["_published/creations/my-art.svg"] == svg

    def test_gallery_html_published(self):
        from library.publish import render_and_publish

        ctx = _FakeCtx({
            "creations/poem.md": _md_creation("My Poem", "poetry", "Roses are red."),
        })
        render_and_publish(ctx)

        assert "_published/creations.html" in ctx._files
        gallery = ctx._files["_published/creations.html"]
        assert "My Poem" in gallery

    def test_music_not_published_individually(self):
        """Music is fully rendered inline via abcjs — no separate file needed."""
        from library.publish import render_and_publish

        ctx = _FakeCtx({
            "creations/tune.abc": _comment_creation("A Tune", "music", "X:1\nT:Test\nK:C\nCDEF|"),
        })
        render_and_publish(ctx)

        # gallery published but no individual music file
        assert "_published/creations.html" in ctx._files
        assert "_published/creations/tune.abc" not in ctx._files

    def test_gallery_links_game_file(self):
        from library.publish import render_and_publish

        ctx = _FakeCtx({
            "creations/adventure.html": _comment_creation("Adventure", "game", "<html>fun</html>"),
        })
        render_and_publish(ctx)

        gallery = ctx._files.get("_published/creations.html", "")
        assert "creations/adventure.html" in gallery

    def test_gallery_links_svg_file(self):
        from library.publish import render_and_publish

        svg = '<svg><rect width="10" height="10"/></svg>'
        ctx = _FakeCtx({
            "creations/abstract.svg": _comment_creation("Abstract", "art", svg),
        })
        render_and_publish(ctx)

        gallery = ctx._files.get("_published/creations.html", "")
        assert "creations/abstract.svg" in gallery

    def test_mixed_types_all_handled(self):
        from library.publish import render_and_publish

        ctx = _FakeCtx({
            "creations/poem.md": _md_creation("Poem", "poetry", "Line one."),
            "creations/art.svg": _comment_creation("Art", "art", '<svg><circle r="5"/></svg>'),
            "creations/tune.abc": _comment_creation("Tune", "music", "X:1\nT:T\nK:C\nCDEF|"),
            "creations/game.html": _comment_creation("Game", "game", "<html>game</html>"),
        })
        render_and_publish(ctx)

        assert "_published/creations.html" in ctx._files
        assert "_published/creations/art.svg" in ctx._files
        assert "_published/creations/game.html" in ctx._files
        # poetry and music have no individual published file
        assert "_published/creations/poem.md" not in ctx._files
        assert "_published/creations/tune.abc" not in ctx._files
