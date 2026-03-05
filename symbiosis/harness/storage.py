"""Namespaced file storage — all paths auto-scoped to an instance."""

from __future__ import annotations

from pathlib import Path


class NamespacedStorage:
    def __init__(self, base_dir: str | Path, namespace: str):
        self._root = Path(base_dir) / namespace / "memory"
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def _resolve(self, path: str) -> Path:
        resolved = (self._root / path).resolve()
        if not str(resolved).startswith(str(self._root.resolve())):
            raise ValueError(f"Path escapes namespace: {path}")
        return resolved

    def read(self, path: str) -> str:
        target = self._resolve(path)
        if not target.exists():
            return ""
        return target.read_text(encoding="utf-8")

    def write(self, path: str, content: str) -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def list(self, prefix: str = "") -> list[str]:
        search_dir = self._resolve(prefix) if prefix else self._root
        if not search_dir.exists():
            return []
        root = self._root.resolve()
        results = []
        for p in sorted(search_dir.rglob("*")):
            if p.is_file():
                results.append(str(p.resolve().relative_to(root)))
        return results

    def exists(self, path: str) -> bool:
        return self._resolve(path).exists()

    def delete(self, path: str) -> bool:
        target = self._resolve(path)
        if target.exists():
            target.unlink()
            return True
        return False
