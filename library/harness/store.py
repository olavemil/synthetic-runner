"""Structured store — SQLite-backed typed key/value with atomic operations."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path


class NamespacedStore:
    """Key/value store scoped to a namespace within a SQLite database."""

    def __init__(self, db: StoreDB, namespace: str):
        self._db = db
        self._namespace = namespace

    def _key(self, key: str) -> str:
        return f"{self._namespace}:{key}"

    def put(self, key: str, value) -> None:
        self._db.put(self._key(key), value)

    def get(self, key: str):
        return self._db.get(self._key(key))

    def delete(self, key: str) -> None:
        self._db.delete(self._key(key))

    def scan(self, prefix: str = "") -> list[tuple[str, object]]:
        full_prefix = self._key(prefix)
        results = self._db.scan(full_prefix)
        strip = len(self._namespace) + 1
        return [(k[strip:], v) for k, v in results]

    def scan_items(self, prefix: str = "") -> list[tuple[str, object, str | None]]:
        """Like scan() but also returns owner column: [(key, value, owner), ...]."""
        full_prefix = self._key(prefix)
        results = self._db.scan_items(full_prefix)
        strip = len(self._namespace) + 1
        return [(k[strip:], v, o) for k, v, o in results]

    def count(self, prefix: str = "") -> int:
        return self._db.count(self._key(prefix))

    def claim(self, key: str, owner_id: str) -> bool:
        return self._db.claim(self._key(key), owner_id)

    def release(self, key: str, owner_id: str) -> bool:
        return self._db.release(self._key(key), owner_id)


class StoreDB:
    """Low-level SQLite store. Thread-safe via a lock."""

    def __init__(self, db_path: str | Path = ":memory:"):
        self._path = str(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kv (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                owner TEXT DEFAULT NULL
            )
            """
        )
        self._conn.commit()

    def close(self):
        self._conn.close()

    def put(self, key: str, value) -> None:
        serialized = json.dumps(value)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)",
                (key, serialized),
            )
            self._conn.commit()

    def get(self, key: str):
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM kv WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def delete(self, key: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM kv WHERE key = ?", (key,))
            self._conn.commit()

    def scan(self, prefix: str = "") -> list[tuple[str, object]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, value FROM kv WHERE key LIKE ? ORDER BY key",
                (prefix + "%",),
            ).fetchall()
        return [(k, json.loads(v)) for k, v in rows]

    def scan_items(self, prefix: str = "") -> list[tuple[str, object, str | None]]:
        """Return (key, value, owner) triples for all rows matching prefix."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, value, owner FROM kv WHERE key LIKE ? ORDER BY key",
                (prefix + "%",),
            ).fetchall()
        return [(k, json.loads(v), o) for k, v, o in rows]

    def count(self, prefix: str = "") -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM kv WHERE key LIKE ?",
                (prefix + "%",),
            ).fetchone()
        return row[0]

    def claim(self, key: str, owner_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT owner FROM kv WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO kv (key, value, owner) VALUES (?, ?, ?)",
                    (key, json.dumps(None), owner_id),
                )
                self._conn.commit()
                return True
            if row[0] is None or row[0] == owner_id:
                self._conn.execute(
                    "UPDATE kv SET owner = ? WHERE key = ?",
                    (owner_id, key),
                )
                self._conn.commit()
                return True
            return False

    def release(self, key: str, owner_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT owner FROM kv WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return False
            if row[0] != owner_id:
                return False
            self._conn.execute(
                "UPDATE kv SET owner = NULL WHERE key = ?", (key,)
            )
            self._conn.commit()
            return True


def open_store(db_path: str | Path = ":memory:") -> StoreDB:
    """Create or open a SQLite-backed store."""
    return StoreDB(db_path)
