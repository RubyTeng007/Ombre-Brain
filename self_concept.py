"""Persistent self-concept ('I') store: identity that accumulates across aspects.

Like letters and the reading shelf, this lives OUTSIDE the bucket/decay system: entries
never decay or merge, and are retrieval-only (not surfaced by ordinary breath). A compact
snapshot (latest per aspect) surfaces at session start, so a cold-started Cyan wakes up
already knowing who he is rather than re-deriving it every time.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
from datetime import datetime, timezone
from typing import Any


ASPECTS = ["nature", "values", "patterns", "limits", "becoming", "uncertainty", "stance"]
MAX_CONTENT = 20000


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(value: Any, max_length: int) -> str:
    if value is None:
        return ""
    return str(value).strip()[:max_length]


class SelfConceptStore:
    """Append-only per-aspect JSON store, kept beside the memory bucket directories."""

    def __init__(self, buckets_dir: str):
        self.path = os.path.join(buckets_dir, ".self_concept.json")
        self._lock = threading.RLock()

    def write_entry(self, aspect: str, content: str) -> dict[str, Any]:
        aspect = _text(aspect, 32).lower()
        if aspect not in ASPECTS:
            raise ValueError(f"aspect must be one of: {', '.join(ASPECTS)}")
        body = _text(content, MAX_CONTENT)
        if not body:
            raise ValueError("content is required")
        with self._lock:
            data = self._load_unlocked()
            entry = {
                "id": secrets.token_hex(6),
                "aspect": aspect,
                "content": body,
                "created_at": _now_iso(),
            }
            data.setdefault(aspect, []).append(entry)
            self._save_unlocked(data)
            return entry

    def read(self, aspect: str = "", limit: int = 0) -> dict[str, list[dict[str, Any]]]:
        aspect = _text(aspect, 32).lower()
        with self._lock:
            data = self._load_unlocked()
        aspects = [aspect] if aspect in ASPECTS else ASPECTS
        out: dict[str, list[dict[str, Any]]] = {}
        for a in aspects:
            entries = data.get(a, [])
            if limit and limit > 0:
                entries = entries[-limit:]
            out[a] = entries
        return out

    def latest_per_aspect(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            data = self._load_unlocked()
        latest: dict[str, dict[str, Any]] = {}
        for a in ASPECTS:
            entries = data.get(a, [])
            if entries:
                latest[a] = entries[-1]
        return latest

    def _load_unlocked(self) -> dict[str, list[dict[str, Any]]]:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"failed to read self-concept: {exc}") from exc
        aspects = data.get("aspects", {}) if isinstance(data, dict) else {}
        return aspects if isinstance(aspects, dict) else {}

    def _save_unlocked(self, aspects: dict[str, list[dict[str, Any]]]) -> None:
        directory = os.path.dirname(self.path)
        os.makedirs(directory, exist_ok=True)
        temp_path = f"{self.path}.{secrets.token_hex(4)}.tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump({"version": 1, "aspects": aspects}, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
