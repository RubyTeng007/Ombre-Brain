"""Persistent storage for handoff / correspondence letters.

Letters are deliberately OUTSIDE the bucket/decay system (like the reading shelf):
they are never decayed, never merged, never auto-resolved, and the original text is
preserved verbatim. The latest letter from each author auto-surfaces at session start.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

# Letters live on Ruby's clock: "today" is her today, not the server's UTC day.
# 信活在 Ruby 的時鐘上：「今天」是台北的今天，不是伺服器的 UTC 日。
TAIPEI = ZoneInfo("Asia/Taipei")

VALID_AUTHORS = {"Ruby", "Cyan"}
MAX_CONTENT = 50000


def _now_iso() -> str:
    return datetime.now(TAIPEI).isoformat()


def _today() -> str:
    return datetime.now(TAIPEI).date().isoformat()


def _text(value: Any, max_length: int) -> str:
    if value is None:
        return ""
    return str(value).strip()[:max_length]


def normalize_letter(payload: dict[str, Any], existing: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate and normalize a letter payload for storage."""
    if not isinstance(payload, dict):
        raise ValueError("letter must be an object")

    author = _text(payload.get("author"), 20)
    if author not in VALID_AUTHORS:
        raise ValueError("author must be 'Ruby' or 'Cyan'")

    content = _text(payload.get("content"), MAX_CONTENT)
    if not content:
        raise ValueError("content is required")

    current = existing or {}
    letter_date = _text(payload.get("letter_date", current.get("letter_date", "")), 32) or _today()

    raw_tags = payload.get("tags", [])
    if isinstance(raw_tags, str):
        raw_tags = raw_tags.split(",")
    tags: list[str] = []
    if isinstance(raw_tags, list):
        for tag in raw_tags[:20]:
            clean = _text(tag, 50)
            if clean and clean not in tags:
                tags.append(clean)

    now = _now_iso()
    return {
        "id": current.get("id") or secrets.token_hex(6),
        "author": author,
        "title": _text(payload.get("title", current.get("title", "")), 200),
        "content": content,
        "letter_date": letter_date,
        "tags": tags,
        "created_at": current.get("created_at") or now,
    }


class LetterStore:
    """Append-only JSON store for letters, kept beside the memory bucket directories.

    No delete method by design: letters are permanent. Mirrors ReadingShelfStore's
    atomic-write + threading-lock pattern.
    """

    def __init__(self, buckets_dir: str):
        self.path = os.path.join(buckets_dir, ".letters.json")
        self._lock = threading.RLock()

    def list_letters(self) -> list[dict[str, Any]]:
        with self._lock:
            letters = self._load_unlocked()
            return sorted(
                letters,
                key=lambda letter: (letter.get("letter_date") or "", letter.get("created_at") or ""),
                reverse=True,
            )

    def write_letter(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            letters = self._load_unlocked()
            letter = normalize_letter(payload)
            letters.append(letter)
            self._save_unlocked(letters)
            return letter

    def read_letters(
        self,
        query: str = "",
        author: str = "",
        date_from: str = "",
        date_to: str = "",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        clean_query = _text(query, 500).casefold()
        clean_author = _text(author, 20)
        df = _text(date_from, 32)
        dt = _text(date_to, 32)
        limit = max(1, min(int(limit), 50))

        matches = []
        for letter in self.list_letters():
            if clean_author and letter.get("author") != clean_author:
                continue
            ld = letter.get("letter_date") or ""
            if df and ld < df:
                continue
            if dt and ld > dt:
                continue
            if clean_query and clean_query not in self._search_text(letter):
                continue
            matches.append(letter)
            if len(matches) >= limit:
                break
        return matches

    def latest_per_author(self) -> dict[str, dict[str, Any]]:
        """Most recent letter from each author (for session-start surfacing)."""
        latest: dict[str, dict[str, Any]] = {}
        for letter in self.list_letters():  # already newest-first
            author = letter.get("author")
            if author in VALID_AUTHORS and author not in latest:
                latest[author] = letter
            if len(latest) == len(VALID_AUTHORS):
                break
        return latest

    @staticmethod
    def _search_text(letter: dict[str, Any]) -> str:
        parts = [
            letter.get("title", ""),
            letter.get("content", ""),
            letter.get("author", ""),
            letter.get("letter_date", ""),
            " ".join(letter.get("tags", [])),
        ]
        return "\n".join(str(part) for part in parts).casefold()

    def _load_unlocked(self) -> list[dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"failed to read letters: {exc}") from exc

        letters = data.get("letters", []) if isinstance(data, dict) else []
        return letters if isinstance(letters, list) else []

    def _save_unlocked(self, letters: list[dict[str, Any]]) -> None:
        directory = os.path.dirname(self.path)
        os.makedirs(directory, exist_ok=True)
        temp_path = f"{self.path}.{secrets.token_hex(4)}.tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump({"version": 1, "letters": letters}, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
            try:
                os.chmod(self.path, 0o660)
            except OSError:
                pass
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
