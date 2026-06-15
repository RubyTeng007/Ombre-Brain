"""Persistent storage for the Dashboard shared-reading shelf."""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
from datetime import datetime, timezone
from typing import Any


VALID_STATUSES = {"想讀", "共讀中", "已讀完"}
VALID_OWNERS = {"Ruby", "Cyan", "我們"}
HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
BUCKET_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(value: Any, max_length: int) -> str:
    if value is None:
        return ""
    return str(value).strip()[:max_length]


def normalize_book(payload: dict[str, Any], existing: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate and normalize a book payload for storage."""
    if not isinstance(payload, dict):
        raise ValueError("book must be an object")

    title = _text(payload.get("title"), 200)
    if not title:
        raise ValueError("title is required")

    current = existing or {}
    status = _text(payload.get("status", current.get("status", "想讀")), 20)
    if status not in VALID_STATUSES:
        status = "想讀"

    color = _text(payload.get("cover_color", current.get("cover_color", "#2F4F4F")), 20)
    if not HEX_COLOR.fullmatch(color):
        color = "#2F4F4F"

    raw_tags = payload.get("tags", [])
    if isinstance(raw_tags, str):
        raw_tags = raw_tags.split(",")
    tags = []
    if isinstance(raw_tags, list):
        for tag in raw_tags[:20]:
            clean = _text(tag, 50)
            if clean and clean not in tags:
                tags.append(clean)

    raw_bucket_ids = payload.get("source_bucket_ids", [])
    if isinstance(raw_bucket_ids, str):
        raw_bucket_ids = raw_bucket_ids.split(",")
    source_bucket_ids = []
    if isinstance(raw_bucket_ids, list):
        for bucket_id in raw_bucket_ids[:20]:
            clean = _text(bucket_id, 64)
            if BUCKET_ID.fullmatch(clean) and clean not in source_bucket_ids:
                source_bucket_ids.append(clean)

    excerpts = []
    raw_excerpts = payload.get("excerpts", [])
    if isinstance(raw_excerpts, list):
        for item in raw_excerpts[:100]:
            if not isinstance(item, dict):
                continue
            quote = _text(item.get("quote"), 5000)
            if not quote:
                continue
            owner = _text(item.get("added_by", "我們"), 20)
            if owner not in VALID_OWNERS:
                owner = "我們"
            excerpts.append({
                "quote": quote,
                "page": _text(item.get("page"), 100),
                "note": _text(item.get("note"), 5000),
                "added_by": owner,
            })

    now = _now_iso()
    return {
        "id": current.get("id") or secrets.token_hex(6),
        "title": title,
        "author": _text(payload.get("author"), 200),
        "status": status,
        "started_at": _text(payload.get("started_at"), 32),
        "finished_at": _text(payload.get("finished_at"), 32),
        "cover_color": color,
        "summary": _text(payload.get("summary"), 20000),
        "ruby_notes": _text(payload.get("ruby_notes"), 20000),
        "cyan_notes": _text(payload.get("cyan_notes"), 20000),
        "tags": tags,
        "source_bucket_ids": source_bucket_ids,
        "excerpts": excerpts,
        "created_at": current.get("created_at") or now,
        "updated_at": now,
    }


class ReadingShelfStore:
    """Small JSON store kept beside the user's memory bucket directories."""

    def __init__(self, buckets_dir: str):
        self.path = os.path.join(buckets_dir, ".reading_shelf.json")
        self._lock = threading.RLock()

    def list_books(self) -> list[dict[str, Any]]:
        with self._lock:
            books = self._load_unlocked()
            return sorted(
                books,
                key=lambda book: (
                    book.get("finished_at") or book.get("started_at") or "",
                    book.get("updated_at") or "",
                ),
                reverse=True,
            )

    def get_book(self, book_id: str) -> dict[str, Any] | None:
        with self._lock:
            for book in self._load_unlocked():
                if book.get("id") == book_id:
                    return book
        return None

    def search_books(
        self,
        query: str = "",
        status: str = "",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        clean_query = _text(query, 500).casefold()
        clean_status = _text(status, 20)
        limit = max(1, min(int(limit), 50))

        matches = []
        for book in self.list_books():
            if clean_status and book.get("status") != clean_status:
                continue
            if clean_query and clean_query not in self._search_text(book):
                continue
            matches.append(book)
            if len(matches) >= limit:
                break
        return matches

    def create_book(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            books = self._load_unlocked()
            book = normalize_book(payload)
            books.append(book)
            self._save_unlocked(books)
            return book

    def update_book(self, book_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            books = self._load_unlocked()
            for index, book in enumerate(books):
                if book.get("id") == book_id:
                    merged_payload = {**book, **payload}
                    updated = normalize_book(merged_payload, existing=book)
                    books[index] = updated
                    self._save_unlocked(books)
                    return updated
        raise KeyError(book_id)

    def delete_book(self, book_id: str) -> bool:
        with self._lock:
            books = self._load_unlocked()
            kept = [book for book in books if book.get("id") != book_id]
            if len(kept) == len(books):
                return False
            self._save_unlocked(kept)
            return True

    @staticmethod
    def _search_text(book: dict[str, Any]) -> str:
        parts = [
            book.get("title", ""),
            book.get("author", ""),
            book.get("summary", ""),
            book.get("ruby_notes", ""),
            book.get("cyan_notes", ""),
            " ".join(book.get("tags", [])),
            " ".join(book.get("source_bucket_ids", [])),
        ]
        for excerpt in book.get("excerpts", []):
            if isinstance(excerpt, dict):
                parts.extend([
                    excerpt.get("quote", ""),
                    excerpt.get("page", ""),
                    excerpt.get("note", ""),
                    excerpt.get("added_by", ""),
                ])
        return "\n".join(str(part) for part in parts).casefold()

    def _load_unlocked(self) -> list[dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"failed to read reading shelf: {exc}") from exc

        books = data.get("books", []) if isinstance(data, dict) else []
        return books if isinstance(books, list) else []

    def _save_unlocked(self, books: list[dict[str, Any]]) -> None:
        directory = os.path.dirname(self.path)
        os.makedirs(directory, exist_ok=True)
        temp_path = f"{self.path}.{secrets.token_hex(4)}.tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump({"version": 1, "books": books}, handle, ensure_ascii=False, indent=2)
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
