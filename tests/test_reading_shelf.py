import json
import os

import pytest

from reading_shelf import ReadingShelfStore, normalize_book


def test_normalize_book_requires_title():
    with pytest.raises(ValueError, match="title is required"):
        normalize_book({"title": "   "})


def test_normalize_book_cleans_nested_fields():
    book = normalize_book({
        "title": "長日將盡",
        "status": "已讀完",
        "cover_color": "not-a-color",
        "tags": ["石黑一雄", "石黑一雄", ""],
        "source_bucket_ids": ["332c39744623", "../bad"],
        "excerpts": [
            {"quote": "節錄", "page": "p. 42", "added_by": "Ruby"},
            {"quote": "", "added_by": "unknown"},
        ],
    })

    assert book["cover_color"] == "#2F4F4F"
    assert book["tags"] == ["石黑一雄"]
    assert book["source_bucket_ids"] == ["332c39744623"]
    assert book["excerpts"] == [{
        "quote": "節錄",
        "page": "p. 42",
        "note": "",
        "added_by": "Ruby",
    }]


def test_store_crud_and_atomic_file(tmp_path):
    store = ReadingShelfStore(str(tmp_path))
    created = store.create_book({
        "title": "長日將盡",
        "author": "石黑一雄",
        "summary": "共同摘要",
    })

    assert store.list_books()[0]["id"] == created["id"]
    assert os.path.exists(store.path)
    assert not list(tmp_path.glob("*.tmp"))

    updated = store.update_book(created["id"], {
        **created,
        "status": "已讀完",
        "ruby_notes": "從無聊讀到哭。",
    })
    assert updated["status"] == "已讀完"
    assert updated["ruby_notes"] == "從無聊讀到哭。"
    assert updated["created_at"] == created["created_at"]

    with open(store.path, encoding="utf-8") as handle:
        persisted = json.load(handle)
    assert persisted["version"] == 1
    assert persisted["books"][0]["status"] == "已讀完"

    assert store.delete_book(created["id"]) is True
    assert store.delete_book(created["id"]) is False
    assert store.list_books() == []


def test_store_rejects_corrupt_json(tmp_path):
    store = ReadingShelfStore(str(tmp_path))
    with open(store.path, "w", encoding="utf-8") as handle:
        handle.write("{broken")

    with pytest.raises(RuntimeError, match="failed to read reading shelf"):
        store.list_books()
