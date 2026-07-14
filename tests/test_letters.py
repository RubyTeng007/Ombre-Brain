from letters import LetterStore


def test_machine_letter_idempotency_key_returns_original(tmp_path):
    store = LetterStore(str(tmp_path))
    first = store.write_letter({
        "author": "Cyan",
        "title": "交接信",
        "content": "第一份內容",
        "tags": ["next-self-letter"],
        "idempotency_key": "swap:session-a:2026-07-14T10:00:00Z",
    })
    retry = store.write_letter({
        "author": "Cyan",
        "title": "重送不應另存",
        "content": "網路逾時後的重送",
        "tags": ["next-self-letter"],
        "idempotency_key": "swap:session-a:2026-07-14T10:00:00Z",
    })

    assert retry == first
    assert retry["content"] == "第一份內容"
    assert len(store.list_letters()) == 1


def test_distinct_idempotency_keys_remain_distinct_letters(tmp_path):
    store = LetterStore(str(tmp_path))
    for key in ("checkpoint:one", "checkpoint:two"):
        store.write_letter({
            "author": "Cyan",
            "content": key,
            "idempotency_key": key,
        })

    assert len(store.list_letters()) == 2
