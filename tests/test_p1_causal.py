# ============================================================
# Tests for P1 (2026-07-12 深夜) — causal receipts & typed refs
# P1 測試：因果收據 API、typed 執念來源、wake reason 落欄位
#
# Covers:
# 1. /api/desire/wake stores wake_id in its own event field (the frontend
#    chains on e.wake_id) and the first-person reason in note; idempotent
#    across old (note-embedded) and new formats.
# 2. /api/desire/ledger groups wake → closure into receipts, parses degree,
#    says "unknown" for what old data can't fill, and never guesses.
# 3. _desire_fixation_buckets(with_detail=True) returns typed refs, keeps
#    expired-but-active plans visible (feeding=False) while still cutting
#    them out of the kernel feed.
# ============================================================

import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

import desire as dk
import server as srv


def _run(coro):
    return asyncio.run(coro)


class FakeRequest:
    def __init__(self, body=None, query=None):
        self.path_params = {}
        self._body = body if body is not None else {}
        self.query_params = query or {}

    async def json(self):
        return self._body


def _payload(resp):
    return json.loads(resp.body)


@pytest.fixture
def wired(test_config, bucket_mgr, mock_embedding_engine, mock_dehydrator, monkeypatch):
    decay_stub = MagicMock()
    decay_stub.is_running = True
    decay_stub.calculate_score = lambda meta: 5.0
    monkeypatch.setattr(srv, "config", test_config)
    monkeypatch.setattr(srv, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(srv, "embedding_engine", mock_embedding_engine)
    monkeypatch.setattr(srv, "dehydrator", mock_dehydrator)
    monkeypatch.setattr(srv, "decay_engine", decay_stub)
    monkeypatch.setattr(srv, "desire_store", dk.DesireStore(test_config["buckets_dir"]))
    monkeypatch.setattr(srv, "_require_read_access", lambda req: None)
    monkeypatch.setattr(srv, "_require_hook_access", lambda req: None)
    return srv


# ---------------------------------------------------------
# 1. wake 事件：wake_id 落欄位、reason 落 note
# ---------------------------------------------------------
class TestWakeEvent:
    def test_wake_id_in_field_and_reason_in_note(self, wired):
        resp = _run(srv.api_desire_wake(FakeRequest({
            "drive": "curiosity", "wake_id": "w123", "action": "explore",
            "score": "0.87", "reason": "夢裡那條線還勾著我",
        })))
        assert resp.status_code == 200
        st = srv.desire_store.load()
        ev = [e for e in st["events"] if e.get("kind") == "wake:curiosity"]
        assert len(ev) == 1
        assert ev[0].get("wake_id") == "w123"
        assert ev[0].get("note") == "夢裡那條線還勾著我"

    def test_idempotent_new_and_old_format(self, wired):
        _run(srv.api_desire_wake(FakeRequest({"drive": "duty", "wake_id": "w200", "reason": "r"})))
        _run(srv.api_desire_wake(FakeRequest({"drive": "duty", "wake_id": "w200", "reason": "r"})))
        st = srv.desire_store.load()
        assert len([e for e in st["events"] if e.get("wake_id") == "w200"]) == 1
        # old format: wake_id embedded in note only → still recognized
        def _old(s):
            dk._log_event(s, datetime.now(), "wake:social", "w300 browse score=0.7")
            return s
        srv.desire_store.mutate(_old)
        _run(srv.api_desire_wake(FakeRequest({"drive": "social", "wake_id": "w300"})))
        st = srv.desire_store.load()
        assert len([e for e in st["events"] if "w300" in (e.get("note", "") + str(e.get("wake_id", "")))]) == 1

    def test_reason_missing_falls_back_to_technical_note(self, wired):
        _run(srv.api_desire_wake(FakeRequest({
            "drive": "creation", "wake_id": "w400", "action": "create", "score": "0.61",
        })))
        st = srv.desire_store.load()
        ev = [e for e in st["events"] if e.get("wake_id") == "w400"][0]
        assert "create" in ev["note"]


# ---------------------------------------------------------
# 2. ledger 收據
# ---------------------------------------------------------
class TestLedgerReceipts:
    def test_tool_requires_explicit_satisfaction_degree(self, wired):
        msg = _run(wired.desire(
            action="satisfy", verb="explore", event="只讀了一點", wake_id="w-degree"
        ))
        assert "需要明確 degree" in msg
        assert not any(e.get("wake_id") == "w-degree" for e in wired.desire_store.load()["events"])

    def test_tool_defer_and_outreach_round_trip(self, wired):
        _run(srv.api_desire_wake(FakeRequest({
            "drive": "creation", "wake_id": "w-tool", "reason": "想做東西",
        })))
        _run(wired.desire(
            action="defer", drive="creation", reason="先陪 Ruby", wake_id="w-tool"
        ))
        _run(wired.desire(
            action="outreach", medium="sticker", event="貼圖送達", wake_id="w-tool"
        ))
        data = _payload(_run(srv.api_desire_ledger(FakeRequest())))
        r = next(x for x in data["receipts"] if x["wake_id"] == "w-tool")
        assert r["outcomes"][0]["choice"] == "defer"
        assert r["contacted_ruby"] is True

    def test_wake_closure_grouped_with_degree(self, wired):
        _run(srv.api_desire_wake(FakeRequest({
            "drive": "curiosity", "wake_id": "w500", "reason": "想去讀那篇文",
        })))
        srv.desire_store.mutate(
            lambda s: dk.satisfy(s, "explore", datetime.now(), note="讀完了", degree=0.5, wake_id="w500"))
        resp = _run(srv.api_desire_ledger(FakeRequest()))
        assert resp.status_code == 200
        data = _payload(resp)
        r = next(x for x in data["receipts"] if x["wake_id"] == "w500")
        assert r["drive"] == "curiosity"
        assert r["reason"] == "想去讀那篇文"
        assert r["delivery"] == "delivered"
        assert r["choice"] == "satisfy"
        assert r["verb"] == "explore"
        assert abs(r["degree"] - 0.5) < 1e-9
        assert r["closed"] is True
        assert r["result_note"] == "讀完了"
        assert r["outcomes"] == [{
            "choice": "satisfy", "verb": "explore", "degree": 0.5,
            "result_note": "讀完了", "ts": r["outcomes"][0]["ts"],
        }]

    def test_multiple_outcomes_and_outreach_are_not_overwritten(self, wired):
        _run(srv.api_desire_wake(FakeRequest({
            "drive": "reflection", "wake_id": "w-multi", "reason": "想沉一沉",
        })))
        now = datetime.now()
        srv.desire_store.mutate(
            lambda s: dk.satisfy(s, "dream_feel", now, note="寫了 feel", degree=0.4, wake_id="w-multi"))
        srv.desire_store.mutate(
            lambda s: dk.engage(s, "browse", now, note="看了一圈", wake_id="w-multi", drive="social"))
        srv.desire_store.mutate(
            lambda s: dk.outreach(s, "text", now, note="短訊已送達", wake_id="w-multi"))
        data = _payload(_run(srv.api_desire_ledger(FakeRequest())))
        r = next(x for x in data["receipts"] if x["wake_id"] == "w-multi")
        assert [(x["choice"], x["verb"]) for x in r["outcomes"]] == [
            ("satisfy", "dream_feel"), ("engage", "browse"),
        ]
        assert r["contacted_ruby"] is True
        assert [x["medium"] for x in r["outreach"]] == ["text"]

    def test_outcome_written_before_wake_post_still_attaches(self, wired):
        now = datetime.now()
        srv.desire_store.mutate(
            lambda s: dk.engage(s, "create", now, note="先做了一點", wake_id="w-race", drive="creation"))
        _run(srv.api_desire_wake(FakeRequest({
            "drive": "creation", "wake_id": "w-race", "reason": "手癢了",
        })))
        data = _payload(_run(srv.api_desire_ledger(FakeRequest())))
        r = next(x for x in data["receipts"] if x["wake_id"] == "w-race")
        assert r["closed"] is True
        assert r["outcomes"][0]["verb"] == "create"

    def test_unclosed_wake_says_unknown(self, wired):
        _run(srv.api_desire_wake(FakeRequest({"drive": "social", "wake_id": "w600"})))
        data = _payload(_run(srv.api_desire_ledger(FakeRequest())))
        r = next(x for x in data["receipts"] if x["wake_id"] == "w600")
        assert r["choice"] == "unknown"
        assert r["closed"] is False

    def test_loose_events_not_invented_into_receipts(self, wired):
        srv.desire_store.mutate(
            lambda s: dk.feed(s, "miss_ruby", -0.06, datetime.now(), event="她來說話"))
        data = _payload(_run(srv.api_desire_ledger(FakeRequest())))
        assert all(not r["wake_id"].startswith("feed") for r in data["receipts"])
        assert any(str(e.get("kind", "")).startswith("feed:") for e in data["loose"])


# ---------------------------------------------------------
# 3. typed refs（boosts_detail）
# ---------------------------------------------------------
class TestBoostsDetail:
    def test_expired_plan_visible_but_not_feeding(self, wired, bucket_mgr):
        yesterday = (datetime.now() - timedelta(days=1)).date().isoformat()
        bid = _run(bucket_mgr.create(
            content="過期夢種子", name="過期夢種子", domain=["待辦"], bucket_type="plan",
            extra_meta={"status": "active", "kind": "question",
                        "target_drive": "curiosity", "weight": 0.4, "due_at": yesterday}))
        feed_list, detail = _run(srv._desire_fixation_buckets(with_detail=True))
        assert not any(b.get("id") == bid for b in feed_list)  # kernel 斷餵
        ref = next(x for x in detail["curiosity"] if x["id"] == bid)
        assert ref["expired"] is True
        assert ref["feeding"] is False
        assert ref["type"] == "plan"

    def test_memory_fixation_typed_and_mapped(self, wired, bucket_mgr):
        bid = _run(bucket_mgr.create(content="鯁著的心事", name="鯁著的心事", domain=["自省"]))
        _run(bucket_mgr.update(bid, affects_desire=True))
        feed_list, detail = _run(srv._desire_fixation_buckets(with_detail=True))
        assert any(b.get("id") == bid for b in feed_list)
        ref = next(x for x in detail["reflection"] if x["id"] == bid)
        assert ref["type"] == "memory"
        assert ref["feeding"] is True

    def test_payload_ships_boosts_detail(self, wired, bucket_mgr):
        payload = _run(srv._desire_state_payload())
        assert "boosts_detail" in payload
