# ============================================================
# Tests for batch-4a (2026-07-12 晚) — the web door learns bucket types
# batch-4a 測試：web 門的型別語義修正
#
# Covers:
# 1. /api/bucket/{id}/trace type dispatch — a plan 沉底 from the web now
#    writes the plan lifecycle (`status`), so it actually stops feeding
#    desire fixations; feel/mirage refuse resolve; pin is decay-only.
# 2. /api/desire/state payload — expired vetoes stay home, `saturated`
#    ships so the frontend can explain a falling bottle.
# 3. /api/status full pulse (feel/plan/mirage/letters) and /api/buckets
#    carrying `last_surfaced` for a truthful 「正在浮現」.
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
    """Minimal stand-in for a Starlette request hitting a custom route."""

    def __init__(self, bucket_id=None, body=None):
        self.path_params = {"bucket_id": bucket_id} if bucket_id else {}
        self._body = body if body is not None else {}

    async def json(self):
        return self._body


def _payload(resp):
    return json.loads(resp.body)


@pytest.fixture
def wired(test_config, bucket_mgr, mock_embedding_engine, mock_dehydrator, monkeypatch):
    decay_stub = MagicMock()
    decay_stub.is_running = True
    decay_stub.ensure_started = AsyncMock()
    decay_stub.calculate_score = lambda meta: 5.0
    decay_stub.heartbeat = MagicMock(return_value={
        "running": True, "last_cycle_at": None, "last_cycle_result": None, "overdue": False,
    })
    usage_stub = MagicMock()
    usage_stub.check_all = AsyncMock(return_value={"ok": True, "warnings": []})
    letters_stub = MagicMock()
    letters_stub.list_letters = MagicMock(return_value=[{"id": "a"}, {"id": "b"}])
    monkeypatch.setattr(srv, "config", test_config)
    monkeypatch.setattr(srv, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(srv, "embedding_engine", mock_embedding_engine)
    monkeypatch.setattr(srv, "dehydrator", mock_dehydrator)
    monkeypatch.setattr(srv, "decay_engine", decay_stub)
    monkeypatch.setattr(srv, "api_usage_guard", usage_stub)
    monkeypatch.setattr(srv, "letter_store", letters_stub)
    monkeypatch.setattr(srv, "desire_store", dk.DesireStore(test_config["buckets_dir"]))
    monkeypatch.setattr(srv, "_require_read_access", lambda req: None)
    return srv


def _mk_plan(bucket_mgr, name="測試帳票", drive="duty", status="active"):
    return _run(bucket_mgr.create(
        content=name, name=name, domain=["待辦"], bucket_type="plan",
        extra_meta={"status": status, "kind": "task", "target_drive": drive, "weight": 0.6},
    ))


# ---------------------------------------------------------
# 1. trace 端點：plan 的沉底寫進生命週期
# ---------------------------------------------------------
class TestTracePlanLifecycle:
    def test_web_resolve_on_plan_writes_status(self, wired, bucket_mgr):
        bid = _mk_plan(bucket_mgr)
        resp = _run(srv.api_bucket_trace(FakeRequest(bid, {"resolved": 1})))
        assert resp.status_code == 200
        meta = _run(bucket_mgr.get(bid))["metadata"]
        assert meta["status"] == "resolved"
        assert meta["resolved"] is True

    def test_web_unresolve_on_plan_reactivates(self, wired, bucket_mgr):
        bid = _mk_plan(bucket_mgr, status="resolved")
        resp = _run(srv.api_bucket_trace(FakeRequest(bid, {"resolved": 0})))
        assert resp.status_code == 200
        meta = _run(bucket_mgr.get(bid))["metadata"]
        assert meta["status"] == "active"
        assert meta["resolved"] is False

    def test_explicit_status_abandoned(self, wired, bucket_mgr):
        bid = _mk_plan(bucket_mgr)
        resp = _run(srv.api_bucket_trace(FakeRequest(bid, {"status": "abandoned"})))
        assert resp.status_code == 200
        meta = _run(bucket_mgr.get(bid))["metadata"]
        assert meta["status"] == "abandoned"
        assert meta["resolved"] is True  # lifecycle closed → decay track follows

    def test_bogus_status_rejected(self, wired, bucket_mgr):
        bid = _mk_plan(bucket_mgr)
        resp = _run(srv.api_bucket_trace(FakeRequest(bid, {"status": "done"})))
        assert resp.status_code == 400

    def test_status_is_plan_only(self, wired, bucket_mgr):
        bid = _run(bucket_mgr.create(content="普通記憶", name="普通記憶"))
        resp = _run(srv.api_bucket_trace(FakeRequest(bid, {"status": "resolved"})))
        assert resp.status_code == 400

    def test_resolved_plan_stops_feeding_fixations(self, wired, bucket_mgr):
        """The actual bug: a web 沉底 used to leave status=active → the plan
        kept boosting its drive forever. End to end through the fixation
        collector."""
        bid = _mk_plan(bucket_mgr, name="沉底後不該再餵", drive="duty")
        before = _run(srv._desire_fixation_buckets())
        assert any(b.get("id") == bid for b in before)
        _run(srv.api_bucket_trace(FakeRequest(bid, {"resolved": 1})))
        after = _run(srv._desire_fixation_buckets())
        assert not any(b.get("id") == bid for b in after)


# ---------------------------------------------------------
# 2. trace 端點：feel / mirage / pin 的型別邊界
# ---------------------------------------------------------
class TestTraceTypeBoundaries:
    def test_feel_refuses_resolve(self, wired, bucket_mgr):
        bid = _run(bucket_mgr.create(content="一段感受", name="一段感受", bucket_type="feel"))
        resp = _run(srv.api_bucket_trace(FakeRequest(bid, {"resolved": 1})))
        assert resp.status_code == 400

    def test_mirage_refuses_resolve_and_pin(self, wired, bucket_mgr):
        bid = _run(bucket_mgr.create(content="一場夢", name="一場夢", bucket_type="mirage"))
        assert _run(srv.api_bucket_trace(FakeRequest(bid, {"resolved": 1}))).status_code == 400
        assert _run(srv.api_bucket_trace(FakeRequest(bid, {"pinned": 1}))).status_code == 400

    def test_plan_refuses_pin(self, wired, bucket_mgr):
        bid = _mk_plan(bucket_mgr)
        resp = _run(srv.api_bucket_trace(FakeRequest(bid, {"pinned": 1})))
        assert resp.status_code == 400

    def test_dynamic_keeps_old_behavior(self, wired, bucket_mgr):
        bid = _run(bucket_mgr.create(content="普通記憶", name="普通記憶"))
        r1 = _run(srv.api_bucket_trace(FakeRequest(bid, {"resolved": 1})))
        assert r1.status_code == 200
        r2 = _run(srv.api_bucket_trace(FakeRequest(bid, {"pinned": 1})))
        assert r2.status_code == 200
        meta = _run(bucket_mgr.get(bid))["metadata"]
        assert meta["resolved"] is True
        assert meta["pinned"] is True
        assert meta["importance"] == 10  # pin locks importance, unchanged

    def test_missing_bucket_404(self, wired):
        resp = _run(srv.api_bucket_trace(FakeRequest("nope00000000", {"resolved": 1})))
        assert resp.status_code == 404


# ---------------------------------------------------------
# 2b. 4c hardening：web 門的只讀邊界收到最終形
# ---------------------------------------------------------
class TestTraceReadOnlyBoundaries:
    def test_delete_is_dynamic_only(self, wired, bucket_mgr):
        """Feel/mirage are Cyan's inner objects (deleted by his hand via MCP,
        never a web button); plans retire via abandoned, keeping the ledger."""
        for btype in ("feel", "mirage"):
            bid = _run(bucket_mgr.create(content="內在物", name="內在物", bucket_type=btype))
            resp = _run(srv.api_bucket_trace(FakeRequest(bid, {"delete": True})))
            assert resp.status_code == 400, btype
        pid = _mk_plan(bucket_mgr)
        assert _run(srv.api_bucket_trace(FakeRequest(pid, {"delete": True}))).status_code == 400
        did = _run(bucket_mgr.create(content="普通", name="普通"))
        resp = _run(srv.api_bucket_trace(FakeRequest(did, {"delete": True})))
        assert resp.status_code == 200
        assert _payload(resp)["deleted"] is True

    def test_feel_fully_read_only(self, wired, bucket_mgr):
        bid = _run(bucket_mgr.create(content="一段感受", name="一段感受", bucket_type="feel"))
        for body in ({"resolved": 1}, {"pinned": 1}, {"delete": True}):
            assert _run(srv.api_bucket_trace(FakeRequest(bid, body))).status_code == 400

    def test_portrait_refuses_everything(self, wired, bucket_mgr):
        bid = _run(bucket_mgr.create(content="畫像正文", name="畫像Ruby 的模樣",
                                     domain=["畫像"], pinned=True))
        for body in ({"resolved": 1}, {"pinned": 0}, {"delete": True}, {"progress": 0.5}):
            assert _run(srv.api_bucket_trace(FakeRequest(bid, body))).status_code == 400

    def test_plan_progress_updates(self, wired, bucket_mgr):
        bid = _mk_plan(bucket_mgr)
        resp = _run(srv.api_bucket_trace(FakeRequest(bid, {"progress": 0.7})))
        assert resp.status_code == 200
        meta = _run(bucket_mgr.get(bid))["metadata"]
        assert abs(float(meta["progress"]) - 0.7) < 1e-9

    def test_progress_is_plan_only_and_validated(self, wired, bucket_mgr):
        did = _run(bucket_mgr.create(content="普通", name="普通"))
        assert _run(srv.api_bucket_trace(FakeRequest(did, {"progress": 0.5}))).status_code == 400
        pid = _mk_plan(bucket_mgr)
        assert _run(srv.api_bucket_trace(FakeRequest(pid, {"progress": "abc"}))).status_code == 400
        resp = _run(srv.api_bucket_trace(FakeRequest(pid, {"progress": 1.7})))
        assert resp.status_code == 200  # clamped
        assert float(_run(bucket_mgr.get(pid))["metadata"]["progress"]) == 1.0


# ---------------------------------------------------------
# 3. desire payload：過期 veto 不出門、saturated 端出
# ---------------------------------------------------------
class TestDesirePayload:
    def test_expired_vetoes_filtered_saturated_shipped(self, wired):
        now = datetime.now()
        store = srv.desire_store
        st = store.load(now)
        st["drives"]["libido"] = 0.80  # above SAT_FLOOR so decay keeps the flag
        st["saturated"] = {"libido": True}
        st["veto_until"] = {
            "social": (now - timedelta(hours=2)).isoformat(timespec="seconds"),
            "creation": (now + timedelta(hours=2)).isoformat(timespec="seconds"),
            "duty": "not-a-timestamp",
        }
        store.save(st)
        payload = _run(srv._desire_state_payload())
        assert "creation" in payload["veto_until"]      # still cooling → shown
        assert "social" not in payload["veto_until"]    # expired → stays home
        assert "duty" not in payload["veto_until"]      # unparseable → dropped
        assert payload["saturated"].get("libido") is True


# ---------------------------------------------------------
# 4. 脈搏補全與 last_surfaced
# ---------------------------------------------------------
class TestFullPulseAndSurfacing:
    def test_status_counts_all_organs(self, wired, bucket_mgr):
        _run(bucket_mgr.create(content="普通", name="普通"))
        _run(bucket_mgr.create(content="感受", name="感受", bucket_type="feel"))
        _mk_plan(bucket_mgr)
        _run(bucket_mgr.create(content="夢", name="夢", bucket_type="mirage"))
        resp = _run(srv.api_system_status(FakeRequest()))
        assert resp.status_code == 200
        buckets = _payload(resp)["buckets"]
        assert buckets["dynamic"] == 1
        assert buckets["feel"] == 1
        assert buckets["plan"] == 1
        assert buckets["mirage"] == 1
        assert buckets["letters"] == 2  # from the stubbed letter store
        assert buckets["total"] == 1    # meaning unchanged: active memory only

    def test_buckets_carry_last_surfaced(self, wired, bucket_mgr):
        bid = _run(bucket_mgr.create(content="會浮現的", name="會浮現的"))
        resp1 = _run(srv.api_buckets(FakeRequest()))
        row1 = next(r for r in _payload(resp1) if r["id"] == bid)
        assert row1["last_surfaced"] == ""  # never surfaced yet
        _run(bucket_mgr.mark_surfaced(bid))
        resp2 = _run(srv.api_buckets(FakeRequest()))
        row2 = next(r for r in _payload(resp2) if r["id"] == bid)
        assert row2["last_surfaced"] != ""
