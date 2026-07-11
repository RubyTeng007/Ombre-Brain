# ============================================================
# Tests for the 2026-07-12 batch-0 fixes (Non-spec review batch)
# 2026-07-12 第零批止血修復的測試（Non 規格評審批次）
#
# Covers: fail-closed merge semantic gate, search recall without the
# vector pre-filter, dream plan-tail on an empty digestion window,
# and the plan kind/target_drive/progress/due_at schema.
# ============================================================

import asyncio
import re
import pytest
from unittest.mock import AsyncMock, MagicMock

import server as srv


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def wired(test_config, bucket_mgr, mock_embedding_engine, mock_dehydrator, monkeypatch):
    """Wire server module globals to the isolated test environment."""
    decay_stub = MagicMock()
    decay_stub.ensure_started = AsyncMock()
    decay_stub.calculate_score = lambda meta: 1.0

    monkeypatch.setattr(srv, "config", test_config)
    monkeypatch.setattr(srv, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(srv, "embedding_engine", mock_embedding_engine)
    monkeypatch.setattr(srv, "dehydrator", mock_dehydrator)
    monkeypatch.setattr(srv, "decay_engine", decay_stub)
    return srv


def _fake_search_hit(bucket_id, content, score=99.0, pinned=False):
    return [{
        "id": bucket_id,
        "content": content,
        "score": score,
        "metadata": {"pinned": pinned, "protected": False, "valence": 0.5,
                     "arousal": 0.3, "tags": [], "domain": ["日常"]},
    }]


# ---------------------------------------------------------
# 1. Merge semantic gate is FAIL-CLOSED
#    合併語義閘 fail-closed：驗證不了相似度就不合併
# ---------------------------------------------------------
class TestMergeGateFailClosed:
    def test_gate_exception_creates_new_bucket(self, wired, bucket_mgr, monkeypatch):
        old_id = _run(bucket_mgr.create(content="舊桶內容", name="舊桶"))
        monkeypatch.setattr(
            bucket_mgr, "search",
            AsyncMock(return_value=_fake_search_hit(old_id, "舊桶內容")),
        )
        wired.embedding_engine.enabled = True
        wired.embedding_engine._generate_embedding = AsyncMock(side_effect=RuntimeError("boom"))

        result_id, merged = _run(wired._merge_or_create(
            content="新內容", tags=[], importance=5, domain=["日常"],
            valence=0.5, arousal=0.3,
        ))
        assert merged is False
        assert result_id != old_id

    def test_missing_embedding_creates_new_bucket(self, wired, bucket_mgr, monkeypatch):
        old_id = _run(bucket_mgr.create(content="舊桶內容", name="舊桶"))
        monkeypatch.setattr(
            bucket_mgr, "search",
            AsyncMock(return_value=_fake_search_hit(old_id, "舊桶內容")),
        )
        wired.embedding_engine.enabled = True
        wired.embedding_engine._generate_embedding = AsyncMock(return_value=[0.1, 0.2])
        wired.embedding_engine.get_embedding = AsyncMock(return_value=None)

        result_id, merged = _run(wired._merge_or_create(
            content="新內容", tags=[], importance=5, domain=["日常"],
            valence=0.5, arousal=0.3,
        ))
        assert merged is False
        assert result_id != old_id

    def test_engine_disabled_creates_new_bucket(self, wired, bucket_mgr, monkeypatch):
        old_id = _run(bucket_mgr.create(content="舊桶內容", name="舊桶"))
        monkeypatch.setattr(
            bucket_mgr, "search",
            AsyncMock(return_value=_fake_search_hit(old_id, "舊桶內容")),
        )
        wired.embedding_engine.enabled = False

        result_id, merged = _run(wired._merge_or_create(
            content="新內容", tags=[], importance=5, domain=["日常"],
            valence=0.5, arousal=0.3,
        ))
        assert merged is False
        assert result_id != old_id

    def test_verified_similarity_still_merges(self, wired, bucket_mgr, monkeypatch):
        old_id = _run(bucket_mgr.create(content="舊桶內容", name="舊桶"))
        monkeypatch.setattr(
            bucket_mgr, "search",
            AsyncMock(return_value=_fake_search_hit(old_id, "舊桶內容")),
        )
        wired.embedding_engine.enabled = True
        vec = [0.5, 0.5, 0.5]
        wired.embedding_engine._generate_embedding = AsyncMock(return_value=vec)
        wired.embedding_engine.get_embedding = AsyncMock(return_value=vec)
        wired.embedding_engine._cosine_similarity = lambda a, b: 1.0

        result_id, merged = _run(wired._merge_or_create(
            content="新內容", tags=[], importance=5, domain=["日常"],
            valence=0.5, arousal=0.3,
        ))
        assert merged is True

    def test_pinned_target_never_merges(self, wired, bucket_mgr, monkeypatch):
        old_id = _run(bucket_mgr.create(content="釘選桶", name="釘選桶"))
        monkeypatch.setattr(
            bucket_mgr, "search",
            AsyncMock(return_value=_fake_search_hit(old_id, "釘選桶", pinned=True)),
        )
        wired.embedding_engine.enabled = True
        vec = [0.5, 0.5]
        wired.embedding_engine._generate_embedding = AsyncMock(return_value=vec)
        wired.embedding_engine.get_embedding = AsyncMock(return_value=vec)
        wired.embedding_engine._cosine_similarity = lambda a, b: 1.0

        result_id, merged = _run(wired._merge_or_create(
            content="新內容", tags=[], importance=5, domain=["日常"],
            valence=0.5, arousal=0.3,
        ))
        assert merged is False


# ---------------------------------------------------------
# 2. Search recall: the vector pre-filter no longer hides exact hits
#    搜尋召回：向量預篩不再吃掉精確命中
# ---------------------------------------------------------
class TestSearchUnionRecall:
    def test_exact_hit_survives_unrelated_vector_top(self, test_config):
        from bucket_manager import BucketManager

        fake_engine = MagicMock()
        fake_engine.enabled = True
        mgr = BucketManager(test_config, embedding_engine=fake_engine)

        target_id = _run(mgr.create(content="海底電纜維修的完整紀錄", name="海底電纜維修紀錄"))
        other_id = _run(mgr.create(content="完全無關的日常雜記", name="日常雜記"))
        # Vector channel only knows about the unrelated bucket — before the
        # 2026-07-12 fix this REPLACED the candidate set and dropped the
        # exact-name hit before precision ranking.
        fake_engine.search_similar = AsyncMock(return_value=[(other_id, 0.9)])

        results = _run(mgr.search("海底電纜", limit=5))
        found_ids = [r["id"] for r in results]
        assert target_id in found_ids


# ---------------------------------------------------------
# 3. Dream shows the plan ledger even with nothing to digest
#    空消化窗口也要看見 plan 帳本
# ---------------------------------------------------------
class TestDreamPlanTailOnEmptyWindow:
    def test_plan_tail_survives_empty_candidates(self, wired, bucket_mgr):
        out = _run(wired.plan(content="測試中的承諾", weight=0.7, kind="promise"))
        assert "🪢plan→" in out

        result = _run(wired.dream(max_results=3))
        assert "沒有需要消化的新記憶" in result
        assert "記掛著的事" in result
        assert "測試中的承諾" in result


# ---------------------------------------------------------
# 4. Plan schema: kind / target_drive / progress / due_at
# ---------------------------------------------------------
class TestPlanSchema:
    def _create(self, wired, **kwargs):
        out = _run(wired.plan(content=kwargs.pop("content", "做一件事"), **kwargs))
        m = re.search(r"plan→(\w+)", out)
        return out, (m.group(1) if m else None)

    def test_default_kind_is_task_duty(self, wired, bucket_mgr):
        out, bid = self._create(wired)
        assert bid, out
        meta = _run(bucket_mgr.get(bid))["metadata"]
        assert meta["kind"] == "task"
        assert meta["target_drive"] == "duty"
        assert meta["domain"] == ["待辦"]
        assert meta["progress"] == 0.0

    def test_promise_maps_to_miss_ruby_and_promise_domain(self, wired, bucket_mgr):
        out, bid = self._create(wired, content="陪她去花市", kind="promise")
        meta = _run(bucket_mgr.get(bid))["metadata"]
        assert meta["kind"] == "promise"
        assert meta["target_drive"] == "miss_ruby"
        assert meta["domain"] == ["約定"]

    def test_explicit_target_drive_wins(self, wired, bucket_mgr):
        out, bid = self._create(wired, kind="question", target_drive="reflection")
        meta = _run(bucket_mgr.get(bid))["metadata"]
        assert meta["target_drive"] == "reflection"

    def test_invalid_kind_rejected(self, wired):
        out = _run(wired.plan(content="x", kind="wish"))
        assert "kind 只接受" in out

    def test_invalid_target_drive_rejected(self, wired):
        out = _run(wired.plan(content="x", target_drive="possess"))
        assert "target_drive 只接受" in out

    def test_invalid_due_at_rejected(self, wired):
        out = _run(wired.plan(content="x", due_at="下週吧"))
        assert "due_at" in out and "ISO" in out

    def test_trace_updates_plan_fields(self, wired, bucket_mgr):
        out, bid = self._create(wired)
        r = _run(wired.trace(bid, progress=0.4, target_drive="creation",
                             kind="maintenance", due_at="2026-07-20"))
        assert "已修改" in r
        meta = _run(bucket_mgr.get(bid))["metadata"]
        assert meta["progress"] == 0.4
        assert meta["target_drive"] == "creation"
        assert meta["kind"] == "maintenance"
        assert meta["due_at"] == "2026-07-20"

    def test_trace_plan_fields_rejected_on_dynamic_bucket(self, wired, bucket_mgr):
        bid = _run(bucket_mgr.create(content="普通記憶", name="普通記憶"))
        r = _run(wired.trace(bid, progress=0.5))
        assert "只能用在 plan 桶" in r
        r2 = _run(wired.trace(bid, kind="task"))
        assert "只能用在 plan 桶" in r2
