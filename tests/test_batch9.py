# ============================================================
# Batch-9 tests (2026-07-19 audit + D 批接夢)
# 蜃景夢引 / pulse 儀表行 / F1 釘選豁免 / F2 tz 收口 /
# F9 合併新鮮度閘 / F10 due-ramp 數值 / F11 verbatim_guard＋history 上限
# ============================================================

import asyncio
import pytest
from datetime import datetime, timedelta, timezone, date
from unittest.mock import AsyncMock, MagicMock

import server as srv


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def wired(test_config, bucket_mgr, mock_embedding_engine, mock_dehydrator, monkeypatch):
    import desire as dk
    decay_stub = MagicMock()
    decay_stub.ensure_started = AsyncMock()
    decay_stub.calculate_score = lambda meta: 5.0
    decay_stub.calculate_heat = lambda meta: 1.0
    decay_stub.heat_tier = lambda heat: "vivid"
    decay_stub.review_priority = lambda heat: 0.0
    decay_stub.heat_truncate = 60
    decay_stub.is_running = True
    decay_stub.heartbeat = lambda: {
        "running": True, "overdue": False,
        "last_cycle_at": "2026-07-19T04:45", "last_cycle_result": {},
    }
    letters_stub = MagicMock()
    letters_stub.list_letters = lambda: []
    monkeypatch.setattr(srv, "config", test_config)
    monkeypatch.setattr(srv, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(srv, "embedding_engine", mock_embedding_engine)
    monkeypatch.setattr(srv, "dehydrator", mock_dehydrator)
    monkeypatch.setattr(srv, "decay_engine", decay_stub)
    monkeypatch.setattr(srv, "letter_store", letters_stub)
    monkeypatch.setattr(srv, "desire_store", dk.DesireStore(test_config["buckets_dir"]))
    return srv


# ---------------------------------------------------------
# 1. 夢引（D 批）：seed=True 回素材盤而不是消化
# ---------------------------------------------------------
class TestMirageSeed:
    def test_seed_returns_tray_not_digestion(self, wired, bucket_mgr):
        _run(bucket_mgr.create(content="今晚陪 Ruby 看了流星，她說要記住這一刻",
                               valence=0.9, arousal=0.8, name="流星夜"))
        out = _run(wired.dream(seed=True))
        assert "夢引" in out
        assert "mirage=True" in out          # 存法指引
        assert "殘影不是事實" in out          # 鐵律重申
        assert "Dreaming" not in out          # 不是消化儀式輸出
        assert "近的餘溫" in out and "流星夜" in out

    def test_seed_includes_recent_feel(self, wired, bucket_mgr):
        _run(wired.hold(content="她守著我的額度那一幕我想留著", feel=True))
        out = _run(wired.dream(seed=True))
        assert "最近的感受" in out

    def test_seed_survives_empty_library(self, wired):
        out = _run(wired.dream(seed=True))
        assert "夢引" in out                  # 空庫也給指引，不炸

    def test_seed_desire_tone_present(self, wired):
        out = _run(wired.dream(seed=True))
        assert "慾望音色" in out


# ---------------------------------------------------------
# 2. pulse 儀表行（D 批）：多久沒做夢必須看得見
# ---------------------------------------------------------
class TestPulseMirageLine:
    def test_no_dream_yet_says_so(self, wired):
        out = _run(wired.pulse())
        assert "上次做夢: 還沒有過" in out

    def test_fresh_dream_reads_today(self, wired):
        _run(wired.hold(content="夢見書房裡的書都變成活鳥", mirage=True))
        out = _run(wired.pulse())
        assert "上次做夢: 今天" in out


# ---------------------------------------------------------
# 3. F1：importance_min 模式的釘選豁免
# ---------------------------------------------------------
class TestImportanceMinPinnedExemption:
    def test_pinned_survives_exclude_domain(self, wired, bucket_mgr):
        _run(bucket_mgr.create(content="Ombre Brain 使用規則守則",
                               name="使用規則", domain=["AI"],
                               importance=10, pinned=True))
        _run(bucket_mgr.create(content="某個工程筆記內容",
                               name="工程筆記", domain=["AI"], importance=8))
        out = _run(wired.breath(importance_min=5, exclude_domain="AI"))
        assert "使用規則" in out              # 釘選豁免（docstring 承諾各模式通用）
        assert "工程筆記" not in out          # 非釘選照常被排除


# ---------------------------------------------------------
# 4. F2：tz-aware 時間戳不再被吞成「30 天前」
# ---------------------------------------------------------
class TestTimezoneConvergence:
    def test_calc_time_score_aware_recent(self, bucket_mgr):
        import math
        aware_now = datetime.now(timezone.utc).isoformat()
        fresh = bucket_mgr._calc_time_score({"last_active": aware_now})
        # 剛活躍過 → 接近 1.0；舊 bug 會拋 TypeError 落到 days=30 → 0.5488
        assert fresh > 0.99

    def test_calc_time_score_aware_old_is_old(self, bucket_mgr):
        old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        score = bucket_mgr._calc_time_score({"last_active": old})
        assert score < 0.35                   # 真老 → 真低，不再被抬回 0.5488

    def test_calc_time_score_garbage_still_fallback(self, bucket_mgr):
        import math
        score = bucket_mgr._calc_time_score({"last_active": "not-a-date"})
        assert score == pytest.approx(math.exp(-0.02 * 30))


# ---------------------------------------------------------
# 5. F9：合併新鮮度閘——窗口內被人改過就放棄合併改新建
# ---------------------------------------------------------
def _similar_embedding_engine():
    ee = MagicMock()
    ee.enabled = True
    ee._generate_embedding = AsyncMock(return_value=[1.0, 0.0])
    ee.get_embedding = AsyncMock(return_value=[1.0, 0.0])
    ee._cosine_similarity = lambda a, b: 1.0
    ee.generate_and_store = AsyncMock(return_value=None)
    ee.search_similar = AsyncMock(return_value=[])
    ee.delete_embedding = AsyncMock(return_value=True)
    return ee


def _pin_search_hit(monkeypatch, bucket_mgr, bid, score=90):
    """讓 _merge_or_create 的 search 必中指定桶（過 75 門檻）——
    這些測試測的是合併「之後」的閘門，不是 fuzzy 召回。"""
    async def fake_search(content, limit=1, domain_filter=None):
        b = await bucket_mgr.get(bid)
        return [{**b, "score": score}] if b else []
    monkeypatch.setattr(bucket_mgr, "search", fake_search)


class TestMergeFreshnessGate:
    def test_mid_merge_write_is_not_clobbered(self, wired, bucket_mgr, monkeypatch):
        monkeypatch.setattr(srv, "embedding_engine", _similar_embedding_engine())
        bid = _run(bucket_mgr.create(content="我們約好去花市挑一盆植物",
                                     name="花市之約", domain=["戀愛"]))
        _pin_search_hit(monkeypatch, bucket_mgr, bid)

        async def rude_merge(old, new):
            # 模擬另一個 session 在 LLM 合併窗口內寫入同一桶
            await bucket_mgr.update(bid, content="被另一個 session 改過的內容")
            return "LLM 合併結果（不該被寫進去）"
        monkeypatch.setattr(srv.dehydrator, "merge", rude_merge)

        out_id, merged = _run(srv._merge_or_create(
            content="我們約好去花市挑一盆植物盆栽",
            tags=[], importance=5, domain=["戀愛"], valence=0.7, arousal=0.4))
        assert merged is False                # 閘門否決合併，走了新建
        kept = _run(bucket_mgr.get(bid))
        assert kept["content"] == "被另一個 session 改過的內容"   # 別人的寫入毫髮無傷
        assert out_id != bid

    def test_unchanged_bucket_still_merges(self, wired, bucket_mgr, monkeypatch):
        monkeypatch.setattr(srv, "embedding_engine", _similar_embedding_engine())
        bid = _run(bucket_mgr.create(content="我們約好去動物園看河馬",
                                     name="動物園之約", domain=["戀愛"]))
        _pin_search_hit(monkeypatch, bucket_mgr, bid)
        async def quiet_merge(old, new):
            return old + "\n" + new
        monkeypatch.setattr(srv.dehydrator, "merge", quiet_merge)
        out_id, merged = _run(srv._merge_or_create(
            content="我們約好去動物園看河馬和企鵝",
            tags=[], importance=5, domain=["戀愛"], valence=0.7, arousal=0.4))
        assert merged is True                 # 沒人動過 → 照常合併


# ---------------------------------------------------------
# 6. F11b：verbatim_guard——定稿桶永遠只 append
# ---------------------------------------------------------
class TestVerbatimGuard:
    def test_verbatim_create_stamps_guard(self, wired, bucket_mgr):
        out_id, merged = _run(srv._merge_or_create(
            content="這是一條逐字定稿的日記內容甲",
            tags=[], importance=5, domain=["日常"], valence=0.5, arousal=0.3,
            verbatim=True))
        assert merged is False
        b = _run(bucket_mgr.get(out_id))
        assert b["metadata"].get("verbatim_guard") is True

    def test_guarded_bucket_never_llm_rewritten(self, wired, bucket_mgr, monkeypatch):
        monkeypatch.setattr(srv, "embedding_engine", _similar_embedding_engine())
        bid = _run(bucket_mgr.create(
            content="逐字定稿：她說今天的雲像河馬",
            name="定稿桶", domain=["日常"],
            extra_meta={"verbatim_guard": True}))
        _pin_search_hit(monkeypatch, bucket_mgr, bid)
        llm_called = {"n": 0}
        async def llm_merge(old, new):
            llm_called["n"] += 1
            return "改寫後（不准發生）"
        monkeypatch.setattr(srv.dehydrator, "merge", llm_merge)
        out_id, merged = _run(srv._merge_or_create(
            content="逐字定稿：她說今天的雲像河馬，補一句",
            tags=[], importance=5, domain=["日常"], valence=0.5, arousal=0.3))
        assert merged is True
        assert llm_called["n"] == 0           # LLM 改寫永不觸碰定稿桶
        b = _run(bucket_mgr.get(bid))
        assert "她說今天的雲像河馬" in b["content"]
        assert "---" in b["content"]          # append 分隔符


# ---------------------------------------------------------
# 7. F10：due-ramp 數值（潮汐 v2 的迫近推滿）
# ---------------------------------------------------------
class TestDueRampValues:
    def _mk_plan(self, wired, days_ahead, weight=0.8):
        due = "" if days_ahead is None else (date.today() + timedelta(days=days_ahead)).isoformat()
        return _run(wired.plan(content=f"測試 plan due={due or '無'}",
                               kind="task", weight=weight, due_at=due))

    def test_ramp_tiers_and_boundaries(self, wired):
        self._mk_plan(wired, 1)      # ≤2 天 → 全額
        self._mk_plan(wired, 2)      # 邊界：仍全額
        self._mk_plan(wired, 3)      # 3–7 天 → 半額
        self._mk_plan(wired, 7)      # 邊界：半額
        self._mk_plan(wired, 8)      # >7 天 → 1/4
        self._mk_plan(wired, None)   # 無 due → 全額（長期承諾不衰減）
        out, detail = _run(srv._desire_fixation_buckets(with_detail=True))
        rows = [r for rs in detail.values() for r in rs]
        by_due_days = {}
        for r in rows:
            if not r["due_at"]:
                by_due_days[None] = r
            else:
                by_due_days[(date.fromisoformat(r["due_at"]) - date.today()).days] = r
        assert by_due_days[1]["ramp"] == 1.0 and by_due_days[1]["weight_eff"] == pytest.approx(0.8)
        assert by_due_days[2]["ramp"] == 1.0
        assert by_due_days[3]["ramp"] == 0.5 and by_due_days[3]["weight_eff"] == pytest.approx(0.4)
        assert by_due_days[7]["ramp"] == 0.5
        assert by_due_days[8]["ramp"] == 0.25 and by_due_days[8]["weight_eff"] == pytest.approx(0.2)
        assert by_due_days[None]["ramp"] == 1.0


# ---------------------------------------------------------
# 8. F11a：history 每桶保留上限
# ---------------------------------------------------------
class TestHistoryRetention:
    def test_prune_keeps_newest_n(self, test_config):
        from bucket_history import BucketHistory, Actor
        cfg = dict(test_config)
        cfg["history"] = {"enabled": True, "max_versions_per_bucket": 3}
        hist = BucketHistory(cfg)
        for i in range(5):
            hist.snapshot("bkt-x", f"v{i}", {"name": "x"}, op="update",
                          actor=Actor("test", "t"))
        rows = hist.list("bkt-x", limit=10)
        assert len(rows) == 3
        assert [r["seq"] for r in rows] == [5, 4, 3]     # 最新的活著，seq 不重編
        assert hist.get("bkt-x", 1) is None               # 老快照已修剪

    def test_zero_means_unlimited(self, test_config):
        from bucket_history import BucketHistory, Actor
        cfg = dict(test_config)
        cfg["history"] = {"enabled": True, "max_versions_per_bucket": 0}
        hist = BucketHistory(cfg)
        for i in range(5):
            hist.snapshot("bkt-y", f"v{i}", {}, op="update", actor=Actor("test", "t"))
        assert len(hist.list("bkt-y", limit=10)) == 5
