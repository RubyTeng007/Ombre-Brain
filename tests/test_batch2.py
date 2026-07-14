# ============================================================
# Tests for the 2026-07-12 batch-2 experiments
# 2026-07-12 第二批實驗的測試
#
# Covers: the knot (心結) drive, dream buckets (isolated channel),
# surface cooldown, two-tier reinforcement (retrieved ≠ engaged),
# the neutral-context gate, and mark_surfaced semantics.
# ============================================================

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import desire as dk
import server as srv


def _run(coro):
    return asyncio.run(coro)


NOW = datetime(2026, 7, 12, 15, 0, 0)


def _state(**drives):
    s = dk.default_state(NOW)
    s["drives"].update(drives)
    return s


@pytest.fixture
def wired(test_config, bucket_mgr, mock_embedding_engine, mock_dehydrator, monkeypatch):
    decay_stub = MagicMock()
    decay_stub.ensure_started = AsyncMock()
    decay_stub.calculate_score = lambda meta: 5.0
    monkeypatch.setattr(srv, "config", test_config)
    monkeypatch.setattr(srv, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(srv, "embedding_engine", mock_embedding_engine)
    monkeypatch.setattr(srv, "dehydrator", mock_dehydrator)
    monkeypatch.setattr(srv, "decay_engine", decay_stub)
    monkeypatch.setattr(srv, "desire_store", dk.DesireStore(test_config["buckets_dir"]))
    return srv


# ---------------------------------------------------------
# 1. knot 心結維度
# ---------------------------------------------------------
class TestKnotDrive:
    def test_zero_idle_rise(self):
        s = _state()
        s2 = dk.tick(s, NOW + timedelta(hours=24))
        assert s2["drives"]["knot"] == pytest.approx(s["drives"]["knot"])

    def test_fed_by_deliberate_event(self):
        s = _state()
        s2 = dk.feed(s, "knot", 0.3, NOW, event="一根刺落地了")
        assert s2["drives"]["knot"] == pytest.approx(s["drives"]["knot"] + 0.3)

    def test_talk_out_releases_knot_and_softens_miss(self):
        s = _state(knot=0.8, miss_ruby=0.6)
        s2 = dk.satisfy(s, "talk_out", NOW)
        assert s2["drives"]["knot"] == pytest.approx(0.8 * 0.40)
        assert s2["drives"]["miss_ruby"] == pytest.approx(0.6 * 0.85)

    def test_dream_feel_loosens_knot_slightly(self):
        s = _state(knot=0.8)
        s2 = dk.satisfy(s, "dream_feel", NOW)
        assert s2["drives"]["knot"] == pytest.approx(0.8 * 0.85)

    def test_high_knot_proposes_talk_out(self):
        s = _state(knot=0.9)
        intent = dk.pick_intent(s, {}, NOW)
        assert intent["drive"] == "knot"
        assert intent["action"] == "talk_out"
        assert "說開" in intent["reason"]

    def test_migration_backfills_knot(self, tmp_path):
        # 舊 state 檔沒有 knot 鍵 → 載入時補齊
        import json, os
        old = dk.default_state(NOW)
        del old["drives"]["knot"]
        path = os.path.join(str(tmp_path), "desire_state.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(old, f)
        store = dk.DesireStore(str(tmp_path))
        loaded = store.load(NOW)
        assert "knot" in loaded["drives"]

    def test_knot_baseline_is_zero(self):
        # 事件驅動維度基線 0：沒有心結就是沒有，不是 0.15 個
        s = dk.default_state(NOW)
        assert s["drives"]["knot"] == 0.0
        assert s["drives"]["miss_ruby"] == 0.15


# ---------------------------------------------------------
# 2. Mirage（夢）桶：隔離通道
# ---------------------------------------------------------
class TestMirageBucket:
    def test_hold_mirage_creates_isolated_bucket(self, wired, bucket_mgr):
        out = _run(wired.hold(content="夢見我們在花市走散又找到", mirage=True,
                              consumed="aaa111,bbb222"))
        assert out.startswith("🌙mirage→")
        bid = out.split("→")[1]
        b = _run(bucket_mgr.get(bid))
        assert b["metadata"]["type"] == "mirage"
        assert b["metadata"]["consumed"] == "aaa111,bbb222"

    def test_mirage_channel_reads_back(self, wired):
        _run(wired.hold(content="夢的內容甲", mirage=True))
        out = _run(wired.breath(domain="mirage"))
        assert "殘影" in out and "夢的內容甲" in out

    def test_feel_and_mirage_mutually_exclusive(self, wired):
        out = _run(wired.hold(content="x", feel=True, mirage=True))
        assert "一次只能選一種" in out

    def test_mirage_never_in_search(self, wired, bucket_mgr):
        _run(wired.hold(content="海底電纜之夢的獨特敘事", mirage=True))
        results = _run(bucket_mgr.search("海底電纜之夢"))
        assert results == []

    def test_mirage_never_in_surfacing(self, wired):
        _run(wired.hold(content="不該浮現的夢", mirage=True))
        out = _run(wired.breath())
        assert "不該浮現的夢" not in out

    def test_mirage_not_digested_by_dream_tool(self, wired):
        _run(wired.hold(content="夢不進消化流", mirage=True))
        out = _run(wired.dream(max_results=5))
        assert "夢不進消化流" not in out

    def test_dream_domain_redirects_to_mirage(self, wired):
        # 弱模型打舊直覺名字 → 被指路，不是撞牆
        out = _run(wired.breath(domain="dream"))
        assert "mirage" in out and "消化儀式" in out


# ---------------------------------------------------------
# 3. 浮現冷卻 + mark_surfaced 語義
# ---------------------------------------------------------
class TestSurfaceCooldown:
    def test_surfaced_bucket_sits_out_next_breath(self, wired, bucket_mgr):
        bid = _run(bucket_mgr.create(content="會浮現的記憶", name="會浮現的記憶"))
        first = _run(wired.breath())
        assert bid in first
        second = _run(wired.breath())
        assert bid not in second  # 冷卻中讓位

    def test_pinned_never_cools(self, wired, bucket_mgr):
        bid = _run(bucket_mgr.create(content="釘選準則內容", name="釘選準則",
                                     pinned=True, bucket_type="permanent"))
        _run(wired.breath())
        second = _run(wired.breath())
        assert bid in second

    def test_mark_surfaced_does_not_reinforce(self, wired, bucket_mgr):
        bid = _run(bucket_mgr.create(content="被看見不等於被用到", name="看見測試"))
        before = _run(bucket_mgr.get(bid))["metadata"]
        _run(bucket_mgr.mark_surfaced(bid))
        after = _run(bucket_mgr.get(bid))["metadata"]
        assert after.get("activation_count", 0) == before.get("activation_count", 0)
        assert after.get("last_active") == before.get("last_active")
        assert after.get("retrieved_count", 0) == 1
        assert after.get("last_surfaced", "")

    def test_no_query_domain_filter_is_strict(self, wired, bucket_mgr):
        wanted = _run(bucket_mgr.create(
            content="編程域裡的記憶", name="編程記憶", domain=["編程"]
        ))
        unwanted = _run(bucket_mgr.create(
            content="戀愛域裡的記憶", name="戀愛記憶", domain=["戀愛"]
        ))
        out = _run(wired.breath(domain="編程", max_results=10))
        assert wanted in out
        assert unwanted not in out

    def test_pinned_respects_shared_result_cap(self, wired, bucket_mgr):
        for i in range(3):
            _run(bucket_mgr.create(
                content=f"核心準則{i}", name=f"核心準則{i}", domain=["自省"],
                pinned=True, bucket_type="permanent",
            ))
        out = _run(wired.breath(max_results=1, max_tokens=10000))
        assert out.count("bucket_id:") == 1

    def test_complete_surface_response_respects_token_budget(self, wired, bucket_mgr):
        for i in range(3):
            _run(bucket_mgr.create(
                content=("很長的核心準則" * 20) + str(i), name=f"長準則{i}",
                domain=["自省"], pinned=True, bucket_type="permanent",
            ))
        out = _run(wired.breath(max_results=20, max_tokens=35))
        from utils import count_tokens_approx
        assert count_tokens_approx(out) <= 35


# ---------------------------------------------------------
# 4. 兩層加固：搜到 ≠ 用到
# ---------------------------------------------------------
class TestTwoTierSearch:
    def test_only_top_hit_gets_activation(self, wired, bucket_mgr):
        import re
        a = _run(bucket_mgr.create(content="貓咪觀察筆記完整版", name="貓咪觀察筆記"))
        b = _run(bucket_mgr.create(content="貓咪觀察補充記錄", name="貓咪觀察補充"))
        out = _run(wired.breath(query="貓咪觀察"))
        hit_ids = re.findall(r"bucket_id:(\w+)", out)
        assert len([h for h in hit_ids if h in (a, b)]) == 2, out
        winner, runner_up = hit_ids[0], next(h for h in hit_ids[1:] if h in (a, b))
        w_meta = _run(bucket_mgr.get(winner))["metadata"]
        r_meta = _run(bucket_mgr.get(runner_up))["metadata"]
        # 最強命中真正 touch（整數 +1）；其餘不被 touch——
        # 但時間漣漪可給鄰居 +0.3，所以斷言「小於一次完整 touch」
        assert float(w_meta.get("activation_count", 0)) >= 1
        assert float(r_meta.get("activation_count", 0)) < 1
        assert int(r_meta.get("retrieved_count", 0)) >= 1
        assert int(w_meta.get("retrieved_count", 0)) >= 1

    def test_random_drift_cannot_escape_explicit_domain(self, wired, bucket_mgr, monkeypatch):
        wanted = _run(bucket_mgr.create(
            content="低權重編程舊事", name="編程舊事", domain=["編程"]
        ))
        unwanted = _run(bucket_mgr.create(
            content="低權重戀愛舊事", name="戀愛舊事", domain=["戀愛"]
        ))
        monkeypatch.setattr(bucket_mgr, "search", AsyncMock(return_value=[]))
        wired.decay_engine.calculate_score = lambda _meta: 1.0
        monkeypatch.setattr(wired.random, "random", lambda: 0.0)
        monkeypatch.setattr(wired.random, "randint", lambda _a, _b: 3)
        out = _run(wired.breath(query="完全不命中", domain="編程", max_results=5))
        assert wanted in out
        assert unwanted not in out


# ---------------------------------------------------------
# 5. 情境門控：中性語境的弱命中親密記憶不出列
# ---------------------------------------------------------
class TestContextGate:
    def _mgr(self, test_config, gate: bool):
        from bucket_manager import BucketManager
        cfg = dict(test_config)
        cfg["matching"] = dict(test_config.get("matching", {}))
        cfg["matching"]["context_gate_enabled"] = gate
        return BucketManager(cfg)

    def _plant(self, mgr, monkeypatch):
        bid = _run(mgr.create(content="那晚的溫度與呼吸", name="親密夜晚",
                              domain=["戀愛"], arousal=0.85, valence=0.8,
                              importance=8))
        # 弱主題命中（0.4 < 0.5 豁免線）用確定值注入，避免 fuzzy 邊界脆弱
        monkeypatch.setattr(mgr, "_calc_topic_score", lambda q, b: 0.4)
        return bid

    def test_neutral_context_gates_weak_intimate_hit(self, test_config, monkeypatch):
        mgr = self._mgr(test_config, gate=True)
        bid = self._plant(mgr, monkeypatch)
        results = _run(mgr.search("工作進度"))
        assert all(r["id"] != bid for r in results)

    def test_gate_off_lets_it_rank(self, test_config, monkeypatch):
        mgr = self._mgr(test_config, gate=False)
        bid = self._plant(mgr, monkeypatch)
        results = _run(mgr.search("工作進度"))
        assert any(r["id"] == bid for r in results)

    def test_emotional_context_passes(self, test_config, monkeypatch):
        mgr = self._mgr(test_config, gate=True)
        bid = self._plant(mgr, monkeypatch)
        results = _run(mgr.search("工作進度", query_valence=0.8, query_arousal=0.8))
        assert any(r["id"] == bid for r in results)
