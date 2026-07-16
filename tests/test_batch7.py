# ============================================================
# Tests for the 2026-07-16 batch-7 memory-metabolism 檔1
# 2026-07-16 批7 記憶代謝檔1 的測試
#
# Covers: per-domain metabolism factors on the heat half-life,
# the pure-engineering surfacing quota (guardrail-with-backfill),
# and breath's exclude_domain parameter across all four modes.
#
# Written as RULES, not pinned values (the 07-16 lesson: three tests
# once nailed a bug in place by asserting the buggy composition).
# 全部斷言「規則」而非「值」——07-16 教訓：三個測試曾把 bug 釘死。
# ============================================================

import asyncio
import copy

import pytest
from unittest.mock import AsyncMock, MagicMock

import server as srv
from decay_engine import DecayEngine
from datetime import datetime, timedelta


def _run(coro):
    return asyncio.run(coro)


def _meta(days_ago=0.0, **kw):
    m = {
        "type": "dynamic",
        "importance": 5,
        "activation_count": 1,
        "arousal": 0.3,
        "last_active": (datetime.now() - timedelta(days=days_ago)).isoformat(),
    }
    m.update(kw)
    return m


METAB = {
    "編程": 0.5, "工作": 0.5, "AI": 0.7,
    "戀愛": 1.5, "人際": 1.3,
    "default": 1.0,
}


@pytest.fixture
def metab_config(test_config):
    cfg = copy.deepcopy(test_config)
    cfg["decay"]["metabolism_factor"] = dict(METAB)
    cfg["decay"]["metabolism_exempt_importance"] = 8
    return cfg


@pytest.fixture
def metab_eng(metab_config, bucket_mgr):
    return DecayEngine(metab_config, bucket_mgr)


@pytest.fixture
def plain_eng(test_config, bucket_mgr):
    return DecayEngine(test_config, bucket_mgr)


# =============================================================
class TestMetabolismFactor:
    """§3.3: H × factor(domains) — 只乘半衰期，score 一個字都不動。"""

    def test_no_config_means_feature_off(self, plain_eng):
        # 沒有 metabolism_factor 配置 → 一律 1.0 → 行為與批7之前逐位相同
        for m in (_meta(domain=["編程"]), _meta(domain=["戀愛"]), _meta(), _meta(domain=[])):
            assert plain_eng.metabolism_factor(m) == 1.0

    def test_eng_domain_gets_its_factor(self, metab_eng):
        assert metab_eng.metabolism_factor(_meta(domain=["編程"])) == 0.5
        assert metab_eng.metabolism_factor(_meta(domain=["AI"])) == 0.7

    def test_multi_domain_takes_the_max_lean_preservative(self, metab_eng):
        # 「凌晨三點陪 debug」＝編程×戀愛 → 跟最慢的代謝走
        assert metab_eng.metabolism_factor(_meta(domain=["編程", "戀愛"])) == 1.5
        # 純工程多域取工程內最大
        assert metab_eng.metabolism_factor(_meta(domain=["編程", "AI"])) == 0.7

    def test_unknown_and_empty_domains_fail_open(self, metab_eng):
        assert metab_eng.metabolism_factor(_meta(domain=["未分類"])) == 1.0
        assert metab_eng.metabolism_factor(_meta(domain=["從沒見過的域"])) == 1.0
        assert metab_eng.metabolism_factor(_meta(domain=[])) == 1.0
        assert metab_eng.metabolism_factor(_meta()) == 1.0

    def test_high_importance_never_accelerates(self, metab_eng):
        # 教訓/事故（imp>=8）不加速淡忘：係數下限 1.0
        assert metab_eng.metabolism_factor(_meta(domain=["編程"], importance=8)) == 1.0
        assert metab_eng.metabolism_factor(_meta(domain=["編程"], importance=10)) == 1.0

    def test_exemption_is_a_floor_not_a_cap(self, metab_eng):
        # 保護性 >1 係數在高 importance 照舊——豁免只擋加速，不擋保留
        assert metab_eng.metabolism_factor(_meta(domain=["戀愛"], importance=9)) == 1.5

    def test_blank_importance_does_not_crash_or_exempt(self, metab_eng):
        # 手改成 importance:（空值）的桶：走 or 5 路徑 → 5 < 8 → 照常加速
        assert metab_eng.metabolism_factor(_meta(domain=["編程"], importance=None)) == 0.5

    def test_nonpositive_config_factors_are_dropped(self, test_config, bucket_mgr):
        # 0 會把半衰期打到地板、把記憶藏起來 → 這種配置垃圾直接不收
        cfg = copy.deepcopy(test_config)
        cfg["decay"]["metabolism_factor"] = {"編程": 0, "工作": -1, "default": 1.0}
        eng = DecayEngine(cfg, bucket_mgr)
        assert eng.metabolism_factor(_meta(domain=["編程"])) == 1.0
        assert eng.metabolism_factor(_meta(domain=["工作"])) == 1.0

    def test_factor_multiplies_the_half_life(self, metab_eng, plain_eng):
        # H×0.5 ⟺ 同一天數下等於「年齡加倍」：heat_metab(d) == heat_plain(2d)
        d = 7.0
        got = metab_eng.calculate_heat(_meta(days_ago=d, domain=["編程"]))
        want = plain_eng.calculate_heat(_meta(days_ago=2 * d, domain=["編程"]))
        assert got == pytest.approx(want, abs=1e-9)

    def test_heat_ordering_follows_metabolism(self, metab_eng):
        # 同齡同參數：工程 < 未分類 < 戀愛
        d = 10.0
        h_eng = metab_eng.calculate_heat(_meta(days_ago=d, domain=["編程"]))
        h_neutral = metab_eng.calculate_heat(_meta(days_ago=d, domain=["未分類"]))
        h_rel = metab_eng.calculate_heat(_meta(days_ago=d, domain=["戀愛"]))
        assert h_eng < h_neutral < h_rel

    def test_exempt_types_still_render_full_heat(self, metab_eng):
        for btype in ("permanent", "feel", "plan", "mirage"):
            assert metab_eng.calculate_heat(_meta(days_ago=999, type=btype, domain=["編程"])) == 1.0

    def test_score_is_untouched_by_metabolism_config(self, metab_eng, plain_eng):
        # score 管歸檔分流——代謝係數對它一個字都不動
        for m in (
            _meta(days_ago=3, domain=["編程"]),
            _meta(days_ago=30, domain=["編程"], importance=3),
            _meta(days_ago=10, domain=["戀愛"], arousal=0.9),
        ):
            assert metab_eng.calculate_score(m) == plain_eng.calculate_score(m)

    def test_errors_fail_as_if_the_feature_were_not_there(self, metab_eng):
        assert metab_eng.metabolism_factor(None) == 1.0
        assert metab_eng.metabolism_factor({"domain": "不是列表"}) >= 0.0  # no crash


# =============================================================
ENG_SET = frozenset(["編程", "工作", "AI", "數字", "硬件"])


class TestPureEngPredicate:
    """純工程桶 = 非空 domains ⊆ 工程域集合。"""

    def test_subset_semantics(self):
        assert srv._is_pure_eng({"domain": ["編程"]}, ENG_SET)
        assert srv._is_pure_eng({"domain": ["編程", "工作"]}, ENG_SET)

    def test_any_non_eng_domain_exempts(self):
        # 雙域記憶永不被配額壓制
        assert not srv._is_pure_eng({"domain": ["編程", "戀愛"]}, ENG_SET)
        assert not srv._is_pure_eng({"domain": ["AI", "記憶"]}, ENG_SET)

    def test_unclassified_and_empty_are_not_pure(self):
        assert not srv._is_pure_eng({"domain": ["未分類"]}, ENG_SET)
        assert not srv._is_pure_eng({"domain": []}, ENG_SET)
        assert not srv._is_pure_eng({}, ENG_SET)

    def test_empty_eng_set_disables_the_quota(self):
        assert not srv._is_pure_eng({"domain": ["編程"]}, frozenset())


class TestQuotaPick:
    """配額是護欄不是牆：封頂、讓位、回填，名額永不燒掉。"""

    @staticmethod
    def _b(i, eng):
        return {"id": f"{'e' if eng else 'n'}{i}", "metadata": {"domain": ["編程"] if eng else ["戀愛"]}}

    @pytest.mark.parametrize("quota", [0, 1, 2, 5])
    def test_pick_count_invariant(self, quota):
        # 不變量：選取數 = min(limit, 候選數)，與 quota 無關
        ordered = [self._b(i, eng=(i % 2 == 0)) for i in range(10)]
        assert len(srv._pick_with_eng_quota(ordered, 6, quota, ENG_SET)) == 6
        assert len(srv._pick_with_eng_quota(ordered[:3], 6, quota, ENG_SET)) == 3

    def test_cap_respected_when_non_eng_plentiful(self):
        ordered = [self._b(i, eng=True) for i in range(5)] + [self._b(i, eng=False) for i in range(5)]
        picked = srv._pick_with_eng_quota(ordered, 4, 1, ENG_SET)
        eng_picked = [b for b in picked if b["id"].startswith("e")]
        assert len(picked) == 4
        assert len(eng_picked) == 1

    def test_backfill_when_non_eng_insufficient(self):
        # 護欄不是牆：非工程不夠時，被排開的工程桶回填
        ordered = [self._b(0, True), self._b(1, True), self._b(2, True), self._b(0, False)]
        picked = srv._pick_with_eng_quota(ordered, 4, 1, ENG_SET)
        assert len(picked) == 4
        assert [b["id"] for b in picked] == ["e0", "n0", "e1", "e2"]  # 回填排在後

    def test_zero_quota_still_backfills(self):
        ordered = [self._b(i, eng=True) for i in range(3)]
        picked = srv._pick_with_eng_quota(ordered, 3, 0, ENG_SET)
        assert len(picked) == 3

    def test_non_eng_relative_order_preserved(self):
        ordered = [self._b(0, True), self._b(0, False), self._b(1, True), self._b(1, False)]
        picked = srv._pick_with_eng_quota(ordered, 4, 1, ENG_SET)
        n_ids = [b["id"] for b in picked if b["id"].startswith("n")]
        assert n_ids == ["n0", "n1"]


# =============================================================
@pytest.fixture
def wired(test_config, bucket_mgr, mock_embedding_engine, mock_dehydrator, monkeypatch):
    """Same wiring as test_batch2: real manager, stubbed heat (all vivid)."""
    decay_stub = MagicMock()
    decay_stub.ensure_started = AsyncMock()
    decay_stub.calculate_score = lambda meta: 5.0
    decay_stub.calculate_heat = lambda meta: 1.0
    decay_stub.heat_tier = lambda heat: "vivid"
    decay_stub.review_priority = lambda heat: 0.0
    decay_stub.heat_truncate = 60
    monkeypatch.setattr(srv, "config", test_config)
    monkeypatch.setattr(srv, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(srv, "embedding_engine", mock_embedding_engine)
    monkeypatch.setattr(srv, "dehydrator", mock_dehydrator)
    monkeypatch.setattr(srv, "decay_engine", decay_stub)
    return srv


class TestExcludeDomain:
    """exclude_domain：domain 交集判定取反，四模式一致；釘選豁免。"""

    def test_surfacing_excludes_matching_buckets(self, wired, bucket_mgr):
        eng_id = _run(bucket_mgr.create(content="工程流水記錄", name="工程流水", domain=["編程"]))
        rel_id = _run(bucket_mgr.create(content="關係記憶內容", name="關係記憶", domain=["戀愛"]))
        out = _run(wired.breath(exclude_domain="編程,工作,AI,數字,硬件"))
        assert eng_id not in out
        assert rel_id in out

    def test_multi_domain_bucket_is_excluded_on_any_hit(self, wired, bucket_mgr):
        # 語義是交集判定取反：帶任一被排除域就不出列（與 include 對稱）
        dual_id = _run(bucket_mgr.create(content="凌晨三點的雙域記憶", name="雙域", domain=["編程", "戀愛"]))
        out = _run(wired.breath(exclude_domain="編程"))
        assert dual_id not in out

    def test_pinned_core_principles_are_exempt(self, wired, bucket_mgr):
        # 防污染的護欄不能讓一次醒來失去身分錨點（使用規則桶帶 domain=AI）
        pin_id = _run(bucket_mgr.create(content="使用規則", name="使用規則", domain=["AI"], pinned=True))
        dyn_id = _run(bucket_mgr.create(content="普通 AI 筆記", name="AI筆記", domain=["AI"]))
        out = _run(wired.breath(exclude_domain="AI"))
        assert pin_id in out
        assert dyn_id not in out

    def test_query_mode_excludes(self, wired, bucket_mgr):
        eng_id = _run(bucket_mgr.create(content="天空之城獨特關鍵詞 工程版", name="天空工程", domain=["編程"]))
        rel_id = _run(bucket_mgr.create(content="天空之城獨特關鍵詞 戀愛版", name="天空戀愛", domain=["戀愛"]))
        out = _run(wired.breath(query="天空之城獨特關鍵詞", exclude_domain="編程"))
        assert eng_id not in out
        assert rel_id in out

    def test_catalog_mode_excludes(self, wired, bucket_mgr):
        eng_id = _run(bucket_mgr.create(content="工程目錄項", name="工程目錄項", domain=["編程"]))
        rel_id = _run(bucket_mgr.create(content="關係目錄項", name="關係目錄項", domain=["戀愛"]))
        out = _run(wired.breath(catalog=True, exclude_domain="編程"))
        assert eng_id not in out
        assert rel_id in out

    def test_importance_min_mode_excludes(self, wired, bucket_mgr):
        eng_id = _run(bucket_mgr.create(content="重要工程教訓", name="工程教訓", domain=["編程"], importance=9))
        rel_id = _run(bucket_mgr.create(content="重要關係時刻", name="關係時刻", domain=["戀愛"], importance=9))
        out = _run(wired.breath(importance_min=8, exclude_domain="編程"))
        assert eng_id not in out
        assert rel_id in out

    def test_include_and_exclude_compose(self, wired, bucket_mgr):
        eng_id = _run(bucket_mgr.create(content="工程", name="組合工程", domain=["編程"]))
        rel_id = _run(bucket_mgr.create(content="戀愛", name="組合戀愛", domain=["戀愛"]))
        out = _run(wired.breath(domain="編程,戀愛", exclude_domain="編程"))
        assert eng_id not in out
        assert rel_id in out

    def test_no_exclude_changes_nothing(self, wired, bucket_mgr):
        eng_id = _run(bucket_mgr.create(content="工程流水", name="無排除工程", domain=["編程"]))
        out = _run(wired.breath())
        assert eng_id in out


class TestSurfacingQuotaEndToEnd:
    """一般名額配額規則：工程 ≤ ceil(名額×1/3)（默認），非工程不夠時回填。"""

    def test_eng_capped_when_relation_plentiful(self, wired, bucket_mgr):
        eng_ids = [
            _run(bucket_mgr.create(content=f"工程記錄{i}", name=f"工程{i}", domain=["編程"]))
            for i in range(6)
        ]
        rel_ids = [
            _run(bucket_mgr.create(content=f"關係記錄{i}", name=f"關係{i}", domain=["戀愛"]))
            for i in range(6)
        ]
        out = _run(wired.breath(max_results=6))
        surfaced_eng = [i for i in eng_ids if i in out]
        surfaced_rel = [i for i in rel_ids if i in out]
        # 規則：名額用滿（6），工程 ≤ ceil(6×1/3)=2
        assert len(surfaced_eng) + len(surfaced_rel) == 6
        assert len(surfaced_eng) <= 2

    def test_backfill_keeps_the_breath_full(self, wired, bucket_mgr):
        # 只有工程桶時：護欄不是牆，名額照樣填滿
        eng_ids = [
            _run(bucket_mgr.create(content=f"純工程池{i}", name=f"純工程{i}", domain=["編程"]))
            for i in range(6)
        ]
        out = _run(wired.breath(max_results=6))
        assert len([i for i in eng_ids if i in out]) == 6

    def test_dual_domain_buckets_never_capped(self, wired, bucket_mgr):
        # 多域桶不算純工程：全部雙域時配額形同虛設
        dual_ids = [
            _run(bucket_mgr.create(content=f"雙域{i}", name=f"雙域{i}", domain=["編程", "戀愛"]))
            for i in range(6)
        ]
        out = _run(wired.breath(max_results=6))
        assert len([i for i in dual_ids if i in out]) == 6
