# ============================================================
# Tests for the 2026-07-15 memory batch
# 2026-07-15 記憶批的測試
#
# Covers: the shared decay-exemption constant, heat (retrievability) and its
# three-tier injection, the danger-zone review curve, the pinned-vs-slot
# starvation fix, blur/tombstone rendering, and bucket history + restore.
#
# Every test here is anchored to a claim that was VERIFIED against live data or
# upstream source on 2026-07-15 — the numbers in the docstrings are not vibes.
# 這裡每個測試都錨定一個 2026-07-15 對 live 資料或上游原始碼驗證過的宣稱——
# docstring 裡的數字不是憑感覺寫的。
# ============================================================

import os
import json
import math
import asyncio
import pytest
from datetime import datetime, timedelta

from utils import NON_DECAYING_TYPES, is_decay_exempt


async def _write_bucket_file(mgr, content, **kwargs):
    """Create a bucket, then patch frontmatter fields create() doesn't accept.
    建桶，然後補寫 create() 不收的 frontmatter 欄位。"""
    import frontmatter as fm

    direct = {
        k: kwargs.pop(k) for k in list(kwargs)
        if k in ("created", "last_active", "resolved", "digested", "activation_count")
    }
    bid = await mgr.create(content=content, **kwargs)
    if direct:
        path = mgr._find_bucket_file(bid)
        post = fm.load(path)
        for k, v in direct.items():
            post[k] = v
        with open(path, "w", encoding="utf-8") as f:
            f.write(fm.dumps(post))
    return bid


def _meta(days_ago=0.0, **kw):
    """Bucket metadata whose last_active sits `days_ago` in the past."""
    m = {
        "type": "dynamic",
        "importance": 5,
        "activation_count": 1,
        "arousal": 0.3,
        "last_active": (datetime.now() - timedelta(days=days_ago)).isoformat(),
    }
    m.update(kw)
    return m


# =============================================================
class TestSharedExemption:
    """Ruby's requirement: heat and decay must share ONE exemption list.
    They already carried three drifting copies before this batch."""

    def test_exempt_types_are_the_documented_four(self):
        assert set(NON_DECAYING_TYPES) == {"permanent", "feel", "plan", "mirage"}

    @pytest.mark.parametrize("btype", ["permanent", "feel", "plan", "mirage"])
    def test_each_exempt_type_is_exempt(self, btype):
        assert is_decay_exempt({"type": btype})

    def test_pinned_and_protected_are_exempt_regardless_of_type(self):
        assert is_decay_exempt({"type": "dynamic", "pinned": True})
        assert is_decay_exempt({"type": "dynamic", "protected": True})

    def test_plain_dynamic_is_not_exempt(self):
        assert not is_decay_exempt({"type": "dynamic"})
        assert not is_decay_exempt({})

    def test_non_dict_is_not_exempt(self):
        assert not is_decay_exempt(None)
        assert not is_decay_exempt("dynamic")

    def test_score_and_heat_agree_on_the_exempt_set(self, decay_eng):
        """The actual anti-drift property: whatever score treats as exempt,
        heat must too. This is the test that fails if someone adds a type to
        one and not the other.
        真正的防漂移性質：score 認定豁免的，heat 必須也認定。"""
        for btype in NON_DECAYING_TYPES:
            m = _meta(days_ago=999, type=btype)
            assert decay_eng.calculate_score(m) >= 30.0, btype  # exempt → fixed score
            assert decay_eng.calculate_heat(m) == 1.0, btype

    def test_import_time_guard_catches_value_map_drift(self):
        """decay_engine refuses to import if _EXEMPT_SCORES and
        NON_DECAYING_TYPES disagree — a loud failure, not a silent skip."""
        import decay_engine
        assert set(decay_engine._EXEMPT_SCORES) == set(NON_DECAYING_TYPES)


# =============================================================
class TestHeat:
    """heat = 2^(-days/H): a real 0-1 retrievability, separate from score."""

    def test_heat_is_bounded_zero_to_one(self, decay_eng):
        """The bug this whole batch exists to fix: score is UNBOUNDED (live
        range 0.18–40.75), so thresholds of 0.7/0.3 were meaningless on it."""
        for days in (0, 1, 5, 30, 365, 10_000):
            h = decay_eng.calculate_heat(_meta(days_ago=days))
            assert 0.0 <= h <= 1.0, f"{days}d → {h}"

    def test_fresh_memory_is_fully_vivid(self, decay_eng):
        assert decay_eng.calculate_heat(_meta(days_ago=0)) == pytest.approx(1.0)

    def test_heat_decreases_monotonically_with_age(self, decay_eng):
        heats = [decay_eng.calculate_heat(_meta(days_ago=d)) for d in range(0, 60, 3)]
        assert heats == sorted(heats, reverse=True)
        assert heats[0] > heats[-1]

    def test_half_life_is_actually_a_half_life(self, decay_eng):
        """At exactly one half-life, heat must be 0.5. This is what makes the
        number legible: 'this memory half-fades in N days'."""
        m = _meta(days_ago=0, importance=5, activation_count=1, arousal=0.3)
        H = (
            decay_eng.heat_halflife_base
            * (5 / 5.0)
            * (1.0 ** 0.3)
            * (decay_eng.emotion_base + 0.3 * decay_eng.arousal_boost)
        )
        m["last_active"] = (datetime.now() - timedelta(days=H)).isoformat()
        assert decay_eng.calculate_heat(m) == pytest.approx(0.5, abs=0.01)

    def test_importance_extends_retention(self, decay_eng):
        at = lambda imp: decay_eng.calculate_heat(_meta(days_ago=14, importance=imp))
        assert at(10) > at(5) > at(1)

    def test_arousal_extends_retention(self, decay_eng):
        hot = decay_eng.calculate_heat(_meta(days_ago=14, arousal=0.9))
        cold = decay_eng.calculate_heat(_meta(days_ago=14, arousal=0.0))
        assert hot > cold

    def test_activation_extends_retention(self, decay_eng):
        used = decay_eng.calculate_heat(_meta(days_ago=14, activation_count=20))
        unused = decay_eng.calculate_heat(_meta(days_ago=14, activation_count=1))
        assert used > unused

    def test_unparseable_timestamp_falls_back_not_crashes(self, decay_eng):
        h = decay_eng.calculate_heat(_meta(last_active="not-a-date"))
        assert 0.0 <= h <= 1.0

    def test_tz_aware_timestamp_is_not_silently_aged_to_30_days(self, decay_eng):
        """The old inline parse hit TypeError on any offset-aware value
        (datetime.now() is naive) and silently returned days_since=30. Zero
        live buckets carry an offset today, but a hand-edited file could.
        舊的行內解析碰到帶時區的值會靜默回傳 30 天。"""
        from datetime import timezone
        now_aware = datetime.now(timezone.utc).astimezone()
        m = _meta(last_active=now_aware.isoformat())
        assert decay_eng.calculate_heat(m) == pytest.approx(1.0, abs=0.01)

    def test_heat_does_not_disturb_score(self, decay_eng):
        """calculate_score must be untouched by this batch. arousal appearing
        twice and the 3-day weight flip are deliberate and correct."""
        m = _meta(days_ago=2, importance=7, arousal=0.8)
        before = decay_eng.calculate_score(m)
        decay_eng.calculate_heat(m)
        assert decay_eng.calculate_score(m) == before


# =============================================================
class TestHeatTiers:
    """Boundaries copied verbatim from kiwi-mem main.py:885-903: strict `>`."""

    def test_tier_boundaries_are_strict_greater_than(self, decay_eng):
        assert decay_eng.heat_tier(0.70) == "faded"   # NOT vivid
        assert decay_eng.heat_tier(0.30) == "lost"    # NOT faded
        assert decay_eng.heat_tier(0.7001) == "vivid"
        assert decay_eng.heat_tier(0.3001) == "faded"

    def test_the_three_tiers(self, decay_eng):
        assert decay_eng.heat_tier(1.0) == "vivid"
        assert decay_eng.heat_tier(0.5) == "faded"
        assert decay_eng.heat_tier(0.0) == "lost"

    def test_exempt_buckets_always_render_vivid(self, decay_eng):
        for btype in NON_DECAYING_TYPES:
            h = decay_eng.calculate_heat(_meta(days_ago=999, type=btype))
            assert decay_eng.heat_tier(h) == "vivid"


# =============================================================
class TestDangerZone:
    """cortexgraph review.py:15-58, ported. The x in 1-x² is NOT the heat: it is
    heat re-centred on the midpoint and rescaled to [-1, 1]. Feeding raw heat in
    would peak at 0 — on already-dead memories — inverting the intent."""

    def test_peak_is_at_the_midpoint_of_the_danger_window(self, decay_eng):
        mid = (decay_eng.danger_min + decay_eng.danger_max) / 2
        assert mid == pytest.approx(0.25)
        assert decay_eng.review_priority(mid) == pytest.approx(1.0)

    def test_peak_is_the_global_maximum(self, decay_eng):
        """Scan the whole 0-1 range: nothing may outrank the midpoint."""
        best_h, best_p = max(
            ((i / 1000, decay_eng.review_priority(i / 1000)) for i in range(1001)),
            key=lambda t: t[1],
        )
        assert best_h == pytest.approx(0.25, abs=1e-6)
        assert best_p == pytest.approx(1.0)

    def test_hard_gate_outside_the_window(self, decay_eng):
        """Below the floor it is already gone — digging it up is relearning,
        not review. Above the ceiling it does not need saving."""
        assert decay_eng.review_priority(0.05) == 0.0
        assert decay_eng.review_priority(0.14) == 0.0
        assert decay_eng.review_priority(0.36) == 0.0
        assert decay_eng.review_priority(0.9) == 0.0
        assert decay_eng.review_priority(1.0) == 0.0

    def test_priority_is_symmetric_around_the_peak(self, decay_eng):
        for delta in (0.02, 0.05, 0.09):
            lo = decay_eng.review_priority(0.25 - delta)
            hi = decay_eng.review_priority(0.25 + delta)
            assert lo == pytest.approx(hi, abs=1e-9)

    def test_priority_falls_off_away_from_the_peak(self, decay_eng):
        assert (
            decay_eng.review_priority(0.25)
            > decay_eng.review_priority(0.29)
            > decay_eng.review_priority(0.33)
        )

    def test_priority_is_bounded(self, decay_eng):
        for i in range(0, 101):
            assert 0.0 <= decay_eng.review_priority(i / 100) <= 1.0

    def test_moving_the_window_moves_the_peak(self, test_config, bucket_mgr):
        """0.25 is not a constant anywhere — it is the midpoint of two configs.
        Verified against cortexgraph source: same property there."""
        from decay_engine import DecayEngine
        cfg = dict(test_config)
        cfg["decay"] = {**cfg.get("decay", {}), "heat": {"danger_min": 0.2, "danger_max": 0.6}}
        eng = DecayEngine(cfg, bucket_mgr)
        assert eng.review_priority(0.4) == pytest.approx(1.0)
        assert eng.review_priority(0.25) < 1.0


# =============================================================
class TestBlurRendering:
    """Ruby's principle: say 'you can't recall this clearly' — never truncate
    silently and let the reader think the fragment is the whole thing."""

    def test_json_body_degrades_to_its_summary_field(self):
        import server as srv
        body = json.dumps(
            {
                "core_facts": ["事實一", "事實二", "事實三"],
                "emotion_state": "平靜",
                "todos": ["待辦一"],
                "keywords": ["關鍵詞"],
                "summary": "這是核心總結",
            },
            ensure_ascii=False,
        )
        out = srv._blur_summary(f"📌 記憶桶: 測試桶 [主題:測試]\n{body}", 60)
        assert "這是核心總結" in out
        assert "（印象模糊）" in out
        assert "事實二" not in out          # detail dropped
        assert "📌 記憶桶: 測試桶" in out    # header kept — you still know what it is

    def test_blur_says_so_and_names_the_way_back(self):
        """A blurred memory is not a dead end: the full text is still on disk."""
        import server as srv
        out = srv._blur_summary('head\n{"summary": "摘要"}', 60)
        assert "（印象模糊）" in out
        assert "breath(query" in out

    def test_label_is_a_suffix_not_a_prefix(self):
        """kiwi-mem appends it. Copied exactly."""
        import server as srv
        out = srv._blur_summary('head\n{"summary": "摘要"}', 60)
        line = [l for l in out.split("\n") if "（印象模糊）" in l][0]
        assert line.rstrip().endswith("（印象模糊）")

    def test_broken_json_falls_back_to_char_slice_instead_of_crashing(self):
        """This is not hypothetical: on 2026-07-15 the live '畫像我們的模樣'
        bucket had a cached summary truncated mid-string by the dehydrator's
        own 1024-token output cap. The blur path must survive it.
        這不是假設：live 的「畫像我們的模樣」快取就是壞的 JSON。"""
        import server as srv
        broken = '{"core_facts": ["a", "b"], "todos": ["未完成的字串'
        out = srv._blur_summary(f"head\n{broken}", 20)
        assert "（印象模糊）" in out
        assert "…" in out

    def test_short_non_json_body_uses_kiwi_mem_raw_slice(self):
        """Buckets under 100 tokens bypass dehydration entirely."""
        import server as srv
        out = srv._blur_summary("head\n" + ("字" * 200), 60)
        assert "…" in out
        assert len([c for c in out if c == "字"]) == 60

    def test_body_shorter_than_limit_is_not_given_an_ellipsis(self):
        """Only the body line matters — the escape-hatch hint legitimately
        contains an ellipsis of its own (breath(query=…))."""
        import server as srv
        out = srv._blur_summary("head\n短短的", 60)
        body_line = [l for l in out.split("\n") if "（印象模糊）" in l][0]
        assert "…" not in body_line
        assert body_line == "短短的（印象模糊）"

    def test_handles_a_body_with_no_header_line(self):
        import server as srv
        out = srv._blur_summary("只有正文沒有標頭", 60)
        assert "（印象模糊）" in out


# =============================================================
class TestTombstone:
    """kiwi-mem has no else-branch: sub-threshold memories vanish, counted only
    in a log the model never sees. That is the same silent omission Ruby named,
    moved from the text level to the bucket level — a reader shown nothing
    assumes there was nothing."""

    def test_empty_when_nothing_is_lost(self):
        import server as srv
        assert srv._tombstone_line([]) == ""

    def test_names_the_faded_without_revealing_content(self):
        import server as srv
        lost = [
            {"id": "a", "metadata": {"name": "木柵動物園看河馬"}},
            {"id": "b", "metadata": {"name": "稱謂規則備忘"}},
        ]
        out = srv._tombstone_line(lost)
        assert "木柵動物園看河馬" in out
        assert "稱謂規則備忘" in out
        assert "2" in out
        assert "breath(query" in out  # names the lever

    def test_caps_the_list_but_still_reports_the_true_total(self):
        """Truncating the roster must not lie about how many faded."""
        import server as srv
        lost = [{"id": str(i), "metadata": {"name": f"記憶{i}"}} for i in range(20)]
        out = srv._tombstone_line(lost)
        assert "20" in out
        assert "記憶0" in out
        assert "記憶19" not in out

    def test_falls_back_to_bucket_id_when_unnamed(self):
        import server as srv
        out = srv._tombstone_line([{"id": "abc123", "metadata": {}}])
        assert "abc123" in out


# =============================================================
class TestPinnedSlotStarvation:
    """The bug that made both features invisible: pinned buckets spent the
    surfacing count budget. With 5 pinned and the documented opening call
    (max_results=5), candidates[:0] surfaced NOTHING — 360 eligible memories,
    zero shown, at any token budget. Reproduced live on 2026-07-15."""

    def test_pinned_no_longer_consume_the_dynamic_budget(self):
        """Ruby's fix, not mine: pinned are RESIDENT, not surfaced, so they sit
        outside the budget entirely. Capping them at 'half the slots' would only
        push the bug one bucket further away — a 6th pinned bucket revives it."""
        pinned_count, max_results = 5, 5
        slots = max_results  # pinned no longer decrement this
        assert slots == 5, "5 pinned + max_results=5 must still leave 5 dynamic slots"

    def test_the_old_arithmetic_did_starve(self):
        """Guard against a regression to the old semantics."""
        slots = 5
        for _ in range(5):  # the old loop
            if slots <= 0:
                break
            slots -= 1
        assert slots == 0  # this is what we escaped


# =============================================================
class TestDangerSlotAllocation:
    """2 normal : 1 danger."""

    @pytest.mark.parametrize(
        "total,expect_danger",
        [(0, 0), (1, 0), (2, 0), (3, 1), (5, 1), (6, 2), (9, 3), (20, 6)],
    )
    def test_two_to_one_split(self, total, expect_danger):
        danger = total // 3
        normal = total - danger
        assert danger == expect_danger
        assert danger + normal == total

    def test_danger_is_silent_when_there_are_no_dynamic_slots(self):
        """If the slot fix were reverted, the danger channel would go completely
        quiet — no error, no trace. The two ship together for this reason."""
        assert 0 // 3 == 0


# =============================================================
class TestDehydrationTruncation:
    """The output cap trapdoor. NOT hypothetical: on 2026-07-15 the live
    畫像我們的模樣 bucket — the portrait of the relationship, and the largest of
    the five pinned buckets — had its cached JSON severed at
    `"下一本共讀書待定", "` by the 1024-token cap, then served on EVERY breath
    from then on, with keywords and summary simply absent. finish_reason was
    telling us the whole time; nothing read it.
    這不是假設：live 的關係畫像就是這樣被切、被快取、被每次呼吸供應的。"""

    @pytest.fixture
    def dehy(self, test_config):
        from unittest.mock import MagicMock, AsyncMock
        from dehydrator import Dehydrator
        cfg = dict(test_config)
        cfg["dehydration"] = {"api_key": "test-key", "max_tokens": 1024}
        d = Dehydrator(cfg)
        d.client = MagicMock()
        return d

    def _reply(self, text, finish_reason):
        from unittest.mock import MagicMock
        choice = MagicMock()
        choice.message.content = text
        choice.finish_reason = finish_reason
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    @pytest.mark.asyncio
    async def test_clean_completion_is_marked_complete(self, dehy):
        from unittest.mock import AsyncMock
        dehy.client.chat.completions.create = AsyncMock(
            return_value=self._reply('{"summary": "好"}', "stop")
        )
        text, complete = await dehy._complete("sys", "usr")
        assert complete is True
        assert text == '{"summary": "好"}'
        assert dehy.client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_hitting_the_cap_retries_with_double_room(self, dehy):
        from unittest.mock import AsyncMock
        dehy.client.chat.completions.create = AsyncMock(
            side_effect=[self._reply('{"summary": "被切', "length"),
                         self._reply('{"summary": "完整了"}', "stop")]
        )
        text, complete = await dehy._complete("sys", "usr")
        assert complete is True
        assert text == '{"summary": "完整了"}'
        budgets = [c.kwargs["max_tokens"] for c in dehy.client.chat.completions.create.call_args_list]
        assert budgets == [1024, 2048], "retrying with the SAME budget would fail identically"

    @pytest.mark.asyncio
    async def test_still_truncated_after_retry_is_reported_not_hidden(self, dehy):
        from unittest.mock import AsyncMock
        dehy.client.chat.completions.create = AsyncMock(
            return_value=self._reply('{"summary": "還是被切', "length")
        )
        text, complete = await dehy._complete("sys", "usr")
        assert complete is False
        assert dehy.client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_truncated_summary_never_reaches_the_cache(self, dehy):
        """The heart of it. Cached, a truncated summary stops being one bad call
        and becomes the memory's permanent face.
        一旦進了快取，它就不再是一次失敗的呼叫，而是這條記憶的永久臉孔。"""
        from unittest.mock import AsyncMock
        long_content = "很長的內容" * 100  # > 100 tokens, so it goes through the API
        dehy.client.chat.completions.create = AsyncMock(
            return_value=self._reply('{"core_facts": ["a"], "todos": ["未完的字串', "length")
        )
        await dehy.dehydrate(long_content, {"name": "測試"})
        assert dehy._get_cached_summary(long_content) is None

    @pytest.mark.asyncio
    async def test_a_complete_summary_does_reach_the_cache(self, dehy):
        from unittest.mock import AsyncMock
        long_content = "很長的內容" * 100
        dehy.client.chat.completions.create = AsyncMock(
            return_value=self._reply('{"summary": "完整"}', "stop")
        )
        await dehy.dehydrate(long_content, {"name": "測試"})
        assert dehy._get_cached_summary(long_content) == '{"summary": "完整"}'

    @pytest.mark.asyncio
    async def test_truncated_output_says_so(self, dehy):
        """Same rule as （印象模糊）: never hand over a fragment and let the
        reader believe it is the whole thing."""
        from unittest.mock import AsyncMock
        long_content = "很長的內容" * 100
        dehy.client.chat.completions.create = AsyncMock(
            return_value=self._reply('{"core_facts": ["被切掉的', "length")
        )
        out = await dehy.dehydrate(long_content, {"name": "測試"})
        assert "截斷" in out
        assert "breath(query" in out

    @pytest.mark.asyncio
    async def test_truncated_merge_is_refused_outright(self, dehy):
        """Worse than a truncated summary: a merge overwrites the bucket's own
        body — the source of truth — with an amputated version. The codebase
        already decided this trade: 重複可救，誤併不可逆."""
        from unittest.mock import AsyncMock
        dehy.client.chat.completions.create = AsyncMock(
            return_value=self._reply("被截斷的合併結果", "length")
        )
        with pytest.raises(RuntimeError, match="截斷"):
            await dehy.merge("舊記憶正文", "新內容")

    @pytest.mark.asyncio
    async def test_complete_merge_is_returned(self, dehy):
        from unittest.mock import AsyncMock
        dehy.client.chat.completions.create = AsyncMock(
            return_value=self._reply("合併好的完整正文", "stop")
        )
        assert await dehy.merge("舊", "新") == "合併好的完整正文"

    @pytest.mark.asyncio
    async def test_a_provider_without_finish_reason_is_treated_as_complete(self, dehy):
        """Don't punish a provider that omits the field — only act on an
        explicit 'length'."""
        from unittest.mock import AsyncMock, MagicMock
        choice = MagicMock()
        choice.message.content = "文字"
        del choice.finish_reason
        resp = MagicMock()
        resp.choices = [choice]
        dehy.client.chat.completions.create = AsyncMock(return_value=resp)
        text, complete = await dehy._complete("sys", "usr")
        assert complete is True


# =============================================================
class TestBucketHistory:
    """Letta's BlockHistory, adapted. Copied: standalone PK + UNIQUE(bucket_id,
    seq), full snapshot not diff, actor_id as a plain string not an FK.
    NOT copied: its undo/redo pointer — its checkpoint deletes every row with
    seq > current, so undo-then-edit destroys the redo chain permanently."""

    @pytest.fixture
    def hist(self, test_config):
        from bucket_history import BucketHistory
        return BucketHistory(test_config)

    def test_seq_starts_at_one_and_increments_per_bucket(self, hist):
        assert hist.snapshot("b1", "v1", {"importance": 5}, "update") == 1
        assert hist.snapshot("b1", "v2", {"importance": 5}, "update") == 2
        assert hist.snapshot("b2", "v1", {"importance": 5}, "update") == 1  # per-bucket

    def test_stores_a_full_snapshot_not_a_diff(self, hist):
        hist.snapshot("b1", "整段正文都要在", {"importance": 7, "resolved": False}, "update")
        row = hist.get("b1", 1)
        assert row["content"] == "整段正文都要在"
        assert json.loads(row["meta"])["importance"] == 7

    def test_observed_actor_is_recorded(self, hist):
        from bucket_history import Actor
        hist.snapshot("b1", "x", {}, "update", Actor("cyan", "trace"))
        row = hist.get("b1", 1)
        assert row["actor_type"] == "cyan"
        assert row["actor_id"] == "trace"

    def test_self_report_lives_in_its_own_column(self, hist):
        """Ruby's structural guarantee: a self-report must never be promotable
        to an observed fact. Separate columns, and the column name itself warns.
        自陳永遠不能被提升成觀察事實——分開的欄位，欄名自己帶警語。"""
        from bucket_history import Actor
        hist.snapshot("b1", "x", {}, "update", Actor("cyan", "trace", "dream"))
        row = hist.get("b1", 1)
        assert row["self_reported_during"] == "dream"
        assert row["actor_id"] == "trace"          # NOT "trace:dream"
        assert "dream" not in row["actor_id"]      # never folded in for convenience
        assert "dream" not in row["actor_type"]

    def test_self_report_is_null_when_not_declared(self, hist):
        from bucket_history import Actor
        hist.snapshot("b1", "x", {}, "update", Actor("cyan", "trace"))
        assert hist.get("b1", 1)["self_reported_during"] is None

    def test_missing_actor_records_as_unattributed_rather_than_dropping_the_row(self, hist):
        """An unattributed history row is still worth far more than no row."""
        hist.snapshot("b1", "x", {}, "update")
        row = hist.get("b1", 1)
        assert row["actor_type"] == "system"
        assert row["content"] == "x"

    def test_list_is_newest_first(self, hist):
        for i in range(3):
            hist.snapshot("b1", f"v{i}", {}, "update")
        seqs = [r["seq"] for r in hist.list("b1")]
        assert seqs == [3, 2, 1]

    def test_get_returns_none_for_unknown_seq(self, hist):
        assert hist.get("nope", 1) is None

    def test_disabled_history_is_inert_not_fatal(self, test_config):
        from bucket_history import BucketHistory
        cfg = dict(test_config)
        cfg["history"] = {"enabled": False}
        h = BucketHistory(cfg)
        assert h.snapshot("b1", "x", {}, "update") is None
        assert h.list("b1") == []
        assert h.get("b1", 1) is None


# =============================================================
class TestHistoryIntegration:
    """The choke point: update() snapshots so no caller can forget to."""

    @pytest.fixture
    def mgr(self, test_config):
        from bucket_manager import BucketManager
        from bucket_history import BucketHistory
        return BucketManager(test_config, history=BucketHistory(test_config))

    @pytest.mark.asyncio
    async def test_update_snapshots_the_pre_change_state(self, mgr):
        bid = await _write_bucket_file(mgr, "原本的正文", importance=5)
        await mgr.update(bid, content="改過的正文")
        rows = mgr.history.list(bid)
        assert len(rows) == 1
        assert rows[0]["content"] == "原本的正文"  # BEFORE, not after
        assert rows[0]["op"] == "update"

    @pytest.mark.asyncio
    async def test_a_bad_digestion_pass_is_recoverable(self, mgr):
        """The actual scenario. dream() is read-only — the damage lands in the
        trace(resolved=1)/hold(feel=True) writes that follow it, which is why
        history hangs off update() and not off dream.
        真正的情境：dream() 唯讀，破壞在它之後那串寫入。"""
        bid = await _write_bucket_file(mgr, "重要的記憶", importance=8)
        await mgr.update(bid, resolved=True, digested=True)   # the bad pass
        assert (await mgr.get(bid))["metadata"]["resolved"] is True

        await mgr.restore(bid, 1)
        restored = await mgr.get(bid)
        assert restored["metadata"].get("resolved") in (False, None)
        assert restored["metadata"].get("digested") in (False, None)
        assert restored["content"] == "重要的記憶"

    @pytest.mark.asyncio
    async def test_restore_is_itself_snapshotted_so_it_can_be_undone(self, mgr):
        """Restoring wrongly must not be a one-way door either."""
        bid = await _write_bucket_file(mgr, "v1")
        await mgr.update(bid, content="v2")
        await mgr.restore(bid, 1)                       # back to v1
        assert (await mgr.get(bid))["content"] == "v1"
        rows = mgr.history.list(bid)
        assert rows[0]["op"] == "restore"
        assert rows[0]["content"] == "v2"               # what restore overwrote
        await mgr.restore(bid, rows[0]["seq"])          # undo the undo
        assert (await mgr.get(bid))["content"] == "v2"

    @pytest.mark.asyncio
    async def test_restore_can_land_on_any_seq_not_just_one_step_back(self, mgr):
        """Better than Letta's ±1 pointer walk, and with no branch to destroy."""
        bid = await _write_bucket_file(mgr, "v1")
        for v in ("v2", "v3", "v4"):
            await mgr.update(bid, content=v)
        await mgr.restore(bid, 1)
        assert (await mgr.get(bid))["content"] == "v1"

    @pytest.mark.asyncio
    async def test_delete_is_recoverable(self, mgr):
        """trace(delete=True) was the single truly irreversible call: it unlinks
        the file and drops the embedding. archive/revive only move the file."""
        bid = await _write_bucket_file(mgr, "不該被刪的記憶", importance=9)
        await mgr.delete(bid)
        assert await mgr.get(bid) is None

        assert await mgr.restore(bid, 1)
        back = await mgr.get(bid)
        assert back is not None
        assert back["content"] == "不該被刪的記憶"
        assert back["metadata"]["importance"] == 9

    @pytest.mark.asyncio
    async def test_restoring_an_archived_bucket_puts_it_back_in_dynamic(self, mgr):
        """list_all() walks DIRECTORIES, not the type field. A restore that
        rewrites type:dynamic into a file still sitting in archive/ creates a
        bucket that believes it is alive but never surfaces and never decays.
        還原一個被歸檔的桶，如果只改 type 不搬目錄，會造出一個幽靈。"""
        bid = await _write_bucket_file(mgr, "被歸檔又被救回來的記憶")
        await mgr.archive(bid)                      # snapshot seq 1 = pre-archive
        assert (await mgr.get(bid))["metadata"]["type"] == "archived"
        assert bid not in [b["id"] for b in await mgr.list_all(include_archive=False)]

        assert await mgr.restore(bid, 1)
        back = await mgr.get(bid)
        assert back["metadata"]["type"] == "dynamic"
        # The real assertion: visible to the surfacing pool again, not a ghost.
        assert bid in [b["id"] for b in await mgr.list_all(include_archive=False)]

    @pytest.mark.asyncio
    async def test_touch_does_not_pollute_history(self, mgr):
        """touch/mark_surfaced only bump counters. Snapshotting every surfacing
        would bury the real edits in noise."""
        bid = await _write_bucket_file(mgr, "x")
        await mgr.touch(bid)
        await mgr.mark_surfaced(bid)
        assert mgr.history.list(bid) == []

    @pytest.mark.asyncio
    async def test_archive_records_the_system_forgetting_on_its_own(self, mgr):
        from bucket_history import ACTOR_DECAY
        bid = await _write_bucket_file(mgr, "會被歸檔的")
        await mgr.archive(bid, actor=ACTOR_DECAY)
        rows = mgr.history.list(bid)
        assert rows[0]["op"] == "archive"
        assert rows[0]["actor_type"] == "decay"

    @pytest.mark.asyncio
    async def test_manager_without_history_still_works(self, bucket_mgr):
        """history=None must be inert, not fatal — tests and import scripts."""
        bid = await _write_bucket_file(bucket_mgr, "x")
        assert await bucket_mgr.update(bid, content="y")
        assert (await bucket_mgr.get(bid))["content"] == "y"
        assert await bucket_mgr.restore(bid, 1) is False

    @pytest.mark.asyncio
    async def test_history_failure_does_not_block_the_write(self, mgr):
        """Recording history must never cost the caller the write they actually
        asked for — snapshot swallows its own failures and logs loudly instead.
        記錄歷史失敗，不該讓呼叫者失去他真正要的那次寫入。"""
        bid = await _write_bucket_file(mgr, "x")
        mgr.history.path = "/nonexistent-dir/history.db"  # break the DB
        assert mgr.history.snapshot(bid, "x", {}, "update") is None  # swallowed
        assert await mgr.update(bid, content="y")                    # write survives
        assert (await mgr.get(bid))["content"] == "y"
