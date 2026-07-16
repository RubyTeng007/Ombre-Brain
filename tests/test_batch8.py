# ============================================================
# Tests for the 2026-07-16 audit batch-8: poison-bucket resilience
# and the decay heartbeat.
# 2026-07-16 審計批8 的測試：毒桶韌性＋衰減心跳。
#
# The two P1s this batch closes:
# 1. importance:null (hand-edited bucket) used to kill calculate_score,
#    which sat under BOTH the decay loop (every bucket after the poison
#    one froze forever) and the breath surfacing sort (whole breath died).
# 2. A tz-aware last_active fell into the inline parse's 999-day fallback
#    and auto-resolved a memory written yesterday.
# Plus: the heartbeat, so a halted loop can no longer hide behind the flag.
# 這批關掉的兩顆 P1：
# 1. importance:null 曾讓 calculate_score 拋錯——它墊在衰減迴圈（毒桶之後
#    全部凍結）與浮現排序（整口呼吸死掉）底下。
# 2. 帶時區的 last_active 掉進行內解析的 999 天 fallback，
#    昨天寫的記憶被自動結案。
# 外加心跳：停擺的迴圈再也不能躲在旗標後面。
# ============================================================

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from decay_engine import DecayEngine
from utils import select_importance_tiers


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


@pytest.fixture
def engine(test_config, bucket_mgr):
    return DecayEngine(test_config, bucket_mgr)


# ---------------------------------------------------------
# 1. Poison metadata must cost the bucket its ranking, not the caller
# 1. 毒 metadata 只能賠上該桶的排序，不能賠上呼叫端
# ---------------------------------------------------------

class TestPoisonBucketScore:
    @pytest.mark.parametrize("poison", [None, "high", "", [], {}])
    def test_score_survives_poison_importance(self, engine, poison):
        score = engine.calculate_score(_meta(importance=poison))
        assert isinstance(score, float)
        # Defaults, not zero: the bucket should rank as an ordinary one.
        assert score == pytest.approx(engine.calculate_score(_meta(importance=5)))

    @pytest.mark.parametrize("poison", [None, "many", ""])
    def test_score_survives_poison_activation(self, engine, poison):
        score = engine.calculate_score(_meta(activation_count=poison))
        assert isinstance(score, float)

    def test_heat_and_score_agree_on_poison(self, engine):
        # calculate_heat already had the `or 5` guard; the two paths must
        # both survive the same poison (the asymmetry was the bug).
        meta = _meta(importance=None)
        assert isinstance(engine.calculate_heat(meta), float)
        assert isinstance(engine.calculate_score(meta), float)

    def test_tier_reservation_survives_poison(self):
        buckets = [
            {"id": "a", "metadata": _meta(importance=None)},
            {"id": "b", "metadata": _meta(importance=9)},
        ]
        picked = select_importance_tiers(buckets, cap=2)
        assert {b["id"] for b in picked} == {"a", "b"}


class TestPoisonBucketCycle:
    def _inject_poison(self, test_config, bucket_id, patch):
        """Hand-edit the bucket file the way Obsidian would — this is the
        documented workflow that produces poison values in the first place.
        像 Obsidian 手編輯那樣直接改桶檔——毒值本來就是從這條
        明文支援的工作流進來的。"""
        from pathlib import Path

        import frontmatter

        for path in Path(test_config["buckets_dir"]).rglob(f"*_{bucket_id}.md"):
            post = frontmatter.load(str(path))
            post.metadata.update(patch)
            path.write_text(frontmatter.dumps(post), encoding="utf-8")
            return True
        return False

    def test_cycle_survives_poison_bucket_and_reaches_the_rest(
        self, engine, bucket_mgr, test_config,
    ):
        """One poison bucket must not freeze every bucket after it.
        一顆毒桶不能凍結它之後的所有桶。"""
        poison_id = _run(bucket_mgr.create(content="毒", name="毒桶"))
        old_id = _run(bucket_mgr.create(content="老", name="老桶"))
        assert self._inject_poison(
            test_config, poison_id, {"importance": None, "resolved": False},
        )
        assert self._inject_poison(
            test_config, old_id, {
                "importance": 1,
                "arousal": 0.0,
                "last_active": (datetime.now() - timedelta(days=400)).isoformat(),
                "created": (datetime.now() - timedelta(days=400)).isoformat(),
            },
        )
        result = _run(engine.run_decay_cycle())
        assert "error" not in result
        # The cycle walked past the poison bucket to the rest of the list.
        assert result["checked"] >= 2

    def test_cycle_skips_non_dict_metadata(self, engine, bucket_mgr, monkeypatch):
        good = {"id": "good", "metadata": _meta(days_ago=0)}
        bad = {"id": "bad", "metadata": "not a dict"}

        async def fake_list_all(include_archive=False):
            return [bad, good]

        monkeypatch.setattr(engine.bucket_mgr, "list_all", fake_list_all)
        result = _run(engine.run_decay_cycle())
        assert "error" not in result
        assert result["checked"] == 1  # good only; bad skipped, cycle alive


# ---------------------------------------------------------
# 2. Auto-resolve: fail toward retention
# 2. 自動結案：失敗方向是保留
# ---------------------------------------------------------

class TestAutoResolveFailDirection:
    def _cycle_with(self, engine, monkeypatch, meta):
        bucket = {"id": "x", "metadata": meta}
        resolved_calls = []

        async def fake_list_all(include_archive=False):
            return [bucket]

        async def fake_update(bid, **kw):
            resolved_calls.append((bid, kw))
            return True

        async def fake_archive(bid, actor=None):
            return True

        async def fake_hygiene():
            return {}

        monkeypatch.setattr(engine.bucket_mgr, "list_all", fake_list_all)
        monkeypatch.setattr(engine.bucket_mgr, "update", fake_update)
        monkeypatch.setattr(engine.bucket_mgr, "archive", fake_archive)
        monkeypatch.setattr(engine, "_vector_hygiene", fake_hygiene)
        _run(engine.run_decay_cycle())
        return resolved_calls

    def test_tz_aware_fresh_memory_is_not_auto_resolved(self, engine, monkeypatch):
        """The old inline parse aged a tz-aware fresh memory to 999 days.
        舊行內解析把帶時區的新記憶算成 999 天。"""
        meta = _meta(importance=3, resolved=False)
        meta["last_active"] = datetime.now(timezone.utc).isoformat()
        calls = self._cycle_with(engine, monkeypatch, meta)
        assert calls == []

    def test_unparseable_timestamp_is_not_auto_resolved(self, engine, monkeypatch):
        meta = _meta(importance=3, resolved=False)
        meta["last_active"] = "not-a-date"
        calls = self._cycle_with(engine, monkeypatch, meta)
        assert calls == []

    def test_genuinely_old_low_importance_still_resolves(self, engine, monkeypatch):
        """The gate itself must keep working — fail-safe, not lobotomy.
        門本身要照常運作——是 fail-safe，不是把功能切掉。"""
        meta = _meta(days_ago=40, importance=3, resolved=False)
        calls = self._cycle_with(engine, monkeypatch, meta)
        assert len(calls) == 1 and calls[0][1].get("resolved") is True

    def test_poison_importance_is_not_auto_resolved(self, engine, monkeypatch):
        # importance None defaults to 5 → above the ≤4 gate → untouched.
        meta = _meta(days_ago=40, importance=None, resolved=False)
        calls = self._cycle_with(engine, monkeypatch, meta)
        assert calls == []


# ---------------------------------------------------------
# 3. Heartbeat: the flag can no longer stand in for the pulse
# 3. 心跳：旗標不能再冒充脈搏
# ---------------------------------------------------------

class TestDecayHeartbeat:
    def test_cycle_stamps_heartbeat(self, engine, bucket_mgr):
        assert engine.heartbeat()["last_cycle_at"] is None
        _run(engine.run_decay_cycle())
        hb = engine.heartbeat()
        assert hb["last_cycle_at"] is not None
        assert isinstance(hb["last_cycle_result"], dict)
        assert "checked" in hb["last_cycle_result"]

    def test_running_without_any_cycle_is_overdue(self, engine):
        engine._running = True
        assert engine.heartbeat()["overdue"] is True

    def test_stale_cycle_is_overdue(self, engine):
        engine._running = True
        stale_h = 2 * float(engine.check_interval) + 1
        engine._last_cycle_at = (
            datetime.now() - timedelta(hours=stale_h)
        ).isoformat()
        engine._last_cycle_result = {"checked": 1}
        assert engine.heartbeat()["overdue"] is True

    def test_fresh_cycle_is_not_overdue(self, engine):
        engine._running = True
        engine._last_cycle_at = datetime.now().isoformat()
        engine._last_cycle_result = {"checked": 1}
        hb = engine.heartbeat()
        assert hb["overdue"] is False and hb["running"] is True

    def test_stopped_engine_reports_not_running(self, engine):
        hb = engine.heartbeat()
        assert hb["running"] is False and hb["overdue"] is False

    def test_dead_task_flips_the_flag(self, engine, monkeypatch):
        """A loop that dies outside its own except must not leave the flag
        claiming 'running' — the exact halt-invisibility failure.
        迴圈死在自己的 except 之外時，旗標不能繼續說「運行中」——
        正是停擺不可見的那個形狀。"""
        async def scenario():
            async def dying_loop():
                raise TypeError("bad config type")

            monkeypatch.setattr(engine, "_background_loop", dying_loop)
            await engine.start()
            assert engine.is_running is True
            await asyncio.sleep(0)  # let the task die and the callback run
            await asyncio.sleep(0)
            assert engine.is_running is False

        _run(scenario())
