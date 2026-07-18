# ============================================================
# Tests for the 2026-07-12 batch-1 metabolism upgrade
# 2026-07-12 第一批代謝升級的測試
#
# Covers: saturation hysteresis, near-top weighted intent pick,
# satisfy degree + engage ledger event, libido cross-inhibition,
# plan-aware drive_boosts with the plan strength cap, the narrowed
# fixation collection (plans + affects_desire only), the trace
# affects_desire flag, and the DesireStore append-only ledger.
# ============================================================

import asyncio
import json
import os
import re
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import desire as dk
import server as srv


def _run(coro):
    return asyncio.run(coro)


NOW = datetime(2026, 7, 12, 10, 0, 0)


def _state(**drives):
    s = dk.default_state(NOW)
    s["drives"].update(drives)
    return s


# ---------------------------------------------------------
# 1. Saturation hysteresis 高位消退
# ---------------------------------------------------------
class TestSaturationHysteresis:
    def test_rise_to_ceiling_enters_decay_mode(self):
        s = _state(curiosity=0.84)
        s2 = dk.tick(s, NOW + timedelta(hours=1))  # 0.84 + 0.025 = 0.865 ≥ 0.85
        assert s2["saturated"].get("curiosity") is True

    def test_saturated_drive_falls_instead_of_rising(self):
        s = _state(curiosity=0.86)
        s["saturated"] = {"curiosity": True}
        s2 = dk.tick(s, NOW + timedelta(minutes=30))
        assert s2["drives"]["curiosity"] < 0.86

    def test_exits_at_floor_and_stops_there(self):
        # curiosity fall speed: (0.85-0.65)/1h — one hour from ceil reaches floor
        s = _state(curiosity=0.85)
        s["saturated"] = {"curiosity": True}
        s2 = dk.tick(s, NOW + timedelta(hours=2))
        assert s2["drives"]["curiosity"] == pytest.approx(dk.SAT_FLOOR)
        assert "curiosity" not in s2["saturated"]

    def test_below_floor_value_is_never_lifted_back(self):
        # satisfy already dropped it to 0.30 while the flag was still on
        s = _state(curiosity=0.30)
        s["saturated"] = {"curiosity": True}
        s2 = dk.tick(s, NOW + timedelta(minutes=10))
        assert s2["drives"]["curiosity"] <= 0.30
        assert "curiosity" not in s2["saturated"]

    def test_deep_drives_fall_slower_than_light(self):
        deep = _state(miss_ruby=0.85)
        deep["saturated"] = {"miss_ruby": True}
        light = _state(social=0.85)
        light["saturated"] = {"social": True}
        d2 = dk.tick(deep, NOW + timedelta(hours=1))
        l2 = dk.tick(light, NOW + timedelta(hours=1))
        drop_deep = 0.85 - d2["drives"]["miss_ruby"]
        drop_light = 0.85 - l2["drives"]["social"]
        assert drop_light > drop_deep


# ---------------------------------------------------------
# 2. Near-top weighted pick 近高位加權抽選
# ---------------------------------------------------------
class TestWeightedIntentPick:
    def test_clear_winner_is_deterministic(self):
        s = _state(curiosity=0.9, social=0.55)
        for _ in range(5):
            intent = dk.pick_intent(s, {}, NOW)
            assert intent["drive"] == "curiosity"

    def test_band_pick_is_stable_within_window(self):
        s = _state(curiosity=0.9, creation=0.85, social=0.82)
        first = dk.pick_intent(s, {}, NOW)["drive"]
        for sec in range(1, 200, 40):
            assert dk.pick_intent(s, {}, NOW + timedelta(seconds=sec))["drive"] == first

    def test_band_pick_varies_across_windows(self):
        s = _state(curiosity=0.9, creation=0.85, social=0.82)
        picks = {
            dk.pick_intent(s, {}, NOW + timedelta(seconds=300 * i))["drive"]
            for i in range(40)
        }
        assert len(picks) >= 2  # 榜首不再永遠霸榜

    def test_out_of_band_never_picked(self):
        s = _state(curiosity=0.9, creation=0.85, social=0.60)
        picks = {
            dk.pick_intent(s, {}, NOW + timedelta(seconds=300 * i))["drive"]
            for i in range(40)
        }
        assert "social" not in picks


# ---------------------------------------------------------
# 3. satisfy degree + engage
# ---------------------------------------------------------
class TestSatisfyDegreeAndEngage:
    def test_full_degree_matches_old_behavior(self):
        # 潮汐 v2（2026-07-19）：主維 mult 平方，explore 0.50→0.25
        s = _state(curiosity=0.8)
        s2 = dk.satisfy(s, "explore", NOW, degree=1.0)
        assert s2["drives"]["curiosity"] == pytest.approx(0.8 * 0.25)

    def test_half_degree_halves_the_drop(self):
        s = _state(curiosity=0.8)
        s2 = dk.satisfy(s, "explore", NOW, degree=0.5)
        # 潮汐 v2：eff = 1 - 0.5*(1-0.25) = 0.625
        assert s2["drives"]["curiosity"] == pytest.approx(0.8 * 0.625)
        assert any(e["kind"] == "satisfy:explore:d0.50" for e in s2["events"])

    def test_zero_degree_changes_nothing(self):
        s = _state(curiosity=0.8)
        s2 = dk.satisfy(s, "explore", NOW, degree=0.0)
        assert s2["drives"]["curiosity"] == pytest.approx(0.8)

    def test_engage_logs_without_moving_levels(self):
        s = _state(curiosity=0.8, libido=0.5)
        s2 = dk.engage(s, "explore", NOW, note="翻了半篇論文")
        assert s2["drives"] == s["drives"]
        assert any(e["kind"] == "engage:explore" for e in s2["events"])

    def test_engage_rejects_unknown_verb(self):
        with pytest.raises(ValueError):
            dk.engage(_state(), "procrastinate", NOW)


# ---------------------------------------------------------
# 4. Cross-inhibition 互相制約
# ---------------------------------------------------------
class TestCrossInhibition:
    @pytest.mark.parametrize("verb", ["explore", "browse", "create", "chore"])
    def test_self_directed_verbs_gently_lower_libido(self, verb):
        s = _state(libido=0.6)
        s2 = dk.satisfy(s, verb, NOW)
        assert s2["drives"]["libido"] == pytest.approx(0.6 * 0.95)

    def test_murmur_does_not_touch_libido(self):
        s = _state(libido=0.6)
        s2 = dk.satisfy(s, "murmur", NOW)
        assert s2["drives"]["libido"] == pytest.approx(0.6)


# ---------------------------------------------------------
# 5. drive_boosts with plans 執念層的 plan 接線
# ---------------------------------------------------------
class TestDriveBoostsWithPlans:
    def test_plan_feeds_its_target_drive(self):
        boosts = dk.drive_boosts([
            {"id": "p1", "name": "陪她去花市", "kind": "plan", "drive": "miss_ruby", "weight": 0.6},
        ])
        assert "miss_ruby" in boosts
        assert boosts["miss_ruby"]["boost"] == pytest.approx(dk.FIXATION_BOOST * 0.6 / 3, abs=1e-4)
        assert "陪她去花市" in boosts["miss_ruby"]["sources"]

    def test_plan_share_is_capped(self):
        plans = [
            {"id": f"p{i}", "name": f"任務{i}", "kind": "plan", "drive": "duty", "weight": 1.0}
            for i in range(3)
        ]
        boosts = dk.drive_boosts(plans)
        # 三條滿重 plan：strength 被 cap 到 0.6×3/3=0.6 → boost 0.21，不打滿 0.35
        assert boosts["duty"]["boost"] == pytest.approx(dk.FIXATION_BOOST * dk.PLAN_STRENGTH_CAP, abs=1e-4)

    def test_memories_alone_unchanged_scale(self):
        boosts = dk.drive_boosts([
            {"id": "m1", "name": "還鯁著的事", "domains": ["自省"], "score": 20.0},
        ])
        assert boosts["reflection"]["boost"] == pytest.approx(dk.FIXATION_BOOST * 1.0 / 3, abs=1e-4)

    def test_mixed_memory_and_plan(self):
        boosts = dk.drive_boosts([
            {"id": "m1", "name": "心事", "domains": ["約定"], "score": 20.0},
            {"id": "p1", "name": "承諾", "kind": "plan", "drive": "miss_ruby", "weight": 1.0},
        ])
        assert boosts["miss_ruby"]["boost"] == pytest.approx(dk.FIXATION_BOOST * 2.0 / 3, abs=1e-4)
        assert len(boosts["miss_ruby"]["sources"]) == 2

    def test_plan_without_valid_drive_ignored(self):
        boosts = dk.drive_boosts([
            {"id": "p1", "name": "壞資料", "kind": "plan", "drive": "possess", "weight": 1.0},
        ])
        assert boosts == {}


# ---------------------------------------------------------
# 6. Narrowed fixation collection 收窄後的執念收集（server 端）
# ---------------------------------------------------------
@pytest.fixture
def wired(test_config, bucket_mgr, mock_embedding_engine, mock_dehydrator, monkeypatch):
    decay_stub = MagicMock()
    decay_stub.ensure_started = AsyncMock()
    decay_stub.calculate_score = lambda meta: 10.0
    monkeypatch.setattr(srv, "config", test_config)
    monkeypatch.setattr(srv, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(srv, "embedding_engine", mock_embedding_engine)
    monkeypatch.setattr(srv, "dehydrator", mock_dehydrator)
    monkeypatch.setattr(srv, "decay_engine", decay_stub)
    monkeypatch.setattr(srv, "desire_store", dk.DesireStore(test_config["buckets_dir"]))
    return srv


class TestNarrowedFixationCollection:
    def test_only_plans_and_flagged_buckets_feed(self, wired, bucket_mgr):
        _run(wired.plan(content="修好執念接線", kind="task"))
        plain = _run(bucket_mgr.create(content="普通未解決記憶", name="普通記憶", domain=["自省"]))
        flagged = _run(bucket_mgr.create(content="還鯁著的心事", name="鯁著的心事", domain=["自省"]))
        _run(bucket_mgr.update(flagged, affects_desire=True))

        items = _run(wired._desire_fixation_buckets())
        kinds = {(i.get("kind", "memory"), i["name"]) for i in items}
        assert ("plan", "修好執念接線") in kinds
        assert ("memory", "鯁著的心事") in kinds
        assert all(i["name"] != "普通記憶" for i in items)

    def test_plan_without_target_drive_not_fed(self, wired, bucket_mgr):
        bid = _run(bucket_mgr.create(
            content="舊 plan", name="舊plan", bucket_type="plan",
            extra_meta={"status": "active", "weight": 0.7},
        ))
        items = _run(wired._desire_fixation_buckets())
        assert all(i["id"] != bid for i in items)

    def test_resolved_flagged_bucket_not_fed(self, wired, bucket_mgr):
        bid = _run(bucket_mgr.create(content="已談開的心事", name="已談開", domain=["自省"]))
        _run(bucket_mgr.update(bid, affects_desire=True, resolved=True))
        items = _run(wired._desire_fixation_buckets())
        assert all(i["id"] != bid for i in items)


class TestTraceAffectsDesire:
    def test_flag_and_unflag_dynamic_bucket(self, wired, bucket_mgr):
        bid = _run(bucket_mgr.create(content="心事一樁", name="心事一樁"))
        r = _run(wired.trace(bid, affects_desire=1))
        assert "已掛成執念" in r
        assert _run(bucket_mgr.get(bid))["metadata"]["affects_desire"] is True
        r2 = _run(wired.trace(bid, affects_desire=0))
        assert "已從執念取下" in r2

    def test_rejected_on_plan_bucket(self, wired, bucket_mgr):
        out = _run(wired.plan(content="一個承諾", kind="promise"))
        bid = re.search(r"plan→(\w+)", out).group(1)
        r = _run(wired.trace(bid, affects_desire=1))
        assert "只能用在動態桶" in r


# ---------------------------------------------------------
# 7. DesireStore append-only ledger
# ---------------------------------------------------------
class TestDesireLedger:
    def test_mutated_events_are_appended(self, tmp_path):
        store = dk.DesireStore(str(tmp_path))
        store.mutate(lambda st: dk.feed(st, "curiosity", 0.1, NOW, event="測試餵入"), NOW)
        store.mutate(lambda st: dk.engage(st, "explore", NOW, note="看了一圈"), NOW)
        lines = open(os.path.join(str(tmp_path), "desire_ledger.jsonl"), encoding="utf-8").read().splitlines()
        kinds = [json.loads(l)["kind"] for l in lines]
        assert "feed:curiosity:+0.10" in kinds
        assert "engage:explore" in kinds

    def test_tick_only_mutation_appends_nothing(self, tmp_path):
        store = dk.DesireStore(str(tmp_path))
        store.mutate(lambda st: st, NOW)
        assert not os.path.exists(os.path.join(str(tmp_path), "desire_ledger.jsonl"))
