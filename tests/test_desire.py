# ============================================================
# Test: Desire System Kernel — pure local, no LLM, no network
# 測試：慾望系統內核 —— 純本地
#
# Verifies:
#   - tick 時間流逝（上漲/clamp/fatigue 回復/異常時鐘防護）
#   - drive_boosts 執念映射（繁體 domain → 維度、top-N、權重歸一）
#   - pick_intent 提案（argmax、執念翻盤、fatigue 閘、intimacy 閘、
#     veto 冷卻、quiet 低水位、第一人稱 reason）
#   - satisfy 乘性回落與交叉沾光
#   - feed / veto / set_gate 事件紀錄與鐵律
#   - DesireStore 持久化（round-trip、壞檔容錯、惰性 tick、純函數性）
# ============================================================

import json
import os
from datetime import datetime, timedelta

import pytest

import desire
from desire import (
    DRIVE_KEYS, FATIGUE_GATE, MIN_INTENT_SCORE, MAX_EVENTS,
    default_state, tick, drive_boosts, pick_intent, satisfy, engage, defer, outreach, feed, veto,
    set_gate, DesireStore,
)

NOW = datetime(2026, 7, 4, 12, 0, 0)


def fresh(now=NOW):
    return default_state(now)


# ---------------------------------------------------------
# tick：時間流逝
# ---------------------------------------------------------

class TestTick:
    def test_drives_rise_over_time(self):
        s = fresh()
        s2 = tick(s, NOW + timedelta(hours=10))
        for k in DRIVE_KEYS:
            if desire.RISE_PER_HOUR[k] == 0.0:
                # 事件驅動維度（knot）：零自漲是設計，不隨時間長
                assert s2["drives"][k] == pytest.approx(s["drives"][k]), k
            else:
                assert s2["drives"][k] > s["drives"][k], k

    def test_rise_rate_matches_constant(self):
        s = fresh()
        s2 = tick(s, NOW + timedelta(hours=5))
        expected = s["drives"]["miss_ruby"] + desire.RISE_PER_HOUR["miss_ruby"] * 5
        assert s2["drives"]["miss_ruby"] == pytest.approx(expected, abs=1e-3)

    def test_clamped_at_one(self):
        s = fresh()
        s["drives"]["miss_ruby"] = 0.99
        s2 = tick(s, NOW + timedelta(hours=48))
        assert s2["drives"]["miss_ruby"] == 1.0

    def test_fatigue_recovers(self):
        s = fresh()
        s["fatigue"] = 0.5
        s2 = tick(s, NOW + timedelta(hours=10))
        assert s2["fatigue"] == pytest.approx(0.5 - 0.03 * 10, abs=1e-3)

    def test_fatigue_floor_zero(self):
        s = fresh()
        s["fatigue"] = 0.01
        s2 = tick(s, NOW + timedelta(hours=10))
        assert s2["fatigue"] == 0.0

    def test_clock_anomaly_capped(self):
        """時鐘跳一年也只算 MAX_TICK_HOURS，不會全維暴衝到頂。"""
        s = fresh()
        s2 = tick(s, NOW + timedelta(days=365))
        expected = min(1.0, s["drives"]["duty"] + desire.RISE_PER_HOUR["duty"] * desire.MAX_TICK_HOURS)
        assert s2["drives"]["duty"] == pytest.approx(expected, abs=1e-3)

    def test_bad_timestamp_means_no_time_passes(self):
        s = fresh()
        s["updated_at"] = "not-a-date"
        s2 = tick(s, NOW)
        assert s2["drives"] == s["drives"]

    def test_pure_no_mutation(self):
        s = fresh()
        before = json.dumps(s, sort_keys=True)
        tick(s, NOW + timedelta(hours=5))
        assert json.dumps(s, sort_keys=True) == before


# ---------------------------------------------------------
# drive_boosts：執念層（Ombre 桶 → 召喚力加成）
# ---------------------------------------------------------

class TestDriveBoosts:
    def test_traditional_domain_maps_to_drive(self):
        buckets = [{"id": "a", "name": "像素小家", "domains": ["創作"], "score": 20}]
        b = drive_boosts(buckets)
        assert "creation" in b
        assert b["creation"]["boost"] > 0
        assert "像素小家" in b["creation"]["sources"]

    def test_full_strength_needs_top_n_buckets(self):
        """單桶滿分只給 1/N 的滿加成；N 桶滿分才頂到 FIXATION_BOOST。"""
        one = drive_boosts([{"id": "a", "name": "x", "domains": ["編程"], "score": 20}])
        n = drive_boosts([
            {"id": str(i), "name": f"x{i}", "domains": ["編程"], "score": 20}
            for i in range(desire.FIXATION_TOP_N)
        ])
        assert one["curiosity"]["boost"] == pytest.approx(desire.FIXATION_BOOST / desire.FIXATION_TOP_N, abs=1e-3)
        assert n["curiosity"]["boost"] == pytest.approx(desire.FIXATION_BOOST, abs=1e-3)

    def test_unmapped_domain_ignored(self):
        b = drive_boosts([{"id": "a", "name": "x", "domains": ["穿搭"], "score": 20}])
        assert b == {}

    def test_one_bucket_feeds_one_drive_only(self):
        buckets = [{"id": "a", "name": "x", "domains": ["戀愛", "編程"], "score": 20}]
        b = drive_boosts(buckets)
        assert "miss_ruby" in b and "curiosity" not in b

    def test_empty_and_garbage_safe(self):
        assert drive_boosts([]) == {}
        assert drive_boosts([{"id": "a", "name": "x", "domains": ["編程"], "score": "??"}]) == {}


# ---------------------------------------------------------
# pick_intent：提案
# ---------------------------------------------------------

class TestPickIntent:
    def test_argmax_wins(self):
        s = fresh()
        s["drives"]["curiosity"] = 0.9
        intent = pick_intent(s, {}, NOW)
        assert intent["drive"] == "curiosity"
        assert intent["action"] == "explore"

    def test_fixation_boost_can_flip_winner(self):
        s = fresh()
        s["drives"]["curiosity"] = 0.70
        s["drives"]["duty"] = 0.60
        boosts = {"duty": {"boost": 0.30, "sources": ["StackChan 安全層"]}}
        intent = pick_intent(s, boosts, NOW)
        assert intent["drive"] == "duty"
        assert "StackChan 安全層" in intent["reason"]

    def test_fatigue_gate_beats_everything(self):
        s = fresh()
        s["drives"]["miss_ruby"] = 1.0
        s["fatigue"] = FATIGUE_GATE
        intent = pick_intent(s, {}, NOW)
        assert intent["action"] == "rest"

    def test_quiet_when_all_low(self):
        s = fresh()  # 全維 0.15，低於 MIN_INTENT_SCORE
        intent = pick_intent(s, {}, NOW)
        assert intent["action"] == "quiet"
        assert intent["drive"] is None

    def test_intimacy_gate_excludes_libido(self):
        s = fresh()
        s["drives"]["libido"] = 1.0
        s["drives"]["miss_ruby"] = 0.8
        s["gates"]["intimacy_ok"] = False
        intent = pick_intent(s, {}, NOW)
        assert intent["drive"] == "miss_ruby"  # libido 被門擋住，次高上

    def test_intimacy_open_lets_libido_win(self):
        s = fresh()
        s["drives"]["libido"] = 1.0
        s["drives"]["miss_ruby"] = 0.8
        s = set_gate(s, "intimacy_ok", True, NOW, note="測試：Ruby 開的門")
        intent = pick_intent(s, {}, NOW)
        assert intent["drive"] == "libido"
        assert intent["action"] == "tease"

    def test_intimacy_gate_fails_closed(self):
        # fail-close 鐵律（2026-07-05）：全新狀態、或 gates 鍵遺失（檔案損壞重建），
        # 閘門都必須是關的——「開」只能來自 Ruby 親口的 set_gate。
        assert fresh()["gates"]["intimacy_ok"] is False
        s = fresh()
        s["drives"]["libido"] = 1.0
        s["drives"]["miss_ruby"] = 0.8
        del s["gates"]  # 模擬狀態檔損壞、gates 整塊遺失
        intent = pick_intent(s, {}, NOW)
        assert intent["drive"] == "miss_ruby"  # libido 仍被擋，fallback 也是關

    def test_vetoed_drive_skipped_until_cooldown(self):
        s = fresh()
        s["drives"]["social"] = 0.9
        s["drives"]["creation"] = 0.7
        s2 = veto(s, "social", NOW, "現在不想逛")
        intent = pick_intent(s2, {}, NOW + timedelta(minutes=10))
        assert intent["drive"] == "creation"
        # 冷卻過後可以再提
        intent_later = pick_intent(s2, {}, NOW + timedelta(hours=desire.VETO_COOLDOWN_HOURS + 1))
        assert intent_later["drive"] in ("social", "creation")

    def test_reason_is_first_person_without_source(self):
        s = fresh()
        s["drives"]["miss_ruby"] = 0.9
        intent = pick_intent(s, {}, NOW)
        assert intent["reason"] == "有點想她，心裡冒了句話。"

    def test_score_includes_boost(self):
        s = fresh()
        s["drives"]["creation"] = 0.6
        boosts = {"creation": {"boost": 0.2, "sources": ["獻花動畫"]}}
        intent = pick_intent(s, boosts, NOW)
        assert intent["score"] == pytest.approx(0.8, abs=1e-3)


# ---------------------------------------------------------
# satisfy / feed / veto / gate
# ---------------------------------------------------------

class TestSatisfy:
    def test_main_drive_falls_hard_neighbors_lightly(self):
        s = fresh()
        s["drives"]["libido"] = 0.8
        s["drives"]["miss_ruby"] = 0.6
        s2 = satisfy(s, "tease", NOW)
        assert s2["drives"]["libido"] == pytest.approx(0.8 * 0.5, abs=1e-3)
        assert s2["drives"]["miss_ruby"] == pytest.approx(0.6 * 0.75, abs=1e-3)

    def test_rest_halves_fatigue(self):
        s = fresh()
        s["fatigue"] = 0.8
        s2 = satisfy(s, "rest", NOW)
        assert s2["fatigue"] == pytest.approx(0.4, abs=1e-3)

    def test_unknown_action_raises(self):
        with pytest.raises(ValueError):
            satisfy(fresh(), "hack_the_planet", NOW)

    def test_event_logged(self):
        s2 = satisfy(fresh(), "create", NOW, note="像素客廳畫完了")
        assert any(e["kind"] == "satisfy:create" for e in s2["events"])

    def test_pure_no_mutation(self):
        s = fresh()
        before = json.dumps(s, sort_keys=True)
        satisfy(s, "murmur", NOW)
        assert json.dumps(s, sort_keys=True) == before

    def test_same_wake_same_verb_is_idempotent(self):
        s = fresh()
        s["drives"]["curiosity"] = 0.8
        s2 = satisfy(s, "explore", NOW, degree=0.5, wake_id="w-retry")
        s3 = satisfy(s2, "explore", NOW, degree=0.5, wake_id="w-retry")
        assert s3["drives"]["curiosity"] == s2["drives"]["curiosity"]
        assert len([e for e in s3["events"] if e.get("wake_id") == "w-retry"]) == 1

    def test_same_wake_stays_idempotent_after_event_window_eviction(self):
        s = fresh()
        s["drives"]["curiosity"] = 0.8
        s2 = satisfy(s, "explore", NOW, degree=0.5, wake_id="w-old-retry")
        first_value = s2["drives"]["curiosity"]
        for i in range(MAX_EVENTS + 20):
            s2 = feed(s2, "duty", 0.0, NOW, event=f"evict-{i}")
        assert not any(e.get("wake_id") == "w-old-retry" for e in s2["events"])
        retried = satisfy(s2, "explore", NOW, degree=0.5, wake_id="w-old-retry")
        assert retried["drives"]["curiosity"] == first_value
        assert "w-old-retry|satisfy:explore" in retried["processed_wake_receipts"]


class TestEngageDeferOutreach:
    def test_engage_keeps_water_and_rechecks_later(self):
        s = fresh()
        s["drives"]["curiosity"] = 0.8
        before = dict(s["drives"])
        s2 = engage(s, "explore", NOW, note="只讀了一半", wake_id="w-engage", drive="curiosity")
        assert s2["drives"] == before
        assert "curiosity" in s2["recheck_until"]
        assert pick_intent(s2, {}, NOW + timedelta(minutes=30))["drive"] != "curiosity"

    def test_defer_keeps_water_and_is_idempotent(self):
        s = fresh()
        s["drives"]["creation"] = 0.8
        s2 = defer(s, "creation", NOW, reason="現在想陪她", wake_id="w-defer")
        s3 = defer(s2, "creation", NOW, reason="重試", wake_id="w-defer")
        assert s3["drives"]["creation"] == 0.8
        assert len([e for e in s3["events"] if e.get("wake_id") == "w-defer"]) == 1

    def test_outreach_records_receipt_without_moving_water(self):
        s = fresh()
        before = dict(s["drives"])
        s2 = outreach(s, "text", NOW, note="短訊已送達", wake_id="w-out")
        s3 = outreach(s2, "text", NOW, note="重試", wake_id="w-out")
        assert s3["drives"] == before
        assert len([e for e in s3["events"] if e.get("kind") == "outreach:text"]) == 1


class TestFeedVetoGate:
    def test_feed_bumps_drive(self):
        s2 = feed(fresh(), "curiosity", 0.3, NOW, event="讀到一篇好文")
        assert s2["drives"]["curiosity"] == pytest.approx(0.45, abs=1e-3)
        assert any("讀到一篇好文" in e["note"] for e in s2["events"])

    def test_feed_negative_and_clamp(self):
        s = fresh()
        s2 = feed(s, "social", -0.9, NOW)
        assert s2["drives"]["social"] == 0.0
        s3 = feed(s, "social", 5.0, NOW)  # amount 本身被夾在 ±1
        assert s3["drives"]["social"] <= 1.0

    def test_feed_fatigue_channel(self):
        s2 = feed(fresh(), "fatigue", 0.4, NOW, event="今天燒了很多 token")
        assert s2["fatigue"] == pytest.approx(0.4, abs=1e-3)

    def test_feed_unknown_drive_raises(self):
        with pytest.raises(ValueError):
            feed(fresh(), "hunger", 0.2, NOW)

    def test_veto_damps_and_sets_cooldown(self):
        s = fresh()
        s["drives"]["social"] = 0.9
        s2 = veto(s, "social", NOW, "不是現在")
        assert s2["drives"]["social"] == pytest.approx(0.9 * desire.VETO_DAMP, abs=1e-3)
        assert "social" in s2["veto_until"]
        assert any(e["kind"] == "veto:social" and e["note"] == "不是現在" for e in s2["events"])

    def test_intimacy_gate_toggles(self):
        s2 = set_gate(fresh(), "intimacy_ok", False, NOW, note="月經期")
        assert s2["gates"]["intimacy_ok"] is False
        s3 = set_gate(s2, "intimacy_ok", True, NOW, note="捉i日")
        assert s3["gates"]["intimacy_ok"] is True

    def test_driven_gate_locked_in_phase1(self):
        with pytest.raises(ValueError):
            set_gate(fresh(), "driven", True, NOW)

    def test_events_capped(self):
        s = fresh()
        for i in range(MAX_EVENTS + 20):
            s = feed(s, "curiosity", 0.001, NOW, event=f"e{i}")
        assert len(s["events"]) == MAX_EVENTS
        assert s["events"][-1]["note"] == f"e{MAX_EVENTS + 19}"


# ---------------------------------------------------------
# DesireStore：持久化
# ---------------------------------------------------------

class TestStore:
    def test_round_trip(self, tmp_path):
        store = DesireStore(str(tmp_path))
        s = store.load(NOW)
        s = feed(s, "creation", 0.5, NOW, event="想做東西")
        store.save(s)
        s2 = store.load(NOW)
        assert s2["drives"]["creation"] == pytest.approx(s["drives"]["creation"], abs=1e-3)

    def test_lazy_tick_on_load(self, tmp_path):
        store = DesireStore(str(tmp_path))
        store.save(default_state(NOW))
        s = store.load(NOW + timedelta(hours=10))
        assert s["drives"]["miss_ruby"] > default_state(NOW)["drives"]["miss_ruby"]

    def test_corrupted_file_recovers_last_good_without_reset(self, tmp_path):
        store = DesireStore(str(tmp_path))
        state = default_state(NOW)
        state["drives"]["miss_ruby"] = 0.73
        store.save(state)
        with open(store.path, "w") as f:
            f.write("{not json!!")
        with pytest.warns(RuntimeWarning):
            s = store.load(NOW)
        assert s["drives"]["miss_ruby"] == 0.73

    def test_corrupted_primary_and_backup_fail_closed(self, tmp_path):
        store = DesireStore(str(tmp_path))
        store.save(default_state(NOW))
        for path in (store.path, store.backup_path):
            with open(path, "w") as f:
                f.write("{not json!!")
        with pytest.raises(RuntimeError, match="refusing|unreadable"):
            store.load(NOW)

    def test_missing_keys_backfilled(self, tmp_path):
        store = DesireStore(str(tmp_path))
        with open(store.path, "w") as f:
            json.dump({"version": 1, "updated_at": NOW.isoformat(),
                       "drives": {"miss_ruby": 0.7}}, f)
        s = store.load(NOW)
        for k in DRIVE_KEYS:
            assert k in s["drives"]
        assert "gates" in s and "events" in s and "recheck_until" in s

    def test_mutate_flow(self, tmp_path):
        store = DesireStore(str(tmp_path))
        store.mutate(lambda st: feed(st, "duty", 0.4, NOW, event="還有工作"), NOW)
        s = store.load(NOW)
        assert s["drives"]["duty"] >= 0.4
        assert os.path.exists(store.path)
        # 檔案裡是合法 JSON（atomic write 成功）
        with open(store.path) as f:
            assert "drives" in json.load(f)

    def test_ledger_outbox_retries_without_duplicate(self, tmp_path):
        store = DesireStore(str(tmp_path))
        real_append = store._append_ledger
        store._append_ledger = lambda _events: False
        state = store.mutate(
            lambda st: feed(st, "duty", 0.1, NOW, event="ledger outage"), NOW
        )
        assert len(state["ledger_pending"]) == 1
        assert not os.path.exists(store.ledger_path)

        store._append_ledger = real_append
        store.mutate(lambda st: st, NOW)
        saved = json.loads(open(store.path, encoding="utf-8").read())
        assert saved["ledger_pending"] == []
        lines = open(store.ledger_path, encoding="utf-8").read().splitlines()
        assert len(lines) == 1

        # Simulate crash after ledger append but before pending-clear commit.
        saved["ledger_pending"] = [json.loads(lines[0])]
        store.save(saved)
        store.mutate(lambda st: st, NOW)
        assert len(open(store.ledger_path, encoding="utf-8").read().splitlines()) == 1
