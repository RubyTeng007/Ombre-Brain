# ============================================================
# Tests for the 2026-07-10 upstream-inspired batch
# 2026-07-10 借鑑上游批次的測試
#
# Covers: datetime normalization at the read layer, embedding text LRU,
# letter-prefix vector separation, importance tier reservation, plan bucket
# lifecycle (no decay / no search / dream-tail data), clean_llm_json,
# and configurable API timeouts.
# ============================================================

import os
import json
import asyncio
import pytest
from datetime import datetime, timedelta

from utils import (
    clean_llm_json,
    load_config,
    parse_bucket_ts,
    select_importance_tiers,
)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------
# 1. datetime normalization — hand-edited buckets with unquoted
#    timestamps must not poison sorts or JSON serialization
# ---------------------------------------------------------
class TestDatetimeNormalization:
    def test_unquoted_timestamps_come_back_as_strings(self, bucket_mgr, tmp_path):
        raw = (
            "---\n"
            "id: handedit01234\n"
            "name: 手編輯桶\n"
            "created: 2026-06-23T16:13:25.980248+00:00\n"
            "last_active: 2026-07-01T01:19:06+08:00\n"
            "importance: 5\n"
            "type: dynamic\n"
            "---\n"
            "被 Obsidian 手編輯過的桶\n"
        )
        target_dir = os.path.join(bucket_mgr.dynamic_dir, "未分類")
        os.makedirs(target_dir, exist_ok=True)
        with open(os.path.join(target_dir, "handedit01234.md"), "w", encoding="utf-8") as f:
            f.write(raw)

        bucket = _run(bucket_mgr.get("handedit01234"))
        assert bucket is not None
        assert isinstance(bucket["metadata"]["created"], str)
        assert isinstance(bucket["metadata"]["last_active"], str)
        # JSON serialization must not raise
        json.dumps(bucket["metadata"])

    def test_mixed_timestamp_sort_survives(self, bucket_mgr):
        raw = (
            "---\n"
            "id: handedit99999\n"
            "name: 手編輯桶二\n"
            "created: 2026-06-23T16:13:25+00:00\n"
            "importance: 5\n"
            "type: dynamic\n"
            "---\n"
            "正文\n"
        )
        target_dir = os.path.join(bucket_mgr.dynamic_dir, "未分類")
        os.makedirs(target_dir, exist_ok=True)
        with open(os.path.join(target_dir, "handedit99999.md"), "w", encoding="utf-8") as f:
            f.write(raw)
        _run(bucket_mgr.create(content="正常寫入的桶", tags=[], domain=["日常"]))

        buckets = _run(bucket_mgr.list_all())
        # This sort raised TypeError before normalization (str vs datetime)
        buckets.sort(key=lambda b: str(b["metadata"].get("created", "")), reverse=True)
        assert len(buckets) >= 2


# ---------------------------------------------------------
# 2. parse_bucket_ts — naive/aware mixing
# ---------------------------------------------------------
class TestParseBucketTs:
    def test_naive_passthrough(self):
        ts = parse_bucket_ts("2026-07-10T01:00:00")
        assert ts == datetime(2026, 7, 10, 1, 0, 0)

    def test_aware_converted_to_naive_local(self):
        ts = parse_bucket_ts("2026-07-10T01:00:00+08:00")
        assert ts is not None and ts.tzinfo is None

    def test_garbage_returns_none(self):
        assert parse_bucket_ts("not-a-date") is None
        assert parse_bucket_ts(None) is None


# ---------------------------------------------------------
# 3. embedding text LRU — same text embeds once
# ---------------------------------------------------------
class TestEmbeddingLRU:
    def _engine(self, tmp_path):
        from embedding_engine import EmbeddingEngine
        cfg = {
            "buckets_dir": str(tmp_path / "buckets"),
            "embedding": {"api_key": "fake-key", "enabled": True},
            "dehydration": {},
        }
        return EmbeddingEngine(cfg)

    def test_same_text_hits_cache(self, tmp_path):
        engine = self._engine(tmp_path)
        calls = {"n": 0}

        async def fake_embed(truncated):
            calls["n"] += 1
            return [1.0, 0.0, 0.0]

        engine._embed_uncached = fake_embed
        v1 = _run(engine._generate_embedding("同一個查詢"))
        v2 = _run(engine._generate_embedding("同一個查詢"))
        assert v1 == v2 == [1.0, 0.0, 0.0]
        assert calls["n"] == 1

    def test_failed_embed_not_cached(self, tmp_path):
        engine = self._engine(tmp_path)
        calls = {"n": 0}

        async def fake_embed(truncated):
            calls["n"] += 1
            return []

        engine._embed_uncached = fake_embed
        _run(engine._generate_embedding("失敗的查詢"))
        _run(engine._generate_embedding("失敗的查詢"))
        assert calls["n"] == 2  # failures retry, never cached


# ---------------------------------------------------------
# 4. letter-prefix vector separation
# ---------------------------------------------------------
class TestLetterPrefixVectors:
    def test_default_search_excludes_letters_and_prefix_selects_them(self, tmp_path):
        from embedding_engine import EmbeddingEngine
        cfg = {
            "buckets_dir": str(tmp_path / "buckets"),
            "embedding": {"api_key": "fake-key", "enabled": True},
            "dehydration": {},
        }
        engine = EmbeddingEngine(cfg)
        engine._store_embedding("bucket001", [1.0, 0.0])
        engine._store_embedding("letter:abc123", [1.0, 0.0])

        async def fake_embed(truncated):
            return [1.0, 0.0]

        engine._embed_uncached = fake_embed

        default_hits = {bid for bid, _ in _run(engine.search_similar("q", top_k=10))}
        assert default_hits == {"bucket001"}

        letter_hits = {bid for bid, _ in _run(engine.search_similar("q", top_k=10, id_prefix="letter:"))}
        assert letter_hits == {"letter:abc123"}


# ---------------------------------------------------------
# 5. importance tier reservation
# ---------------------------------------------------------
class TestImportanceTiers:
    def _bucket(self, bid, importance, last_active):
        return {
            "id": bid,
            "metadata": {"importance": importance, "last_active": last_active},
        }

    def test_demoted_bucket_keeps_a_seat(self):
        now = datetime.now()
        # 25 importance-10 buckets + 1 freshly-demoted importance-9 bucket
        tens = [
            self._bucket(f"t{i:02d}", 10, (now - timedelta(days=i + 1)).isoformat())
            for i in range(25)
        ]
        demoted = self._bucket("demoted", 9, now.isoformat())
        filtered = sorted(tens + [demoted],
                          key=lambda b: b["metadata"]["importance"], reverse=True)
        picked = select_importance_tiers(filtered, cap=20)
        assert any(b["id"] == "demoted" for b in picked)
        assert len(picked) == 20

    def test_plain_case_unchanged(self):
        now = datetime.now()
        buckets = [
            self._bucket(f"b{i}", 10 - i, (now - timedelta(days=i)).isoformat())
            for i in range(5)
        ]
        picked = select_importance_tiers(buckets, cap=20)
        assert [b["id"] for b in picked] == [b["id"] for b in buckets]


# ---------------------------------------------------------
# 6. plan bucket lifecycle
# ---------------------------------------------------------
class TestPlanBuckets:
    def _create_plan(self, bucket_mgr, content="陪 Ruby 去建國花市挑植物"):
        return _run(bucket_mgr.create(
            content=content,
            tags=["plan"],
            importance=5,
            domain=["約定"],
            valence=0.5,
            arousal=0.4,
            name=content[:24],
            bucket_type="plan",
            extra_meta={"status": "active", "weight": 0.7,
                        "related_bucket": "", "why_remembered": "答應過的事"},
        ))

    def test_plan_lands_in_plan_dir_with_meta(self, bucket_mgr):
        bid = self._create_plan(bucket_mgr)
        bucket = _run(bucket_mgr.get(bid))
        assert bucket["metadata"]["type"] == "plan"
        assert bucket["metadata"]["status"] == "active"
        assert bucket["metadata"]["weight"] == 0.7
        assert bucket["metadata"]["why_remembered"] == "答應過的事"
        assert os.path.commonpath([bucket["path"], bucket_mgr.plan_dir]) == bucket_mgr.plan_dir

    def test_plan_never_decays_or_archives(self, bucket_mgr, decay_eng):
        bid = self._create_plan(bucket_mgr)
        bucket = _run(bucket_mgr.get(bid))
        assert decay_eng.calculate_score(bucket["metadata"]) == 50.0
        result = _run(decay_eng.run_decay_cycle())
        assert _run(bucket_mgr.get(bid))["metadata"]["type"] == "plan"
        assert result["archived"] == 0

    def test_plan_excluded_from_search(self, bucket_mgr):
        self._create_plan(bucket_mgr, content="花市 挑植物 承諾")
        results = _run(bucket_mgr.search("花市 挑植物"))
        assert all(b["metadata"].get("type") != "plan" for b in results)

    def test_plan_status_update(self, bucket_mgr):
        bid = self._create_plan(bucket_mgr)
        ok = _run(bucket_mgr.update(bid, status="resolved", weight=0.2))
        assert ok
        meta = _run(bucket_mgr.get(bid))["metadata"]
        assert meta["status"] == "resolved"
        assert meta["weight"] == 0.2

    def test_invalid_status_ignored(self, bucket_mgr):
        bid = self._create_plan(bucket_mgr)
        _run(bucket_mgr.update(bid, status="nonsense"))
        assert _run(bucket_mgr.get(bid))["metadata"]["status"] == "active"

    def test_stats_count_plans(self, bucket_mgr):
        self._create_plan(bucket_mgr)
        stats = _run(bucket_mgr.get_stats())
        assert stats["plan_count"] == 1


# ---------------------------------------------------------
# 7. clean_llm_json
# ---------------------------------------------------------
class TestCleanLlmJson:
    def test_plain_json_passthrough(self):
        assert json.loads(clean_llm_json('{"a": 1}')) == {"a": 1}

    def test_code_fence_stripped(self):
        raw = "```json\n[{\"content\": \"x\"}]\n```"
        assert json.loads(clean_llm_json(raw)) == [{"content": "x"}]

    def test_chatter_around_payload(self):
        raw = "好的，以下是拆分結果：\n[{\"content\": \"x\"}]\n希望對你有幫助！"
        assert json.loads(clean_llm_json(raw)) == [{"content": "x"}]

    def test_no_json_raises(self):
        with pytest.raises(ValueError):
            clean_llm_json("這裡完全沒有 JSON")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            clean_llm_json("")


# ---------------------------------------------------------
# 8. configurable timeouts via env
# ---------------------------------------------------------
class TestTimeoutConfig:
    def test_env_overrides(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OMBRE_BUCKETS_DIR", str(tmp_path / "buckets"))
        monkeypatch.setenv("OMBRE_COMPRESS_TIMEOUT_SECONDS", "90")
        monkeypatch.setenv("OMBRE_EMBED_TIMEOUT_SECONDS", "45")
        cfg = load_config(config_path=str(tmp_path / "missing.yaml"))
        assert cfg["dehydration"]["timeout_seconds"] == 90.0
        assert cfg["embedding"]["timeout_seconds"] == 45.0

    def test_invalid_env_ignored(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OMBRE_BUCKETS_DIR", str(tmp_path / "buckets"))
        monkeypatch.setenv("OMBRE_COMPRESS_TIMEOUT_SECONDS", "not-a-number")
        cfg = load_config(config_path=str(tmp_path / "missing.yaml"))
        assert "timeout_seconds" not in cfg.get("dehydration", {})
