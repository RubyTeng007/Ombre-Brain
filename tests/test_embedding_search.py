# ============================================================
# Regression tests for EmbeddingEngine.search_similar
# 向量檢索回歸測試 —— 守住 numpy 批次化後的排序與分數正確性
#
# The default suite runs with embedding disabled, so the vector
# ranking math had no coverage. These tests seed embeddings.db
# with known vectors, stub out the (network) query embedding, and
# assert the ranking + scores match a reference cosine — so the
# numpy batched path can never silently drift from correctness.
# ============================================================

import asyncio
import math
import os

import pytest

from embedding_engine import EmbeddingEngine


def _engine(tmp_path):
    buckets_dir = str(tmp_path / "buckets")
    os.makedirs(buckets_dir, exist_ok=True)
    return EmbeddingEngine({
        "buckets_dir": buckets_dir,
        "embedding": {"enabled": True, "api_key": "test-key", "model": "test-model"},
    })


def _ref_cos(a, b):
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _stub_query(ee, vec):
    async def _fake(text, is_query=False):
        return list(vec)
    ee._generate_embedding = _fake


def test_ranks_by_cosine_and_scores_match_reference(tmp_path):
    ee = _engine(tmp_path)
    vectors = {
        "a": [1.0, 0.0, 0.0],
        "b": [0.0, 1.0, 0.0],
        "c": [0.9, 0.1, 0.0],
        "d": [-1.0, 0.0, 0.0],
    }
    for bid, v in vectors.items():
        ee._store_embedding(bid, v)
    query = [1.0, 0.0, 0.0]
    _stub_query(ee, query)

    res = asyncio.run(ee.search_similar("q", top_k=4))
    ids = [r[0] for r in res]
    assert ids == ["a", "c", "b", "d"]  # cos: 1.0, ~0.994, 0.0, -1.0

    scored = dict(res)
    for bid, v in vectors.items():
        assert abs(scored[bid] - _ref_cos(query, v)) < 1e-9


def test_top_k_limits_results(tmp_path):
    ee = _engine(tmp_path)
    for i in range(10):
        ee._store_embedding(f"b{i}", [float(i), 1.0, 0.0])
    _stub_query(ee, [1.0, 1.0, 0.0])
    assert len(asyncio.run(ee.search_similar("q", top_k=3))) == 3


def test_letter_rows_excluded_by_default_and_isolable(tmp_path):
    ee = _engine(tmp_path)
    ee._store_embedding("bucket1", [1.0, 0.0])
    ee._store_embedding("letter:001", [1.0, 0.0])
    _stub_query(ee, [1.0, 0.0])

    default = [r[0] for r in asyncio.run(ee.search_similar("q", top_k=10))]
    assert "bucket1" in default and "letter:001" not in default

    letters = [r[0] for r in asyncio.run(ee.search_similar("q", top_k=10, id_prefix="letter:"))]
    assert letters == ["letter:001"]


def test_edge_rows_zero_and_wrongdim_kept_at_zero(tmp_path):
    ee = _engine(tmp_path)
    ee._store_embedding("normal", [1.0, 0.0, 0.0])
    ee._store_embedding("zero", [0.0, 0.0, 0.0])       # zero-norm → 0.0
    ee._store_embedding("wrongdim", [1.0, 2.0])         # dim mismatch → 0.0
    _stub_query(ee, [1.0, 0.0, 0.0])

    res = dict(asyncio.run(ee.search_similar("q", top_k=10)))
    assert res["normal"] == pytest.approx(1.0)
    assert res["zero"] == 0.0
    assert res["wrongdim"] == 0.0


def test_zero_norm_query_returns_all_zero_without_crash(tmp_path):
    ee = _engine(tmp_path)
    ee._store_embedding("a", [1.0, 2.0, 3.0])
    _stub_query(ee, [0.0, 0.0, 0.0])
    res = asyncio.run(ee.search_similar("q", top_k=10))
    assert res == [("a", 0.0)]


def test_empty_corpus_returns_empty(tmp_path):
    ee = _engine(tmp_path)
    _stub_query(ee, [1.0, 0.0, 0.0])
    assert asyncio.run(ee.search_similar("q", top_k=10)) == []


def test_disabled_engine_returns_empty(tmp_path):
    buckets_dir = str(tmp_path / "buckets")
    os.makedirs(buckets_dir, exist_ok=True)
    ee = EmbeddingEngine({
        "buckets_dir": buckets_dir,
        "embedding": {"enabled": False, "api_key": ""},
    })
    assert asyncio.run(ee.search_similar("q", top_k=10)) == []


# --- parsed-vector cache (option B) ---

def test_cache_hit_returns_identical_results(tmp_path):
    ee = _engine(tmp_path)
    for i in range(20):
        ee._store_embedding(f"b{i}", [float(i), 1.0, 0.5])
    _stub_query(ee, [3.0, 1.0, 0.5])
    r1 = asyncio.run(ee.search_similar("q", top_k=20))
    r2 = asyncio.run(ee.search_similar("q", top_k=20))  # served from cache
    assert r1 == r2
    assert len(ee._matrix_cache) >= 1  # cache populated


def test_cache_invalidated_on_store(tmp_path):
    ee = _engine(tmp_path)
    ee._store_embedding("a", [1.0, 0.0, 0.0])
    _stub_query(ee, [1.0, 0.0, 0.0])
    assert [x[0] for x in asyncio.run(ee.search_similar("q", 10))] == ["a"]
    ee._store_embedding("b", [1.0, 0.0, 0.0])  # add — must invalidate the cache
    assert sorted(x[0] for x in asyncio.run(ee.search_similar("q", 10))) == ["a", "b"]


def test_cache_invalidated_on_delete(tmp_path):
    ee = _engine(tmp_path)
    ee._store_embedding("a", [1.0, 0.0])
    ee._store_embedding("b", [0.0, 1.0])
    _stub_query(ee, [1.0, 0.0])
    assert len(asyncio.run(ee.search_similar("q", 10))) == 2
    ee.delete_embedding("b")  # remove — must invalidate the cache
    assert [x[0] for x in asyncio.run(ee.search_similar("q", 10))] == ["a"]


def test_cache_detects_external_process_write(tmp_path):
    # Another process (backfill/import) writing straight to the DB bypasses the
    # same-process cache clear — the (COUNT, MAX) signature must still rebuild.
    import sqlite3
    import json as _json
    from utils import now_iso

    ee = _engine(tmp_path)
    ee._store_embedding("a", [1.0, 0.0, 0.0])
    _stub_query(ee, [1.0, 0.0, 0.0])
    assert [x[0] for x in asyncio.run(ee.search_similar("q", 10))] == ["a"]  # warms cache

    conn = sqlite3.connect(ee.db_path)  # separate connection, ee._matrix_cache untouched
    conn.execute(
        "INSERT OR REPLACE INTO embeddings (bucket_id, embedding, updated_at, model) VALUES (?,?,?,?)",
        ("ext", _json.dumps([1.0, 0.0, 0.0]), now_iso(), ee.model),
    )
    conn.commit()
    conn.close()
    # count changed 1 -> 2, so the signature differs and the cache rebuilds
    assert sorted(x[0] for x in asyncio.run(ee.search_similar("q", 10))) == ["a", "ext"]


def test_cache_separates_filters(tmp_path):
    ee = _engine(tmp_path)
    ee._store_embedding("bucket1", [1.0, 0.0])
    ee._store_embedding("letter:1", [1.0, 0.0])
    _stub_query(ee, [1.0, 0.0])
    assert [x[0] for x in asyncio.run(ee.search_similar("q", 10))] == ["bucket1"]
    assert [x[0] for x in asyncio.run(ee.search_similar("q", 10, id_prefix="letter:"))] == ["letter:1"]
    # both filters cached under distinct keys; neither leaks into the other
    assert [x[0] for x in asyncio.run(ee.search_similar("q", 10))] == ["bucket1"]
    assert len(ee._matrix_cache) == 2
