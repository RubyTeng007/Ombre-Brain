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
