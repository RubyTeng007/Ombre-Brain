# ============================================================
# Module: Embedding Engine (embedding_engine.py)
# 模塊：向量化引擎
#
# Generates embeddings via Gemini API (OpenAI-compatible),
# stores them in SQLite, and provides cosine similarity search.
# 通過 Gemini API（OpenAI 兼容）生成 embedding，
# 存儲在 SQLite 中，提供餘弦相似度搜索。
#
# Depended on by: server.py, bucket_manager.py
# 被誰依賴：server.py, bucket_manager.py
# ============================================================

import os
import json
import math
import sqlite3
import logging
from collections import OrderedDict

from openai import AsyncOpenAI

logger = logging.getLogger("ombre_brain.embedding")

# In-process LRU for text→vector: one breath(query=...) embeds the same query
# in the pre-filter (bucket_mgr.search) AND the vector channel; hold/grow embed
# the same new content in the semantic merge gate and again on store. Same text
# + same model = same vector, so caching kills those duplicate API calls.
# 同一次 breath 檢索會在預篩和向量通道各嵌一次同樣的查詢；快取直接省掉重複呼叫。
_TEXT_CACHE_MAXSIZE = 64


class EmbeddingEngine:
    """
    Embedding generation + SQLite vector storage + cosine search.
    向量生成 + SQLite 向量存儲 + 餘弦搜索。
    """

    def __init__(self, config: dict):
        dehy_cfg = config.get("dehydration", {})
        embed_cfg = config.get("embedding", {})

        self.api_key = (embed_cfg.get("api_key") or dehy_cfg.get("api_key") or "").strip()
        self.base_url = (
            (embed_cfg.get("base_url") or "").strip()
            or (dehy_cfg.get("base_url") or "").strip()
            or "https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.model = embed_cfg.get("model", "gemini-embedding-001")
        self.enabled = bool(self.api_key) and embed_cfg.get("enabled", True)
        self.last_error = ""
        self.last_success = False
        self._text_cache: "OrderedDict[str, list[float]]" = OrderedDict()

        # --- SQLite path: buckets_dir/embeddings.db ---
        db_path = os.path.join(config["buckets_dir"], "embeddings.db")
        self.db_path = db_path

        try:
            timeout_seconds = float(embed_cfg.get("timeout_seconds", 30.0))
        except (TypeError, ValueError):
            timeout_seconds = 30.0

        # --- Initialize client ---
        if self.enabled:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=timeout_seconds,
            )
        else:
            self.client = None

        # --- Initialize SQLite ---
        self._init_db()

    def _init_db(self):
        """Create embeddings table if not exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        # WAL 模式：大幅降低並發存取時 "database is locked" 機率（檔案層級設定，持久生效）
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                bucket_id TEXT PRIMARY KEY,
                embedding TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    async def generate_and_store(self, bucket_id: str, content: str) -> bool:
        """
        Generate embedding for content and store in SQLite.
        為內容生成 embedding 並存入 SQLite。
        Returns True on success, False on failure.
        """
        if not self.enabled or not content or not content.strip():
            self.last_success = False
            if not self.enabled:
                self.last_error = "embedding disabled or API key missing"
            return False

        try:
            embedding = await self._generate_embedding(content)
            if not embedding:
                self.last_success = False
                if not self.last_error:
                    self.last_error = "embedding API returned no vector"
                return False
            self._store_embedding(bucket_id, embedding)
            self.last_success = True
            self.last_error = ""
            return True
        except Exception as e:
            self.last_success = False
            self.last_error = str(e)
            logger.warning(f"Embedding generation failed for {bucket_id}: {e}")
            return False

    async def _generate_embedding(self, text: str) -> list[float]:
        """Call API to generate embedding vector (LRU-cached per text)."""
        # Truncate to avoid token limits
        truncated = text[:2000]
        cached = self._text_cache.get(truncated)
        if cached is not None:
            self._text_cache.move_to_end(truncated)
            return list(cached)
        embedding = await self._embed_uncached(truncated)
        if embedding:
            self._text_cache[truncated] = list(embedding)
            self._text_cache.move_to_end(truncated)
            while len(self._text_cache) > _TEXT_CACHE_MAXSIZE:
                self._text_cache.popitem(last=False)
        return embedding

    async def _embed_uncached(self, truncated: str) -> list[float]:
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=truncated,
            )
            if response.data and len(response.data) > 0:
                return response.data[0].embedding
            self.last_error = "embedding API returned no data"
            return []
        except Exception as e:
            self.last_error = str(e)
            logger.warning(f"Embedding API call failed: {e}")
            return []

    def _store_embedding(self, bucket_id: str, embedding: list[float]):
        """Store embedding in SQLite."""
        from utils import now_iso
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO embeddings (bucket_id, embedding, updated_at) VALUES (?, ?, ?)",
            (bucket_id, json.dumps(embedding), now_iso()),
        )
        conn.commit()
        conn.close()

    def delete_embedding(self, bucket_id: str):
        """Remove embedding when bucket is deleted."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM embeddings WHERE bucket_id = ?", (bucket_id,))
        conn.commit()
        conn.close()

    def list_ids(self) -> set[str]:
        """All bucket_ids that currently have a stored embedding (for hygiene sweeps)."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT bucket_id FROM embeddings").fetchall()
        conn.close()
        return {r[0] for r in rows}

    async def get_embedding(self, bucket_id: str) -> list[float] | None:
        """Retrieve stored embedding for a bucket. Returns None if not found."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT embedding FROM embeddings WHERE bucket_id = ?", (bucket_id,)
        ).fetchone()
        conn.close()
        if row:
            try:
                return json.loads(row[0])
            except json.JSONDecodeError:
                return None
        return None

    async def search_similar(
        self, query: str, top_k: int = 10, id_prefix: str | None = None
    ) -> list[tuple[str, float]]:
        """
        Search for buckets similar to query text.
        Returns list of (bucket_id, similarity_score) sorted by score desc.
        id_prefix=None searches bucket vectors only (non-bucket rows like
        "letter:*" are excluded); id_prefix="letter:" searches just that family.
        搜索與查詢文本相似的桶。默認只搜桶向量（排除 letter: 等前綴列），
        傳 id_prefix 則只搜該前綴家族。
        """
        if not self.enabled:
            return []

        try:
            query_embedding = await self._generate_embedding(query)
            if not query_embedding:
                return []
        except Exception as e:
            logger.warning(f"Query embedding failed: {e}")
            return []

        # Load candidate embeddings from SQLite
        conn = sqlite3.connect(self.db_path)
        if id_prefix:
            rows = conn.execute(
                "SELECT bucket_id, embedding FROM embeddings WHERE bucket_id LIKE ?",
                (f"{id_prefix}%",),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT bucket_id, embedding FROM embeddings WHERE bucket_id NOT LIKE 'letter:%'"
            ).fetchall()
        conn.close()

        if not rows:
            return []

        # Calculate cosine similarity
        results = []
        for bucket_id, emb_json in rows:
            try:
                stored_embedding = json.loads(emb_json)
                sim = self._cosine_similarity(query_embedding, stored_embedding)
                results.append((bucket_id, sim))
            except (json.JSONDecodeError, Exception):
                continue

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
