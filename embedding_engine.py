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

import numpy as np
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
        # --- Provider: "gemini" (cloud, default) or "local" (fastembed ONNX).
        # The local path is a pre-built PARACHUTE: implemented, tested and the
        # model pre-downloaded, but dormant until config flips provider=local.
        # Vectors are namespaced per model (model column), so a switch never
        # mixes vector spaces and switching back costs nothing.
        # --- provider："gemini"（雲端，默認）或 "local"（fastembed ONNX）。
        # 本地路徑是預先摺好的降落傘：實作齊、測試過、模型已預下載，
        # 平時不載入；向量按 model 分空間存，切換不混、切回不重嵌。---
        self.provider = str(embed_cfg.get("provider", "gemini")).strip().lower()
        self.local_model_name = embed_cfg.get("local_model", "BAAI/bge-small-zh-v1.5")
        # Model cache lives OUTSIDE buckets_dir (backups must not swallow ~100MB
        # of model weights) and never depends on the service user having a home.
        # 模型快取放在 buckets 外（備份不吞模型權重），也不依賴服務帳號的 home。
        self.local_cache_dir = embed_cfg.get("local_cache_dir") or os.path.join(
            os.path.dirname(os.path.abspath(config["buckets_dir"])), ".fastembed-cache"
        )
        # bge zh v1.5 檢索建議：查詢端加指令前綴，文檔端不加
        self.local_query_prefix = embed_cfg.get(
            "local_query_prefix", "为这个句子生成表示以用于检索相关文章："
        )
        self._local_encoder = None  # lazy — only loaded when provider == "local"
        if self.provider == "local":
            self.model = self.local_model_name
        else:
            self.model = embed_cfg.get("model", "gemini-embedding-001")
        cloud_ready = bool(self.api_key)
        self.enabled = embed_cfg.get("enabled", True) and (
            self.provider == "local" or cloud_ready
        )
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

        # --- Initialize client (cloud provider only) ---
        if self.enabled and self.provider != "local":
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
        """Create embeddings table if not exists; migrate to (bucket_id, model) PK.

        Vector spaces are namespaced per model with a COMPOSITE primary key, so
        a provider switch writes into its own space and switching back costs
        nothing — the old vectors are still there. Legacy rows (no model column
        or bucket_id-only PK) are rebuilt and stamped as the cloud default.
        向量按 (bucket_id, model) 複合主鍵分空間並存：切換各寫各的、切回零成本。
        舊表（無 model 欄或單欄主鍵）重建遷移，舊列蓋雲端默認模型章。"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        # WAL 模式：大幅降低並發存取時 "database is locked" 機率（檔案層級設定，持久生效）
        conn.execute("PRAGMA journal_mode=WAL")
        info = list(conn.execute("PRAGMA table_info(embeddings)"))
        if not info:
            conn.execute("""
                CREATE TABLE embeddings (
                    bucket_id TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    model TEXT NOT NULL,
                    PRIMARY KEY (bucket_id, model)
                )
            """)
        else:
            cols = {row[1] for row in info}
            model_is_pk = any(row[1] == "model" and row[5] > 0 for row in info)
            if not model_is_pk:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS embeddings_v2 (
                        bucket_id TEXT NOT NULL,
                        embedding TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        model TEXT NOT NULL,
                        PRIMARY KEY (bucket_id, model)
                    )
                """)
                model_expr = "COALESCE(model, 'gemini-embedding-001')" if "model" in cols else "'gemini-embedding-001'"
                conn.execute(
                    f"INSERT OR REPLACE INTO embeddings_v2 "
                    f"SELECT bucket_id, embedding, updated_at, {model_expr} FROM embeddings"
                )
                conn.execute("DROP TABLE embeddings")
                conn.execute("ALTER TABLE embeddings_v2 RENAME TO embeddings")
        conn.commit()
        conn.close()

    # ---------------------------------------------------------
    # Local provider (fastembed ONNX) — the dormant parachute
    # 本地供應者（fastembed ONNX）——摺好的降落傘
    # ---------------------------------------------------------
    def _get_local_encoder(self):
        """Lazy-load the fastembed encoder (only ever called when provider=local)."""
        if self._local_encoder is None:
            from fastembed import TextEmbedding  # deferred: heavy import, dormant path
            os.makedirs(self.local_cache_dir, exist_ok=True)
            self._local_encoder = TextEmbedding(
                model_name=self.local_model_name, cache_dir=self.local_cache_dir
            )
            logger.info(f"Local embedding encoder loaded: {self.local_model_name}")
        return self._local_encoder

    async def _embed_local(self, text: str) -> list[float]:
        import asyncio
        def _encode():
            encoder = self._get_local_encoder()
            vectors = list(encoder.embed([text]))
            return [float(x) for x in vectors[0]] if vectors else []
        try:
            # ONNX inference is CPU-bound; keep the event loop free.
            return await asyncio.to_thread(_encode)
        except Exception as e:
            self.last_error = str(e)
            logger.warning(f"Local embedding failed: {e}")
            return []

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

    async def _generate_embedding(self, text: str, is_query: bool = False) -> list[float]:
        """Generate an embedding vector (LRU-cached per final text).
        is_query matters for the local bge models (query instruction prefix);
        the prefix is applied before caching so query/doc vectors never collide."""
        if self.provider == "local" and is_query and self.local_query_prefix:
            text = self.local_query_prefix + text
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
        if self.provider == "local":
            return await self._embed_local(truncated)
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
        """Store embedding in SQLite (stamped with the active model)."""
        from utils import now_iso
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO embeddings (bucket_id, embedding, updated_at, model) VALUES (?, ?, ?, ?)",
            (bucket_id, json.dumps(embedding), now_iso(), self.model),
        )
        conn.commit()
        conn.close()

    def delete_embedding(self, bucket_id: str):
        """Remove embedding when bucket is deleted (all models — the bucket is gone)."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM embeddings WHERE bucket_id = ?", (bucket_id,))
        conn.commit()
        conn.close()

    def list_ids(self) -> set[str]:
        """Bucket_ids with a stored embedding in the ACTIVE model space (for
        hygiene sweeps — after a provider switch every bucket looks missing,
        which is exactly what drives the automatic re-embed)."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT bucket_id FROM embeddings WHERE model = ?", (self.model,)
        ).fetchall()
        conn.close()
        return {r[0] for r in rows}

    async def get_embedding(self, bucket_id: str) -> list[float] | None:
        """Retrieve the active-model embedding for a bucket. None if not found."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT embedding FROM embeddings WHERE bucket_id = ? AND model = ?",
            (bucket_id, self.model),
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
            query_embedding = await self._generate_embedding(query, is_query=True)
            if not query_embedding:
                return []
        except Exception as e:
            logger.warning(f"Query embedding failed: {e}")
            return []

        # Load candidate embeddings from SQLite (active model space only)
        conn = sqlite3.connect(self.db_path)
        if id_prefix:
            rows = conn.execute(
                "SELECT bucket_id, embedding FROM embeddings WHERE bucket_id LIKE ? AND model = ?",
                (f"{id_prefix}%", self.model),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT bucket_id, embedding FROM embeddings WHERE bucket_id NOT LIKE 'letter:%' AND model = ?",
                (self.model,),
            ).fetchall()
        conn.close()

        if not rows:
            return []

        # Rank by cosine similarity, batched with numpy instead of a per-row
        # Python loop. Mathematically identical to calling _cosine_similarity
        # on every row — dot(row, q) / (‖row‖·‖q‖) — but computed as one matrix
        # op, so it stays fast as the corpus grows.
        # 用 numpy 一次矩陣運算取代逐行迴圈，數學等價，向量多時快得多。
        ids: list[str] = []
        vecs: list[np.ndarray] = []
        for bucket_id, emb_json in rows:
            try:
                raw = json.loads(emb_json)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue  # unparseable row — skip, exactly like the old loop
            if not isinstance(raw, list):
                continue  # non-vector JSON (null/number/object) — skip, like before
            try:
                vec = np.asarray(raw, dtype=np.float64)
            except (ValueError, TypeError):
                continue  # list with non-numeric contents — skip, like before
            ids.append(bucket_id)
            vecs.append(vec)

        if not ids:
            return []

        query_dim = len(query_embedding)
        q = np.asarray(query_embedding, dtype=np.float64)
        q_norm = float(np.linalg.norm(q))

        # Rows whose dimension matches the query go through the batched cosine;
        # any odd-length row scores 0.0 (mirrors _cosine_similarity's len guard),
        # so one stray legacy vector can never crash the whole search.
        sims_by_index: dict[int, float] = {}
        aligned_idx = [i for i, v in enumerate(vecs) if v.ndim == 1 and v.shape[0] == query_dim]
        if aligned_idx and q_norm > 0.0:
            mat = np.stack([vecs[i] for i in aligned_idx])       # (M, D)
            denom = np.linalg.norm(mat, axis=1) * q_norm         # (M,)
            dots = mat @ q                                        # (M,)
            sims = np.zeros_like(dots)
            nz = denom > 0.0
            sims[nz] = dots[nz] / denom[nz]                      # zero-norm rows stay 0.0
            for i, s in zip(aligned_idx, sims):
                sims_by_index[i] = float(s)

        results = [(ids[i], sims_by_index.get(i, 0.0)) for i in range(len(ids))]
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
