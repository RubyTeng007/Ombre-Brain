# ============================================================
# Module: Memory Decay Engine (decay_engine.py)
# 模塊：記憶衰減引擎
#
# Simulates human forgetting curve; auto-decays inactive memories and archives them.
# 模擬人類遺忘曲線，自動衰減不活躍記憶並歸檔。
#
# Core formula (improved Ebbinghaus + emotion coordinates):
# 核心公式（改進版艾賓浩斯遺忘曲線 + 情感座標）：
#   Score = Importance × (activation_count^0.3) × e^(-λ×days) × emotion_weight
#
# Emotion weight (continuous coordinate, not discrete labels):
# 情感權重（基於連續座標而非離散列舉）：
#   emotion_weight = base + (arousal × arousal_boost)
#   Higher arousal → higher emotion weight → slower decay
#   喚醒度越高 → 情感權重越大 → 記憶衰減越慢
#
# Depended on by: server.py
# 被誰依賴：server.py
# ============================================================

import math
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger("ombre_brain.decay")


class DecayEngine:
    """
    Memory decay engine — periodically scans all dynamic buckets,
    calculates decay scores, auto-archives low-activity buckets
    to simulate natural forgetting.
    記憶衰減引擎 —— 定期掃描所有動態桶，
    計算衰減得分，將低活躍桶自動歸檔，模擬自然遺忘。
    """

    def __init__(self, config: dict, bucket_mgr, embedding_engine=None):
        # --- Load decay parameters / 加載衰減參數 ---
        decay_cfg = config.get("decay", {})
        self.decay_lambda = decay_cfg.get("lambda", 0.05)
        self.threshold = decay_cfg.get("threshold", 0.3)
        self.check_interval = decay_cfg.get("check_interval_hours", 24)

        # --- Emotion weight params (continuous arousal coordinate) ---
        # --- 情感權重參數（基於連續 arousal 座標）---
        emotion_cfg = decay_cfg.get("emotion_weights", {})
        self.emotion_base = emotion_cfg.get("base", 1.0)
        self.arousal_boost = emotion_cfg.get("arousal_boost", 0.8)

        self.bucket_mgr = bucket_mgr
        # Optional: embedding engine for the daily vector-hygiene sweep
        # 可選：embedding 引擎，用於每日向量衛生（清孤兒 + 補缺）
        self.embedding_engine = embedding_engine

        # --- Background task control / 後臺任務控制 ---
        self._task: asyncio.Task | None = None
        self._running = False

    @property
    def is_running(self) -> bool:
        """Whether the decay engine is running in the background.
        衰減引擎是否正在後臺運行。"""
        return self._running

    # ---------------------------------------------------------
    # Core: calculate decay score for a single bucket
    # 核心：計算單個桶的衰減得分
    #
    # Higher score = more vivid memory; below threshold → archive
    # 得分越高 = 記憶越鮮活，低於閾值則歸檔
    # Permanent buckets never decay / 固化桶永遠不衰減
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Freshness bonus: continuous exponential decay
    # 新鮮度加成：連續指數衰減
    # bonus = 1.0 + 1.0 × e^(-t/36), t in hours
    # t=0 → 2.0×, t≈25h(半衰) → 1.5×, t≈72h → ≈1.14×, t→∞ → 1.0×
    # ---------------------------------------------------------
    @staticmethod
    def _calc_time_weight(days_since: float) -> float:
        """
        Freshness bonus multiplier: 1.0 + e^(-t/36), t in hours.
        新鮮度加成乘數：剛存入×2.0，~36小時半衰，72小時後趨近×1.0。
        """
        hours = days_since * 24.0
        return 1.0 + 1.0 * math.exp(-hours / 36.0)

    def calculate_score(self, metadata: dict) -> float:
        """
        Calculate current activity score for a memory bucket.
        計算一個記憶桶的當前活躍度得分。

        New model: short-term vs long-term weight separation.
        新模型：短期/長期權重分離。
        - Short-term (≤3 days): time_weight dominates, emotion amplifies
        - Long-term (>3 days): emotion_weight dominates, time decays to floor
        短期（≤3天）：時間權重主導，情感放大
        長期（>3天）：情感權重主導，時間衰減到底線
        """
        if not isinstance(metadata, dict):
            return 0.0

        # --- Pinned/protected buckets: never decay, importance locked to 10 ---
        if metadata.get("pinned") or metadata.get("protected"):
            return 999.0

        # --- Permanent buckets never decay ---
        if metadata.get("type") == "permanent":
            return 999.0

        # --- Feel buckets: never decay, fixed moderate score ---
        if metadata.get("type") == "feel":
            return 50.0

        importance = max(1, min(10, int(metadata.get("importance", 5))))
        activation_count = max(1.0, float(metadata.get("activation_count", 1)))

        # --- Days since last activation ---
        last_active_str = metadata.get("last_active", metadata.get("created", ""))
        try:
            last_active = datetime.fromisoformat(str(last_active_str))
            days_since = max(0.0, (datetime.now() - last_active).total_seconds() / 86400)
        except (ValueError, TypeError):
            days_since = 30

        # --- Emotion weight ---
        try:
            arousal = max(0.0, min(1.0, float(metadata.get("arousal", 0.3))))
        except (ValueError, TypeError):
            arousal = 0.3
        emotion_weight = self.emotion_base + arousal * self.arousal_boost

        # --- Time weight ---
        time_weight = self._calc_time_weight(days_since)

        # --- Short-term vs Long-term weight separation ---
        # 短期（≤3天）：time_weight 佔 70%，emotion 佔 30%
        # 長期（>3天）：emotion 佔 70%，time_weight 佔 30%
        if days_since <= 3.0:
            # Short-term: time dominates, emotion amplifies
            combined_weight = time_weight * 0.7 + emotion_weight * 0.3
        else:
            # Long-term: emotion dominates, time provides baseline
            combined_weight = emotion_weight * 0.7 + time_weight * 0.3

        # --- Base score ---
        base_score = (
            importance
            * (activation_count ** 0.3)
            * math.exp(-self.decay_lambda * days_since)
            * combined_weight
        )

        # --- Weight pool modifiers ---
        # resolved + digested (has feel) → accelerated fade: ×0.02
        # resolved only → ×0.05
        # digested only → ×0.2 (dream's feel-writing flow must sink the source
        # even when it wasn't explicitly resolved)
        # 已處理+已消化（寫過feel）→ 加速淡化：×0.02
        # 僅已處理 → ×0.05
        # 僅已消化 → ×0.2（dream 只寫 feel 不 resolve 時，源記憶也要能沉下去）
        resolved = metadata.get("resolved", False)
        digested = metadata.get("digested", False)  # set when feel is written for this memory
        if resolved and digested:
            resolved_factor = 0.02
        elif resolved:
            resolved_factor = 0.05
        elif digested:
            resolved_factor = 0.2
        else:
            resolved_factor = 1.0
        urgency_boost = 1.5 if (arousal > 0.7 and not resolved) else 1.0

        return round(base_score * resolved_factor * urgency_boost, 4)

    # ---------------------------------------------------------
    # Execute one decay cycle
    # 執行一輪衰減週期
    # Scan all dynamic buckets → score → archive those below threshold
    # 掃描所有動態桶 → 算分 → 低於閾值的歸檔
    # ---------------------------------------------------------
    async def run_decay_cycle(self) -> dict:
        """
        Execute one decay cycle: iterate dynamic buckets, archive those
        scoring below threshold.
        執行一輪衰減：遍歷動態桶，歸檔得分低於閾值的桶。

        Returns stats: {"checked": N, "archived": N, "lowest_score": X}
        """
        try:
            buckets = await self.bucket_mgr.list_all(include_archive=False)
        except Exception as e:
            logger.error(f"Failed to list buckets for decay / 衰減週期列桶失敗: {e}")
            return {"checked": 0, "archived": 0, "lowest_score": 0, "error": str(e)}

        checked = 0
        archived = 0
        auto_resolved = 0
        lowest_score = float("inf")

        for bucket in buckets:
            meta = bucket.get("metadata", {})

            # Skip permanent / pinned / protected / feel buckets
            # 跳過固化桶、釘選/保護桶和 feel 桶
            if meta.get("type") in ("permanent", "feel") or meta.get("pinned") or meta.get("protected"):
                continue

            checked += 1

            # --- Auto-resolve: imp≤4 + >30 days old + not resolved → auto resolve ---
            # --- 自動結案：重要度≤4 + 超過30天 + 未解決 → 自動 resolve ---
            if not meta.get("resolved", False):
                imp = int(meta.get("importance", 5))
                last_active_str = meta.get("last_active", meta.get("created", ""))
                try:
                    last_active = datetime.fromisoformat(str(last_active_str))
                    days_since = (datetime.now() - last_active).total_seconds() / 86400
                except (ValueError, TypeError):
                    days_since = 999
                if imp <= 4 and days_since > 30:
                    try:
                        await self.bucket_mgr.update(bucket["id"], resolved=True)
                        meta["resolved"] = True  # refresh local meta so resolved_factor applies this cycle
                        auto_resolved += 1
                        logger.info(
                            f"Auto-resolved / 自動結案: "
                            f"{meta.get('name', bucket['id'])} "
                            f"(imp={imp}, days={days_since:.0f})"
                        )
                    except Exception as e:
                        logger.warning(f"Auto-resolve failed / 自動結案失敗: {e}")

            try:
                score = self.calculate_score(meta)
            except Exception as e:
                logger.warning(
                    f"Score calculation failed for {bucket.get('id', '?')} / "
                    f"計算得分失敗: {e}"
                )
                continue

            lowest_score = min(lowest_score, score)

            # --- Below threshold → archive (simulate forgetting) ---
            # --- 低於閾值 → 歸檔（模擬遺忘）---
            if score < self.threshold:
                try:
                    success = await self.bucket_mgr.archive(bucket["id"])
                    if success:
                        archived += 1
                        logger.info(
                            f"Decay archived / 衰減歸檔: "
                            f"{meta.get('name', bucket['id'])} "
                            f"(score={score:.4f}, threshold={self.threshold})"
                        )
                except Exception as e:
                    logger.warning(
                        f"Archive failed for {bucket.get('id', '?')} / "
                        f"歸檔失敗: {e}"
                    )

        # --- Vector hygiene: drop orphan embeddings, backfill missing ones ---
        # --- 向量衛生：清理孤兒 embedding，補齊缺失的（每輪最多補20個）---
        hygiene = await self._vector_hygiene()

        result = {
            "checked": checked,
            "archived": archived,
            "auto_resolved": auto_resolved,
            "lowest_score": lowest_score if checked > 0 else 0,
            **hygiene,
        }
        logger.info(f"Decay cycle complete / 衰減週期完成: {result}")
        return result

    async def _vector_hygiene(self) -> dict:
        """
        Keep embeddings.db consistent with the bucket store:
        delete embeddings whose bucket no longer exists, and (re)generate
        embeddings for buckets that lost theirs (bounded per cycle).
        讓 embeddings.db 與桶存儲保持一致：刪掉桶已不存在的孤兒向量，
        為缺向量的桶補齊（每輪限量，防止配額雪崩）。
        """
        if not self.embedding_engine or not getattr(self.embedding_engine, "enabled", False):
            return {}
        try:
            all_buckets = await self.bucket_mgr.list_all(include_archive=True)
            bucket_ids = {b["id"] for b in all_buckets}
            emb_ids = self.embedding_engine.list_ids()

            orphans_removed = 0
            for orphan in emb_ids - bucket_ids:
                try:
                    self.embedding_engine.delete_embedding(orphan)
                    orphans_removed += 1
                except Exception:
                    continue

            backfilled = 0
            missing = [b for b in all_buckets if b["id"] not in emb_ids][:20]
            for b in missing:
                try:
                    if await self.embedding_engine.generate_and_store(b["id"], b["content"]):
                        backfilled += 1
                except Exception:
                    continue

            if orphans_removed or backfilled:
                logger.info(
                    f"Vector hygiene / 向量衛生: removed {orphans_removed} orphans, "
                    f"backfilled {backfilled} embeddings"
                )
            return {"emb_orphans_removed": orphans_removed, "emb_backfilled": backfilled}
        except Exception as e:
            logger.warning(f"Vector hygiene failed / 向量衛生失敗: {e}")
            return {}

    # ---------------------------------------------------------
    # Background decay task management
    # 後臺衰減任務管理
    # ---------------------------------------------------------
    async def ensure_started(self) -> None:
        """
        Ensure the decay engine is started (lazy init on first call).
        確保衰減引擎已啟動（懶加載，首次調用時啟動）。
        """
        if not self._running:
            await self.start()

    async def start(self) -> None:
        """Start the background decay loop.
        啟動後臺衰減循環。"""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._background_loop())
        logger.info(
            f"Decay engine started, interval: {self.check_interval}h / "
            f"衰減引擎已啟動，檢查間隔: {self.check_interval} 小時"
        )

    async def stop(self) -> None:
        """Stop the background decay loop.
        停止後臺衰減循環。"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Decay engine stopped / 衰減引擎已停止")

    async def _background_loop(self) -> None:
        """Background loop: run decay → sleep → repeat.
        後臺循環體：執行衰減 → 睡眠 → 重複。"""
        while self._running:
            try:
                await self.run_decay_cycle()
            except Exception as e:
                logger.error(f"Decay cycle error / 衰減週期出錯: {e}")
            # --- Wait for next cycle / 等待下一個週期 ---
            try:
                await asyncio.sleep(self.check_interval * 3600)
            except asyncio.CancelledError:
                break
