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

from utils import NON_DECAYING_TYPES, is_decay_exempt, parse_bucket_ts
from bucket_history import ACTOR_DECAY
from bucket_manager import normalize_domains

logger = logging.getLogger("ombre_brain.decay")

# Fixed scores for the types that never decay. Membership lives in ONE place
# (utils.NON_DECAYING_TYPES) and is shared with calculate_heat; only the values
# are local, because score and heat answer different questions about the same
# exempt set.
# 不衰減桶型的固定分。成員資格只有一份（utils.NON_DECAYING_TYPES），與熱度共用；
# 只有「值」留在本地，因為分數和熱度對同一組豁免問的是不同的問題。
_EXEMPT_SCORES = {
    "permanent": 999.0,
    "feel": 50.0,
    "plan": 50.0,
    "mirage": 30.0,
}

# Fail at import, not inside calculate_score: run_decay_cycle swallows every
# exception from it, so a KeyError there would silently look like "skipped".
# 在載入時就炸，不要炸在 calculate_score 裡面：run_decay_cycle 會吞掉它拋的
# 所有例外，KeyError 在那裡看起來就只是「跳過」。
if set(_EXEMPT_SCORES) != set(NON_DECAYING_TYPES):
    raise RuntimeError(
        "decay exemption drift / 衰減豁免清單漂移: "
        f"NON_DECAYING_TYPES={NON_DECAYING_TYPES} vs _EXEMPT_SCORES={tuple(_EXEMPT_SCORES)}"
    )


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

        # --- Heat params (retrievability, a separate 0-1 quantity) ---
        # --- 熱度參數（可提取度，獨立的 0~1 量）---
        # halflife_base_days=7 is calibrated against THIS corpus, not copied:
        # kiwi-mem's 3-day half-life would push our median bucket (9.9 days
        # since last engaged) to heat 0.10 and mute ~90% of the pool. Ours is a
        # long-lived store, not a high-churn chat companion.
        # 7 天是照我們自己的語料校準的，不是抄來的：kiwi-mem 的 3 天半衰期
        # 會把我們的中位桶（距上次真正被用到 9.9 天）壓到 0.10，九成記憶消音。
        heat_cfg = decay_cfg.get("heat", {})
        self.heat_halflife_base = float(heat_cfg.get("halflife_base_days", 7.0))
        self.heat_high = float(heat_cfg.get("threshold_high", 0.7))
        self.heat_medium = float(heat_cfg.get("threshold_medium", 0.3))
        self.heat_truncate = int(heat_cfg.get("medium_truncate", 60))
        # Danger zone for review priority — cortexgraph's shipped defaults.
        # 0.25 is NOT a constant anywhere: it is the midpoint of these two, so
        # moving either moves the peak.
        # 危險區（複習優先度）——cortexgraph 實裝預設。0.25 不是常數，
        # 是這兩個的中點，改任一個峰值就跟著跑。
        self.danger_min = float(heat_cfg.get("danger_min", 0.15))
        self.danger_max = float(heat_cfg.get("danger_max", 0.35))

        # --- Per-domain metabolism factors (2026-07-16 batch-7 檔1) ---
        # Multiplies the heat HALF-LIFE only. calculate_score (archive triage)
        # is untouched: a fast-metabolizing bucket fades from surfacing sooner
        # but is archived on the same schedule — 「淡忘不刪除」 holds. Verified
        # before shipping: lost-tier buckets never occupy surfacing slots
        # (server.py's normal_pick filters tier != lost BEFORE slicing), so
        # accelerating heat cannot create tombstones that hog the breath.
        # --- per-domain 代謝係數（檔1）：只乘熱度「半衰期」。score（歸檔分流）
        # 不動——代謝快的桶更早淡出浮現，但歸檔節奏照舊。實作前已讀碼驗證：
        # lost 桶不佔浮現名額（normal_pick 先濾 lost 再切名額），
        # 加速 heat 不會製造霸名額的墓碑。---
        # Empty config = factor 1.0 everywhere = feature off. Unknown domains
        # fail OPEN to `default` (the taxonomy is LLM-assigned with no
        # whitelist, so an unrecognized domain must never be punished).
        # Non-positive factors are config nonsense and are dropped: a 0 here
        # would floor the half-life and hide memories — heat must fail vivid.
        # 空配置＝全 1.0＝功能關閉。未知域 fail-open 用 default（taxonomy 是
        # LLM 軟給的、無白名單，不認得的域絕不能被罰）。非正數係數直接丟棄：
        # 0 會把半衰期打到地板、把記憶藏起來——heat 壞掉必須偏鮮明。
        metab_cfg = decay_cfg.get("metabolism_factor", {}) or {}
        try:
            self.metabolism_default = float(metab_cfg.get("default", 1.0))
        except (ValueError, TypeError):
            self.metabolism_default = 1.0
        if self.metabolism_default <= 0:
            self.metabolism_default = 1.0
        self.metabolism_factors: dict[str, float] = {}
        for key, value in metab_cfg.items():
            if key == "default":
                continue
            try:
                factor = float(value)
            except (ValueError, TypeError):
                continue
            if factor <= 0:
                continue
            # Normalize the config side the same way bucket domains are
            # normalized at write time, so a simplified-Chinese key still lands.
            # config 端照桶寫入時的同一套正規化，簡體鍵也能對上。
            for norm in normalize_domains([str(key)]):
                self.metabolism_factors[norm] = factor
        try:
            self.metabolism_exempt_importance = int(
                decay_cfg.get("metabolism_exempt_importance", 8)
            )
        except (ValueError, TypeError):
            self.metabolism_exempt_importance = 8

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

    @staticmethod
    def _days_since_active(metadata: dict) -> float:
        """
        Days since this bucket was last ENGAGED with (touch), not merely shown
        (mark_surfaced). One parser for both score and heat — two would drift.
        距上次「真正被用到」（touch）的天數，不是「被看到」（mark_surfaced）。
        分數與熱度共用一個解析器——兩個遲早會不一致。

        Uses parse_bucket_ts, which normalizes tz-aware timestamps to local
        naive. The old inline parse fell into its 30-day fallback on any
        tz-aware value (datetime.now() is naive → TypeError), silently aging a
        fresh memory to a month old. Zero live buckets carry an offset today
        (verified 2026-07-15: 648/648 naive), so this changes no current score
        — it just closes the trapdoor for a hand-edited file.
        用 parse_bucket_ts（會把帶時區的正規化成本地 naive）。舊的行內解析
        碰到帶時區的值會掉進 30 天 fallback（datetime.now() 是 naive → TypeError），
        把新記憶靜默算成一個月大。今天 live 沒有任何桶帶時區（2026-07-15 驗過
        648/648 naive），所以這不改變任何現有分數，只是把手編輯的活門關上。
        """
        ts = parse_bucket_ts(metadata.get("last_active", metadata.get("created", "")))
        if ts is None:
            return 30.0
        return max(0.0, (datetime.now() - ts).total_seconds() / 86400)

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

        # --- Exempt buckets: never decay, fixed score by type ---
        # --- 豁免桶：不衰減，依型別給固定分 ---
        # Pinned/protected is checked FIRST and separately: a pinned bucket of
        # any type is locked at the top, so this order must stay ahead of the
        # type map (a pinned feel bucket scores 999, not 50).
        # 釘選/保護要先檢查且分開檢查：任何型別的釘選桶都鎖在頂端，
        # 所以這個順序必須排在型別表前面（釘選的 feel 桶是 999 不是 50）。
        # Values live here; membership lives in utils.NON_DECAYING_TYPES,
        # shared with calculate_heat so the two can never drift apart.
        # 值在這裡；成員資格在 utils.NON_DECAYING_TYPES，與熱度共用，永不分家。
        # feel/plan = 50 (sediment), mirage = 30 (a dream is residue, not
        # sediment), permanent = 999.
        if metadata.get("pinned") or metadata.get("protected"):
            return 999.0
        bucket_type = metadata.get("type")
        if bucket_type in NON_DECAYING_TYPES:
            return _EXEMPT_SCORES[bucket_type]

        importance = max(1, min(10, int(metadata.get("importance", 5))))
        activation_count = max(1.0, float(metadata.get("activation_count", 1)))

        # --- Days since last activation ---
        days_since = self._days_since_active(metadata)

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
    # Heat: how retrievable a memory still is. A SEPARATE quantity from score.
    # 熱度：這條記憶還有多想得起來。與 score 是「兩個」量，不是同一個。
    #
    # score answers "should this stay?"    → unbounded (live range 0.18–40.75),
    #                                        drives archive triage.
    # heat  answers "do I still recall it?" → a true 0-1 probability-shaped
    #                                        number, drives injection tier.
    # score 問「該不該留」→ 無界（live 實測 0.18–40.75），管歸檔分流。
    # heat  問「還記不記得清」→ 真的 0~1，管注入分層。
    #
    # These were conflated, which is exactly why there was no middle tier: the
    # only way to score low was resolved(×0.05)/digested(×0.2), and surfacing
    # excludes both — so "fading" and "can surface" were mutually exclusive and
    # the surfacing pool's floor sat at 1.09. Applying 0.7/0.3 to score would
    # have tiered 86% / 13% / ONE bucket, and the 13% were all resolved, i.e.
    # already excluded: a no-op.
    # 這兩件事被混在一起，正是「沒有中間格」的根因：低分的唯一來路是
    # resolved(×0.05)/digested(×0.2)，而浮現把兩者都排除——所以「正在淡掉」和
    # 「能被浮現」互斥，浮現池的地板卡在 1.09。0.7/0.3 套在 score 上會分成
    # 86%/13%/一個桶，而那 13% 全是 resolved（本來就被排除）＝空操作。
    #
    # Form is kiwi-mem's (2^(-age/half_life), main.py:868 + database.py:804-904);
    # the stability model is ours — kiwi-mem has only two half-lives (3d/7d),
    # we scale continuously. NOTE: the "retrievability" framing is Ombre's own
    # design, NOT inherited: cortexgraph's score is unbounded too (verified
    # 2026-07-15, decay.py:26-78) and has no retrievability concept.
    # 形式抄 kiwi-mem；穩定度模型是我們自己的（它只有 3天/7天 兩檔，我們連續）。
    # 注意：「可提取度」這個框架是 Ombre 自己的設計，不是繼承來的——
    # cortexgraph 的 score 一樣無界（2026-07-15 驗證），它沒有可提取度概念。
    #
    # calculate_score above is UNTOUCHED by this addition: arousal appearing
    # twice and the weight flip at 3 days are deliberate and correct.
    # 上面的 calculate_score 完全未被本次新增改動：arousal 出現兩次、
    # 三天處權重翻轉，都是刻意且正確的。
    # ---------------------------------------------------------
    def calculate_heat(self, metadata: dict) -> float:
        """
        Retrievability of a memory, 0.0–1.0. 1.0 = fully vivid, 0.25 = you'd
        fail to recall it three times in four.
        記憶的可提取度 0.0–1.0。1.0＝完全鮮明，0.25＝四次有三次想不起來。

        heat = 2 ^ (-days_since / H)
        H    = halflife_base × (importance/5) × (activation^0.3) × emotion_weight

        Exempt buckets return 1.0 — same exemption set as calculate_score.
        豁免桶回 1.0——與 calculate_score 同一組豁免。

        H additionally carries metabolism_factor(domains) — per-domain
        half-life scaling (2026-07-16 batch-7); see metabolism_factor below.
        H 另乘 per-domain 代謝係數（檔1），見下方 metabolism_factor。
        """
        if not isinstance(metadata, dict):
            return 0.0
        if is_decay_exempt(metadata):
            return 1.0

        days_since = self._days_since_active(metadata)

        importance = max(1, min(10, int(metadata.get("importance", 5) or 5)))
        try:
            activation = max(1.0, float(metadata.get("activation_count", 1) or 1))
        except (ValueError, TypeError):
            activation = 1.0
        try:
            arousal = max(0.0, min(1.0, float(metadata.get("arousal", 0.3))))
        except (ValueError, TypeError):
            arousal = 0.3

        # Same emotion coefficients as the score — one claim ("high arousal
        # makes memory durable"), so retuning it must move both.
        # 情感係數與分數共用——同一個主張（高喚醒讓記憶耐久），
        # 重新調校時兩邊必須一起動。
        emotion_weight = self.emotion_base + arousal * self.arousal_boost
        half_life = (
            self.heat_halflife_base
            * (importance / 5.0)
            * (activation ** 0.3)
            * emotion_weight
            * self.metabolism_factor(metadata)
        )
        return max(0.0, min(1.0, 2.0 ** (-days_since / max(0.1, half_life))))

    def metabolism_factor(self, metadata: dict) -> float:
        """
        Per-domain metabolism multiplier for the heat half-life, 2026-07-16
        batch-7 檔1. <1 = fades faster (engineering chatter), >1 = retained
        longer (relationship memory).
        per-domain 代謝係數（乘在熱度半衰期上）：<1 淡得快（工程流水），
        >1 留得久（關係記憶）。

        Rules / 規則:
        - Multi-domain takes the MAX factor — lean preservative. A dual-domain
          memory (「凌晨三點陪 debug」＝編程×戀愛) inherits its slowest
          metabolism; acceleration can only ever hit pure-engineering buckets.
          多域取最大＝偏保留：雙域記憶跟著最慢的代謝走，加速永遠只落在
          純工程桶上。
        - Unknown / empty domains fail OPEN to `default` (1.0).
          未知／未分類域 fail-open 用 default（1.0）。
        - importance >= metabolism_exempt_importance never accelerates: the
          factor floors at 1.0 (protective >1 factors are kept). Lessons and
          incidents (檔0 rule: importance >= 7-8) must not be fast-forgotten.
          importance 達豁免線的桶不加速：係數下限 1.0（保護性 >1 照舊）。
          教訓／事故不加速淡忘。
        - Any error returns 1.0 — the feature fails as if it weren't there,
          same contract as the surfacing heat helpers.
          任何錯誤回 1.0——壞掉時就當這功能不存在，與浮現側熱度輔助同一契約。
        """
        try:
            domains = metadata.get("domain", []) or []
            if domains:
                factor = max(
                    self.metabolism_factors.get(str(d), self.metabolism_default)
                    for d in domains
                )
            else:
                factor = self.metabolism_default
            if factor < 1.0:
                try:
                    importance = max(1, min(10, int(metadata.get("importance", 5) or 5)))
                except (ValueError, TypeError):
                    importance = 5
                if importance >= self.metabolism_exempt_importance:
                    factor = 1.0
            return factor
        except Exception:
            return 1.0

    def heat_tier(self, heat: float) -> str:
        """
        Injection tier for a heat value: vivid | faded | lost.
        注入分層：鮮明 | 模糊 | 已失去。

        Boundaries are STRICT > (copied exactly from kiwi-mem main.py:885-903):
        heat == 0.7 is 'faded', heat == 0.3 is 'lost'.
        邊界是嚴格 >（逐字照抄 kiwi-mem）：0.7 算模糊，0.3 算已失去。
        """
        if heat > self.heat_high:
            return "vivid"
        if heat > self.heat_medium:
            return "faded"
        return "lost"

    def review_priority(self, heat: float) -> float:
        """
        How badly a memory needs resurfacing before it is lost, 0.0–1.0.
        Peaks at the danger-zone midpoint (0.25 by default) and is hard-gated
        to 0.0 outside [danger_min, danger_max] — past the floor it is already
        gone, and digging it up is relearning, not review.
        一條記憶在消失前多需要被撈上來，0.0–1.0。峰值在危險區中點（預設 0.25），
        窗口外硬歸零——低於地板就是已經沒了，挖它是重學不是複習。

        Ported from cortexgraph review.py:15-58. The x fed to 1-x² is NOT the
        heat: it is heat re-centered on the midpoint and rescaled to [-1, 1].
        Feeding raw heat in would peak at 0 (i.e. on already-dead memories) —
        the exact inversion of the intent.
        移植自 cortexgraph review.py:15-58。餵進 1-x² 的 x 不是熱度本身，
        是「以中點為原點、重新縮放到 [-1,1]」後的偏移量。直接餵熱度會讓峰值
        落在 0（已經死掉的記憶）——與本意完全相反。
        """
        if heat < self.danger_min or heat > self.danger_max:
            return 0.0
        midpoint = (self.danger_min + self.danger_max) / 2.0
        range_half = (self.danger_max - self.danger_min) / 2.0
        if range_half <= 0:
            return 0.0
        normalized = (heat - midpoint) / range_half
        return max(0.0, min(1.0, 1.0 - normalized ** 2))

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

            # Skip permanent / pinned / protected / feel / plan / mirage buckets
            # 跳過固化桶、釘選/保護桶、feel、plan、mirage 桶
            # Was a local tuple — decay_engine used to carry two separate
            # representations of this same set (this tuple and calculate_score's
            # if-chain). Now both go through the one shared predicate.
            # 這裡本來是一個本地 tuple——decay_engine 自己曾有兩份同一組豁免的
            # 寫法（這個 tuple 和 calculate_score 的 if 串）。現在共用一個判斷。
            if is_decay_exempt(meta):
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
                        await self.bucket_mgr.update(bucket["id"], resolved=True, actor=ACTOR_DECAY)
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
                    success = await self.bucket_mgr.archive(bucket["id"], actor=ACTOR_DECAY)
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
                # Prefixed rows (letter:*) are letter vectors, not bucket orphans —
                # letters are permanent and live outside the bucket store.
                # letter: 前綴是信件向量，不是孤兒——信件永久保存且不在桶存儲裡。
                if orphan.startswith("letter:"):
                    continue
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
