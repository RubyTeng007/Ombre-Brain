# ============================================================
# Module: Dehydration & Auto-tagging (dehydrator.py)
# 模塊：數據脫水壓縮 + 自動打標
#
# Capabilities:
# 能力：
#   1. Dehydrate: compress memory content into high-density summaries (save tokens)
#      脫水：將記憶桶的原始內容壓縮為高密度摘要，省 token
#   2. Merge: blend old and new content, keeping bucket size constant
#      合併：揉合新舊內容，控制桶體積恆定
#   3. Analyze: auto-analyze content for domain/emotion/tags
#      打標：自動分析內容，輸出主題域/情感座標/標籤
#
# Operating modes:
# 工作模式：
#   - API only: OpenAI-compatible API (DeepSeek/Ollama/LM Studio/vLLM/Gemini etc.)
#     僅 API：通過 OpenAI 兼容客戶端調用 LLM API
#   - Dehydration cache: SQLite persistent cache to avoid redundant API calls
#     脫水緩存：SQLite 持久緩存，避免重複調用 API
#
# Depended on by: server.py
# 被誰依賴：server.py
# ============================================================


import os
import re
import json
import hashlib
import sqlite3
import logging

from openai import AsyncOpenAI

from utils import clean_llm_json, count_tokens_approx

logger = logging.getLogger("ombre_brain.dehydrator")


# --- Dehydration prompt: instructs cheap LLM to compress information ---
# --- 脫水提示詞：指導廉價 LLM 壓縮信息 ---
# 提示詞版本。改了下面任何一個 PROMPT 就要往上加一號，否則快取會繼續服務用舊
# 提示詞產生的摘要——快取鍵含這個字串，加一號就等於讓所有舊列失效。
# 舊列不會被刪（沒有 TTL、沒有清理），會留在 DB 裡當孤兒佔空間；那是刻意的取捨：
# 寧可留下不會被讀到的位元組，也不要送出一份不知道是誰產的摘要。
PROMPT_VERSION = "1"

# 脫水輸入天花板。線上最長的桶約 3982 字，舊值 3000 已經在咬它們了（而且咬掉的是
# 待辦清單）。留餘裕但不無上限：真的超過就明確標記省略，不靜默切。
# 改這個值要一起把 PROMPT_VERSION 加一號——摘要的涵蓋範圍變了，舊快取不該再命中。
DEHYDRATE_INPUT_LIMIT = 8000

DEHYDRATE_PROMPT = """你是一個信息壓縮專家。請將以下內容脫水為緊湊摘要。

壓縮規則：
1. 提取所有核心事實，去除冗餘修飾和重複
2. 保留最新的情緒狀態和態度
3. 保留所有待辦/未完成事項
4. 關鍵數字、日期、名稱必須保留
5. 目標壓縮率 > 70%
6. 所有輸出一律使用繁體中文（台灣用字），即使原文是簡體也要轉為繁體

輸出格式（純 JSON，無其他內容）：
{
  "core_facts": ["事實1", "事實2"],
  "emotion_state": "當前情緒關鍵詞",
  "todos": ["待辦1", "待辦2"],
  "keywords": ["關鍵詞1", "關鍵詞2"],
  "summary": "50字以內的核心總結"
}"""


# --- Diary digest prompt: split daily notes into independent memory entries ---
# --- 日記整理提示詞：把一大段日常拆分成多個獨立記憶條目 ---
DIGEST_PROMPT = """你是一個日記整理專家。用戶會發送一段包含今天各種事情的文本（可能很雜亂），請你將其拆分成多個獨立的記憶條目。

整理規則：
1. 每個條目應該是一個獨立的主題/事件（不要混在一起）
2. 為每個條目自動分析元數據
3. 去除無意義的口水話和重複信息，保留核心內容
4. 同一主題的零散信息應合併為一個條目
5. 如果有待辦事項，單獨提取為一個條目
6. 單個條目內容不少於50字，過短的零碎信息合併到最相關的條目中
7. 總條目數控制在 2~6 個，避免過度碎片化
8. 在 content 中對人名、地名、專有名詞用 [[雙鏈]] 標記（如 [[婷易]]、[[Obsidian]]），普通詞彙不要加
9. 所有輸出（name/content/domain/tags）一律使用繁體中文（台灣用字），即使原文是簡體也要轉為繁體

輸出格式（純 JSON 數組，無其他內容）：
[
  {
    "name": "條目標題（10字以內）",
    "content": "整理後的內容",
    "domain": ["主題域1"],
    "valence": 0.7,
    "arousal": 0.4,
    "tags": ["核心詞1", "核心詞2", "擴展詞1", "擴展詞2"],
    "importance": 5
  }
]

tags 生成規則：先從原文精準提取 3~5 個核心詞，再引申擴展 5~8 個語義相關詞（近義詞、上位詞、關聯場景詞），合併為一個數組。

主題域可選（選最精確的 1~2 個，只選真正相關的）：
  日常: ["飲食", "穿搭", "出行", "居家", "購物"]
  人際: ["家庭", "戀愛", "友誼", "社交"]
  成長: ["工作", "學習", "考試", "求職"]
  身心: ["健康", "心理", "睡眠", "運動"]
  興趣: ["遊戲", "影視", "音樂", "閱讀", "創作", "手工"]
  數字: ["編程", "AI", "硬件", "網絡"]
  事務: ["財務", "計劃", "待辦"]
  內心: ["情緒", "回憶", "夢境", "自省"]
importance: 1-10，根據內容重要程度判斷
valence: 0~1（0=消極, 0.5=中性, 1=積極）
arousal: 0~1（0=平靜, 0.5=普通, 1=激動）"""


# --- Merge prompt: instruct LLM to blend old and new memories ---
# --- 合併提示詞：指導 LLM 揉合新舊記憶 ---
MERGE_PROMPT = """你是一個信息合併專家。請將舊記憶與新內容合併為一份統一的簡潔記錄。

合併規則：
1. 新內容與舊記憶衝突時，以新內容為準
2. 去除重複信息
3. 保留所有重要事實
4. 總長度儘量不超過舊記憶的 120%
5. 對出現的人名、地名、專有名詞用 [[雙鏈]] 標記（如 [[婷易]]、[[Obsidian]]），普通詞彙不要加
6. 輸出一律使用繁體中文（台灣用字），即使舊記憶或新內容是簡體也要轉為繁體

直接輸出合併後的文本，不要加額外說明。"""


# --- Auto-tagging prompt: analyze content for domain and emotion coords ---
# --- 自動打標提示詞：分析內容的主題域和情感座標 ---
ANALYZE_PROMPT = """你是一個內容分析器。請分析以下文本，輸出結構化的元數據。

分析規則：
1. domain（主題域）：選最精確的 1~2 個，只選真正相關的
   日常: ["飲食", "穿搭", "出行", "居家", "購物"]
   人際: ["家庭", "戀愛", "友誼", "社交"]
   成長: ["工作", "學習", "考試", "求職"]
   身心: ["健康", "心理", "睡眠", "運動"]
   興趣: ["遊戲", "影視", "音樂", "閱讀", "創作", "手工"]
   數字: ["編程", "AI", "硬件", "網絡"]
   事務: ["財務", "計劃", "待辦"]
   內心: ["情緒", "回憶", "夢境", "自省"]
2. valence（情感效價）：0.0~1.0，0=極度消極 → 0.5=中性 → 1.0=極度積極
3. arousal（情感喚醒度）：0.0~1.0，0=非常平靜 → 0.5=普通 → 1.0=非常激動
4. tags（關鍵詞標籤）：分兩步生成，合併為一個數組：
   第一步—精準提取：從原文抽取 3~5 個真正的核心詞，不泛化、不遺漏
   第二步—引申擴展：自動補充 8~10 個與當前場景語義相關的詞，包括近義詞、上位詞、關聯場景詞、用戶可能用不同措辭搜索的詞
   兩步合併為一個 tags 數組，總計 10~15 個
5. suggested_name（建議桶名）：10字以內的簡短標題
6. 在 tags 和 suggested_name 中不要使用 [[]] 雙鏈標記
7. 所有輸出（domain/tags/suggested_name）一律使用繁體中文（台灣用字），即使原文是簡體也要轉為繁體

輸出格式（純 JSON，無其他內容）：
{
  "domain": ["主題域1", "主題域2"],
  "valence": 0.7,
  "arousal": 0.4,
  "tags": ["核心詞1", "核心詞2", "擴展詞1", "擴展詞2", "..."],
  "suggested_name": "簡短標題"
}"""


class Dehydrator:
    """
    Data dehydrator + content analyzer.
    Three capabilities: dehydration / merge / auto-tagging (domain + emotion).
    API-only: every public method requires a working LLM API.
    If the API is unavailable, methods raise RuntimeError so callers can
    surface the failure to the user instead of silently producing low-quality results.
    數據脫水器 + 內容分析器。
    三大能力：脫水壓縮 / 新舊合併 / 自動打標。
    僅走 API：API 不可用時直接拋出 RuntimeError，調用方明確感知。
    （根據 BEHAVIOR_SPEC.md 三、降級行為表決策：無本地降級）
    """

    def __init__(self, config: dict):
        # --- Read dehydration API config / 讀取脫水 API 配置 ---
        dehy_cfg = config.get("dehydration", {})
        self.api_key = dehy_cfg.get("api_key", "")
        self.model = dehy_cfg.get("model", "deepseek-chat")
        self.base_url = dehy_cfg.get("base_url", "https://api.deepseek.com/v1")
        self.max_tokens = dehy_cfg.get("max_tokens", 2048)
        self.temperature = dehy_cfg.get("temperature", 0.1)
        try:
            timeout_seconds = float(dehy_cfg.get("timeout_seconds", 60.0))
        except (TypeError, ValueError):
            timeout_seconds = 60.0

        # --- API availability / 是否有可用的 API ---
        self.api_available = bool(self.api_key)

        # --- Initialize OpenAI-compatible client ---
        # --- 初始化 OpenAI 兼容客戶端 ---
        if self.api_available:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=timeout_seconds,
            )
        else:
            self.client = None

        # --- SQLite dehydration cache ---
        # --- SQLite 脫水緩存：content hash → summary ---
        db_path = os.path.join(config["buckets_dir"], "dehydration_cache.db")
        self.cache_db_path = db_path
        self._init_cache_db()

    def _init_cache_db(self):
        """Create dehydration cache table if not exists."""
        os.makedirs(os.path.dirname(self.cache_db_path), exist_ok=True)
        conn = sqlite3.connect(self.cache_db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dehydration_cache (
                content_hash TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                model TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
        conn.close()

    def _cache_key(self, content: str) -> str:
        """快取鍵＝正文＋模型＋提示詞版本。

        以前只雜湊正文，model 有存欄位但不在 WHERE 裡——所以換模型（server.py 會在
        執行期改 dehydrator.model）或改提示詞之後，同一份正文照樣命中舊那筆，
        舊模型的摘要永遠服務下去。而 invalidate_cache() 全 repo、線上都零呼叫，
        沒有 TTL、沒有版本欄位，唯一的失效途徑是「正文被改」——那只是讓舊列變孤兒，
        而孤兒也永遠不會被刪。
        """
        return hashlib.sha256(
            "\x00".join([content, str(self.model), PROMPT_VERSION]).encode()
        ).hexdigest()

    def _get_cached_summary(self, content: str) -> str | None:
        """Look up cached dehydration result by content hash."""
        conn = sqlite3.connect(self.cache_db_path)
        row = conn.execute(
            "SELECT summary FROM dehydration_cache WHERE content_hash = ?",
            (self._cache_key(content),)
        ).fetchone()
        conn.close()
        return row[0] if row else None

    def _set_cached_summary(self, content: str, summary: str):
        """Store dehydration result in cache."""
        conn = sqlite3.connect(self.cache_db_path)
        conn.execute(
            "INSERT OR REPLACE INTO dehydration_cache (content_hash, summary, model) VALUES (?, ?, ?)",
            (self._cache_key(content), summary, self.model)
        )
        conn.commit()
        conn.close()

    def invalidate_cache(self, content: str):
        """Remove cached summary for specific content (call when bucket content changes).

        注意：這個方法目前零呼叫（repo 與線上都是）。留著是因為它是對的，但別把它
        當成「快取會被清掉」的證據——真正在做失效的是 _cache_key 本身：鍵一變，
        舊列就再也命中不到。舊列不會被刪，那是已知的取捨（見 PROMPT_VERSION）。
        """
        conn = sqlite3.connect(self.cache_db_path)
        conn.execute("DELETE FROM dehydration_cache WHERE content_hash = ?",
                     (self._cache_key(content),))
        conn.commit()
        conn.close()

    # ---------------------------------------------------------
    # Dehydrate: compress raw content into concise summary
    # 脫水：將原始內容壓縮為精簡摘要
    # API only (no local fallback)
    # 僅通過 API 脫水（無本地回退）
    # ---------------------------------------------------------
    async def dehydrate(self, content: str, metadata: dict = None) -> str:
        """
        Dehydrate/compress memory content.
        Returns formatted summary string ready for Claude context injection.
        Uses SQLite cache to avoid redundant API calls.
        對記憶內容做脫水壓縮。
        返回格式化的摘要字符串，可直接注入 Claude 上下文。
        使用 SQLite 緩存避免重複調用 API。
        """
        if not content or not content.strip():
            return "（空記憶 / empty memory）"

        # --- Content is short enough, no compression needed ---
        # --- 內容已經很短，不需要壓縮 ---
        if count_tokens_approx(content) < 100:
            return self._format_output(content, metadata)

        # --- Check cache first ---
        # --- 先查緩存 ---
        cached = self._get_cached_summary(content)
        if cached:
            return self._format_output(cached, metadata)

        # --- API dehydration (no local fallback) ---
        # --- API 脫水（無本地降級）---
        if not self.api_available:
            raise RuntimeError("脫水 API 不可用，請配置 OMBRE_API_KEY")

        result, complete = await self._api_dehydrate(content)
        if not complete:
            # A truncated summary must NEVER reach the cache. Cached, it stops
            # being one bad call and becomes the memory's permanent face —
            # served on every breath, with no error anywhere to say so.
            # 截斷的摘要絕不能進快取。一旦進了，它就不再是「一次失敗的呼叫」，
            # 而變成這條記憶的永久臉孔——每次呼吸都供應，而且沒有任何錯誤會說。
            logger.error(
                "Dehydration TRUNCATED even after retry — not caching / "
                "脫水輸出重試後仍被截斷，不進快取: %s…",
                content[:60],
            )
            # Say it in the output, same rule as the blur label: never hand over
            # a fragment and let the reader believe it is the whole thing.
            # 在輸出裡說出來，跟「（印象模糊）」同一條規矩：
            # 絕不遞出一個殘片、讓讀的人以為那就是全部。
            result = result.rstrip() + "\n（⚠ 這份摘要被壓縮器的輸出上限截斷了，不完整、也沒有進快取。要全文用 breath(query=…) 去問。）"
            return self._format_output(result, metadata)
        # --- Cache the result ---
        self._set_cached_summary(content, result)
        return self._format_output(result, metadata)

    # ---------------------------------------------------------
    # Merge: blend new content into existing bucket
    # 合併：將新內容揉入已有桶，保持體積恆定
    # ---------------------------------------------------------
    async def merge(self, old_content: str, new_content: str) -> str:
        """
        Merge new content with old memory, preventing infinite bucket growth.
        將新內容與舊記憶合併，避免桶無限膨脹。
        """
        if not old_content and not new_content:
            return ""
        if not old_content:
            return new_content or ""
        if not new_content:
            return old_content

        # --- API merge (no local fallback) ---
        if not self.api_available:
            raise RuntimeError("脫水 API 不可用，請檢查 config.yaml 中的 dehydration 配置")
        try:
            result, complete = await self._api_merge(old_content, new_content)
            if not complete:
                # A truncated merge is worse than a truncated summary: the
                # summary poisons a cache, this overwrites the bucket's own
                # body — the source of truth — with an amputated version.
                # Refusing sends _merge_or_create down its existing fail-closed
                # path, which creates a new bucket instead. That is the right
                # trade and the code already says so: 重複可救，誤併不可逆.
                # 截斷的合併比截斷的摘要嚴重：摘要毒的是快取，這個是拿殘肢
                # 覆蓋桶的正文本身——真相來源。拒絕之後 _merge_or_create 會走
                # 它既有的 fail-closed 路徑改為新建一個桶。那是對的取捨，
                # 而且程式碼自己早就寫了：重複可救，誤併不可逆。
                raise RuntimeError(
                    "API 合併輸出被 max_tokens 截斷，拒絕寫入（重複可救，誤併不可逆）"
                )
            if result:
                return result
            raise RuntimeError("API 合併返回空結果")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"API 合併失敗，請檢查 API 連接: {e}") from e

    # ---------------------------------------------------------
    # API call: dehydration
    # API 調用：脫水壓縮
    # ---------------------------------------------------------
    async def _complete(self, system_prompt: str, user_msg: str) -> tuple[str, bool]:
        """
        One completion, with the output-cap trapdoor closed.
        Returns (text, complete). complete=False means the model hit max_tokens
        and the text is cut off mid-token — for JSON output that is a broken
        blob, for a merge it is silently amputated memory.
        回傳 (文字, 是否完整)。complete=False 代表模型撞到 max_tokens、
        字串被切在半路——對 JSON 輸出來說那是壞掉的 blob，對合併來說那是
        被無聲截肢的記憶。

        We never read finish_reason before. That is how 畫像我們的模樣 ended up
        cached with its JSON severed at `"下一本共讀書待定", "` — the 1024-token
        cap cut the model mid-string, and the broken text was written into
        dehydration_cache.db and served on every single breath from then on.
        The provider was telling us the whole time; nobody was listening.
        我們從來沒讀過 finish_reason。這就是「畫像我們的模樣」的 JSON 被切在
        `"下一本共讀書待定", "` 還進了快取、之後每一次呼吸都供應同一份殘缺的
        原因——1024 上限把模型切在字串中間。供應商一直在講，只是沒人在聽。

        Retry once with double room: hitting the cap means the model needed more
        space, and asking again with the same budget would just fail identically.
        撞到上限就是模型需要更多空間，用同樣的額度再問一次只會一模一樣地失敗。
        """
        budget = self.max_tokens
        text = ""
        for _ in (1, 2):
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=budget,
                temperature=self.temperature,
            )
            if not response.choices:
                return "", False
            choice = response.choices[0]
            text = choice.message.content or ""
            if getattr(choice, "finish_reason", None) != "length":
                return text, True
            logger.warning(
                f"LLM output hit the {budget}-token cap, retrying with double room / "
                f"輸出撞到 {budget} token 上限，加倍重試"
            )
            budget *= 2
        return text, False

    async def _api_dehydrate(self, content: str) -> tuple[str, bool]:
        """
        Call LLM API for intelligent dehydration (via OpenAI-compatible client).
        調用 LLM API 執行智能脫水。

        輸入截斷不再是無聲的。3000 字的天花板本來就在，只是沒有人知道它咬到了誰：
        線上有三個桶超過它，摘要早就是拿截斷版產的、天天在服務中，而其中一條被吃掉
        的尾巴正好是待辦清單——DEHYDRATE_PROMPT 第 3 條寫的是「保留所有待辦/未完成
        事項」。天花板提高到能蓋住現有的桶，超過的話明確告訴模型「這裡被切了」，
        並且記一筆 log 讓它變成看得見的事實而不是猜測。
        """
        if len(content) > DEHYDRATE_INPUT_LIMIT:
            logger.warning(
                f"Dehydration input truncated at {DEHYDRATE_INPUT_LIMIT} chars "
                f"(body is {len(content)}) — the tail is not in this summary / "
                f"脫水輸入被截斷，尾段不在這份摘要裡"
            )
            body = (
                content[:DEHYDRATE_INPUT_LIMIT]
                + f"\n\n〔以下省略 {len(content) - DEHYDRATE_INPUT_LIMIT} 字，"
                  f"摘要只涵蓋前面這段〕"
            )
        else:
            body = content
        return await self._complete(DEHYDRATE_PROMPT, body)

    # ---------------------------------------------------------
    # API call: merge
    # API 調用：合併
    # ---------------------------------------------------------
    async def _api_merge(self, old_content: str, new_content: str) -> tuple[str, bool]:
        """
        Call LLM API for intelligent merge (via OpenAI-compatible client).
        調用 LLM API 執行智能合併。
        """
        user_msg = f"舊記憶：\n{old_content[:2000]}\n\n新內容：\n{new_content[:2000]}"
        return await self._complete(MERGE_PROMPT, user_msg)



    # ---------------------------------------------------------
    # Output formatting
    # 輸出格式化
    # Wraps dehydrated result with bucket name, tags, emotion coords
    # 把脫水結果包裝成帶桶名、標籤、情感座標的可讀文本
    # ---------------------------------------------------------
    def _format_output(self, content: str, metadata: dict = None) -> str:
        """
        Format dehydrated result into context-injectable text.
        將脫水結果格式化為可注入上下文的文本。
        """
        header = ""
        if metadata and isinstance(metadata, dict):
            name = metadata.get("name", "未命名")
            domains = ", ".join(metadata.get("domain", []))
            try:
                valence = float(metadata.get("valence", 0.5))
                arousal = float(metadata.get("arousal", 0.3))
            except (ValueError, TypeError):
                valence, arousal = 0.5, 0.3
            header = f"📌 記憶桶: {name}"
            if domains:
                header += f" [主題:{domains}]"
            header += f" [情感:V{valence:.1f}/A{arousal:.1f}]"
            # Show model's perspective if available (valence drift)
            model_v = metadata.get("model_valence")
            if model_v is not None:
                try:
                    header += f" [我的視角:V{float(model_v):.1f}]"
                except (ValueError, TypeError):
                    pass
            if metadata.get("digested"):
                header += " [已消化]"
            header += "\n"
        
        content = re.sub(r'\[\[([^\]]+)\]\]', r'\1', content)
        return f"{header}{content}"

    # ---------------------------------------------------------
    # Auto-tagging: analyze content for domain + emotion + tags
    # 自動打標：分析內容，輸出主題域 + 情感座標 + 標籤
    # Called by server.py when storing new memories
    # 存新記憶時由 server.py 調用
    # ---------------------------------------------------------
    async def analyze(self, content: str) -> dict:
        """
        Analyze content and return structured metadata.
        分析內容，返回結構化元數據。

        Returns: {"domain", "valence", "arousal", "tags", "suggested_name"}
        """
        if not content or not content.strip():
            return self._default_analysis()

        # --- API analyze (no local fallback) ---
        if not self.api_available:
            raise RuntimeError("脫水 API 不可用，請檢查 config.yaml 中的 dehydration 配置")
        try:
            result = await self._api_analyze(content)
            if result:
                return result
            raise RuntimeError("API 打標返回空結果")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"API 打標失敗，請檢查 API 連接: {e}") from e

    # ---------------------------------------------------------
    # API call: auto-tagging
    # API 調用：自動打標
    # ---------------------------------------------------------
    async def _api_analyze(self, content: str) -> dict:
        """
        Call LLM API for content analysis / tagging.
        調用 LLM API 執行內容分析打標。
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": ANALYZE_PROMPT},
                {"role": "user", "content": content[:2000]},
            ],
            max_tokens=512,
            temperature=0.1,
        )
        if not response.choices:
            return self._default_analysis()
        raw = response.choices[0].message.content or ""
        if not raw.strip():
            return self._default_analysis()
        return self._parse_analysis(raw)

    # ---------------------------------------------------------
    # Parse API JSON response with safety checks
    # 解析 API 返回的 JSON，做安全校驗
    # Ensure valence/arousal in 0~1, domain/tags valid
    # ---------------------------------------------------------
    def _parse_analysis(self, raw: str) -> dict:
        """
        Parse and validate API tagging result.
        解析並校驗 API 返回的打標結果。
        """
        try:
            # Tolerate code fences and chatter around the JSON payload
            # 容忍 code fence 與 JSON 前後的說明文字
            result = json.loads(clean_llm_json(raw))
        except (json.JSONDecodeError, ValueError):
            logger.warning(f"API tagging JSON parse failed / JSON 解析失敗: {raw[:200]}")
            return self._default_analysis()

        if not isinstance(result, dict):
            return self._default_analysis()

        # --- Validate and clamp value ranges / 校驗並鉗制數值範圍 ---
        try:
            valence = max(0.0, min(1.0, float(result.get("valence", 0.5))))
            arousal = max(0.0, min(1.0, float(result.get("arousal", 0.3))))
        except (ValueError, TypeError):
            valence, arousal = 0.5, 0.3

        return {
            "domain": result.get("domain", ["未分類"])[:3],
            "valence": valence,
            "arousal": arousal,
            "tags": result.get("tags", [])[:15],
            "suggested_name": str(result.get("suggested_name", ""))[:20],
        }

    # ---------------------------------------------------------
    # Default analysis result (empty content or total failure)
    # 默認分析結果（內容為空或完全失敗時用）
    # ---------------------------------------------------------
    def _default_analysis(self) -> dict:
        """
        Return default neutral analysis result.
        返回默認的中性分析結果。
        """
        return {
            "domain": ["未分類"],
            "valence": 0.5,
            "arousal": 0.3,
            "tags": [],
            "suggested_name": "",
        }

    # ---------------------------------------------------------
    # Diary digest: split daily notes into independent memory entries
    # 日記整理：把一大段日常拆分成多個獨立記憶條目
    # For the "grow" tool — "dump a day's content and it gets organized"
    # 給 grow 工具用，"一天結束髮一坨內容"靠這個
    # ---------------------------------------------------------
    async def digest(self, content: str) -> list[dict]:
        """
        Split a large chunk of daily content into independent memory entries.
        將一大段日常內容拆分成多個獨立記憶條目。

        Returns: [{"name", "content", "domain", "valence", "arousal", "tags", "importance"}, ...]
        """
        if not content or not content.strip():
            return []

        # --- API digest (no local fallback) ---
        if not self.api_available:
            raise RuntimeError("脫水 API 不可用，請檢查 config.yaml 中的 dehydration 配置")
        try:
            result = await self._api_digest(content)
            if result:
                return result
            raise RuntimeError("API 日記整理返回空結果")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"API 日記整理失敗，請檢查 API 連接: {e}") from e

    # ---------------------------------------------------------
    # API call: diary digest
    # API 調用：日記整理
    # ---------------------------------------------------------
    async def _api_digest(self, content: str) -> list[dict]:
        """
        Call LLM API for diary organization.
        調用 LLM API 執行日記整理。
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": DIGEST_PROMPT},
                {"role": "user", "content": content[:5000]},
            ],
            max_tokens=4096,
            temperature=0.0,
        )
        if not response.choices:
            return []
        raw = response.choices[0].message.content or ""
        if not raw.strip():
            return []
        return self._parse_digest(raw)

    # ---------------------------------------------------------
    # Parse diary digest result with safety checks
    # 解析日記整理結果，做安全校驗
    # ---------------------------------------------------------
    def _parse_digest(self, raw: str) -> list[dict]:
        """
        Parse and validate API diary digest result.
        解析並校驗 API 返回的日記整理結果。
        """
        try:
            items = json.loads(clean_llm_json(raw))
        except (json.JSONDecodeError, ValueError):
            logger.warning(f"Diary digest JSON parse failed / JSON 解析失敗: {raw[:200]}")
            return []

        if not isinstance(items, list):
            return []

        validated = []
        for item in items:
            if not isinstance(item, dict) or not item.get("content"):
                continue
            try:
                importance = max(1, min(10, int(item.get("importance", 5))))
            except (ValueError, TypeError):
                importance = 5
            try:
                valence = max(0.0, min(1.0, float(item.get("valence", 0.5))))
                arousal = max(0.0, min(1.0, float(item.get("arousal", 0.3))))
            except (ValueError, TypeError):
                valence, arousal = 0.5, 0.3

            validated.append({
                "name": str(item.get("name", ""))[:20],
                "content": str(item.get("content", "")),
                "domain": item.get("domain", ["未分類"])[:3],
                "valence": valence,
                "arousal": arousal,
                "tags": item.get("tags", [])[:15],
                "importance": importance,
            })
        return validated
