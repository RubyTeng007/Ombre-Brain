# ============================================================
# Module: Memory Bucket Manager (bucket_manager.py)
# 模塊：記憶桶管理器
#
# CRUD operations, multi-dimensional index search, activation updates
# for memory buckets.
# 記憶桶的增刪改查、多維索引搜索、激活更新。
#
# Core design:
# 核心邏輯：
#   - Each bucket = one Markdown file (YAML frontmatter + body)
#     每個記憶桶 = 一個 Markdown 文件
#   - Storage by type: permanent / dynamic / archive
#     存儲按類型分目錄
#   - Multi-dimensional soft index: domain + valence/arousal + fuzzy text
#     多維軟索引：主題域 + 情感座標 + 文本模糊匹配
#   - Search strategy: domain pre-filter → weighted multi-dim ranking
#     搜索策略：主題域預篩 → 多維加權精排
#   - Emotion coordinates based on Russell circumplex model:
#     情感座標基於環形情感模型（Russell circumplex）：
#       valence (0~1): 0=negative → 1=positive
#       arousal (0~1): 0=calm → 1=excited
#
# Depended on by: server.py, decay_engine.py
# 被誰依賴：server.py, decay_engine.py
# ============================================================

import os
import json
import math
import random
import secrets
import logging
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from typing import NamedTuple

import frontmatter
from rapidfuzz import fuzz


class DeleteResult(NamedTuple):
    """delete() 的結果。刻意不是 bool：「檔案刪掉了嗎」和「救得回來嗎」是兩件事，
    混成一個真假值正是 restore 指令會說謊的原因。ok=False 代表桶還在。
    seq=None 代表刪掉了但沒存下快照——不可復原，別對使用者宣稱可以。"""
    ok: bool
    seq: Optional[int]

from utils import generate_bucket_id, sanitize_name, safe_path, now_iso, parse_bucket_ts

# --- Domain normalization: canonical form is Traditional Chinese (Ruby's call) ---
# --- 主題域正規化：一律繁體（Ruby 拍板：全部繁體）。---
# Prefer OpenCC at runtime (catches any simplified domain the LLM emits);
# the static map is the no-dependency fallback for historically seen domains.
# 執行期優先用 OpenCC（LLM 吐什麼簡體都接得住）；靜態表是無依賴保底。
try:
    from opencc import OpenCC as _OpenCC
    _S2TW = _OpenCC("s2tw")
except Exception:
    _S2TW = None

DOMAIN_NORMALIZE = {
    "编程": "編程",
    "恋爱": "戀愛",
    "数字": "數字",
    "计划": "計劃",
    "事务": "事務",
    "人际": "人際",
    "兴趣": "興趣",
    "内心": "內心",
    "创作": "創作",
    "友谊": "友誼",
    "影视": "影視",
    "待办": "待辦",
    "情绪": "情緒",
    "成长": "成長",
    "游戏": "遊戲",
    "阅读": "閱讀",
    "音乐": "音樂",
    "饮食": "飲食",
    "网络": "網絡",
    "未分类": "未分類",
    "沉淀物": "沉澱物",
    "学习": "學習",
    "购物": "購物",
    "运动": "運動",
    "梦境": "夢境",
    "回忆": "回憶",
    "财务": "財務",
    "健康": "健康",
}

# --- Activation cap: bounds time-ripple / touch inflation ---
# --- 激活次數上限：防止 time_ripple/touch 無限通脹扭曲權重 ---
ACTIVATION_CAP = 50


def normalize_domains(domains: list[str]) -> list[str]:
    """Map any simplified-variant domain onto Traditional Chinese, dedup preserving order."""
    out = []
    for d in domains or []:
        if _S2TW is not None:
            try:
                c = _S2TW.convert(d)
            except Exception:
                c = DOMAIN_NORMALIZE.get(d, d)
        else:
            c = DOMAIN_NORMALIZE.get(d, d)
        if c not in out:
            out.append(c)
    return out


# Plan bucket lifecycle states + whitelisted extra metadata keys.
# plan 桶的生命週期狀態 + 附加元數據白名單。
_PLAN_STATUSES = ("active", "resolved", "abandoned")
_EXTRA_META_KEYS = (
    "status", "weight", "related_bucket", "why_remembered",
    # plan schema (2026-07-12): the fixation wiring reads target_drive
    # directly — never the domain→drive map.
    # plan schema（2026-07-12）：執念接線直接讀 target_drive，不走 domain 映射。
    "kind", "target_drive", "due_at", "progress",
    # dream provenance (2026-07-12 batch-2): which buckets the dream consumed.
    # dream 出處（第二批）：這個夢消化了哪些桶。
    "consumed",
    # verbatim guard (2026-07-19 audit F11b): bucket holds final-text content —
    # merges must append, never LLM-rewrite.
    # 逐字保護（audit F11b）：桶裡住著定稿——合併只准 append，不准 LLM 改寫。
    "verbatim_guard",
)
_FLOAT_META_KEYS = ("weight", "progress")


def _normalize_meta_datetimes(metadata: dict) -> dict:
    """
    Buckets written by this system quote their timestamps, but a hand-edited
    file (e.g. via Obsidian) can leave them unquoted and YAML parses those into
    datetime/date objects — which then break string sorts and JSON responses.
    Normalize every datetime-ish value back to an ISO string at the read layer.
    手編輯的桶檔若時間戳沒加引號，YAML 會解析成 datetime 物件，
    排序與 JSON 序列化都會炸。在讀取層統一轉回 ISO 字串。
    """
    for key, value in metadata.items():
        if isinstance(value, datetime):
            metadata[key] = value.isoformat()
        elif isinstance(value, date):
            metadata[key] = value.isoformat()
    return metadata


def atomic_write_text(file_path: str, text: str) -> None:
    """Write file via tmp + fsync + rename so a crash never leaves a half-written bucket.
    與 letters/self_concept 相同的原子寫入模式；桶是最重要的存儲，不該比它們脆。"""
    tmp = f"{file_path}.{secrets.token_hex(4)}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, file_path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

logger = logging.getLogger("ombre_brain.bucket")


class BucketManager:
    """
    Memory bucket manager — entry point for all bucket CRUD operations.
    Buckets are stored as Markdown files with YAML frontmatter for metadata
    and body for content. Natively compatible with Obsidian browsing/editing.
    記憶桶管理器 —— 所有桶的 CRUD 操作入口。
    桶以 Markdown 文件存儲，YAML frontmatter 存元數據，正文存內容。
    天然兼容 Obsidian 直接瀏覽和編輯。
    """

    def __init__(self, config: dict, embedding_engine=None, history=None):
        # --- Optional bucket_history.BucketHistory: snapshots every
        # destructive write so it can be undone. None = no history recorded
        # (tests, imports). Injected like embedding_engine.
        # --- 可選的歷史表：每次破壞性寫入前存快照，讓它可以被還原。
        # None = 不記歷史（測試、匯入腳本）。比照 embedding_engine 注入。---
        self.history = history

        # --- Read storage paths from config / 從配置中讀取存儲路徑 ---
        self.base_dir = config["buckets_dir"]
        self.permanent_dir = os.path.join(self.base_dir, "permanent")
        self.dynamic_dir = os.path.join(self.base_dir, "dynamic")
        self.archive_dir = os.path.join(self.base_dir, "archive")
        self.feel_dir = os.path.join(self.base_dir, "feel")
        self.mirage_dir = os.path.join(self.base_dir, "mirage")
        self.plan_dir = os.path.join(self.base_dir, "plan")
        self.fuzzy_threshold = config.get("matching", {}).get("fuzzy_threshold", 50)
        self.max_results = config.get("matching", {}).get("max_results", 5)
        # --- BM25 keyword channel: pre-built but dormant (matching.bm25_enabled).
        # Flip trigger: corpus outgrows rapidfuzz (~2000 buckets) or real recall
        # misses. When on, BM25 hits join the candidate set and blend into the
        # topic score — recall insurance, not a ranking rewrite.
        # --- BM25 關鍵詞通道：預建休眠。開了之後 BM25 命中補進候選集並融入
        # topic 分——是召回保險，不是重寫排序。---
        self.bm25_enabled = bool(config.get("matching", {}).get("bm25_enabled", False))
        # --- Context gate (2026-07-12 batch-2): neutral-context queries damp
        # weak-hit high-arousal intimate buckets out of the ranking.
        # --- 情境門控：中性語境查詢把弱命中的高喚醒親密桶擋在榜外。---
        _m = config.get("matching", {})
        self.context_gate_enabled = bool(_m.get("context_gate_enabled", True))
        self.context_gate_arousal = float(_m.get("context_gate_arousal", 0.75))
        self.context_gate_damp = float(_m.get("context_gate_damp", 0.5))
        self.context_gate_domains = set(_m.get("context_gate_domains", ["戀愛"]))
        # Vector-channel admissibility (2026-07-12 batch-3): the semantic
        # supplement channel needs a STRONG hit to carry a gated bucket into
        # a neutral-context result — 0.5-grade grazes were a side door.
        # 向量通道入場線：門控類桶要進中性語境結果，語義命中必須夠強。
        self.context_gate_vector_sim = float(_m.get("context_gate_vector_sim", 0.75))
        self._bm25 = None

        # --- Wikilink config / 雙鏈配置 ---
        wikilink_cfg = config.get("wikilink", {})
        self.wikilink_enabled = wikilink_cfg.get("enabled", True)
        self.wikilink_use_tags = wikilink_cfg.get("use_tags", False)
        self.wikilink_use_domain = wikilink_cfg.get("use_domain", True)
        self.wikilink_use_auto_keywords = wikilink_cfg.get("use_auto_keywords", True)
        self.wikilink_auto_top_k = wikilink_cfg.get("auto_top_k", 8)
        self.wikilink_min_len = wikilink_cfg.get("min_keyword_len", 2)
        self.wikilink_exclude_keywords = set(wikilink_cfg.get("exclude_keywords", []))
        self.wikilink_stopwords = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
            "都", "一個", "上", "也", "很", "到", "說", "要", "去",
            "你", "會", "著", "沒有", "看", "好", "自己", "這", "他", "她",
            "我們", "你們", "他們", "然後", "今天", "昨天", "明天", "一下",
            "the", "and", "for", "are", "but", "not", "you", "all", "can",
            "had", "her", "was", "one", "our", "out", "has", "have", "with",
            "this", "that", "from", "they", "been", "said", "will", "each",
        }
        self.wikilink_stopwords |= {w.lower() for w in self.wikilink_exclude_keywords}

        # --- Search scoring weights / 檢索權重配置 ---
        scoring = config.get("scoring_weights", {})
        self.w_topic = scoring.get("topic_relevance", 4.0)
        self.w_emotion = scoring.get("emotion_resonance", 2.0)
        self.w_time = scoring.get("time_proximity", 1.5)
        self.w_importance = scoring.get("importance", 1.0)
        self.content_weight = scoring.get("content_weight", 1.0)  # body×1, per spec

        # --- Optional embedding engine for pre-filtering / 可選 embedding 引擎，用於預篩候選集 ---
        self.embedding_engine = embedding_engine

        # --- mtime-keyed bucket cache: list_all() re-parses only changed files ---
        # --- 以 mtime 為鍵的桶緩存：list_all() 只重新解析有變化的文件 ---
        self._bucket_cache: dict[str, tuple[float, dict]] = {}

    # ---------------------------------------------------------
    # Create a new bucket
    # 創建新桶
    # Write content and metadata into a .md file
    # 將內容和元數據寫入一個 .md 文件
    # ---------------------------------------------------------
    async def create(
        self,
        content: str,
        tags: list[str] = None,
        importance: int = 5,
        domain: list[str] = None,
        valence: float = 0.5,
        arousal: float = 0.3,
        bucket_type: str = "dynamic",
        name: str = None,
        pinned: bool = False,
        protected: bool = False,
        extra_meta: dict = None,
    ) -> str:
        """
        Create a new memory bucket, return bucket ID.
        創建一個新的記憶桶，返回桶 ID。

        pinned/protected=True: bucket won't be merged, decayed, or have importance changed.
        Importance is locked to 10 for pinned/protected buckets.
        pinned/protected 桶不參與合併與衰減，importance 強制鎖定為 10。
        """
        bucket_id = generate_bucket_id()
        bucket_name = sanitize_name(name) if name else bucket_id
        # feel/mirage buckets are allowed to have empty domain; others default to ["未分類"]
        if bucket_type in ("feel", "mirage"):
            domain = normalize_domains(domain) if domain is not None else []
        else:
            domain = normalize_domains(domain) or ["未分類"]
        tags = tags or []
        # Pinned non-permanent must become permanent BEFORE metadata is built,
        # otherwise the file lands in permanent/ with frontmatter still saying dynamic.
        # 釘選桶必須在構建 metadata 之前就轉為 permanent，否則目錄與 type 不一致。
        if pinned and bucket_type not in ("permanent",):
            bucket_type = "permanent"
        linked_content = content  # wikilink injection disabled; LLM adds [[]] via prompt

        # --- Pinned/protected buckets: lock importance to 10 ---
        # --- 釘選/保護桶：importance 強制鎖定為 10 ---
        if pinned or protected:
            importance = 10

        # --- Build YAML frontmatter metadata / 構建元數據 ---
        metadata = {
            "id": bucket_id,
            "name": bucket_name,
            "tags": tags,
            "domain": domain,
            "valence": max(0.0, min(1.0, valence)),
            "arousal": max(0.0, min(1.0, arousal)),
            "importance": max(1, min(10, importance)),
            "type": bucket_type,
            "created": now_iso(),
            "last_active": now_iso(),
            "activation_count": 0,
        }
        if pinned:
            metadata["pinned"] = True
        if protected:
            metadata["protected"] = True
        # --- Whitelisted extra metadata (plan fields, provenance notes) ---
        # --- 白名單制的附加元數據（plan 欄位、記錄原因）---
        for key in _EXTRA_META_KEYS:
            value = (extra_meta or {}).get(key)
            if value is None or value == "":
                continue
            if key in _FLOAT_META_KEYS:
                try:
                    metadata[key] = max(0.0, min(1.0, float(value)))
                except (TypeError, ValueError):
                    continue
            elif key == "status":
                if str(value) in _PLAN_STATUSES:
                    metadata[key] = str(value)
            elif key == "verbatim_guard":
                # 布林旗標存真 bool（str 化會讓 'False' 也為真）。
                metadata[key] = bool(value)
            else:
                metadata[key] = str(value)[:200]

        # --- Assemble Markdown file (frontmatter + body) ---
        # --- 組裝 Markdown 文件 ---
        post = frontmatter.Post(linked_content, **metadata)

        # --- Choose directory by type + primary domain ---
        # --- 按類型 + 主題域選擇存儲目錄 ---
        if bucket_type == "permanent" or pinned:
            type_dir = self.permanent_dir
        elif bucket_type == "feel":
            type_dir = self.feel_dir
        elif bucket_type == "mirage":
            type_dir = self.mirage_dir
        elif bucket_type == "plan":
            type_dir = self.plan_dir
        else:
            type_dir = self.dynamic_dir
        if bucket_type == "feel":
            primary_domain = "沉澱物"  # feel subfolder name
        elif bucket_type == "mirage":
            primary_domain = "蜃景"  # mirage subfolder name
        else:
            primary_domain = sanitize_name(domain[0]) if domain else "未分類"
        target_dir = os.path.join(type_dir, primary_domain)
        os.makedirs(target_dir, exist_ok=True)

        # --- Filename: readable_name_bucketID.md (Obsidian friendly) ---
        # --- 文件名：可讀名稱_桶ID.md ---
        if bucket_name and bucket_name != bucket_id:
            filename = f"{bucket_name}_{bucket_id}.md"
        else:
            filename = f"{bucket_id}.md"
        file_path = safe_path(target_dir, filename)

        try:
            atomic_write_text(str(file_path), frontmatter.dumps(post))
        except OSError as e:
            logger.error(f"Failed to write bucket file / 寫入桶文件失敗: {file_path}: {e}")
            raise

        logger.info(
            f"Created bucket / 創建記憶桶: {bucket_id} ({bucket_name}) → {primary_domain}/"
            + (" [PINNED]" if pinned else "") + (" [PROTECTED]" if protected else "")
        )
        return bucket_id

    # ---------------------------------------------------------
    # Read bucket content
    # 讀取桶內容
    # Returns {"id", "metadata", "content", "path"} or None
    # ---------------------------------------------------------
    async def get(self, bucket_id: str) -> Optional[dict]:
        """
        Read a single bucket by ID.
        根據 ID 讀取單個桶。
        """
        if not bucket_id or not isinstance(bucket_id, str):
            return None
        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return None
        return self._load_bucket(file_path)

    # ---------------------------------------------------------
    # Move bucket between directories
    # 在目錄間移動桶文件
    # ---------------------------------------------------------
    def _move_bucket(self, file_path: str, target_type_dir: str, domain: list[str] = None) -> str:
        """
        Move a bucket file to a new type directory, preserving domain subfolder.
        Returns new file path.
        """
        primary_domain = sanitize_name(domain[0]) if domain else "未分類"
        target_dir = os.path.join(target_type_dir, primary_domain)
        os.makedirs(target_dir, exist_ok=True)
        filename = os.path.basename(file_path)
        new_path = safe_path(target_dir, filename)
        if os.path.normpath(file_path) != os.path.normpath(new_path):
            # shutil.move 而非 os.rename：buckets_dir 可能是 iCloud/外掛磁碟，
            # os.rename 跨檔案系統會丟 EXDEV。archive() 早就為了這個用 shutil.move。
            shutil.move(file_path, str(new_path))
            logger.info(f"Moved bucket / 移動記憶桶: {filename} → {target_dir}/")
        return new_path

    # ---------------------------------------------------------
    # Update bucket
    # 更新桶
    # Supports: content, tags, importance, valence, arousal, name, resolved
    # ---------------------------------------------------------
    async def update(self, bucket_id: str, actor=None, **kwargs) -> bool:
        """
        Update bucket content or metadata fields.
        更新桶的內容或元數據字段。

        actor: a bucket_history.Actor saying who is doing this. Snapshots the
        pre-change state so the write is recoverable. Defaults to an unattributed
        system write rather than refusing — an unattributed history row is still
        worth far more than no row.
        actor：bucket_history.Actor，說明這是誰做的。會先存變更前的快照讓這次寫入
        可復原。沒傳就記成無歸屬的系統寫入而不是拒絕——沒有歸屬的歷史列，
        仍然遠比沒有列有價值。
        """
        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return False

        try:
            post = frontmatter.load(file_path)
        except Exception as e:
            logger.warning(f"Failed to load bucket for update / 加載桶失敗: {file_path}: {e}")
            return False

        # --- Snapshot BEFORE any mutation ---
        # --- 任何變更之前先存快照 ---
        # This sits inside update() rather than at the call sites on purpose:
        # update() is the single choke point every destructive metadata/content
        # write funnels through, so nothing can forget to checkpoint. (Letta
        # requires an explicit checkpoint_block call, which can be forgotten.)
        # touch()/mark_surfaced() deliberately do NOT come through here — they
        # only bump counters, and snapshotting every surfacing would bury the
        # real edits in noise.
        # 這放在 update() 裡而不是放在各呼叫點，是刻意的：update() 是所有破壞性
        # 寫入的唯一收口，所以不可能有人忘記存檔。（Letta 要求明確呼叫 checkpoint，
        # 那是會被忘記的。）touch()/mark_surfaced() 刻意不走這裡——它們只加計數器，
        # 每次浮現都存快照會把真正的編輯埋進雜訊裡。
        if self.history is not None:
            self.history.snapshot(
                bucket_id,
                post.content,
                dict(post.metadata),
                op="update",
                actor=actor,
            )

        # --- Pinned/protected buckets: lock importance to 10, ignore importance changes ---
        # --- 釘選/保護桶：importance 不可修改，強制保持 10 ---
        is_pinned = post.get("pinned", False) or post.get("protected", False)
        if is_pinned:
            kwargs.pop("importance", None)  # silently ignore importance update

        # --- Update only fields that were passed in / 只改傳入的字段 ---
        if "content" in kwargs:
            post.content = kwargs["content"]  # wikilink injection disabled; LLM adds [[]] via prompt
        if "tags" in kwargs:
            post["tags"] = kwargs["tags"]
        if "importance" in kwargs:
            post["importance"] = max(1, min(10, int(kwargs["importance"])))
        if "domain" in kwargs:
            post["domain"] = normalize_domains(kwargs["domain"])
        if "valence" in kwargs:
            post["valence"] = max(0.0, min(1.0, float(kwargs["valence"])))
        if "arousal" in kwargs:
            post["arousal"] = max(0.0, min(1.0, float(kwargs["arousal"])))
        if "name" in kwargs:
            post["name"] = sanitize_name(kwargs["name"])
        if "resolved" in kwargs:
            post["resolved"] = bool(kwargs["resolved"])
        if "pinned" in kwargs:
            was_pinned = bool(post.get("pinned", False))
            post["pinned"] = bool(kwargs["pinned"])
            if kwargs["pinned"]:
                # Remember pre-pin importance so unpin can restore it
                # 記住釘選前的重要度，取消釘選時恢復
                if not was_pinned and "importance_prepin" not in post.metadata:
                    try:
                        post["importance_prepin"] = int(post.get("importance", 5) or 5)
                    except (ValueError, TypeError):
                        post["importance_prepin"] = 5
                post["importance"] = 10  # pinned → lock importance to 10
            elif was_pinned and "importance_prepin" in post.metadata:
                post["importance"] = max(1, min(10, int(post.metadata.pop("importance_prepin"))))
        if "digested" in kwargs:
            post["digested"] = bool(kwargs["digested"])
        if "model_valence" in kwargs:
            post["model_valence"] = max(0.0, min(1.0, float(kwargs["model_valence"])))
        if "status" in kwargs and str(kwargs["status"]) in _PLAN_STATUSES:
            post["status"] = str(kwargs["status"])
        if "weight" in kwargs:
            try:
                post["weight"] = max(0.0, min(1.0, float(kwargs["weight"])))
            except (TypeError, ValueError):
                pass
        if "progress" in kwargs:
            try:
                post["progress"] = max(0.0, min(1.0, float(kwargs["progress"])))
            except (TypeError, ValueError):
                pass
        if "kind" in kwargs:
            post["kind"] = str(kwargs["kind"])[:32]
        if "target_drive" in kwargs:
            post["target_drive"] = str(kwargs["target_drive"])[:32]
        if "due_at" in kwargs:
            post["due_at"] = str(kwargs["due_at"])[:64]
        if "affects_desire" in kwargs:
            post["affects_desire"] = bool(kwargs["affects_desire"])
        if "why_remembered" in kwargs:
            post["why_remembered"] = str(kwargs["why_remembered"])[:200]
        if "related_bucket" in kwargs:
            post["related_bucket"] = str(kwargs["related_bucket"])[:64]
        if "verbatim_guard" in kwargs:
            # 逐字保護（2026-07-19 audit F11b）：桶裡一旦住進「一字不動」的定稿，
            # 日後任何合併都只准 append、不准 LLM 改寫——承諾要對已入庫的內容永久成立。
            post["verbatim_guard"] = bool(kwargs["verbatim_guard"])

        # --- Auto-refresh activation time / 自動刷新激活時間 ---
        post["last_active"] = now_iso()

        try:
            atomic_write_text(file_path, frontmatter.dumps(post))
        except OSError as e:
            logger.error(f"Failed to write bucket update / 寫入桶更新失敗: {file_path}: {e}")
            return False

        # --- Auto-move: pinned → permanent/ ---
        # --- 自動移動：釘選 → permanent/ ---
        # NOTE: resolved buckets are NOT auto-archived here.
        # They stay in dynamic/ and decay naturally until score < threshold.
        # 注意：resolved 桶不在此自動歸檔，留在 dynamic/ 隨衰減引擎自然歸檔。
        domain = post.get("domain", ["未分類"])
        if kwargs.get("pinned") and post.get("type") != "permanent":
            post["type"] = "permanent"
            atomic_write_text(file_path, frontmatter.dumps(post))
            self._move_bucket(file_path, self.permanent_dir, domain)
        elif "pinned" in kwargs and not kwargs["pinned"] and post.get("type") == "permanent" and not post.get("protected"):
            # --- Reverse: unpin → demote permanent back to dynamic/ ---
            # --- 反向：取消釘選 → 從 permanent/ 降回 dynamic/，讓衰減引擎接手 ---
            post["type"] = "dynamic"
            atomic_write_text(file_path, frontmatter.dumps(post))
            file_path = self._move_bucket(file_path, self.dynamic_dir, domain)
        elif post.get("type") == "archived" and "resolved" in kwargs and not kwargs["resolved"]:
            # --- Revive: re-activating an archived bucket pulls it back to dynamic/ ---
            # --- 復活：對歸檔桶 resolved=0 視為喚回，搬回 dynamic/ 重新參與生命週期 ---
            # Move FIRST, write second — restore()'s reasoning applies here
            # too: writing type:dynamic and then failing the move leaves a
            # file in archive/ that believes it is alive (list_all can't see
            # it, revive() refuses it). Move-first fails toward a repairable
            # state: a dynamic/-located file still saying archived.
            # 先搬再寫——restore() 的理由同樣適用：先寫 type:dynamic 再搬失敗，
            # 會在 archive/ 留下自以為活著的幽靈（list_all 看不見、revive()
            # 不認）。先搬的失敗方向是可修復的：檔案已在 dynamic/、type 還是
            # archived，revive() 修得回來。
            file_path = self._move_bucket(file_path, self.dynamic_dir, domain)
            post["type"] = "dynamic"
            atomic_write_text(file_path, frontmatter.dumps(post))
            logger.info(f"Revived bucket from archive / 歸檔桶復活: {bucket_id}")

        logger.info(f"Updated bucket / 更新記憶桶: {bucket_id}")
        return True

    # ---------------------------------------------------------
    # Wikilink injection — DISABLED
    # 自動添加 Obsidian 雙鏈 — 已禁用
    # Now handled by LLM prompts (Gemini adds [[]] for proper nouns)
    # 現在由 LLM prompt 處理（Gemini 對人名/地名/專有名詞加 [[]]）
    # ---------------------------------------------------------
    # def _apply_wikilinks(self, content, tags, domain, name): ...
    # def _collect_wikilink_keywords(self, content, tags, domain, name): ...
    # def _normalize_keywords(self, keywords): ...
    # def _extract_auto_keywords(self, content): ...

    # ---------------------------------------------------------
    # Delete bucket
    # 刪除桶
    # ---------------------------------------------------------
    async def delete(self, bucket_id: str, actor=None) -> "DeleteResult":
        """
        Delete a memory bucket file. Returns DeleteResult(ok, seq) — check .ok,
        never the tuple's truthiness (a NamedTuple is always truthy).
        刪除指定的記憶桶文件。回傳 DeleteResult(ok, seq)——要看 .ok，
        不要直接拿整個 tuple 當真假值判斷（NamedTuple 恆為真）。

        Snapshots first. This is the one operation with no other way back:
        archive/revive only move the file (it still exists on disk), but delete
        unlinks it and drops the embedding. Before this snapshot existed,
        trace(delete=True) was the single truly irreversible call in the system.
        先存快照。這是唯一沒有別條退路的操作：archive/revive 只是搬檔案（檔案還在），
        但 delete 會 unlink 並刪掉向量。在有這個快照之前，trace(delete=True)
        是整個系統裡唯一真正不可逆的呼叫。
        """
        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return DeleteResult(False, None)

        # snapshot() 回傳的 seq 一定要接住往上傳。以前它被丟掉，呼叫端只好事後用
        # COUNT(*) 重算一個——快照失敗時那個數字會指向更舊的版本，使用者照著印出來的
        # restore 指令做，會拿回舊正文而且沒有任何錯誤。快照失敗＝這次刪除不可復原，
        # 那就要說得出口，不能猜一個看起來能用的號碼。
        seq: Optional[int] = None
        if self.history is not None:
            try:
                post = frontmatter.load(file_path)
                seq = self.history.snapshot(
                    bucket_id, post.content, dict(post.metadata),
                    op="delete", actor=actor,
                )
            except Exception as e:
                logger.warning(f"Pre-delete snapshot failed / 刪除前快照失敗: {bucket_id}: {e}")

        try:
            os.remove(file_path)
        except OSError as e:
            logger.error(f"Failed to delete bucket file / 刪除桶文件失敗: {file_path}: {e}")
            return DeleteResult(False, None)

        if seq is None:
            logger.error(
                f"Deleted WITHOUT a usable history snapshot (unrecoverable) / "
                f"刪除了但沒有可用的歷史快照（救不回來）: {bucket_id}"
            )
        logger.info(f"Deleted bucket / 刪除記憶桶: {bucket_id}")
        return DeleteResult(True, seq)

    async def restore(self, bucket_id: str, seq: int, actor=None) -> bool:
        """
        Restore a bucket to the state recorded at history `seq`.
        把桶還原到歷史第 seq 版的狀態。

        The restore is itself a write, so it snapshots the current state first —
        which means an unwanted restore is itself undoable, and you can land on
        any seq rather than stepping ±1. Works on a deleted bucket too: the file
        is recreated from the snapshot.
        還原本身也是一次寫入，所以它會先把當下狀態存起來——這代表「還原錯了」
        本身也可以再還原回去，而且能直接跳到任一版而不是一步步 ±1。
        對已刪除的桶也有效：檔案會從快照重建。
        """
        if self.history is None:
            return False
        row = self.history.get(bucket_id, seq)
        if not row:
            return False

        try:
            meta = json.loads(row["meta"])
        except Exception as e:
            logger.error(f"Restore failed, unreadable snapshot / 快照無法解析: {bucket_id}#{seq}: {e}")
            return False

        domain = meta.get("domain", ["未分類"])
        target_dir = self._dir_for_type(meta.get("type", "dynamic"))

        file_path = self._find_bucket_file(bucket_id)
        if file_path:
            try:
                current = frontmatter.load(file_path)
                self.history.snapshot(
                    bucket_id, current.content, dict(current.metadata),
                    op="restore", actor=actor,
                )
            except Exception as e:
                logger.warning(f"Pre-restore snapshot failed / 還原前快照失敗: {bucket_id}: {e}")
        else:
            # Bucket was deleted — recreate it where its snapshot says it lives.
            # 桶已被刪除——照快照說它該在的地方重建。
            primary = sanitize_name(domain[0]) if domain else "未分類"
            subdir = os.path.join(target_dir, primary)
            os.makedirs(subdir, exist_ok=True)
            name = sanitize_name(str(meta.get("name", bucket_id)))
            file_path = str(safe_path(subdir, f"{name}_{bucket_id}.md"))

        # --- Put the file in the directory the restored type belongs to, FIRST ---
        # --- 先把檔案搬到「還原後的型別」該待的目錄，再寫內容 ---
        # list_all() walks DIRECTORIES, not the type field. A file sitting in
        # archive/ whose metadata says type:dynamic is a bucket that believes it
        # is alive but never surfaces and never decays — invisible to everything
        # except a direct get(), and revive() refuses to repair it because it
        # only touches type=="archived". That is the one state we must never
        # leave behind, so the move goes first: if it fails, nothing has changed
        # and we say so. Writing type:dynamic before the move would put the
        # failure case squarely on the ghost.
        # list_all() 走的是「目錄」不是 type 欄位。躺在 archive/ 卻寫著 type:dynamic
        # 的檔案，是個自以為活著、永不浮現也永不衰減的幽靈——除了直接 get() 對所有
        # 東西都隱形，而 revive() 只認 type=="archived"，救不了它。那是唯一絕對不能
        # 留下的狀態，所以搬移排在寫入前面：搬不動就什麼都沒動，並且照實回報。
        # （archive() 的順序相反卻是對的：它失敗會落在 dynamic/+archived，
        #   revive()/decay 都修得回來——兩邊都是朝安全的方向失敗。）
        if os.path.dirname(os.path.realpath(file_path)) != os.path.realpath(
            os.path.join(target_dir, sanitize_name(domain[0]) if domain else "未分類")
        ):
            try:
                file_path = self._move_bucket(file_path, target_dir, domain)
            except Exception as e:
                logger.error(
                    f"Restore could not relocate the bucket; nothing changed / "
                    f"還原搬移失敗，桶維持原狀: {bucket_id}: {e}"
                )
                return False

        post = frontmatter.Post(row["content"], **meta)
        try:
            atomic_write_text(file_path, frontmatter.dumps(post))
        except OSError as e:
            logger.error(f"Restore write failed / 還原寫入失敗: {file_path}: {e}")
            return False

        logger.info(f"Restored bucket / 還原記憶桶: {bucket_id} → seq {seq}")
        return True

    def _dir_for_type(self, bucket_type: str) -> str:
        """Which directory a bucket of this type lives in. list_all() walks
        directories, so type and location must never disagree.
        某個型別的桶該待在哪個目錄。list_all() 走目錄，所以型別和位置永遠不能矛盾。"""
        return {
            "permanent": self.permanent_dir,
            "archived": self.archive_dir,
            "feel": self.feel_dir,
            "mirage": self.mirage_dir,
            "plan": self.plan_dir,
        }.get(bucket_type, self.dynamic_dir)

    def vector_admissible(
        self,
        meta: dict,
        sim: float,
        query_valence: float | None = None,
        query_arousal: float | None = None,
        domain_filter: list | None = None,
    ) -> bool:
        """Admissibility gate for the vector supplement channel (2026-07-12
        batch-3): the semantic side door must honor the same boundaries as
        the ranked channel. An explicit domain filter binds vector hits too;
        in a NEUTRAL query (no emotion coordinates), a gated-domain
        high-arousal bucket needs a strong hit (sim ≥ context_gate_vector_sim)
        to enter — a 0.5-grade graze must not surface or revive it. Queries
        that carry emotion coordinates are exempt, same as the text gate.
        向量補充通道的入場閘：補位通道不能變成繞過情境門控與 domain filter
        的側門。明確給了 domain filter → 向量結果也要在域內；中性查詢碰到
        門控域的高喚醒桶 → 相似度必須夠強才入列（弱擦邊不浮現、不復活）；
        帶情緒座標的查詢照舊豁免——有意去找就找得到。"""
        if domain_filter and not (set(meta.get("domain", []) or []) & set(domain_filter)):
            return False
        if (
            self.context_gate_enabled
            and query_valence is None and query_arousal is None
            and sim < self.context_gate_vector_sim
            and float(meta.get("arousal", 0.3)) >= self.context_gate_arousal
            and set(meta.get("domain", []) or []) & self.context_gate_domains
        ):
            return False
        return True

    # ---------------------------------------------------------
    # Touch bucket (refresh activation time + increment count)
    # 觸碰桶（刷新激活時間 + 累加激活次數）
    # Called on every recall hit; affects decay score.
    # 每次檢索命中時調用，影響衰減得分。
    # ---------------------------------------------------------
    async def mark_surfaced(self, bucket_id: str) -> None:
        """Record that a bucket was shown (surfaced or returned by search)
        WITHOUT reinforcing it: last_surfaced + retrieved_count only —
        last_active / activation_count stay untouched, so being looked at
        is not the same as being engaged with (2026-07-12 batch-2).
        記錄「被看見」但不加固：只寫 last_surfaced 與 retrieved_count，
        不動 last_active / activation_count——被看到 ≠ 被用到。"""
        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return
        try:
            post = frontmatter.load(file_path)
            post["last_surfaced"] = now_iso()
            try:
                post["retrieved_count"] = int(post.get("retrieved_count", 0)) + 1
            except (ValueError, TypeError):
                post["retrieved_count"] = 1
            atomic_write_text(file_path, frontmatter.dumps(post))
        except Exception as e:
            logger.warning(f"Failed to mark surfaced / 記錄浮現失敗: {bucket_id}: {e}")

    async def touch(self, bucket_id: str, ripple: bool = False) -> None:
        """
        Update a bucket's last activation time and count (capped at ACTIVATION_CAP).
        ripple=True additionally wakes a random sample of temporal neighbors —
        callers should only ripple the strongest recall, not every search hit.
        更新桶的最後激活時間和激活次數（封頂 ACTIVATION_CAP）。
        ripple=True 才觸發時間漣漪；調用方只對最強的那次命中開漣漪，避免通脹。
        """
        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return

        try:
            post = frontmatter.load(file_path)
            post["last_active"] = now_iso()
            try:
                current = float(post.get("activation_count", 0))
            except (ValueError, TypeError):
                current = 0.0
            post["activation_count"] = min(round(current + 1, 1), ACTIVATION_CAP)

            atomic_write_text(file_path, frontmatter.dumps(post))

            # --- Time ripple: boost memories created within ±48h of this one ---
            # --- 時間漣漪：喚醒與本桶創建時間相鄰（±48h）的記憶 ---
            # tz 收口（2026-07-19 audit F2）：aware 時間戳（手改/舊 import）在這裡
            # 曾直接 fromisoformat，naive−aware 相減拋 TypeError 被外層吞掉——
            # 統一走 parse_bucket_ts（decay_engine 07-15 已收口，這邊補齊另一半）。
            created_str = post.get("created", "")
            _ripple_ref = parse_bucket_ts(created_str) if (ripple and created_str) else None
            if _ripple_ref is not None:
                await self._time_ripple(bucket_id, _ripple_ref)
        except Exception as e:
            logger.warning(f"Failed to touch bucket / 觸碰桶失敗: {bucket_id}: {e}")

    async def _time_ripple(self, source_id: str, reference_time: datetime, hours: float = 48.0) -> None:
        """
        Slightly boost activation_count of buckets created near the reference time.
        Samples up to 5 eligible neighbors at random (walk order would always feed
        the same few buckets); counts stay capped at ACTIVATION_CAP.
        輕微提升時間相鄰桶的激活次數（+0.3）：在全部合格鄰居中隨機取樣最多5個，
        避免固定餵養目錄順序靠前的桶；封頂 ACTIVATION_CAP。
        """
        try:
            all_buckets = await self.list_all(include_archive=False)
        except Exception:
            return

        eligible = []
        for bucket in all_buckets:
            if bucket["id"] == source_id:
                continue
            meta = bucket.get("metadata", {})
            # Skip pinned/permanent/feel/mirage and buckets already settled or digested
            if meta.get("pinned") or meta.get("protected") or meta.get("type") in ("permanent", "feel", "mirage"):
                continue
            if meta.get("resolved") or meta.get("digested"):
                continue

            created = parse_bucket_ts(meta.get("created", ""))
            if created is None:
                continue
            delta_hours = abs((reference_time - created).total_seconds()) / 3600
            if delta_hours <= hours:
                eligible.append(bucket["id"])

        for target_id in random.sample(eligible, min(5, len(eligible))):
            file_path = self._find_bucket_file(target_id)
            if not file_path:
                continue
            try:
                post = frontmatter.load(file_path)
                try:
                    current_count = float(post.get("activation_count", 1))
                except (ValueError, TypeError):
                    current_count = 1.0
                # Fractional boost, don't change last_active (avoids recursive wake)
                post["activation_count"] = min(round(current_count + 0.3, 1), ACTIVATION_CAP)
                atomic_write_text(file_path, frontmatter.dumps(post))
            except Exception:
                continue

    # ---------------------------------------------------------
    # Multi-dimensional search (core feature)
    # 多維搜索（核心功能）
    #
    # Strategy: domain pre-filter → weighted multi-dim ranking
    # 策略：主題域預篩 → 多維加權精排
    #
    # Ranking formula:
    #   total = topic(×w_topic) + emotion(×w_emotion)
    #           + time(×w_time) + importance(×w_importance)
    #
    # Per-dimension scores (normalized to 0~1):
    #   topic     = rapidfuzz weighted match (name/tags/domain/body)
    #   emotion   = 1 - Euclidean distance (query v/a vs bucket v/a)
    #   time      = e^(-0.02 × days) (recent memories first)
    #   importance = importance / 10
    # ---------------------------------------------------------
    async def search(
        self,
        query: str,
        limit: int = None,
        domain_filter: list[str] = None,
        query_valence: float = None,
        query_arousal: float = None,
        exclude_domains: list[str] = None,
    ) -> list[dict]:
        """
        Multi-dimensional indexed search for memory buckets.
        多維索引搜索記憶桶。

        domain_filter: pre-filter by domain (None = search all)
        query_valence/arousal: emotion coordinates for resonance scoring
        exclude_domains: drop buckets carrying ANY of these domains (the
        inverse of domain_filter; batch-7 檔1)
        """
        if not query or not query.strip():
            return []

        limit = limit or self.max_results
        all_buckets = await self.list_all(include_archive=False)

        # --- Feels are a private channel (breath domain="feel" only), never search results.
        # Plans live in dream's tail only — keeping them out of search also keeps
        # merge_or_create from ever merging ordinary memories into a plan. ---
        # --- feel 是獨立私人通道，永遠不進普通搜索；plan 只活在 dream 尾端，
        # 排除在搜索外也保證合併管線永遠不會把普通記憶併進 plan。---
        all_buckets = [b for b in all_buckets if b["metadata"].get("type") not in ("feel", "plan", "mirage")]

        # --- exclude_domains applies at the POOL level, not as a post-filter:
        # BM25 (layer 1.6) re-adds strong keyword hits from all_buckets, so an
        # exclusion any later would leak through that rejoin path.
        # --- 排除落在「底池」層而不是事後過濾：BM25（第1.6層）會從 all_buckets
        # 補回強關鍵詞命中，排除放後面就會被那條補回路徑漏過。---
        if exclude_domains:
            ex_set = {d.lower() for d in normalize_domains(exclude_domains)}
            if ex_set:
                all_buckets = [
                    b for b in all_buckets
                    if not ({d.lower() for d in b["metadata"].get("domain", []) or []} & ex_set)
                ]

        if not all_buckets:
            return []

        # --- Layer 1: domain pre-filter (fast scope reduction) ---
        # --- 第一層：主題域預篩（快速縮小範圍）---
        # Normalize the query side too: a simplified domain query must still
        # match the (now Traditional) canonical domains.
        # 查詢端也正規化：用簡體查 domain 也要能命中繁體正典域。
        if domain_filter:
            domain_filter = normalize_domains(domain_filter)
        if domain_filter:
            filter_set = {d.lower() for d in domain_filter}
            candidates = [
                b for b in all_buckets
                if {d.lower() for d in b["metadata"].get("domain", [])} & filter_set
            ]
        else:
            candidates = all_buckets

        # --- Layer 1.5 removed (2026-07-12): the embedding pre-filter REPLACED the
        # candidate set with the vector top-50, so an exact name/tag hit outside
        # that set was dropped before precision ranking ever saw it. At this corpus
        # size full fuzzy ranking costs milliseconds; semantic recall has its own
        # parallel vector channel in breath (server.py) and BM25 (layer 1.6) covers
        # full-body keywords. Recall beats latency here.
        # --- 第1.5層已移除（2026-07-12）：embedding 預篩會「取代」候選集，
        # 精確的名字/標籤命中若不在向量前 50 就在精排前被丟掉。這個語料規模
        # 全量精排只要毫秒級；語義召回由 breath 的並聯向量通道負責，
        # 全文關鍵詞由 BM25（第1.6層）兜底。召回優先於延遲。---

        # --- Layer 1.6: BM25 keyword channel (dormant unless matching.bm25_enabled) ---
        # Recall insurance: strong keyword hits rejoin the candidate set even if
        # the domain/embedding pre-filters dropped them, and the normalized BM25
        # score later blends into topic relevance.
        # --- 第1.6層：BM25 關鍵詞通道（休眠旗標）。強關鍵詞命中補回候選集，
        # 正規化分數稍後融入 topic 相關度。---
        bm25_norm: dict[str, float] = {}
        if self.bm25_enabled:
            try:
                bm25_norm = self._bm25_scores(query, all_buckets)
                if bm25_norm:
                    candidate_ids = {b["id"] for b in candidates}
                    by_id = {b["id"]: b for b in all_buckets}
                    top_ids = sorted(bm25_norm, key=bm25_norm.get, reverse=True)[:20]
                    for bid in top_ids:
                        if bid not in candidate_ids and bid in by_id:
                            candidates.append(by_id[bid])
                            candidate_ids.add(bid)
            except Exception as e:
                logger.warning(f"BM25 channel failed, continuing without / BM25 通道失敗: {e}")
                bm25_norm = {}

        # --- Layer 2: weighted multi-dim ranking ---
        # --- 第二層：多維加權精排 ---
        scored = []
        for bucket in candidates:
            meta = bucket.get("metadata", {})

            try:
                # Dim 1: topic relevance (fuzzy text, 0~1; BM25 blends in when enabled)
                topic_score = self._calc_topic_score(query, bucket)
                if bm25_norm:
                    topic_score = max(topic_score, bm25_norm.get(bucket["id"], 0.0))

                # Dim 2: emotion resonance (coordinate distance, 0~1)
                emotion_score = self._calc_emotion_score(
                    query_valence, query_arousal, meta
                )

                # Dim 3: time proximity (exponential decay, 0~1)
                time_score = self._calc_time_score(meta)

                # Dim 4: importance (direct normalization)
                importance_score = max(1, min(10, int(meta.get("importance", 5)))) / 10.0

                # --- Weighted sum / 加權求和 ---
                total = (
                    topic_score * self.w_topic
                    + emotion_score * self.w_emotion
                    + time_score * self.w_time
                    + importance_score * self.w_importance
                )
                # Normalize to 0~100 for readability
                weight_sum = self.w_topic + self.w_emotion + self.w_time + self.w_importance
                normalized = (total / weight_sum) * 100 if weight_sum > 0 else 0

                # --- Context gate (2026-07-12 batch-2): in a NEUTRAL query context
                # (no emotion coordinates passed), high-arousal intimate memories
                # need a strong topic hit to rank — a work-context search should
                # not surface bedroom memories on a weak fuzzy match. Strong topic
                # relevance (≥0.5) passes untouched; vector channel is unaffected
                # (semantic similarity IS a strong signal).
                # --- 情境門控（2026-07-12 第二批）：中性語境（沒帶情緒座標的查詢）
                # 裡，高喚醒的親密記憶要有夠強的主題命中才排得上——工作語境的
                # 搜尋不該因為模糊分擦邊就浮出臥室記憶。主題相關 ≥0.5 原樣放行；
                # 向量通道不受影響（語義相似本身就是強信號）。---
                if (
                    self.context_gate_enabled
                    and query_valence is None and query_arousal is None
                    and topic_score < 0.5
                    and float(meta.get("arousal", 0.3)) >= self.context_gate_arousal
                    and set(meta.get("domain", [])) & self.context_gate_domains
                ):
                    normalized *= self.context_gate_damp

                # Threshold check uses raw (pre-penalty) score so resolved buckets
                # 閾值用原始分數判定，確保 resolved 桶在關鍵詞命中時仍可被搜出
                # remain reachable by keyword (penalty applied only to ranking).
                if normalized >= self.fuzzy_threshold:
                    # Resolved/digested buckets get ranking penalty (still reachable by keyword)
                    # 已解決/已消化的桶僅在排序時降權（關鍵詞仍可召回）
                    if meta.get("resolved", False) or meta.get("digested", False):
                        normalized *= 0.3
                    bucket["score"] = round(normalized, 2)
                    scored.append(bucket)
            except Exception as e:
                logger.warning(
                    f"Scoring failed for bucket {bucket.get('id', '?')} / "
                    f"桶評分失敗: {e}"
                )
                continue

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    # ---------------------------------------------------------
    # Topic relevance sub-score:
    # name(×3) + domain(×2.5) + tags(×2) + body(×1)
    # 文本相關性子分：桶名(×3) + 主題域(×2.5) + 標籤(×2) + 正文(×1)
    # ---------------------------------------------------------
    def _bm25_scores(self, query: str, buckets: list[dict]) -> dict[str, float]:
        """Normalized (0~1) BM25 scores for the query over the given buckets.
        The index rebuilds only when the corpus fingerprint changes — at our
        scale that is tens of milliseconds, paid rarely.
        對候選語料算正規化 BM25 分；語料指紋沒變就沿用既有索引。"""
        from bm25_index import Bm25Index
        version = (
            len(buckets),
            hash(tuple(sorted(
                (b["id"], str(b["metadata"].get("last_active", ""))) for b in buckets
            ))),
        )
        if self._bm25 is None or self._bm25.version != version:
            docs = {}
            for b in buckets:
                meta = b["metadata"]
                docs[b["id"]] = " ".join([
                    str(meta.get("name", "")),
                    " ".join(meta.get("tags", []) or []),
                    " ".join(meta.get("domain", []) or []),
                    (b.get("content") or "")[:1000],
                ])
            index = Bm25Index()
            index.build(docs, version=version)
            self._bm25 = index
        hits = self._bm25.search(query, top_k=50)
        if not hits:
            return {}
        top = hits[0][1] or 1.0
        return {doc_id: score / top for doc_id, score in hits}

    def _calc_topic_score(self, query: str, bucket: dict) -> float:
        """
        Calculate text dimension relevance score (0~1).
        計算文本維度的相關性得分。
        """
        meta = bucket.get("metadata", {})

        name_score = fuzz.partial_ratio(query, meta.get("name", "")) * 3
        domain_score = (
            max(
                (fuzz.partial_ratio(query, d) for d in meta.get("domain", [])),
                default=0,
            )
            * 2.5
        )
        tag_score = (
            max(
                (fuzz.partial_ratio(query, tag) for tag in meta.get("tags", [])),
                default=0,
            )
            * 2
        )
        content_score = fuzz.partial_ratio(query, bucket.get("content", "")[:1000]) * self.content_weight

        return (name_score + domain_score + tag_score + content_score) / (100 * (3 + 2.5 + 2 + self.content_weight))

    # ---------------------------------------------------------
    # Emotion resonance sub-score:
    # Based on Russell circumplex Euclidean distance
    # 情感共鳴子分：基於環形情感模型的歐氏距離
    # No emotion in query → neutral 0.5 (doesn't affect ranking)
    # ---------------------------------------------------------
    def _calc_emotion_score(
        self, q_valence: float, q_arousal: float, meta: dict
    ) -> float:
        """
        Calculate emotion resonance score (0~1, closer = higher).
        計算情感共鳴度（0~1，越近越高）。
        """
        if q_valence is None or q_arousal is None:
            return 0.5  # No emotion coordinates → neutral / 無情感座標時給中性分

        try:
            b_valence = float(meta.get("valence", 0.5))
            b_arousal = float(meta.get("arousal", 0.3))
        except (ValueError, TypeError):
            return 0.5

        # Euclidean distance, max sqrt(2) ≈ 1.414
        dist = math.sqrt((q_valence - b_valence) ** 2 + (q_arousal - b_arousal) ** 2)
        return max(0.0, 1.0 - dist / 1.414)

    # ---------------------------------------------------------
    # Time proximity sub-score:
    # More recent activation → higher score
    # 時間親近子分：距上次激活越近分越高
    # ---------------------------------------------------------
    def _calc_time_score(self, meta: dict) -> float:
        """
        Calculate time proximity score (0~1, more recent = higher).
        計算時間親近度。
        """
        # parse_bucket_ts 而非 raw fromisoformat（2026-07-19 audit F2）：aware 時間戳
        # 曾在此拋 TypeError 被吞成 days=30——「30 天前」與真實新舊脫鉤，兩個方向都錯。
        last_active = parse_bucket_ts(meta.get("last_active", meta.get("created", "")))
        if last_active is None:
            days = 30.0
        else:
            days = max(0.0, (datetime.now() - last_active).total_seconds() / 86400)
        return math.exp(-0.02 * days)

    # ---------------------------------------------------------
    # List all buckets
    # 列出所有桶
    # ---------------------------------------------------------
    async def list_all(self, include_archive: bool = False) -> list[dict]:
        """
        Recursively walk directories (including domain subdirs), list all buckets.
        遞歸遍歷目錄（含域子目錄），列出所有記憶桶。
        """
        buckets = []

        dirs = [self.permanent_dir, self.dynamic_dir, self.feel_dir, self.plan_dir, self.mirage_dir]
        if include_archive:
            dirs.append(self.archive_dir)

        for dir_path in dirs:
            if not os.path.exists(dir_path):
                continue
            for root, _, files in os.walk(dir_path):
                for filename in files:
                    if not filename.endswith(".md"):
                        continue
                    file_path = os.path.join(root, filename)
                    # mtime cache: only re-parse files that actually changed
                    # mtime 緩存：只重新解析有變化的文件
                    try:
                        mtime = os.path.getmtime(file_path)
                    except OSError:
                        continue
                    cached = self._bucket_cache.get(file_path)
                    if cached and cached[0] == mtime:
                        buckets.append({**cached[1]})
                        continue
                    bucket = self._load_bucket(file_path)
                    if bucket:
                        self._bucket_cache[file_path] = (mtime, bucket)
                        buckets.append({**bucket})

        # Bound cache growth (stale paths from moved/deleted files)
        if len(self._bucket_cache) > 5000:
            self._bucket_cache.clear()

        return buckets

    # ---------------------------------------------------------
    # Statistics (counts per category + total size)
    # 統計信息（各分類桶數量 + 總體積）
    # ---------------------------------------------------------
    async def get_stats(self) -> dict:
        """
        Return memory bucket statistics (including domain subdirs).
        返回記憶桶的統計數據。
        """
        stats = {
            "permanent_count": 0,
            "dynamic_count": 0,
            "archive_count": 0,
            "feel_count": 0,
            "plan_count": 0,
            "mirage_count": 0,
            "total_size_kb": 0.0,
            "domains": {},
        }

        for subdir, key in [
            (self.permanent_dir, "permanent_count"),
            (self.dynamic_dir, "dynamic_count"),
            (self.archive_dir, "archive_count"),
            (self.feel_dir, "feel_count"),
            (self.plan_dir, "plan_count"),
            (self.mirage_dir, "mirage_count"),
        ]:
            if not os.path.exists(subdir):
                continue
            for root, _, files in os.walk(subdir):
                for f in files:
                    if f.endswith(".md"):
                        stats[key] += 1
                        fpath = os.path.join(root, f)
                        try:
                            stats["total_size_kb"] += os.path.getsize(fpath) / 1024
                        except OSError:
                            pass
                        # Per-domain counts / 每個域的桶數量
                        domain_name = os.path.basename(root)
                        if domain_name != os.path.basename(subdir):
                            stats["domains"][domain_name] = stats["domains"].get(domain_name, 0) + 1

        return stats

    # ---------------------------------------------------------
    # Archive bucket (move from permanent/dynamic into archive)
    # 歸檔桶（從 permanent/dynamic 移入 archive）
    # Called by decay engine to simulate "forgetting"
    # 由衰減引擎調用，模擬"遺忘"
    # ---------------------------------------------------------
    async def archive(self, bucket_id: str, actor=None) -> bool:
        """
        Move a bucket into the archive directory (preserving domain subdirs).
        將指定桶移入歸檔目錄（保留域子目錄結構）。

        Archiving is already reversible on its own (the file still exists, and
        revive() brings it back), so the snapshot here isn't insurance — it's
        the record of the system forgetting something on its own initiative,
        which is the one kind of write nobody was around to witness.
        歸檔本來就可逆（檔案還在，revive() 就搬得回來），所以這裡的快照不是保險，
        是「系統自己主動忘掉一件事」的紀錄——那是唯一沒有人在場見證的那種寫入。
        """
        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return False

        try:
            # Read once, get domain info and update type / 一次性讀取
            post = frontmatter.load(file_path)
            if self.history is not None:
                self.history.snapshot(
                    bucket_id, post.content, dict(post.metadata),
                    op="archive", actor=actor,
                )
            domain = post.get("domain", ["未分類"])
            primary_domain = sanitize_name(domain[0]) if domain else "未分類"
            archive_subdir = os.path.join(self.archive_dir, primary_domain)
            os.makedirs(archive_subdir, exist_ok=True)

            dest = safe_path(archive_subdir, os.path.basename(file_path))

            # Update type marker then move file / 更新類型標記後移動文件
            post["type"] = "archived"
            atomic_write_text(file_path, frontmatter.dumps(post))

            # Use shutil.move for cross-filesystem safety
            # 使用 shutil.move 保證跨文件系統安全
            shutil.move(file_path, str(dest))
        except Exception as e:
            logger.error(
                f"Failed to archive bucket / 歸檔桶失敗: {bucket_id}: {e}"
            )
            return False

        logger.info(f"Archived bucket / 歸檔記憶桶: {bucket_id} → archive/{primary_domain}/")
        return True

    # ---------------------------------------------------------
    # Revive bucket (move from archive back into dynamic)
    # 復活桶（從 archive 搬回 dynamic）
    # Called when the semantic channel recalls an archived memory:
    # forgetting is reversible when something genuinely reminds us.
    # 語義通道勾迴歸檔記憶時調用——真的被想起來，就回來。
    # ---------------------------------------------------------
    async def revive(self, bucket_id: str, resolved: bool = True, actor=None) -> bool:
        """
        Move an archived bucket back to dynamic/. Comes back resolved by default
        so a revival doesn't flood the surfacing pool.
        默認以 resolved 狀態迴歸，避免復活桶立刻擠佔浮現位。
        """
        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return False
        try:
            post = frontmatter.load(file_path)
            if post.get("type") != "archived":
                return False
            if self.history is not None:
                self.history.snapshot(
                    bucket_id, post.content, dict(post.metadata),
                    op="revive", actor=actor,
                )
            # Move FIRST, write second (same reasoning as restore()): a ghost
            # in archive/ claiming type:dynamic is the one unrecoverable
            # state; a dynamic/-located file still saying archived is exactly
            # what this function repairs on the next call.
            # 先搬再寫（與 restore() 同一套理由）：archive/ 裡寫著 type:dynamic
            # 的幽靈是唯一救不回的狀態；檔案在 dynamic/ 而 type 還是 archived，
            # 正是本函數下次呼叫就能修好的形狀。
            file_path = self._move_bucket(
                file_path, self.dynamic_dir, post.get("domain", ["未分類"]),
            )
            post["type"] = "dynamic"
            post["resolved"] = resolved
            post["last_active"] = now_iso()
            atomic_write_text(file_path, frontmatter.dumps(post))
            logger.info(f"Revived bucket / 復活記憶桶: {bucket_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to revive bucket / 復活桶失敗: {bucket_id}: {e}")
            return False

    # ---------------------------------------------------------
    # Internal: find bucket file across all three directories
    # 內部：在三個目錄中查找桶文件
    # ---------------------------------------------------------
    def _find_bucket_file(self, bucket_id: str) -> Optional[str]:
        """
        Recursively search permanent/dynamic/archive for a bucket file
        matching the given ID.
        在 permanent/dynamic/archive 中遞歸查找指定 ID 的桶文件。
        """
        if not bucket_id:
            return None
        for dir_path in [self.permanent_dir, self.dynamic_dir, self.archive_dir, self.feel_dir, self.plan_dir, self.mirage_dir]:
            if not os.path.exists(dir_path):
                continue
            for root, _, files in os.walk(dir_path):
                for fname in files:
                    if not fname.endswith(".md"):
                        continue
                    # Match by exact ID segment in filename
                    # 通過文件名中的 ID 片段精確匹配
                    name_part = fname[:-3]  # remove .md
                    if name_part == bucket_id or name_part.endswith(f"_{bucket_id}"):
                        return os.path.join(root, fname)
        return None

    # ---------------------------------------------------------
    # Internal: load bucket data from .md file
    # 內部：從 .md 文件加載桶數據
    # ---------------------------------------------------------
    def _load_bucket(self, file_path: str) -> Optional[dict]:
        """
        Parse a Markdown file and return structured bucket data.
        解析 Markdown 文件，返回桶的結構化數據。
        """
        try:
            post = frontmatter.load(file_path)
            return {
                "id": post.get("id", Path(file_path).stem),
                "metadata": _normalize_meta_datetimes(dict(post.metadata)),
                "content": post.content,
                "path": file_path,
            }
        except Exception as e:
            logger.warning(
                f"Failed to load bucket file / 加載桶文件失敗: {file_path}: {e}"
            )
            return None
