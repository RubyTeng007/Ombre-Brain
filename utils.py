# ============================================================
# Module: Common Utilities (utils.py)
# 模塊：通用工具函數
#
# Provides config loading, logging init, path safety, ID generation, etc.
# 提供配置加載、日誌初始化、路徑安全校驗、ID 生成等基礎能力
#
# Depended on by: server.py, bucket_manager.py, dehydrator.py, decay_engine.py
# 被誰依賴：server.py, bucket_manager.py, dehydrator.py, decay_engine.py
# ============================================================

import os
import re
import uuid
import yaml
import logging
from pathlib import Path
from datetime import datetime


def load_config(config_path: str = None) -> dict:
    """
    Load configuration file.
    加載配置文件。

    Priority: environment variables > config.yaml > built-in defaults.
    優先級：環境變量 > config.yaml > 內置默認值。
    """
    # --- Built-in defaults (fallback so it runs even without config.yaml) ---
    # --- 內置默認配置（兜底，保證即使沒有 config.yaml 也能跑）---
    defaults = {
        "transport": "stdio",
        "log_level": "INFO",
        "buckets_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "buckets"),
        "merge_threshold": 75,
        "dehydration": {
            "model": "deepseek-chat",
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "",
            "max_tokens": 1024,
            "temperature": 0.1,
        },
        "decay": {
            "lambda": 0.05,
            "threshold": 0.3,
            "check_interval_hours": 24,
            "emotion_weights": {
                "base": 1.0,
                "arousal_boost": 0.8,
            },
        },
        "matching": {
            "fuzzy_threshold": 50,
            "max_results": 5,
        },
    }

    # --- Load user config from YAML file ---
    # --- 從 YAML 文件加載用戶自定義配置 ---
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config.yaml"
        )

    config = defaults.copy()
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f) or {}
            if isinstance(file_config, dict):
                config = _deep_merge(defaults, file_config)
            else:
                logging.warning(
                    f"Config file is not a valid YAML dict, using defaults / "
                    f"配置文件不是有效的 YAML 字典，使用默認配置: {config_path}"
                )
        except yaml.YAMLError as e:
            logging.warning(
                f"Failed to parse config file, using defaults / "
                f"配置文件解析失敗，使用默認配置: {e}"
            )

    # --- Environment variable overrides (highest priority) ---
    # --- 環境變量覆蓋敏感/運行時配置（優先級最高）---
    env_api_key = os.environ.get("OMBRE_API_KEY", "")
    if env_api_key:
        config.setdefault("dehydration", {})["api_key"] = env_api_key

    env_base_url = os.environ.get("OMBRE_BASE_URL", "")
    if env_base_url:
        config.setdefault("dehydration", {})["base_url"] = env_base_url

    env_transport = os.environ.get("OMBRE_TRANSPORT", "")
    if env_transport:
        config["transport"] = env_transport

    env_buckets_dir = os.environ.get("OMBRE_BUCKETS_DIR", "")
    if env_buckets_dir:
        config["buckets_dir"] = env_buckets_dir

    # OMBRE_DEHYDRATION_MODEL (with OMBRE_MODEL alias) overrides dehydration.model
    env_dehy_model = os.environ.get("OMBRE_DEHYDRATION_MODEL", "") or os.environ.get("OMBRE_MODEL", "")
    if env_dehy_model:
        config.setdefault("dehydration", {})["model"] = env_dehy_model

    # OMBRE_DEHYDRATION_BASE_URL overrides dehydration.base_url
    env_dehy_base_url = os.environ.get("OMBRE_DEHYDRATION_BASE_URL", "")
    if env_dehy_base_url:
        config.setdefault("dehydration", {})["base_url"] = env_dehy_base_url

    # OMBRE_EMBEDDING_MODEL overrides embedding.model
    env_embed_model = os.environ.get("OMBRE_EMBEDDING_MODEL", "")
    if env_embed_model:
        config.setdefault("embedding", {})["model"] = env_embed_model

    # OMBRE_EMBEDDING_BASE_URL overrides embedding.base_url
    env_embed_base_url = os.environ.get("OMBRE_EMBEDDING_BASE_URL", "")
    if env_embed_base_url:
        config.setdefault("embedding", {})["base_url"] = env_embed_base_url

    # OMBRE_EMBEDDING_API_KEY lets embeddings use a separate Gemini key.
    # If omitted, embedding_engine falls back to dehydration.api_key for
    # backward compatibility.
    env_embed_api_key = os.environ.get("OMBRE_EMBEDDING_API_KEY", "")
    if env_embed_api_key:
        config.setdefault("embedding", {})["api_key"] = env_embed_api_key

    env_deepseek_low_balance = os.environ.get("OMBRE_DEEPSEEK_LOW_BALANCE_USD", "")
    if env_deepseek_low_balance:
        try:
            config.setdefault("api_usage_guard", {})["deepseek_low_balance_usd"] = float(env_deepseek_low_balance)
        except ValueError:
            logging.warning("OMBRE_DEEPSEEK_LOW_BALANCE_USD is not a valid number; ignored")

    # OMBRE_COMPRESS_TIMEOUT_SECONDS / OMBRE_EMBED_TIMEOUT_SECONDS override the
    # LLM / embedding client timeouts (config keys dehydration.timeout_seconds /
    # embedding.timeout_seconds; defaults 60 / 30).
    for env_name, section, key in (
        ("OMBRE_COMPRESS_TIMEOUT_SECONDS", "dehydration", "timeout_seconds"),
        ("OMBRE_EMBED_TIMEOUT_SECONDS", "embedding", "timeout_seconds"),
    ):
        raw = os.environ.get(env_name, "")
        if raw:
            try:
                config.setdefault(section, {})[key] = float(raw)
            except ValueError:
                logging.warning(f"{env_name} is not a valid number; ignored")

    # --- Ensure bucket storage directories exist ---
    # --- 確保記憶桶存儲目錄存在 ---
    buckets_dir = config["buckets_dir"]
    for subdir in ["permanent", "dynamic", "archive"]:
        os.makedirs(os.path.join(buckets_dir, subdir), exist_ok=True)

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Deep-merge two dicts; override values take precedence.
    深度合併兩個字典，override 的值覆蓋 base。
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def setup_logging(level: str = "INFO") -> None:
    """
    Initialize logging system.
    初始化日誌系統。

    Note: In MCP stdio mode, stdout is occupied by the protocol;
    logs must go to stderr.
    注意：MCP stdio 模式下 stdout 被協議佔用，日誌只能走 stderr。
    """
    log_level = getattr(logging, level.upper(), None)
    if not isinstance(log_level, int):
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],  # StreamHandler defaults to stderr
    )


def generate_bucket_id() -> str:
    """
    Generate a unique bucket ID (12-char short UUID for readability).
    生成唯一的記憶桶 ID（12 位短 UUID，方便人類閱讀）。
    """
    return uuid.uuid4().hex[:12]


def strip_wikilinks(text: str) -> str:
    """
    Remove Obsidian wikilink brackets: [[word]] → word
    去除 Obsidian 雙鏈括號
    """
    return re.sub(r"\[\[([^\]]+)\]\]", r"\1", text) if text else text


def sanitize_name(name: str) -> str:
    """
    Sanitize bucket name, keeping only safe characters.
    Prevents path traversal attacks (e.g. ../../etc/passwd).
    清洗桶名稱，只保留安全字符。防止路徑遍歷攻擊。
    """
    if not isinstance(name, str):
        return "unnamed"
    cleaned = re.sub(r"[^\w\s\u4e00-\u9fff-]", "", name, flags=re.UNICODE)
    cleaned = cleaned.strip()[:80]
    return cleaned if cleaned else "unnamed"


def safe_path(base_dir: str, filename: str) -> Path:
    """
    Construct a safe file path, ensuring it stays within base_dir.
    Prevents directory traversal.
    構造安全的文件路徑，確保最終路徑始終在 base_dir 內部。
    """
    base = Path(base_dir).resolve()
    target = (base / filename).resolve()
    if not str(target).startswith(str(base)):
        raise ValueError(
            f"Path safety check failed / 路徑安全檢查失敗: "
            f"{target} is not inside / 不在 {base} 內"
        )
    return target


def count_tokens_approx(text: str) -> int:
    """
    Rough token count estimate.
    粗略估算 token 數。

    Chinese ≈ 1 char = 1.5 tokens, English ≈ 1 word = 1.3 tokens.
    Used to decide whether dehydration is needed; precision not required.
    中文 ≈ 1字=1.5token，英文 ≈ 1詞=1.3token。
    用於判斷是否需要脫水壓縮，不追求精確。
    """
    if not text:
        return 0
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    english_words = len(re.findall(r"[a-zA-Z]+", text))
    return int(chinese_chars * 1.5 + english_words * 1.3 + len(text) * 0.05)


def now_iso() -> str:
    """
    Return current time as ISO format string.
    返回當前時間的 ISO 格式字符串。
    """
    return datetime.now().isoformat(timespec="seconds")


def parse_bucket_ts(value):
    """Parse a bucket timestamp into naive local datetime (tz-aware → local, tz dropped).
    Returns None on unparseable input.
    把桶時間戳解析成 naive 本地時間（帶時區的先轉本地再去時區），混用也能比較。"""
    try:
        ts = datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None
    if ts.tzinfo is not None:
        ts = ts.astimezone().replace(tzinfo=None)
    return ts


# ---------------------------------------------------------
# Bucket types that sit outside the decay lifecycle.
# 不參與衰減生命週期的桶型。
#
# ONE list, shared by decay_engine.calculate_score (how vivid → archive triage)
# and decay_engine.calculate_heat (how retrievable → injection tier). Kept here
# rather than in either caller because a second copy is not a hypothetical risk:
# on 2026-07-15 decay_engine already carried two representations of this same
# set (five if-branches in calculate_score, a tuple in run_decay_cycle), and
# server.py carried it in three different orders/subsets.
# 一份清單，衰減分與熱度共用。放這裡而不是放在任一呼叫端，是因為「第二份」
# 不是假設性風險：2026-07-15 盤點時 decay_engine 自己就有兩種寫法，
# server.py 有三種順序／子集。
# ---------------------------------------------------------
NON_DECAYING_TYPES = ("permanent", "feel", "plan", "mirage")


def is_decay_exempt(metadata: dict) -> bool:
    """Whether a bucket is outside the decay lifecycle entirely: never scored
    down, never archived, and heat pinned at 1.0 (always injected in full).
    桶是否完全不參與衰減：不掉分、不歸檔、熱度恆為 1.0（永遠全文注入）。"""
    if not isinstance(metadata, dict):
        return False
    return bool(
        metadata.get("pinned")
        or metadata.get("protected")
        or metadata.get("type") in NON_DECAYING_TYPES
    )


def select_importance_tiers(filtered: list, cap: int = 20) -> list:
    """
    Pick up to `cap` buckets from an importance-desc-sorted list, but reserve
    one slot per importance tier first (most recently updated bucket of each
    tier). Without this, 20+ importance-10 buckets crowd out a bucket that was
    just demoted to 9 — trace looks like it "didn't take".
    重要度批量拉取的檔位保留：每個檔位先保一個「最近更新」的席位，
    否則高分桶塞滿上限時，剛被降級的桶會被擠出清單、看似 trace 沒生效。
    """
    def _last_active(b: dict) -> str:
        return str(b["metadata"].get("last_active", b["metadata"].get("created", "")))

    reserved_ids = set()
    tiers_seen = set()
    for b in sorted(filtered, key=_last_active, reverse=True):
        try:
            tier = int(b["metadata"].get("importance", 0) or 0)
        except (ValueError, TypeError):
            tier = 0
        if tier not in tiers_seen:
            tiers_seen.add(tier)
            reserved_ids.add(b["id"])
    reserved = [b for b in filtered if b["id"] in reserved_ids][:cap]
    remaining = [b for b in filtered if b["id"] not in reserved_ids]
    return (reserved + remaining)[:cap]


def clean_llm_json(raw: str) -> str:
    """
    Extract the JSON payload from an LLM reply: strip markdown code fences and
    any chatter before/after the first JSON array or object. Returns a string
    for json.loads(); raises ValueError when no JSON-looking span is found.
    從 LLM 回覆中抽出 JSON：剝掉 code fence 與前後說明文字，
    找不到 JSON 片段時拋 ValueError（讓調用方走既有的解析失敗路徑）。
    """
    if not raw or not raw.strip():
        raise ValueError("empty LLM reply")
    cleaned = raw.strip()

    # Prefer the content of a fenced block when present
    fence = re.search(r"```(?:json)?\s*\n?(.*?)```", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()

    if cleaned.startswith("{") or cleaned.startswith("["):
        return cleaned

    # Chatter around the payload: take the widest span from the first opening
    # bracket to its matching closer's last occurrence.
    starts = [i for i in (cleaned.find("["), cleaned.find("{")) if i != -1]
    if not starts:
        raise ValueError("no JSON payload found in LLM reply")
    start = min(starts)
    closer = "]" if cleaned[start] == "[" else "}"
    end = cleaned.rfind(closer)
    if end <= start:
        raise ValueError("unterminated JSON payload in LLM reply")
    return cleaned[start:end + 1]
