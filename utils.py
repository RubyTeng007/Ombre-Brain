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
