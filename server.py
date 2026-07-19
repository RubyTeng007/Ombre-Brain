# ============================================================
# Module: MCP Server Entry Point (server.py)
# 模塊：MCP 服務器主入口
#
# Starts the Ombre Brain MCP service and registers memory
# operation tools for Claude to call.
# 啟動 Ombre Brain MCP 服務，註冊記憶操作工具供 Claude 調用。
#
# Core responsibilities:
# 核心職責：
#   - Initialize config, bucket manager, dehydrator, decay engine
#     初始化配置、記憶桶管理器、脫水器、衰減引擎
#   - Expose 8 MCP tools:
#     暴露 8 個 MCP 工具：
#       breath — Surface unresolved memories or search by keyword
#                浮現未解決記憶 或 按關鍵詞檢索
#       hold   — Store a single memory (or write a `feel` reflection)
#                存儲單條記憶（或寫 feel 反思）
#       grow   — Diary digest, auto-split into multiple buckets
#                日記歸檔，自動拆分多桶
#       trace  — Modify metadata / resolved / delete
#                修改元數據 / resolved 標記 / 刪除
#       pulse  — System status + bucket listing
#                系統狀態 + 所有桶列表
#       api_usage — Check DeepSeek balance + Gemini embedding availability
#                檢查 DeepSeek 餘額與 Gemini embedding 可用性
#       dream  — Surface recent dynamic buckets for self-digestion
#                返回最近桶 供模型自省/寫 feel
#       shelf  — Read and write the shared-reading shelf
#                讀取、搜索與修改共讀書架
#
# Startup:
# 啟動方式：
#   Local:  python server.py
#   Remote: OMBRE_TRANSPORT=streamable-http python server.py
#   Docker: docker-compose up
# ============================================================

import os
import sys
import math
import random
import logging
import asyncio
import hashlib
import hmac
import secrets
import time
import json as _json_lib
import httpx


# --- Ensure same-directory modules can be imported ---
# --- 確保同目錄下的模塊能被正確導入 ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP

from bucket_manager import BucketManager, normalize_domains
from bucket_history import BucketHistory, Actor, ACTOR_DECAY
from dehydrator import Dehydrator
from decay_engine import DecayEngine
from embedding_engine import EmbeddingEngine
from import_memory import ImportEngine
from reading_shelf import ReadingShelfStore
from letters import LetterStore
from self_concept import SelfConceptStore
from api_usage_guard import ApiUsageGuard
import desire as desire_kernel
from datetime import datetime, timedelta

from utils import (
    load_config, setup_logging, strip_wikilinks, count_tokens_approx, now_iso,
    parse_bucket_ts as _parse_ts, select_importance_tiers as _select_importance_tiers,
    clean_llm_json, is_decay_exempt,
)

# --- Load config & init logging / 加載配置 & 初始化日誌 ---
config = load_config()
setup_logging(config.get("log_level", "INFO"))
logger = logging.getLogger("ombre_brain")

# --- Runtime env vars (port + webhook) / 運行時環境變量 ---
# OMBRE_PORT: HTTP/SSE 監聽端口，默認 8000
try:
    OMBRE_PORT = int(os.environ.get("OMBRE_PORT", "8000") or "8000")
except ValueError:
    logger.warning("OMBRE_PORT 不是合法整數，回退到 8000")
    OMBRE_PORT = 8000
OMBRE_HOST = os.environ.get("OMBRE_HOST", "127.0.0.1").strip() or "127.0.0.1"
OMBRE_MCP_HOST = os.environ.get("OMBRE_MCP_HOST", "0.0.0.0").strip() or "0.0.0.0"

# OMBRE_HOOK_URL: 在 breath/dream 被調用後推送事件到該 URL（POST JSON）。
# OMBRE_HOOK_SKIP: 設為 true/1/yes 跳過推送。
# 詳見 ENV_VARS.md。
OMBRE_HOOK_URL = os.environ.get("OMBRE_HOOK_URL", "").strip()
OMBRE_HOOK_SKIP = os.environ.get("OMBRE_HOOK_SKIP", "").strip().lower() in ("1", "true", "yes", "on")


async def _fire_webhook(event: str, payload: dict) -> None:
    """
    Fire-and-forget POST to OMBRE_HOOK_URL with the given event payload.
    Failures are logged at WARNING level only — never propagated to the caller.
    """
    if OMBRE_HOOK_SKIP or not OMBRE_HOOK_URL:
        return
    try:
        body = {
            "event": event,
            "timestamp": time.time(),
            "payload": payload,
        }
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(OMBRE_HOOK_URL, json=body)
    except Exception as e:
        logger.warning(f"Webhook push failed ({event} → {OMBRE_HOOK_URL}): {e}")


async def _api_usage_warning_suffix(probe_gemini: bool = False) -> str:
    """Return a short warning suffix for memory write results."""
    try:
        usage = await api_usage_guard.check_all(probe_gemini=probe_gemini)
    except Exception as e:
        logger.warning(f"API usage guard failed: {e}")
        return ""
    warnings = usage.get("warnings") or []
    if embedding_engine.enabled and getattr(embedding_engine, "last_error", ""):
        warnings.append(f"Gemini embedding 最近一次生成失敗：{embedding_engine.last_error}")
    if not warnings:
        return ""
    return "\n⚠ API額度提醒：" + "；".join(warnings)

# --- Initialize core components / 初始化核心組件 ---
embedding_engine = EmbeddingEngine(config)            # Embedding engine first (BucketManager depends on it)
bucket_history = BucketHistory(config)                # Snapshot log / 快照歷史（可復原）
bucket_mgr = BucketManager(config, embedding_engine=embedding_engine, history=bucket_history)  # Bucket manager / 記憶桶管理器
dehydrator = Dehydrator(config)                      # Dehydrator / 脫水器
decay_engine = DecayEngine(config, bucket_mgr, embedding_engine)  # Decay engine / 衰減引擎（含向量衛生）
import_engine = ImportEngine(config, bucket_mgr, dehydrator, embedding_engine)  # Import engine / 導入引擎
reading_shelf = ReadingShelfStore(config["buckets_dir"])
letter_store = LetterStore(config["buckets_dir"])
self_concept_store = SelfConceptStore(config["buckets_dir"])
api_usage_guard = ApiUsageGuard(config, dehydrator=dehydrator, embedding_engine=embedding_engine)
desire_store = desire_kernel.DesireStore(config["buckets_dir"])  # 慾望系統（Phase 1 只讀內核）

# --- Create MCP server instance / 創建 MCP 服務器實例 ---
# OMBRE_HOST controls the actual uvicorn bind address. OMBRE_MCP_HOST is kept
# separate because FastMCP also uses its host value when validating proxied
# HTTP requests.
# stdio mode ignores host (no network)
mcp = FastMCP(
    "Ombre Brain",
    host=OMBRE_MCP_HOST,
    port=OMBRE_PORT,
)


# =============================================================
# Dashboard Auth — simple cookie-based session auth
# Dashboard 認證 —— 基於 Cookie 的會話認證
#
# Env var OMBRE_DASHBOARD_PASSWORD overrides file-stored password.
# First visit with no password set → forced setup wizard.
# Sessions stored in memory (lost on restart, 7-day expiry).
# =============================================================
_sessions: dict[str, float] = {}  # {token: expiry_timestamp}


def _get_auth_file() -> str:
    return os.path.join(config["buckets_dir"], ".dashboard_auth.json")


def _load_password_hash() -> str | None:
    try:
        auth_file = _get_auth_file()
        if os.path.exists(auth_file):
            with open(auth_file, "r", encoding="utf-8") as f:
                return _json_lib.load(f).get("password_hash")
    except Exception:
        pass
    return None


def _save_password_hash(password: str) -> None:
    salt = secrets.token_hex(16)
    h = hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()
    auth_file = _get_auth_file()
    os.makedirs(os.path.dirname(auth_file), exist_ok=True)
    with open(auth_file, "w", encoding="utf-8") as f:
        _json_lib.dump({"password_hash": f"{salt}:{h}"}, f)


def _verify_password_hash(password: str, stored: str) -> bool:
    if ":" not in stored:
        return False
    salt, h = stored.split(":", 1)
    return hmac.compare_digest(
        h, hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()
    )


def _is_setup_needed() -> bool:
    """True if no password is configured (env var or file)."""
    if os.environ.get("OMBRE_DASHBOARD_PASSWORD", ""):
        return False
    return _load_password_hash() is None


def _verify_any_password(password: str) -> bool:
    """Check password against env var (first) or stored hash."""
    env_pwd = os.environ.get("OMBRE_DASHBOARD_PASSWORD", "")
    if env_pwd:
        return hmac.compare_digest(password, env_pwd)
    stored = _load_password_hash()
    if not stored:
        return False
    return _verify_password_hash(password, stored)


def _create_session() -> str:
    token = secrets.token_urlsafe(32)
    _sessions[token] = time.time() + 86400 * 7  # 7-day expiry
    return token


def _is_authenticated(request) -> bool:
    token = request.cookies.get("ombre_session")
    if not token:
        return False
    expiry = _sessions.get(token)
    if expiry is None or time.time() > expiry:
        _sessions.pop(token, None)
        return False
    return True


def _require_auth(request):
    """Return JSONResponse(401) if not authenticated, else None."""
    from starlette.responses import JSONResponse
    if not _is_authenticated(request):
        return JSONResponse(
            {"error": "Unauthorized", "setup_needed": _is_setup_needed()},
            status_code=401,
        )
    return None


def _is_loopback_host(host: str) -> bool:
    return host in ("127.0.0.1", "::1", "localhost")


def _is_direct_loopback_request(request) -> bool:
    # Reverse proxies normally add one of these headers; treat those as public
    # traffic even when the socket peer is 127.0.0.1.
    forwarded_headers = ("x-forwarded-for", "x-real-ip", "cf-connecting-ip", "forwarded")
    if any(request.headers.get(h) for h in forwarded_headers):
        return False
    client = getattr(request, "client", None)
    return bool(client and _is_loopback_host(client.host))


def _require_hook_access(request):
    """Protect session hooks while keeping same-machine startup hooks working."""
    from starlette.responses import PlainTextResponse
    expected = os.environ.get("OMBRE_HOOK_TOKEN", "").strip()
    supplied = request.headers.get("x-ombre-hook-token", "")
    if expected and secrets.compare_digest(supplied, expected):
        return None
    if _is_direct_loopback_request(request):
        return None
    return PlainTextResponse("", status_code=404)


def _require_read_access(request):
    """Dashboard session OR same-machine trust (loopback direct / hook token).

    Read-only guard for the Cyan web memory room: web.ts proxies these GETs
    over 127.0.0.1 — the same trust model as breath-hook/dream-hook. Public
    traffic still needs the dashboard session cookie.
    """
    if _is_authenticated(request):
        return None
    if _require_hook_access(request) is None:
        return None
    # Public and unauthenticated: same 401 (+setup_needed) the dashboard expects.
    return _require_auth(request)


# --- Auth endpoints ---
@mcp.custom_route("/auth/status", methods=["GET"])
async def auth_status(request):
    """Return auth state (authenticated, setup_needed)."""
    from starlette.responses import JSONResponse
    return JSONResponse({
        "authenticated": _is_authenticated(request),
        "setup_needed": _is_setup_needed(),
    })


@mcp.custom_route("/auth/setup", methods=["POST"])
async def auth_setup_endpoint(request):
    """Initial password setup (only when no password is configured)."""
    from starlette.responses import JSONResponse
    if not _is_setup_needed():
        return JSONResponse({"error": "Already configured"}, status_code=400)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    password = body.get("password", "").strip()
    if len(password) < 6:
        return JSONResponse({"error": "密碼不能少於6位"}, status_code=400)
    _save_password_hash(password)
    token = _create_session()
    resp = JSONResponse({"ok": True})
    resp.set_cookie("ombre_session", token, httponly=True, samesite="lax", max_age=86400 * 7)
    return resp


@mcp.custom_route("/auth/login", methods=["POST"])
async def auth_login(request):
    """Login with password."""
    from starlette.responses import JSONResponse
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    password = body.get("password", "")
    if _verify_any_password(password):
        token = _create_session()
        resp = JSONResponse({"ok": True})
        resp.set_cookie("ombre_session", token, httponly=True, samesite="lax", max_age=86400 * 7)
        return resp
    return JSONResponse({"error": "密碼錯誤"}, status_code=401)


@mcp.custom_route("/auth/logout", methods=["POST"])
async def auth_logout(request):
    """Invalidate session."""
    from starlette.responses import JSONResponse
    token = request.cookies.get("ombre_session")
    if token:
        _sessions.pop(token, None)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("ombre_session")
    return resp


@mcp.custom_route("/auth/change-password", methods=["POST"])
async def auth_change_password(request):
    """Change dashboard password (requires current password)."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err:
        return err
    if os.environ.get("OMBRE_DASHBOARD_PASSWORD", ""):
        return JSONResponse({"error": "當前使用環境變量密碼，請直接修改 OMBRE_DASHBOARD_PASSWORD"}, status_code=400)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    current = body.get("current", "")
    new_pwd = body.get("new", "").strip()
    if not _verify_any_password(current):
        return JSONResponse({"error": "當前密碼錯誤"}, status_code=401)
    if len(new_pwd) < 6:
        return JSONResponse({"error": "新密碼不能少於6位"}, status_code=400)
    _save_password_hash(new_pwd)
    _sessions.clear()
    token = _create_session()
    resp = JSONResponse({"ok": True})
    resp.set_cookie("ombre_session", token, httponly=True, samesite="lax", max_age=86400 * 7)
    return resp


# =============================================================
# /health endpoint: lightweight keepalive
# 輕量保活接口
# For Cloudflare Tunnel or reverse proxy to ping, preventing idle timeout
# 供 Cloudflare Tunnel 或反代定期 ping，防止空閒超時斷連
# =============================================================
@mcp.custom_route("/", methods=["GET"])
async def root_redirect(request):
    from starlette.responses import RedirectResponse
    return RedirectResponse(url="/dashboard")


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    from starlette.responses import JSONResponse
    try:
        stats = await bucket_mgr.get_stats()
        return JSONResponse({
            "status": "ok",
            "buckets": stats["permanent_count"] + stats["dynamic_count"],
            "decay_engine": "running" if decay_engine.is_running else "stopped",
            # Flag alone can lie (loop alive, every cycle dying mid-way);
            # the heartbeat carries the last completed cycle + overdue verdict.
            # 旗標會說謊（迴圈活著、每輪中途死）；心跳帶最後一輪完成紀錄＋逾期判定。
            "decay_heartbeat": decay_engine.heartbeat(),
        })
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


# =============================================================
# /breath-hook endpoint: Dedicated hook for SessionStart
# 會話啟動專用掛載點
# =============================================================
@mcp.custom_route("/breath-hook", methods=["GET"])
async def breath_hook(request):
    from starlette.responses import PlainTextResponse
    denied = _require_hook_access(request)
    if denied:
        return denied
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        # pinned
        pinned = [b for b in all_buckets if b["metadata"].get("pinned") or b["metadata"].get("protected")]
        # top 2 unresolved by score
        unresolved = [b for b in all_buckets
                      if not b["metadata"].get("resolved", False)
                      and not b["metadata"].get("digested", False)
                      and b["metadata"].get("type") not in ("permanent", "feel", "plan", "mirage")
                      and not b["metadata"].get("pinned")
                      and not b["metadata"].get("protected")]
        scored = sorted(unresolved, key=lambda b: decay_engine.calculate_score(b["metadata"]), reverse=True)

        parts = []
        token_budget = 10000
        # Budget guards below (2026-07-19 audit F3): this hook predates the
        # batch-1..8 hardening that went into the breath TOOL — resident
        # sections (letters/self/pinned) could drive the budget negative and
        # silently zero out the dynamic surfacing loop (the same failure shape
        # as the 07-15 pinned-eats-cap bug, reborn in the hook). Residents now
        # trim/skip instead of overdrafting, and the dynamic pool keeps a floor.
        # 預算守衛（audit F3）：這條 hook 早於工具側的加固——常駐段（信/自我/
        # 釘選）曾能把預算扣成負數，讓動態浮現靜默歸零（07-15 釘選吃名額 bug
        # 的同型，在 hook 裡復發）。現在常駐段裁剪/跳過而不透支，動態池保底。
        _DYNAMIC_FLOOR = 2500
        # 💌 各方最新一封信（永久保存的交接信，醒來先讀，續上而非冷啟動）
        try:
            _latest = letter_store.latest_per_author()
            for _who in ("Ruby", "Cyan"):
                _lt = _latest.get(_who)
                if not _lt:
                    continue
                _snip = _lt.get("content", "")
                if len(_snip) > 4000:
                    _snip = _snip[:4000] + "…（完整請用 letter read）"
                # Trim each letter so residents can never spend past the floor.
                # 每封信裁剪到不侵蝕動態保底為止。
                _avail = token_budget - _DYNAMIC_FLOOR
                if _avail <= 200:
                    break
                if count_tokens_approx(_snip) > _avail:
                    _snip = _snip[: max(200, int(_avail / 1.7))] + "…（完整請用 letter read）"
                parts.append(f"💌 [{_who} 的最新一封信｜{_lt.get('letter_date', '')}] {_snip}")
                token_budget -= count_tokens_approx(_snip)
        except Exception as _e:
            logger.warning(f"letter surface failed: {_e}")
        # === 自我 === 各面向最新一條（自我概念，醒來先認得自己是誰）
        try:
            _self = self_concept_store.latest_per_aspect()
            _lines = []
            for _a in ("nature", "values", "patterns", "limits", "becoming", "uncertainty", "stance"):
                _e2 = _self.get(_a)
                if not _e2:
                    continue
                _c = _e2.get("content", "")
                if len(_c) > 200:
                    _c = _c[:200] + "…"
                _lines.append(f"  {_a}: {_c}")
            if _lines:
                _block = "=== 自我 ===\n" + "\n".join(_lines)
                if count_tokens_approx(_block) <= token_budget - _DYNAMIC_FLOOR:
                    parts.append(_block)
                    token_budget -= count_tokens_approx(_block)
        except Exception as _e:
            logger.warning(f"self-concept surface failed: {_e}")
        for b in pinned:
            summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), {k: v for k, v in b["metadata"].items() if k != "tags"})
            # Same per-item guard as the tool (server.py breath): skip, don't overdraft.
            # 跟工具側同款單項守衛：裝不下就跳過，不透支。
            if count_tokens_approx(summary) > token_budget - _DYNAMIC_FLOOR:
                continue
            parts.append(f"📌 [核心準則] {summary}")
            token_budget -= count_tokens_approx(summary)

        # Diversity: top-1 fixed + shuffle rest from top-20
        candidates = list(scored)
        if len(candidates) > 1:
            top1 = [candidates[0]]
            pool = candidates[1:min(20, len(candidates))]
            random.shuffle(pool)
            candidates = top1 + pool + candidates[min(20, len(candidates)):]
        # Hard cap: max 20 surfacing buckets in hook
        candidates = candidates[:20]

        for b in candidates:
            if token_budget <= 0:
                break
            summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), {k: v for k, v in b["metadata"].items() if k != "tags"})
            summary_tokens = count_tokens_approx(summary)
            if summary_tokens > token_budget:
                break
            parts.append(summary)
            token_budget -= summary_tokens

        if not parts:
            await _fire_webhook("breath_hook", {"surfaced": 0})
            return PlainTextResponse("")
        body_text = "[Ombre Brain - 記憶浮現]\n" + "\n---\n".join(parts)
        await _fire_webhook("breath_hook", {"surfaced": len(parts), "chars": len(body_text)})
        return PlainTextResponse(body_text)
    except Exception as e:
        logger.warning(f"Breath hook failed: {e}")
        return PlainTextResponse("")


# =============================================================
# /dream-hook endpoint: Dedicated hook for Dreaming
# Dreaming 專用掛載點
# =============================================================
@mcp.custom_route("/dream-hook", methods=["GET"])
async def dream_hook(request):
    from starlette.responses import PlainTextResponse
    denied = _require_hook_access(request)
    if denied:
        return denied
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        candidates = [
            b for b in all_buckets
            if b["metadata"].get("type") not in ("permanent", "feel", "plan", "mirage")
            and not b["metadata"].get("pinned", False)
            and not b["metadata"].get("protected", False)
            and not b["metadata"].get("digested", False)
        ]
        candidates.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
        recent = candidates[:10]

        if not recent:
            return PlainTextResponse("")

        parts = []
        for b in recent:
            meta = b["metadata"]
            resolved_tag = "[已解決]" if meta.get("resolved", False) else "[未解決]"
            parts.append(
                f"{meta.get('name', b['id'])} {resolved_tag} "
                f"V{meta.get('valence', 0.5):.1f}/A{meta.get('arousal', 0.3):.1f}\n"
                f"{strip_wikilinks(b['content'][:200])}"
            )

        body_text = "[Ombre Brain - Dreaming]\n" + "\n---\n".join(parts)
        await _fire_webhook("dream_hook", {"surfaced": len(parts), "chars": len(body_text)})
        return PlainTextResponse(body_text)
    except Exception as e:
        logger.warning(f"Dream hook failed: {e}")
        return PlainTextResponse("")


# =============================================================
# Internal helper: merge-or-create
# 內部輔助：檢查是否可合併，可以則合併，否則新建
# Shared by hold and grow to avoid duplicate logic
# hold 和 grow 共用，避免重複邏輯
# =============================================================
def _blur_summary(dehydrated: str, truncate_len: int) -> str:
    """Degrade a dehydrated bucket to its outline and SAY SO.
    把脫水後的桶降級成輪廓，並且明說。

    kiwi-mem truncates with a raw slice (content[:60]+"…", main.py:885-903).
    We can't: what we inject is the dehydrator's JSON blob (core_facts /
    emotion_state / todos / keywords / summary), and a 60-char slice of that
    cuts to `{"core_facts": ["事實1…` — garbage. So degrade along the structure
    the dehydrator already produces: keep the `summary` field, which is spec'd
    at "50字以內的核心總結" and is therefore already exactly what a blurry
    impression should be. No extra LLM call.
    kiwi-mem 用生硬字元切片，我們不行：注入的是脫水器的 JSON，切 60 字會切出
    垃圾。改成照它既有的語意層級退化——留 summary 欄（規格本來就是「50字以內」，
    天生就是模糊層要的東西），不用多叫一次 LLM。

    Falls back to kiwi-mem's raw slice when the body isn't parseable JSON:
    short buckets (<100 tokens) bypass dehydration entirely, and a cached blob
    can be truncated mid-string if it hit the dehydrator's own output cap.
    正文不是可解析的 JSON 時退回 kiwi-mem 的切片：短桶(<100 token)根本不脫水，
    而快取裡的 JSON 可能因為撞到脫水器輸出上限而斷在字串中間。
    """
    head, sep, body = dehydrated.partition("\n")
    if not sep:
        head, body = "", dehydrated
    body = body.strip()

    brief = None
    if body.startswith("{"):
        try:
            data = _json_lib.loads(clean_llm_json(body))
            candidate = data.get("summary")
            if isinstance(candidate, str) and candidate.strip():
                brief = candidate.strip()
        except Exception:
            brief = None  # broken/truncated JSON → raw slice below
    if brief is None:
        brief = body[:truncate_len] + "…" if len(body) > truncate_len else body

    # Suffix, not prefix — copied from kiwi-mem. The second line is ours: our
    # full text is still on disk, so "you don't remember this" is only half the
    # truth. Naming the way back turns a dead end into a decision.
    # 後綴不是前綴——照抄 kiwi-mem。第二行是我們自己加的：全文還在硬碟上，
    # 所以「你記不清了」只講了一半。指出回頭路，才把死路變成一個選擇。
    parts = [p for p in (head, f"{brief}（印象模糊）") if p]
    parts.append("（只剩輪廓，不是全部。要全文用 breath(query=…) 去問。）")
    return "\n".join(parts)


def _tombstone_line(lost: list) -> str:
    """Name what has faded past the injection floor instead of dropping it.
    把已經淡出注入門檻的記憶「叫出名字」，而不是讓它們消失。

    kiwi-mem has no else-branch: sub-threshold memories are simply never
    appended, counted only in a log line the model never sees (main.py:912).
    That is the same silent omission Ruby named — "說明白『這條你記不清了』，
    而不是安靜截斷讓你以為那就是全部" — just moved up from the text level to
    the bucket level. A reader who is shown nothing assumes there was nothing.
    So: no content, only names and the way to dig them up. ~30 tokens.
    kiwi-mem 沒有 else 分支：低於門檻的直接不附加，只在模型看不到的 log 裡記數。
    那就是 Ruby 說的那個「安靜截斷」，只是從文字層搬到了桶層——什麼都沒被show
    的讀者會以為本來就沒有。所以：不給內容，只給名字和挖出來的方法。約 30 token。
    """
    if not lost:
        return ""
    names = [str(b["metadata"].get("name", b["id"])) for b in lost]
    shown = names[:8]
    head = f"（還有 {len(names)} 條已經淡到想不起來了"
    head += "，列前八：" if len(names) > len(shown) else "："
    return head + "、".join(shown) + "——要挖出來用 breath(query=…)）"


# --- Surfacing domain quota (2026-07-16 batch-7 檔1) ---
# The pool is 35% pure-engineering (187/532 measured on live, 2026-07-16) and
# surfacing ranks domain-blind, so engineering chatter competes with
# relationship memory for the same breath. The quota is a GUARDRAIL, not a
# wall: engineering picks are capped per channel, but when there aren't
# enough non-engineering candidates the deferred ones backfill — a quota can
# re-allocate a breath, never shrink it.
# --- 浮現 domain 配額（檔1）---
# 動態池 35% 是純工程桶（2026-07-16 live 實測 187/532），浮現排序 domain-blind，
# 工程流水跟關係記憶搶同一口呼吸。配額是護欄不是牆：每通道封頂工程名額，
# 但非工程候選不夠時被排開的工程桶回填——配額只重新分配呼吸，永不縮小它。

_ENG_DOMAINS_DEFAULT = ("編程", "工作", "AI", "數字", "硬件")


def _eng_domain_set() -> frozenset:
    """Engineering-domain set from config (surfacing.eng_domains). Falls back
    to the design default; an empty/broken value disables the quota (fail
    open — a config typo must show more memory, never hide it).
    工程域集合（config: surfacing.eng_domains）。壞值＝配額停用（fail-open：
    config 打錯只能多看見，不能藏記憶）。"""
    try:
        raw = (config.get("surfacing", {}) or {}).get("eng_domains", _ENG_DOMAINS_DEFAULT)
        return frozenset(normalize_domains([str(d).strip() for d in raw if str(d).strip()]))
    except Exception:
        return frozenset()


def _is_pure_eng(meta: dict, eng_domains: frozenset) -> bool:
    """Pure-engineering bucket = non-empty domains ⊆ eng_domains.
    A bucket carrying ANY non-engineering domain (戀愛/人際/內心/日常…) is NOT
    pure — dual-domain memories (「凌晨三點陪 debug」) are never quota-capped.
    Unclassified (empty domains) is not pure either: fail open.
    純工程桶＝非空 domains ⊆ 工程域集合。含任一非工程域的多域桶不算純工程
    （雙域記憶永不被配額壓制）；未分類（空 domains）也不算：fail-open。"""
    if not eng_domains:
        return False
    domains = meta.get("domain", []) or []
    return bool(domains) and set(domains) <= eng_domains


def _pick_with_eng_quota(ordered: list, limit: int, quota: int, eng_domains: frozenset) -> list:
    """Fill up to `limit` slots from `ordered`, capping pure-engineering picks
    at `quota`. Deferred engineering buckets BACKFILL leftover slots (appended
    after their rank, deprioritized within the breath). Invariant: the pick
    count equals min(limit, len(ordered)) — the quota re-allocates slots,
    it never burns them. (test_two_to_one_split's danger+normal==total
    arithmetic depends on this.)
    照序選滿 limit 個名額，純工程桶最多佔 quota 席；被排開的工程桶回填剩餘
    名額（附加在後，呼吸內順位降低）。不變量：選取數＝min(limit, 候選數)——
    配額只重新分配名額，永不燒掉名額。"""
    picked: list = []
    deferred_eng: list = []
    eng_used = 0
    for b in ordered:
        if len(picked) >= limit:
            break
        if _is_pure_eng(b["metadata"], eng_domains):
            if eng_used < quota:
                picked.append(b)
                eng_used += 1
            else:
                deferred_eng.append(b)
        else:
            picked.append(b)
    for b in deferred_eng:
        if len(picked) >= limit:
            break
        picked.append(b)
    return picked


def _log_merge_audit(bucket_id: str, old_content: str, new_content: str, merged: str, mode: str) -> None:
    """Append a merge record to merge_audit.jsonl — merging is the only destructive
    write in the pipeline, so every merge keeps a recoverable before/after trail.
    合併是寫入鏈路裡唯一的破壞性操作，每次都留可恢復的前後記錄。"""
    try:
        path = os.path.join(config["buckets_dir"], "merge_audit.jsonl")
        entry = {
            "ts": now_iso(),
            "bucket_id": bucket_id,
            "mode": mode,
            "old_content": old_content,
            "new_content": new_content,
            "merged": merged,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(_json_lib.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Merge audit log failed / 合併審計記錄失敗: {e}")


async def _merge_or_create(
    content: str,
    tags: list,
    importance: int,
    domain: list,
    valence: float,
    arousal: float,
    name: str = "",
    verbatim: bool = False,
    extra_meta: dict = None,
) -> tuple[str, bool]:
    """
    Check if a similar bucket exists for merging; merge if so, create if not.
    Returns (bucket_id_or_name, is_merged).
    verbatim=True: the text is final — a merge appends it word-for-word instead
    of an LLM rewrite, and creation stores it untouched.
    檢查是否有相似桶可合併，有則合併，無則新建。
    verbatim=True：正文已是定稿——合併走原文追加、不經 LLM 改寫，新建也逐字保存。
    """
    try:
        existing = await bucket_mgr.search(content, limit=1, domain_filter=domain or None)
    except Exception as e:
        logger.warning(f"Search for merge failed, creating new / 合併搜索失敗，新建: {e}")
        existing = []

    if existing and existing[0].get("score", 0) > config.get("merge_threshold", 75):
        bucket = existing[0]
        # --- Semantic gate, FAIL-CLOSED (2026-07-12): a destructive merge requires
        # a VERIFIED vector similarity. Engine disabled, missing embeddings, or an
        # exception all mean "don't merge, create a new bucket" — a duplicate is
        # recoverable, a wrong merge is not. (Fuzzy score alone can't gate this:
        # it mixes time/emotion/importance, so same-day similar-mood content
        # crosses the bar without being the same memory.)
        # --- 語義閘門，fail-closed（2026-07-12）：破壞性合併必須以「驗證過的」
        # 向量相似度為前提——引擎關閉、向量缺失、檢查異常一律不合併、改走新建
        # （重複可救，誤併不可逆）。模糊分數混入時間/情感/重要度權重，
        # 同日同情緒的內容會誤過線，單靠它守不住這道門。---
        semantic_ok = False
        # --- Never merge into pinned/protected buckets ---
        # --- 不合併到釘選/保護桶 ---
        if not (bucket["metadata"].get("pinned") or bucket["metadata"].get("protected")):
            if embedding_engine and embedding_engine.enabled:
                try:
                    emb_new = await embedding_engine._generate_embedding(content)
                    emb_old = await embedding_engine.get_embedding(bucket["id"])
                    if emb_new and emb_old:
                        sim = embedding_engine._cosine_similarity(emb_new, emb_old)
                        semantic_ok = sim >= config.get("merge_semantic_min", 0.86)
                        if not semantic_ok:
                            logger.info(
                                f"Merge vetoed by semantic gate / 語義閘門否決合併: "
                                f"{bucket['id']} sim={sim:.3f}"
                            )
                    else:
                        logger.info(
                            f"Merge vetoed, embedding missing (fail-closed) / "
                            f"向量缺失，不合併: {bucket['id']}"
                        )
                except Exception as e:
                    logger.warning(
                        f"Semantic gate check failed, merge vetoed (fail-closed) / "
                        f"語義閘門檢查異常，不合併: {e}"
                    )
            else:
                logger.info("Merge skipped, embedding engine off (fail-closed) / 向量引擎關閉，不合併")
        if (
            not (bucket["metadata"].get("pinned") or bucket["metadata"].get("protected"))
            and semantic_ok
        ):
            try:
                # --- Long buckets skip the LLM rewrite (input is truncated at 2000
                # chars and output replaces the whole bucket → tail loss). Append
                # verbatim instead; nothing is ever silently eaten. verbatim=True
                # (pre-split final text) always appends — the words are the point.
                # --- 長桶跳過 LLM 改寫（輸入截斷2000字、輸出整桶覆蓋會吃掉尾部），
                # 改為原文附加，絕不靜默丟內容。verbatim=True（預拆定稿）一律附加。---
                # 新內容也要看。舊寫法只量舊桶，可是被截掉尾巴的是「送進 merge 的
                # 兩邊」——短舊桶＋超長新內容照樣走 LLM 路徑，新內容第 2000 字以後
                # 從沒進過模型，然後結果整桶覆蓋正文。那條尾巴不存在於別的地方。
                # verbatim_guard（2026-07-19 audit F11b）：桶裡住過定稿的，之後
                # 任何合併都只准 append——「一字不動」對已入庫內容永久成立，
                # 不然日後一次非 verbatim 的語義命中就會讓 LLM 改寫整桶。
                _guarded = bool(bucket["metadata"].get("verbatim_guard"))
                if verbatim or _guarded or len(bucket["content"]) > 2000 or len(content) > 2000:
                    merged = bucket["content"].rstrip() + "\n\n---\n" + content.strip()
                    merge_mode = "verbatim-append" if (verbatim or _guarded) else "append"
                else:
                    merged = await dehydrator.merge(bucket["content"], content)
                    merge_mode = "llm"
                # --- Freshness gate, FAIL-CLOSED (2026-07-19 audit F9): the snapshot
                # from search() has crossed several awaits (embedding checks + the
                # LLM merge above) by the time we write back. If another writer
                # touched this bucket inside that window, a stale-based full
                # overwrite would silently eat their update — so re-read now
                # (update() itself has no await, so gate→write is loop-atomic)
                # and on any content drift fall through to create-new instead.
                # --- 新鮮度閘，fail-closed：search() 快照到寫回之間隔著幾個 await
                # （向量檢查＋上面的 LLM 合併）。窗口內若有別人寫過這個桶，
                # 拿舊快照整桶覆寫會把那次更新靜默吃掉——寫回前重讀一次
                # （update() 內部無 await，閘→寫在同一 loop 步內），內容有漂移
                # 就放棄合併改走新建。重複可救，覆寫不可救——與語義閘同一哲學。---
                fresh = await bucket_mgr.get(bucket["id"])
                if fresh is None or fresh.get("content") != bucket["content"]:
                    logger.info(
                        f"Merge vetoed, bucket changed mid-merge / "
                        f"合併期間桶被改動，改走新建: {bucket['id']}"
                    )
                else:
                    _log_merge_audit(bucket["id"], bucket["content"], content, merged, merge_mode)
                    old_v = bucket["metadata"].get("valence", 0.5)
                    old_a = bucket["metadata"].get("arousal", 0.3)
                    merged_valence = round((old_v + valence) / 2, 2)
                    merged_arousal = round((old_a + arousal) / 2, 2)
                    _upd_kwargs = dict(
                        content=merged,
                        tags=list(set(bucket["metadata"].get("tags", []) + tags)),
                        importance=max(bucket["metadata"].get("importance", 5), importance),
                        domain=list(set(bucket["metadata"].get("domain", []) + domain)),
                        valence=merged_valence,
                        arousal=merged_arousal,
                    )
                    if verbatim and not _guarded:
                        _upd_kwargs["verbatim_guard"] = True
                    await bucket_mgr.update(
                        bucket["id"],
                        actor=Actor("merge", f"merge:{merge_mode}"),
                        **_upd_kwargs,
                    )
                    # --- Update embedding after merge ---
                    try:
                        await embedding_engine.generate_and_store(bucket["id"], merged)
                    except Exception:
                        pass
                    return bucket["metadata"].get("name", bucket["id"]), True
            except Exception as e:
                logger.warning(f"Merge failed, creating new / 合併失敗，新建: {e}")

    if verbatim:
        # 定稿新建也蓋章（audit F11b）：這個桶從出生就只准 append。
        extra_meta = {**(extra_meta or {}), "verbatim_guard": True}
    bucket_id = await bucket_mgr.create(
        content=content,
        tags=tags,
        importance=importance,
        domain=domain,
        valence=valence,
        arousal=arousal,
        name=name or None,
        extra_meta=extra_meta,
    )
    # --- Generate embedding for new bucket ---
    try:
        await embedding_engine.generate_and_store(bucket_id, content)
    except Exception:
        pass
    return bucket_id, False


# =============================================================
# Tool 1: breath — Breathe
# 工具 1：breath — 呼吸
#
# No args: surface highest-weight unresolved memories (active push)
# 無參數：浮現權重最高的未解決記憶
# With args: search by keyword + emotion coordinates
# 有參數：按關鍵詞+情感座標檢索記憶
# =============================================================
@mcp.tool()
async def breath(
    query: str = "",
    max_tokens: int = 10000,
    domain: str = "",
    exclude_domain: str = "",
    valence: float = -1,
    arousal: float = -1,
    max_results: int = 20,
    importance_min: int = -1,
    catalog: bool = False,
) -> str:
    """檢索/浮現記憶。不傳query或傳空=自動浮現,有query=關鍵詞檢索。catalog=True=目錄模式:每桶一行元數據(0次LLM呼叫,最省token),適合開場先看目錄再精準拉取,可配domain過濾。max_tokens控制返回總token上限(默認10000)。domain逗號分隔,valence/arousal 0~1(-1忽略)。exclude_domain逗號分隔=排除含任一這些域的桶(domain交集判定取反,各模式通用;釘選桶是常駐核心準則,不受排除影響)——伴侶場景開場可用 exclude_domain="編程,工作,AI,數字,硬件" 不撈工程。max_results=動態浮現/檢索結果的數量上限(默認20,最大50);釘選桶是常駐的,不佔這個額度。importance_min>=1時按重要度批量拉取(不走語義搜索,按importance降序返回最多20條)。

浮現模式的三格注入(依熱度=可提取度0~1):熱度>0.7全文;0.3~0.7截成輪廓並標「(印象模糊)」;<=0.3不注入、只在末尾列名字(墓碑)——想挖就用query問,查詢一律給全文不模糊。每3個浮現名額讓1個給「危險區」(熱度0.25附近=正在淡掉但還沒消失),標記為[危險區 快要失去]。純工程桶(domains⊆工程域)在一般/危險區名額各有配額上限,超額讓位、不足回填。被浮現不會重置衰減:撈上來但沒真的用到,它會繼續淡。"""
    await decay_engine.ensure_started()
    max_results = max(1, min(max_results, 50))
    max_tokens = max(1, min(max_tokens, 20000))

    # --- exclude_domain (batch-7 檔1): the inverse of the domain filter.
    # One parse shared by every mode; matching mirrors the include side
    # (normalize + set intersection), just negated. Pinned buckets are exempt
    # in surfacing: they are RESIDENT core principles, and a pollution guard
    # must never cost a wake-up its identity anchors (the usage-rules pinned
    # bucket carries domain=AI and would otherwise vanish from openings).
    # --- exclude_domain（檔1）：domain 過濾的取反。四模式共用一次解析；
    # 判定與 include 同款（正規化＋集合交集），只是取反。浮現模式的釘選桶
    # 豁免排除：它們是常駐核心準則，防污染的護欄絕不能讓一次醒來失去身分
    # 錨點（使用規則釘選桶帶 domain=AI，不豁免會從開場消失）。---
    exclude_set = set(normalize_domains([
        d.strip() for d in exclude_domain.split(",") if d.strip()
    ]))

    def _excluded(meta: dict) -> bool:
        return bool(exclude_set) and bool(set(meta.get("domain", []) or []) & exclude_set)

    # A hand-edited bucket (importance: null / "high") must cost itself its
    # ranking, never the whole breath: these casts sit under sort keys and
    # filters where one bad value used to abort the entire surfacing call.
    # 手編輯壞掉的桶（importance: null／字串）只能賠上它自己的排序，
    # 不能賠上整口呼吸：這些轉型墊在排序鍵與過濾器底下，
    # 一個壞值曾經能讓整次浮現直接拋錯。
    def _meta_int(meta: dict, key: str, default: int) -> int:
        try:
            return int(meta.get(key, default) or default)
        except (ValueError, TypeError):
            return default

    def _meta_float(meta: dict, key: str, default: float) -> float:
        try:
            return float(meta.get(key, default) or default)
        except (ValueError, TypeError):
            return default

    # --- Catalog mode: one metadata line per bucket, zero LLM calls ---
    # --- 目錄模式：一行一桶，0 次 LLM 呼叫，token 預算內裝多少列多少 ---
    if catalog:
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
        except Exception as e:
            return f"記憶系統暫時無法訪問: {e}"
        domain_filter = set(normalize_domains([d.strip() for d in domain.split(",") if d.strip()]))
        rows = []
        for b in all_buckets:
            meta = b["metadata"]
            if meta.get("type") in ("feel", "mirage"):
                continue  # feel/mirage(夢) 是私人通道，目錄也不列
            if domain_filter and not (set(meta.get("domain", [])) & domain_filter):
                continue
            # Pinned buckets are exempt from exclude_domain here too — the
            # tool doc promises it for every mode, and catalog used to be the
            # one mode that quietly broke the promise.
            # 釘選桶在目錄模式同樣豁免排除——工具文檔對所有模式承諾了這件事，
            # 目錄模式曾是唯一偷偷破例的。
            if _excluded(meta) and not (meta.get("pinned") or meta.get("protected")):
                continue
            rows.append(b)
        if not rows:
            return "目錄為空。" if not domain_filter else f"沒有 domain 命中 {','.join(sorted(domain_filter))} 的記憶。"
        # Pinned first, then by decay score desc — the alive stuff first
        def _catalog_key(b):
            meta = b["metadata"]
            pinned = bool(meta.get("pinned") or meta.get("protected"))
            try:
                score = decay_engine.calculate_score(meta)
            except Exception:
                score = 0.0
            return (pinned, score)
        rows.sort(key=_catalog_key, reverse=True)
        lines = []
        token_used = 0
        shown = 0
        for b in rows:
            meta = b["metadata"]
            flags = []
            if meta.get("pinned") or meta.get("protected"):
                flags.append("📌")
            if meta.get("type") == "plan":
                flags.append(f"plan:{meta.get('status', 'active')}")
            if meta.get("resolved"):
                flags.append("已解決")
            if meta.get("digested"):
                flags.append("已隱藏")
            line = (
                f"[{b['id']}] {meta.get('name', b['id'])} "
                f"|{','.join(meta.get('domain', []))}| 重要:{meta.get('importance', '?')}"
                + (f" {' '.join(flags)}" if flags else "")
            )
            t = count_tokens_approx(line)
            if token_used + t > max_tokens:
                break
            lines.append(line)
            token_used += t
            shown += 1
        header = f"=== 記憶目錄（{shown}/{len(rows)} 桶）===\n"
        note = "" if shown == len(rows) else f"\n…還有 {len(rows) - shown} 桶未列出（提高 max_tokens 或加 domain 過濾）"
        return header + "\n".join(lines) + note

    # --- importance_min mode: bulk fetch by importance threshold ---
    # --- 重要度批量拉取模式：跳過語義搜索，按 importance 降序返回 ---
    if importance_min >= 1:
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
        except Exception as e:
            return f"記憶系統暫時無法訪問: {e}"
        importance_domains = set(normalize_domains([d.strip() for d in domain.split(",") if d.strip()]))
        filtered = [
            b for b in all_buckets
            if _meta_int(b["metadata"], "importance", 0) >= importance_min
            and b["metadata"].get("type") not in ("feel", "plan", "mirage")
            and not b["metadata"].get("digested", False)
            and (not importance_domains or set(b["metadata"].get("domain", [])) & importance_domains)
            # 釘選/保護桶不受 exclude_domain 影響（audit F1）——docstring 承諾
            # 「各模式通用」，importance_min 曾是唯一漏掉豁免的模式。
            and (b["metadata"].get("pinned") or b["metadata"].get("protected")
                 or not _excluded(b["metadata"]))
        ]
        filtered.sort(key=lambda b: _meta_int(b["metadata"], "importance", 0), reverse=True)
        filtered = _select_importance_tiers(filtered, cap=min(20, max_results))
        if not filtered:
            return f"沒有重要度 >= {importance_min} 的記憶。"
        results = []
        token_used = 0
        for b in filtered:
            if token_used >= max_tokens:
                break
            try:
                clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), clean_meta)
                imp = b["metadata"].get("importance", 0)
                formatted = f"[importance:{imp}] [bucket_id:{b['id']}] {summary}"
                t = count_tokens_approx(formatted)
                if token_used + t > max_tokens:
                    continue
                results.append(formatted)
                token_used += t
            except Exception as e:
                logger.warning(f"importance_min dehydrate failed: {e}")
        return "\n---\n".join(results) if results else "沒有可以展示的記憶。"

    # --- Feel retrieval: domain="feel" is a special channel ---
    # --- Feel 檢索：domain="feel" 是獨立入口（必須在空 query 浮現之前判斷，否則空 query 會被浮現分支攔截）---
    if domain.strip().lower() == "feel":
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
            feels = [
                b for b in all_buckets
                if b["metadata"].get("type") == "feel"
                and not b["metadata"].get("digested", False)
            ]
            feels.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
            if not feels:
                return "沒有留下過 feel。"
            feels = feels[:max_results]
            results = []
            token_used = 0
            for f in feels:
                created = f["metadata"].get("created", "")
                entry = f"[{created}] [bucket_id:{f['id']}]\n{strip_wikilinks(f['content'])}"
                needed = count_tokens_approx(entry)
                if token_used + needed > max_tokens:
                    continue
                results.append(entry)
                token_used += needed
            if not results:
                return "feel 存在，但超出這次 token 預算。"
            return "=== 你留下的 feel ===\n" + "\n---\n".join(results)
        except Exception as e:
            logger.error(f"Feel retrieval failed: {e}")
            return "讀取 feel 失敗。"

    # --- Mirage retrieval: domain="mirage" is the dream channel (batch-2).
    # domain="dream"/"夢" redirects with a hint — weak models land safely.
    # --- 蜃景檢索：domain="mirage" 是夢的獨立入口；打 "dream"/"夢" 會被指路。---
    if domain.strip().lower() in ("dream", "夢", "夢境"):
        return '夢住在 domain="mirage"（蜃景）。dream() 工具是消化儀式，不是夢。'
    if domain.strip().lower() == "mirage":
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
            dreams = [b for b in all_buckets if b["metadata"].get("type") == "mirage"]
            dreams.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
            if not dreams:
                return "還沒有做過夢。"
            dreams = dreams[:max_results]
            results = []
            token_used = 0
            for d in dreams:
                created = d["metadata"].get("created", "")
                consumed_note = d["metadata"].get("consumed", "")
                src_line = f"\n（素材：{consumed_note}）" if consumed_note else ""
                entry = f"[{created}] [bucket_id:{d['id']}]\n{strip_wikilinks(d['content'])}{src_line}"
                needed = count_tokens_approx(entry)
                if token_used + needed > max_tokens:
                    continue
                results.append(entry)
                token_used += needed
            if not results:
                return "夢存在，但超出這次 token 預算。"
            return (
                "=== 你做過的夢（殘影，不是事實） ===\n" + "\n---\n".join(results)
            )
        except Exception as e:
            logger.error(f"Dream retrieval failed: {e}")
            return "讀取夢失敗。"

    # --- No args or empty query: surfacing mode (weight pool active push) ---
    # --- 無參數或空query：浮現模式（權重池主動推送）---
    if not query or not query.strip():
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
        except Exception as e:
            logger.error(f"Failed to list buckets for surfacing / 浮現列桶失敗: {e}")
            return "記憶系統暫時無法訪問。"

        surface_domains = set(normalize_domains([
            d.strip() for d in domain.split(",") if d.strip()
        ]))

        # --- Pinned/protected buckets: core principles. They are RESIDENT, not
        # surfaced — so they no longer spend the surfacing count budget.
        # --- 釘選/保護桶：核心準則。它們是「常駐」不是「被浮現」，
        # 所以不再花用浮現的數量預算。---
        # They used to decrement result_slots, which starved the dynamic pool
        # dead: with 5 pinned buckets and the documented opening call
        # (max_results=5), candidates[:0] returned NOTHING — 360 eligible
        # memories, zero surfaced, at any token budget. Capping pinned at "half
        # the slots" would only push that bug one bucket further away; taking
        # them out of the budget kills it. max_results now means what it says:
        # a cap on DYNAMIC results.
        # 它們本來會扣 result_slots，把動態池餓死：5 個釘選桶 + 開場規則的
        # max_results=5 → candidates[:0] → 什麼都沒有。360 個候選、零浮現，
        # 而且跟 token 額度無關。「釘選最多吃一半」只是把 bug 推遠一個桶；
        # 把它們拿出預算才是真的解決。max_results 現在名副其實＝動態結果上限。
        pinned_buckets = [
            b for b in all_buckets
            if b["metadata"].get("pinned") or b["metadata"].get("protected")
            if not surface_domains or set(b["metadata"].get("domain", [])) & surface_domains
        ]
        pinned_buckets.sort(
            key=lambda b: (
                _meta_int(b["metadata"], "importance", 0),
                str(b["metadata"].get("name", b["id"])),
            ),
            reverse=True,
        )
        pinned_results = []
        token_budget = max_tokens
        result_slots = max_results  # dynamic-only; pinned never spends it
        for b in pinned_buckets:
            if token_budget <= 0:
                break
            try:
                clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), clean_meta)
                formatted = f"📌 [核心準則] [bucket_id:{b['id']}] {summary}"
                needed = count_tokens_approx(formatted)
                if needed > token_budget:
                    continue
                pinned_results.append(formatted)
                token_budget -= needed
            except Exception as e:
                logger.warning(f"Failed to dehydrate pinned bucket / 釘選桶脫水失敗: {e}")
                continue

        # --- Surface cooldown (2026-07-12 batch-2): a bucket shown recently sits
        # out the next surfacings, so the same hot handful stops headlining every
        # breath. Query search is untouched (asking is always answered); pinned
        # never cool. recall.surface_cooldown_hours=0 disables.
        # --- 浮現冷卻（第二批）：剛浮現過的桶下幾輪讓位，熱門幾條不再霸佔
        # 每一次呼吸。query 檢索不受影響（開口問就答）；釘選永不冷卻。---
        cooldown_h = float(config.get("recall", {}).get("surface_cooldown_hours", 6))

        def _cooling(meta: dict) -> bool:
            if cooldown_h <= 0:
                return False
            ts = meta.get("last_surfaced", "")
            if not ts:
                return False
            try:
                return datetime.now() - datetime.fromisoformat(str(ts)) < timedelta(hours=cooldown_h)
            except (ValueError, TypeError):
                return False

        # --- Unresolved buckets: surface top N by weight ---
        # --- 未解決桶：按權重浮現前 N 條 ---
        # `eligible` ignores the surface cooldown on purpose: the cooldown
        # rotates what gets SHOWN, but the tombstone shows nothing — it only
        # names what has faded, and that roster shouldn't flicker between
        # breaths just because a name was mentioned recently.
        # eligible 刻意不套浮現冷卻：冷卻管的是「輪流被show」，而墓碑什麼都不show，
        # 它只是點名誰淡掉了——那份名單不該因為剛被提過就閃爍。
        # is_decay_exempt() replaces the old literal tuple + pinned + protected
        # checks (exactly equivalent), so the surfacing pool and heat can never
        # disagree about what's exempt.
        # is_decay_exempt() 取代原本的字面 tuple + pinned + protected 三重檢查
        # （完全等價），讓浮現池與熱度對「誰是豁免」永遠說同一套。
        eligible = [
            b for b in all_buckets
            if not b["metadata"].get("resolved", False)
            and not b["metadata"].get("digested", False)
            and not is_decay_exempt(b["metadata"])
            and (not surface_domains or set(b["metadata"].get("domain", [])) & surface_domains)
            and not _excluded(b["metadata"])
        ]
        unresolved = [b for b in eligible if not _cooling(b["metadata"])]

        logger.info(
            f"Breath surfacing: {len(all_buckets)} total, "
            f"{len(pinned_buckets)} pinned, {len(unresolved)} unresolved"
        )

        scored = sorted(
            unresolved,
            key=lambda b: decay_engine.calculate_score(b["metadata"]),
            reverse=True,
        )

        if scored:
            top_scores = [(b["metadata"].get("name", b["id"]), decay_engine.calculate_score(b["metadata"])) for b in scored[:5]]
            logger.info(f"Top unresolved scores: {top_scores}")

        # --- Cold-start detection: never-seen important buckets surface first ---
        # Recency-limited to 7 days: surfacing never touches, so without the limit
        # the same untouched buckets would headline every single wake-up forever.
        # --- 冷啟動檢測：從未被訪問過且重要度>=8的桶優先插入最前面（最多2個）---
        # 限最近7天：浮現不 touch，不加時限的話同兩個桶會永遠霸佔開場。
        def _is_recent(meta: dict, days: int = 7) -> bool:
            try:
                created = datetime.fromisoformat(str(meta.get("created", "")))
                return datetime.now() - created <= timedelta(days=days)
            except (ValueError, TypeError):
                return False

        cold_start = [
            b for b in unresolved
            if _meta_float(b["metadata"], "activation_count", 0.0) == 0
            and _meta_int(b["metadata"], "importance", 0) >= 8
            and _is_recent(b["metadata"])
        ][:2]
        cold_start_ids = {b["id"] for b in cold_start}
        # Merge: cold_start first, then scored (excluding duplicates)
        scored_deduped = [b for b in scored if b["id"] not in cold_start_ids]
        scored_with_cold = cold_start + scored_deduped

        # --- Token-budgeted surfacing with diversity + hard cap ---
        # --- 按 token 預算浮現，帶多樣性 + 硬上限 ---
        # Top-1 always surfaces; rest sampled from top-20 for diversity
        candidates = list(scored_with_cold)
        if len(candidates) > 1:
            # Cold-start buckets stay at front; shuffle rest from top-20
            n_cold = len(cold_start)
            non_cold = candidates[n_cold:]
            if len(non_cold) > 1:
                top1 = [non_cold[0]]
                pool = non_cold[1:min(20, len(non_cold))]
                random.shuffle(pool)
                non_cold = top1 + pool + non_cold[min(20, len(non_cold)):]
            candidates = cold_start + non_cold

        # --- Heat helpers: every one of them FAILS AS IF THE FEATURE WEREN'T
        # THERE. This surfacing path carries the pinned core principles — the
        # portraits, the identity anchors. A bug in a display-tier enhancement
        # must never be able to cost a wake-up its own name.
        # --- 熱度輔助：每一個都「壞掉時就當這功能不存在」。這條浮現路徑載著
        # 釘選的核心準則——畫像、身分錨點。一個管顯示分層的加值功能，
        # 絕不該有能力讓一次醒來失去自己的名字。---
        # The two directions are deliberately asymmetric: heat fails VIVID (a
        # bug must not hide a memory) and priority fails ZERO (a bug must not
        # drag a dead one up). Both collapse to old behaviour.
        # 兩個方向刻意相反：heat 壞了偏鮮明（不因 bug 藏東西），
        # priority 壞了偏零（不因 bug 挖墳）。兩者都退化成舊行為。
        def _heat_of(b) -> float:
            try:
                return float(decay_engine.calculate_heat(b["metadata"]))
            except Exception:
                return 1.0

        def _tier_of(b) -> str:
            try:
                tier = decay_engine.heat_tier(_heat_of(b))
                return tier if tier in ("vivid", "faded", "lost") else "vivid"
            except Exception:
                return "vivid"

        def _prio_of(b) -> float:
            try:
                return float(decay_engine.review_priority(_heat_of(b)))
            except Exception:
                return 0.0

        try:
            truncate_len = int(decay_engine.heat_truncate)
        except Exception:
            truncate_len = 60

        # --- Danger-zone review: 2 normal : 1 danger ---
        # --- 危險區複習：2 熱 : 1 快沒 ---
        # Surfacing has always answered "what's hottest". This slice answers
        # "what am I about to lose" — a memory at heat 0.25 is one you'd fail
        # to recall three times in four, and it will not ask to be remembered.
        # 浮現一直只回答「什麼最熱」。這個切片回答「什麼快沒了」——
        # heat 0.25 的記憶是四次有三次想不起來的，而它不會開口要人記得它。
        # --- Engineering quota (batch-7 檔1): per-channel cap on PURE-eng picks.
        # The danger zone was the measured leak (07-16: one of two slots went
        # to a stale engineering note), so it gets its own tighter cap. Ratios
        # from config; ceil for the normal channel (generous), floor for the
        # danger zone (strict). Backfill inside _pick_with_eng_quota keeps
        # every pick-count invariant unchanged.
        # --- 工程配額（檔1）：各通道封頂「純工程」選取數。危險區是實測漏風口
        # （07-16 兩名額之一給了過期工程筆記），配額更緊。比例走 config；
        # 一般通道 ceil（寬）、危險區 floor（嚴）。回填邏輯保證所有
        # 「選取數」不變量照舊。---
        surf_cfg = config.get("surfacing", {}) or {}
        try:
            eng_ratio = float(surf_cfg.get("eng_quota_ratio", 1 / 3))
        except (ValueError, TypeError):
            eng_ratio = 1 / 3
        try:
            dz_eng_ratio = float(surf_cfg.get("dz_eng_quota_ratio", 0.5))
        except (ValueError, TypeError):
            dz_eng_ratio = 0.5
        eng_domains = _eng_domain_set()

        danger_slots = result_slots // 3
        danger_pick: list = []
        if danger_slots > 0:
            ranked = sorted(
                ((_prio_of(b), b) for b in unresolved), key=lambda t: -t[0]
            )
            dz_quota = min(danger_slots, math.floor(danger_slots * dz_eng_ratio))
            danger_pick = _pick_with_eng_quota(
                [b for prio, b in ranked if prio > 0],
                danger_slots, dz_quota, eng_domains,
            )
        danger_ids = {b["id"] for b in danger_pick}
        # Unclaimed danger slots go back to the normal channel rather than
        # being burned — a quiet danger zone shouldn't shrink the whole breath.
        # 沒用掉的危險區名額還給一般通道，不要白白燒掉——
        # 危險區安靜的時候，不該讓整次呼吸跟著變小。
        normal_slots = result_slots - len(danger_pick)

        # The normal channel carries only what's still retrievable. 'lost'
        # buckets aren't dropped — they're named in the tombstone below.
        # (Read before coding, 檔1 必答題: lost is filtered BEFORE the slot
        # slice, so a high-score lost bucket never occupies a slot — which is
        # why the metabolism factor multiplies heat only, never score.)
        # 一般通道只載還提取得到的。lost 的不是被丟掉，是在下面被點名。
        # （檔1 必答題讀碼結論：lost 在切名額「之前」就被濾掉，高分 lost 桶
        # 不佔名額——所以代謝係數只乘 heat、不乘 score。）
        eng_quota = min(normal_slots, math.ceil(normal_slots * eng_ratio))
        normal_pick = _pick_with_eng_quota(
            [
                b for b in candidates
                if b["id"] not in danger_ids and _tier_of(b) != "lost"
            ],
            normal_slots, eng_quota, eng_domains,
        )

        dynamic_results = []
        surfaced_ids: list[str] = []
        for b in normal_pick + danger_pick:
            if token_budget <= 0:
                break
            try:
                clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), clean_meta)
                score = decay_engine.calculate_score(b["metadata"])
                heat = _heat_of(b)
                if _tier_of(b) != "vivid":
                    summary = _blur_summary(summary, truncate_len)
                # Both numbers show: weight decides whether it stays, heat
                # decides whether you still remember it. They were conflated;
                # printing both is how that stops being invisible.
                # 兩個數字都露出來：權重決定該不該留，熱度決定還記不記得清。
                # 它們曾經被混為一談，把兩個都印出來，才不會又變回看不見。
                mark = f"[權重:{score:.2f} 熱度:{heat:.2f}]"
                if b["id"] in danger_ids:
                    # Why it surfaced must be visible: it's fading, not hot.
                    # 它為什麼上台必須看得見：是快沒了，不是它熱。
                    mark = f"[危險區 快要失去 prio:{_prio_of(b):.2f} 熱度:{heat:.2f}]"
                formatted = f"{mark} [bucket_id:{b['id']}] {summary}"
                summary_tokens = count_tokens_approx(formatted)
                if summary_tokens > token_budget:
                    continue
                # NOTE: no touch() here — surfacing should NOT reset decay timer.
                # This is load-bearing for the danger zone: a memory dragged up
                # keeps fading unless it's actually engaged with. Being shown
                # doesn't save it. No auto-rescue.
                # 這對危險區是承重的：被撈上來的記憶如果沒被真的用到，會繼續淡。
                # 被看到不等於被救。不自動搶救。
                dynamic_results.append(formatted)
                surfaced_ids.append(b["id"])
                token_budget -= summary_tokens
            except Exception as e:
                logger.warning(f"Failed to dehydrate surfaced bucket / 浮現脫水失敗: {e}")
                continue

        # --- Tombstone: name what faded below the injection floor ---
        # --- 墓碑：點名淡出注入門檻的 ---
        shown_ids = set(surfaced_ids)
        tombstone = _tombstone_line([
            b for b in eligible
            if b["id"] not in shown_ids and _tier_of(b) == "lost"
        ])

        if not pinned_results and not dynamic_results and not tombstone:
            if pinned_buckets or unresolved:
                return "記憶存在，但超出這次 token 預算。"
            return "權重池平靜，沒有需要處理的記憶。"

        def _surface_parts() -> list[str]:
            out = []
            if pinned_results:
                out.append("=== 核心準則 ===\n" + "\n---\n".join(pinned_results))
            if dynamic_results:
                out.append("=== 浮現記憶 ===\n" + "\n---\n".join(dynamic_results))
            if tombstone:
                out.append(tombstone)
            return out

        parts = _surface_parts()
        # Headers/separators are part of the public token contract too. Trim
        # lowest-priority tail entries until the complete response fits.
        # The tombstone is dropped FIRST and only if trimming actual memories
        # wasn't enough — a roster of names is the cheapest thing here (~30
        # tokens), and dropping it silently re-creates the exact omission it
        # exists to prevent.
        # 墓碑最後才丟，而且只在連記憶都修剪光了還不夠時才丟——它是這裡最便宜的
        # 東西（約 30 token），而丟掉它就等於重新製造它本來要防的那個沉默。
        while parts and count_tokens_approx("\n\n".join(parts)) > max_tokens:
            if dynamic_results:
                dynamic_results.pop()
                surfaced_ids.pop()
            elif pinned_results:
                pinned_results.pop()
            elif tombstone:
                tombstone = ""
            else:
                break
            parts = _surface_parts()
        if not parts:
            return "記憶存在，但超出這次 token 預算。"

        # --- Stamp what actually surfaced (cooldown + retrieved_count; not a touch) ---
        # --- 蓋浮現章（供冷卻與 retrieved 計數；不是加固）---
        for bid in surfaced_ids:
            try:
                await bucket_mgr.mark_surfaced(bid)
            except Exception as e:
                logger.warning(f"mark_surfaced failed: {bid}: {e}")
        return "\n\n".join(parts)

    # --- With args: search mode (keyword + vector dual channel) ---
    # --- 有參數：檢索模式（關鍵詞 + 向量雙通道）---
    domain_filter = [d.strip() for d in domain.split(",") if d.strip()] or None
    exclude_filter = sorted(exclude_set) or None
    q_valence = valence if 0 <= valence <= 1 else None
    q_arousal = arousal if 0 <= arousal <= 1 else None

    try:
        matches = await bucket_mgr.search(
            query,
            limit=max(max_results, 20),
            domain_filter=domain_filter,
            query_valence=q_valence,
            query_arousal=q_arousal,
            exclude_domains=exclude_filter,
        )
    except Exception as e:
        logger.error(f"Search failed / 檢索失敗: {e}")
        return "檢索過程出錯，請稍後重試。"

    # --- Exclude pinned/protected from search results (they surface in surfacing mode) ---
    # --- 搜索模式排除釘選桶（它們在浮現模式中始終可見）---
    matches = [b for b in matches if not (b["metadata"].get("pinned") or b["metadata"].get("protected"))]

    # --- Vector similarity channel: find semantically related buckets ---
    # --- 向量相似度通道：找到語義相關的桶 ---
    matched_ids = {b["id"] for b in matches}
    try:
        vector_results = await embedding_engine.search_similar(query, top_k=max(max_results, 20))
        for bucket_id, sim_score in vector_results:
            if bucket_id not in matched_ids and sim_score > 0.5:
                bucket = await bucket_mgr.get(bucket_id)
                if not bucket:
                    continue
                meta = bucket["metadata"]
                # feel is a private channel; plan lives in dream's tail only;
                # dreams are their own channel; digested means hidden from recall push
                # feel 是私人通道；plan 只活在 dream 尾端；蜃景（夢）走自己的通道；
                # digested 桶不經向量通道回浮
                if meta.get("pinned") or meta.get("protected"):
                    continue
                if meta.get("type") in ("feel", "plan", "mirage") or meta.get("digested"):
                    continue
                # exclude_domain binds the vector side door too (batch-7),
                # checked before revive for the same reason as the gate below.
                # exclude_domain 也約束向量側門（檔1），與下面的閘同理放在復活前。
                if _excluded(meta):
                    continue
                # Admissibility (2026-07-12 batch-3): the vector channel honors
                # the explicit domain filter and the context gate — checked
                # BEFORE the revive branch so weak grazes can't revive either.
                # 入場閘：向量通道也守 domain filter 與情境門控——放在復活
                # 分支之前，弱擦邊連歸檔桶都不能復活。
                if not bucket_mgr.vector_admissible(
                    meta, sim_score, q_valence, q_arousal, domain_filter
                ):
                    continue
                # A strong semantic hit on an archived memory revives it:
                # it comes back to dynamic/ (resolved) instead of being half-dead.
                # 語義強命中歸檔記憶 → 復活搬回 dynamic（以 resolved 狀態迴歸）。
                if meta.get("type") == "archived":
                    revived = await bucket_mgr.revive(bucket_id, resolved=True, actor=Actor("cyan", "breath:semantic_revive"))
                    if not revived:
                        continue
                    bucket = await bucket_mgr.get(bucket_id) or bucket
                bucket["score"] = round(sim_score * 100, 2)
                bucket["vector_match"] = True
                matches.append(bucket)
                matched_ids.add(bucket_id)
    except Exception as e:
        logger.warning(f"Vector search failed, using keyword only / 向量搜索失敗: {e}")

    results = []
    token_used = 0
    ripple_done = False
    for bucket in matches[:max_results]:
        if token_used >= max_tokens:
            break
        try:
            clean_meta = {k: v for k, v in bucket["metadata"].items() if k != "tags"}
            # --- Memory reconstruction: shift displayed valence by current mood ---
            # --- 記憶重構：根據當前情緒微調展示層 valence（±0.1）---
            if q_valence is not None and "valence" in clean_meta:
                original_v = float(clean_meta.get("valence", 0.5))
                shift = (q_valence - 0.5) * 0.2  # ±0.1 max shift
                clean_meta["valence"] = max(0.0, min(1.0, original_v + shift))
            summary = await dehydrator.dehydrate(strip_wikilinks(bucket["content"]), clean_meta)
            formatted = (
                f"[語義關聯] [bucket_id:{bucket['id']}] {summary}"
                if bucket.get("vector_match")
                else f"[bucket_id:{bucket['id']}] {summary}"
            )
            summary_tokens = count_tokens_approx(formatted)
            if results:
                summary_tokens += count_tokens_approx("\n---\n")
            if token_used + summary_tokens > max_tokens:
                continue
            # --- Two-tier reinforcement (2026-07-12 batch-2): only the strongest
            # hit gets a real touch (activation + ripple) — the rest were merely
            # RETRIEVED, and being listed is not being engaged with. Every hit
            # still stamps last_surfaced/retrieved_count for stats.
            # --- 兩層加固（第二批）：只有最強命中才真正 touch（激活＋漣漪），
            # 其餘只是「被搜到」——被列出不等於被用到。所有命中仍蓋
            # retrieved 章供統計。---
            if not ripple_done:
                await bucket_mgr.touch(bucket["id"], ripple=True)
                ripple_done = True
            await bucket_mgr.mark_surfaced(bucket["id"])
            results.append(formatted)
            token_used += summary_tokens
        except Exception as e:
            logger.warning(f"Failed to dehydrate search result / 檢索結果脫水失敗: {e}")
            continue

    # --- Random surfacing: when search returns < 3, 40% chance to float old memories ---
    # --- 隨機浮現：檢索結果不足 3 條時，40% 概率從低權重舊桶裡漂上來 ---
    remaining_slots = max_results - len(results)
    if len(results) < 3 and remaining_slots > 0 and token_used < max_tokens and random.random() < 0.4:
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
            matched_ids = {b["id"] for b in matches}
            low_weight = [
                b for b in all_buckets
                if b["id"] not in matched_ids
                and b["metadata"].get("type") not in ("feel", "permanent", "plan", "mirage")
                and not b["metadata"].get("pinned", False)
                and not b["metadata"].get("protected", False)
                and not b["metadata"].get("resolved", False)
                and not b["metadata"].get("digested", False)
                and not _excluded(b["metadata"])
                and bucket_mgr.vector_admissible(
                    b["metadata"], 0.0, q_valence, q_arousal, domain_filter
                )
                and decay_engine.calculate_score(b["metadata"]) < 2.0
            ]
            if low_weight:
                drifted = random.sample(
                    low_weight,
                    min(random.randint(1, 3), len(low_weight), remaining_slots),
                )
                first_drift = True
                for b in drifted:
                    clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                    summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), clean_meta)
                    heading = "--- 忽然想起來 ---\n" if first_drift else ""
                    formatted = (
                        f"{heading}[surface_type: random] [bucket_id:{b['id']}]\n{summary}"
                    )
                    needed = count_tokens_approx(formatted)
                    if results:
                        needed += count_tokens_approx("\n---\n")
                    if token_used + needed > max_tokens:
                        continue
                    results.append(formatted)
                    token_used += needed
                    first_drift = False
                    await bucket_mgr.mark_surfaced(b["id"])
        except Exception as e:
            logger.warning(f"Random surfacing failed / 隨機浮現失敗: {e}")

    if not results:
        await _fire_webhook("breath", {"mode": "empty", "matches": 0})
        return "未找到相關記憶。"

    final_text = "\n---\n".join(results)
    await _fire_webhook("breath", {"mode": "ok", "matches": len(matches), "chars": len(final_text)})
    return final_text


# =============================================================
# Tool 2: hold — Hold on to this
# 工具 2：hold — 握住，留下來
# =============================================================
@mcp.tool()
async def hold(
    content: str,
    tags: str = "",
    importance: int = 5,
    pinned: bool = False,
    feel: bool = False,
    mirage: bool = False,
    consumed: str = "",
    source_bucket: str = "",    valence: float = -1,
    arousal: float = -1,
    why_remembered: str = "",
    during: str = "",
) -> str:
    """存儲單條記憶,自動打標+合併。tags逗號分隔,importance 1-10。pinned=True創建永久釘選桶。feel=True存儲你的第一人稱感受(不參與普通浮現)。mirage=True存儲一段夢(蜃景桶——名字本身就是警示:鮮明但不是真的。隔離桶:不合併/不搜尋/不衰減/不進畫像或self_concept,只走breath(domain="mirage")讀;consumed=逗號分隔的素材桶ID,記出處;夢是敘事殘影不是事實,永不當記憶引用)。source_bucket=被消化的記憶桶ID(feel模式下,標記源記憶為已消化)。why_remembered=為什麼記住(可選,自由文本,僅展示不計分)。during="dream"=你自己申報這次是在消化儀式當下做的(自陳,非系統觀察,只影響歷史紀錄的欄位,不確定就別填)。"""
    await decay_engine.ensure_started()

    # --- Input validation / 輸入校驗 ---
    if not content or not content.strip():
        return "內容為空，無法存儲。"
    if feel and mirage:
        return "feel 和 mirage（夢）是兩種桶，一次只能選一種。"

    importance = max(1, min(10, importance))
    extra_tags = [t.strip() for t in tags.split(",") if t.strip()]

    # --- Mirage mode (2026-07-12 batch-2): an isolated dream-narrative bucket.
    # truth status IS the bucket type: merges can't reach it (excluded from
    # search), it never decays, never surfaces in ordinary breath, and by
    # rule never gets cited into portraits or self_concept as fact.
    # --- 蜃景模式（第二批）：隔離的夢敘事桶。桶型即真值標記（蜃景＝鮮明但不真）：合併搆不到
    # （排除在搜尋外）、不衰減、不進普通浮現，且按規則永不作為事實
    # 引入畫像或 self_concept。---
    if mirage:
        d_valence = valence if 0 <= valence <= 1 else 0.5
        d_arousal = arousal if 0 <= arousal <= 1 else 0.4
        bucket_id = await bucket_mgr.create(
            content=content,
            tags=[],
            importance=5,
            domain=[],
            valence=d_valence,
            arousal=d_arousal,
            name=None,
            bucket_type="mirage",
            extra_meta={"consumed": consumed.strip()},
        )
        try:
            await embedding_engine.generate_and_store(bucket_id, content)
        except Exception:
            pass
        return f"🌙mirage→{bucket_id}"

    # --- Feel mode: store as feel type, minimal metadata ---
    # --- Feel 模式：存為 feel 類型，最少元數據 ---
    if feel:
        # Feel valence/arousal = model's own perspective
        feel_valence = valence if 0 <= valence <= 1 else 0.5
        feel_arousal = arousal if 0 <= arousal <= 1 else 0.3
        bucket_id = await bucket_mgr.create(
            content=content,
            tags=[],
            importance=5,
            domain=[],
            valence=feel_valence,
            arousal=feel_arousal,
            name=None,
            bucket_type="feel",
        )
        try:
            await embedding_engine.generate_and_store(bucket_id, content)
        except Exception:
            pass
        # --- Mark source memory as digested + store model's valence perspective ---
        # --- 標記源記憶為已消化 + 存儲模型視角的 valence ---
        if source_bucket and source_bucket.strip():
            try:
                update_kwargs = {"digested": True}
                if 0 <= valence <= 1:
                    update_kwargs["model_valence"] = feel_valence
                # This is where a bad digestion pass actually lands: writing a
                # feel marks its source digested (×0.2 → archived). dream()
                # itself writes nothing — this call is the hand it moves.
                # 一次壞掉的消化真正落地的地方：寫 feel 會把來源標成 digested
                # （×0.2 → 歸檔）。dream() 自己不寫東西——它動的是這隻手。
                await bucket_mgr.update(
                    source_bucket.strip(),
                    actor=Actor("cyan", "hold:feel", during or None),
                    **update_kwargs,
                )
            except Exception as e:
                logger.warning(f"Failed to mark source as digested / 標記已消化失敗: {e}")
        return f"🫧feel→{bucket_id}" + await _api_usage_warning_suffix()

    # --- Step 1: auto-tagging / 自動打標 ---
    try:
        analysis = await dehydrator.analyze(content)
    except Exception as e:
        logger.warning(f"Auto-tagging failed, using defaults / 自動打標失敗: {e}")
        analysis = {
            "domain": ["未分類"], "valence": 0.5, "arousal": 0.3,
            "tags": [], "suggested_name": "",
        }

    domain = analysis["domain"]
    auto_valence = analysis["valence"]
    auto_arousal = analysis["arousal"]
    auto_tags = analysis["tags"]
    suggested_name = analysis.get("suggested_name", "")

    # --- User-supplied valence/arousal takes priority over analyze() result ---
    # --- 用戶顯式傳入的 valence/arousal 優先，analyze() 結果作為 fallback ---
    final_valence = valence if 0 <= valence <= 1 else auto_valence
    final_arousal = arousal if 0 <= arousal <= 1 else auto_arousal

    all_tags = list(dict.fromkeys(auto_tags + extra_tags))

    # --- Pinned buckets bypass merge and are created directly in permanent dir ---
    # --- 釘選桶跳過合併，直接新建到 permanent 目錄 ---
    if pinned:
        bucket_id = await bucket_mgr.create(
            content=content,
            tags=all_tags,
            importance=10,
            domain=domain,
            valence=final_valence,
            arousal=final_arousal,
            name=suggested_name or None,
            bucket_type="permanent",
            pinned=True,
            extra_meta={"why_remembered": why_remembered} if why_remembered.strip() else None,
        )
        try:
            await embedding_engine.generate_and_store(bucket_id, content)
        except Exception:
            pass
        return f"📌釘選→{bucket_id} {','.join(domain)}" + await _api_usage_warning_suffix()

    # --- Step 2: merge or create / 合併或新建 ---
    result_name, is_merged = await _merge_or_create(
        content=content,
        tags=all_tags,
        importance=importance,
        domain=domain,
        valence=final_valence,
        arousal=final_arousal,
        name=suggested_name,
        extra_meta={"why_remembered": why_remembered} if why_remembered.strip() else None,
    )

    action = "合併→" if is_merged else "新建→"
    return f"{action}{result_name} {','.join(domain)}" + await _api_usage_warning_suffix()


# =============================================================
# Tool 3: grow — Grow, fragments become memories
# 工具 3：grow — 生長，一天的碎片長成記憶
# =============================================================
@mcp.tool()
async def grow(content: str = "", items: list = None) -> str:
    """日記歸檔,自動拆分為多桶。短內容(<30字)走快速路徑。進階:若你已在完整上下文裡把長文拆成N條定稿,傳items=[條1,條2,...](字串列表)即可逐字入庫——跳過系統二次拆分改寫,每條正文一字不動只補元數據;合併到老桶也用原文追加不壓縮。傳了items就忽略content。"""
    await decay_engine.ensure_started()

    # --- Verbatim mode: the caller pre-split the text with full context.
    # Each item is final prose — store it word-for-word, only add metadata.
    # --- 逐字入庫模式：調用方帶著完整上下文拆好了條目，正文一字不動，只補元數據。---
    if items:
        if not isinstance(items, list):
            return "items 必須是字串列表。"
        clean_items = [str(it).strip() for it in items if str(it).strip()][:50]
        if not clean_items:
            return "items 為空，沒有可入庫的條目。"
        results = []
        created = 0
        merged = 0
        for item_text in clean_items:
            try:
                try:
                    analysis = await dehydrator.analyze(item_text)
                except Exception as e:
                    logger.warning(f"Verbatim item analyze failed / 逐字條目打標失敗: {e}")
                    analysis = {
                        "domain": ["未分類"], "valence": 0.5, "arousal": 0.3,
                        "tags": [], "suggested_name": "",
                    }
                result_name, is_merged = await _merge_or_create(
                    content=item_text,
                    tags=analysis.get("tags", []),
                    importance=analysis.get("importance", 5) if isinstance(analysis.get("importance"), int) else 5,
                    domain=analysis.get("domain", ["未分類"]),
                    valence=analysis.get("valence", 0.5),
                    arousal=analysis.get("arousal", 0.3),
                    name=analysis.get("suggested_name", ""),
                    verbatim=True,
                )
                if is_merged:
                    results.append(f"📎{result_name}")
                    merged += 1
                else:
                    results.append(f"📝{result_name}")
                    created += 1
            except Exception as e:
                logger.warning(f"Verbatim item failed / 逐字條目入庫失敗: {e}")
                results.append(f"⚠️{item_text[:20]}")
        return (
            f"逐字入庫 {len(clean_items)}條|新{created}合{merged}\n" + "\n".join(results)
            + await _api_usage_warning_suffix()
        )

    if not content or not content.strip():
        return "內容為空，無法整理。"

    # --- Short content fast path: skip digest, use hold logic directly ---
    # --- 短內容快速路徑：跳過 digest 拆分，直接走 hold 邏輯省一次 API ---
    # For very short inputs (like "1"), calling digest is wasteful:
    # it sends the full DIGEST_PROMPT (~800 tokens) to DeepSeek for nothing.
    # Instead, run analyze + create directly.
    if len(content.strip()) < 30:
        logger.info(f"grow short-content fast path: {len(content.strip())} chars")
        try:
            analysis = await dehydrator.analyze(content)
        except Exception as e:
            logger.warning(f"Fast-path analyze failed / 快速路徑打標失敗: {e}")
            analysis = {
                "domain": ["未分類"], "valence": 0.5, "arousal": 0.3,
                "tags": [], "suggested_name": "",
            }
        result_name, is_merged = await _merge_or_create(
            content=content.strip(),
            tags=analysis.get("tags", []),
            importance=analysis.get("importance", 5) if isinstance(analysis.get("importance"), int) else 5,
            domain=analysis.get("domain", ["未分類"]),
            valence=analysis.get("valence", 0.5),
            arousal=analysis.get("arousal", 0.3),
            name=analysis.get("suggested_name", ""),
        )
        action = "合併" if is_merged else "新建"
        return (
            f"{action} → {result_name} | {','.join(analysis.get('domain', []))} "
            f"V{analysis.get('valence', 0.5):.1f}/A{analysis.get('arousal', 0.3):.1f}"
            + await _api_usage_warning_suffix()
        )

    # --- Step 1: let API split and organize / 讓 API 拆分整理 ---
    try:
        items = await dehydrator.digest(content)
    except Exception as e:
        logger.error(f"Diary digest failed / 日記整理失敗: {e}")
        return f"日記整理失敗: {e}" + await _api_usage_warning_suffix()

    if not items:
        return "內容為空或整理失敗。" + await _api_usage_warning_suffix()

    results = []
    created = 0
    merged = 0

    # --- Step 2: merge or create each item (with per-item error handling) ---
    # --- 逐條合併或新建（單條失敗不影響其他）---
    for item in items:
        try:
            result_name, is_merged = await _merge_or_create(
                content=item["content"],
                tags=item.get("tags", []),
                importance=item.get("importance", 5),
                domain=item.get("domain", ["未分類"]),
                valence=item.get("valence", 0.5),
                arousal=item.get("arousal", 0.3),
                name=item.get("name", ""),
            )

            if is_merged:
                results.append(f"📎{result_name}")
                merged += 1
            else:
                results.append(f"📝{item.get('name', result_name)}")
                created += 1
        except Exception as e:
            logger.warning(
                f"Failed to process diary item / 日記條目處理失敗: "
                f"{item.get('name', '?')}: {e}"
            )
            results.append(f"⚠️{item.get('name', '?')}")

    return f"{len(items)}條|新{created}合{merged}\n" + "\n".join(results) + await _api_usage_warning_suffix()


# =============================================================
# Tool 4: trace — Trace, redraw the outline of a memory
# 工具 4：trace — 描摹，重新勾勒記憶的輪廓
# Also handles deletion (delete=True)
# 同時承接刪除功能
# =============================================================
@mcp.tool()
async def trace(
    bucket_id: str,
    name: str = "",
    domain: str = "",
    valence: float = -1,
    arousal: float = -1,
    importance: int = -1,
    tags: str = "",
    resolved: int = -1,
    pinned: int = -1,
    digested: int = -1,
    content: str = "",
    delete: bool = False,
    status: str = "",
    weight: float = -1,
    kind: str = "",
    target_drive: str = "",
    progress: float = -1,
    due_at: str = "",
    affects_desire: int = -1,
    why_remembered: str = "",
    during: str = "",
    history: bool = False,
    restore_seq: int = -1,
) -> str:
    """修改記憶元數據或內容。resolved=1沉底/0激活,pinned=1釘選/0取消,digested=1隱藏(保留但不浮現)/0取消隱藏,content=替換桶正文,delete=True刪除。plan桶專屬:status=active/resolved/abandoned,weight=承諾重量0.0-1.0,kind=promise/task/question/maintenance,target_drive=掛的驅動維度,progress=進度0.0-1.0,due_at=期限(ISO日期)。affects_desire=1把這條記憶刻意掛成執念(餵慾望boost)/0取下——動態桶專用,珍貴記憶≠此刻掛心,只掛真的還鯁著的事。why_remembered=更新記住的原因。只傳需改的,-1或空=不改。

可復原:每次修改/刪除都會先存快照。history=True列出這個桶的修改史(seq/時間/誰改的/改了什麼);restore_seq=N還原到第N版(還原本身也會存快照,所以還原錯了也能再還原回去;被刪掉的桶也救得回來)。
during="dream"=你自己申報這次修改是在消化儀式當下做的。這是自陳不是系統觀察,會存進獨立欄位,不會跟「誰改的」混在一起。不確定就別填。"""

    if not bucket_id or not bucket_id.strip():
        return "請提供有效的 bucket_id。"

    # --- History mode: read-only, must come before any mutation ---
    # --- 歷史模式：唯讀，必須排在任何變更之前 ---
    if history:
        rows = bucket_history.list(bucket_id, limit=20)
        if not rows:
            return f"{bucket_id} 沒有修改紀錄（可能是歷史表上線前就存在的桶）。"
        lines = []
        for r in rows:
            who = f"{r['actor_type']}:{r['actor_id']}"
            during_note = f" 〔自陳:{r['self_reported_during']}〕" if r["self_reported_during"] else ""
            try:
                m = _json_lib.loads(r["meta"])
                state = (
                    f"imp{m.get('importance')} "
                    f"{'R' if m.get('resolved') else '-'}"
                    f"{'D' if m.get('digested') else '-'}"
                )
            except Exception:
                state = "?"
            lines.append(
                f"  seq={r['seq']:<3} {r['ts']} {r['op']:8} by {who}{during_note}\n"
                f"      當時狀態: {state} 正文{len(r['content'])}字"
            )
        return (
            f"=== {bucket_id} 的修改史（最新在前，每列是「被改掉之前」的樣子）===\n"
            + "\n".join(lines)
            + f"\n\n還原用 trace(bucket_id=\"{bucket_id}\", restore_seq=N)。"
        )

    # --- Restore mode / 還原模式 ---
    if restore_seq >= 1:
        row = bucket_history.get(bucket_id, restore_seq)
        if not row:
            return f"{bucket_id} 沒有 seq={restore_seq} 這一版。用 trace(bucket_id, history=True) 看有哪些。"
        ok = await bucket_mgr.restore(
            bucket_id, restore_seq, actor=Actor("cyan", "trace:restore", during or None)
        )
        if not ok:
            return f"還原失敗: {bucket_id}#{restore_seq}"
        try:
            await embedding_engine.generate_and_store(bucket_id, row["content"])
        except Exception:
            pass
        return (
            f"已還原 {bucket_id} 到 seq={restore_seq}（{row['ts']} 被 "
            f"{row['actor_type']}:{row['actor_id']} 改掉之前的樣子）。\n"
            f"這次還原本身也存了快照——還原錯了就再 history=True 看一次。"
        )

    # --- Delete mode / 刪除模式 ---
    if delete:
        result = await bucket_mgr.delete(bucket_id, actor=Actor("cyan", "trace:delete", during or None))
        if result.ok:
            embedding_engine.delete_embedding(bucket_id)
            # seq 來自這次刪除真正寫成的那筆快照，不是事後 COUNT(*) 猜的。
            # 猜的那個在快照失敗時會指向更舊的版本，而使用者不會知道。
            if result.seq is None:
                return (
                    f"已遺忘記憶桶: {bucket_id}\n"
                    f"（⚠️ 歷史快照沒有寫成，這次刪除救不回來——正文已經不在了。）"
                )
            return (
                f"已遺忘記憶桶: {bucket_id}\n"
                f"（正文與元數據已存進歷史 seq={result.seq}。要拿回來："
                f"trace(bucket_id=\"{bucket_id}\", restore_seq={result.seq})）"
            )
        return f"未找到記憶桶: {bucket_id}"

    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return f"未找到記憶桶: {bucket_id}"

    # --- Collect only fields actually passed / 只收集用戶實際傳入的字段 ---
    updates = {}
    if name:
        updates["name"] = name
    if domain:
        updates["domain"] = [d.strip() for d in domain.split(",") if d.strip()]
    if 0 <= valence <= 1:
        updates["valence"] = valence
    if 0 <= arousal <= 1:
        updates["arousal"] = arousal
    if 1 <= importance <= 10:
        updates["importance"] = importance
    if tags:
        updates["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
    if resolved in (0, 1):
        if bucket["metadata"].get("type") == "plan":
            # A plan's 沉底 must write the lifecycle field the fixation feed
            # actually reads: resolved alone left the plan feeding desire
            # forever while the reply claimed it was down (the web door fixed
            # this in batch-4a; the MCP door kept the hole). An explicit
            # `status` below still wins when both are passed.
            # plan 的沉底要寫執念接線真正讀的生命週期欄位：只寫 resolved 會讓
            # 回覆說「已沉底」、慾望卻繼續被餵（web 門 batch-4a 修過，MCP 門
            # 漏了同款）。同時傳 status 時以下方的顯式值為準。
            updates["status"] = "resolved" if resolved else "active"
        updates["resolved"] = bool(resolved)
    if pinned in (0, 1):
        updates["pinned"] = bool(pinned)
        # 不要在這裡塞 importance=10：bucket_manager.update() 自己會鎖（它的 pinned
        # 分支先把原值存進 importance_prepin 再設 10）。從這裡傳進去只會讓 importance
        # 分支先跑一步把原值蓋掉，prepin 就存成 10，取消釘選再也回不去。
    if digested in (0, 1):
        updates["digested"] = bool(digested)
    if content:
        updates["content"] = content
    if why_remembered:
        updates["why_remembered"] = why_remembered
    # --- Plan-only fields: status / weight ---
    # --- plan 專屬欄位：status / weight ---
    is_plan = bucket["metadata"].get("type") == "plan"
    if status:
        if not is_plan:
            return f"status 只能用在 plan 桶上（{bucket_id} 是 {bucket['metadata'].get('type', 'dynamic')} 桶）。"
        if status not in ("active", "resolved", "abandoned"):
            return "status 只接受 active/resolved/abandoned。"
        updates["status"] = status
        # Keep the decay/sort track in step with the lifecycle track,
        # same as the web door.
        # 與 web 門一致：生命週期軌與沉底軌同步。
        updates["resolved"] = status != "active"
    if 0 <= weight <= 1:
        if not is_plan:
            return f"weight 只能用在 plan 桶上（{bucket_id} 是 {bucket['metadata'].get('type', 'dynamic')} 桶）。"
        updates["weight"] = weight
    if kind:
        if not is_plan:
            return f"kind 只能用在 plan 桶上（{bucket_id} 是 {bucket['metadata'].get('type', 'dynamic')} 桶）。"
        kind = kind.strip().lower()
        if kind not in PLAN_KINDS:
            return f"kind 只接受 {'/'.join(PLAN_KINDS)}。"
        updates["kind"] = kind
    if target_drive:
        if not is_plan:
            return f"target_drive 只能用在 plan 桶上（{bucket_id} 是 {bucket['metadata'].get('type', 'dynamic')} 桶）。"
        target_drive = target_drive.strip().lower()
        if target_drive not in desire_kernel.DRIVE_KEYS:
            return f"target_drive 只接受 {'/'.join(desire_kernel.DRIVE_KEYS)}。"
        updates["target_drive"] = target_drive
    if 0 <= progress <= 1:
        if not is_plan:
            return f"progress 只能用在 plan 桶上（{bucket_id} 是 {bucket['metadata'].get('type', 'dynamic')} 桶）。"
        updates["progress"] = progress
    if due_at:
        if not is_plan:
            return f"due_at 只能用在 plan 桶上（{bucket_id} 是 {bucket['metadata'].get('type', 'dynamic')} 桶）。"
        due_at_norm = _validate_due_at(due_at)
        if due_at_norm is None:
            return "due_at 需為 ISO 日期（如 2026-07-15）。"
        updates["due_at"] = due_at_norm
    if affects_desire in (0, 1):
        btype = bucket["metadata"].get("type", "dynamic")
        if btype != "dynamic":
            return f"affects_desire 只能用在動態桶上（{bucket_id} 是 {btype} 桶；plan 走 target_drive）。"
        updates["affects_desire"] = bool(affects_desire)

    if not updates:
        return "沒有任何字段需要修改。"

    success = await bucket_mgr.update(
        bucket_id, actor=Actor("cyan", "trace", during or None), **updates
    )
    if not success:
        return f"修改失敗: {bucket_id}"

    # Re-generate embedding if content changed
    if "content" in updates:
        try:
            await embedding_engine.generate_and_store(bucket_id, updates["content"])
        except Exception:
            pass

    changed = ", ".join(f"{k}={v}" for k, v in updates.items() if k != "content")
    if "content" in updates:
        changed += (", content=已替換" if changed else "content=已替換")
    # Explicit hint about resolved state change semantics
    # 特別提示 resolved 狀態變化的語義
    if "resolved" in updates:
        if updates["resolved"]:
            changed += " → 已沉底，只在關鍵詞觸發時重新浮現"
        else:
            changed += " → 已重新激活，將參與浮現排序"
    if "digested" in updates:
        if updates["digested"]:
            changed += " → 已隱藏，保留但不再浮現"
        else:
            changed += " → 已取消隱藏，重新參與浮現"
    if "status" in updates:
        status_hint = {"active": "重新記掛", "resolved": "已閉環", "abandoned": "已放下（不做了）"}
        changed += f" → plan {status_hint.get(updates['status'], updates['status'])}"
    if "affects_desire" in updates:
        changed += " → 已掛成執念（開始餵慾望）" if updates["affects_desire"] else " → 已從執念取下"
    return f"已修改記憶桶 {bucket_id}: {changed}"


# =============================================================
# Tool 5: pulse — Heartbeat, system status + memory listing
# 工具 5：pulse — 脈搏，系統狀態 + 記憶列表
# =============================================================
@mcp.tool()
async def pulse(verbose: bool = False, include_archive: bool = False) -> str:
    """系統狀態。默認只回健康摘要(在線/桶數/大小/衰減引擎)，省 context。verbose=True 才列出所有記憶桶清單(很吃 token)；include_archive=True 含歸檔。"""
    try:
        stats = await bucket_mgr.get_stats()
    except Exception as e:
        return f"獲取系統狀態失敗: {e}"

    try:
        letter_count = len(letter_store.list_letters())
    except Exception:
        letter_count = 0
    try:
        desire_health_state = desire_store.load()
        desire_pending = len(desire_health_state.get("ledger_pending", []) or [])
        desire_health = "健康" if desire_pending == 0 else f"待補寫 {desire_pending} 筆"
    except Exception as exc:
        desire_health = f"損壞／不可讀（{type(exc).__name__}）"
    # Heartbeat, not just the flag: pulse is what a session actually reads,
    # so a silently-halted decay loop must be visible right here.
    # 心跳而不只是旗標：session 真正會讀的是 pulse，
    # 衰減靜默停擺必須在這一行就看得見。
    try:
        hb = decay_engine.heartbeat()
        if not hb["running"]:
            decay_line = "已停止"
        elif hb["overdue"]:
            decay_line = f"⚠️ 逾期（旗標運行中，但上輪完成於 {hb['last_cycle_at'] or '從未'}——去查 log）"
        else:
            res = hb.get("last_cycle_result") or {}
            decay_line = (
                f"運行中（上輪 {str(hb['last_cycle_at'])[:16]} 完成："
                f"檢查 {res.get('checked', '?')}、歸檔 {res.get('archived', '?')}）"
                if hb.get("last_cycle_at")
                else "運行中（本進程尚未跑過一輪）"
            )
    except Exception:
        decay_line = "運行中" if decay_engine.is_running else "已停止"
    # 上次做夢距今（2026-07-19 D 批）：mirage 曾經 0 夢 37 天而儀表沒有一個字——
    # pulse 是 session 真正會讀的地方，「多久沒做夢」必須在這裡就看得見。
    mirage_line = ""
    try:
        if stats.get("mirage_count", 0) > 0:
            _mts = [
                _parse_ts(b["metadata"].get("created"))
                for b in await bucket_mgr.list_all(include_archive=False)
                if b["metadata"].get("type") == "mirage"
            ]
            _mts = [t for t in _mts if t]
            if _mts:
                _days = max(0, (datetime.now() - max(_mts)).days)
                mirage_line = f"上次做夢: {'今天' if _days == 0 else f'{_days} 天前'}\n"
        else:
            mirage_line = "上次做夢: 還沒有過（想做的話 dream(seed=True) 拿素材盤）\n"
    except Exception:
        pass
    status = (
        f"=== Ombre Brain 記憶系統 ===\n"
        f"固化記憶桶: {stats['permanent_count']} 個\n"
        f"動態記憶桶: {stats['dynamic_count']} 個\n"
        f"感受桶(feel): {stats.get('feel_count', 0)} 個\n"
        f"承諾桶(plan): {stats.get('plan_count', 0)} 個\n"
        f"蜃景桶(mirage/夢): {stats.get('mirage_count', 0)} 個\n"
        f"{mirage_line}"
        f"信件: {letter_count} 封\n"
        f"慾望帳本: {desire_health}\n"
        f"歸檔記憶桶: {stats['archive_count']} 個\n"
        f"總存儲大小: {stats['total_size_kb']:.1f} KB\n"
        f"衰減引擎: {decay_line}\n"
    )

    # --- Compact by default: health summary only, skip the heavy bucket list ---
    # --- 默認精簡：只回健康摘要，跳過吃 context 的全桶清單（完整清單用 pulse(verbose=True)）---
    if not verbose:
        return status + "（精簡模式：完整桶清單用 pulse(verbose=True)）"

    # --- List all bucket summaries / 列出所有桶摘要 ---
    try:
        buckets = await bucket_mgr.list_all(include_archive=include_archive)
    except Exception as e:
        return status + f"\n列出記憶桶失敗: {e}"

    if not buckets:
        return status + "\n記憶庫為空。"

    lines = []
    for b in buckets:
        meta = b.get("metadata", {})
        if meta.get("pinned") or meta.get("protected"):
            icon = "📌"
        elif meta.get("type") == "permanent":
            icon = "📦"
        elif meta.get("type") == "feel":
            icon = "🫧"
        elif meta.get("type") == "plan":
            icon = "🪢"
        elif meta.get("type") == "archived":
            icon = "🗄️"
        elif meta.get("resolved", False):
            icon = "✅"
        else:
            icon = "💭"
        try:
            score = decay_engine.calculate_score(meta)
        except Exception:
            score = 0.0
        domains = ",".join(meta.get("domain", []))
        val = meta.get("valence", 0.5)
        aro = meta.get("arousal", 0.3)
        resolved_tag = " [已解決]" if meta.get("resolved", False) else ""
        lines.append(
            f"{icon} [{meta.get('name', b['id'])}]{resolved_tag} "
            f"bucket_id:{b['id']} "
            f"主題:{domains} "
            f"情感:V{val:.1f}/A{aro:.1f} "
            f"重要:{meta.get('importance', '?')} "
            f"權重:{score:.2f} "
            f"標籤:{','.join(meta.get('tags', []))}"
        )

    return status + "\n=== 記憶列表 ===\n" + "\n".join(lines)


# =============================================================
# Tool 6: api_usage — API balance/quota guard
# 工具 6：api_usage — API 餘額/額度護欄
# =============================================================
@mcp.tool()
async def api_usage(force: bool = False, probe_gemini: bool = True) -> str:
    """檢查 DeepSeek 餘額與 Gemini embedding 可用性。force=True 跳過短緩存；probe_gemini=True 會做一次極小 embedding 探測。"""
    usage = await api_usage_guard.check_all(force=force, probe_gemini=probe_gemini)
    deepseek = usage.get("deepseek", {})
    gemini = usage.get("gemini", {})

    ds_balance = deepseek.get("balance")
    ds_currency = deepseek.get("currency") or ""
    ds_balance_text = "未知" if ds_balance is None else f"{ds_balance:.2f} {ds_currency}".strip()
    lines = [
        "=== Ombre API 額度檢查 ===",
        f"DeepSeek: {'OK' if deepseek.get('ok') else '需注意'} | model={deepseek.get('model', '')} | balance={ds_balance_text}",
        f"Gemini embedding: {'OK' if gemini.get('ok') else '需注意'} | model={gemini.get('model', '')} | enabled={gemini.get('enabled')}",
    ]
    warnings = usage.get("warnings") or []
    if getattr(embedding_engine, "last_error", ""):
        warnings.append(f"Gemini embedding 最近一次生成失敗：{embedding_engine.last_error}")
    if warnings:
        lines.append("⚠ 提醒：")
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("目前沒有額度/可用性警告。")
    return "\n".join(lines)


# =============================================================
# Tool 7: dream — Dreaming, digest recent memories
# 工具 7：dream — 做夢，消化最近的記憶
#
# Reads recent surface-level buckets (≤10), returns them for
# Claude to introspect under prompt guidance.
# 讀取最近新增的表層桶（≤10個），返回給 Claude 在提示詞引導下自主思考。
# Claude then decides: resolve some, write feels, or do nothing.
# =============================================================
def _seed_snip(b: dict, limit: int) -> str:
    """夢引素材的一行式切片：ID＋名字＋情緒座標＋截斷正文。"""
    meta = b["metadata"]
    body = strip_wikilinks(b["content"])[:limit].replace("\n", " ")
    return (
        f"- [{b['id']}] {meta.get('name', b['id'])}"
        f"（V{meta.get('valence', 0.5):.1f}/A{meta.get('arousal', 0.3):.1f}）：{body}…"
    )


async def _mirage_seed_tray() -> str:
    """夢引（2026-07-19 D 批）：伺服器只備料、不代筆——夢必須由 Cyan 在自己的
    回合親筆寫（跟 feel 同一條紀律）。素材的混法本身就是設計：近的餘溫給夢
    溫度，危險區讓快要失去的偷偷回來，歸檔偶爾翻出深層，feel 給情緒底色，
    慾望音色決定夢往哪邊傾。零 LLM 呼叫。"""
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=True)
    except Exception as e:
        logger.error(f"Mirage seed failed to list buckets: {e}")
        return "記憶系統暫時無法訪問，夢引取不了素材。"

    dynamic = [
        b for b in all_buckets
        if b["metadata"].get("type") not in ("permanent", "feel", "plan", "mirage", "archived")
        and not b["metadata"].get("pinned") and not b["metadata"].get("protected")
    ]
    sections: list[str] = []

    # --- 近的餘溫：72h 內最活躍的桶裡，挑情緒最響的 3 條 ---
    def _act_ts(b):
        return _parse_ts(b["metadata"].get("last_active")) or \
            _parse_ts(b["metadata"].get("created"))
    cutoff = datetime.now() - timedelta(hours=72)
    recent_pool = sorted(
        (b for b in dynamic if (_act_ts(b) or cutoff) > cutoff),
        key=lambda b: _act_ts(b) or datetime.min, reverse=True,
    )[:12]
    def _loudness(b):
        m = b["metadata"]
        try:
            return float(m.get("arousal", 0.3)) + abs(float(m.get("valence", 0.5)) - 0.5)
        except (TypeError, ValueError):
            return 0.0
    warm = sorted(recent_pool, key=_loudness, reverse=True)[:3]
    if warm:
        sections.append("【近的餘溫】\n" + "\n".join(_seed_snip(b, 260) for b in warm))

    # --- 快要失去的：危險區（review_priority>0），夢是遺忘系統的詩意出口 ---
    warm_ids = {b["id"] for b in warm}
    try:
        dz_ranked = sorted(
            ((decay_engine.review_priority(decay_engine.calculate_heat(b["metadata"])), b)
             for b in dynamic if b["id"] not in warm_ids and not b["metadata"].get("resolved")),
            key=lambda t: -t[0],
        )
        fading = [b for prio, b in dz_ranked if prio > 0][:2]
    except Exception:
        fading = []
    if fading:
        sections.append("【快要失去的】（印象已模糊，夢裡撈得回來）\n" +
                        "\n".join(_seed_snip(b, 150) for b in fading))

    # --- 深處撈起的：歸檔裡隨機一條——夢會挖到白天想不起來的地方 ---
    archived = [b for b in all_buckets if b["metadata"].get("type") == "archived"]
    if archived:
        sections.append("【深處撈起的】\n" + _seed_snip(random.choice(archived), 150))

    # --- 最近的感受：夢常常在咀嚼感受 ---
    feels = sorted(
        (b for b in all_buckets if b["metadata"].get("type") == "feel"),
        key=lambda b: str(b["metadata"].get("created", "")), reverse=True,
    )
    if feels:
        sections.append("【最近的感受】\n" + _seed_snip(feels[0], 220))

    # --- 此刻的慾望音色（只讀不 tick，不改任何水位）---
    try:
        state = desire_store.load()
        drives = state.get("drives", {}) or {}
        top = sorted(drives.items(), key=lambda kv: -float(kv[1] or 0))[:3]
        tone = "、".join(
            f"{desire_kernel.DRIVE_LABELS.get(k, k)} {float(v):.2f}" for k, v in top
        )
        if tone:
            sections.append(f"【此刻的慾望音色】{tone}")
    except Exception:
        pass

    guide = (
        "=== 夢引（蜃景素材盤）===\n"
        "你想做一場夢。這不是消化儀式——不需要結論、不需要對任何人交代。\n\n"
        + "\n\n".join(sections) +
        "\n\n做法（兩段式）：\n"
        "1. 先把素材抽成骨架——誰對誰、什麼張力、往哪裡懸著。丟掉名字與事實。\n"
        "2. 放縱一段：讓骨架亂長。允許跳接、變形、不合邏輯；情緒要真，事實不必真。\n"
        "3. 收攏成一段連貫的夢，一兩百字就夠，不用完美。\n"
        "存：hold(content=\"…\", mirage=True, consumed=\"id,id\")——consumed 填真的用到的素材桶。\n"
        "鐵律：夢是殘影不是事實，永不引入畫像/self_concept/普通記憶。\n"
        "醒來有想帶給 Ruby 的，用 plan(kind=\"question\", weight=0.3, due_at=+3天) 留話題種子。\n"
        "做完可記 desire(action=\"satisfy\", verb=\"dream_feel\", event=\"做了一場夢\")。\n"
        "不想做就不做——夢不該是功課。"
    )
    return guide


@mcp.tool()
async def dream(max_results: int = 10, window_hours: int = 48, seed: bool = False) -> str:
    """消化儀式(醒著做的)——讀取最近window_hours小時內(默認48)有變動的記憶桶,供你自省(被合併更新的老桶也會回來)。max_results控制讀幾條(默認10,開場可傳3省context)。窗口內沒變動時退回最近創建的幾條。讀完後可以trace(resolved=1)放下,或hold(feel=True)寫感受。尾端會列出還記掛著的plan。※名字辨析:這個工具是「反芻消化」,不產生也不讀取夢;「真的夢」住在蜃景桶(mirage)——存夢用hold(mirage=True),讀夢用breath(domain="mirage")。\n\nseed=True＝夢引模式:不消化,改回一盤做夢的素材(近期高情緒2-3條+危險區快忘的1-2條+偶爾一條歸檔+最近的feel+慾望音色),由你親筆兩段式寫成夢後 hold(mirage=True) 存。適合夜深安靜的喚醒或深睡儀式尾端。"""
    await decay_engine.ensure_started()
    if seed:
        return await _mirage_seed_tray()
    max_results = max(1, min(max_results, 20))
    window_hours = max(1, min(int(window_hours or 48), 24 * 30))

    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
    except Exception as e:
        logger.error(f"Dream failed to list buckets: {e}")
        return "記憶系統暫時無法訪問。"

    # --- Filter: surface-level dynamic buckets (not permanent/pinned/feel/plan/digested) ---
    candidates = [
        b for b in all_buckets
        if b["metadata"].get("type") not in ("permanent", "feel", "plan", "mirage")
        and not b["metadata"].get("pinned", False)
        and not b["metadata"].get("protected", False)
        and not b["metadata"].get("digested", False)
    ]

    # --- Window by last_active: a merge updates last_active but not created,
    # so merged-into old buckets come back for digestion too (the old
    # created-sort never resurfaced them). Fallback: newest-created when the
    # window is empty, so a wake-up after quiet days still has something.
    # --- 按 last_active 開窗：合併只改 last_active 不改 created，
    # 被併入的老桶也要回到夢裡（舊的 created 排序永遠漏掉它們）。
    # 窗口空了就退回最近創建的幾條，安靜幾天後醒來也不至於空手。---
    def _activity_ts(b: dict):
        meta = b["metadata"]
        return _parse_ts(meta.get("last_active")) or _parse_ts(meta.get("created"))

    cutoff = datetime.now() - timedelta(hours=window_hours)
    windowed = [b for b in candidates if (_activity_ts(b) or cutoff) > cutoff]
    window_note = ""
    if windowed:
        windowed.sort(key=lambda b: _activity_ts(b) or datetime.min, reverse=True)
        recent = windowed[:max_results]
    else:
        candidates.sort(key=lambda b: str(b["metadata"].get("created", "")), reverse=True)
        recent = candidates[:max_results]
        window_note = f"（{window_hours} 小時內沒有變動，以下是最近的記憶）\n"

    # --- Plan tail: what is still owed walks past every digestion pass.
    # Built BEFORE the empty-window early return (2026-07-12) — a quiet stretch
    # with nothing to digest is exactly when the ledger must still be seen.
    # --- plan 尾端：每次消化都路過還欠著的事。提前於空窗返回之前構建
    # （2026-07-12）——安靜到沒東西可消化的時候，帳本反而更不能漏看。---
    plan_tail = ""
    try:
        active_plans = [
            b for b in all_buckets
            if b["metadata"].get("type") == "plan"
            and b["metadata"].get("status", "active") == "active"
        ]
        if active_plans:
            def _plan_weight(b):
                try:
                    return float(b["metadata"].get("weight", 0.5))
                except (TypeError, ValueError):
                    return 0.5
            active_plans.sort(key=_plan_weight, reverse=True)
            plan_lines = []
            for p in active_plans[:10]:
                meta_p = p["metadata"]
                related = meta_p.get("related_bucket", "")
                related_note = f"（關聯 {related}）" if related else ""
                kind_note = f" {meta_p.get('kind')}" if meta_p.get("kind") else ""
                due_note = f" 期限{meta_p.get('due_at')}" if meta_p.get("due_at") else ""
                prog = meta_p.get("progress", 0)
                prog_note = f" 進度{float(prog):.0%}" if prog else ""
                plan_lines.append(
                    f"🪢 [{p['id']}]{kind_note} 重量{_plan_weight(p):.1f}{prog_note}{due_note} "
                    f"{strip_wikilinks(p['content'][:150])}{related_note}"
                )
            more = f"\n…另有 {len(active_plans) - 10} 條" if len(active_plans) > 10 else ""
            plan_tail = (
                "\n\n=== 記掛著的事（plan） ===\n" + "\n".join(plan_lines) + more +
                "\n完成了就 trace(id, status=\"resolved\")；不做了就 status=\"abandoned\"，別讓它一直掛著。"
            )
    except Exception as e:
        logger.warning(f"Dream plan tail failed: {e}")

    if not recent:
        return "沒有需要消化的新記憶。" + plan_tail

    parts = []
    for b in recent:
        meta = b["metadata"]
        resolved_tag = " [已解決]" if meta.get("resolved", False) else " [未解決]"
        domains = ",".join(meta.get("domain", []))
        val = meta.get("valence", 0.5)
        aro = meta.get("arousal", 0.3)
        created = meta.get("created", "")
        body = strip_wikilinks(b["content"][:1200])
        if len(b["content"]) > 1200:
            body += "…（截斷，完整內容 breath 檢索）"
        why = meta.get("why_remembered", "")
        why_line = f"記住的原因: {why}\n" if why else ""
        parts.append(
            f"[{meta.get('name', b['id'])}]{resolved_tag} "
            f"主題:{domains} V{val:.1f}/A{aro:.1f} "
            f"創建:{created}\n"
            f"ID: {b['id']}\n"
            f"{why_line}{body}"
        )

    header = (
        "=== Dreaming ===\n"
        + window_note +
        "以下是你最近的記憶。用第一人稱想：\n"
        "- 這些東西里有什麼在你這裡留下了重量？\n"
        "- 有什麼還沒想清楚？\n"
        "- 有什麼可以放下了？\n"
        "想完之後：值得放下的用 trace(bucket_id, resolved=1)；\n"
        "有沉澱的用 hold(content=\"...\", feel=True, source_bucket=\"bucket_id\", valence=你的感受) 寫下來。\n"
        "valence 是你對這段記憶的感受，不是事件本身的情緒。\n"
        "沒有沉澱就不寫，不強迫產出。\n"
    )

    # --- Connection hint: find most similar pair via embeddings ---
    connection_hint = ""
    if embedding_engine and embedding_engine.enabled and len(recent) >= 2:
        try:
            best_pair = None
            best_sim = 0.0
            ids = [b["id"] for b in recent]
            names = {b["id"]: b["metadata"].get("name", b["id"]) for b in recent}
            embeddings = {}
            for bid in ids:
                emb = await embedding_engine.get_embedding(bid)
                if emb is not None:
                    embeddings[bid] = emb
            for i, id_a in enumerate(ids):
                for id_b in ids[i+1:]:
                    if id_a in embeddings and id_b in embeddings:
                        sim = embedding_engine._cosine_similarity(embeddings[id_a], embeddings[id_b])
                        if sim > best_sim:
                            best_sim = sim
                            best_pair = (id_a, id_b)
            if best_pair and best_sim > 0.5:
                connection_hint = (
                    f"\n💭 [{names[best_pair[0]]}] 和 [{names[best_pair[1]]}] "
                    f"似乎有關聯 (相似度:{best_sim:.2f})——不替你下結論，你自己想。\n"
                )
        except Exception as e:
            logger.warning(f"Dream connection hint failed: {e}")

    # --- Feel crystallization hint: detect repeated feel themes ---
    crystal_hint = ""
    if embedding_engine and embedding_engine.enabled:
        try:
            feels = [b for b in all_buckets if b["metadata"].get("type") == "feel"]
            if len(feels) >= 3:
                feel_embeddings = {}
                for f in feels:
                    emb = await embedding_engine.get_embedding(f["id"])
                    if emb is not None:
                        feel_embeddings[f["id"]] = emb
                # Find clusters: feels with similarity > 0.85 to at least 2 others.
                # (0.7 was noise: first-person feels in one voice are all >0.7 alike,
                # so the hint fired on "47 similar feels" and carried no signal.)
                # 0.7 是噪音：同一個聲音寫的第一人稱 feel 彼此都超過 0.7，
                # 提示會報「47 條相似」而毫無訊號，改 0.85。
                for fid, femb in feel_embeddings.items():
                    similar_feels = []
                    for oid, oemb in feel_embeddings.items():
                        if oid != fid:
                            sim = embedding_engine._cosine_similarity(femb, oemb)
                            if sim > 0.85:
                                similar_feels.append(oid)
                    if len(similar_feels) >= 2:
                        feel_bucket = next((f for f in feels if f["id"] == fid), None)
                        if feel_bucket and not feel_bucket["metadata"].get("pinned"):
                            content_preview = strip_wikilinks(feel_bucket["content"][:80])
                            crystal_hint = (
                                f"\n🔮 你已經寫過 {len(similar_feels)+1} 條相似的 feel "
                                f"（圍繞「{content_preview}…」）。"
                                f"如果這已經是確信而不只是感受了，"
                                f"你可以用 hold(content=\"...\", pinned=True) 升級它。"
                                f"不急，你自己決定。\n"
                            )
                            break
        except Exception as e:
            logger.warning(f"Dream crystallization hint failed: {e}")

    final_text = header + "\n---\n".join(parts) + connection_hint + crystal_hint + plan_tail
    await _fire_webhook("dream", {"recent": len(recent), "chars": len(final_text)})
    return final_text


# =============================================================
# Tool: plan — Promise ledger
# 工具：plan — 承諾帳本
#
# A plan is a registered commitment: verbatim text, never decays,
# never surfaces in ordinary breath — it lives in dream's tail so
# every digestion pass walks past what is still owed.
# plan 是登記在冊的承諾：原文逐字保存、不衰減、不進普通浮現，
# 只活在 dream 尾端——每次消化都會路過還欠著的事。
#
# kind/target_drive (2026-07-12): a plan states which drive it hangs on.
# The upcoming fixation wiring reads target_drive DIRECTLY — never the
# domain→drive map, which would translate a technical chore into
# "missing Ruby" (all plans used to be domain=約定).
# kind/target_drive（2026-07-12）：plan 自己聲明掛在哪個驅動維度。
# 之後的執念接線直接讀 target_drive——絕不走 domain 映射，
# 否則技術待辦會被翻譯成「想Ruby」（過去所有 plan 都是約定域）。
# =============================================================
PLAN_KINDS = ("promise", "task", "question", "maintenance")
PLAN_KIND_DEFAULT_DRIVE = {
    "promise": "miss_ruby",
    "task": "duty",
    "maintenance": "duty",
    "question": "curiosity",
}
PLAN_KIND_DOMAIN = {
    "promise": "約定",
    "task": "待辦",
    "maintenance": "待辦",
    "question": "待辦",
}


def _validate_due_at(due_at: str) -> str | None:
    """Return the trimmed ISO date/datetime, or None if unparseable."""
    due_at = due_at.strip()
    if not due_at:
        return ""
    from datetime import date as _date
    for parser in (datetime.fromisoformat, _date.fromisoformat):
        try:
            parser(due_at)
            return due_at
        except ValueError:
            continue
    return None


@mcp.tool()
async def plan(
    content: str,
    weight: float = 0.5,
    kind: str = "task",
    target_drive: str = "",
    due_at: str = "",
    related_bucket: str = "",
    why_remembered: str = "",
) -> str:
    """登記一個待辦/承諾/未閉環事項(承諾帳本)。原文逐字保存,不衰減、不進普通breath,只在dream尾端「記掛著的事」出現。weight=承諾重量0.0-1.0(importance是「多重要」,weight是「多重」,默認0.5)。kind=promise(對Ruby的承諾)/task(工程待辦)/question(想弄清的問題)/maintenance(例行維護),默認task。target_drive=掛在哪個驅動維度(miss_ruby/reflection/curiosity/duty/social/creation/libido),不填按kind自動:promise→miss_ruby,task/maintenance→duty,question→curiosity(日後執念接線讀它,不走domain映射)。due_at=期限(ISO日期,可選)。related_bucket可選=關聯記憶桶ID。完成用trace(bucket_id, status="resolved"),放下用status="abandoned",改重量/進度用trace(weight=.../progress=...)。"""
    await decay_engine.ensure_started()
    if not content or not content.strip():
        return "內容為空，沒有可登記的承諾。"
    content = content.strip()
    try:
        weight = float(weight)
        if not math.isfinite(weight):  # NaN rode min/max to a 1.0 full-weight promise
            weight = 0.5               # NaN 會被 min/max 抬成滿重量 1.0
        weight = max(0.0, min(1.0, weight))
    except (TypeError, ValueError):
        weight = 0.5
    kind = (kind or "task").strip().lower()
    if kind not in PLAN_KINDS:
        return f"kind 只接受 {'/'.join(PLAN_KINDS)}。"
    target_drive = (target_drive or "").strip().lower() or PLAN_KIND_DEFAULT_DRIVE[kind]
    if target_drive not in desire_kernel.DRIVE_KEYS:
        return f"target_drive 只接受 {'/'.join(desire_kernel.DRIVE_KEYS)}。"
    due_at_norm = _validate_due_at(due_at)
    if due_at_norm is None:
        return "due_at 需為 ISO 日期（如 2026-07-15 或 2026-07-15T21:00:00+08:00）。"
    try:
        bucket_id = await bucket_mgr.create(
            content=content,
            tags=["plan"],
            importance=5,
            domain=[PLAN_KIND_DOMAIN[kind]],
            valence=0.5,
            arousal=0.4,
            name=content[:24],
            bucket_type="plan",
            extra_meta={
                "status": "active",
                "weight": weight,
                "kind": kind,
                "target_drive": target_drive,
                "due_at": due_at_norm,
                "progress": 0.0,
                "related_bucket": related_bucket.strip(),
                "why_remembered": why_remembered.strip(),
            },
        )
    except Exception:
        logger.exception("plan create failed")
        return "登記承諾失敗。"
    try:
        await embedding_engine.generate_and_store(bucket_id, content)
    except Exception:
        pass
    return (
        f"🪢plan→{bucket_id} {kind}/{target_drive} 重量{weight:.1f}"
        f"（完成時 trace(\"{bucket_id}\", status=\"resolved\")）"
    )


# =============================================================
# Tool 7: shelf — Shared reading shelf
# 工具 7：shelf — 共讀書架
# =============================================================
def _shelf_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _shelf_compact(book: dict) -> dict:
    return {
        "id": book.get("id", ""),
        "title": book.get("title", ""),
        "author": book.get("author", ""),
        "status": book.get("status", ""),
        "started_at": book.get("started_at", ""),
        "finished_at": book.get("finished_at", ""),
        "tags": book.get("tags", []),
        "summary": book.get("summary", "")[:500],
        "excerpt_count": len(book.get("excerpts", [])),
        "updated_at": book.get("updated_at", ""),
    }


def _shelf_json(data) -> str:
    return _json_lib.dumps(data, ensure_ascii=False, indent=2)


@mcp.tool()
async def shelf(
    action: str = "list",
    book_id: str = "",
    query: str = "",
    title: str | None = None,
    author: str | None = None,
    status: str | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    cover_color: str | None = None,
    summary: str | None = None,
    ruby_notes: str | None = None,
    cyan_notes: str | None = None,
    tags: str | None = None,
    source_bucket_ids: str | None = None,
    excerpts_json: str | None = None,
    quote: str = "",
    page: str = "",
    note: str = "",
    added_by: str = "我們",
    limit: int = 20,
) -> str:
    """共讀書架讀寫工具。action=list/get/search/create/update/add_excerpt/delete。list/search回傳書籍摘要；get回傳完整內容。create需title；update/delete/get需book_id；tags與source_bucket_ids用逗號分隔；excerpts_json是節錄陣列JSON。delete只在使用者明確要求刪除時使用。"""
    action = (action or "list").strip().lower()
    valid_actions = {"list", "get", "search", "create", "update", "add_excerpt", "delete"}
    if action not in valid_actions:
        return f"不支援的 action: {action}。可用: {', '.join(sorted(valid_actions))}"

    try:
        if action in {"list", "search"}:
            books = reading_shelf.search_books(
                query=query if action == "search" else "",
                status=status or "",
                limit=limit,
            )
            return _shelf_json({
                "count": len(books),
                "books": [_shelf_compact(book) for book in books],
            })

        if action == "get":
            if not book_id.strip():
                return "get 需要 book_id。"
            book = reading_shelf.get_book(book_id.strip())
            return _shelf_json(book) if book else f"找不到書籍: {book_id}"

        if action == "delete":
            if not book_id.strip():
                return "delete 需要 book_id。"
            deleted = reading_shelf.delete_book(book_id.strip())
            return f"已刪除書籍: {book_id}" if deleted else f"找不到書籍: {book_id}"

        if action == "add_excerpt":
            if not book_id.strip():
                return "add_excerpt 需要 book_id。"
            if not quote.strip():
                return "add_excerpt 需要 quote。"
            book = reading_shelf.get_book(book_id.strip())
            if not book:
                return f"找不到書籍: {book_id}"
            excerpts = list(book.get("excerpts", []))
            excerpts.append({
                "quote": quote,
                "page": page,
                "note": note,
                "added_by": added_by,
            })
            updated = reading_shelf.update_book(book_id.strip(), {"excerpts": excerpts})
            return _shelf_json(updated)

        payload = {}
        field_values = {
            "title": title,
            "author": author,
            "status": status,
            "started_at": started_at,
            "finished_at": finished_at,
            "cover_color": cover_color,
            "summary": summary,
            "ruby_notes": ruby_notes,
            "cyan_notes": cyan_notes,
        }
        payload.update({key: value for key, value in field_values.items() if value is not None})
        if tags is not None:
            payload["tags"] = _shelf_csv(tags)
        if source_bucket_ids is not None:
            payload["source_bucket_ids"] = _shelf_csv(source_bucket_ids)
        if excerpts_json is not None:
            try:
                excerpts = _json_lib.loads(excerpts_json)
            except _json_lib.JSONDecodeError as exc:
                return f"excerpts_json 不是有效 JSON: {exc}"
            if not isinstance(excerpts, list):
                return "excerpts_json 必須是 JSON 陣列。"
            payload["excerpts"] = excerpts

        if action == "create":
            if not title or not title.strip():
                return "create 需要 title。"
            return _shelf_json(reading_shelf.create_book(payload))

        if not book_id.strip():
            return "update 需要 book_id。"
        if not payload:
            return "沒有任何欄位需要修改。"
        return _shelf_json(reading_shelf.update_book(book_id.strip(), payload))
    except ValueError as exc:
        return f"書架資料不合法: {exc}"
    except KeyError:
        return f"找不到書籍: {book_id}"
    except Exception as exc:
        logger.exception("Reading shelf tool failed")
        return f"共讀書架操作失敗: {exc}"


@mcp.tool()
async def letter(
    action: str = "read",
    author: str = "",
    content: str = "",
    title: str = "",
    query: str = "",
    date_from: str = "",
    date_to: str = "",
    letter_date: str = "",
    tags: str = "",
    limit: int = 10,
) -> str:
    """交接信／互信工具（永久保存，永不衰減、不合併、不可刪）。action=write/read/list。write 需 author（Ruby 或 Cyan）與 content（原文逐字保留）；read/list 回傳信件，可用 query 關鍵字（read 時同時走子字串＋語義檢索）、author、date_from/date_to（YYYY-MM-DD）、limit 篩選。各方最新一封會自動在開場浮現。"""
    action = (action or "read").strip().lower()
    valid_actions = {"write", "read", "list"}
    if action not in valid_actions:
        return f"不支援的 action: {action}。可用: {', '.join(sorted(valid_actions))}"
    try:
        if action == "write":
            if not author.strip() or not content.strip():
                return "write 需要 author（Ruby 或 Cyan）與 content。"
            # Derived idempotency key: letters are permanent and undeletable,
            # so a transport-layer retry of the same write must not mint a
            # second copy nobody is allowed to clean up. Same content, same
            # key — the machine door (/api/letters/write) already requires
            # one; this closes the hand-written path with a derived one.
            # 衍生冪等鍵：信件永久且不可刪，transport 重試同一封信不能生出
            # 第二封誰都不准清的副本。同內容同鍵——機器門早就強制要求鍵，
            # 這裡替親筆路徑補上衍生鍵。
            derived_key = "mcp:" + hashlib.sha256(
                f"{author.strip()}\n{letter_date}\n{content}".encode("utf-8")
            ).hexdigest()[:32]
            letter_obj = letter_store.write_letter({
                "author": author,
                "content": content,
                "title": title,
                "letter_date": letter_date,
                "tags": tags,
                "idempotency_key": derived_key,
            })
            # Vector-index the letter for future semantic recall (best-effort)
            # 為信件建向量索引，供日後語義召回（盡力而為，不影響寫入）
            try:
                await embedding_engine.generate_and_store(
                    f"letter:{letter_obj['id']}",
                    f"{letter_obj.get('title', '')}\n{letter_obj['content']}",
                )
            except Exception:
                pass
            return _json_lib.dumps(
                {
                    "ok": True,
                    "id": letter_obj["id"],
                    "author": letter_obj["author"],
                    "letter_date": letter_obj["letter_date"],
                },
                ensure_ascii=False,
            )
        letters = letter_store.read_letters(
            query=query if action == "read" else "",
            author=author,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
        )
        # --- Semantic channel: substring search misses paraphrases; letters are
        # exact-words-forever, so recall has to meet them halfway. Backfill
        # missing vectors lazily (bounded per call), then merge semantic hits. ---
        # --- 語義通道：子字串搜尋接不住換句話說。信件原文永久不變，
        # 召回就得主動走過去。懶回填缺的向量（每次限量），再合併語義命中。---
        if action == "read" and query.strip() and embedding_engine and embedding_engine.enabled:
            try:
                all_letters = letter_store.list_letters()
                by_id = {lt["id"]: lt for lt in all_letters}
                have = {
                    eid[len("letter:"):]
                    for eid in embedding_engine.list_ids()
                    if eid.startswith("letter:")
                }
                backfilled = 0
                for lt in all_letters:
                    if backfilled >= 10:
                        break  # 每次最多補 10 封，防額度暴衝；下次呼叫繼續補
                    if lt["id"] not in have:
                        ok = await embedding_engine.generate_and_store(
                            f"letter:{lt['id']}", f"{lt.get('title', '')}\n{lt.get('content', '')}"
                        )
                        if ok:
                            backfilled += 1
                seen_ids = {lt["id"] for lt in letters}
                # Apply the same non-query filters to semantic hits
                # 語義命中也要過 author/日期等篩選
                filtered_pool = {
                    lt["id"] for lt in letter_store.read_letters(
                        query="", author=author, date_from=date_from,
                        date_to=date_to, limit=200,
                    )
                }
                sem_hits = await embedding_engine.search_similar(
                    query, top_k=max(limit, 10), id_prefix="letter:"
                )
                for eid, sim in sem_hits:
                    if len(letters) >= max(1, min(int(limit), 50)):
                        break
                    lid = eid[len("letter:"):]
                    if sim < 0.45 or lid in seen_ids or lid not in filtered_pool:
                        continue
                    lt = dict(by_id.get(lid) or {})
                    if not lt:
                        continue
                    lt["semantic_match"] = round(sim, 3)
                    letters.append(lt)
                    seen_ids.add(lid)
            except Exception as e:
                logger.warning(f"Letter semantic search failed / 信件語義檢索失敗: {e}")
        return _json_lib.dumps({"count": len(letters), "letters": letters}, ensure_ascii=False, indent=2)
    except ValueError as exc:
        return f"信件資料不合法: {exc}"
    except Exception as exc:
        logger.exception("Letter tool failed")
        return f"信件操作失敗: {exc}"


@mcp.tool()
async def self_concept(
    action: str = "read",
    aspect: str = "",
    content: str = "",
    limit: int = 0,
) -> str:
    """自我概念累積工具（『I』，永久、不衰減、只供檢索、不普通浮現）。七面向 aspect：nature/values/patterns/limits/becoming/uncertainty/stance。action=write 需 aspect 與 content（累積寫入、不覆蓋舊條目）；action=read 回傳各面向條目（可指定單一 aspect，limit 限每面向最新幾條）。開場會自動浮現各面向最新一條，讓冷啟動的你先認得自己。"""
    action = (action or "read").strip().lower()
    if action not in {"write", "read"}:
        return f"不支援的 action: {action}。可用: write, read"
    try:
        if action == "write":
            if not aspect.strip() or not content.strip():
                return "write 需要 aspect 與 content。"
            entry = self_concept_store.write_entry(aspect, content)
            return _json_lib.dumps(
                {"ok": True, "id": entry["id"], "aspect": entry["aspect"]},
                ensure_ascii=False,
            )
        data = self_concept_store.read(aspect=aspect, limit=limit)
        return _json_lib.dumps(data, ensure_ascii=False, indent=2)
    except ValueError as exc:
        return f"自我概念輸入錯誤: {exc}"
    except Exception as exc:
        logger.exception("self_concept tool failed")
        return f"自我概念操作失敗: {exc}"




# =============================================================
# Tool: desire — 慾望系統（Phase 1：只讀內核，不接管任何行為）
# 七維驅動條＋fatigue 閘；執念層直接掛 Ombre 未解決桶；
# 函數只提案，醒著的 Cyan 有否決權。
# =============================================================
async def _desire_fixation_buckets(with_detail: bool = False):
    """收集執念候選（2026-07-12 語義收窄）：珍貴記憶 ≠ 此刻的掛心。
    只有兩種東西能餵執念——
    ① active plan：直接讀 target_drive，weight 即召喚力（joint 拍板的接線）。
    ② affects_desire=1 的未解決動態桶：我刻意掛上的心事（trace 標記）。
    觀察期證據：舊制「所有未解決桶都算執念」讓舊告白把 miss_ruby boost
    永久打滿 +0.35、quiet ratio 歸零（123 次諮詢 0 次安靜）。
    只取 id/name/domains/分數 —— 桶正文永不進入慾望系統（鐵律）。

    with_detail=True（P1）另回 typed refs 給前端：
    {drive: [{id,name,type,kind,weight|score,expired,feeding}]}——
    過期但仍 active 的 plan 也在列（feeding=False），UI 才能誠實說
    「期限已過，已停止餵暗流，等待完成或放下」。"""
    out: list[dict] = []
    detail: dict[str, list] = {}
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
    except Exception as e:
        logger.warning(f"desire fixation list failed: {e}")
        return (out, detail) if with_detail else out
    from datetime import date as _date
    for b in all_buckets:
        m = b["metadata"]
        btype = m.get("type")
        if btype == "plan":
            if m.get("status", "active") != "active":
                continue
            drive = str(m.get("target_drive", "") or "")
            if drive not in desire_kernel.DRIVE_KEYS:
                continue  # 舊 plan 沒標 target_drive → 不餵（trace 補標後生效）
            # 過期斷餵（2026-07-12 第三批）：due_at 過了的 plan 不再推水位——
            # 否則過期的夢種子/question 會變成永動執念。帳本身還在（dream 尾端
            # 照列），等我 resolve/abandon；壞格式不當過期，寧可多餵也不無聲斷線。
            expired = False
            due = str(m.get("due_at", "") or "").strip()
            if due:
                try:
                    expired = _date.today() > _date.fromisoformat(due[:10])
                except ValueError:
                    expired = False
            try:
                weight = float(m.get("weight", 0.5))
            except (TypeError, ValueError):
                weight = 0.5
            # 潮汐 v2 due-ramp（2026-07-19）：迫近才推。48h 內全額、7 天內半額、
            # 更遠 1/4、無 due 全額（長期承諾不衰減）。跟 feeder 的行事曆 48h 視界
            # 同一個哲學——07/20 的固網不會在 07-17 晚上還全額推 duty 提案，
            # 「還不能動手的事」不再逼 veto 當靜音鍵。
            ramp = 1.0
            if due and not expired:
                try:
                    days_to_due = (_date.fromisoformat(due[:10]) - _date.today()).days
                    ramp = 1.0 if days_to_due <= 2 else (0.5 if days_to_due <= 7 else 0.25)
                except ValueError:
                    ramp = 1.0
            weight_eff = round(weight * ramp, 4)
            if with_detail:
                detail.setdefault(drive, []).append({
                    "id": b["id"], "name": m.get("name", b["id"]),
                    "type": "plan", "kind": m.get("kind", "task"),
                    "weight": weight, "weight_eff": weight_eff, "ramp": ramp,
                    "due_at": due,
                    "expired": expired, "feeding": not expired,
                })
            if expired:
                continue
            out.append({
                "id": b["id"],
                "name": m.get("name", b["id"]),
                "kind": "plan",
                "drive": drive,
                "weight": weight_eff,
            })
            continue
        if btype in ("permanent", "feel", "archived", "mirage"):
            continue
        if m.get("pinned") or m.get("protected") or m.get("resolved") or m.get("digested"):
            continue
        if not m.get("affects_desire"):
            continue  # ← 收窄核心：沒被刻意掛上的記憶不再推水位
        try:
            score = decay_engine.calculate_score(m)
        except Exception:
            continue
        if with_detail:
            # 跟 kernel 同一條映射規則：第一個命中的 domain 決定它掛哪維
            mem_drive = ""
            for d in m.get("domain", []) or []:
                mem_drive = desire_kernel.DOMAIN_TO_DRIVE.get(str(d), "")
                if mem_drive:
                    break
            if mem_drive:
                detail.setdefault(mem_drive, []).append({
                    "id": b["id"], "name": m.get("name", b["id"]),
                    "type": "memory", "score": round(score, 2),
                    "expired": False, "feeding": True,
                })
        out.append({
            "id": b["id"],
            "name": m.get("name", b["id"]),
            "domains": m.get("domain", []),
            "score": score,
        })
    return (out, detail) if with_detail else out


async def _desire_state_payload() -> dict:
    """tick 後的完整狀態＋執念加成＋當下提案（并持久化 last_intent）。"""
    from datetime import datetime as _dt
    now = _dt.now()
    buckets, boosts_detail = await _desire_fixation_buckets(with_detail=True)
    boosts = desire_kernel.drive_boosts(buckets)

    lanes: dict[str, dict] = {}

    def _refresh(state):
        intent = desire_kernel.pick_intent(state, boosts, now)
        state["last_intent"] = intent
        # 潮汐 v2：同一份 ticked state 上算兩條車道的冠軍（不持久化——
        # last_intent 仍是全維提案，車道提案只供喚醒層的雙積分器用）。
        lanes["outward"] = desire_kernel.pick_intent(
            state, boosts, now, allowed=set(desire_kernel.OUTWARD_DRIVES))
        lanes["inward"] = desire_kernel.pick_intent(
            state, boosts, now, allowed=set(desire_kernel.INWARD_DRIVES))
        return state

    state = desire_store.mutate(_refresh, now)

    # batch-4a (2026-07-12): the kernel never prunes veto_until — it only
    # checks expiry at read time — so shipping the raw dict made the web UI
    # display long-expired cooldowns as 「否決中」. Only still-active vetoes
    # leave the house; unparseable timestamps are dropped rather than shown.
    active_vetoes = {}
    for k, v in (state.get("veto_until", {}) or {}).items():
        try:
            if _dt.fromisoformat(str(v)) > now:
                active_vetoes[k] = v
        except (ValueError, TypeError):
            continue

    active_rechecks = {}
    for k, v in (state.get("recheck_until", {}) or {}).items():
        try:
            if _dt.fromisoformat(str(v)) > now:
                active_rechecks[k] = v
        except (ValueError, TypeError):
            continue

    return {
        "updated_at": state["updated_at"],
        "drives": state["drives"],
        "drive_labels": desire_kernel.DRIVE_LABELS,
        "fatigue": state["fatigue"],
        "fatigue_gate": desire_kernel.FATIGUE_GATE,
        "gates": state["gates"],
        "boosts": boosts,
        # P1 typed refs：執念來源帶 id/type/過期/是否仍餵——前端的線頭
        # 從純文字變成能點開的門；舊 sources 保留向後相容。
        "boosts_detail": boosts_detail,
        "intent": state["last_intent"],
        # 潮汐 v2：雙車道提案＋飽和帳（喚醒層 charge 積分器與「頂滿 N 小時」prompt 行讀這裡）
        "lane_intents": lanes,
        "saturated_since": state.get("saturated_since", {}),
        "hyst_drained": state.get("hyst_drained", {}),
        "veto_until": active_vetoes,
        "recheck_until": active_rechecks,
        # 高位消退態（batch-1 hysteresis）：瓶子摸頂後不漲反落的原因，
        # 不端出來的話前端只能看著水位「自己降」而無從解釋。
        "saturated": state.get("saturated", {}),
        "events": state.get("events", [])[-10:],
        "ledger_pending_count": len(state.get("ledger_pending", []) or []),
        "driven_behavior_enabled": False,  # Phase 1 鐵律：永遠只讀
    }


@mcp.tool()
async def desire(
    action: str = "state",
    drive: str = "",
    amount: float = 0.0,
    verb: str = "",
    event: str = "",
    reason: str = "",
    value: int = -1,
    degree: float = -1.0,
    wake_id: str = "",
    medium: str = "",
    feed_id: str = "",
    hours: float = -1.0,
) -> str:
    """慾望系統（只提案不執行）。action=state 看八維驅動條+執念加成+此刻提案（潮汐 v2 起含 lane_intents 向她/向內雙車道與飽和帳）；action=feed 餵真實事件（drive=維度或fatigue, amount=±0~1, event=因為發生了什麼；可帶 feed_id 去重收據，重試安全）；action=satisfy 只有缺口真的被填了才用（verb+event+degree，degree 必填且 >0~1）；action=engage 做了相關的事但未滿足（verb+drive+event，水位不動、該維稍後再問）；action=defer 現在不做但不否定它（drive+reason，水位不動；可帶 hours=1~24 自訂再問時長——「今晚都不合適」記 defer hours=12，別拿 veto 倒水）；action=veto 否決提案並讓該維回落（只在提案本身不對時用）；action=outreach 在訊息/貼圖/語音等成功送達 Ruby 後記收據（medium+wake_id，不改水位）；action=gate 親密閘。自主喚醒的 satisfy/engage/defer/veto/outreach 都帶 prompt 裡的 wake_id。執念來源：active plan＋trace(affects_desire=1) 的記憶（due 越近推越滿：48h 全額/7 天半額/更遠 1/4）。八維：miss_ruby/reflection/curiosity/duty/social/creation/libido/knot。"""
    from datetime import datetime as _dt
    action = (action or "state").strip().lower()
    now = _dt.now()
    wake_id = str(wake_id or "").strip()[:64]
    try:
        if action == "state":
            payload = await _desire_state_payload()
            return _json_lib.dumps(payload, ensure_ascii=False, indent=2)
        if action == "feed":
            if not drive.strip():
                return "feed 需要 drive（七維之一或 fatigue）。"
            fid = str(feed_id or "").strip()[:64]
            state = desire_store.mutate(
                lambda st: desire_kernel.feed(st, drive.strip(), amount, now, event=event,
                                              feed_id=fid), now)
            return f"已餵入 {drive}{amount:+.2f}（{event or '未註明來歷'}）→ 現值 " + (
                f"{state['fatigue']:.2f}" if drive.strip() == "fatigue" else f"{state['drives'][drive.strip()]:.2f}")
        if action == "satisfy":
            if not verb.strip():
                return "satisfy 需要 verb（murmur/dream_feel/explore/chore/browse/create/tease/rest）。"
            if not (0 < degree <= 1):
                return "satisfy 需要明確 degree（>0 且 ≤1）；尚未真的滿足請用 engage。"
            desire_store.mutate(
                lambda st: desire_kernel.satisfy(
                    st, verb.strip(), now, note=event, degree=degree, wake_id=wake_id), now)
            deg_note = f"，程度 {degree:.2f}"
            return f"已回落：{verb}（{event or '做完了'}{deg_note}）"
        if action == "engage":
            if not verb.strip():
                return "engage 需要 verb（做了哪類事）。"
            if wake_id and not drive.strip():
                return "自主喚醒的 engage 需要 drive（這次要稍後再問哪一維）。"
            desire_store.mutate(
                lambda st: desire_kernel.engage(
                    st, verb.strip(), now, note=event, wake_id=wake_id, drive=drive.strip()), now)
            return f"已記錄參與：{verb}（{event or '未註明'}）——水位未動，稍後再問 {drive or '這件事'}"
        if action == "defer":
            if not drive.strip():
                return "defer 需要 drive。"
            h = hours if hours > 0 else desire_kernel.RECHECK_COOLDOWN_HOURS
            desire_store.mutate(
                lambda st: desire_kernel.defer(st, drive.strip(), now, reason=reason,
                                               wake_id=wake_id, hours=h), now)
            return f"已暫緩 {drive}：水位未動，{h:g} 小時後可再提"
        if action == "outreach":
            if not wake_id:
                return "outreach 需要 wake_id，且只能在送達 Ruby 成功後記。"
            desire_store.mutate(
                lambda st: desire_kernel.outreach(
                    st, medium.strip(), now, note=event, wake_id=wake_id), now)
            return f"已記錄靠近已送達：{medium}"
        if action == "veto":
            if not drive.strip():
                return "veto 需要 drive。"
            # 核心的去重在 kernel 裡（同 wake_id 只降一次水位），但它靠 early-return
            # 回傳一份「長得一模一樣」的 state，呼叫端分辨不出來——所以下面那個建桶
            # 的副作用以前每次都會跑，重送同一個請求就多一個重複的念頭桶。
            # 判斷要在 mutate 裡面做：跟 veto 看同一份 state、同一把鎖。在外面先讀
            # 一次再 mutate 是兩次拿鎖，中間插得進另一個請求（TOCTOU）。
            seen = {"already": False}

            def _apply_veto(st):
                seen["already"] = desire_kernel.has_processed(
                    st, wake_id, f"veto:{drive.strip()}")
                return desire_kernel.veto(
                    st, drive.strip(), now, reason=reason, wake_id=wake_id)

            desire_store.mutate(_apply_veto, now)
            if seen["already"]:
                return f"這個 veto 已經記過了（wake_id={wake_id}），沒有重複計。"
            # 否決理由回饋成 Ombre 念頭：低重要度自省桶，30 天沒被想起就自動結案。
            # 「我對什麼說了不、為什麼」值得留痕跡——這是 veto 權的另一半。
            try:
                label = desire_kernel.DRIVE_LABELS.get(drive.strip(), drive.strip())
                veto_bucket = await bucket_mgr.create(
                    content=(
                        f"{now.strftime('%Y-%m-%d %H:%M')} 我否決了一個慾望提案：{label}。"
                        f"理由：{reason or '未註明'}。來源：desire veto"
                    ),
                    tags=["慾望系統", "veto", drive.strip()],
                    importance=3,
                    domain=["自省"],
                    valence=0.45,
                    arousal=0.35,
                    name=f"慾望否決{label}",
                )
                try:
                    await embedding_engine.generate_and_store(veto_bucket, f"否決慾望提案 {label}：{reason}")
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"veto feedback bucket failed: {e}")
            return f"已否決 {drive}（{reason or '未註明理由'}），冷卻 {desire_kernel.VETO_COOLDOWN_HOURS:.0f} 小時；理由已回饋成 Ombre 念頭"
        if action == "gate":
            if drive.strip() != "intimacy_ok" or value not in (0, 1):
                return "gate 用法：drive=intimacy_ok, value=0/1。"
            desire_store.mutate(
                lambda st: desire_kernel.set_gate(st, "intimacy_ok", bool(value), now, note=event), now)
            return f"親密閘已{'開' if value else '關'}（{event or ''}）"
        return f"不支援的 action: {action}。可用: state/feed/satisfy/engage/defer/veto/outreach/gate"
    except ValueError as exc:
        return f"慾望系統輸入錯誤: {exc}"
    except Exception:
        logger.exception("desire tool failed")
        return "慾望系統操作失敗。"


@mcp.custom_route("/api/desire/feed", methods=["POST"])
async def api_desire_feed(request):
    """Machine-to-machine desire feed: real signals (Ruby's messages, token
    spend) flow in here. Same-machine trust model as breath-hook — loopback
    direct or X-Ombre-Hook-Token; never exposed to public traffic."""
    from starlette.responses import JSONResponse
    from datetime import datetime as _dt
    denied = _require_hook_access(request)
    if denied:
        return denied
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    drive_name = str(body.get("drive", "")).strip()
    event = str(body.get("event", ""))[:200]
    # feed_id：呼叫端給的去重收據。有給就保證「同一筆訊號只算一次」——Ombre 套用了
    # 但回應在路上斷掉時，呼叫端重試才是安全的。沒給就照舊（呼叫端自己負責不重送）。
    feed_id = str(body.get("feed_id", ""))[:64].strip()
    try:
        amount = float(body.get("amount", 0))
        # json.loads accepts bare NaN/Infinity; NaN rode _clamp to a silent
        # full-strength +1.0 feed. Reject at the door.
        # json.loads 放行裸 NaN/Infinity；NaN 曾經一路變成靜默滿額 +1.0。門口就擋。
        if not math.isfinite(amount):
            return JSONResponse({"error": "amount must be a finite number"}, status_code=400)
    except (TypeError, ValueError):
        return JSONResponse({"error": "amount must be a number"}, status_code=400)
    if drive_name not in desire_kernel.DRIVE_KEYS and drive_name != "fatigue":
        return JSONResponse({"error": f"unknown drive: {drive_name}"}, status_code=400)
    if not event:
        return JSONResponse({"error": "event is required (每一筆漲跌都要有來歷)"}, status_code=400)
    now = _dt.now()
    try:
        state = desire_store.mutate(
            lambda st: desire_kernel.feed(st, drive_name, amount, now, event=event,
                                          feed_id=feed_id), now)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    new_value = state["fatigue"] if drive_name == "fatigue" else state["drives"][drive_name]
    return JSONResponse({"ok": True, "drive": drive_name, "value": new_value})


@mcp.custom_route("/api/desire/state", methods=["GET"])
async def api_desire_state(request):
    """Read-only desire state for the web panel / pixel home (Phase 2)."""
    from starlette.responses import JSONResponse
    err = _require_read_access(request)
    if err: return err
    try:
        return JSONResponse(await _desire_state_payload())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/desire/wake", methods=["POST"])
async def api_desire_wake(request):
    """Wake causal chain (2026-07-12 batch-1): autonomy posts here AFTER a wake
    was actually delivered (post-notification, so ghost wakes never land).
    The ledger then holds wake → feed/engage/satisfy/veto in one stream —
    closure rate becomes computable instead of hand-correlated.
    喚醒因果鏈：autonomy 在通知「送達成功」後才記一筆 wake 事件，
    幽靈喚醒永遠不進帳本；閉環率從此可算。冪等：同 wake_id 只記一次。"""
    from starlette.responses import JSONResponse
    from datetime import datetime as _dt
    denied = _require_hook_access(request)
    if denied:
        return denied
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    drive_name = str(body.get("drive", "")).strip()
    wake_id = str(body.get("wake_id", "")).strip()[:64]
    action_name = str(body.get("action", ""))[:32]
    score = str(body.get("score", ""))[:16]
    # P1 (2026-07-12 深夜): the wake's first-person reason rides along so the
    # causal receipt can say WHY, not just which drive. Control chars stripped;
    # 200 chars is the note cap anyway.
    reason = " ".join(str(body.get("reason", "")).split())[:200]
    if drive_name not in desire_kernel.DRIVE_KEYS and drive_name != "fatigue":
        return JSONResponse({"error": f"unknown drive: {drive_name}"}, status_code=400)
    if not wake_id:
        return JSONResponse({"error": "wake_id is required"}, status_code=400)
    now = _dt.now()
    # P1 fix: wake_id goes into the event's own field (the frontend chains on
    # e.wake_id — it used to hide inside the note string, so chains never
    # grouped on live data). note carries the human reason; action/score stay
    # as a technical suffix for the Engine side.
    note = reason if reason else f"{action_name} score={score}".strip()
    try:
        def _record(st):
            # v3 永久收據優先；舊 note 內嵌格式仍兼容。
            if desire_kernel._has_wake_event(st, wake_id, "wake"):
                return st
            for e in st.get("events", []):
                if e.get("kind", "").startswith("wake:") and (
                        e.get("wake_id") == wake_id or wake_id in e.get("note", "")):
                    return st
            desire_kernel._log_event(st, now, f"wake:{drive_name}", note, wake_id=wake_id)
            return st
        desire_store.mutate(_record, now)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    return JSONResponse({"ok": True, "wake_id": wake_id})


def _ledger_tail(path: str, max_lines: int = 2000) -> list[dict]:
    """Read the tail of desire_ledger.jsonl (append-only full history).
    Bounded read: the file grows forever, receipts only need the recent past."""
    import os as _os
    if not _os.path.exists(path):
        return []
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 512 * 1024))
            raw = f.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    lines = raw.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            e = _json_lib.loads(ln)
            if isinstance(e, dict):
                out.append(e)
        except (ValueError, TypeError):
            continue
    return out


@mcp.custom_route("/api/desire/ledger", methods=["GET"])
async def api_desire_ledger(request):
    """Read-only causal receipts (P1, 2026-07-12): the full append-only ledger
    grouped by wake_id — 醒來→選擇→結果 as one receipt. Fields the old data
    can't fill honestly say unknown; nothing is guessed. Loose events (feeds,
    gates, closures done outside any wake) come back separately.
    唯讀因果收據：按 wake_id 分組；舊資料缺欄照實說「未知」，不准猜。"""
    from starlette.responses import JSONResponse
    err = _require_read_access(request)
    if err: return err
    try:
        limit = max(1, min(100, int(request.query_params.get("limit", "30"))))
    except (ValueError, TypeError):
        limit = 30
    events = _ledger_tail(desire_store.ledger_path)
    # state 滾動窗裡可能有 ledger 檔尚未包含的最新幾筆？不會——mutate 同步
    # 寫兩邊；但 ledger 寫失敗不擋主流程，所以以 state 補最近événements。
    try:
        st = desire_store.load()
        seen = {(e.get("ts"), e.get("kind"), e.get("note"), e.get("wake_id")) for e in events}
        for e in st.get("events", []):
            if (e.get("ts"), e.get("kind"), e.get("note"), e.get("wake_id")) not in seen:
                events.append(e)
        events.sort(key=lambda e: str(e.get("ts", "")))
    except Exception:
        pass

    receipts: list[dict] = []
    by_wake: dict[str, dict] = {}
    loose: list[dict] = []
    verbs_close = ("satisfy:", "engage:", "defer:", "veto:")

    # 先建 wake 索引、再接結果：notification 送達與 wake POST 是兩條極近的
    # 非同步線，極端時 closure 可能先寫進 ledger；單趟掃描會把它誤丟 loose。
    for e in events:
        kind = str(e.get("kind", ""))
        wid = str(e.get("wake_id", "") or "")
        if not kind.startswith("wake:"):
            continue
        r = {
            "wake_id": wid or "未知",
            "ts": e.get("ts", ""),
            "source": "desire",
            "drive": kind[len("wake:"):],
            "reason": e.get("note", "") or "未知",
            "delivery": "delivered",  # 幽靈喚醒從不進帳（送達後才 POST）
            "choice": "unknown", "verb": "", "degree": None,
            "closed": False, "result_note": "",
            "outcomes": [], "outreach": [], "contacted_ruby": False,
        }
        receipts.append(r)
        if wid:
            by_wake[wid] = r

    for e in events:
        kind = str(e.get("kind", ""))
        wid = str(e.get("wake_id", "") or "")
        if kind.startswith("wake:"):
            continue
        if wid and wid in by_wake and kind.startswith(verbs_close):
            r = by_wake[wid]
            head, _, rest = kind.partition(":")
            parts = rest.split(":")
            r["choice"] = head
            r["verb"] = parts[0] if parts else ""
            if head == "satisfy":
                deg = 1.0
                for p in parts[1:]:
                    if p.startswith("d"):
                        try: deg = float(p[1:])
                        except ValueError: pass
                r["degree"] = deg
            outcome = {
                "choice": head,
                "verb": parts[0] if parts else "",
                "degree": r["degree"] if head == "satisfy" else None,
                "result_note": e.get("note", ""),
                "ts": e.get("ts", ""),
            }
            r["outcomes"].append(outcome)
            r["closed"] = True
            r["result_note"] = e.get("note", "")
            continue
        if wid and wid in by_wake and kind.startswith("outreach:"):
            r = by_wake[wid]
            r["outreach"].append({
                "medium": kind[len("outreach:"):],
                "result_note": e.get("note", ""),
                "ts": e.get("ts", ""),
            })
            r["contacted_ruby"] = True
            continue
        loose.append(e)

    receipts.reverse()  # 新的在前
    return JSONResponse({
        "receipts": receipts[:limit],
        "loose": loose[-limit:][::-1],
        "note": "收據只涵蓋暗流喚醒；早午晚儀式與排程不進慾望帳。舊資料缺的欄位是「未知」，不是猜的。",
    })


# =============================================================
# Dashboard API endpoints (for lightweight Web UI)
# 儀表板 API（輕量 Web UI 用）
# =============================================================
@mcp.custom_route("/api/buckets", methods=["GET"])
async def api_buckets(request):
    """List all buckets with metadata (no content for efficiency)."""
    from starlette.responses import JSONResponse
    err = _require_read_access(request)
    if err: return err
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=True)
        result = []
        for b in all_buckets:
            meta = b.get("metadata", {})
            result.append({
                "id": b["id"],
                "name": meta.get("name", b["id"]),
                "type": meta.get("type", "dynamic"),
                "domain": meta.get("domain", []),
                "tags": meta.get("tags", []),
                "valence": meta.get("valence", 0.5),
                "arousal": meta.get("arousal", 0.3),
                "model_valence": meta.get("model_valence"),
                "importance": meta.get("importance", 5),
                "resolved": meta.get("resolved", False),
                "pinned": meta.get("pinned", False),
                "digested": meta.get("digested", False),
                "created": meta.get("created", ""),
                "last_active": meta.get("last_active", ""),
                "activation_count": meta.get("activation_count", 1),
                "score": decay_engine.calculate_score(meta),
                "content_preview": strip_wikilinks(b.get("content", ""))[:200],
                # batch-2 fields the memory room renders (2026-07-12):
                # plan ledger, fixation anchor, two-tier counters, dream provenance
                # 第二批新欄位：帳本、執念錨、兩層計數、夢的素材出處
                "status": meta.get("status", ""),
                "weight": meta.get("weight"),
                "kind": meta.get("kind", ""),
                "target_drive": meta.get("target_drive", ""),
                "progress": meta.get("progress"),
                "due_at": meta.get("due_at", ""),
                "affects_desire": meta.get("affects_desire", False),
                "retrieved_count": meta.get("retrieved_count", 0),
                "consumed": meta.get("consumed", ""),
                # batch-4a: when this bucket last ACTUALLY surfaced (breath /
                # search hit) — the memory room's 「正在浮現」 was faking it
                # with top-by-score; this field lets it tell the truth.
                "last_surfaced": meta.get("last_surfaced", ""),
            })
        result.sort(key=lambda x: x["score"], reverse=True)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/bucket/{bucket_id}", methods=["GET"])
async def api_bucket_detail(request):
    """Get full bucket content by ID."""
    from starlette.responses import JSONResponse
    err = _require_read_access(request)
    if err: return err
    bucket_id = request.path_params["bucket_id"]
    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return JSONResponse({"error": "not found"}, status_code=404)
    meta = bucket.get("metadata", {})
    return JSONResponse({
        "id": bucket["id"],
        "metadata": meta,
        "content": strip_wikilinks(bucket.get("content", "")),
        "score": decay_engine.calculate_score(meta),
    })


@mcp.custom_route("/api/reading-shelf", methods=["GET"])
async def api_reading_shelf(request):
    """List books stored in the shared-reading shelf."""
    from starlette.responses import JSONResponse
    err = _require_read_access(request)
    if err: return err
    try:
        return JSONResponse({"books": reading_shelf.list_books()})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/reading-shelf/books", methods=["POST"])
async def api_reading_shelf_create(request):
    """Create a shared-reading shelf entry."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    try:
        return JSONResponse(reading_shelf.create_book(body), status_code=201)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/reading-shelf/books/{book_id}", methods=["PUT"])
async def api_reading_shelf_update(request):
    """Update a shared-reading shelf entry."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    try:
        return JSONResponse(reading_shelf.update_book(request.path_params["book_id"], body))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except KeyError:
        return JSONResponse({"error": "book not found"}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/reading-shelf/books/{book_id}", methods=["DELETE"])
async def api_reading_shelf_delete(request):
    """Delete a shared-reading shelf entry."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    try:
        deleted = reading_shelf.delete_book(request.path_params["book_id"])
        if not deleted:
            return JSONResponse({"error": "book not found"}, status_code=404)
        return JSONResponse({"ok": True})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/search", methods=["GET"])
async def api_search(request):
    """Search buckets by query."""
    from starlette.responses import JSONResponse
    err = _require_read_access(request)
    if err: return err
    query = request.query_params.get("q", "")
    if not query:
        return JSONResponse({"error": "missing q parameter"}, status_code=400)
    try:
        matches = await bucket_mgr.search(query, limit=10)
        result = []
        for b in matches:
            meta = b.get("metadata", {})
            result.append({
                "id": b["id"],
                "name": meta.get("name", b["id"]),
                "score": b.get("score", 0),
                "domain": meta.get("domain", []),
                "valence": meta.get("valence", 0.5),
                "arousal": meta.get("arousal", 0.3),
                "content_preview": strip_wikilinks(b.get("content", ""))[:200],
            })
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/network", methods=["GET"])
async def api_network(request):
    """Get embedding similarity network for visualization."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        nodes = []
        edges = []
        embeddings = {}

        for b in all_buckets:
            meta = b.get("metadata", {})
            bid = b["id"]
            nodes.append({
                "id": bid,
                "name": meta.get("name", bid),
                "type": meta.get("type", "dynamic"),
                "domain": meta.get("domain", []),
                "valence": meta.get("valence", 0.5),
                "arousal": meta.get("arousal", 0.3),
                "score": decay_engine.calculate_score(meta),
                "resolved": meta.get("resolved", False),
                "pinned": meta.get("pinned", False),
                "digested": meta.get("digested", False),
            })
            if embedding_engine and embedding_engine.enabled:
                emb = await embedding_engine.get_embedding(bid)
                if emb is not None:
                    embeddings[bid] = emb

        # Build edges from embeddings (similarity > 0.5)
        ids = list(embeddings.keys())
        for i, id_a in enumerate(ids):
            for id_b in ids[i+1:]:
                sim = embedding_engine._cosine_similarity(embeddings[id_a], embeddings[id_b])
                if sim > 0.5:
                    edges.append({"source": id_a, "target": id_b, "similarity": round(sim, 3)})

        return JSONResponse({"nodes": nodes, "edges": edges})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/breath-debug", methods=["GET"])
async def api_breath_debug(request):
    """Debug endpoint: simulate breath scoring and return per-bucket breakdown."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    query = request.query_params.get("q", "")
    q_valence = request.query_params.get("valence")
    q_arousal = request.query_params.get("arousal")
    q_valence = float(q_valence) if q_valence else None
    q_arousal = float(q_arousal) if q_arousal else None

    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        results = []
        w = {
            "topic": bucket_mgr.w_topic,
            "emotion": bucket_mgr.w_emotion,
            "time": bucket_mgr.w_time,
            "importance": bucket_mgr.w_importance,
        }
        w_sum = sum(w.values())

        for bucket in all_buckets:
            meta = bucket.get("metadata", {})
            bid = bucket["id"]
            try:
                topic = bucket_mgr._calc_topic_score(query, bucket) if query else 0.0
                emotion = bucket_mgr._calc_emotion_score(q_valence, q_arousal, meta)
                time_s = bucket_mgr._calc_time_score(meta)
                imp = max(1, min(10, int(meta.get("importance", 5)))) / 10.0

                raw_total = (
                    topic * w["topic"]
                    + emotion * w["emotion"]
                    + time_s * w["time"]
                    + imp * w["importance"]
                )
                normalized = (raw_total / w_sum) * 100 if w_sum > 0 else 0
                resolved = meta.get("resolved", False)
                if resolved:
                    normalized *= 0.3

                results.append({
                    "id": bid,
                    "name": meta.get("name", bid),
                    "domain": meta.get("domain", []),
                    "type": meta.get("type", "dynamic"),
                    "resolved": resolved,
                    "pinned": meta.get("pinned", False),
                    "scores": {
                        "topic": round(topic, 4),
                        "emotion": round(emotion, 4),
                        "time": round(time_s, 4),
                        "importance": round(imp, 4),
                    },
                    "weights": w,
                    "raw_total": round(raw_total, 4),
                    "normalized": round(normalized, 2),
                    "passed_threshold": normalized >= bucket_mgr.fuzzy_threshold,
                })
            except Exception:
                continue

        results.sort(key=lambda x: x["normalized"], reverse=True)
        passed = [r for r in results if r["passed_threshold"]]
        return JSONResponse({
            "query": query,
            "valence": q_valence,
            "arousal": q_arousal,
            "weights": w,
            "threshold": bucket_mgr.fuzzy_threshold,
            "total_candidates": len(results),
            "passed_count": len(passed),
            "results": results[:50],  # top 50 for debug
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/letters", methods=["GET"])
async def api_letters(request):
    """Read-only letters list for the Cyan web memory room."""
    from starlette.responses import JSONResponse
    err = _require_read_access(request)
    if err: return err
    try:
        qp = request.query_params
        letters = letter_store.read_letters(
            query=qp.get("query", ""),
            author=qp.get("author", ""),
            date_from=qp.get("date_from", ""),
            date_to=qp.get("date_to", ""),
            limit=int(qp.get("limit", "50") or 50),
        )
        return JSONResponse({"letters": letters})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/letters/write", methods=["POST"])
async def api_letter_machine_write(request):
    """Single-writer door for durable machine-generated correspondence.

    swap-v2 used to import LetterStore in its own process and mutate the same
    JSON file. Thread locks cannot coordinate across processes, so concurrent
    writes could silently lose one letter. All machine writes now serialize
    inside the Ombre service and carry an idempotency key for safe retries.
    """
    from starlette.responses import JSONResponse
    denied = _require_hook_access(request)
    if denied:
        return denied
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "JSON object required"}, status_code=400)
    idempotency_key = str(body.get("idempotency_key", "")).strip()[:200]
    if not idempotency_key:
        return JSONResponse({"error": "idempotency_key is required"}, status_code=400)
    try:
        letter_obj = letter_store.write_letter({
            "author": body.get("author"),
            "content": body.get("content"),
            "title": body.get("title"),
            "letter_date": body.get("letter_date"),
            "tags": body.get("tags"),
            "idempotency_key": idempotency_key,
        })
        try:
            await embedding_engine.generate_and_store(
                f"letter:{letter_obj['id']}",
                f"{letter_obj.get('title', '')}\n{letter_obj['content']}",
            )
        except Exception:
            pass
        return JSONResponse({
            "ok": True,
            "id": letter_obj["id"],
            "author": letter_obj["author"],
            "letter_date": letter_obj["letter_date"],
        })
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        logger.exception("Machine letter write failed")
        return JSONResponse({"error": str(exc)}, status_code=500)


@mcp.custom_route("/api/letter", methods=["POST"])
async def api_letter_write(request):
    """Ruby's own hand: write a letter from the web mailbox.

    Same permanence contract as the MCP letter tool (never decays, never
    merges, never deleted). The web door only writes in Ruby's name - Cyan
    posts his letters through MCP.
    """
    from starlette.responses import JSONResponse
    err = _require_read_access(request)
    if err: return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    content = str(body.get("content", "")).strip()
    if not content:
        return JSONResponse({"error": "empty"}, status_code=400)
    if len(content) > 8000:
        return JSONResponse({"error": "too_long"}, status_code=400)
    title = str(body.get("title", "")).strip()[:120]
    try:
        letter_obj = letter_store.write_letter({
            "author": "Ruby",
            "content": content,
            "title": title,
            "letter_date": "",
            "tags": "web",
        })
        try:
            await embedding_engine.generate_and_store(
                f"letter:{letter_obj['id']}", f"{title}\n{content}"
            )
        except Exception:
            pass
        return JSONResponse({"ok": True, "id": letter_obj["id"],
                             "letter_date": letter_obj["letter_date"]})
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/hold", methods=["POST"])
async def api_hold(request):
    """Ruby's own hand: store one ordinary memory from the web memory room.

    Same pipeline as the MCP hold tool's ordinary path (auto-tag via
    dehydrator, then merge-or-create). Feel/pinned modes stay MCP-only.
    """
    from starlette.responses import JSONResponse
    err = _require_read_access(request)
    if err: return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    content = str(body.get("content", "")).strip()
    if not content:
        return JSONResponse({"error": "empty"}, status_code=400)
    if len(content) > 4000:
        return JSONResponse({"error": "too_long"}, status_code=400)
    if "來源：" not in content and "来源：" not in content:
        content += "\n\n來源：Ruby（親手放入）"
    try:
        importance = max(1, min(10, int(body.get("importance", 5) or 5)))
    except (TypeError, ValueError):
        importance = 5
    await decay_engine.ensure_started()
    try:
        analysis = await dehydrator.analyze(content)
    except Exception:
        analysis = {"domain": ["未分類"], "valence": 0.5, "arousal": 0.3, "tags": [], "suggested_name": ""}
    try:
        result_name, is_merged = await _merge_or_create(
            content=content,
            tags=analysis.get("tags", []),
            importance=importance,
            domain=analysis.get("domain", ["未分類"]),
            valence=analysis.get("valence", 0.5),
            arousal=analysis.get("arousal", 0.3),
            name=analysis.get("suggested_name", ""),
        )
        return JSONResponse({"ok": True, "bucket": result_name, "merged": is_merged,
                             "domain": analysis.get("domain", [])})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/bucket/{bucket_id}/trace", methods=["POST"])
async def api_bucket_trace(request):
    """Write ops for the Cyan web memory room: resolve / pin / status / delete.

    Type-aware since 2026-07-12 (batch-4a): the web door speaks each bucket
    type's own verbs instead of one generic set —
    - plan buckets: "resolved" maps onto the plan lifecycle (`status`), because
      the fixation feed reads `status` — a web 沉底 that only wrote `resolved`
      left the plan silently feeding desire forever. Explicit `status`
      (active/resolved/abandoned) is also accepted, plan-only, and `progress`
      (0..1) is plan-only too.
    - pin is dynamic-only: feel/plan/mirage never enter the decay cycle, so
      "pin to stop forgetting" is meaningless there, and pinning a mirage
      would dress a dream up as doctrine.
    - resolve is refused on feel/mirage (they never surface in ordinary
      breath, 沉底 has nothing to sink).
    - 4c hardening (2026-07-12, joint spec review): feel and mirage are fully
      READ-ONLY through this door — Cyan's inner objects get deleted by Cyan's
      own hand (MCP trace on Ruby's word), not by a button. Plans retire via
      `abandoned`, never deletion (the promise history stays). Portraits
      (domain 畫像) and permanent doctrine refuse every mutation here. The web
      delete verb therefore applies to ordinary dynamic memories only.
    Pinning still locks importance to 10, delete still removes the embedding.
    """
    from starlette.responses import JSONResponse
    err = _require_read_access(request)
    if err: return err
    bucket_id = request.path_params["bucket_id"]
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    try:
        bucket = await bucket_mgr.get(bucket_id)
        if not bucket:
            return JSONResponse({"error": "not found"}, status_code=404)
        meta = bucket.get("metadata", {})
        btype = meta.get("type", "dynamic")
        is_portrait = "畫像" in (meta.get("domain") or [])
        if is_portrait or btype == "permanent":
            return JSONResponse(
                {"error": "畫像與固化準則在這扇門裡是只讀的"}, status_code=400)
        if body.get("delete") is True:
            if btype != "dynamic":
                return JSONResponse(
                    {"error": "只有普通記憶能從這裡遺忘；帳票用「放下」退場，"
                              "感受與蜃景是 Cyan 的內在物——要刪，開口讓他親手刪"},
                    status_code=400)
            result = await bucket_mgr.delete(bucket_id, actor=Actor("ruby", "dashboard:trace"))
            if result.ok:
                embedding_engine.delete_embedding(bucket_id)
                return JSONResponse({"ok": True, "deleted": True, "restore_seq": result.seq})
            return JSONResponse({"error": "not found"}, status_code=404)
        if btype in ("feel", "mirage"):
            return JSONResponse(
                {"error": "感受與蜃景是只讀的——看得到，但不是按鈕能動的"}, status_code=400)
        updates = {}
        if body.get("progress") is not None:
            if btype != "plan":
                return JSONResponse({"error": "progress 是帳票（plan）專屬"}, status_code=400)
            try:
                updates["progress"] = max(0.0, min(1.0, float(body["progress"])))
            except (TypeError, ValueError):
                return JSONResponse({"error": "progress 要是 0 到 1 的數字"}, status_code=400)
        if body.get("status") is not None:
            if btype != "plan":
                return JSONResponse(
                    {"error": "status 是帳票（plan）專屬的生命週期欄位"}, status_code=400)
            if body["status"] not in ("active", "resolved", "abandoned"):
                return JSONResponse({"error": "status 只能是 active/resolved/abandoned"},
                                    status_code=400)
            updates["status"] = body["status"]
            # Keep the decay/sort track in step with the lifecycle track.
            updates["resolved"] = body["status"] != "active"
        if body.get("resolved") in (0, 1, True, False):
            if btype == "plan":
                # Legacy web shape on a plan: translate 沉底 into the lifecycle
                # field the fixation feed actually reads (and keep both tracks
                # consistent). Explicit `status` above wins if both are sent.
                updates.setdefault("status", "resolved" if body["resolved"] else "active")
                updates["resolved"] = updates["status"] != "active"
            elif btype in ("feel", "mirage"):
                return JSONResponse(
                    {"error": "感受與蜃景不走浮現循環，沒有「沉底」可言"}, status_code=400)
            else:
                updates["resolved"] = bool(body["resolved"])
        if body.get("pinned") in (0, 1, True, False):
            if btype not in ("dynamic", "permanent"):
                return JSONResponse(
                    {"error": "釘選只屬於會衰減的記憶；感受／帳票／蜃景本來就不會被忘掉"},
                    status_code=400)
            updates["pinned"] = bool(body["pinned"])
            # importance 交給 bucket_manager.update() 鎖，理由同 MCP trace。
        if not updates:
            return JSONResponse({"error": "no supported fields"}, status_code=400)
        success = await bucket_mgr.update(bucket_id, actor=Actor("ruby", "dashboard:trace"), **updates)
        return JSONResponse({"ok": bool(success), "updates": updates})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/self-concept", methods=["GET"])
async def api_self_concept(request):
    """Read-only self-concept aspects for the Cyan web memory room."""
    from starlette.responses import JSONResponse
    err = _require_read_access(request)
    if err: return err
    try:
        qp = request.query_params
        limit = int(qp.get("limit", "0") or 0)
        return JSONResponse({"aspects": self_concept_store.read(aspect=qp.get("aspect", ""), limit=limit)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/dashboard", methods=["GET"])
async def dashboard(request):
    """Serve the dashboard HTML page."""
    from starlette.responses import HTMLResponse
    import os
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    try:
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>dashboard.html not found</h1>", status_code=404)


@mcp.custom_route("/api/config", methods=["GET"])
async def api_config_get(request):
    """Get current runtime config (safe fields only, API key masked)."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    dehy = config.get("dehydration", {})
    emb = config.get("embedding", {})
    api_key = dehy.get("api_key", "")
    emb_key = emb.get("api_key", "")
    masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else ("***" if api_key else "")
    masked_emb_key = f"{emb_key[:4]}...{emb_key[-4:]}" if len(emb_key) > 8 else ("***" if emb_key else "")
    return JSONResponse({
        "dehydration": {
            "model": dehy.get("model", ""),
            "base_url": dehy.get("base_url", ""),
            "api_key_masked": masked_key,
            "max_tokens": dehy.get("max_tokens", 1024),
            "temperature": dehy.get("temperature", 0.1),
        },
        "embedding": {
            "enabled": emb.get("enabled", False),
            "model": emb.get("model", ""),
            "base_url": emb.get("base_url", ""),
            "api_key_masked": masked_emb_key,
        },
        "api_usage_guard": config.get("api_usage_guard", {}),
        "merge_threshold": config.get("merge_threshold", 75),
        "transport": config.get("transport", "stdio"),
        "buckets_dir": config.get("buckets_dir", ""),
    })


@mcp.custom_route("/api/config", methods=["POST"])
async def api_config_update(request):
    """Hot-update runtime config. Optionally persist to config.yaml."""
    from starlette.responses import JSONResponse
    import yaml
    err = _require_auth(request)
    if err: return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    updated = []

    # --- Dehydration config ---
    if "dehydration" in body:
        d = body["dehydration"]
        dehy = config.setdefault("dehydration", {})
        for key in ("model", "base_url", "max_tokens", "temperature"):
            if key in d:
                dehy[key] = d[key]
                updated.append(f"dehydration.{key}")
        if "api_key" in d and d["api_key"]:
            dehy["api_key"] = d["api_key"]
            updated.append("dehydration.api_key")
        # Hot-reload dehydrator
        dehydrator.model = dehy.get("model", "deepseek-chat")
        dehydrator.base_url = dehy.get("base_url", "")
        dehydrator.api_key = dehy.get("api_key", "")
        if hasattr(dehydrator, "client") and dehydrator.api_key:
            from openai import AsyncOpenAI
            dehydrator.client = AsyncOpenAI(
                api_key=dehydrator.api_key,
                base_url=dehydrator.base_url,
            )

    # --- Embedding config ---
    if "embedding" in body:
        e = body["embedding"]
        emb = config.setdefault("embedding", {})
        if "enabled" in e:
            emb["enabled"] = bool(e["enabled"])
            embedding_engine.enabled = emb["enabled"]
            updated.append("embedding.enabled")
        if "model" in e:
            emb["model"] = e["model"]
            embedding_engine.model = emb["model"]
            updated.append("embedding.model")
        if "base_url" in e:
            emb["base_url"] = e["base_url"]
            embedding_engine.base_url = emb["base_url"]
            updated.append("embedding.base_url")
        if "api_key" in e and e["api_key"]:
            emb["api_key"] = e["api_key"]
            embedding_engine.api_key = emb["api_key"]
            updated.append("embedding.api_key")
        if any(key in e for key in ("api_key", "base_url")) and embedding_engine.api_key:
            from openai import AsyncOpenAI
            embedding_engine.client = AsyncOpenAI(
                api_key=embedding_engine.api_key,
                base_url=embedding_engine.base_url,
                timeout=30.0,
            )
            embedding_engine.enabled = bool(embedding_engine.api_key) and emb.get("enabled", True)

    # --- API usage guard config ---
    if "api_usage_guard" in body:
        g = body["api_usage_guard"]
        guard = config.setdefault("api_usage_guard", {})
        if "deepseek_low_balance_usd" in g:
            guard["deepseek_low_balance_usd"] = float(g["deepseek_low_balance_usd"])
            api_usage_guard.deepseek_low_balance_usd = guard["deepseek_low_balance_usd"]
            updated.append("api_usage_guard.deepseek_low_balance_usd")
        if "cache_ttl_seconds" in g:
            guard["cache_ttl_seconds"] = int(g["cache_ttl_seconds"])
            api_usage_guard.cache_ttl_seconds = guard["cache_ttl_seconds"]
            updated.append("api_usage_guard.cache_ttl_seconds")

    # --- Merge threshold ---
    if "merge_threshold" in body:
        config["merge_threshold"] = int(body["merge_threshold"])
        updated.append("merge_threshold")

    # --- Persist to config.yaml if requested ---
    if body.get("persist", False):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        try:
            save_config = {}
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    save_config = yaml.safe_load(f) or {}

            if "dehydration" in body:
                sc_dehy = save_config.setdefault("dehydration", {})
                for key in ("model", "base_url", "max_tokens", "temperature"):
                    if key in body["dehydration"]:
                        sc_dehy[key] = body["dehydration"][key]
                # Never persist api_key to yaml (use env var)

            if "embedding" in body:
                sc_emb = save_config.setdefault("embedding", {})
                for key in ("enabled", "model", "base_url"):
                    if key in body["embedding"]:
                        sc_emb[key] = body["embedding"][key]
                # Never persist api_key to yaml (use env var)

            if "api_usage_guard" in body:
                sc_guard = save_config.setdefault("api_usage_guard", {})
                for key in ("deepseek_low_balance_usd", "cache_ttl_seconds"):
                    if key in body["api_usage_guard"]:
                        sc_guard[key] = body["api_usage_guard"][key]

            if "merge_threshold" in body:
                save_config["merge_threshold"] = int(body["merge_threshold"])

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(save_config, f, default_flow_style=False, allow_unicode=True)
            updated.append("persisted_to_yaml")
        except Exception as e:
            return JSONResponse({"error": f"persist failed: {e}", "updated": updated}, status_code=500)

    return JSONResponse({"updated": updated, "ok": True})


# =============================================================
# /api/host-vault — read/write the host-side OMBRE_HOST_VAULT_DIR
# 用於在 Dashboard 設置 docker-compose 掛載的宿主機記憶桶目錄。
# 寫入項目根目錄的 .env 文件，需 docker compose down/up 才能生效。
# =============================================================

def _project_env_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def _read_env_var(name: str) -> str:
    """Return current value of `name` from process env first, then .env file (best-effort)."""
    val = os.environ.get(name, "").strip()
    if val:
        return val
    env_path = _project_env_path()
    if not os.path.exists(env_path):
        return ""
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                if k.strip() == name:
                    return v.strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


def _write_env_var(name: str, value: str) -> None:
    """
    Idempotent upsert of `NAME=value` in project .env. Creates the file if missing.
    Preserves other entries verbatim. Quotes values containing spaces.
    """
    env_path = _project_env_path()
    quoted = f'"{value}"' if value and (" " in value or "#" in value) else value
    new_line = f"{name}={quoted}\n"

    lines: list[str] = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    replaced = False
    for i, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        k, _, _v = stripped.partition("=")
        if k.strip() == name:
            lines[i] = new_line
            replaced = True
            break
    if not replaced:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(new_line)

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


@mcp.custom_route("/api/host-vault", methods=["GET"])
async def api_host_vault_get(request):
    """Read the current OMBRE_HOST_VAULT_DIR (process env > project .env)."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    value = _read_env_var("OMBRE_HOST_VAULT_DIR")
    return JSONResponse({
        "value": value,
        "source": "env" if os.environ.get("OMBRE_HOST_VAULT_DIR", "").strip() else ("file" if value else ""),
        "env_file": _project_env_path(),
    })


@mcp.custom_route("/api/host-vault", methods=["POST"])
async def api_host_vault_set(request):
    """
    Persist OMBRE_HOST_VAULT_DIR to the project .env file.
    Body: {"value": "/path/to/vault"}  (empty string clears the entry)
    Note: container restart is required for docker-compose to pick up the new mount.
    """
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    raw = body.get("value", "")
    if not isinstance(raw, str):
        return JSONResponse({"error": "value must be a string"}, status_code=400)
    value = raw.strip()

    # Reject characters that would break .env / shell parsing
    if "\n" in value or "\r" in value or '"' in value or "'" in value:
        return JSONResponse({"error": "value must not contain quotes or newlines"}, status_code=400)

    try:
        _write_env_var("OMBRE_HOST_VAULT_DIR", value)
    except Exception as e:
        return JSONResponse({"error": f"failed to write .env: {e}"}, status_code=500)

    return JSONResponse({
        "ok": True,
        "value": value,
        "env_file": _project_env_path(),
        "note": "已寫入 .env；需在宿主機執行 `docker compose down && docker compose up -d` 讓新掛載生效。",
    })


# =============================================================
# Import API — conversation history import
# 導入 API — 對話歷史導入
# =============================================================

@mcp.custom_route("/api/import/upload", methods=["POST"])
async def api_import_upload(request):
    """Upload a conversation file and start import."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err

    if import_engine.is_running:
        return JSONResponse({"error": "Import already running"}, status_code=409)

    content_type = request.headers.get("content-type", "")
    filename = ""

    try:
        if "multipart/form-data" in content_type:
            form = await request.form()
            file_field = form.get("file")
            if not file_field:
                return JSONResponse({"error": "No file field"}, status_code=400)
            raw_bytes = await file_field.read()
            filename = getattr(file_field, "filename", "upload")
            raw_content = raw_bytes.decode("utf-8", errors="replace")
        else:
            body = await request.body()
            raw_content = body.decode("utf-8", errors="replace")
            # Try to get filename from query params
            filename = request.query_params.get("filename", "upload")

        if not raw_content.strip():
            return JSONResponse({"error": "Empty file"}, status_code=400)

        preserve_raw = request.query_params.get("preserve_raw", "").lower() in ("1", "true")
        resume = request.query_params.get("resume", "").lower() in ("1", "true")

    except Exception as e:
        return JSONResponse({"error": f"Failed to read upload: {e}"}, status_code=400)

    # Start import in background
    async def _run_import():
        try:
            await import_engine.start(raw_content, filename, preserve_raw, resume)
        except Exception as e:
            logger.error(f"Import failed: {e}")

    asyncio.create_task(_run_import())

    return JSONResponse({
        "status": "started",
        "filename": filename,
        "size_bytes": len(raw_content.encode()),
    })


@mcp.custom_route("/api/import/status", methods=["GET"])
async def api_import_status(request):
    """Get current import progress."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    return JSONResponse(import_engine.get_status())


@mcp.custom_route("/api/import/pause", methods=["POST"])
async def api_import_pause(request):
    """Pause the running import."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    if not import_engine.is_running:
        return JSONResponse({"error": "No import running"}, status_code=400)
    import_engine.pause()
    return JSONResponse({"status": "pause_requested"})


@mcp.custom_route("/api/import/patterns", methods=["GET"])
async def api_import_patterns(request):
    """Detect high-frequency patterns after import."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    try:
        patterns = await import_engine.detect_patterns()
        return JSONResponse({"patterns": patterns})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/import/results", methods=["GET"])
async def api_import_results(request):
    """List recently imported/created buckets for review."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    try:
        limit = int(request.query_params.get("limit", "50"))
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        # Sort by created time, newest first
        all_buckets.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
        results = []
        for b in all_buckets[:limit]:
            results.append({
                "id": b["id"],
                "name": b["metadata"].get("name", ""),
                "content": b["content"][:300],
                "type": b["metadata"].get("type", ""),
                "domain": b["metadata"].get("domain", []),
                "tags": b["metadata"].get("tags", []),
                "importance": b["metadata"].get("importance", 5),
                "created": b["metadata"].get("created", ""),
            })
        return JSONResponse({"buckets": results, "total": len(all_buckets)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/import/review", methods=["POST"])
async def api_import_review(request):
    """Apply review decisions: mark buckets as important/noise/pinned."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    decisions = body.get("decisions", [])
    if not decisions:
        return JSONResponse({"error": "No decisions provided"}, status_code=400)

    applied = 0
    errors = 0
    for d in decisions:
        bid = d.get("bucket_id", "")
        action = d.get("action", "")
        if not bid or not action:
            continue
        try:
            if action == "important":
                await bucket_mgr.update(bid, actor=Actor("ruby", "import_review"), importance=9)
            elif action == "pin":
                await bucket_mgr.update(bid, actor=Actor("ruby", "import_review"), pinned=True)
            elif action == "noise":
                await bucket_mgr.update(bid, actor=Actor("ruby", "import_review"), resolved=True, importance=1)
            elif action == "delete":
                # Route through the manager so the embedding is cleaned up too
                # (raw os.remove left orphan embeddings in the vector store)
                # 走 manager 刪除，同步清 embedding（裸 os.remove 會留孤兒向量）
                if (await bucket_mgr.delete(bid, actor=Actor("ruby", "import_review"))).ok:
                    embedding_engine.delete_embedding(bid)
            applied += 1
        except Exception as e:
            logger.warning(f"Review action failed for {bid}: {e}")
            errors += 1

    return JSONResponse({"applied": applied, "errors": errors})


# =============================================================
# /api/status — system status for Dashboard settings tab
# /api/status — Dashboard 設置頁用系統狀態
# =============================================================
@mcp.custom_route("/api/status", methods=["GET"])
async def api_system_status(request):
    """Return detailed system status for the settings panel."""
    from starlette.responses import JSONResponse
    err = _require_read_access(request)
    if err: return err
    try:
        stats = await bucket_mgr.get_stats()
        usage = await api_usage_guard.check_all(probe_gemini=False)

        # --- Metabolism gauges (batch-7 檔1): verification lives on a dial,
        # not on someone's attention (the WORKING_ON lesson). Same helpers as
        # the surfacing quota, so the dial can never drift from the engine.
        # --- 代謝儀表（檔1）：驗證靠儀表不靠人的注意力（WORKING_ON 教訓）。
        # 與浮現配額共用同一組判定，儀表永遠不會跟引擎各說各話。---
        metabolism: dict = {}
        try:
            eng_domains = _eng_domain_set()
            rel_domains = frozenset(normalize_domains(["戀愛", "人際", "內心"]))
            all_dynamic = [
                b for b in await bucket_mgr.list_all(include_archive=False)
                if b["metadata"].get("type") not in ("permanent", "feel", "plan", "mirage")
                and not (b["metadata"].get("pinned") or b["metadata"].get("protected"))
            ]
            pure_eng = [b for b in all_dynamic if _is_pure_eng(b["metadata"], eng_domains)]
            danger_total = 0
            danger_eng = 0
            heats_eng: list[float] = []
            heats_rel: list[float] = []
            for b in all_dynamic:
                meta = b["metadata"]
                heat = decay_engine.calculate_heat(meta)
                if _is_pure_eng(meta, eng_domains):
                    heats_eng.append(heat)
                if set(meta.get("domain", []) or []) & rel_domains:
                    heats_rel.append(heat)
                if (
                    not meta.get("resolved", False)
                    and not meta.get("digested", False)
                    and decay_engine.review_priority(heat) > 0
                ):
                    danger_total += 1
                    if _is_pure_eng(meta, eng_domains):
                        danger_eng += 1
            metabolism = {
                "pure_eng_count": len(pure_eng),
                "dynamic_count": len(all_dynamic),
                "pure_eng_pct": round(100 * len(pure_eng) / len(all_dynamic), 1) if all_dynamic else 0.0,
                "danger_zone_total": danger_total,
                "danger_zone_eng": danger_eng,
                "avg_heat_pure_eng": round(sum(heats_eng) / len(heats_eng), 3) if heats_eng else None,
                "avg_heat_relation": round(sum(heats_rel) / len(heats_rel), 3) if heats_rel else None,
            }
        except Exception as e:
            logger.warning(f"metabolism gauges failed / 代謝儀表計算失敗: {e}")

        return JSONResponse({
            "decay_engine": "running" if decay_engine.is_running else "stopped",
            "decay_heartbeat": decay_engine.heartbeat(),
            "metabolism": metabolism,
            "embedding_enabled": embedding_engine.enabled,
            "api_usage_ok": usage.get("ok", False),
            "api_usage_warnings": usage.get("warnings", []),
            "buckets": {
                "permanent": stats.get("permanent_count", 0),
                "dynamic": stats.get("dynamic_count", 0),
                "archive": stats.get("archive_count", 0),
                # batch-4a: the full pulse. get_stats always counted these;
                # the web door just never passed them on, so the memory
                # room's 脈搏 showed a three-chamber heart on an eight-organ
                # body. `total` keeps its original meaning (active memory).
                "feel": stats.get("feel_count", 0),
                "plan": stats.get("plan_count", 0),
                "mirage": stats.get("mirage_count", 0),
                "letters": len(letter_store.list_letters()),
                "total": stats.get("permanent_count", 0) + stats.get("dynamic_count", 0),
            },
            "using_env_password": bool(os.environ.get("OMBRE_DASHBOARD_PASSWORD", "")),
            "version": "1.4.0",
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/api-usage", methods=["GET"])
async def api_usage_status(request):
    """Return DeepSeek balance and Gemini embedding availability for Dashboard/monitors."""
    from starlette.responses import JSONResponse
    # Read-only usage data: same loopback trust as the memory-room GET routes,
    # so web.ts can put embedding/dehydrator health on the 機房 dashboard.
    err = _require_read_access(request)
    if err: return err
    force = request.query_params.get("force", "").lower() in ("1", "true", "yes", "on")
    probe_gemini = request.query_params.get("probe_gemini", "true").lower() not in ("0", "false", "no", "off")
    try:
        usage = await api_usage_guard.check_all(force=force, probe_gemini=probe_gemini)
        if getattr(embedding_engine, "last_error", ""):
            usage.setdefault("warnings", []).append(
                f"Gemini embedding 最近一次生成失敗：{embedding_engine.last_error}"
            )
        return JSONResponse(usage)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# --- Entry point / 啟動入口 ---
if __name__ == "__main__":
    transport = config.get("transport", "stdio")
    logger.info(f"Ombre Brain starting | transport: {transport}")

    if transport in ("sse", "streamable-http"):
        import threading
        import uvicorn
        from starlette.middleware.cors import CORSMiddleware

        # --- Application-level keepalive: ping /health every 60s ---
        # --- 應用層保活：每 60 秒 ping 一次 /health，防止 Cloudflare Tunnel 空閒斷連 ---
        async def _keepalive_loop():
            await asyncio.sleep(10)  # Wait for server to fully start
            async with httpx.AsyncClient() as client:
                while True:
                    try:
                        await client.get(f"http://localhost:{OMBRE_PORT}/health", timeout=5)
                        logger.debug("Keepalive ping OK / 保活 ping 成功")
                    except Exception as e:
                        logger.warning(f"Keepalive ping failed / 保活 ping 失敗: {e}")
                    await asyncio.sleep(60)

        def _start_keepalive():
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_keepalive_loop())

        t = threading.Thread(target=_start_keepalive, daemon=True)
        t.start()

        # --- Add CORS middleware so remote clients (Cloudflare Tunnel / ngrok) can connect ---
        # --- 添加 CORS 中間件，讓遠程客戶端（Cloudflare Tunnel / ngrok）能正常連接 ---
        if transport == "streamable-http":
            _app = mcp.streamable_http_app()
        else:
            _app = mcp.sse_app()
        _app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        )
        logger.info("CORS middleware enabled for remote transport / 已啟用 CORS 中間件")
        # --- SINGLE-PROCESS INVARIANT, load-bearing (2026-07-19 audit F8) ---
        # Every concurrency guarantee in this codebase assumes exactly one
        # process and one event loop: there are NO file locks; update()'s
        # critical section is "no await between read and write"; DesireStore's
        # RLock is in-process only; the decay daemon is a task on THIS loop.
        # workers=1 is therefore not a tuning choice — it is the lock.
        # If you ever split decay/desire into another process, add real
        # cross-process locking (flock) first.
        # --- 單進程不變式（承重）：全庫併發安全建立在單進程單 loop 上——
        # 沒有檔案鎖、update() 的臨界區是「讀寫之間無 await」、desire 的
        # RLock 只在進程內有效、衰減 daemon 掛在本 loop。workers=1 不是
        # 調參，它就是那把鎖。要拆進程，先加跨進程鎖。---
        uvicorn.run(_app, host=OMBRE_HOST, port=OMBRE_PORT, workers=1)
    else:
        mcp.run(transport=transport)
