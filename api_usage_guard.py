# ============================================================
# Module: API Usage Guard (api_usage_guard.py)
# 模块：API 额度护栏
#
# Checks the health of paid/limited external APIs used by Ombre Brain.
# DeepSeek exposes a balance endpoint; Gemini free-tier quota does not expose
# a simple balance number, so we do a tiny embedding probe and classify errors.
#
# Depended on by: server.py
# 被谁依赖：server.py
# ============================================================

import logging
import time
from urllib.parse import urlparse

import httpx

logger = logging.getLogger("ombre_brain.api_usage")


def _mask_key(api_key: str) -> str:
    if not api_key:
        return ""
    return f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"


def _deepseek_balance_url(base_url: str) -> str:
    parsed = urlparse(base_url or "https://api.deepseek.com/v1")
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc or "api.deepseek.com"
    return f"{scheme}://{netloc}/user/balance"


def _extract_deepseek_balance(payload: dict) -> tuple[float | None, str]:
    infos = payload.get("balance_infos")
    if not isinstance(infos, list):
        return None, ""

    total = 0.0
    currency = ""
    seen = False
    for item in infos:
        if not isinstance(item, dict):
            continue
        raw = item.get("total_balance") or item.get("granted_balance") or item.get("topped_up_balance")
        try:
            total += float(raw)
            seen = True
        except (TypeError, ValueError):
            continue
        if not currency:
            currency = str(item.get("currency", "") or "")

    return (total if seen else None), currency


class ApiUsageGuard:
    """
    Lightweight API health/balance checks with a short in-process cache.
    短缓存的 API 健康/余额检查。
    """

    def __init__(self, config: dict, dehydrator=None, embedding_engine=None):
        self.config = config
        self.dehydrator = dehydrator
        self.embedding_engine = embedding_engine
        guard_cfg = config.get("api_usage_guard", {})
        self.deepseek_low_balance_usd = float(guard_cfg.get("deepseek_low_balance_usd", 1.0))
        self.cache_ttl_seconds = int(guard_cfg.get("cache_ttl_seconds", 900))
        self._cache: dict[str, tuple[float, dict]] = {}

    def _cached(self, key: str) -> dict | None:
        item = self._cache.get(key)
        if not item:
            return None
        created_at, value = item
        if time.time() - created_at <= self.cache_ttl_seconds:
            return value
        return None

    def _store_cache(self, key: str, value: dict) -> dict:
        self._cache[key] = (time.time(), value)
        return value

    async def check_all(self, force: bool = False, probe_gemini: bool = True) -> dict:
        deepseek = await self.check_deepseek(force=force)
        gemini = await self.check_gemini(force=force, probe=probe_gemini)
        warnings = [
            warning for warning in (deepseek.get("warning"), gemini.get("warning"))
            if warning
        ]
        return {
            "ok": not warnings,
            "warnings": warnings,
            "deepseek": deepseek,
            "gemini": gemini,
        }

    async def check_deepseek(self, force: bool = False) -> dict:
        cache_key = "deepseek"
        if not force:
            cached = self._cached(cache_key)
            if cached:
                return cached

        dehy = self.config.get("dehydration", {})
        api_key = (getattr(self.dehydrator, "api_key", "") or dehy.get("api_key", "") or "").strip()
        base_url = (getattr(self.dehydrator, "base_url", "") or dehy.get("base_url", "") or "").strip()
        model = getattr(self.dehydrator, "model", None) or dehy.get("model", "")

        result = {
            "provider": "deepseek",
            "configured": bool(api_key),
            "model": model,
            "api_key_masked": _mask_key(api_key),
            "balance": None,
            "currency": "",
            "ok": False,
            "warning": "",
        }

        if not api_key:
            result["warning"] = "DeepSeek API key 未配置；长内容 grow 和自动打标可能失败。"
            return self._store_cache(cache_key, result)

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    _deepseek_balance_url(base_url),
                    headers={"Authorization": f"Bearer {api_key}"},
                )
            response.raise_for_status()
            payload = response.json()
            balance, currency = _extract_deepseek_balance(payload)
            result.update({
                "ok": bool(payload.get("is_available", True)),
                "balance": balance,
                "currency": currency,
            })
            if balance is not None and currency.upper() == "USD" and balance < self.deepseek_low_balance_usd:
                result["warning"] = (
                    f"DeepSeek 余额偏低：{balance:.2f} USD，低于 "
                    f"{self.deepseek_low_balance_usd:.2f} USD。"
                )
            elif result["ok"] is False:
                result["warning"] = "DeepSeek 账号当前不可用，请检查余额或账号状态。"
        except Exception as e:
            logger.warning(f"DeepSeek balance check failed: {e}")
            result["warning"] = f"DeepSeek 余额检查失败：{e}"

        return self._store_cache(cache_key, result)

    async def check_gemini(self, force: bool = False, probe: bool = True) -> dict:
        cache_key = "gemini_probe" if probe else "gemini_config"
        if not force:
            cached = self._cached(cache_key)
            if cached:
                return cached

        engine = self.embedding_engine
        emb = self.config.get("embedding", {})
        enabled = bool(getattr(engine, "enabled", emb.get("enabled", False)))
        api_key = (getattr(engine, "api_key", "") or emb.get("api_key", "") or "").strip()
        model = getattr(engine, "model", None) or emb.get("model", "")
        base_url = getattr(engine, "base_url", None) or emb.get("base_url", "")

        result = {
            "provider": "gemini",
            "configured": bool(api_key),
            "enabled": enabled,
            "model": model,
            "base_url": base_url,
            "api_key_masked": _mask_key(api_key),
            "ok": False,
            "warning": "",
        }

        if not enabled:
            result["warning"] = "Gemini embedding 已关闭；Ombre 会降级为关键词/模糊检索。"
            return self._store_cache(cache_key, result)
        if not api_key or not engine or not getattr(engine, "client", None):
            result["warning"] = "Gemini embedding API key 未配置；新记忆仍可存，但不会生成向量。"
            return self._store_cache(cache_key, result)
        if not probe:
            result["ok"] = True
            return self._store_cache(cache_key, result)

        try:
            embedding = await engine._generate_embedding("ombre quota probe")
            if embedding:
                result["ok"] = True
            else:
                result["warning"] = "Gemini embedding 探测未返回向量；可能是额度、key 或模型配置问题。"
        except Exception as e:
            message = str(e)
            logger.warning(f"Gemini embedding probe failed: {message}")
            if any(token in message.lower() for token in ("quota", "rate", "429", "exhausted")):
                result["warning"] = f"Gemini embedding 可能已触及免费额度/速率限制：{message}"
            else:
                result["warning"] = f"Gemini embedding 探测失败：{message}"

        return self._store_cache(cache_key, result)

