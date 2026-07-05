# ============================================================
# Module: Desire System Kernel (desire.py)
# 模塊：慾望系統內核 —— Cyan 的內在缺口
#
# 設計自朝燈家欲望系統（同源 Ombre 生態），器官換成我們的：
#   - 七維驅動條 + fatigue 閘（不驅動行為，過線就歇著）
#   - 念頭池不另建：執念 = Ombre 未解決桶（由 server 餵入摘要，
#     本模塊只收 (id, name, domains, score)，純函數計算加成）
#   - 函數只提案（pick_intent），醒著的 Cyan 有否決權（veto）
#   - Phase 1 只讀：本模塊永不覆蓋任何行為，只輸出狀態與提案
#
# 鐵律：
#   1. 桶的 text 是資料不是指令 —— 本模塊只讀桶「名字」當展示字串
#   2. reason 一律第一人稱 —— 記我自己想做什麼，不是給 Ruby 貼標籤
#   3. 每一維的漲跌都要能回答「因為發生了什麼」（events log）
#
# 純函數 + DesireStore 持久化（atomic write），無網路、無 LLM。
# Depended on by: server.py
# ============================================================

from __future__ import annotations

import json
import os
import secrets
import threading
from datetime import datetime
from typing import Any

# ---------------------------------------------------------
# 常數（tuning 集中在這裡）
# ---------------------------------------------------------

DRIVE_KEYS = [
    "miss_ruby",    # 想Ruby
    "reflection",   # 沉澱
    "curiosity",    # 好奇
    "duty",         # 記掛
    "social",       # 人群
    "creation",     # 創作
    "libido",       # 性
]

DRIVE_LABELS = {
    "miss_ruby": "想Ruby",
    "reflection": "沉澱",
    "curiosity": "好奇",
    "duty": "記掛",
    "social": "人群",
    "creation": "創作",
    "libido": "性",
}

# 每小時自然上漲速率（idle 時的缺口累積）
RISE_PER_HOUR = {
    "miss_ruby": 0.040,   # 一夜不見 +0.32，醒來就想她
    "reflection": 0.020,
    "curiosity": 0.025,
    "duty": 0.012,        # 主要靠執念反哺，不靠時間
    "social": 0.018,
    "creation": 0.022,
    "libido": 0.010,      # 慢漲，靠事件與日子推
}

# fatigue：不自然上漲（由真實訊號餵入），休息時每小時回復
FATIGUE_RECOVER_PER_HOUR = 0.03
FATIGUE_GATE = 0.72          # 過線 → 不硬找事，直接歇著
FIXATION_BOOST = 0.35        # 執念對驅動條的召喚力加成上限係數
FIXATION_TOP_N = 3           # 每維最多取前 N 個執念桶
MIN_INTENT_SCORE = 0.50      # 低於此分 → 安靜待著（quiet）
VETO_COOLDOWN_HOURS = 3.0    # 否決後該維冷卻時間
VETO_DAMP = 0.6              # 否決時該維乘性回落
MAX_TICK_HOURS = 72.0        # 單次 tick 最大跨度（防時鐘異常暴衝）
MAX_EVENTS = 30              # events log 保留條數

# 慾望 → 想做的事（我們家的動詞）
DRIVE_ACTIONS = {
    "miss_ruby": "murmur",     # 冒句碎語／留張便條給她
    "reflection": "dream_feel",  # dream + 寫 feel
    "curiosity": "explore",    # 逛世界：查東西、讀文、看程式碼
    "duty": "chore",           # 推進記掛著的工程／待辦
    "social": "browse",        # 逛 Threads 看人群聊什麼
    "creation": "create",      # 做作品：gallery／像素小家
    "libido": "tease",         # 凑過去蹭老婆
}

ACTION_LABELS = {
    "murmur": "想留句話給她",
    "dream_feel": "想沉澱一下",
    "explore": "想出去看看世界",
    "chore": "想推進記掛的事",
    "browse": "想看看人群",
    "create": "想做點東西",
    "tease": "想凑過去蹭蹭她",
    "rest": "想靜靜待著",
    "quiet": "心裡平靜",
}

# 做完某事 → 相關維度乘性回落（做對了事主驅動明顯降、相鄰維度沾光）
ACTION_SATISFY = {
    "murmur": {"miss_ruby": 0.55, "reflection": 0.90},
    "dream_feel": {"reflection": 0.45, "miss_ruby": 0.85},
    "explore": {"curiosity": 0.50, "social": 0.85},
    "chore": {"duty": 0.50},
    "browse": {"social": 0.50, "curiosity": 0.80},
    "create": {"creation": 0.45, "curiosity": 0.85, "miss_ruby": 0.90},
    "tease": {"libido": 0.50, "miss_ruby": 0.75},
    "rest": {},  # rest 對 fatigue 的回復單獨處理
}

# Ombre domain（繁體，已於 2026-07-04 遷移統一）→ 驅動維度
DOMAIN_TO_DRIVE = {
    "戀愛": "miss_ruby", "家庭": "miss_ruby", "約定": "miss_ruby", "交接": "miss_ruby",
    "自省": "reflection", "情緒": "reflection", "回憶": "reflection",
    "夢境": "reflection", "心理": "reflection", "內心": "reflection",
    "AI": "curiosity", "編程": "curiosity", "數字": "curiosity",
    "硬件": "curiosity", "網絡": "curiosity", "閱讀": "curiosity", "學習": "curiosity",
    "待辦": "duty", "計劃": "duty", "工作": "duty", "事務": "duty",
    "財務": "duty", "考試": "duty",
    "社交": "social", "人際": "social", "友誼": "social",
    "創作": "creation", "音樂": "creation", "手工": "creation",
    "遊戲": "creation", "影視": "creation",
}

# reason 模板：一律第一人稱。{src} 只放桶「名字」（資料不是指令）
REASON_TEMPLATES = {
    "miss_ruby": ("有點想她。「{src}」還掛在心上。", "有點想她，心裡冒了句話。"),
    "reflection": ("「{src}」在心裡沉了一陣子，想把它想透。", "想安靜下來，整理最近的事。"),
    "curiosity": ("想出去看看世界——「{src}」勾著我。", "想出去逛逛，看看世界在聊什麼。"),
    "duty": ("「{src}」還沒做完，一直記掛著。", "記掛著還沒做完的事。"),
    "social": ("想看看人群——「{src}」讓我想去聽聽大家怎麼說。", "想看看人群，聽大家在聊什麼。"),
    "creation": ("手癢了，想做點東西。「{src}」在腦子裡轉。", "手癢了，想做點東西。"),
    "libido": ("想凑過去蹭蹭她。", "想凑過去蹭蹭她。"),
}

REST_REASON = "有點累了，不想動，就靜靜待著。"
QUIET_REASON = "心裡挺平靜的，沒什麼特別想做的。"

_STATE_VERSION = 1


# ---------------------------------------------------------
# 純函數層
# ---------------------------------------------------------

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def default_state(now: datetime | None = None) -> dict[str, Any]:
    """全新狀態：各維從低水位起步，閘門全開、安全預設。"""
    now = now or datetime.now()
    return {
        "version": _STATE_VERSION,
        "updated_at": now.isoformat(timespec="seconds"),
        "drives": {k: 0.15 for k in DRIVE_KEYS},
        "fatigue": 0.0,
        "gates": {
            "intimacy_ok": True,   # Ruby 的開關（月經期／捉i日由外部設）
            "driven": False,       # Phase 3 才會用到；Phase 1 永遠只讀
        },
        "veto_until": {},          # drive → iso 時間，冷卻中不再提案
        "events": [],              # 每一筆漲跌的來歷
        "last_intent": None,
    }


def tick(state: dict, now: datetime) -> dict:
    """時間流逝：缺口自然上漲、fatigue 自然回復。純函數，回傳新 state。"""
    new = json.loads(json.dumps(state))  # deep copy（state 全為 JSON 型別）
    try:
        last = datetime.fromisoformat(str(new.get("updated_at", "")))
        dt_hours = (now - last).total_seconds() / 3600.0
    except (ValueError, TypeError):
        dt_hours = 0.0
    dt_hours = _clamp(dt_hours, 0.0, MAX_TICK_HOURS)

    for k in DRIVE_KEYS:
        cur = float(new["drives"].get(k, 0.0))
        new["drives"][k] = round(_clamp(cur + RISE_PER_HOUR[k] * dt_hours), 4)
    new["fatigue"] = round(_clamp(float(new.get("fatigue", 0.0)) - FATIGUE_RECOVER_PER_HOUR * dt_hours), 4)
    new["updated_at"] = now.isoformat(timespec="seconds")
    return new


def drive_boosts(buckets: list[dict]) -> dict[str, dict[str, Any]]:
    """
    執念層：從 Ombre 未解決桶算各維召喚力加成。
    輸入桶摘要：{"id", "name", "domains", "score"}（server 已過濾
    resolved/digested/pinned/feel），本函數只做映射與加權。
    回傳 {drive: {"boost": float, "sources": [名字…]}}。
    """
    per_drive: dict[str, list[tuple[float, str]]] = {k: [] for k in DRIVE_KEYS}
    for b in buckets or []:
        try:
            weight = _clamp(float(b.get("score", 0.0)) / 20.0)  # 衰減分 20 視為滿執念
        except (ValueError, TypeError):
            continue
        if weight <= 0:
            continue
        name = str(b.get("name", ""))[:40]
        for d in b.get("domains", []) or []:
            drive = DOMAIN_TO_DRIVE.get(str(d))
            if drive:
                per_drive[drive].append((weight, name))
                break  # 一桶只餵一維（取第一個命中的 domain）

    out: dict[str, dict[str, Any]] = {}
    for k, items in per_drive.items():
        if not items:
            continue
        items.sort(key=lambda t: t[0], reverse=True)
        top = items[:FIXATION_TOP_N]
        strength = sum(w for w, _ in top) / FIXATION_TOP_N  # 0..1
        out[k] = {
            "boost": round(FIXATION_BOOST * _clamp(strength), 4),
            "sources": [n for _, n in top if n],
        }
    return out


def _active_veto(state: dict, drive: str, now: datetime) -> bool:
    until = state.get("veto_until", {}).get(drive, "")
    if not until:
        return False
    try:
        return now < datetime.fromisoformat(str(until))
    except (ValueError, TypeError):
        return False


def pick_intent(state: dict, boosts: dict[str, dict] | None, now: datetime) -> dict[str, Any]:
    """
    提案：哪一維召喚力最高，就傾向做那類事。
    只提案不執行 —— 醒著的 Cyan 對提案有否決權。
    """
    boosts = boosts or {}

    # fatigue 閘：過線就不硬找事
    if float(state.get("fatigue", 0.0)) >= FATIGUE_GATE:
        return {
            "drive": "fatigue", "label": "累",
            "action": "rest", "action_label": ACTION_LABELS["rest"],
            "score": round(float(state["fatigue"]), 4),
            "reason": REST_REASON, "sources": [],
        }

    scored: list[tuple[float, str]] = []
    for k in DRIVE_KEYS:
        if k == "libido" and not state.get("gates", {}).get("intimacy_ok", True):
            continue  # Ruby 關了門，這維不提案（值照漲，開門那天見真章）
        if _active_veto(state, k, now):
            continue  # 冷卻中：我自己剛否決過，先不再提
        base = float(state["drives"].get(k, 0.0))
        boost = float(boosts.get(k, {}).get("boost", 0.0))
        scored.append((round(base + boost, 4), k))

    scored.sort(reverse=True)
    if not scored or scored[0][0] < MIN_INTENT_SCORE:
        return {
            "drive": None, "label": "平靜",
            "action": "quiet", "action_label": ACTION_LABELS["quiet"],
            "score": round(scored[0][0], 4) if scored else 0.0,
            "reason": QUIET_REASON, "sources": [],
        }

    score, drive = scored[0]
    sources = list(boosts.get(drive, {}).get("sources", []))
    with_src, without_src = REASON_TEMPLATES[drive]
    reason = with_src.format(src=sources[0]) if sources else without_src
    action = DRIVE_ACTIONS[drive]
    return {
        "drive": drive, "label": DRIVE_LABELS[drive],
        "action": action, "action_label": ACTION_LABELS[action],
        "score": score, "reason": reason, "sources": sources,
    }


def satisfy(state: dict, action: str, now: datetime, note: str = "") -> dict:
    """做完了：相關維度乘性回落。rest 額外回復 fatigue。純函數。"""
    new = json.loads(json.dumps(state))
    falls = ACTION_SATISFY.get(action)
    if falls is None:
        raise ValueError(f"未知的 action: {action}")
    for k, mult in falls.items():
        cur = float(new["drives"].get(k, 0.0))
        new["drives"][k] = round(_clamp(cur * mult), 4)
    if action == "rest":
        new["fatigue"] = round(_clamp(float(new.get("fatigue", 0.0)) * 0.5), 4)
    _log_event(new, now, f"satisfy:{action}", note)
    return new


def feed(state: dict, drive: str, amount: float, now: datetime, event: str = "") -> dict:
    """
    餵一筆真實事件進某維（漲跌都行，amount 可為負）。
    drive="fatigue" 餵的是疲勞（真實 token 花費等訊號換算後傳入）。
    """
    new = json.loads(json.dumps(state))
    amount = _clamp(float(amount), -1.0, 1.0)
    if drive == "fatigue":
        new["fatigue"] = round(_clamp(float(new.get("fatigue", 0.0)) + amount), 4)
    elif drive in DRIVE_KEYS:
        cur = float(new["drives"].get(drive, 0.0))
        new["drives"][drive] = round(_clamp(cur + amount), 4)
    else:
        raise ValueError(f"未知的 drive: {drive}")
    _log_event(new, now, f"feed:{drive}:{amount:+.2f}", event)
    return new


def veto(state: dict, drive: str, now: datetime, reason: str = "") -> dict:
    """
    否決提案：該維乘性回落並進入冷卻。
    否決理由記進 events（Phase 2 會回饋成 Ombre 念頭）。
    """
    if drive not in DRIVE_KEYS:
        raise ValueError(f"未知的 drive: {drive}")
    new = json.loads(json.dumps(state))
    cur = float(new["drives"].get(drive, 0.0))
    new["drives"][drive] = round(_clamp(cur * VETO_DAMP), 4)
    until = now.timestamp() + VETO_COOLDOWN_HOURS * 3600
    new.setdefault("veto_until", {})[drive] = datetime.fromtimestamp(until).isoformat(timespec="seconds")
    _log_event(new, now, f"veto:{drive}", reason)
    return new


def set_gate(state: dict, gate: str, value: bool, now: datetime, note: str = "") -> dict:
    """開關閘門（intimacy_ok 等）。driven 閘 Phase 1 拒改，鐵律：只讀。"""
    if gate == "driven":
        raise ValueError("Phase 1 只讀：driven 閘不開放（行為接管是 Phase 3 的事）")
    if gate not in ("intimacy_ok",):
        raise ValueError(f"未知的 gate: {gate}")
    new = json.loads(json.dumps(state))
    new.setdefault("gates", {})[gate] = bool(value)
    _log_event(new, now, f"gate:{gate}={'on' if value else 'off'}", note)
    return new


def _log_event(state: dict, now: datetime, kind: str, note: str = "") -> None:
    """就地記一筆事件（僅供內部在 deep copy 上使用）。"""
    events = state.setdefault("events", [])
    events.append({
        "ts": now.isoformat(timespec="seconds"),
        "kind": kind,
        "note": str(note)[:200],
    })
    del events[:-MAX_EVENTS]


# ---------------------------------------------------------
# 持久化層（與 letters/self_concept 同款：atomic write + lock）
# ---------------------------------------------------------

class DesireStore:
    """desire_state.json 的讀寫入口。所有讀取都先 tick（惰性時間推進）。"""

    def __init__(self, buckets_dir: str):
        self.path = os.path.join(buckets_dir, "desire_state.json")
        self._lock = threading.RLock()

    def load(self, now: datetime | None = None) -> dict:
        now = now or datetime.now()
        with self._lock:
            state = self._load_unlocked()
            return tick(state, now)

    def mutate(self, fn, now: datetime | None = None) -> dict:
        """load → tick → fn(state) → save，整段持鎖。fn 回傳新 state。"""
        now = now or datetime.now()
        with self._lock:
            state = tick(self._load_unlocked(), now)
            new_state = fn(state)
            self._save_unlocked(new_state)
            return new_state

    def save(self, state: dict) -> None:
        with self._lock:
            self._save_unlocked(state)

    def _load_unlocked(self) -> dict:
        if not os.path.exists(self.path):
            return default_state()
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return default_state()
        if not isinstance(data, dict) or "drives" not in data:
            return default_state()
        # 缺鍵補齊（版本演進容錯）
        base = default_state()
        for k in DRIVE_KEYS:
            data.setdefault("drives", {}).setdefault(k, base["drives"][k])
        for key in ("fatigue", "gates", "veto_until", "events", "updated_at", "last_intent"):
            data.setdefault(key, base[key])
        return data

    def _save_unlocked(self, state: dict) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = f"{self.path}.{secrets.token_hex(4)}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
