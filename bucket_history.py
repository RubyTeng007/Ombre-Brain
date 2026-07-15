# ============================================================
# Module: Bucket History (bucket_history.py)
# 模塊：記憶桶歷史（可復原）
#
# Every destructive write to a bucket first snapshots the state it is about to
# overwrite, with who did it. Restoring is itself a write, so it snapshots too:
# nothing is ever a one-way door.
# 每一次破壞性寫入，都先把「即將被蓋掉的狀態」連同「誰做的」存下來。
# 還原本身也是一次寫入，所以它也會先存快照：沒有任何一扇門是單向的。
#
# Why this exists / 為什麼有這個東西：
#   dream() is READ-ONLY — it lists buckets and prints them, it writes nothing.
#   So "a bad dream" is never dream's own doing: the damage is in the
#   trace(resolved=1) / hold(feel=True, source_bucket=X) calls that FOLLOW it,
#   and those land on bucket_manager.update(). That is why this hangs off the
#   update choke point and not off dream.
#   dream() 完全唯讀——它列桶、印字，一個字都不寫。所以「壞掉的 dream」從來
#   不是 dream 自己幹的：破壞在它「之後」那串 trace/hold，而它們全都落在
#   bucket_manager.update()。所以這個東西掛在 update 這個收口，不掛在 dream。
#
# Prior art: Letta's BlockHistory (letta/orm/block_history.py). Copied: the
# standalone PK + UNIQUE(bucket_id, seq) index rather than a composite PK; the
# full-value snapshot rather than a diff; actor_id as a plain string and NOT a
# foreign key (the writer may not exist by the time you read the row).
# NOT copied: its current_history_entry_id pointer with undo/redo. Its
# checkpoint_block_async deletes every row with seq > current, so undo-then-edit
# destroys the redo chain permanently. We make restore a normal write instead —
# one mechanism, no pointer, no branch destruction, and it can land on ANY seq
# rather than only ±1. Simpler than Letta and strictly more capable.
# 抄 Letta：獨立 PK + UNIQUE(bucket_id, seq)、全量快照不存 diff、actor_id 是純
# 字串不是外鍵（讀到這列時寫入者可能已經不存在了）。不抄：它的 undo/redo 指標
# ——它的 checkpoint 會刪掉所有 seq > current 的列，undo 之後再編輯，redo 鏈
# 永久消失。我們讓「還原」就是一次普通寫入：一個機制、沒有指標、沒有分支銷毀，
# 而且能還原到任意一點而不只是 ±1。
#
# Depended on by: bucket_manager.py, server.py
# ============================================================

import os
import json
import uuid
import sqlite3
import logging
from typing import NamedTuple, Optional

from utils import now_iso

logger = logging.getLogger("ombre_brain.history")


# --- Actor types: what the SYSTEM observed about the origin of a write ---
# --- actor 型別：系統「觀察到」的寫入來源 ---
ACTOR_TYPES = ("ruby", "cyan", "decay", "merge", "system")

# --- Operations / 操作 ---
OPS = ("create", "update", "merge", "delete", "archive", "revive", "restore")


class Actor(NamedTuple):
    """Who made a write.

    `type` and `id` are OBSERVED: they are stamped at the tool/endpoint
    boundary from which handler actually ran, so a caller cannot claim to be
    something else. They are facts about the system.

    `during` is SELF-REPORTED: Cyan declaring the state it believes it was in
    ("I was digesting"). It is not a fact the system checked. It lives in its
    own column named `self_reported_during` precisely so that no reader — human,
    query, or future model skimming the schema — can mistake it for an
    observation. It must NEVER be folded into actor_id for convenience.

    type/id 是「觀察」：在工具/端點邊界由實際跑的那個 handler 蓋章，呼叫者無法
    謊報，它們是關於系統的事實。
    during 是「自陳」：Cyan 自己宣告它以為當時的狀態（「我在消化」）。系統沒有
    查證過它。它住在自己的欄位 self_reported_during 裡，正是為了讓任何讀它的東西
    ——人、查詢、未來掃過 schema 的模型——都不可能把它誤讀成觀察。
    永遠不准為了方便把它併進 actor_id。

    This is the same rule as the mirage bucket: a dream is residue, not fact, so
    it gets its own bucket, its own channel, and never enters the portraits.
    Different epistemic status, different container.
    這條規矩就是 mirage 桶的同一條：夢是殘影不是事實，所以它有自己的桶、自己的
    通道，永不進畫像。不同的知識論地位，就該有不同的容器。
    """

    type: str = "system"
    id: str = "unknown"
    during: Optional[str] = None


# Convenience actors for the fixed, non-negotiable origins.
# 固定來源的現成 actor。
ACTOR_DECAY = Actor("decay", "decay_cycle")
ACTOR_SYSTEM = Actor("system", "unknown")


class BucketHistory:
    """Append-only per-bucket snapshot log with restore.
    每桶一條的追加式快照日誌，可還原。"""

    def __init__(self, config: dict):
        self.base_dir = config["buckets_dir"]
        self.path = os.path.join(self.base_dir, "history.db")
        hist_cfg = config.get("history", {}) or {}
        self.enabled = bool(hist_cfg.get("enabled", True))
        if self.enabled:
            self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        try:
            os.makedirs(self.base_dir, exist_ok=True)
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS bucket_history (
                        id                   TEXT PRIMARY KEY,
                        bucket_id            TEXT NOT NULL,
                        seq                  INTEGER NOT NULL,
                        ts                   TEXT NOT NULL,
                        -- observed by the system / 系統觀察到的
                        actor_type           TEXT NOT NULL,
                        actor_id             TEXT NOT NULL,
                        op                   TEXT NOT NULL,
                        -- self-reported by Cyan; never an observation
                        -- Cyan 自陳；永遠不是觀察
                        self_reported_during TEXT,
                        -- the state being overwritten / 即將被蓋掉的狀態
                        content              TEXT NOT NULL,
                        meta                 TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ix_bucket_history_bucket_seq "
                    "ON bucket_history(bucket_id, seq)"
                )
        except Exception as e:
            logger.error(f"History DB init failed / 歷史表初始化失敗: {e}")
            self.enabled = False

    def snapshot(
        self,
        bucket_id: str,
        content: str,
        meta: dict,
        op: str,
        actor: Optional[Actor] = None,
    ) -> Optional[int]:
        """Record the state about to be overwritten. Returns the new seq.
        記下即將被蓋掉的狀態，回傳新的 seq。

        Never raises: a failure to record history must not block the write the
        caller actually asked for. It is logged loudly instead.
        永不拋例外：記錄歷史失敗不該擋住呼叫者真正要做的那次寫入，改為大聲記 log。
        """
        if not self.enabled:
            return None
        actor = actor or ACTOR_SYSTEM
        try:
            with self._connect() as conn:
                for attempt in (1, 2):
                    row = conn.execute(
                        "SELECT COALESCE(MAX(seq), 0) + 1 FROM bucket_history WHERE bucket_id = ?",
                        (bucket_id,),
                    ).fetchone()
                    seq = int(row[0])
                    try:
                        conn.execute(
                            "INSERT INTO bucket_history (id, bucket_id, seq, ts, actor_type, "
                            "actor_id, op, self_reported_during, content, meta) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                f"hist-{uuid.uuid4()}",
                                bucket_id,
                                seq,
                                now_iso(),
                                str(actor.type)[:32],
                                str(actor.id)[:64],
                                str(op)[:32],
                                str(actor.during)[:32] if actor.during else None,
                                content or "",
                                json.dumps(meta, ensure_ascii=False, default=str),
                            ),
                        )
                        return seq
                    except sqlite3.IntegrityError:
                        # Lost a seq race; recompute once. The UNIQUE index is
                        # what makes this safe rather than silently overwriting.
                        # seq 撞號；重算一次。UNIQUE index 才是讓這裡安全、
                        # 而不是靜默覆蓋的東西。
                        if attempt == 2:
                            raise
        except Exception as e:
            logger.error(
                f"History snapshot FAILED (write proceeds unrecorded) / "
                f"歷史快照失敗（該次寫入將無紀錄）: {bucket_id} op={op}: {e}"
            )
        return None

    def list(self, bucket_id: str, limit: int = 20) -> list[dict]:
        """Newest first. 最新的在前。"""
        if not self.enabled:
            return []
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM bucket_history WHERE bucket_id = ? "
                    "ORDER BY seq DESC LIMIT ?",
                    (bucket_id, max(1, int(limit))),
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.warning(f"History list failed / 讀歷史失敗: {bucket_id}: {e}")
            return []

    def get(self, bucket_id: str, seq: int) -> Optional[dict]:
        if not self.enabled:
            return None
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM bucket_history WHERE bucket_id = ? AND seq = ?",
                    (bucket_id, int(seq)),
                ).fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.warning(f"History get failed / 讀歷史失敗: {bucket_id}#{seq}: {e}")
            return None

    def count(self, bucket_id: str) -> int:
        if not self.enabled:
            return 0
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM bucket_history WHERE bucket_id = ?",
                    (bucket_id,),
                ).fetchone()
                return int(row[0])
        except Exception:
            return 0
