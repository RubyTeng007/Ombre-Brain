#!/usr/bin/env python3
"""Deploy Ombre Brain source files to the VPS (/opt/ombre-brain).

Mirrors the deploy_telegram.py discipline, sized for this repo:
refuse a dirty git tree → local pytest + py_compile gate → live-drift check
against the last deployed commit → stage → remote py_compile with the live
venv → backup → two-phase install in one ssh script → restart → hash verify →
health check → stamp the deployed commit on the VPS.

Ombre Brain 部署腳本（比照 telegram 部署紀律）：
髒工作樹拒部 → 本地測試門 → live 漂移檢查 → 暫存 → 遠端編譯檢查 →
備份 → 單一 ssh 兩段式安裝 → 重啟 → hash 驗證 → 健康檢查 → 蓋部署章。
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_HOST = "root@45.76.219.5"
DEFAULT_KEY = "/Users/dengyuru/.ssh/cyan_vps_ed25519"
REMOTE_DIR = "/opt/ombre-brain"
SERVICE = "ombre-brain"
OWNER = "ombre"
MARKER = f"{REMOTE_DIR}/.deployed-commit"

# Active source files: everything server.py imports, plus the dashboard.
# 部署清單：server.py 的全部專案內依賴＋dashboard。config.yaml 是 live 資料，永不覆蓋。
FILES = [
    "server.py",
    "bucket_manager.py",
    "bucket_history.py",
    "bm25_index.py",
    "decay_engine.py",
    "dehydrator.py",
    "embedding_engine.py",
    "utils.py",
    "desire.py",
    "letters.py",
    "self_concept.py",
    "reading_shelf.py",
    "api_usage_guard.py",
    "import_memory.py",
    "dashboard.html",
]
PY_FILES = [f for f in FILES if f.endswith(".py")]


def run(cmd: list[str], *, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=capture, check=check)


def ssh(args: argparse.Namespace, remote: list[str], *, capture: bool = False, check: bool = True):
    return run(["ssh", "-i", args.key, "-o", "BatchMode=yes", args.host, *remote],
               capture=capture, check=check)


def sha256_local(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_remote(args: argparse.Namespace, files: list[str]) -> dict[str, str]:
    """Missing remote files simply don't appear in the result (a file that is
    new this deploy has nothing to drift from), so no check=True here.
    遠端不存在的檔案就不出現在結果裡（本次新增檔沒有漂移可查），不視為錯誤。"""
    paths = [f"{REMOTE_DIR}/{f}" for f in files]
    out = ssh(args, ["sha256sum", *paths], capture=True, check=False).stdout or ""
    hashes: dict[str, str] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            hashes[os.path.basename(parts[-1])] = parts[0]
    return hashes


def git_show(commit: str, path: str) -> bytes | None:
    proc = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT,
                          capture_output=True)
    return proc.stdout if proc.returncode == 0 else None


def step(msg: str) -> None:
    print(f"\n== {msg}")


def live_drift_gate(args: argparse.Namespace) -> int | None:
    """Fail-closed live drift gate. Returns an exit code to abort, None to proceed.
    fail-closed 的 live 漂移守門。回傳 exit code＝擋下，None＝放行。
    抽成函數是為了讓 tests/test_deploy_ombre.py 打得到它——2026-07-16 的
    教訓之一就是「守門存在但沒有任何測試執行它」。"""
    # 2026-07-16 審計把三條 fail-open 全部翻成 fail-closed。回滾部署是無聲的
    # （部署成功、健康檢查全綠、程式碼就是沒了），所以「驗不了」永遠不能拼成
    # 「放行」。三條舊活門：①章讀不到/是垃圾 → 舊碼當「首次使用」跳過；
    # ②章 commit 不在本 clone（07-15 實證發生過：/opt/ombre-brain 藏著 5 個
    # 誰都沒有的 commit）→ git_show 全回 None → 全部被「新檔」continue 吃掉
    # → 印出「live == 部署章」這句假話；③遠端 hash 整批失敗 → remote_hashes
    # 空 → 同樣全綠。
    step("live drift check")
    marker_proc = ssh(
        args,
        [f"cat {MARKER} 2>/dev/null || echo NO-MARKER; echo MARKERCHECK-DONE"],
        capture=True, check=False,
    )
    marker_out = (marker_proc.stdout or "").strip().splitlines()
    if "MARKERCHECK-DONE" not in marker_out:
        print("讀不到主機（連部署章都問不到）——拒絕部署。--force 不豁免這一關。")
        print("修好 ssh／網路再來。")
        return 1
    marker = marker_out[0].strip() if marker_out else "NO-MARKER"
    if marker == "NO-MARKER":
        # 章不在＝守門下線。首次部署與「有人刪了章」在這裡長得一樣，
        # 所以要人親口確認（--force），不能靜默放行。
        if not args.force:
            print("VPS 上沒有部署章——首次部署或章被刪都長這樣，無法驗證 live。")
            print("確認這真的是首次（或接受風險）就 --force。")
            return 1
        print("沒有部署章，--force 放行（跳過漂移檢查）。")
    elif not re.fullmatch(r"[0-9a-f]{40}", marker):
        print(f"部署章內容不是 commit hash（{marker[:60]!r}）——拒絕部署。")
        print("有人動過章檔。先上主機看 " + MARKER)
        return 1
    elif subprocess.run(
        ["git", "cat-file", "-e", f"{marker}^{{commit}}"],
        cwd=ROOT, capture_output=True,
    ).returncode != 0:
        # 這正是 07-15 的形狀：live 自己有本 clone 沒有的歷史。
        # 舊碼在這裡印「live == 部署章」然後整包倒退。
        print(f"live 的部署章 {marker[:12]} 不在本 clone——你缺 live 的歷史，")
        print("整包部署會把那段內容無聲倒退。先把 live 的 git 歷史收回來")
        print(f"（ssh 主機在 {REMOTE_DIR} 開 git bundle / push），再部署。")
        print("--force 不豁免這一關：缺歷史時沒有任何東西能算出漂移。")
        return 1
    else:
        script = "; ".join(
            [f"sha256sum {REMOTE_DIR}/{f} 2>/dev/null || echo ABSENT {f}" for f in FILES]
            + ["echo DRIFTCHECK-DONE"]
        )
        out = ssh(args, [script], capture=True, check=False).stdout or ""
        if "DRIFTCHECK-DONE" not in out:
            print("遠端 hash 讀取沒跑完——拒絕部署。--force 不豁免這一關。")
            return 1
        remote_hashes: dict[str, str] = {}
        absent: set[str] = set()
        for line in out.splitlines():
            parts = line.split()
            if line.startswith("ABSENT ") and len(parts) >= 2:
                absent.add(parts[-1])
            elif len(parts) >= 2 and re.fullmatch(r"[0-9a-f]{64}", parts[0]):
                remote_hashes[os.path.basename(parts[-1])] = parts[0]
        drifted = []
        unreadable = []
        for f in FILES:
            if f in absent:
                continue  # live 沒有這個檔＝本次新檔，沒有漂移可查
            if f not in remote_hashes:
                unreadable.append(f)  # 主機沒說 ABSENT 也沒給 hash＝讀取失敗
                continue
            expected = git_show(marker, f)
            if expected is None:
                continue  # 章 commit 已驗證存在；檔案不在那個 commit＝本次新檔
            if hashlib.sha256(expected).hexdigest() != remote_hashes[f]:
                drifted.append(f)
        if unreadable:
            print("這些檔案在主機上讀不到 hash（不是 ABSENT，是失敗）——拒絕部署：")
            for f in unreadable:
                print(f"  UNREADABLE {f}")
            return 1
        if drifted and not args.force:
            print(f"live 檔案偏離上次部署章 {marker[:12]}（可能有人直接改了 live）：")
            for f in drifted:
                print(f"  DRIFT {f}")
            print("先把 live 的改動撈回 mirror，或確認可覆蓋再 --force。")
            return 1
        print(f"live == 部署章 {marker[:12]}" if not drifted else f"忽略 {len(drifted)} 個漂移（--force）")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("OMBRE_VPS_HOST", DEFAULT_HOST))
    parser.add_argument("--key", default=os.environ.get("OMBRE_VPS_KEY", DEFAULT_KEY))
    parser.add_argument("--force", action="store_true",
                        help="skip the dirty-tree guard, a KNOWN live drift, and the "
                             "missing-marker refusal. It does NOT skip: unreadable host, "
                             "garbage marker, or a marker commit this clone doesn't have "
                             "— when nothing can be verified there is no flag.")
    parser.add_argument("--check", action="store_true",
                        help="only compare live hashes against local files, no deploy")
    parser.add_argument("--skip-tests", action="store_true",
                        help="skip the local pytest gate (emergencies only)")
    args = parser.parse_args()

    os.chdir(ROOT)

    if args.check:
        step("live vs local hash check")
        remote_hashes = sha256_remote(args, FILES)
        drift = 0
        for f in FILES:
            local = sha256_local(ROOT / f)
            live = remote_hashes.get(f, "<missing>")
            mark = "OK  " if local == live else "DIFF"
            if local != live:
                drift += 1
            print(f"{mark} {f}")
        return 1 if drift else 0

    # --- 1. clean tree guard ---
    step("git tree check")
    dirty = run(["git", "status", "--porcelain"], capture=True).stdout.strip()
    if dirty and not args.force:
        print("工作樹不乾淨——先 commit（或確定是自己的改動就 --force）：")
        print(dirty)
        return 1
    head = run(["git", "rev-parse", "HEAD"], capture=True).stdout.strip()
    print(f"HEAD {head[:12]}{' (dirty, --force)' if dirty else ''}")

    # --- 2. local gate: compile + tests ---
    step("local py_compile + pytest gate")
    run([sys.executable, "-m", "py_compile", *PY_FILES])
    if args.skip_tests:
        print("pytest SKIPPED (--skip-tests)")
    else:
        py = str(ROOT / ".venv/bin/python") if (ROOT / ".venv/bin/python").exists() else sys.executable
        run([py, "-m", "pytest", "tests/", "-q"])

    # --- 3. live drift check against the last deployed commit ---
    rc = live_drift_gate(args)
    if rc is not None:
        return rc

    # --- 4. stage files on the VPS ---
    # mktemp instead of a predictable second-granularity name: a compromised
    # low-privilege user could pre-create /tmp/ombre-deploy-<timestamp> and
    # swap payloads between the remote compile check and root's install.
    # mktemp 取代可預測的秒戳目錄名：被入侵的低權帳號可以預佔未來的秒戳目錄，
    # 在遠端編譯檢查與 root 安裝之間換包。（deploy_telegram 從一開始就這樣做。）
    deploy_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    step("stage (remote mktemp)")
    stage = (ssh(args, ["mktemp", "-d", "/tmp/ombre-deploy.XXXXXX"],
                 capture=True).stdout or "").strip()
    if not re.fullmatch(r"/tmp/ombre-deploy\.[A-Za-z0-9]+", stage):
        print(f"mktemp 回傳的路徑不合法（{stage!r}）——拒絕部署。")
        return 1
    print(f"stage → {stage}")
    run(["scp", "-i", args.key, "-q", *[str(ROOT / f) for f in FILES], f"{args.host}:{stage}/"])

    # --- 5. remote compile check with the LIVE venv ---
    step("remote py_compile (live venv)")
    ssh(args, [f"cd {stage} && {REMOTE_DIR}/.venv/bin/python -m py_compile " + " ".join(PY_FILES)])

    # --- 6. backup + two-phase install in ONE ssh script ---
    backup_dir = f"{REMOTE_DIR}/.backups/{deploy_id}"
    step(f"backup → {backup_dir}, two-phase install")
    lines = [
        "set -e",
        f"mkdir -p {backup_dir}",
        *[f"cp -p {REMOTE_DIR}/{f} {backup_dir}/ 2>/dev/null || true" for f in FILES],
        # phase 1: land everything beside its target
        *[f"install -m 644 -o {OWNER} -g {OWNER} {stage}/{f} {REMOTE_DIR}/{f}.new-{deploy_id}" for f in FILES],
        # phase 2: atomic renames (no mixed tree on a mid-script failure)
        *[f"mv {REMOTE_DIR}/{f}.new-{deploy_id} {REMOTE_DIR}/{f}" for f in FILES],
        # plan buckets dir for the promise ledger + dream dir for the dream channel
        f"install -d -o {OWNER} -g {OWNER} {REMOTE_DIR}/buckets/plan",
        f"install -d -o {OWNER} -g {OWNER} {REMOTE_DIR}/buckets/mirage",
        f"rm -rf {stage}",
    ]
    ssh(args, ["\n".join(lines)])

    # --- 7. restart + verify ---
    step("restart service")
    ssh(args, ["systemctl", "restart", SERVICE])
    active = (ssh(args, ["systemctl", "is-active", SERVICE], capture=True, check=False).stdout or "").strip()
    print(f"{SERVICE}: {active}")
    if active != "active":
        print("服務沒起來——live 備份在", backup_dir)
        return 1

    step("hash verify")
    remote_hashes = sha256_remote(args, FILES)
    bad = [f for f in FILES if sha256_local(ROOT / f) != remote_hashes.get(f)]
    if bad:
        print("hash 不一致：", bad)
        return 1
    print(f"{len(FILES)} 檔全部一致")

    step("health check")
    health = ssh(args, ["curl", "-s", "-m", "10", "http://127.0.0.1:8000/health"],
                 capture=True, check=False).stdout.strip()
    print(health or "<no response>")
    if not health:
        print("health 端點沒回應——live 備份在", backup_dir)
        return 1

    # --- 8. stamp the deployed commit ---
    ssh(args, [f"echo {head} > {MARKER} && chown {OWNER}:{OWNER} {MARKER}"])
    step(f"done — deployed {head[:12]} ({deploy_id})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
