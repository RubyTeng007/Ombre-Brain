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
    paths = [f"{REMOTE_DIR}/{f}" for f in files]
    out = ssh(args, ["sha256sum", *paths], capture=True).stdout or ""
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("OMBRE_VPS_HOST", DEFAULT_HOST))
    parser.add_argument("--key", default=os.environ.get("OMBRE_VPS_KEY", DEFAULT_KEY))
    parser.add_argument("--force", action="store_true",
                        help="skip the dirty-tree and live-drift guards")
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
    step("live drift check")
    marker = ssh(args, ["cat", MARKER], capture=True, check=False).stdout.strip()
    if marker:
        remote_hashes = sha256_remote(args, FILES)
        drifted = []
        for f in FILES:
            expected = git_show(marker, f)
            if expected is None:
                continue  # file didn't exist at that commit (new file this deploy)
            if hashlib.sha256(expected).hexdigest() != remote_hashes.get(f):
                drifted.append(f)
        if drifted and not args.force:
            print(f"live 檔案偏離上次部署章 {marker[:12]}（可能有人直接改了 live）：")
            for f in drifted:
                print(f"  DRIFT {f}")
            print("先把 live 的改動撈回 mirror，或確認可覆蓋再 --force。")
            return 1
        print(f"live == 部署章 {marker[:12]}" if not drifted else f"忽略 {len(drifted)} 個漂移（--force）")
    else:
        print("VPS 上沒有部署章（首次使用本腳本）——跳過漂移檢查。")

    # --- 4. stage files on the VPS ---
    deploy_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    stage = f"/tmp/ombre-deploy-{deploy_id}"
    step(f"stage → {stage}")
    ssh(args, ["mkdir", "-p", stage])
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
        # plan buckets dir for the promise ledger
        f"install -d -o {OWNER} -g {OWNER} {REMOTE_DIR}/buckets/plan",
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
