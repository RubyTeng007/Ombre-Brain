# ============================================================
# Tests for deploy_ombre.py's live drift gate (2026-07-16 audit).
# deploy_ombre.py 漂移守門的測試（2026-07-16 審計）。
#
# The three fail-open trapdoors this gate closes, each of which used to
# print a green light:
# 1. marker unreadable / garbage → old code treated it as "first deploy"
#    and skipped the check entirely (deleting the marker = guard offline).
# 2. marker commit not in this clone → git_show returned None for every
#    file, the "new file" tolerance swallowed all of them, and the script
#    printed "live == 部署章 X" — a lie — then rolled live back. This
#    EXACTLY happened in shape on 07-15: /opt/ombre-brain carried 5
#    commits no other clone had.
# 3. remote hashing failed wholesale → remote_hashes empty → every file
#    skipped → all green.
# ============================================================

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("deploy_ombre", ROOT / "deploy_ombre.py")
deploy_ombre = importlib.util.module_from_spec(spec)
sys.modules["deploy_ombre"] = deploy_ombre
spec.loader.exec_module(deploy_ombre)


def _fake_ssh(monkeypatch, responses):
    """responses: [marker_out] or [marker_out, drift_out]. Keyed by command
    content so the gate can be invoked more than once per test."""
    marker_out = responses[0]
    drift_out = responses[1] if len(responses) > 1 else ""

    def fake(args, remote, *, capture=False, check=True):
        text = " ".join(remote) if isinstance(remote, list) else str(remote)
        out = marker_out if "MARKERCHECK-DONE" in text else drift_out
        return SimpleNamespace(stdout=out, returncode=0)

    monkeypatch.setattr(deploy_ombre, "ssh", fake)


ARGS = SimpleNamespace(force=False)
GOOD_MARKER = "a" * 40


def test_unreadable_host_refuses(monkeypatch):
    """連章都問不到（沒有哨兵）＝拒絕，--force 也不行。"""
    _fake_ssh(monkeypatch, [""])  # no MARKERCHECK-DONE sentinel
    assert deploy_ombre.live_drift_gate(SimpleNamespace(force=True)) == 1


def test_missing_marker_refuses_without_force(monkeypatch):
    """章不在＝首次部署或章被刪，要人親口 --force，不能靜默放行。"""
    _fake_ssh(monkeypatch, ["NO-MARKER\nMARKERCHECK-DONE"])
    assert deploy_ombre.live_drift_gate(SimpleNamespace(force=False)) == 1
    assert deploy_ombre.live_drift_gate(SimpleNamespace(force=True)) is None


def test_garbage_marker_refuses(monkeypatch):
    _fake_ssh(monkeypatch, ["not-a-commit-hash\nMARKERCHECK-DONE"])
    assert deploy_ombre.live_drift_gate(SimpleNamespace(force=True)) == 1


def test_marker_commit_not_in_clone_refuses(monkeypatch):
    """07-15 的形狀：live 有本 clone 沒有的歷史 → 舊碼印假綠燈後整包倒退。"""
    _fake_ssh(monkeypatch, [f"{GOOD_MARKER}\nMARKERCHECK-DONE"])

    def fake_run(cmd, cwd=None, capture_output=False):
        return SimpleNamespace(returncode=1, stdout=b"", stderr=b"")

    monkeypatch.setattr(deploy_ombre.subprocess, "run", fake_run)
    assert deploy_ombre.live_drift_gate(SimpleNamespace(force=True)) == 1


def _known_commit_env(monkeypatch, drift_out):
    """marker commit exists; second ssh call returns the sha256 script output."""
    _fake_ssh(monkeypatch, [f"{GOOD_MARKER}\nMARKERCHECK-DONE", drift_out])

    def fake_run(cmd, cwd=None, capture_output=False):
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(deploy_ombre.subprocess, "run", fake_run)


def test_truncated_hash_output_refuses(monkeypatch):
    """遠端 hash 腳本沒跑完（缺哨兵）＝拒絕。舊碼在這裡全綠。"""
    _known_commit_env(monkeypatch, "deadbeef" * 8 + "  /opt/ombre-brain/server.py\n")
    assert deploy_ombre.live_drift_gate(SimpleNamespace(force=True)) == 1


def test_unreadable_file_refuses(monkeypatch):
    """主機沒說 ABSENT 也沒給 hash＝讀取失敗，不是新檔，拒絕。"""
    monkeypatch.setattr(deploy_ombre, "FILES", ["server.py"])
    _known_commit_env(monkeypatch, "DRIFTCHECK-DONE\n")
    assert deploy_ombre.live_drift_gate(SimpleNamespace(force=True)) == 1


def test_absent_file_is_new_not_drift(monkeypatch):
    """主機自己說 ABSENT＝本次新檔，放行。"""
    monkeypatch.setattr(deploy_ombre, "FILES", ["server.py"])
    _known_commit_env(monkeypatch, "ABSENT server.py\nDRIFTCHECK-DONE\n")
    assert deploy_ombre.live_drift_gate(SimpleNamespace(force=False)) is None


def test_drift_refuses_without_force(monkeypatch):
    monkeypatch.setattr(deploy_ombre, "FILES", ["server.py"])
    _known_commit_env(
        monkeypatch,
        "b" * 64 + "  /opt/ombre-brain/server.py\nDRIFTCHECK-DONE\n",
    )
    monkeypatch.setattr(deploy_ombre, "git_show", lambda commit, path: b"local content")
    assert deploy_ombre.live_drift_gate(SimpleNamespace(force=False)) == 1
    assert deploy_ombre.live_drift_gate(SimpleNamespace(force=True)) is None


def test_matching_live_passes(monkeypatch):
    import hashlib

    monkeypatch.setattr(deploy_ombre, "FILES", ["server.py"])
    content = b"the deployed content"
    _known_commit_env(
        monkeypatch,
        hashlib.sha256(content).hexdigest()
        + "  /opt/ombre-brain/server.py\nDRIFTCHECK-DONE\n",
    )
    monkeypatch.setattr(deploy_ombre, "git_show", lambda commit, path: content)
    assert deploy_ombre.live_drift_gate(SimpleNamespace(force=False)) is None
