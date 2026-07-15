# ============================================================
# Tests for the 2026-07-16 batch-6 fixes (audit follow-up)
# 2026-07-16 第六批修復的測試（審計跟進）
#
# Covers: pin no longer clobbers the pre-pin importance, restore refuses
# to report success when it leaves a ghost, and delete stops naming a
# history seq it never verified.
# ============================================================

import asyncio
import re

import frontmatter
import pytest
from unittest.mock import AsyncMock, MagicMock

import server as srv


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def mgr(test_config):
    """A manager with history actually wired.

    The shared `bucket_mgr` fixture leaves history=None, so any restore/delete
    recoverability test written against it passes vacuously. That is how these
    bugs survived: restore() returns False for want of any snapshot at all, and
    the test reads that as "refused correctly".
    共用的 bucket_mgr fixture 沒有接歷史，拿它寫的復原測試會空過——這些 bug
    就是這樣活下來的：restore() 因為根本沒有快照而回 False，測試卻讀成「正確拒絕」。
    """
    from bucket_manager import BucketManager
    from bucket_history import BucketHistory
    return BucketManager(test_config, history=BucketHistory(test_config))


@pytest.fixture
def wired(test_config, mgr, mock_embedding_engine, mock_dehydrator, monkeypatch):
    decay_stub = MagicMock()
    decay_stub.ensure_started = AsyncMock()
    decay_stub.calculate_score = lambda meta: 1.0

    monkeypatch.setattr(srv, "config", test_config)
    monkeypatch.setattr(srv, "bucket_mgr", mgr)
    monkeypatch.setattr(srv, "bucket_history", mgr.history)
    monkeypatch.setattr(srv, "embedding_engine", mock_embedding_engine)
    monkeypatch.setattr(srv, "dehydrator", mock_dehydrator)
    monkeypatch.setattr(srv, "decay_engine", decay_stub)
    return srv


def _meta(mgr, bucket_id):
    return dict(frontmatter.load(mgr._find_bucket_file(bucket_id)).metadata)


def _boom(*a, **k):
    raise OSError(18, "Invalid cross-device link")


# ---------------------------------------------------------
# 1. 釘選不可以吃掉釘選前的重要度
#    工具層若自己塞 importance=10，update() 的 importance 分支會先跑，把原值
#    蓋成 10，接著 pinned 分支才把「已經是 10」存進 importance_prepin。取消
#    釘選就再也回不去。病因只在工具層，manager 自己一直是對的。
# ---------------------------------------------------------
class TestPinKeepsPrePinImportance:
    def test_mcp_trace_pin_then_unpin_restores_original(self, wired, mgr):
        bid = _run(mgr.create(content="重要度四的桶", name="四號桶", importance=4))

        _run(wired.trace(bucket_id=bid, pinned=1))
        assert _meta(mgr, bid)["importance_prepin"] == 4, "prepin 必須是原值，不是 10"
        assert _meta(mgr, bid)["importance"] == 10, "釘選時 importance 仍須鎖成 10"

        _run(wired.trace(bucket_id=bid, pinned=0))
        assert _meta(mgr, bid)["importance"] == 4, "取消釘選要回到原值"

    def test_manager_locks_importance_without_the_tool_layer_passing_it(self, mgr):
        # 守「刪掉工具層那兩行之後，釘選仍然會鎖」——否則修法本身就弄壞了功能。
        bid = _run(mgr.create(content="重要度三的桶", name="三號桶", importance=3))
        _run(mgr.update(bid, pinned=True))
        assert _meta(mgr, bid)["importance"] == 10
        assert _meta(mgr, bid)["importance_prepin"] == 3

    def test_repin_does_not_clobber_prepin(self, wired, mgr):
        bid = _run(mgr.create(content="重複釘選", name="重複釘選", importance=2))
        _run(wired.trace(bucket_id=bid, pinned=1))
        _run(wired.trace(bucket_id=bid, pinned=1))
        assert _meta(mgr, bid)["importance_prepin"] == 2
        _run(wired.trace(bucket_id=bid, pinned=0))
        assert _meta(mgr, bid)["importance"] == 2


# ---------------------------------------------------------
# 2. restore 搬移失敗時不可以回報成功，也不可以留下幽靈
#    幽靈態＝檔案在 archive/ 但 metadata 寫 dynamic：list_all 走目錄所以看不到它，
#    revive() 只認 type=="archived" 所以拒修它。舊行為先寫 type=dynamic 再搬，
#    失敗正好落在這一格，然後 return True。
# ---------------------------------------------------------
class TestRestoreNeverReportsAGhost:
    def test_restore_returns_false_when_relocation_fails(self, mgr, monkeypatch):
        bid = _run(mgr.create(content="會被歸檔的桶", name="歸檔桶"))
        _run(mgr.update(bid, content="第二版"))
        assert _run(mgr.archive(bid)) is True

        monkeypatch.setattr(mgr, "_move_bucket", _boom)
        assert _run(mgr.restore(bid, seq=1)) is False, "搬不動就不是還原成功"

    def test_failed_restore_leaves_something_revive_can_repair(self, mgr, monkeypatch):
        bid = _run(mgr.create(content="會被歸檔的桶2", name="歸檔桶2"))
        _run(mgr.update(bid, content="第二版"))
        assert _run(mgr.archive(bid)) is True

        monkeypatch.setattr(mgr, "_move_bucket", _boom)
        _run(mgr.restore(bid, seq=1))

        # 搬移失敗後 type 必須仍是 archived，這樣 revive() 還救得回來。
        assert _meta(mgr, bid)["type"] == "archived", "搬不動就別改 type，留給 revive() 救"

        # 磁碟故障排除後，系統要能自己把它接回來（幽靈的定義就是「再也接不回來」）。
        monkeypatch.undo()
        assert _run(mgr.revive(bid)) is True, "失敗的還原必須留下 revive() 修得動的狀態"
        assert _meta(mgr, bid)["type"] == "dynamic"

    def test_successful_restore_still_works(self, mgr):
        # 守「反轉順序沒把還原本身弄壞」。
        bid = _run(mgr.create(content="原始正文", name="正常還原"))
        _run(mgr.update(bid, content="改過的正文"))
        assert _run(mgr.restore(bid, seq=1)) is True
        assert frontmatter.load(mgr._find_bucket_file(bid)).content.strip() == "原始正文"


# ---------------------------------------------------------
# 3. delete 不可以宣稱一個沒驗證過的 history seq
#    舊行為：delete() 把 snapshot() 的 seq 丟掉，server 再用 COUNT(*) 重建一個。
#    快照失敗時那個數字會指向更舊的版本，使用者照著印出的指令 restore 會拿到
#    舊正文，而且沒有任何錯誤。
# ---------------------------------------------------------
class TestDeleteTellsTheTruthAboutRecovery:
    def test_delete_reports_the_seq_the_snapshot_actually_wrote(self, wired, mgr):
        bid = _run(mgr.create(content="v1", name="誠實桶"))
        _run(mgr.update(bid, content="v2"))
        _run(mgr.update(bid, content="v3"))

        msg = _run(wired.trace(bucket_id=bid, delete=True))
        seq = int(re.search(r"restore_seq=(\d+)", msg).group(1))

        # 印出來的指令就要能用，而且要拿回刪除當下那一版（v3），不是更舊的。
        assert _run(mgr.restore(bid, seq=seq)) is True
        assert frontmatter.load(mgr._find_bucket_file(bid)).content.strip() == "v3"

    def test_delete_admits_it_when_the_snapshot_failed(self, wired, mgr, monkeypatch):
        bid = _run(mgr.create(content="v1", name="快照壞掉的桶"))
        _run(mgr.update(bid, content="v2"))

        # snapshot() 是 fail-open 的：它吞掉例外並回 None。刪除照樣進行——
        # 那是刻意的——但回覆不准再宣稱救得回來。
        monkeypatch.setattr(mgr.history, "snapshot", lambda *a, **k: None)
        msg = _run(wired.trace(bucket_id=bid, delete=True))

        assert "restore_seq=" not in msg, f"快照失敗時不該印出可用的 restore 指令：{msg}"
        assert "救不回來" in msg, f"快照失敗要說出口：{msg}"

    def test_delete_result_is_not_a_bare_bool(self, mgr):
        # 這條擋的是回頭寫 `if await mgr.delete(...)`——NamedTuple 恆為真，
        # 那樣寫會靜默地永遠成立。server.py:4142 本來就是這樣寫的。
        bid = _run(mgr.create(content="v1", name="型別桶"))
        result = _run(mgr.delete(bid))
        assert result.ok is True
        assert isinstance(result.seq, int)
        missing = _run(mgr.delete("nonexistent"))
        assert missing.ok is False and missing.seq is None


# ---------------------------------------------------------
# 4. 脫水快取鍵：正文＋模型＋提示詞版本
#    舊行為只雜湊正文。model 有存欄位，但不在 WHERE 裡——所以換模型
#    （server.py 會在執行期改 dehydrator.model）或改提示詞之後，同一份正文
#    照樣命中舊那筆，舊模型的摘要永遠服務下去。而 invalidate_cache() 零呼叫、
#    沒有 TTL、沒有版本欄位，唯一的失效途徑是「正文被改」。
# ---------------------------------------------------------
@pytest.fixture
def dehy(test_config):
    from unittest.mock import MagicMock
    from dehydrator import Dehydrator
    cfg = dict(test_config)
    cfg["dehydration"] = {"api_key": "test-key", "max_tokens": 1024}
    d = Dehydrator(cfg)
    d.client = MagicMock()
    return d


class TestDehydrationCacheKey:
    def test_same_body_same_model_hits(self, dehy):
        dehy._set_cached_summary("正文", '{"summary": "a"}')
        assert dehy._get_cached_summary("正文") == '{"summary": "a"}'

    def test_switching_model_misses(self, dehy):
        dehy._set_cached_summary("正文", '{"summary": "deepseek 產的"}')
        dehy.model = "some-other-model"
        assert dehy._get_cached_summary("正文") is None, (
            "換了模型還命中舊快取＝新模型的名義下送出舊模型的摘要"
        )

    def test_bumping_prompt_version_misses(self, dehy, monkeypatch):
        import dehydrator as dh
        dehy._set_cached_summary("正文", '{"summary": "舊提示詞產的"}')
        monkeypatch.setattr(dh, "PROMPT_VERSION", "2")
        assert dehy._get_cached_summary("正文") is None, (
            "改了提示詞，舊快取就該失效——否則提示詞怎麼改都不會生效"
        )

    def test_different_bodies_still_separate(self, dehy):
        dehy._set_cached_summary("正文A", '{"summary": "a"}')
        dehy._set_cached_summary("正文B", '{"summary": "b"}')
        assert dehy._get_cached_summary("正文A") == '{"summary": "a"}'
        assert dehy._get_cached_summary("正文B") == '{"summary": "b"}'


# ---------------------------------------------------------
# 5. 輸入截斷不可以是無聲的
# ---------------------------------------------------------
class TestDehydrationInputTruncation:
    @pytest.mark.asyncio
    async def test_long_body_is_marked_not_silently_cut(self, dehy):
        from unittest.mock import AsyncMock
        import dehydrator as dh
        seen = {}

        async def capture(**kw):
            seen["user"] = kw["messages"][1]["content"]
            raise RuntimeError("stop here")

        dehy.client.chat.completions.create = AsyncMock(side_effect=capture)
        body = "字" * (dh.DEHYDRATE_INPUT_LIMIT + 500)
        with pytest.raises(RuntimeError):
            await dehy._api_dehydrate(body)
        assert "省略" in seen["user"], "超過上限就要明說被切了，不能靜默截斷"
        assert "500 字" in seen["user"]

    @pytest.mark.asyncio
    async def test_the_live_long_buckets_now_fit_whole(self, dehy):
        from unittest.mock import AsyncMock
        import dehydrator as dh
        seen = {}

        async def capture(**kw):
            seen["user"] = kw["messages"][1]["content"]
            raise RuntimeError("stop here")

        dehy.client.chat.completions.create = AsyncMock(side_effect=capture)
        # 線上最長的桶約 3982 字（舊上限 3000 咬掉了它 982 字，而那段是待辦清單）。
        body = "字" * 3982
        with pytest.raises(RuntimeError):
            await dehy._api_dehydrate(body)
        assert seen["user"] == body, "現有的長桶要能整份進模型"
        assert "省略" not in seen["user"]
