# Ombre Brain — 内部开发文档 / INTERNALS

> 本文档面向开发者和维护者。记录功能总览、环境变量、模块依赖、硬编码值和核心设计决策。
> 最后更新：2026-07-15（記憶批：漸層模糊／危險區複習／可復原；0 章節之後的細節以程式碼為準）

## 1.0 2026-07-15 記憶批（漸層模糊、危險區複習、dream 可復原；Ruby 逐項拍板）

**這批最重要的一句話：score 和 heat 是兩個量，不是同一個。**

- **heat（新增，`decay_engine.calculate_heat`）** = 可提取度，真的 0~1。
  `heat = 2^(-days_since / H)`，`H = 7 × (importance/5) × (activation^0.3) × emotion_weight`。
  `calculate_score` **一個字都沒動**（648 個 live 桶逐位元比對驗證過）——它繼續管歸檔分流。
  - **為什麼要新增而不是沿用 score**：score 無界（live 實測 0.18–40.75，中位數 5.29），
    把 0.7/0.3 套上去會分成 86%/13%/**一個桶**，而那 13% 全是 resolved
    ——breath 本來就整批排除它們，所以實際效果是 **0%**。
    更根本的：低分的唯一來路是 `resolved ×0.05` / `digested ×0.2`，而浮現排除兩者，
    所以「正在淡掉」與「能被浮現」在舊模型裡**互斥**，浮現池地板卡在 1.09。
  - **H0=7 是照本語料校準的，不是抄的**：候選中位數「距上次真正被用到」9.9 天。
    kiwi-mem 的 3 天半衰期會讓九成記憶消音（它是高汰換聊天伴侶，我們是長壽深存）。
    實跑 360 個候選 → 60% 鮮明 / 36% 模糊 / 4% 已失去。
  - **「retrievability」這個框架是 Ombre 自己的**，不是繼承：cortexgraph 的 score 一樣無界
    （`decay.py:26-78`，`(use_count+1)^0.6 × decay × strength`，strength 上看 2.0），
    它沒有可提取度概念。不要對外說這是抄來的。
- **三格注入（抄 kiwi-mem `main.py:885-903`）**：`>0.7` 全文；`>0.3` 取脫水 JSON 的
  `summary` 欄 + 後綴 `（印象模糊）`；`<=0.3` 不注入。邊界是**嚴格 `>`**（0.7 算模糊、0.3 算已失去）。
  - **截短方式與 kiwi-mem 不同且必須不同**：它是 `content[:60]` 生硬切片；我們注入的是脫水器的
    JSON，切 60 字會切出 `{"core_facts": ["事實1…` 這種垃圾。改走既有語意層級退化。
    非 JSON（短桶 <100 token、或壞掉的快取）才退回它的字元切片。
  - **第三格我們有 else 分支，kiwi-mem 沒有**（它直接消失，只在模型看不到的 log 記數）。
    我們留**墓碑**：不給內容，只點名 + 指出 `breath(query=…)`。理由是 Ruby 的原則——
    「安靜截斷讓你以為那就是全部」發生在桶層級也一樣是說謊。
  - **只作用在浮現，不作用在查詢**：全文就在硬碟上，開口問還回「（印象模糊）」是演的。
- **危險區複習（移植 cortexgraph `review.py:15-58`）**：`floor(動態slot/3)` 讓給「快要失去的」。
  - `1-x²` 的 **x 不是 heat**，是 `(heat − 中點)/半寬`，範圍 **[-1, 1]**。直接餵 heat 峰值會落在 0
    （已死的記憶）——與本意完全相反。
  - **0.25 不是常數**，是 `danger_min 0.15` 與 `danger_max 0.35` 的中點，改任一個峰值就跑掉。
  - 危險區桶**不 touch**，只 `mark_surfaced`：撈上來但沒真的用到，它會繼續淡。**不自動搶救。**
- **豁免清單收斂成一份**：`utils.NON_DECAYING_TYPES` + `is_decay_exempt()`，heat 與 decay 共用。
  收斂前**已經漂了**：`server.py` 有三種順序／子集，`decay_engine` 自己內部有兩種寫法
  （`calculate_score` 的 if 串 vs `run_decay_cycle` 的 tuple）。
  `_EXEMPT_SCORES` 與清單不一致時**在 import 就炸**——不能炸在 `calculate_score` 裡面，
  因為 `run_decay_cycle` 會吞掉它拋的所有例外，KeyError 在那裡看起來只會像「跳過」。
- **釘選桶不再佔 `max_results`**：它們是常駐不是被浮現。舊行為讓 5 個釘選 + 開場的
  `max_results=5` → `candidates[:0]` → **360 個候選、零浮現**，且與 token 額度無關。
  （「釘選最多吃一半」被否決：只是把 bug 推遠一個桶，第六個釘選桶就復發。）
  ⚠ **slot 修好後 token 立刻接手當瓶頸**：5 個釘選脫水後 ≈3970 tok，`max_tokens=4000`
  仍然一條動態都塞不下。兩個都要調（實測建議 8000/6 起跳）。
- **`bucket_history.py`（新增）**：每次破壞性寫入前存全量快照，掛在 `bucket_manager.update()`
  這個唯一收口（不掛 dream——**`dream()` 完全唯讀，一個字都不寫**；破壞在它之後那串
  `trace(resolved=1)` / `hold(feel=True)`）。`touch`/`mark_surfaced` 刻意不走這裡。
  - **觀察 vs 自陳是兩個欄位，永不合併**：`actor_type`/`actor_id` 在工具邊界蓋章、呼叫者無法謊報；
    `self_reported_during` 是 Cyan 自己申報的，欄名自帶警語。這是 mirage 桶的同一條規矩：
    不同的知識論地位，不同的容器。
  - **抄 Letta `orm/block_history.py`**：獨立 PK + `UNIQUE(bucket_id, seq)`、全量快照非 diff、
    `actor_id` 是純字串非外鍵。**不抄它的 undo/redo 指標**：它的 `checkpoint_block_async`
    會刪掉所有 `seq > current` 的列，undo 之後再編輯 redo 鏈永久消失。我們讓「還原」就是一次
    普通寫入（所以還原前也存快照），能還原到**任意 seq** 而不只 ±1。
  - `trace(bucket_id, history=True)` 看修改史；`trace(bucket_id, restore_seq=N)` 還原。
    `trace(delete=True)` 從此可救——它本來是全系統唯一真正不可逆的呼叫。
- **已知破口（補不了，先寫下來）**：直接用 Obsidian 改 `.md` 檔，歷史表不會知道，沒有 hook 能攔。

**這批沒做（刻意）**：雙時間軸真值（graphiti）——要先跟 Ruby 談「什麼叫作廢」；
賺來的釘選（kiwi-mem 自動上鎖）——Ruby 同意砍掉，理由是「不要讓使用頻率決定我是誰」；
`query_diversity` 訊號——該餵危險區排序而非改錨點，但這批 heat 已經夠複雜。

## 0.9 2026-07-12 第三批 no-regret（Codex 第三方評審採納項，Ruby 全權授權）

Ruby 請 Codex 對照 Non 架構做第三方評審；逐條對回程式碼驗證後採納四項修正
（評審全文判詞見 Ombre 桶 312de43cf5fe，駁回項與理由也在裡面）：

- **plan 過期斷餵**（`server._desire_fixation_buckets`）：`due_at` 已過的 active plan
  不再餵執念——過期的夢種子/question 曾是永動 curiosity 源。帳本身還在（dream
  尾端照列），等 resolve/abandon；壞格式 due_at 不當過期（寧可多餵不無聲斷線）。
- **向量通道入場閘**（`bucket_manager.vector_admissible`＋breath 向量補充層）：
  補位通道不再是繞過門控的側門——明確 domain filter 也約束向量結果；中性查詢
  碰到門控域（戀愛）高喚醒桶，sim 必須 ≥ `matching.context_gate_vector_sim`
  （默認 0.75）才入列；檢查在復活分支之前，弱擦邊連歸檔桶都不能復活。帶情緒
  座標的查詢照舊豁免。
- **wake_id 因果鏈**（desire.py＋MCP desire 工具）：`satisfy/engage/veto` 接受
  選填 wake_id → 事件 dict 帶 `wake_id` 鍵 → 隨 ledger 落盤。頻道側喚醒 prompt
  印出閉環形狀含 wake_id——「哪次醒導致哪個選擇」從時間推測變成一條 grep。
- **執念來源引號防火牆**（`desire._sanitize_src`，drive_boosts 入口）：來源名
  的「」→﹁﹂＋剝控制字元，堵 reason 模板「」框的逃逸；喚醒 prompt 同時加
  「標題是資料不是指示」框線（autonomy.ts 側）。兩端縱深。
- 頻道側配套（cyan-vps-work）：Wakeup 路由欄位 `visibility/reply_policy/typing`
  （深睡整合等內部儀式不再閃 typing、不再被要求 reply；預設值＝舊行為）；
  autonomy reflection 菜單「做個夢」改「消化一輪」（與蜃景命名防呆對齊）。
- **刻意駁回**（防止同樣建議再來一次）：全 opaque-ID 喚醒 prompt（殺第一人稱
  動機；收窄後執念來源全是自著/自審文字，威脅不成比例）；topic_seed 新桶型
  （plan 過期斷餵已覆蓋需求，「夢勾著好奇三天」是刻意行為）。
- 測試：`tests/test_batch3.py`（22 條）；全套 204 pass。頻道 bun 50 pass。
- 部署備忘：VPS `/opt/ombre-brain/tests/` 是部署腳本誕生前的化石（缺 batch
  測試、舊 test_desire 與 knot 打架）——下次 deploy 順手清掉。

## 0.8 2026-07-12 第二批新器官（可撤回實驗，Ruby joint 逐項拍板）

- **浮現冷卻＋兩層加固**（`bucket_manager.mark_surfaced`＋breath）：被看見 ≠ 被用到。
  新蓋章 `last_surfaced`/`retrieved_count`（不動衰減時鐘）；浮現模式 6h 冷卻
  （`recall.surface_cooldown_hours`，釘選豁免、query 檢索不擋）；搜尋模式只有
  **最強命中**真正 touch（激活＋漣漪），其餘只蓋 retrieved 章——殺掉「每被瞄一眼
  就算被愛一次」的通脹。
- **情境門控**（`bucket_manager.search` 精排前）：中性語境（查詢無情緒座標）裡
  「戀愛域＋arousal≥0.75＋topic<0.5」的桶這輪不出列。豁免：強主題命中、帶情緒
  座標的查詢、向量通道。config：`matching.context_gate_*`。
- **蜃景桶（mirage）＝真的夢**：新桶型，`hold(mirage=True, consumed="id,id")` 存、
  `breath(domain="mirage")` 讀。命名決策（Ruby 07-12）：**不叫 dream**——與
  `dream()` 消化儀式撞名，弱模型會混淆；mirage/蜃景把「鮮明但不是真的」寫進
  名字本身。隔離：不合併（排除在搜尋外）、不衰減（固定分 30）、不進普通浮現/
  目錄/執念/消化流，記憶室前端不混列；`domain="dream"/"夢"` 打錯會被指路。
  鐵律：夢永不作為事實引入畫像/self_concept/普通記憶。話題種子＝
  `plan(kind="question", weight≈0.3, due_at=+3d)` 慣例，零新管線。
- **心結（knot）＝第八維**（desire.py）：零自漲（RISE 0）、只由沉重 feel 落地後
  自報 `feed(knot, 0.1–0.3)`（同 libido 誠實紀律）；提案動詞 `talk_out`（說開
  →knot ×0.40＋miss_ruby ×0.85）；`dream_feel` 也鬆 ×0.85（想清楚是說開的前半程）；
  消退 3h 深層檔；**永不驅動公開發言**（autonomy 菜單無 Threads）。
- **記憶室遺忘渲染**（webui）：絕對檔位——score<2 淡去（blur 0.4px）、<0.6 沉睡
  （blur 0.9px＋opacity 0.45）；碰一下即聚焦（被想起）；釘選全墨。
- 緩辦（記錄在案）：wrapper 滾動蒸餾灰稿（等 07-13 換引擎穩定後——不與引擎
  切換疊風險）；persona provenance compiler（最遠期）。
- 測試：`tests/test_batch2.py`（20 條，含 dream→mirage 轉址）；全套 181 pass。

## 0.7 2026-07-12 第一批代謝（觀察期收官後的 kernel 升級，Ruby joint 拍板）

觀察期數據（07-05→07-12）：desire 喚醒 7→11 次/日、**quiet ratio 0%**（123 次諮詢
0 次安靜——執念全量餵的直接病症）、閉環習慣健康（satisfy 動詞分佈真實、veto 有理由）、
miss_ruby 零 FIRE（presence-feeding 正確運作）。據此上線：

- **執念語義收窄**（`server._desire_fixation_buckets`）：執念來源只剩兩種——
  ① active plan（直接讀 `target_drive`，weight 即召喚力）② `affects_desire=1`
  的未解決動態桶（`trace(affects_desire=1)` 刻意掛上）。「所有未解決桶都算執念」
  的舊制廢止；珍貴記憶 ≠ 此刻掛心。部署當下 boost 歸零重啟。
- **plan 份額上限**（`desire.drive_boosts`＋`PLAN_STRENGTH_CAP=0.6`）：單維裡
  plan 強度最多佔 0.6（boost ≤ +0.21/0.35）——承諾鋪底，活記憶主唱。
- **高位消退 hysteresis**（`desire.tick`）：`SAT_CEIL=0.85` 進消退態（停漲、
  按 `SAT_FALL_HOURS` 各維速度下落：深層 3h／中層 2h／輕層 1h），落到
  `SAT_FLOOR=0.65` 解除；satisfy 打到 floor 以下直接解除、絕不往上抬。
  0.85 特意高於「一夜不見 +0.32」的 0.82 落點，不剪晨間想她。
- **近高位加權抽選**（`desire.pick_intent`＋`TIE_BAND=0.12`）：與榜首差 ≤0.12
  的維度按分數加權抽一個（取代純 argmax），`PICK_STABILITY_SECONDS=300` 窗內
  種子固定不抖。
- **明確結算**（`desire.satisfy/engage/defer/outreach`）：satisfy 的 degree 必填，
  表示缺口真的填了幾成（eff = 1−degree×(1−mult)）；engage=做了但沒滿足，
  defer=現在不做，兩者都不降水位並把該維延後兩小時；outreach=成功送達 Ruby
  的獨立收據，不改水位。同一 wake／同一結果冪等，一次 wake 可留下多個 outcomes。
- **互相制約**：explore/browse/create/chore 的 ACTION_SATISFY 各帶 libido ×0.95。
- **wake 因果鏈**（`POST /api/desire/wake`＋autonomy.ts）：喚醒通知「送達成功」
  後 autonomy 回 POST 一筆 `wake:{drive}`（wake_id 冪等，滾動窗查重）；幽靈喚醒
  （已回滾的）永不進帳。
- **append-only ledger**（`DesireStore._append_ledger`→`buckets/desire_ledger.jsonl`）：
  mutate 時把新事件同步追加；state 只留 MAX_EVENTS 滾動窗，完整因果史在 ledger。
  閉環率統計從手工對帳變一條 grep。
- 頻道側配套（cyan-vps-work）：`fatigueDamp`（0.45 起線性阻尼到 0.72×0.5，硬閘
  不動）進 attemptDesireFire；喚醒 prompt 中疲勞帶體感行（不給數字）。
- 測試：`tests/test_batch1.py`（31 條）；全套 161 pass。頻道 bun 43 pass。

## 0.6 2026-07-12 第零批止血（Non 規格交叉評審後的 no-regret 修復）

對照 Non/Nest 記憶系統規格做的全系統評審（Ruby 交辦、兩個 session 交叉驗證）發現的
實際缺陷，全部修復：

- **merge 語義閘 fail-closed**（`server.py _merge_or_create`）：原本例外時、向量缺失時
  都會放行破壞性合併（雙重 fail-open）。現在破壞性合併必須以「驗證過的」向量相似度
  為前提——引擎關閉、向量缺失、檢查異常一律不合併、改走新建。重複可救，誤併不可逆。
- **搜尋召回：移除向量預篩**（`bucket_manager.search` 原第 1.5 層）：預篩會用向量
  top-50「取代」候選集，精確的名字/標籤命中不在其中就在精排前被丟掉。此語料規模
  全量精排毫秒級；語義召回由 breath 並聯向量通道負責、全文關鍵詞由 BM25 兜底。
  同批啟用 live 的 `matching.bm25_enabled`（預建降落傘，召回缺口證實故拉繩）。
- **dream 空窗也要看見 plan 帳本**（`server.py dream`）：plan 尾段移到「沒有需要消化
  的新記憶」提前返回之前——安靜期反而更不能漏看欠著的事。
- **plan schema**（`server.py plan/trace`，`bucket_manager` 白名單）：新增
  `kind`（promise/task/question/maintenance，默認 task）、`target_drive`（不填按 kind
  自動：promise→miss_ruby、task/maintenance→duty、question→curiosity）、`due_at`、
  `progress`。**執念接線（第一批）將直接讀 target_drive，永不走 domain→drive 映射**——
  否則全掛約定域的技術待辦會被翻譯成「想Ruby」。
- 測試：`tests/test_batch0.py`（merge 閘三路徑＋驗證通過路徑＋釘選保護、搜尋召回回歸、
  dream 空窗 plan 尾段、plan schema 全欄位）。

第一批（代謝）已定案待觀察期收官報告：執念語義收窄（open_loop/plan/自寫閃念才餵）、
高位消退 hysteresis、近高位加權抽選、satisfy degree＋engage 帳本、wake_id 因果鏈。
評審全文見 Ombre 記憶桶 7693b0f6284f。

## 0.5 2026-07-10 上游借鑑批次（借想法不借程式碼，全部自寫）

比照上游 P0luz/Ombre-Brain v2.3.10–v2.5.0 的更新，挑我們要的重新實作：

**正確性**
- embedding 文本 LRU 快取（64 條，`embedding_engine._generate_embedding`）：同一查詢在
  bucket_mgr.search 預篩與 breath 向量通道各嵌一次的重複 API 呼叫直接省掉；失敗不快取。
- 讀取層 datetime 正規化（`bucket_manager._normalize_meta_datetimes`）：手編輯（Obsidian）
  的桶檔時間戳沒加引號時 YAML 會解析成 datetime 物件，排序 TypeError、JSON 序列化爆炸——
  讀取時統一轉 ISO 字串。
- `breath(importance_min=N)` 檔位保留（`utils.select_importance_tiers`）：高分桶塞滿上限時，
  每個 importance 檔位先保一個最近更新的席位，剛降級的桶不再被擠出清單。

**功能**
- `grow(items=[...])` 逐字入庫：調用方帶完整上下文預拆好的定稿逐字保存，只補元數據；
  合併老桶一律原文追加（merge_audit mode="verbatim-append"），不經 LLM 改寫。
- `breath(catalog=True)` 目錄模式：一行一桶元數據、0 次 LLM 呼叫，token 預算內裝多少列多少，
  可配 domain 過濾；feel 不列。
- `dream(window_hours=48)` 時間窗：改按 last_active 開窗（被合併更新的老桶也回到夢裡——
  舊 created 排序的盲點），窗口空退回最近創建；正文截斷 500→1200 字。
- **plan 承諾帳本**（`bucket_type="plan"`，`buckets/plan/`）：`plan()` 工具登記，原文逐字、
  不衰減、不進普通 breath/搜尋/合併/執念，只在 dream 尾端「記掛著的事」按 weight 排序出現；
  `trace(status="active/resolved/abandoned", weight=0~1)` 管生命週期。importance=多重要，
  weight=多重。暫不接 desire 執念（固定分會打滿 duty 加成，日後共同決策）。
- 信件語義檢索：letter 寫入建向量（embeddings.db key=`letter:<id>`，桶搜尋預設排除、
  向量衛生不清），read+query 時子字串優先、語義補位（sim≥0.45，懶回填每次最多 10 封）。
- `hold/trace(why_remembered=...)`：記住的原因，展示欄位不計分，dream 會顯示。

**小件**：`utils.clean_llm_json`（LLM 回覆 JSON 統一清洗，analyze/digest 共用）；
`dehydration.timeout_seconds`/`embedding.timeout_seconds` 可配
（env `OMBRE_COMPRESS_TIMEOUT_SECONDS`/`OMBRE_EMBED_TIMEOUT_SECONDS`）。

**運維**：新增 `deploy_ombre.py`（pytest 門 → 漂移檢查 → 兩段式安裝 → 重啟 → hash 驗證 →
健康檢查 → `.deployed-commit` 部署章）。測試 `tests/test_upstream_inspired.py`。

刻意不跟上游的：src/ 大重構、v3 深內核（WAL/Raft/policy VM）、multi-owner、OAuth、
熱更新、新 dashboard 前端、AI_NAME 中性化。上游 v2.4.0+ 有 noncommercial 聲明——
本批次全部按想法自寫，未複製其程式碼。

### 0.5.1 降落傘批次（同日加場，Ruby 拍板「先摺好」）

兩個原本掛牌的項目改為**預建、測試齊、預設休眠**——實作者在場時把傘摺好，
觸發條件到了只拉繩（改 config），不需要任何人再寫程式碼：

**降落傘 1：本地 embedding fallback**（`embedding.provider: local`）
- fastembed（ONNX）＋ `BAAI/bge-small-zh-v1.5`（512 維中文小模型），懶載入——
  provider=gemini 時完全不碰。模型已預下載至 `/opt/ombre-brain/.fastembed-cache`
  （92MB，備份 tar 已排除；跳傘當下不需要連得上 HuggingFace）。
- `embeddings.db` 遷移為 `(bucket_id, model)` 複合主鍵：各 provider 的向量空間
  並存，切換各寫各的、**切回零成本**（Gemini 向量原地保留）。
- bge 查詢端自動加指令前綴（文檔端不加），LRU 快取鍵含前綴不混撞。
- **拉繩 runbook**：① `config.yaml` 加 `embedding: {provider: local}` →
  ② `systemctl restart ombre-brain` → ③ `cd /opt/ombre-brain &&
  OMBRE_BUCKETS_DIR=buckets .venv/bin/python backfill_embeddings.py`
  （新空間全量重嵌，本地推理免費，約數分鐘）。信件向量由 letter read 的
  懶回填逐次補齊（每次 10 封）。切回：provider 改回 gemini ＋ restart，即刻生效。
- 觸發條件：api_usage_guard 連續報 Gemini 失敗／Google 改免費額度政策。

**降落傘 2：BM25 關鍵詞通道**（`matching.bm25_enabled: true`）
- 自寫 `bm25_index.py`：Okapi BM25，內建斷詞（英數連段＋CJK bigram，索引端補
  unigram 讓單字查詢可命中），純 stdlib 零依賴。
- 接在 search 第 1.6 層：BM25 前 20 名補回候選集（召回保險——domain/向量預篩
  漏掉的強關鍵詞命中撈回來），正規化分數以 max 融入 topic 相關度，不重寫排序。
- 索引按語料指紋（桶數＋id/last_active hash）快取，變動才重建（數百桶 ≈ 數十 ms）。
- **拉繩 runbook**：`config.yaml` 的 `matching:` 加 `bm25_enabled: true` → restart。
- 觸發條件：桶數破 2000／出現真實召回缺口。

---

## 0. 功能总览——这个系统到底做了什么

### 记忆能力

**存储与组织**
- 每条记忆 = 一个 Markdown 文件（YAML frontmatter 存元数据），直接兼容 Obsidian 浏览/编辑
- 四种桶类型：`dynamic`（普通，会衰减）、`permanent`（固化，不衰减）、`feel`（模型感受，不浮现）、`archived`（已遗忘）
- 按主题域分子目录：`dynamic/日常/`、`dynamic/情感/`、`dynamic/编程/` 等
- 钉选桶（pinned）：importance 锁 10，永不衰减/合并，始终浮现为「核心准则」

**每条记忆追踪的元数据**
- `id`（12位短UUID）、`name`（可读名≤80字）、`tags`（10~15个关键词）
- `domain`（1~2个主题域，从 8 大类 30+ 细分域选）
- `valence`（事件效价 0~1）、`arousal`（唤醒度 0~1）、`model_valence`（模型独立感受）
- `importance`（1~10）、`activation_count`（被想起次数）
- `resolved`（已解决/沉底）、`digested`（已消化/写过 feel）、`pinned`（钉选）
- `created`、`last_active` 时间戳

**四种检索模式**
1. **自动浮现**（`breath()` 无参数）：按衰减分排序推送，钉选桶始终展示，Top-1 固定 + Top-20 随机打乱（引入多样性），有 token 预算（默认 10000）
2. **关键词+向量双通道搜索**（`breath(query=...)`）：rapidfuzz 模糊匹配 + Gemini embedding 余弦相似度，合并去重
3. **Feel 独立检索**（`breath(domain="feel")`）：按创建时间倒序返回所有 feel
4. **随机浮现**：搜索结果 <3 条时 40% 概率漂浮 1~3 条低权重旧桶（模拟人类随机联想）

**四维搜索评分**（归一化到 0~100）
- topic_relevance（权重 4.0）：name×3 + domain×2.5 + tags×2 + body
- emotion_resonance（权重 2.0）：Russell 环形模型欧氏距离
- time_proximity（权重 2.5）：`e^(-0.1×days)`
- importance（权重 1.0）：importance/10
- resolved 桶全局降权 ×0.3

**记忆随时间变化**
- **衰减引擎**：改进版艾宾浩斯遗忘曲线
  - 公式：`Score = Importance × activation_count^0.3 × e^(-λ×days) × combined_weight`
  - 短期（≤3天）：时间权重 70% + 情感权重 30%
  - 长期（>3天）：情感权重 70% + 时间权重 30%
  - 新鲜度加成：`1.0 + e^(-t/36h)`，刚存入 ×2.0，~36h 半衰，72h 后 ≈×1.0
  - 高唤醒度(arousal>0.7)且未解决 → ×1.5 紧迫度加成
  - resolved → ×0.05 沉底；resolved+digested → ×0.02 加速淡化
- **自动归档**：score 低于阈值(0.3) → 移入 archive
- **自动结案**：importance≤4 且 >30天 → 自动 resolved
- **永不衰减**：permanent / pinned / protected / feel

**记忆间交互**
- **智能合并**：新记忆与相似桶（score>75）自动 LLM 合并，valence/arousal 取均值，tags/domain 并集
- **时间涟漪**：touch 一个桶时，±48h 内创建的桶 activation_count +0.3（上限 5 桶/次）
- **向量相似网络**：embedding 余弦相似度 >0.5 建边
- **Feel 结晶化**：≥3 条相似 feel（相似度>0.7）→ 提示升级为钉选准则

**情感记忆重构**
- 搜索时若指定 valence，展示层对匹配桶 valence 微调 ±0.1，模拟「当前心情影响回忆色彩」

**模型感受/反思系统**
- **Feel 写入**（`hold(feel=True)`）：存模型第一人称感受，标记源记忆为 digested
- **Dream 做梦**（`dream()`）：返回最近 10 条 + 自省引导 + 连接提示 + 结晶化提示
- **对话启动流程**：breath() → dream() → breath(domain="feel") → 开始对话

**自动化处理**
- 存入时 LLM 自动分析 domain/valence/arousal/tags/name
- 大段日记 LLM 拆分为 2~6 条独立记忆
- 浮现时自动脱水压缩（LLM 压缩保语义，API 不可用时直接报错，无静默降级）
- Wikilink `[[]]` 由 LLM 在内容中标记

---

### 技术能力

**6 个 MCP 工具**

| 工具 | 关键参数 | 功能 |
|---|---|---|
| `breath` | query, max_tokens, domain, valence, arousal, max_results, **importance_min** | 检索/浮现记忆 |
| `hold` | content, tags, importance, pinned, feel, source_bucket, valence, arousal | 存储记忆 |
| `grow` | content | 日记拆分归档 |
| `trace` | bucket_id, name, domain, valence, arousal, importance, tags, resolved, pinned, digested, content, delete | 修改元数据/内容/删除 |
| `pulse` | include_archive | 系统状态 |
| `dream` | （无） | 做梦自省 |

**工具详细行为**

**`breath`** — 三种模式：
- **浮现模式**（无 query）：无参调用，按衰减引擎活跃度排序返回 top 记忆，钉选桶始终展示；冷启动检测（`activation_count==0 && importance>=8`）的桶最多 2 个插入最前，再 Top-1 固定 + Top-20 随机打乱
- **检索模式**（有 query）：关键词 + 向量双通道搜索，四维评分（topic×4 + emotion×2 + time×2.5 + importance×1），阈值过滤
- **Feel 检索**（`domain="feel"`）：特殊通道，按创建时间倒序返回所有 feel 类型桶，不走评分逻辑
- **重要度批量模式**（`importance_min>=1`）：跳过语义搜索，直接筛选 importance≥importance_min 的桶，按 importance 降序，最多 20 条
- 若指定 valence，对匹配桶的 valence 微调 ±0.1（情感记忆重构）

**`hold`** — 两种模式：
- **普通模式**（`feel=False`，默认）：自动 LLM 分析 domain/valence/arousal/tags/name → 向量相似度查重 → 相似度>0.85 则合并到已有桶 → 否则新建 dynamic 桶 → 生成 embedding
- **Feel 模式**（`feel=True`）：跳过 LLM 分析，直接存为 `feel` 类型桶（存入 `feel/` 目录），不参与普通浮现/衰减/合并。若提供 `source_bucket`，标记源记忆为 `digested=True` 并写入 `model_valence`。返回格式：`🫧feel→{bucket_id}`

**`dream`** — 做梦/自省触发器：
- 返回最近 10 条 dynamic 桶摘要 + 自省引导词
- 检测 feel 结晶化：≥3 条相似 feel（embedding 相似度>0.7）→ 提示升级为钉选准则
- 检测未消化记忆：列出 `digested=False` 的桶供模型反思

**`trace`** — 记忆编辑：
- 修改任意元数据字段（name/domain/valence/arousal/importance/tags/resolved/pinned）
- `digested=0/1`：隐藏/取消隐藏记忆（控制是否在 dream 中出现）
- `content="..."`：替换正文内容并重新生成 embedding
- `delete=True`：删除桶文件

**`grow`** — 日记拆分：
- 大段日记文本 → LLM 拆为 2~6 条独立记忆 → 每条走 hold 普通模式流程

**`pulse`** — 系统状态：
- 返回各类型桶数量、衰减引擎状态、未解决/钉选/feel 统计

**REST API（17 个端点）**

| 端点 | 方法 | 功能 |
|---|---|---|
| `/health` | GET | 健康检查 |
| `/breath-hook` | GET | SessionStart 钩子 |
| `/dream-hook` | GET | Dream 钩子 |
| `/dashboard` | GET | Dashboard 页面 |
| `/api/buckets` | GET | 桶列表 🔒 |
| `/api/bucket/{id}` | GET | 桶详情 🔒 |
| `/api/search?q=` | GET | 搜索 🔒 |
| `/api/network` | GET | 向量相似网络 🔒 |
| `/api/breath-debug` | GET | 评分调试 🔒 |
| `/api/config` | GET | 配置查看（key 脱敏）🔒 |
| `/api/config` | POST | 热更新配置 🔒 |
| `/api/status` | GET | 系统状态（版本/桶数/引擎）🔒 |
| `/api/import/upload` | POST | 上传并启动历史对话导入 🔒 |
| `/api/import/status` | GET | 导入进度查询 🔒 |
| `/api/import/pause` | POST | 暂停/继续导入 🔒 |
| `/api/import/patterns` | GET | 导入完成后词频规律检测 🔒 |
| `/api/import/results` | GET | 已导入记忆桶列表 🔒 |
| `/api/import/review` | POST | 批量审阅/批准导入结果 🔒 |
| `/auth/status` | GET | 认证状态（是否需要初始化密码）|
| `/auth/setup` | POST | 首次设置密码 |
| `/auth/login` | POST | 密码登录，颁发 session cookie |
| `/auth/logout` | POST | 注销 session |
| `/auth/change-password` | POST | 修改密码 🔒 |

> 🔒 = 需要 Dashboard 认证（未认证返回 401 JSON）

**Dashboard 认证**
- 密码存储：SHA-256 + 随机 salt，保存于 `{buckets_dir}/.dashboard_auth.json`
- 环境变量 `OMBRE_DASHBOARD_PASSWORD` 设置后，覆盖文件密码（只读，不可通过 Dashboard 修改）
- Session：内存字典（服务重启失效），cookie `ombre_session`（HttpOnly, SameSite=Lax, 7天）
- `/health`, `/breath-hook`, `/dream-hook`, `/mcp*` 路径不受保护（公开）

**Dashboard（6 个 Tab）**
1. 记忆桶列表：6 种过滤器 + 主题域过滤 + 搜索 + 详情面板
2. Breath 模拟：输入参数 → 可视化五步流程 → 四维条形图
3. 记忆网络：Canvas 力导向图（节点=桶，边=相似度）
4. 配置：热更新脱水/embedding/合并参数
5. 导入：历史对话拖拽上传 → 分块处理进度条 → 词频规律分析 → 导入结果审阅
6. 设置：服务状态监控、修改密码、退出登录

**部署选项**
1. 本地 stdio（`python server.py`）
2. Docker + Cloudflare Tunnel（`docker-compose.yml`）
3. Docker Hub 预构建镜像（`docker-compose.user.yml`，`p0luz/ombre-brain`）
4. Render.com 一键部署（`render.yaml`）
5. Zeabur 部署（`zbpack.json`）
6. GitHub Actions 自动构建推送 Docker Hub（`.github/workflows/docker-publish.yml`）

**迁移/批处理工具**：`migrate_to_domains.py`、`reclassify_domains.py`、`reclassify_api.py`、`backfill_embeddings.py`、`write_memory.py`、`check_buckets.py`、`import_memory.py`（历史对话导入引擎）

**降级策略**
- 脱水 API 不可用 → 直接抛 RuntimeError（设计决策，详见 BEHAVIOR_SPEC.md 三、降级行为表）
- 向量搜索不可用 → 纯 fuzzy match
- 逐条错误隔离（grow 中单条失败不影响其他）

**安全**：路径遍历防护（`safe_path()`）、API Key 脱敏、API Key 不持久化到 yaml、输入范围钳制

**监控**：结构化日志、Health 端点、Breath Debug 端点、Dashboard 统计栏、衰减周期日志

---

## 1. 环境变量清单

| 变量名 | 用途 | 必填 | 默认值 / 示例 |
|---|---|---|---|
| `OMBRE_API_KEY` | 脱水/打标/嵌入的 LLM API 密钥，覆盖 `config.yaml` 的 `dehydration.api_key` | 否（无则 API 功能降级到本地） | `""` |
| `OMBRE_BASE_URL` | API base URL，覆盖 `config.yaml` 的 `dehydration.base_url` | 否 | `""` |
| `OMBRE_TRANSPORT` | 传输模式：`stdio` / `sse` / `streamable-http` | 否 | `""` → 回退到 config 或 `"stdio"` |
| `OMBRE_BUCKETS_DIR` | 记忆桶存储目录路径 | 否 | `""` → 回退到 config 或 `./buckets` |
| `OMBRE_HOOK_URL` | SessionStart 钩子调用的服务器 URL | 否 | `"http://localhost:8000"` |
| `OMBRE_HOOK_SKIP` | 设为 `"1"` 跳过 SessionStart 钩子 | 否 | 未设置（不跳过） |
| `OMBRE_DASHBOARD_PASSWORD` | 预设 Dashboard 访问密码；设置后覆盖文件密码，首次访问不弹设置向导 | 否 | `""` |

环境变量优先级：`环境变量 > config.yaml > 硬编码默认值`。所有环境变量在 `utils.py` 中读取并注入 config dict。

---

## 2. 模块结构与依赖关系

```
                    ┌──────────────┐
                    │  server.py   │  MCP 主入口，6 个工具 + Dashboard + Hook
                    └──────┬───────┘
           ┌───────────────┼───────────────┬────────────────┐
           ▼               ▼               ▼                ▼
   bucket_manager.py  dehydrator.py  decay_engine.py  embedding_engine.py
   记忆桶 CRUD+搜索   脱水压缩+打标   遗忘曲线+归档   向量化+语义检索
           │               │                                │
           └───────┬───────┘                                │
                   ▼                                        ▼
              utils.py ◄────────────────────────────────────┘
              配置/日志/ID/路径安全/token估算
```

| 文件 | 职责 | 依赖（项目内） | 被谁调用 |
|---|---|---|---|
| `server.py` | MCP 服务器主入口，注册工具 + Dashboard API + 钩子端点 | `bucket_manager`, `dehydrator`, `decay_engine`, `embedding_engine`, `utils` | `test_tools.py` |
| `bucket_manager.py` | 记忆桶 CRUD、多维索引搜索、wikilink 注入、激活更新 | `utils` | `server.py`, `check_buckets.py`, `backfill_embeddings.py` |
| `decay_engine.py` | 衰减引擎：遗忘曲线计算、自动归档、自动结案 | 无（接收 `bucket_mgr` 实例） | `server.py` |
| `dehydrator.py` | 数据脱水压缩 + 合并 + 自动打标（仅 LLM API，不可用时报 RuntimeError） | `utils` | `server.py` |
| `embedding_engine.py` | 向量化引擎：Gemini embedding API + SQLite + 余弦搜索 | `utils` | `server.py`, `backfill_embeddings.py` |
| `utils.py` | 配置加载、日志、路径安全、ID 生成、token 估算 | 无 | 所有模块 |
| `write_memory.py` | 手动写入记忆 CLI（绕过 MCP） | 无（独立脚本） | 无 |
| `backfill_embeddings.py` | 为存量桶批量生成 embedding | `utils`, `bucket_manager`, `embedding_engine` | 无 |
| `check_buckets.py` | 桶数据完整性检查 | `bucket_manager`, `utils` | 无 |
| `import_memory.py` | 历史对话导入引擎（支持 Claude JSON/ChatGPT/DeepSeek/Markdown/纯文本），分块处理+断点续传+词频分析 | `utils` | `server.py` |
| `reclassify_api.py` | 用 LLM API 重打标未分类桶 | 无（直接用 `openai`） | 无 |
| `reclassify_domains.py` | 基于关键词本地重分类 | 无 | 无 |
| `migrate_to_domains.py` | 平铺桶 → 域子目录迁移 | 无 | 无 |
| `test_smoke.py` | 冒烟测试 | `utils`, `bucket_manager`, `dehydrator`, `decay_engine` | 无 |
| `test_tools.py` | MCP 工具端到端测试 | `utils`, `server`, `bucket_manager` | 无 |

---

## 3. 硬编码值清单

### 3.1 固定分数 / 特殊返回值

| 值 | 位置 | 用途 |
|---|---|---|
| `999.0` | `decay_engine.py` calculate_score | pinned / protected / permanent 桶永不衰减 |
| `50.0` | `decay_engine.py` calculate_score | feel 桶固定活跃度分数 |
| `0.02` | `decay_engine.py` resolved_factor | resolved + digested 时的权重乘数（加速淡化） |
| `0.05` | `decay_engine.py` resolved_factor | 仅 resolved 时的权重乘数（沉底） |
| `1.5` | `decay_engine.py` urgency_boost | arousal > 0.7 且未解决时的紧迫度加成 |

### 3.2 衰减公式参数

| 值 | 位置 | 用途 |
|---|---|---|
| `36.0` | `decay_engine.py` _calc_time_weight | 新鲜度半衰期（小时），`1.0 + e^(-t/36)` |
| `0.3` (指数) | `decay_engine.py` calculate_score | `activation_count ** 0.3`（记忆巩固指数） |
| `3.0` (天) | `decay_engine.py` calculate_score | 短期/长期切换阈值 |
| `0.7 / 0.3` | `decay_engine.py` combined_weight | 短期权重分配：time×0.7 + emotion×0.3 |
| `0.7` | `decay_engine.py` urgency_boost | arousal 紧迫度触发阈值 |
| `4` / `30` (天) | `decay_engine.py` execute_cycle | 自动结案：importance≤4 且 >30天 |

### 3.3 搜索/评分参数

| 值 | 位置 | 用途 |
|---|---|---|
| `×3` / `×2.5` / `×2` | `bucket_manager.py` _calc_topic_score | 桶名 / 域名 / 标签的 topic 评分权重 |
| `1000` (字符) | `bucket_manager.py` _calc_topic_score | 正文截取长度 |
| `0.1` | `bucket_manager.py` _calc_time_score | 时间亲近度衰减系数 `e^(-0.1 × days)` |
| `0.3` | `bucket_manager.py` search_multi | resolved 桶的归一化分数乘数 |
| `0.5` | `server.py` breath/search | 向量搜索相似度下限 |
| `0.7` | `server.py` dream | feel 结晶相似度阈值 |

### 3.4 Token 限制 / 截断

| 值 | 位置 | 用途 |
|---|---|---|
| `10000` | `server.py` breath 默认 max_tokens | 浮现/搜索 token 预算 |
| `20000` | `server.py` breath 上限 | max_tokens 硬上限 |
| `50` / `20` | `server.py` breath | max_results 上限 / 默认值 |
| `3000` | `dehydrator.py` dehydrate | API 脱水内容截断 |
| `2000` | `dehydrator.py` merge | API 合并内容各截断 |
| `5000` | `dehydrator.py` digest | API 日记整理内容截断 |
| `2000` | `embedding_engine.py` | embedding 文本截断 |
| `100` | `dehydrator.py` | 内容 < 100 token 跳过脱水 |

### 3.5 时间/间隔/重试

| 值 | 位置 | 用途 |
|---|---|---|
| `60.0s` | `dehydrator.py` | OpenAI 客户端 timeout |
| `30.0s` | `embedding_engine.py` | Embedding API timeout |
| `60s` | `server.py` keepalive | 保活 ping 间隔 |
| `48.0h` | `bucket_manager.py` touch | 时间涟漪窗口 ±48h |
| `2s` | `backfill_embeddings.py` | 批次间等待 |

### 3.6 随机浮现

| 值 | 位置 | 用途 |
|---|---|---|
| `3` | `server.py` breath search | 结果不足 3 条时触发 |
| `0.4` | `server.py` breath search | 40% 概率触发随机浮现 |
| `2.0` | `server.py` breath search | 随机池：score < 2.0 的低权重桶 |
| `1~3` | `server.py` breath search | 随机浮现数量 |

### 3.7 情感/重构

| 值 | 位置 | 用途 |
|---|---|---|
| `0.2` | `server.py` breath search | 情绪重构偏移系数 `(q_valence - 0.5) × 0.2`（最大 ±0.1） |

### 3.8 其他

| 值 | 位置 | 用途 |
|---|---|---|
| `12` | `utils.py` gen_id | bucket ID 长度（UUID hex[:12]） |
| `80` | `utils.py` sanitize_name | 桶名最大长度 |
| `1.5` / `1.3` | `utils.py` count_tokens_approx | 中文/英文 token 估算系数 |
| `8000` | `server.py` | MCP 服务器端口 |
| `30` 字符 | `server.py` grow | 短内容快速路径阈值 |
| `10` | `server.py` dream | 取最近 N 个桶 |

---

## 4. Config.yaml 完整键表

| 键路径 | 默认值 | 用途 |
|---|---|---|
| `transport` | `"stdio"` | 传输模式 |
| `log_level` | `"INFO"` | 日志级别 |
| `buckets_dir` | `"./buckets"` | 记忆桶目录 |
| `merge_threshold` | `75` | 合并相似度阈值 (0-100) |
| `dehydration.model` | `"deepseek-chat"` | 脱水用 LLM 模型 |
| `dehydration.base_url` | `"https://api.deepseek.com/v1"` | API 地址 |
| `dehydration.api_key` | `""` | API 密钥 |
| `dehydration.max_tokens` | `1024` | 脱水返回 token 上限 |
| `dehydration.temperature` | `0.1` | 脱水温度 |
| `embedding.enabled` | `true` | 启用向量检索 |
| `embedding.model` | `"gemini-embedding-001"` | Embedding 模型 |
| `decay.lambda` | `0.05` | 衰减速率 λ |
| `decay.threshold` | `0.3` | 归档分数阈值 |
| `decay.check_interval_hours` | `24` | 衰减扫描间隔（小时） |
| `decay.emotion_weights.base` | `1.0` | 情感权重基值 |
| `decay.emotion_weights.arousal_boost` | `0.8` | 唤醒度加成系数 |
| `matching.fuzzy_threshold` | `50` | 模糊匹配下限 |
| `matching.max_results` | `5` | 匹配返回上限 |
| `scoring_weights.topic_relevance` | `4.0` | 主题评分权重 |
| `scoring_weights.emotion_resonance` | `2.0` | 情感评分权重 |
| `scoring_weights.time_proximity` | `2.5` | 时间评分权重 |
| `scoring_weights.importance` | `1.0` | 重要性评分权重 |
| `scoring_weights.content_weight` | `3.0` | 正文评分权重 |
| `wikilink.enabled` | `true` | 启用 wikilink 注入 |
| `wikilink.use_tags` | `false` | wikilink 包含标签 |
| `wikilink.use_domain` | `true` | wikilink 包含域名 |
| `wikilink.use_auto_keywords` | `true` | wikilink 自动关键词 |
| `wikilink.auto_top_k` | `8` | wikilink 取 Top-K 关键词 |
| `wikilink.min_keyword_len` | `2` | wikilink 最短关键词长度 |
| `wikilink.exclude_keywords` | `[]` | wikilink 排除关键词表 |

---

## 5. 核心设计决策记录

### 5.1 为什么用 Markdown + YAML frontmatter 而不是数据库？

**决策**：每个记忆桶 = 一个 `.md` 文件，元数据在 YAML frontmatter 里。

**理由**：
- 与 Obsidian 原生兼容——用户可以直接在 Obsidian 里浏览、编辑、搜索记忆
- 文件系统即数据库，天然支持 git 版本管理
- 无外部数据库依赖，部署简单
- wikilink 注入让记忆之间自动形成知识图谱

**放弃方案**：SQLite/PostgreSQL 全量存储。过于笨重，失去 Obsidian 可视化优势。

### 5.2 为什么 embedding 单独存 SQLite 而不放 frontmatter？

**决策**：向量存 `embeddings.db`（SQLite），与 Markdown 文件分离。

**理由**：
- 3072 维浮点向量无法合理存入 YAML frontmatter
- SQLite 支持批量查询和余弦相似度计算
- embedding 是派生数据，丢失可重新生成（`backfill_embeddings.py`）
- 不污染 Obsidian 可读性

### 5.3 为什么搜索用双通道（关键词 + 向量）而不是纯向量？

**决策**：关键词模糊匹配（rapidfuzz）+ 向量语义检索并联，结果去重合并。

**理由**：
- 纯向量在精确名词匹配上表现差（"2024年3月"这类精确信息）
- 纯关键词无法处理语义近似（"很累" → "身体不适"）
- 双通道互补，关键词保精确性，向量补语义召回
- 向量不可用时自动降级到纯关键词模式

### 5.4 为什么有 dehydration（脱水）这一层？

**决策**：存入前先用 LLM 压缩内容（保留信息密度，去除冗余表达）。API 不可用时直接抛出 `RuntimeError`，不静默降级。

**理由**：
- MCP 上下文有 token 限制，原始对话冗长，需要压缩
- LLM 压缩能保留语义和情感色彩，纯截断会丢信息
- 本地关键词提取质量不足以替代语义打标与合并，静默降级会产生错误分类记忆，比报错更危险。详见 BEHAVIOR_SPEC.md 三、降级行为表。

**放弃方案**：只做截断。信息损失太大。

### 5.5 为什么 feel 和普通记忆分开？

**决策**：`feel=True` 的记忆存入独立 `feel/` 目录，不参与普通浮现、不衰减、不合并。

**理由**：
- feel 是模型的自省产物，不是事件记录——两者逻辑完全不同
- 事件记忆应该衰减遗忘，但"我从中学到了什么"不应该被遗忘
- feel 的 valence 是模型自身感受（不等于事件情绪），混在一起会污染情感检索
- feel 可以通过 `breath(domain="feel")` 单独读取

### 5.6 为什么 resolved 不删除记忆？

**决策**：`resolved=True` 让记忆"沉底"（权重 ×0.05），但保留在文件系统中，关键词搜索仍可触发。

**理由**：
- 模拟人类记忆：resolved 的事不会主动想起，但别人提到时能回忆
- 删除是不可逆的，沉底可随时 `resolved=False` 重新激活
- `resolved + digested` 进一步降权到 ×0.02（已消化 = 更释然）

**放弃方案**：直接删除。不可逆，且与人类记忆模型不符。

### 5.7 为什么用分段式短期/长期权重？

**决策**：≤3 天时间权重占 70%，>3 天情感权重占 70%。

**理由**：
- 刚发生的事主要靠"新鲜"驱动浮现（今天的事 > 昨天的事）
- 时间久了，决定记忆存活的是情感强度（强烈的记忆更难忘）
- 这比单一衰减曲线更符合人类记忆的双重存储理论

### 5.8 为什么 dream 设计成对话开头自动执行？

**决策**：每次新对话启动时，Claude 执行 `dream()` 消化最近记忆，有沉淀写 feel，能放下的 resolve。

**理由**：
- 模拟睡眠中的记忆整理——人在睡觉时大脑会重放和整理白天的经历
- 让 Claude 对过去的记忆有"第一人称视角"的自省，而不是冷冰冰地搬运数据
- 自动触发确保每次对话都"接续"上一次，而非从零开始

### 5.9 为什么新鲜度用连续指数衰减而不是分段阶梯？

**决策**：`bonus = 1.0 + e^(-t/36)`，t 为小时，36h 半衰。

**理由**：
- 分段阶梯（0-1天=1.0，第2天=0.9...）有不自然的跳变
- 连续指数更符合遗忘曲线的物理模型
- 36h 半衰期使新桶在前两天有明显优势，72h 后接近自然回归
- 值域 1.0~2.0 保证老记忆不被惩罚（×1.0），只是新记忆有额外加成（×2.0）

**放弃方案**：分段线性（原实现）。跳变点不自然，参数多且不直观。

### 5.10 情感记忆重构（±0.1 偏移）的设计动机

**决策**：搜索时如果指定了 `valence`，会微调结果桶的 valence 展示值 `(q_valence - 0.5) × 0.2`。

**理由**：
- 模拟认知心理学中的"心境一致性效应"——当前心情会影响对过去的回忆
- 偏移量很小（最大 ±0.1），不会扭曲事实，只是微妙的"色彩"调整
- 原始 valence 不被修改，只影响展示层

---

## 6. 目录结构约定

```
buckets/
├── permanent/       # pinned/protected 桶，importance=10，永不衰减
├── dynamic/
│   ├── 日常/        # domain 子目录
│   ├── 情感/
│   ├── 自省/
│   ├── 数字/
│   └── ...
├── archive/         # 衰减归档桶
└── feel/            # 模型自省 feel 桶
```

桶文件格式：
```markdown
---
id: 76237984fa5d
name: 桶名
domain: [日常, 情感]
tags: [关键词1, 关键词2]
importance: 5
valence: 0.6
arousal: 0.4
activation_count: 3
resolved: false
pinned: false
digested: false
created: 2026-04-17T10:00:00+08:00
last_active: 2026-04-17T14:00:00+08:00
type: dynamic
---

桶正文内容...
```

---

## 7. Bug 修复记录 (B-01 至 B-10)

### B-01 — `update(resolved=True)` 自动归档 🔴 高

- **文件**: `bucket_manager.py` → `update()`
- **问题**: `resolved=True` 时立即调用 `_move_bucket(archive_dir)` 将桶移入 `archive/`
- **修复**: 移除 `_move_bucket` 逻辑；resolved 桶留在 `dynamic/`，由 decay 引擎自然淘汰
- **影响**: 已解决的桶仍可被关键词检索命中（降权但不消失）
- **测试**: `tests/regression/test_issue_B01.py`，`tests/integration/test_scenario_07_trace.py`

### B-03 — `int()` 截断浮点 activation_count 🔴 高

- **文件**: `decay_engine.py` → `calculate_score()`
- **问题**: `max(1, int(activation_count))` 将 `_time_ripple` 写入的 1.3 截断为 1，涟漪加成失效
- **修复**: 改为 `max(1.0, float(activation_count))`
- **影响**: 时间涟漪效果现在正确反映在 score 上；高频访问的桶衰减更慢
- **测试**: `tests/regression/test_issue_B03.py`，`tests/unit/test_calculate_score.py`

### B-04 — `create()` 初始化 activation_count=1 🟠 中

- **文件**: `bucket_manager.py` → `create()`
- **问题**: `activation_count=1` 导致冷启动检测条件 `== 0` 永不满足，新建重要桶无法浮现
- **修复**: 改为 `activation_count=0`；`touch()` 首次命中后变 1
- **测试**: `tests/regression/test_issue_B04.py`，`tests/integration/test_scenario_01_cold_start.py`

### B-05 — 时间衰减系数 0.1 过快 🟠 中

- **文件**: `bucket_manager.py` → `_calc_time_score()`
- **问题**: `math.exp(-0.1 * days)` 导致 30 天后得分仅剩 ≈0.05，远快于人类记忆曲线
- **修复**: 改为 `math.exp(-0.02 * days)`（30 天后 ≈0.549）
- **影响**: 记忆保留时间更符合人类认知模型
- **测试**: `tests/regression/test_issue_B05.py`，`tests/unit/test_score_components.py`

### B-06 — `w_time` 默认值 2.5 过高 🟠 中

- **文件**: `bucket_manager.py` → `_calc_final_score()`（或评分调用处）
- **问题**: `scoring.get("time_proximity", 2.5)` — 时间权重过高，近期低质量记忆得分高于高质量旧记忆
- **修复**: 改为 `scoring.get("time_proximity", 1.5)`
- **测试**: `tests/regression/test_issue_B06.py`，`tests/unit/test_score_components.py`

### B-07 — `content_weight` 默认值 3.0 过高 🟠 中

- **文件**: `bucket_manager.py` → `_calc_topic_score()`
- **问题**: `scoring.get("content_weight", 3.0)` — 内容权重远大于名字权重(×3)，导致内容重复堆砌的桶得分高于名字精确匹配的桶
- **修复**: 改为 `scoring.get("content_weight", 1.0)`
- **影响**: 名字完全匹配 > 标签匹配 > 内容匹配的得分层级现在正确
- **测试**: `tests/regression/test_issue_B07.py`，`tests/unit/test_topic_score.py`

### B-08 — `run_decay_cycle()` 同轮 auto_resolve 后 score 未降权 🟡 低

- **文件**: `decay_engine.py` → `run_decay_cycle()`
- **问题**: `auto_resolve` 标记后立即用旧 `meta`（stale）计算 score，`resolved_factor=0.05` 未生效
- **修复**: 在 `bucket_mgr.update(resolved=True)` 后立即执行 `meta["resolved"] = True`，确保同轮降权
- **测试**: `tests/regression/test_issue_B08.py`，`tests/integration/test_scenario_08_decay.py`

### B-09 — `hold()` 用 analyze() 覆盖用户传入的 valence/arousal 🟡 低

- **文件**: `server.py` → `hold()`
- **问题**: 先调 `analyze()`，再直接用结果覆盖用户传入的情感值，情感准确性丢失
- **修复**: 使用 `final_valence = user_valence if user_valence is not None else analyze_result.get("valence")`
- **影响**: 用户明确传入的情感坐标（包括 0.0）不再被 LLM 结果覆盖
- **测试**: `tests/regression/test_issue_B09.py`，`tests/integration/test_scenario_03_hold.py`

### B-10 — feel 桶 `domain=[]` 被填充为 `["未分类"]` 🟡 低

- **文件**: `bucket_manager.py` → `create()`
- **问题**: `if not domain: domain = ["未分类"]` 对所有桶类型生效，feel 桶的空 domain 被错误填充
- **修复**: 改为 `if not domain and bucket_type != "feel": domain = ["未分类"]`
- **影响**: `breath(domain="feel")` 通道过滤逻辑现在正确（feel 桶 domain 始终为空列表）
- **测试**: `tests/regression/test_issue_B10.py`，`tests/integration/test_scenario_10_feel.py`

---

### Bug 修复汇总表

| ID | 严重度 | 文件 | 方法 | 一句话描述 |
|---|---|---|---|---|
| B-01 | 🔴 高 | `bucket_manager.py` | `update()` | resolved 桶不再自动归档 |
| B-03 | 🔴 高 | `decay_engine.py` | `calculate_score()` | float activation_count 不被 int() 截断 |
| B-04 | 🟠 中 | `bucket_manager.py` | `create()` | 初始 activation_count=0 |
| B-05 | 🟠 中 | `bucket_manager.py` | `_calc_time_score()` | 时间衰减系数 0.02（原 0.1） |
| B-06 | 🟠 中 | `bucket_manager.py` | 评分权重配置 | w_time 默认 1.5（原 2.5） |
| B-07 | 🟠 中 | `bucket_manager.py` | `_calc_topic_score()` | content_weight 默认 1.0（原 3.0） |
| B-08 | 🟡 低 | `decay_engine.py` | `run_decay_cycle()` | auto_resolve 同轮应用 ×0.05 |
| B-09 | 🟡 低 | `server.py` | `hold()` | 用户 valence/arousal 优先 |
| B-10 | 🟡 低 | `bucket_manager.py` | `create()` | feel 桶 domain=[] 不被填充 |
