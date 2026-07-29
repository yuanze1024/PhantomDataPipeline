# PhantomData

把 **Phantom-Data** 的标注对齐到百度 BOS 上的 **Koala** 源视频，产出与 UltraVid57k 同构的
box+ref 训练集。全程从 BOS 在线读帧，**不下载源视频到本地**。

Phantom-Data（ICLR 2026，subject-consistent video generation）建立在 Koala-36M 上，只发布标注、
不发布 mp4。本包解析那些标注、切出 81 帧窗口、修正 subject 框、跑 SAM2 masklet，最后建索引。

## 环境

需要三样东西（路径均可用环境变量覆盖）：

- Phantom 两个 parquet，放在 `$PHANTOM_DATA_DIR`（默认
  `/mnt/pfs/users/yuanze/datasets/phantom_data_koala36m`）：
  - `koala36M_multi_ref_merged_filtered.parquet` — 标注表
  - `koala36M_multi_ref_meta_info_merged.parquet` — `vid` → `youtube_url` + `timestamp`
- BOS video-id→后缀映射 CSV `$PHANTOM_AUDIT_CSV`（默认
  `/mnt/pfs/share/pengchunli/koala36m_thordata_audit/thordata_bos_videos.csv`）
- BOS 凭证 `$PHANTOM_BOS_AKSK`（默认 `<repo-root>/BOS_AKSK`，行 1 = AK，行 2 = SK）

安装与测试：

```bash
pip install -e 'third_party/PhantomData[build]'
pytest third_party/PhantomData/tests        # 纯函数测试，无网络无 GPU
```

模型权重全部从本地加载：Grounding DINO 和 SAM2 走 `HF_HOME`，ID-Sim 走
`/mnt/pfs/users/yuanze/models/id_sim_checkpoint`。SAM2 与 `ultravid_pipeline` 从源码目录经
`PYTHONPATH` 引入，不从 index 安装。

## 管线

```
stage A  plan      parquet 行  → sample spec（选窗口、定 seed 帧、解析 ref 指针）
stage B  extract   spec        → 81 帧 mp4 + 原始 ref 帧 jpg + extracted.jsonl
stage 2' box       extracted   → gate_report.json（候选池 + ID-Sim 排序后的修正框）
stage 3  gate      report      → gated.jsonl（过门的 subject，框已替换）
stage C  segment   gated       → SAM2 masklet + ref 抠图 + segmented.jsonl
stage D  index     segmented   → 质量漏斗 + 去重 + train/eval split
```

### 跑法（在 pod 里，uid 1010）

```bash
export PYTHONPATH=<repo>/third_party/PhantomData/src:<repo>/third_party/UltraVidPipeline/src:<sam2>
export HF_HOME=/mnt/pfs/share/pretrained_model/.cache/huggingface
export NO_PROXY="192.168.0.0/16,10.0.0.0/8,127.0.0.1,localhost"
unset http_proxy https_proxy all_proxy      # BOS 出网要绕代理

D=/mnt/pfs/data/yuanze/phantom_koala_v1

# A: 选样本
python -m phantom_data.build.plan --num-sources 100 --out $D/specs.jsonl

# B: 切 clip + 取 ref 帧
python -m phantom_data.build.extract --specs $D/specs.jsonl --dataset $D --workers 4

# 2': 修正框（GPU）
python tools/tighten_run.py --dataset $D --out-root _box --candidates

# 3: 过门，写 stage C 的输入
python tools/gate_apply.py --dataset $D --out-root _box \
  --rule identity_only --identity-min 0.2 --output gated.jsonl

# C: masklet（GPU）
python -m phantom_data.build.segment --dataset $D --input gated.jsonl

# D: 建索引（关掉 CLIP 门，见下）
python -m phantom_data.build.index --dataset $D --output-name phantom_v1 \
  --min-ref-clip-score 0
```

每一步都可恢复：已有 passed marker 的 sample 跳过，failed 的重试。加 `--force` 忽略 marker。
`--limit N` 只跑前 N 条，适合 smoke。

### stage 2'：框是怎么修正的

Phantom 的框经常偏移，有时框错物体。这一步不信任它的几何，只信任它的语义指向：

1. **提名** — Grounding DINO 用 subject 名词做 query 出候选框，加上 Phantom 自己的框
2. **收紧** — 每个候选各跑一次 SAM2 单帧分割，取 mask 的 tight box，使候选可比
3. **裁决** — ID-Sim 给 ref 侧和 target 侧的候选两两配对打分，选出最像同一个体的那一对

无 caption、无 CLIP 打分。query 只取名词短语（`candidates.subject_noun`）：检测器的置信度是
**对文本 token 取 max**，长 query 会让 `glasses`、`hair` 这类词各自赢下一个部件框。面积下限
（`candidates.plausible_instance`）是第二道防线。

`--detector-top-k` 控制每侧候选数（默认 3）。`--no-idsim` 只出框不打分，用于快速查几何。

报告里每个 subject 都记着完整证据：`ref_pool` / `seed_pool` 是全部候选，`pairs` 是所有配对分数，
`ranking.margin` 是第一名与第二名的差距，`ranking.used_detector` 说明赢家是否来自检测器。
margin 很小意味着两个候选难分，值得人看。

### stage 3：门

只有一条规则 `identity_only`：**ref 和 target 是否同一个体**。框来自 SAM2 分割候选指向的物体，
所以"框有没有套对物体"由构造保证，不需要额外判据确认。

`--identity-min` 是 ID-Sim 相似度下限（1 − distance）。`--clip-min` / `--iou-min` /
`--iou-floor` 仍被接受并记入 provenance，但不参与判定。

`--keep-all` 不过门、全部输出（行上仍记 verdict），用于在过滤前先看 mask。

## 产出

```
<root>/clips/<sample_id>.mp4                     81 帧，fps 16
<root>/ref_frames/<sample_id>_subj<NN>.jpg       原始 ref 帧（未抠图）
<root>/extracted.jsonl                           stage B 产物
<root>/_box/gate_report.json                     候选、配对分数、选中的框
<root>/gated.jsonl                               stage C 的输入
<root>/masklets/<sample_id>.npz                  81 帧 mask，按宽度位压缩
<root>/object_reference/<sample_id>_subj<NN>.jpg 白底抠图
<root>/segmented.jsonl                           stage C 产物
<root>/indexes/<name>/                           stage D：metadata_train.csv 等
```

masklet npz 的字段：`subject_masks_packed` (subjects, frames, H, ceil(W/8))、`mask_width`、
`source_subject_ids`。解包：`np.unpackbits(packed, axis=-1)[..., :mask_width]`。

`extracted.jsonl` 每行的 subject 里，`ref.frame` 是**原始 ref 帧**（待抠图）。故意不叫
`object_reference`：在 UltraVid schema 里那个键专指白底抠图，是 stage C 的产出。

stage D 的 `indexes/<name>/` 与 UltraVid 的 `indexes/bboxref_clean_dedup_iou50` 同构，判定逻辑
全部 import `ultravid_pipeline`（`stages.quality.filter_sample`、`stages.index.deduplicate_sample`
等），本模块只做编排，所以两个数据集永远同一口径。任何阈值偏离都会写进
`funnel.json.threshold_deltas` 并在该目录的 README 顶部警告。train/eval 按 `video_id` hash 切分，
source-disjoint 且代码里 assert 无交集。

### `--min-ref-clip-score 0`：为什么关掉 CLIP 门

UltraVid 的默认值 0.23 是一道 CLIP 门：拿 stage C 顺手算的 `ref_clip_score`（白底抠图 vs
Phantom 的短语）当质量判据，低于阈值就丢。**这条链故意关掉它。**

理由是这个判据的失效方向恰好就是它被使用的方向。CLIP 分高确实说明抠图内容对得上短语，但**分低
不说明样本坏** —— 短名词短语（"woman"、"cat"）的分天然低于 UltraVid 的 VLM 长句（中位 0.267 vs
0.279），而且裸 CLIP 余弦没有校准，量纲随文本长度漂移。同一个理由已经让 CLIP 从 stage 2' 的出框
判据里退场（见 :mod:`phantom_data.redetect`）；在 stage D 保留它等于在链路末端重新引入一个已经
被否决的判据。

实测差异：140 subject 的 pilot 上，0.23 丢 21/135（16%），关掉后 135/135 全过，
`quality_rejection` 全空 —— 也就是说那 21 条**全部**是这道门丢的，没有别的原因。

`ref_clip_score` 字段仍然照算照写：UltraVid 的 CSV 有它，schema 要兼容。**算而不拦。**

身份质控由 stage 3 的 ID-Sim 承担（同一/不同个体分离 AUROC 0.998），几何质控由 mask 派生 tight
box 的构造保证。其余三个 UltraVid 阈值（`min_visible_frames`、`max_mask_area_q75`、
`dedup_max_mask_iou`）保持默认，它们量的是 mask 本身而非图文一致性。

代价：与 UltraVid 混用时两边的质控口径不同。这会写进 `threshold_deltas`，不会静默。

落地经过 `storage.py` 的 backend 接口（当前只有 `local`）。规模化后 clip 要存 BOS，换 backend
即可，调用方不用改。

## 查看结果

```bash
bash tools/run_masklet_viewer.sh      # 默认 :8512
```

逐个 subject 显示 81 帧里的 mask 叠加（半透明蓝灰 + 白描边）和 mask 派生的框，下面是逐帧面积。
判断跟踪质量看三个数：面积远低于中位说明 mask 部分溶解，远高于说明漏到了别的物体上，
`interior_gaps` 非空说明跟踪断过又重新捕获 —— 这三种失败在单帧截图上都看不出来。

环境变量：`PHANTOM_MASKLET_DATASET`、`PHANTOM_MASKLET_MANIFEST`、`PHANTOM_MASKLET_VIEWER_PORT`。

## 两个坑

**`box_space`：最危险的一个字段。** `gate_report.json` 里的 `chosen_box_*` 已经是真实帧像素坐标，
而 `extracted.jsonl` 的 `seed_bbox_768` 是原始标注坐标、stage C 会过 canvas 映射。如果 stage C
把修正后的框**再映射一次**，在 1920×1080 上就是再乘 2.5 —— **不报错，只是把每个框放到别处去**。

所以 `gate_apply.py` 给每行打 `box_space: "frame"`，stage C 按它分派（`segment.resolve_box`）：

| `box_space` | stage C 怎么做 |
|---|---|
| 缺失 / `"annotation"` | 过 canvas 映射 + clamp |
| `"frame"` | **只 clamp，不映射** |

**decord 会在坏源上永久 wedge。** 实测一条 .mkv 让 worker 卡死 65 分钟、RSS 涨到 38 GB，最后把整个
pod OOM 掉（158/159 条已完成却全丢）。线程阻塞在 C 里，Python 层的 `cancel()` 和信号都进不去。
两道防线：`--sample-timeout`（默认 300 s，健康样本约 5 s）到点放弃该样本、记 failed marker、继续跑；
每条完成立刻 append+fsync 到 `extract.partial.jsonl`，收尾折进 `extracted.jsonl`。解码走 decord
直读 BOS presigned URL，**不要**用 ffmpeg 命令行读 URL（静态 ffmpeg 在 https presigned URL 上
segfault）。每 worker 一个独立 `FrameGrabber`（decord reader 非线程安全）。

## bbox 坐标系：y 轴已定，x 轴不可恢复

Phantom 的 bbox 是长边 768 等比坐标。全表 651,031 行 / 1,345,848 个框的标定结论：

- **y 轴严格服从**长边 = 768 等比画布，每个宽高比桶都精确贴合（16:9→432、4:3→576、1:1→768），
  99.25% 的框装得进去。
- **x 轴不服从**：16:9 有 **14.4%** 的框 `x2` 超出拟合宽度（4:3 17-22%、1:1 29%），clamp 值卡在
  768 / 798 / 800 / 832 四个离散点上。已证伪"多画布混合"和"单个各向异性画布"两种解释，且
  **无法从 (W, H) 恢复** —— 标注表只有 3 列，没有可供切分的 provenance。

实用做法是按 `x2 > 拟合宽度` 过滤掉那部分，而不是去猜一个全局 scale，这样保住 y 轴的精确性。
若不过滤而沿用现有映射，那些框会横向出帧、被 stage C clamp 到右边缘。

`canvas.py` 保留了几个候选坐标系，`resolve_box` 默认按长边 768 映射后 clamp。
