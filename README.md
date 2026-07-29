# PhantomData

在线核对 **Phantom-Data** 标注与存放在百度 BOS 上的 **Koala** 源视频。全程从 BOS 在线读帧，**不下载视频到本地**。

Phantom-Data（ICLR2026，subject-consistent video generation）建立在 Koala-36M 上，只发布标注、不发布 mp4。本包把它的标注对齐到已有的 BOS Koala 视频，提供一个分页浏览器逐个查看样本，验证对齐正确性。

## 数据与凭证

需要三样东西（路径均可用环境变量覆盖）：

- Phantom 两个 parquet，放在 `$PHANTOM_DATA_DIR`（默认 `/mnt/pfs/users/yuanze/datasets/phantom_data_koala36m`）：
  - `koala36M_multi_ref_merged_filtered.parquet` — 标注表（`video_id` / `video_caption` / `cross_pair`）
  - `koala36M_multi_ref_meta_info_merged.parquet` — 桥梁（`vid` → `youtube_url` + `timestamp`）
- BOS video-id→后缀映射 CSV `$PHANTOM_AUDIT_CSV`（默认 `/mnt/pfs/share/pengchunli/koala36m_thordata_audit/thordata_bos_videos.csv`）
- BOS 凭证 `$PHANTOM_BOS_AKSK`（默认 `<repo-root>/BOS_AKSK`，行1=AK，行2=SK）。region=bj，endpoint=bj.bcebos.com。

## 对齐链路

```
filtered.video_id = <koala_uuid>_<start>_<end>
  → meta_info[vid].youtube_url  → 11 位 youtube id
  → audit_csv[youtube_id].ext
  → BOS: bucket=external-data, key=Thordata/<yt>/<yt><ext>   (完整源视频)
帧时间 = start + float(frame_idx)*(end-start)
bbox 坐标系 = 长边 768 等比（y 轴已验证；x 轴有 14% 的框溢出且不可恢复，见下）
```

`cross_pair` 里的参考图取自**同源视频的另一时段 clip**，其 vid 同样能在 meta_info 里解析。源视频在 BOS 的覆盖率约 **97.6%**。

### bbox 坐标系：**y 轴已定，x 轴不可恢复**

标定于 2026-07-26 完成：全表 651,031 行 / 1,345,848 个框，三次流式扫描（pod 内各 10-20 秒）。
结论是历史假设"长边缩放到 768"**在 y 轴上完全正确、在 x 轴上部分失效**。

| 桶（源 W×H 举例） | 768 长边画布 | max x2 | max y2 | x2 > 拟合宽度 | y2 > 拟合高度 |
|---|---|---|---|---|---|
| 16:9（1920×1080 / 1280×720） | 768×432 | 928 | 507 | **14.4%** | 0.7% |
| 4:3（1440×1080 / 960×720） | 768×576 | 883 | 574 | **17-22%** | 0.2% |
| 1:1（1080×1080） | 768×768 | 1109 | 851 | **29%** | 1.3% |
| 2.22:1（2400×1080） | 768×345.6 | 768 | 366 | **0%** | 0% |

**y 轴严格服从长边=768 等比画布**：每个宽高比桶都精确贴合（16:9→432、4:3→576、1:1→768、
2.22:1→345.6），99.25% 的框装得进去；16:9 有 **33.5%** 的框 `y2` 恰好等于 432，即贴住画布底边。

**x 轴不服从**，且溢出是 x 轴专有的：16:9 卡在 **768 / 798 / 800 / 832**（`x2` p99 = 832，
32,816 个 target 框精确命中），4:3 卡在 **806.4**（= 768×1.05，取整落在 805/806/807）。

两个曾经的猜测都已证伪：
- **不是"多个等比画布混合"**：`x2 == 832` 的框里有 **51.7%** 的 `y2` 恰好等于 432 —— 而 432 是
  **768** 画布的高度；若这些框来自 832 长边画布（832×468），432 就毫无特殊性，堆应该在 468。
  `x2==768` 与 `x2==832` 两组共享同一堵 y 墙 ⇒ x 与 y 是**独立** clamp，不是联动的画布切换。
- **不是"单个各向异性画布"**：x 有四个 clamp，单画布只会产生一个宽度。
- **也不由源分辨率决定**：1920×1080 与 1280×720 的四个 clamp 集合与占比在 2-3 个百分点内重合
  （832 各占其 overflow 的 36.7% / 31.2%），不存在"某分辨率走 832、另一个走 768"的分工。
  4:3 的 806.4 同样属于整个桶（1440×1080 / 960×720 / 1600×1200 都有）。**x 轴无法从 (W,H) 恢复**，
  重分桶依据在这份数据里不可见；标注表 A 只有 3 列，没有 provenance 列可供切分。

**实用路线**：按 `x2 > 拟合宽度` 过滤掉那 14%（4:3 17-22%、1:1 29%），而不是去猜一个全局 scale
—— 这保住 y 轴的精确性。若不过滤而沿用现有映射，那部分框会横向出帧、被 stage C clamp 到右边缘
（pilot 数据里已观察到 5 个 ref 贴右边缘、22 个贴下边缘）。另需注意"离散 clamp"只解释 80-85% 的
溢出质量：16:9 的溢出摊在 69-81 个不同 `x2` 值上，尾巴延伸到 928（个别 1184）；y 侧 432 是墙但
非硬顶（p99.9 = 465，max 507）。

证据在 `/mnt/pfs/users/yuanze/datasets/phantom_canvas_calib_v1/`：`canvas_estimator.json`（分桶
直方图/百分位）、`long_edge_probe.json`（`L_needed` 与条件 y2 分布）、`xclamp_by_resolution.json`
（逐分辨率 clamp 归属）。**注意 `calib/estimate.py` 的相对 spike 规则
（`count(v) > 5×median(v±4)`）打不着主 clamp** —— 768/832 的邻域中位数被自身抬高，报告里的
"top spikes"是尾部噪声；**读 `x2_hist_top` 原始直方图，别读 spikes 列表**。
同理 `long_edge_probe` 的 per-video span 一致性检验（item 4）无推断力：`L_needed` 是每框下界，
span 大只反映物体尺寸差异。

候选假设集中在 `src/phantom_data/canvas.py`（`HYPOTHESES` 注册表，支持各向异性 sx≠sy）。
带 `_768` 后缀的字段名（`seed_bbox_768` / `bbox_768`）是**误名**，仅为已建数据集的落盘兼容
而保留，含义只是"原始标注坐标"。

## 用法

```bash
pip install -e 'third_party/PhantomData[viewer]'
pytest third_party/PhantomData/tests            # 纯函数测试，无网络
bash third_party/PhantomData/tools/run_viewer.sh  # 起分页浏览器（默认 :8503）
```

浏览器每页展示一个样本：目标帧（绿框）+ 参考帧（青框）+ 名词短语/caption + 原始 `cross_pair` 全字段。

> 注意：构建 `vid → 元数据` 映射会扫 110 万行 parquet（pyarrow 默认多线程）。按项目约定，重扫描/批处理应在 k8s pod 中运行，不在开发登录机上。

## build 管线（`phantom_data.build`）

把标注 + BOS 源视频做成与 UltraVid57k 同构的 box+ref 训练集。

```
stage A  plan     parquet 行  → sample spec（选窗口、定 seed 帧、解析 ref 指针）
stage B  extract  spec       → 81 帧 mp4 + 原始 ref 帧 jpg + extracted.jsonl
   ↓  bbox 修正管线（enrich → redetect → gate_apply，见下节）→ gated.jsonl
stage C  segment  ref 帧抠图 + SAM2 masklet + bbox json → segmented.jsonl
stage D  index    质量漏斗 + 去重 + train/eval split → indexes/<name>/
```

**bbox 修正插在 B 和 C 之间**（2026-07-29 改）。SAM2 只对过了身份门的样本跑，且用修正后的框抠图。
stage C 的输入契约因此有两个：`extracted.jsonl`（原始标注坐标）或 `gated.jsonl`（已是真实帧坐标），
靠行上的 `box_space` 字段区分，见下节"坐标系是最危险的一个字段"。

### 窗口契约

- 目标窗口固定 **81 帧 @ 16fps = 5.0625 s**，保源分辨率（不 resize）。
- clip 时长 < 5.0625 s 的行直接丢弃（`clip_too_short`，实测占 ~25%）。
- 窗口起点 `w0` 必须夹在 clip 内：`clip_start <= w0` 且 `w0 + 5.0625 <= clip_end`。
- 每个 subject 的 seed 绝对时间 `t_i = clip_start + float(frame_idx) * (clip_end - clip_start)`。
  选覆盖 seed 数最多的窗口；平票时让被覆盖的 seed 居中（deterministic）。
- 多 subject 且 seed 跨度超窗口时，**只剔掉覆盖不到的 subject**（进 `dropped_subjects`，
  reason `seed_outside_window`），不丢整行。全部覆盖不到才丢行。
- 窗口内帧号 `seed_frame_index = clamp(round((t_i - w0) * 16), 0, 80)`。
- bbox 原样落盘为标注坐标（字段名带 `_768` 后缀，已是**误名**，见上"bbox 坐标系"节），
  下游当前按假设 `H_768_long`（`scale = max(W, H) / 768`）映射到真实帧。
  **标定结论（2026-07-26，全表 651,031 行 / 1,345,848 框，三次扫描）**：
  - **y 轴严格服从长边=768 等比画布**，每个宽高比桶都精确贴合（16:9→432、4:3→576、
    1:1→768、2.22:1→345.6），99.25% 的框 y 轴装得进去；16:9 有 33.5% 的框 `y2` 恰好 = 432。
  - **x 轴不服从**，溢出是 x 轴专有的：16:9 卡在 768/798/800/832（`x2` p99 = 832，
    32,816 个 target 框精确命中；max 928），4:3 卡在 806.4（= 768×1.05，取整落 805/806/807）。
    只有 2.22:1 两轴干净（x2max 恰好 768、零溢出）。1:1 两轴都溢出（x2 max 1109）。
  - **"多个等比画布混合"已证伪**：`x2 == 832` 的框里 51.7% 的 `y2` 恰好 = 432 —— 那是 **768**
    画布的高度，若来自 832 画布（832×468）则 432 毫无特殊性。x 与 y 是**独立** clamp。
  - **clamp 由宽高比决定，不由源分辨率决定**：1920×1080 与 1280×720 的四个 clamp 集合与占比
    在 2-3 个百分点内重合 → x 轴**无法从 (W,H) 恢复**，重分桶依据在这份数据里不可见。
  - 实用路线：**按 `x2 > 768`（各桶按其拟合宽度）过滤掉那 14%**，而非去猜一个全局 scale
    —— 这保住 y 轴精确性。不过滤则那 14% 会横向出帧、被 stage C clamp 到右边缘（pilot 已见）。
  - 证据：`/mnt/pfs/users/yuanze/datasets/phantom_canvas_calib_v1/`
    （`canvas_estimator.json` / `long_edge_probe.json` / `xclamp_by_resolution.json`）。
    注意 `calib/estimate.py` 的相对 spike 规则打不着主 clamp（邻域中位数被自身抬高），
    **看 `x2_hist_top` 原始直方图，别看 spikes 列表**。

### sample_id 规则

`<koala_uuid>_w<窗口起点毫秒，9 位零填充>`，例如
`784cdb6812944b028c70ee5ac14ef6ad_w000050258`。

Phantom 的 `video_id`（`<uuid>_<start>_<end>`）含小数点，不适合做文件名，所以窗口起点编码成整数毫秒。
同一 (源视频, 窗口起点) 唯一 → deterministic 且无碰撞。

spec 里的 `video_id` 是**源视频粒度**的 koala uuid（不是 clip），train/eval split 按它切就是
source-disjoint。plan 阶段每个源视频只保留 1 行，天然避免同源视觉重复。

### 产出

```
<root>/clips/<sample_id>.mp4                    81 帧，fps 16，源分辨率
<root>/ref_frames/<sample_id>_subj<NN>.jpg      未抠图的原始 ref 帧
<root>/_stages/extract/<sample_id>.json         原子 marker（沿用 MarkerStore）
<root>/extracted.jsonl                          stage C 的输入契约
<root>/_selfcheck/                              画框自检图
```

`extracted.jsonl` 每行的 subject 里，`ref.frame` 是**原始 ref 帧**（待抠图）。
故意不叫 `object_reference`：在 UltraVid schema 里那个键专指白底抠图，是 stage C 的产出。

落地经过 `storage.py` 的 backend 接口（当前只有 `local`）。规模化后 clip 要存 BOS，
换 backend 即可，调用方不用改。

### 跑法（在 pod 里，uid 1010）

```bash
export PYTHONPATH=<repo>/third_party/PhantomData/src:<repo>/third_party/UltraVidPipeline/src
export NO_PROXY="192.168.0.0/16,10.0.0.0/8,127.0.0.1,localhost"
unset http_proxy https_proxy all_proxy   # BOS 出网要绕代理

D=/mnt/pfs/data/yuanze/phantom_koala_bboxref_v1
python -m phantom_data.build.plan --num-sources 100 --out $D/specs_pilot100.jsonl
python -m phantom_data.build.extract --specs $D/specs_pilot100.jsonl --dataset $D --workers 4
python tools/selfcheck_boxes.py $D 4          # 画框自检
python tools/probe_decord_seek.py $D/specs_pilot100.jsonl 3   # 复核 reader 复用是否安全
```

extract 可恢复：已有 passed marker 的 sample 跳过，failed 的会重试。
解码走 decord 直读 BOS presigned URL；**不要**用 ffmpeg 命令行读 URL（静态 ffmpeg 在
https presigned URL 上 segfault）。每 worker 一个独立 `FrameGrabber`（decord reader 非线程安全）。

**decord 会在坏源上永久 wedge**，这是全量规模的头号坑。实测一条 .mkv 让 worker 卡死 65 分钟：
0% CPU、无打开的 socket、RSS 涨到 38GB，最后**把整个 pod OOM 掉**（158/159 条已完成却全丢）。
线程阻塞在 C 里，Python 层 `cancel()` 和信号都进不去。两道防线：

- `--sample-timeout`（默认 300s，健康样本约 5s）到点放弃该样本、记 failed marker、继续跑。
  进程末尾用 `os._exit` 退出，否则解释器会永远 join 那个死线程。
- 每条完成立刻 append+fsync 到 `extract.partial.jsonl`，收尾时折进 `extracted.jsonl`。
  **没有这个的话被杀就等于全丢**：marker 只记形状（`video`/`width`/`height`/`frame_count`），
  重跑又会因为 marker 存在而跳过，最终写出一个近乎空的 manifest。

对**加这两道防线之前**建的数据集，用 specs+marker 重建 manifest（免得重下所有源视频）：

```bash
python tools/recover_extract_manifest.py --dataset $D --specs $D/specs_xxx.jsonl --dry-run
```

逐 subject 的框和 ref 路径取自 spec（marker 里没有），并校验每个引用的 clip/ref jpg 真的在盘上，
缺件的样本跳过并报出来，绝不写出训练时才会炸的行。恢复出的行带 `recovered_from_markers: true`，
且不含 `fps_source`/`source_total_frames`（只存在于丢掉的 manifest 里；stage C 不读这两个）。

### stage D: index

```bash
python -m phantom_data.build.index --dataset $D --output-name phantom_pilot_v1
python -m phantom_data.build.index --dataset $D --output-name phantom_pilot_v1_refclip020 \
  --min-ref-clip-score 0.20
```

产出 `<root>/indexes/<name>/`，与 UltraVid 的 `indexes/bboxref_clean_dedup_iou50` 同构：

```
metadata_train.csv / metadata_eval.csv   列与 UltraVid 完全一致（含空的 vace_video）
funnel.json                              机器可读漏斗 + provenance + 阈值 + split
funnel_lists/*.jsonl                     每阶段清单（含拒绝原因元数据）
quality_decisions.jsonl                  每个 built 样本的漏斗判定（viewer 读这个）
samples.jsonl                            去重后的最终 sample 行
README.md                                人读漏斗表
```

判定逻辑全部 import UltraVidPipeline（`stages.quality.filter_sample`、
`stages.index.deduplicate_sample` / `audit_dataset` / `_is_eval`），本模块只做编排，
所以两个数据集永远同一口径。`vace_video` 列写空串：训练侧从 bbox json 在线渲染控制信号
（`RenderBBoxControlWindow` 会覆盖该列），main_path 设置根本不读它；保留列名只为和
UltraVid 的 CSV header 完全一致。

`prompt` 列 = Phantom 的 `video_caption`（stage B/C 里的 `clip_prompt`），是**源 clip 级**
描述，与 UltraVid 该列口径一致。注意它描述整个 Phantom clip，而我们的窗口只是其中
5.0625 s 子区间，caption 可能提到窗口外的动作 —— 已记入 `funnel.json` 的 pending filters。

train/eval 按 `video_id`（koala 源视频 uuid）hash 切分，source-disjoint 且代码里 assert
无交集。

**阈值口径**：默认沿用 UltraVid 的 `min_ref_clip_score=0.23`。Phantom 的 prompt 是短名词
短语（"woman"/"cat"），CLIP 分天然低于 UltraVid 的 VLM 长句（中位 0.257 vs 0.279），所以
`--min-ref-clip-score` 可放宽。任何偏离都会写进 `funnel.json.threshold_deltas` 并在
README 顶部显式警告"与 UltraVid 索引不同口径"。

### 数据浏览器（本地已建数据集）

```bash
bash third_party/PhantomData/tools/run_build_viewer.sh        # 默认 :8504
PHANTOM_BUILD_INDEX=phantom_pilot_v1_refclip020 bash tools/run_build_viewer.sh
```

分页每页一个样本：clip 视频 + stage C contact sheet + 每个 subject 的白底抠图/名词短语/
`ref_clip_score`/`visible_frame_count`/被哪条阈值卡住 + 分辨率、seed 帧、caption、窗口时间、
Phantom video_id、BOS key。侧栏可筛"仅通过 / 仅被拒（按 rejection code）/ 仅多 subject"。

漏斗判定**不在 viewer 里重算**：只读 `indexes/<name>/quality_decisions.jsonl`，所以 viewer
不碰 masklet npz，也不可能和训练要吃的索引不一致。切换侧栏的 index 就换一套判定。

与 `tools/run_viewer.sh`（:8503，BOS 在线版）互不干扰，端口用
`PHANTOM_BUILD_VIEWER_PORT` 覆盖。

渲染路径验证（streamlit 只在浏览器连上时才执行脚本，端口通不代表页面能渲染）：

用 recording stub 顶替 `streamlit` 模块，跑一遍 `render()`×4 种筛选模式 + 全部 92 个样本的
`render_sample()`，并在 `st.image(use_container_width=...)` 上直接抛错 —— 部署环境是
streamlit **1.23.1**，只认 `use_column_width`，rerun 只有 `st.experimental_rerun`。

### 巡检渲染（`phantom_data.inspect` + :8506）

stage C 的 contact sheet 把框和 mask 轮廓画在同一张小图上，因此答不了两个问题：**框对不对**
（轮廓贴着物体，任何框看起来都合理）、**mask 干不干净**（空洞在轮廓图上完全不可见）。
`inspect` 是只读渲染器，读 stage B/C 已落盘的产物，**不跑 SAM2、不改任何 stage 输出**，
所以可以直接对 pilot 重跑而不污染它。

```bash
python -m phantom_data.inspect --dataset $D          # 落 $D/_inspect/<sample_id>/
python -m phantom_data.inspect --dataset $D --only <sample_id> --out-root _inspect_selfcheck
bash third_party/PhantomData/tools/run_inspect_viewer.sh   # 默认 :8506
```

每条样本三张图：

- `target.png` —— **只画框不画 mask**。黄 = Phantom 标注框（映射+clamp 后，仅 seed 帧）、
  青 = SAM2 该帧 mask 的外接框。
- `mask.png` —— mask 单独渲染，**与 `target.png` 同一组帧、同一顺序**（两者都从
  `pick_frames` 取，改代码时务必保持一致；红斑对不上画面的图比没有图更糟）。
  白 = 前景，**红 = `binary_fill_holes` 会补掉的内部空洞**，灰 = 背景；末尾几格是 ref 抠图的 alpha。
- `reference.png` —— ref **整帧** + 映射后的框，并排白底抠图。整帧是关键：存盘的
  `object_reference` 已经按 mask 裁过，看不出框有没有落在对的物体上。

`metrics.json` 里 `hole_*` 只上报不修正 —— 有些洞是真实几何（实测 `03b2396d` 的 3.45% 是
玩具蜜蜂的镂空翅膜），无条件填会把 mask 弄错。`clamp_report` 会重算「映射后但未 clamp」的框
（stage C 的 `scale_bbox_to_frame` 一步做完映射+clamp，中间值不落盘），把越界量按四条边拆开，
所以 14% 那批 x 轴越界不会被静默吸收进画面边缘。

**空洞的一个已确认成因**：视频分支用 `build_sam2_video_predictor(apply_postprocessing=True)`，
带 `fill_hole_area=8`；ref 分支是 `SAM2ImagePredictor(self.video)`，`max_hole_area=0.0`
是构造默认 → **ref 完全没补洞**（pod 内实测两个 predictor 的这两个属性确实是 8 和 0.0）。
`largest_components` 只删外部碎点，管不到内部空洞。

### ⚠️ 碎点会劫持 bbox（已修）

`bboxes_xyxy` 取的是 mask 的外接框，而**这个框就是训练的条件信号**，所以一个 1 像素的孤立
碎点能把框的边拽偏几百像素。100 条巡检集上实测（11,109 个可见帧）：

- **7.91% 的帧**的框被碎点撑大 >15%，涉及 **46% 的样本**（63/138）
- 最差撑到 **9.7 倍**：真实主体宽 171px，框成 1190px
- 只有 3% 的坏帧在 seed 帧上 → **主因是传播帧，不是 condition frame**
- 碎点面积占比极小（触发排查的那条只占 0.7%）→ **所有面积类指标都是假阴性**，
  该样本的 `max_mask_area_ratio` / 空洞 / `ref_mask_coverage` 全部正常

修法：`merge_directional_masks` 对每帧过一遍 `largest_components`（ref 分支早就在用了，
视频分支一直漏掉）。同一数据集上的安全性实测：64.3% 帧完全不变，35.7% 只削碎点
（中位删 0.119% 面积），**11,109 帧里只有 1 帧掉了 >30%** —— 那是一群 581 个连通块的牛
（多块本身就是 ground truth，不是伪影）。绝不把有主体的帧清空（空 mask 在下游等于"不可见"，
是另一个语义）。`despeckle=False` 可以拿回原始 SAM2 输出。

---

## bbox 修正管线（`enrich` → `redetect` → `gate_apply`，在 SAM2 之前）

Phantom 自带的 bbox 经常歪掉或框错对象，SAM2 抠图跟着废。这条管线用 Grounding DINO 给每个
subject 重打框，逐侧决定用哪个框，再筛掉 ref/target 不是同一个物体的样本。

**顺序（2026-07-29 改）**：

```
A plan → B extract → 1 enrich → 2 redetect → 3 gate_apply → C segment(SAM2) → D index
```

原来这条管线跑在 stage C **之后**，而且**没有任何东西把修正后的框写回去** —— redetect 只产报告，
keep/drop 只在前端实时算。pilot 实测 `gate_report.json` 说 93/140 ref 框和 107/140 seed 框该被替换，
也就是说已建好的 masklet 基本都是用**已知歪掉的框**抠的，而那个框就是训练的条件信号本身。

搬到 SAM2 之前的两个收益：SAM2 只跑过了身份门的样本（pilot 上 `identity_required` 口径 56%），
且抠图用的是修正后的框。

### 四个阶段

| | 跑什么 | 成本 |
|---|---|---|
| stage 1 | `tools/enrich_subjects.py` | 调 LLM，按次花钱，结果落盘缓存 |
| stage 2 | `tools/redetect_run.py` | 1 卡 GPU，140 subject 约 3 分钟；**有 resume** |
| stage 3 | `tools/gate_apply.py` | 纯 CPU 秒级，产 `gated.jsonl` |
| 前端 | `tools/run_gate_viewer.sh`（:8508） | 纯前端，不碰 GPU，随时看 |

分段是因为成本性质不同：stage 1 花钱、stage 2 吃 GPU、stage 3 和前端要反复调。
后面的可以任意重跑而不触发前面的成本。

```bash
D=/mnt/pfs/data/yuanze/phantom_koala_inspect100_v1
export PYTHONPATH=<repo>/third_party/PhantomData/src:<repo>/third_party/UltraVidPipeline/src

# stage 1：cometapi 必须直连，经代理实测 5/20 请求 RemoteDisconnected
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
python tools/enrich_subjects.py --dataset $D --workers 8

# stage 2：读 extracted.jsonl（不再读 segmented.jsonl），权重全在 HF_HOME 本地
python tools/redetect_run.py --dataset $D          # 先 --limit 3 smoke

# stage 3：产 gated.jsonl（stage C 的新输入）+ gated_drops.json（漏斗账）
python tools/gate_apply.py --dataset $D

# stage 3'：巡检用 —— 不过门，全部 subject 都出（含被门判负的），行上记 verdict
python tools/gate_apply.py --dataset $D --rule identity_required --keep-all

# stage C 吃 gated.jsonl
python -m phantom_data.build.segment --dataset $D --input gated.jsonl

# 前端随时看
bash tools/run_gate_viewer.sh
```

`redetect_run.py` 不再依赖 SAM2 跑过：ref 帧用 stage B 已落盘的 `ref.frame` jpg，target 帧解
clip，两个框用 `segment.scale_bbox_to_frame` 从 `seed_bbox_768` / `ref.bbox_768` 现算（纯函数，
canvas 映射 + clamp，没有模型）。**实测 140/140 个框与 stage C 原来预算进 `bbox/*.json` 的值
完全一致**，所以换输入不改任何数字。

代价：stage C 的 `ref_clip_score` / `ref_mask_coverage` 此时还不存在，报告里这两个字段为 None。
两者从来没进过任何判定（`decide` 一个都不读），只是展示上下文；而 `ref_clip_score` 本来就与
`crop_clip_*` 不可比（见下"三个坑"）。字段保留不删，因为 `--no-trust-detector` 对着 stage C
manifest 跑时仍会填上，pilot 报告要能继续用同一个 shape 校验。

模型固定 `qwen3.5-flash` + `enable_thinking: false`（deepseek 的官方 disable 字段在 comet 上
会漏关，thinking 泄进 completion）。API key 读 `<repo>/APIKEY_COMET`。
stage 1 有磁盘缓存 `$D/_enrich/`，重跑不花钱；**失败的不缓存**，gateway 挂了正是该重试的时候。

### `dis`：唯一的文本口径

stage 1 从长 caption（中位 207 词，超 CLIP 的 77 token）抽出一个描述性短语，字段名 `dis`
（`man wearing blue puffer jacket and glasses`）。它同时是**检测器的 query** 和 **CLIP 的文本** ——
所有分数都对着同一句话，所以彼此可比。

Phantom 原短语 65% ≤2 词（`woman`/`fish`），区分不出同类的不同个体，这是 stage 1 存在的理由。

### 选框：默认信检测器（`trust_detector`）

ref 侧和 target 侧**独立**判断（`redetect.pick_side`），两套模式：

**`trust_detector=True`（默认）**：检测器出框就用检测器的，没出框就**过滤掉这个 subject**。

理由是坐标系，不是质量：Grounding DINO 输出的是**真实帧像素坐标**（它看的就是解码后的帧），
不需要任何映射，也就不可能被错的映射搞歪。Phantom 的原始坐标必须过一个**尚未定论**的标注画布
（`H_768_long` 只是工作假设，且实测 x 轴是错的：全表 14% 的框溢出画布，4:3 桶 17-22%，1:1 桶 29%，
见上"bbox 坐标系"节）。所以两个框不是平等的两种意见 —— 一个在帧自己的坐标系里，另一个是对某个
投影的猜测。Phantom 的框降级为 prior，只留两个用途：IoU 一致性这个数，和找到物体的 `dis` 短语。

没出框因此是**过滤**而不是回退（`pick_side` 返回 `NO_BOX`，不是 `PHANTOM`）：回退等于把唯一
没有可信帧坐标的那个框送进产品，正是这个模式要拦的事。报告里 `pick_ref`/`pick_seed` 会是
`no_box`（不是 `phantom`），漏斗才数得清 —— 复用 `PHANTOM` 会让"被过滤"和"主动选了原框"在
报告里长得一模一样。

**`trust_detector=False`（`--no-trust-detector`）**：历史规则，**逐字节保留**，是对照组。

| 原框 crop 的 CLIP 分 | 新框 vs 原框 IoU | 用哪个 |
|---|---|---|
| ≥ 0.21（原标注没问题） | ≥ 0.75 | 新框（小幅微调） |
| ≥ 0.21 | < 0.75 | **原框**（新框跑偏了，信人工标注） |
| < 0.21（原标注可疑） | 任意 | 新框 |
| — | 检测器没出框 | 原框 |

pilot 的 `gate_report.json` 就是这套跑出来的，所以它必须保持可复现。
`tests/test_pick_side.py` 里钉了一份**独立的历史实现副本**，98 组参数逐一比对，
**连 reason 字符串都比**（reason 会写进报告、前端会显示，改词就是改产物）。

为什么现在才需要分模式：修正搬到了 SAM2 之前，选中的框变成了 SAM2 抠图的依据、变成了出厂的
训练条件信号；原来它只是报告里的一个批注，留着原框不花钱。pilot 实测老规则纯因 IoU < 0.75 而
保留原框的有 ref 侧 47/140、target 侧 33/140。

### 过滤：三个门，两套规则

三个测量值，**每个取两侧的较大值**（一侧确认 + 身份门通过就已钉住两侧）：

- **identity** = `dino_cos(ref crop, target crop)`，DINOv2 CLS token 余弦 —— 是不是同一个物体
- **clip** = crop 与 `dis` 的 CLIP 分 —— 框是不是在短语说的东西上
- **IoU** = 新框与原框的重合 —— 检测器和标注员是不是选了同一块区域

前端可切换（`redetect.RULES`），默认第二套：

| 规则 | 公式 |
|---|---|
| `identity_required` | `identity AND (clip OR IoU)` |
| `iou_stands`（默认） | `(clip AND identity) OR IoU` |

两套都要求 clip 或 IoU 至少一个成立，**唯一差别是 IoU 能否替代 identity**。
默认阈值 `identity ≥ 0.6`、`clip ≥ 0.21`、`IoU ≥ 0.75`，前端三个滑条可实时改。

**IoU 替代 identity 要小心**：IoU 是在各自帧内与该帧原框比的，而 ref 帧和 target 帧中位相隔
83 秒 —— IoU 高完全不代表两侧是同一个人。前端把这批标成 `KEEP*` 并单独计数。

⚠️ **pilot 报告存的 verdict 是 `identity_required` 口径，而代码的 `DEFAULT_RULE` 现在是
`iou_stands`** —— 同一份 `gate_report.json` 两套规则重算出来差一倍：

| 规则 | pilot 140 subject 的留存 |
|---|---|
| `identity_required` | **78**（= 56%，报告里存的那批） |
| `iou_stands`（代码默认） | **134**（96%，`gate_apply.py` 不带 `--rule` 时的结果） |

差的 56 条全是身份门判负、被 IoU 单独救回来的（中位 identity 0.485，15 条低于 0.4）。
所以"SAM2 只跑 56%"这个收益**只在 `identity_required` 下成立**；跑 `gate_apply.py` 前先想清楚
要哪套，用 `--rule identity_required` 显式指定。这两个阈值仍待人工定。

### 前端（:8508）

**框是前端实时画的**，不是烧进图里的。stage 2 只存干净帧图（每 subject 两张 jpg），
框坐标全在 `gate_report.json`。改颜色、线宽、开关某个框都是纯前端改动。
keep/drop 判定同样实时算（前端调 `redetect.decide()`），拖滑条立刻生效。

每个 subject 两行（ref / target），每行 = 整帧叠框 + 两个 crop。**红 = 原框，蓝 = grounding dino**，
选中的 crop 标 `← used`。判断看 crop：哪个矩形框对了是猜，crop 里装的是什么一目了然。

前端**不是**判定的落盘处，`gate_apply.py` 才是（前端只用来看和调阈值）。前端读字段全走
`.get()`，所以新顺序下 `ref_clip_score` / `ref_mask_coverage` 为 None 不会让页面崩。

### 坐标系是最危险的一个字段（`box_space`）

**报告里的 `chosen_box_*` 已经是真实帧像素坐标**，而 `extracted.jsonl` 的 `seed_bbox_768` 是
原始标注坐标、stage C 会拿去过 canvas 映射。如果 stage C 把修正后的框**再映射一次**，就是在
1920×1080 上再乘 2.5 —— **不报错，只是把每一个框放到别的地方去**。

所以 `gate_apply.py` 给每行打 `box_space: "frame"`，stage C 按它分派（`segment.resolve_box`）：

| `box_space` | stage C 怎么做 |
|---|---|
| 缺失 / `"annotation"` | 过 canvas 映射 + clamp（历史行为，现存 `extracted.jsonl` 全走这条） |
| `"frame"` | **只 clamp，不映射** |
| 其他任何值 | **抛 ValueError**（不猜） |

两点是刻意的：

- **用数据上的 tag，不用 CLI flag**。flag 会被指到错的输入文件上；tag 跟着它描述的那批行走，
  不可能配错。
- **分派必须穷尽且显式**，未知值大声报错。两种猜错方向都无法在下游被发现（拿 frame 框当
  annotation 会放大，拿 annotation 框当 frame 会不映射），manifest 写错一个字必须让 run 停下来，
  而不是产出一个看着正常、训出来不对的数据集。

字段名仍叫 `seed_bbox_768` / `ref.bbox_768`（值却换了坐标系）—— 那是 stage C 的输入契约
（`parse_sample` 认这两个名字），`_768` 后缀本来就是已记录的误名（意思只是"框字段"），
真正声明坐标系的是行上的 `box_space`。另造 `*_frame` 字段等于让 `parse_sample` 认两套 schema，
然后两条码路会对"谁优先"产生分歧。

### 产出

```
$D/_enrich/<sample_id>_subj<NN>.json          stage 1 短语缓存（dis + text_source）
$D/_redetect100/<sample>/subj<NN>_ref.jpg     干净 ref 帧（不画框）
$D/_redetect100/<sample>/subj<NN>_target.jpg  干净 target 帧
$D/_redetect100/gate_report.json              所有框坐标 + 所有分数 + 汇总
$D/_redetect100/redetect.partial.jsonl        resume 用的增量行（fsync 落盘）
$D/_redetect100/_stages/redetect/<id>.json    resume marker（每 sample 一个）
$D/_redetect100/gated_drops.json              被丢的 subject + 是哪个门丢的（漏斗账）
$D/gated.jsonl                                **stage C 的新输入**（修正框 + box_space=frame）
```

resume 照抄 `build/extract.py`：marker 说"这个 sample 做完了没"，partial jsonl 存"它产出了什么"
（marker 只记 shape，重建不出每 subject 几十个坐标和分数）。**先写行、后写 marker**，
中间崩掉代价是重跑一个 sample，而不是报告里躺半个 sample。marker 存在 `--out-root` 下而不是
dataset 根下，这样换 `--out-root`（换规则、换短语集）各自有独立的 resume 状态，不会互相顶掉。

`recover_partial` 只认 marker 里有的 sample：崩在"写完行、没写 marker"之间的那些行会被重算，
收下它们就会在报告里出现重复 subject。同时按 `(sample_id, subject_id)` 去重，`--force` 追加
写也不会产出两行。

### 三个坑

- **`crop_clip_*` 与 stage C 的 `ref_clip_score` 不可比**。前者是带背景的原始 crop，
  后者看 SAM2 白底抠图，同一个框两个数能差 0.05 以上。阈值 0.21 只对前者成立。
  新顺序下 stage C 还没跑，报告里 `ref_clip_score` / `ref_mask_coverage` 直接是 None。
- **`dino_cos` 的口径局限**：用 DINOv2-base 的 CLS token（256 个 patch token 全丢），
  且 processor 是 `shortest_edge=256` → `center_crop 224` —— **细长框（如 1:3 的人物框）
  只剩中段，头脚都没了**。所以低分有两种可能：真的不是同一个物体，或者被裁掉了关键部分。
  要换口径改 `redetect.Models.dino_embedding` 一处，只需重跑 DINOv2 那一步。
