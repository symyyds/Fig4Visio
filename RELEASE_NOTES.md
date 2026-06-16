# Fig4Visio v0.3.7 Release Notes

This update adds a generalized sparse/no-frame variant inside the Swin Transformer reconstruction category. It is not tied to a filename or single image; the selector uses OCR span plus frame-density evidence to distinguish the standard Swin paper figure from sparse Swin variants where the source lacks large dashed stage frames.

## Core Updates

- Added `swin_transformer_sparse` reconstruction for low-contrast or sparse Swin architecture variants.
- Keeps the standard `swin_transformer` template unchanged for normal framed Swin figures.
- The sparse variant preserves visible modules as editable Visio objects: stage labels, split Patch Merging labels, Swin Transformer Block labels, residual plus stack, MLP/LN/W-MSA/SW-MSA blocks, arrows, and captions.
- No original image, `image_tile`, or raster reference layer is embedded.
- Added regression coverage for the sparse Swin variant.

## Verification

- `python -m pytest tests\test_public_release_smoke.py -q`: 18 passed
- `python -m compileall -q scripts tests gui_app.py sync_to_skill.py`: passed
- Retested the four previously supplied images: all passed screenshot self-check, all had `assets=0`, `image_tiles=0`.

# Fig4Visio v0.3.6 Release Notes

This update adds a category-specific semantic reconstruction path for compact attention mechanism paper figures. It is triggered by the combined OCR/layout signal `Attention mechanism` + `Sigmoid` + `Conv1d` + `Weighted vector` + `High-level features` + `AM-ResNet features`, so it does not change the global trace rules for unrelated images.

## Core Updates

- Added an `attention_mechanism` detector and editable source-coordinate builder.
- Reconstructs the high-level feature bands, dashed attention frame, Conv1d/Sigmoid blocks, weighted vector, multiply operator, AM-ResNet feature grid, connectors, and caption as Visio-editable geometry.
- Keeps `assets: []`, `visual_reference_layer: false`, and `raster_tile_policy: semantic_template_no_raster_tiles`; no full-image embedding or `image_tile` fallback is used.
- Added a regression test requiring the attention mechanism path to output editable module primitives and preserve the key labels.

## Verification

- `python -m pytest tests\test_public_release_smoke.py -q`: 17 passed
- `python -m compileall -q scripts tests gui_app.py sync_to_skill.py`: passed
- `python gui_app.py --smoke`: passed
- User-provided attention mechanism image: GUI workflow passed on the first round, screenshot self-check score `0.633`, `download_allowed=True`, and no image embedding.

# Fig4Visio v0.3.5 更新说明

本次更新修复 cross-attention 论文结构图的自动复现路径。旧版本会把这类图走通用轮廓/trace，生成大量碎线，虽然没有嵌入原图，但 attention 核心和残差分支不可读，且自检会禁止下载。

## 核心更新

- 新增 cross-attention 语义重建路径：识别 `AM-ResNet`、`Wav2vec 2.0`、`Softmax`、`Concat`、`norm`、`Feed forward`、`Cross-fused features` 和 `cross-attention` 后，直接生成可编辑模块。
- Q/K/V token、Softmax、attention 小矩阵、value-weighted 小矩阵、上/下 Concat-norm-feed-forward 残差分支、最终 Concat 和 Cross-fused 输出均为 Visio 可编辑对象。
- 小型 attention 矩阵使用 `grid_matrix` 复现，不使用图片裁片。
- 增加回归测试：cross-attention 命中后必须无 `image_tile`、无 assets、包含至少 4 个 `grid_matrix`，并保留关键标签。

## 验证情况

- `python -m pytest tests\test_public_release_smoke.py -q`：16 passed
- `python gui_app.py --smoke`：通过
- `python -m compileall -q gui_app.py scripts tests sync_to_skill.py`：通过
- 用户提供的 cross-attention 示例：GUI 完整工作流第 1 轮通过，自检评分 `0.5611`，`download_allowed=True`，且 VSDX 无图片嵌入。

# Fig4Visio v0.3.4 更新说明

本次更新修复 mask res-block 论文结构图的复现质量和自检提示误导问题。旧版本会把该类图走通用轮廓提取，输出大块背景和碎线；同时在总分已经高于阈值时，GUI 仍错误提示“评分低于阈值”。

## 核心更新

- 新增 original res-block / mask res-block 语义重建路径：识别 `Conv7-64`、`Batch normalization`、`Max-pooling`、`Original res-block`、`Mask res-block` 后，直接生成可编辑残差 lane、卷积/归一化/ReLU 模块、mask pooling 支路、加法/乘法节点、虚线框、标题和 caption。
- 修正 GUI 自检失败说明：总分通过但结构 gate 未通过时，会明确列出失败项，例如网格密度分布、分区墨迹覆盖或墨迹比例，不再误报“低于阈值”。
- 自检 JSON 升级到 `schema_version: 0.2`，新增 `failed_rules`，便于 GUI、日志和质量报告复用同一套失败原因。
- 增加回归测试：mask res-block 必须走无 `image_tile` 的可编辑模板；GUI 摘要必须正确处理“总分过阈值但 gate 失败”的情况。

## 验证情况

- `python -m pytest tests\test_public_release_smoke.py -q`：15 passed
- `python gui_app.py --smoke`：通过
- `python -m compileall -q gui_app.py scripts tests sync_to_skill.py`：通过
- 用户提供的 mask res-block 示例：GUI 完整工作流第 1 轮通过，自检评分 `0.7244`，`download_allowed=True`，且 VSDX 无图片嵌入。

# Fig4Visio v0.3.3 更新说明

本次更新修复 GUI 对宽幅论文架构图的严重误判：旧版本会把 Swin Transformer 这类黑白模块图拆成碎线和零散文字，且自检仍可能放行。

## 核心更新

- 新增 Swin Transformer architecture 语义重建路径：识别 `Swin Transformer Block`、`Stage`、`Patch Merging`、`W-MSA/SW-MSA`、`MLP/LN` 后，直接生成可编辑 stage 框、Patch/Linear/Swin 模块、右侧 residual block、主干箭头和标题。
- 强化截图自检：新增网格墨迹密度和全局墨迹平衡指标，避免白底图因为“空白区域相似”而通过。
- 增加回归测试：坏输出缺失左侧主干时必须 fail；Swin 架构图模板必须无 `image_tile`、无资产嵌入，并包含关键可编辑模块。

## 验证情况

- `python -m pytest tests\test_public_release_smoke.py -q`：13 passed
- `python gui_app.py --smoke`：通过
- 用户提供的 Swin Transformer 示例：坏输出自检 fail，新模板渲染自检 pass，且无图片嵌入。

# Fig4Visio v0.3.1 更新说明

本次更新聚焦“箭头拓扑审查与重建闭环”。它不是新增某一类固定图形模板，而是让局部拓扑复杂、连接语义强的图在复刻、审查、修复和下一轮重建之间更可追踪。

## 核心更新

- 强化 `metadata.arrow_plan`：严格复刻前先逐条记录原图可见箭头的来源、目标、端点、路径形态、线型、箭头头和语义 intent。
- 强化 scene 绑定规则：可见 edge 需要通过 `arrow_plan_id` 绑定原图箭头事实；一个 arrow plan 默认只能对应一条 scene/motif edge。
- 增加 motif 内部连线可审计能力：`nodes[].motif_edges[]` 可以声明内部 connector，并绑定 `arrow_plan_id`，避免内部线条游离在审查体系之外。
- 增强 review 模板：`make_review_assets.py` 会从 `metadata.arrow_plan` 自动生成 topology checklist，让 reviewer 按箭头 id 逐项检查。
- 增强 repair brief：`review_findings_to_repair_plan.py` 保留 `checklist_refs`，并生成 `arrow_plan_repair_targets`，把问题映射到具体 edge、motif 和可编辑字段。
- 增强 regeneration packet：`prepare_regeneration_packet.py` 会带上 topology checklist、visual checklist、checklist validation 和 arrow repair targets，并展开到 Markdown prompt。
- 强化 validator/audit：新增 arrow-plan 覆盖率、多 edge 误用、端点/路径不匹配、local motif 规则和 motif edge 覆盖检查。

## 适用场景

本次优化适用于局部拓扑复杂、连接语义强、审查修复链路要求高的图形复刻任务，包括论文模块图、神经网络结构图、系统架构图、流程控制图和多分支数据流图等。

它重点解决几类通用问题：局部连接关系密集但整体相似度掩盖错误、箭头端点和路径形态不稳定、内部组件连线无法被审计、视觉审查结果难以映射回 scene 修改目标。

## 相比 v0.3.0

v0.3.0 建立了“原图/复刻图审查 -> rebuild brief -> regeneration packet”的完整复刻闭环；v0.3.1 把其中最容易失真的箭头拓扑进一步结构化：

```text
source arrow inventory
-> metadata.arrow_plan
-> scene edge / motif edge binding
-> topology checklist
-> review findings
-> arrow_plan_repair_targets
-> regeneration prompt
```

这样下一轮 LLM 不只知道“箭头不对”，还会看到应该修改哪个 `edge_id`、哪个 `motif_edges[]` 绑定，以及需要保持的端点、路径和语义约束。

## 注意事项

- `motif_edges` 目前主要用于审计、映射和重建提示，不等同于所有 renderer 内部连线都已完全声明式渲染。
- Python gate 负责发现结构和证据链问题，不能替代原图/复刻图的视觉审查。
- 本工具仍优先支持 Windows + Microsoft Visio 桌面版 + `pywin32`。

## 验证情况

- `python -m compileall -q scripts tests sync_to_skill.py`：通过
- `python -m pytest -q`：通过
- 当前示例 scene 验证通过
