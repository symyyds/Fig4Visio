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
