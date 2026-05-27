# Visiomaster vNext 更新说明

本次更新重点不是单纯增加几个 Visio 组件，而是强化“图片复刻为可编辑 Visio”的完整审查与重建闭环，让复杂论文图、AI 生成流程图、架构图在多轮优化时更可控、更可复现。

## 核心更新

- 新增源图暂存流程：通过 `stage_source_image.py` 固定原图路径并记录 SHA-256，避免后续审查依赖聊天记录、截图记忆或旧输出。
- 新增双图视觉审查流程：要求视觉 LLM 同时查看“原图”和“复刻 PNG”，逐项比较布局、组件、箭头、字体、公式、颜色和局部拓扑。
- 新增 review bundle：通过 `make_review_assets.py` 生成审查 manifest 和模板，统一记录本轮原图、复刻图、scene 和 round 信息。
- 新增审查清单 gate：`review_checklist_gate.py` 检查审查结果是否包含具体 topology checklist、visual checklist 和失败项引用。
- 新增 review findings 到 rebuild brief 的转换：`review_findings_to_repair_plan.py` 将视觉差异转成下一轮重建输入。
- 新增 regeneration packet：`prepare_regeneration_packet.py` 整理下一轮完整重建所需的原图、复刻图、审查摘要和提示。
- 新增 no-op gate：`round_noop_gate.py` 用来发现“只改了元数据、弱样式字段或 PNG 没变化”的无效迭代。
- 强化严格复刻规则：复杂图不再鼓励直接修旧 scene，而是要求从原图视觉 inventory 和 source-pixel region plan 开始重新生成。
- 强化 renderer / validator / audit：增强对字体、公式、复杂箭头、虚线区域、边界端口、loss/feedback 类组件的检查和约束。

## 相比上一版的变化

上一版更像是 `scene.json -> Visio` 的可编辑图形渲染工具；这一版开始加入更完整的“复刻质量闭环”：

```text
原图
-> source staging / visual inventory
-> scene.json
-> Visio 导出
-> 复刻 PNG
-> 原图/复刻图视觉审查
-> 结构化问题清单
-> rebuild brief
-> regeneration packet
-> 下一轮完整重建
```

目标是减少复杂图里常见的局部问题，例如：

- 箭头端点错误或指向错误；
- 曲线路径不自然；
- 字体、字号、公式、上下标不一致；
- 虚线框、边界框、分组区域错位；
- 大图局部模块结构被打乱；
- 多轮修改只改了元数据或弱样式，但视觉结果没有实际变化。

## 新增文件

- `references/review-contract.md`
- `references/reviewer-two-image-prompt.md`
- `references/full-scene-regeneration-prompt.md`
- `references/renderer-effective-fields.json`
- `scripts/stage_source_image.py`
- `scripts/make_review_assets.py`
- `scripts/review_checklist_gate.py`
- `scripts/review_findings_to_repair_plan.py`
- `scripts/prepare_regeneration_packet.py`
- `scripts/round_noop_gate.py`

## 清理和发布准备

- 移除了本地临时输出、回归测试图片包、`.vsdx` 调试文件、`__pycache__` 和 `.pytest_cache`。
- 清理了公开仓库中不应出现的本机绝对路径。
- 更新了 `.gitignore` 和 `.gitattributes`，减少 Windows 换行和临时产物误上传问题。
- 更新了 README 的仓库结构、严格复刻流程和当前限制说明。
- 增加了基础 smoke tests，方便下载后快速确认脚本和示例 scene 没有明显断裂。

## 注意事项

- 本工具仍然优先支持 Windows + Microsoft Visio 桌面版 + `pywin32`。
- Python 脚本只能作为辅助 gate，不能替代视觉 LLM 或人工肉眼审查。
- 复杂论文图和高精度复刻仍可能需要多轮视觉审查和重建。
- 不同 Visio / Office 版本、系统语言和字体安装情况可能导致导出效果有差异。

## 验证情况

本次发布副本已完成以下基础检查：

- `python -m pytest -q`：通过，`3 passed`
- `python -m compileall -q scripts sync_to_skill.py tests`：通过
- 当前保留的 `13` 个 example scene 均通过 `scene_validate.py`
- 本地路径和隐私关键词扫描未发现残留
