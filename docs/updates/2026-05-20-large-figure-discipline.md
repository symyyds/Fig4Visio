# 2026-05-20 大图复杂图流程优化

## 问题判断

多次使用后可以看到一个稳定现象：Visiomaster 对小图、局部模块、结构清晰的论文框图处理效果更好；当原图变宽、模块变多、文字变密、跨模块箭头变多时，错误会明显增加。

这类问题通常不是单个 Visio 组件缺失，而是全局制作流程失控：

- 全图一次性分析会把局部坐标误差放大，导致框、字、箭头在不同区域里逐渐漂移。
- 同一类节点在不同模块里被分别估计字号，最后出现字体大小不统一。
- 大图缩放后，原本很小的文本框或箭头段在 Visio 中更容易出现文字溢出、箭头头部挤压、框线遮挡。
- 没有足够 `group_container` 或 `audit_region` 时，审查脚本只能看全局，难以发现局部模块的子节点数量、输入输出和边界箭头是否合理。
- 大图里“整体像”会掩盖局部错，比如一个模块内部对齐偏下、某条箭头穿过节点、某个边界输出被连到了组件中心。

因此这轮优化重点不是再扩充组件库，而是把大图制作变成更可控的分区流程。

## 优化方向

新增大图纪律：

1. 先建区域蓝图，再做细节。
2. 可见模块使用 `group_container`，不可见逻辑模块使用 `audit_region`。
3. 关键节点必须绑定 `container_id`。
4. 一个区域尽量控制在 12-18 个可见节点以内。
5. 大图先跑复杂度预检，再跑模块审查，再进入 Visio 渲染。
6. 字体按角色统一，而不是每个模块独立估计。
7. 从局部裁剪图制作时，局部坐标只能用于分析，最终必须转换回全页像素坐标。

## 代码变化

### 新增 `scripts/scene_complexity.py`

新增一个轻量级复杂度预检脚本，不依赖 Visio，不做渲染，只输出 `.complexity.md` 报告。

它会检查：

- 可见语义节点数量
- 边数量
- 页面宽高比
- 区域数量
- 节点是否被 `group_container` / `audit_region` 覆盖
- 每个区域包含多少可见节点
- 跨区域边数量
- 同类节点字号范围
- 文字是否可能超出节点框
- 是否存在高风险重叠
- 当前 `scene_validate.py` 的警告/错误摘要

推荐在大图、宽图、密集论文图进入 Visio 渲染前运行：

```powershell
python .\scripts\scene_complexity.py .\work\scene.json
```

### 增强 `scripts/scene_validate.py`

新增大图相关 lint：

- 复杂图但没有设置 `metadata.region_strategy` 时发出警告。
- 复杂图区域数量不足时提示增加 `audit_region` / `group_container`。
- 可见节点没有归属区域时提示显式设置 `container_id`。
- 单个区域超过 18 个可见节点时提示拆分。
- 同一节点类型字号跨度超过 3pt 时提示统一字体尺度。
- 估算文字宽高，发现可能装不下时提示换行、放大框或降低字号。
- 检测非容器节点高风险重叠，避免大图拼接后框体混乱。

这些检查用于提前暴露“字对不齐、框乱、字号漂移”等问题。

### 增强 `scripts/image_to_scene.py`

新增大图起始参数：

- `--pixel-page`：直接使用原图像素宽高作为 scene 页面坐标。
- `--target-width-in` / `--target-height-in`：控制最终渲染到 Visio 的英寸尺寸。
- `--region-strategy`：在 metadata 中记录大图制作策略。
- `--fidelity`：在 metadata 中记录复刻等级，默认 `exact`。

推荐大图起手命令：

```powershell
python .\scripts\image_to_scene.py --image C:\path\source.png --pixel-page --region-strategy region_first --output .\work\scene.json
```

### 更新 `SKILL.md`

新增大图工作规则：

- 大图不要全页一把梭。
- 优先使用 `region_first` 或 `tiled_subscenes`。
- 局部制作后要转换回全页坐标。
- 使用 `scene_complexity.py` 做预检。
- 区域过密时继续拆小。
- 用 `align_group`、`align_to_container`、`container_id`、side-ratio endpoint 控制局部对齐。

### 更新 `references/scene-schema.md`

新增 `metadata.region_strategy` 和 `metadata.font_scale` 说明，并补充“大图纪律”章节。

## 当前能力边界

这轮优化可以显著降低大图失控概率，但它不是自动视觉识别引擎。复杂图仍然需要用户或 AI 在 `scene.json` 中定义模块、节点、边和样式。

新的脚本能提前发现结构和排版风险，但不会自动修复所有问题。后续可以继续优化：

- 自动从 `audit_region` 导出局部裁剪图。
- 支持局部 scene 到全局 scene 的坐标合并工具。
- 在渲染后加入图像级对比，自动标出偏移最大的模块。
- 为大图维护一套更严格的字体 token 和边距 token。
