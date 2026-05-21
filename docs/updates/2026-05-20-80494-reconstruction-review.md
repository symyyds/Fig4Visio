# 2026-05-20 80494 复刻样例局部审查优化

## 样例问题

本次对比对象：

- 生成图：`examples/80494_reconstruction/exports/scene.png`
- 原图：用户提供的 80494 论文模块参考图。

从整体看，生成图已经能表达大体结构，但局部复刻仍有明显失真。这类问题不容易靠全图相似度发现，需要按模块逐项审查。

## 观察到的局部问题

### 1. Environmental perception modeling 内部箭头

用户指出的小图区域里，蓝色小 cuboid 到橙色 Environment Response extractor 的箭头应当是水平短箭头。实际生成图中有些箭头呈轻微倾斜。

根因不是 Visio 箭头样式缺失，而是 scene 中常用：

```json
{
  "type": "arrow_connector",
  "route": "straight",
  "allow_diagonal": true
}
```

当 `from_point` 和 `to_point` 的 y 值只差几像素时，`straight` 会忠实画出斜线。对论文模块图来说，这种短 lane 通常应该强制水平或垂直。

### 2. Quality Head 后的 q 向量公式

原图中 `q = [q_RGB, q_IR, q_SAR]^T` 是紧凑的矩阵/向量公式。生成图里用普通 `text_block` 加 Unicode bracket 字符模拟，容易出现：

- bracket 和条目不贴合
- 行距不稳定
- 字体回退导致公式观感不一致
- `q =` 与向量主体基线不统一

这应该变成专门的公式向量组件，而不是普通多行文本。

### 3. Quality Head / Extractor / Aggregation 模块贴图化

当前样例里 `quality_head`、`environment_extractor`、`aggregation_quality` 等局部模块为了速度使用了小 raster tile。它比整图贴图好，但仍然不是完全可编辑复刻。

这些 wedge/hourglass 模块应优先尝试：

- `trapezoid_node`
- `polygon_node`
- 必要时用多个 polygon/cuboid 组合

只有在局部非常复杂且用户接受速度优先时，才保留小图贴片。

### 4. 审查脚本没有充分提示“看起来还行但局部错”的问题

旧版审查能提示跨容器边界、斜线等粗粒度问题，但对以下情况提示不够明确：

- `allow_diagonal: true` 掩盖本该水平的 paper lane
- 向量公式仍用普通 `text_block`
- wedge 模块仍是小图贴片

### 5. 其他肉眼可见但容易被忽略的局部偏差

本样例还暴露出一类更普遍的问题：整体结构相似时，局部视觉错误会被掩盖。

- 左侧浅层响应模块里，`DWConv`、`1×1 Conv`、`BN`、`SiLU`、`Residual Unit` 原图有轻微 3D 叠层/右侧面，生成图更接近平面矩形，语义正确但论文图观感变弱。
- `Availability Mask`、`P_RGB/P_IR/P_SAR`、`a_m` 这一组竖向构件，原图更紧凑，生成图中部分标签和竖条间距偏松，说明大图一口气布局时局部比例容易漂移。
- 右侧 `GAP` 特征格子和环境编码器局部颜色块，原图是浅色网格/渐变块，生成图容易变成偏实心、偏深的格子，说明特征图应优先用 `feature_map_grid` 的色带参数，而不是单一深色填充。
- `Conditional Variable Generator` 周边的长折线在原图里是贴着面板边界的结构性通路，生成图里容易变成长线段连接到组件中心，后续应优先用 `boundary_port` / `boundary_arrow` 表达“从框边界输出”。

这些问题不是单个 Visio 组件缺失造成的，而是 scene authoring 阶段没有把“局部类型、局部比例、局部拓扑”写得足够明确。skill 侧应把它们变成可审查的规则，而不是等最后看大图时才手动找。

## 本次修改

### 新增 `math_vector`

新增节点类型：

```json
{
  "id": "q_vector",
  "type": "math_vector",
  "prefix": "q =",
  "entries": ["q_RGB", "q_IR", "q_SAR"]
}
```

渲染方式：

- prefix 是可编辑文本
- bracket 是可编辑线段
- entries 是分行可编辑文本

这样比普通 `text_block` 更适合 Quality Head 后的公式向量。

### 新增 `lane_arrow`

新增边类型：

```json
{
  "id": "grad_to_extractor",
  "type": "lane_arrow",
  "from_point": [882, 521],
  "to_point": [912, 521],
  "route": "horizontal",
  "lane_axis": "horizontal"
}
```

用途：

- 蓝色 cuboid 到 extractor 的短箭头
- `GAP -> GMP`
- feature map 到 aggregation 的短箭头
- 其他应当水平/垂直的 paper-flow lane

`lane_arrow` 的目标是防止用 `straight` 画出轻微倾斜的箭头。

### 增强 `scene_validate.py`

新增检查：

- `math_vector` 的 `entries` / `rows` / `prefix` 参数合法性。
- `lane_arrow` 是否仍出现 diagonal segment。
- `lane_arrow` 的 `lane_axis` 是否和实际路径一致。
- 对 `arrow_connector` / `dynamic_connector` 上的 `allow_diagonal: true` 做语义检查：如果边名看起来像 `gap`、`gmp`、`extractor`、`quality`、`aggregation`、`projection`、`environment`、`spine` 这类 paper-flow lane，会提示改成 `lane_arrow` 或强制 axis route。

### 增强 `scene_audit.py`

新增审查提示：

- 如果 `text_block` 里有矩阵 bracket 字符且多行，提示改用 `math_vector`。
- 如果 `image_tile` 看起来是 `quality_head`、`environment_extractor`、`aggregation_quality` 等模块，提示优先改成 editable `trapezoid_node` / `polygon_node`。
- 如果 paper-flow lane 用 `allow_diagonal: true` 放过斜线，提示改用 `lane_arrow`。
- 对公式检测做了收窄，避免把普通的多行统计文字误判成公式向量。

### 新增 smoke 示例

新增：

`templates/examples/lane_math_vector.scene.json`

用于验证：

- `math_vector` 能通过 schema 校验。
- `lane_arrow` 能表达水平短箭头。
- Quality Head 和 Environment Extractor 可以用 editable `trapezoid_node` 而不是贴片。

## 后续建议

这次先把“审查发现 -> skill 规则 -> 校验器提示”的链路补上。后续如果要继续提升 80494 这张图本身的复刻质量，建议优先做：

1. 把 `quality_head` 从 raster tile 改为 `trapezoid_node`。
2. 把 `q_formula` + `q_vector` 合并为一个 `math_vector`。
3. 把 `fs_to_extractor`、`grad_to_extractor`、`cm_to_extractor` 等短箭头改为 `lane_arrow`。
4. 把 `environment_extractor` 和 `aggregation_quality` 从 raster tile 逐步替换成 editable polygons。
5. 把结构状态编码区域的长折线拆成边界端口和短桥接线，避免看起来像组件中心到组件中心的误连。
6. 把局部 3D 块和特征格子统一到 `stacked_process`、`cuboid_node`、`feature_map_grid` 的显式参数，减少大图自动布局导致的比例漂移。
