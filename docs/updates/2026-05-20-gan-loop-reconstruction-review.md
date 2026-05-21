# 2026-05-20 GAN Loop 复刻样例局部审查优化

## 样例问题

本次对比对象：

- 参考图：用户当前消息中提供的 GAN 训练循环图。
- 生成图：`examples/gan_loop/exports/figure2_gan_loop.scene.png`
- 对应 scene：`templates/examples/gan_loop_feedback.scene.json`

整体节点语义基本可读，但这类图最容易失败在“线的语法”：外圈循环、虚线反馈、虚线说明框、箭头指向。单看模块框位置会觉得差不多，但一看训练流程逻辑就会露出问题。

## 观察到的局部问题

### 1. 最外围循环箭头被拆断

原图外圈是顺滑的训练循环曲线，箭头应该贴着曲线切线方向指向 `Latent Vector z` 和底部更新方向。生成图把它拆成：

- `outer_left_loop`
- `outer_bottom_loop`
- `outer_right_loop`
- `outer_left_arrow`
- `outer_right_arrow`

前三个是普通 `line_segment`，后两个是单独短 `arrow_connector`。这种做法会导致：

- 曲线在视觉上断开。
- 箭头和曲线不共线，方向像是单独插上去的。
- 外圈底部曲线和 `Alternating Updates` 文本发生拥挤。

正确语义应该是一个或两个连续的 `loop_arrow` / `curved_arrow` 路径，而不是线段加独立箭头。

### 2. 上方虚线框被当成流程框

原图中 `Forward Reconstruction -> Discriminator Evaluation` 下方的虚线区域是说明/评价区域，不是一个普通流程节点。生成图使用空 `process_box` 画虚线框，导致审查工具之前无法区分它和真实流程模块。

这个区域应该用 `dashed_region` 表达，内部文字用独立 `text_block`。

### 3. 虚线箭头和下方回传路径语义不稳

原图中 dashed loss/backprop paths 是训练反馈路径，应该保持正交、连续、箭头在最后一段。生成图里：

- `loss_to_disc_left` 和 `loss_to_disc_right` 用 `allow_diagonal: true` 放过斜线。
- 下方回传路径被拆成左右路径加多根独立向上箭头，容易在视觉上变成“很多竖箭头”，而不是一组来自 loss backprop 的反馈通路。

这不是箭头样式问题，而是 edge 类型没有表达“虚线反馈路径”。

### 4. 审查脚本误判损失文本

旧逻辑会把 `Reconstruction Loss L_rec / Adversarial Loss L_adv` 这种普通损失说明误判成向量/矩阵公式，提示改用 `math_vector`。这会误导后续制作。

这类文本应保持 `text_block`，只有真正的 bracket/vector 结构才用 `math_vector`。

## 本次修改

### 新增 `loop_arrow` / `curved_arrow`

新增连续路径类边，用于外圈循环或光滑曲线箭头：

```json
{
  "id": "outer_left_loop_to_latent",
  "type": "loop_arrow",
  "from_point": [147, 481],
  "points": [[60, 410], [42, 215], [135, 80], [290, 28]],
  "to_point": [348, 22]
}
```

渲染层会把它画成单个 Visio path，而不是逐段 `DrawLine`。这样箭头头部能贴着路径末端方向，不再像单独补上去。

### 新增 `dashed_region`

新增 visible dashed annotation frame：

```json
{
  "id": "forward_loss_region",
  "type": "dashed_region",
  "x": 292,
  "y": 210,
  "w": 297,
  "h": 90
}
```

它和 `group_container` 一样参与模块审查，但语义上表示“虚线说明/评价区域”，不表示普通流程节点。

### 新增 `dashed_feedback_path`

新增 dashed feedback/training path：

```json
{
  "id": "left_backprop_to_disc",
  "type": "dashed_feedback_path",
  "from_point": [194, 415],
  "points": [[194, 492], [426, 492]],
  "to_point": [426, 368]
}
```

它会作为一个连续虚线路径渲染，箭头只放在最终段。用于 reconstruction loss、adversarial loss、gradient penalty、backpropagation 这类反馈路线。

### 增强 `scene_validate.py`

新增检查：

- 空的虚线 `process_box` 会提示改用 `dashed_region`。
- `outer/loop/cycle` 命名的普通线段会提示改用 `loop_arrow` / `curved_arrow`。
- detached loop arrowhead 会提示合并到曲线路径本身。
- loss/backprop/feedback/gradient/penalty 等虚线箭头如果用 `allow_diagonal: true`，会提示改用 `dashed_feedback_path`。
- `dashed_feedback_path` 如果含 diagonal segment，会提示改成正交路径。

### 增强 `scene_audit.py`

新增审查提示：

- 能识别失败 scene 中被拆开的外圈曲线。
- 能识别 detached outer arrowhead。
- 能识别空虚线流程框。
- 能识别 loss/backprop 斜虚线。
- 收窄公式检测，避免把普通损失说明误判成 `math_vector`。

### 新增 smoke 示例

新增：

`templates/examples/gan_loop_feedback.scene.json`

用于验证：

- `loop_arrow` 可表达连续外圈曲线。
- `dashed_region` 可表达上方虚线评价框。
- `dashed_feedback_path` 可表达正交虚线回传路径。

## 后续建议

如果继续修复 `figure2_gan_loop.scene.json` 本身，建议优先做：

1. 删除 `outer_left_loop` / `outer_bottom_loop` / `outer_right_loop` 和两个 detached outer arrows，改成一到两个 `loop_arrow`。
2. 把 `loss_box` 从空 `process_box` 改成 `dashed_region`，保留内部文字为 `text_block`。
3. 把 `loss_to_disc_left` / `loss_to_disc_right` 改成正交 `dashed_feedback_path`。
4. 把底部左右回传路径合并为更少的连续 `dashed_feedback_path`，只有原图明确显示多根并行上箭头时才保留多根。
5. 统一 `Latent Vector`、`Generator`、`Discriminator`、`Real/Generated` 的字体角色，避免同一图里 Times 字号漂移过大。
