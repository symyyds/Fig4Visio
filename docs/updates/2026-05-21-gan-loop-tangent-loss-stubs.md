# 2026-05-21 GAN/TFR 曲线末端与损失框虚线路由优化

## 背景

最新版 GAN/TFR 复刻图整体已经接近参考图，但仍暴露出两个能力层问题：

- 外围椭圆循环到了箭头末端仍可能不顺滑，箭头方向像是被最后一个采样点硬带过去。
- 中间 `Forward Reconstruction -> Discriminator Evaluation` 这类损失/评价区域，虚线输出容易被画成从虚线框角落到目标左右侧的 L 形路径，看起来像“额外方框加额外箭头”。

这不是单纯坐标没调好，而是 scene 语法没有把“曲线末端切线”和“loss frame 到重叠目标的短 stub”表达成硬约束。

## 本次修改

### 1. 曲线箭头末端切线

- `scene_to_visio.py` 支持 `start_tangent_point` 和 `end_tangent_point`。
- `loop_arrow` 默认 `smooth_samples` 提高到 14，让外圈曲线预览更细腻。
- `scene_autofix.py --recipe gan-tfr` 会给缺少末端切线点的 outer loop 自动补 `end_tangent_point`。
- `scene_validate.py` 和 `scene_audit.py --fail-on-rebuild` 会检查 outer loop 是否缺少 `end_tangent_point`，以及箭头末端转角是否过大。

### 2. loss_region 到目标组件的短 stub 规则

- `scene_autofix.py` 现在能识别 `to: discriminator:left@...` 这种节点端点，不再只处理 `to_point`。
- 当 `loss_region` 与目标组件上下相邻且水平重叠时，recipe 会把 L 形侧边路径改成短竖直边界 stub，例如：

```json
{
  "type": "dashed_feedback_path",
  "from_point": [420, 266],
  "to": "discriminator:top@0.36",
  "route": "vertical"
}
```

- validate/audit 会把 `loss_region -> overlapping target` 的侧边 L 形虚线路由标为需要重建，避免它在视觉上变成额外虚线框。

### 3. GAN/TFR 模板更新

- `templates/examples/gan_tfr_full.scene.json` 的外圈循环添加 `end_tangent_point`。
- 中间 `adv_loss_region` 上移，避免与 `Discriminator` 贴边/重叠。
- `adv_loss_to_disc_left/right` 改为短竖直虚线 stub，直接进入 `discriminator:top@ratio`。
- `templates/examples/gan_loop_feedback.scene.json` 也补了外圈末端切线点。

### 4. math_text 紧凑度

- `draw_math_text` 的片段宽度计算改为“显示框宽度”和“游标推进宽度”分离：`subscript_box_pad_in` 保证 `adv/rec` 不换行，`subscript_pad_in` 控制实际紧凑度，减少 `L_adv`、`L_rec` 被渲染成 `L  adv` 或竖排的风险。
- `templates/visio_components.json` 和 `templates/style_profiles.json` 增加了对应默认值。

## 后续审查重点

- 如果箭头末端仍不自然，先改 `end_tangent_point` 和邻近采样点，不要先移动其他模块。
- 如果中间虚线看起来像多了一个框，优先检查 loss/evaluation 区域是否用了角落到左右侧的 L 形路径。
- 对 GAN/TFR 图，继续把 `scene_audit.py --fail-on-rebuild` 作为渲染后的硬门槛。
