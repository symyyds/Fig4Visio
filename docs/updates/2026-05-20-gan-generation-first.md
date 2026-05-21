# 2026-05-20 GAN/TFR 一次性生成能力优化

## 背景

之前的 GAN/TFR 复刻问题不是单纯审查不够，而是第一遍建图时就缺少稳定的局部语法。常见失败是：Real/Generated TFR 面板被拆成很多散碎节点，虚线 loss 框被当成空流程框，外圈循环被拆成几段曲线加独立箭头，底部回传箭头彼此独立，公式仍是普通文本。

这些问题后期靠坐标微调很难稳定解决。正确方向是把“能不能画出来”的能力前置到模板、复合组件、验证和自动修复流程里。

## 本次新增能力

### 1. 新增 `tfr_panel`

`tfr_panel` 是 Real/Generated/Reconstructed TFR 面板的可编辑复合组件。它把背景圆角框、标题、副标题、TFR 网格、Input 标签和可选内部箭头放到一个语义节点里。

这样可以避免第一遍生成时出现：

- 网格和 Input 标签间距太小。
- 虚线反馈路径穿过 Input 标签。
- 内部 Input 箭头被误当成外部拓扑。
- Real/Generated 两侧网格大小和 y 位置不一致。

### 2. 新增 `loss_region`

`loss_region` 用于 GAN/TFR 图里的虚线 loss/evaluation 框。它把虚线边框、标题和公式内容作为一个局部系统渲染，而不是用一个空 `process_box` 假装虚线框。

适用场景：

- `Forward Reconstruction -> Discriminator Evaluation`
- `Adversarial Loss L_adv`
- `Reconstruction Loss L_rec`
- `Gradient Penalty GP`

### 3. 新增 `gan-tfr` 首版模板

新增模板：

`templates/examples/gan_tfr_full.scene.json`

这个模板直接使用：

- `tfr_panel`
- `loss_region`
- `math_text`
- `loop_arrow`
- `dashed_feedback_path`
- `merge_bus`
- `bundle_id`
- `page_background`

后续遇到 GAN/TFR 训练循环图，可以先用：

```powershell
python scripts/image_to_scene.py --image <source.png> --template gan-tfr --output <scene.json>
```

它的目标是让第一遍 scene 就拥有正确拓扑，而不是后面再反复修碎箭头、碎标签和虚线穿字。

### 4. 新增 `scene_autofix.py`

新增脚本：

```powershell
python scripts/scene_autofix.py <scene.json> --recipe gan-tfr --output <fixed.scene.json>
```

当前 `gan-tfr` recipe 会做这些确定性修复：

- 将 split 的 Real/Generated 面板压缩为 `tfr_panel`。
- 将虚线 loss/evaluation 框压缩为 `loss_region`。
- 将原始 `L_adv` / `L_rec` 文本转成 `math_text`。
- 将外圈循环设置为 smooth `loop_arrow` 并绑定更新标签。
- 修复常见的 `Discriminator -> Generated` 反向箭头。
- 将 loss 框的虚线输出改成从边界离开。
- 将底部多根回传箭头加入统一 `bundle_id`。
- 将面板内部 Input 箭头吸收到 `tfr_panel.input_arrow`，避免它被当成外部边。

## 渲染层修复

### 像素坐标内嵌字段缩放

之前 `page.units: "px"` 的场景只缩放了节点本体 `x/y/w/h`，没有缩放 `tfr_panel` 的内部字段，例如 `grid_y`、`input_y`。这会导致 Visio 导出的 PNG 画布被拉得极高，面板内部网格漂移到远处。

现在 `scene_to_visio.py` 会同步缩放：

- `grid_x`
- `grid_y`
- `grid_w`
- `grid_h`
- `input_y`

### `line: none` 优先级修复

之前文本框继承了父级 `line_dash: dash` 后，即使设置了 `line: none`，仍可能渲染出虚线小框。现在 `apply_style` 明确让 `line: none` 压过 dash 样式，避免 `loss_region` 公式周围出现额外虚线框。

## 文档和流程更新

更新了：

- `SKILL.md`
- `references/scene-schema.md`
- `references/visio-component-map.md`
- `docs/workflow.md`
- `sync_to_skill.py`

新的默认思路是：

1. 判断是否是 GAN/TFR 训练循环图。
2. 如果是，先用 `--template gan-tfr`。
3. 对旧 scene 或手写 scene，先跑 `scene_autofix.py --recipe gan-tfr`。
4. 再跑 validate / audit / render。
5. 如果仍有 `[REBUILD]`，重建局部语法，不继续坐标微调。

## 当前边界

这次优化提升的是“一次性生成时可表达正确结构”的能力，不等于任何 GAN/TFR 图都能完全免审查。复杂图仍然需要视觉对比，但第一版应更少出现结构性错误：断裂外圈箭头、反向 discriminator 箭头、虚线穿字、TFR 面板散碎、loss 框假流程框等。
