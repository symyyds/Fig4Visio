# 2026-05-21 GAN/TFR 一次生成能力与硬门禁优化

## 背景

这轮问题的根因不是单个坐标没有调准，而是生成阶段仍可能绕过语义组件：

- 外圈循环可能用普通线段或缺少末端切线的 `loop_arrow`，导致箭头末端有硬折感。
- 中间 loss/evaluation 区域可能被拆成虚线框、标题、公式和零散虚线箭头，结果虚线穿过标题，或在 `Discriminator` 两侧形成假框。
- `Ladv` / `Lrec` 这类紧凑写法没有进入 `math_text` 的下标渲染路径。
- 底部 backprop 可能由多条无关联短虚线组成，出现孤立箭头、误指 TFR 输入面板、竖直箭头间距不稳定。
- 审查阶段能看出问题，但如果渲染前没有硬门禁，坏语法仍会导出成看似接近但局部错误的 PNG。

所以本次重点不是继续微调某一张图，而是提高“第一遍就进入正确组件路径”的能力，并让错误 scene 无法静默导出。

## 本次修改

### 1. 渲染前自动 GAN/TFR autofix

`scripts/scene_to_visio.py` 现在会在 GAN/TFR 场景渲染前默认执行一次确定性修复：

- 识别并压缩 Real/Generated TFR 面板为 `tfr_panel`。
- 将可压缩的 dashed loss/evaluation 子系统升级为 `loss_region`。
- 将 dashed/loss/backprop 语义的普通 `arrow_connector`、`dynamic_connector` 或带箭头 `line_segment` 转为 `dashed_feedback_path`。
- 将 `loss_region -> Discriminator` 的左右侧 L 形虚线改为短竖直 stub。
- 修正常见的 `Discriminator -> Generated` 方向错误。
- 给 crowded bottom backprop arrows 补 `bundle_id`。
- 归一化 `Ladv`、`Lrec`、`L adv`、`L rec` 为 `L_adv` / `L_rec`。

如果发生改写，渲染器会在输出目录写出：

```text
<basename>.autofixed.scene.json
```

后续调试应优先检查这个文件。只有需要观察原始坏 scene 时才使用 `--no-autofix`。

### 2. loss_region 标题保护

`loss_region` 默认标题布局改为更稳的 inside/protected 方案：

- 长标题可拆成多行。
- 标题与公式之间保留保护间距。
- 虚线框不应该穿过标题。
- `scene_validate.py` 会检查过宽标题、未保护标题和紧凑 loss 公式。

这主要对应 `Forward Reconstruction -> Discriminator Evaluation` 一类区域，避免标题既不像框内标题，也不像规范框外标题。

### 3. 虚线反馈碎片硬拦截

`scene_validate.py` 和 `scene_audit.py --fail-on-rebuild` 增加了更强的判定：

- dashed/loss/backprop 语义不能用普通 connector 逃过 `dashed_feedback_path`。
- 带箭头的短 dashed `line_segment` 会被视为重建问题。
- GAN/TFR 中 feedback/backprop 不能误指 Real/Generated TFR input panel。
- `dashed_feedback_path` 不能穿过 `text_block` 或 `math_text`，除非显式声明允许。
- bottom loss system 如果由过多无关联竖直虚线组成，会要求用 `merge_bus` / `junction_point` / `bundle_id` 统一。

### 4. 外圈 loop_arrow 审查增强

外圈更新循环继续使用 `loop_arrow`，并要求：

- 使用 `curve_mode: "smooth"`。
- 有足够采样点。
- 关键场景包含 `end_tangent_point`。
- 不贴住导出边界。
- 不压住绑定的 loop label，例如 `Alternating Updates (G & D)`。

如果仍然出现硬折或压字，审查会要求重建局部 loop，而不是继续移动无关模块。

### 5. 渲染硬门禁

`scene_to_visio.py` 在 exact 或 GAN/TFR 场景中会自动运行：

```powershell
scene_validate.py --strict
scene_audit.py --fail-on-rebuild
```

如果还有 `[REBUILD]` 项，渲染会停止在 Visio 打开之前。这能阻止“整体看起来差不多，但局部语法已经错了”的结果继续流入 PNG/SVG。

## 使用建议

GAN/TFR 这类图优先从模板开始：

```powershell
python scripts/image_to_scene.py --image <source.png> --template gan-tfr --output <scene.json>
python scripts/scene_to_visio.py <scene.json> --output-dir <exports>
```

如果导出目录出现 `<basename>.autofixed.scene.json`，说明渲染前自动修过局部语法；后续应该以这个文件作为调试对象。

如果渲染被 gate 拦住，先看 audit 里的 `[REBUILD]` 项。此时不要继续调坐标，应重建对应局部组件：外圈 loop、loss_region、dashed_feedback_path、bottom backprop bus。

## 验证重点

- 规范 GAN/TFR 模板能通过 strict validate 和 rebuild audit。
- 人为构造的坏 scene 会被 audit 标出 compact formula、虚线碎片、错误 feedback target、loop 边界/切线等问题。
- 坏 scene 直接渲染会先尝试 autofix；仍存在无法规则修复的问题时，render gate 会返回失败，不导出误导性的最终 PNG。
