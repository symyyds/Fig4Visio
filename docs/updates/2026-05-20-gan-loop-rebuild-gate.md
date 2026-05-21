# 2026-05-20 GAN Loop 复刻失败后的重建闸门优化

## 背景

本次问题来自一次 GAN/TFR 图复刻：初稿很快能画出大体结构，但后续反复审查和微调超过半小时，外圈循环箭头、虚线框、底部反馈虚线、文本穿线等问题仍然没有根本改善。

根因不是缺一个 Visio 组件，而是复刻流程在发现结构性错误后仍继续做坐标微调。错误的局部语法没有被强制替换。

## 失败模式

### 1. 外圈循环被当作椭圆框

失败 scene 使用：

- `ellipse_node` 作为 `outer_loop`
- 两个独立 `arrow_connector` 作为顶部/底部箭头

这会产生“椭圆边框 + 后贴箭头”的观感。原图需要的是训练循环路径，箭头应该属于路径本身。

### 2. 虚线评价框被当作空流程框

失败 scene 使用空 `process_box` 画虚线评价框。这个节点没有流程语义，应该改成 `dashed_region`，内部文字单独放置。

### 3. 虚线反馈路径使用普通箭头

失败 scene 中 loss/backprop 路径大多是带 dash style 的 `arrow_connector`。它们应该是 `dashed_feedback_path`，这样审查器才能按“连续反馈路径”检查是否正交、是否穿字、是否依赖 `allow_diagonal`。

### 4. 虚线穿过文本

底部 feedback 路径穿过 `real_input`、`left_loss_text`、`generated_input`。这类问题不是继续挪字能稳定解决，而是路径语法和避让区域需要重建。

### 5. 背景矩形污染审查

为避免 Visio PNG 导出裁剪，旧 scene 使用空白 `process_box` 作为背景。这会让 route-intersection 检查误以为所有箭头都穿过一个流程节点。

## 本次修改

### 新增 `page_background`

新增节点类型：

```json
{
  "id": "page_background",
  "type": "page_background",
  "x": 0,
  "y": 0,
  "w": 900,
  "h": 575,
  "z": -100
}
```

用途：保留 Visio PNG 导出画布比例。该节点会被语义审查忽略，不再污染路线穿越检查。

### 增强 `scene_validate.py`

新增检查：

- 识别 `page_background` / background role，忽略它的路由穿越。
- 识别 `outer_loop` 这类无文本椭圆作为被动外圈循环框，提示改成 `loop_arrow` / `curved_arrow`。
- 对 dashed/loss/backprop 路径，如果仍使用 `arrow_connector`，提示改成 `dashed_feedback_path`。
- 对 dashed/loss/backprop 路径，检查是否穿过 `text_block`。
- 跳过被动外圈椭圆的普通穿越噪音，避免审查报告被假阳性淹没。

### 增强 `scene_audit.py`

新增 `[REBUILD]` 分类，表示不能继续微调，必须局部重画。

典型输出：

```text
- [ ] [REBUILD] `outer_loop` looks like a passive ellipse used as a training/cycle loop...
- [ ] [REBUILD] `adv_loss_box` is a dashed empty `process_box`...
- [ ] [REBUILD] `left_to_disc_backprop` crosses text node `left_loss_text`...
```

新增参数：

```powershell
python scripts/scene_audit.py <scene.json> --fail-on-rebuild
```

如果存在 `[REBUILD]` 项，脚本返回非零退出码，阻止继续把错误局部当成“只差微调”的问题。

## 测试结果

### 正确 smoke scene

```powershell
python scripts/scene_audit.py templates/examples/gan_loop_feedback.scene.json --fail-on-rebuild
```

结果：退出码 `0`。

### 失败 GAN scene

```powershell
python scripts/scene_audit.py examples/gan_tfr/gan_tfr_recreate.scene.json --fail-on-rebuild
```

结果：退出码 `2`，检测到 `17` 个 `[REBUILD]` 项，包括外圈椭圆、detached loop arrows、空虚线框、普通虚线 feedback arrows、feedback 路径穿字。

## 后续制作规则

以后遇到类似“5 分钟像，30 分钟改不好”的情况：

1. 先运行 `scene_audit.py --fail-on-rebuild`。
2. 只要出现 `[REBUILD]`，停止坐标微调。
3. 按局部系统重建：外圈循环、虚线评价框、底部反馈路径、文本避让。
4. 闸门通过后再进入颜色、字号、位置的微调。
