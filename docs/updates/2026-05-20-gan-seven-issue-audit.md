# 2026-05-20 GAN/TFR 七类细节问题审查优化

## 背景

最新一次 GAN/TFR 图复刻已经不再是“完全贴错组件”的问题，而是整体像、局部细节仍不对：

1. `Generated` 和 `Discriminator` 的主箭头方向反了。
2. 外圈循环箭头虽然成形，但仍像折线装饰边框，不像训练更新流程。
3. 中间虚线评价框内部有多余虚线/短线。
4. 底部 `Loss Backpropagation` 回传虚线拥挤，缺少统一线束语法。
5. `L_adv` / `L_rec` 公式仍是 raw underscore 文本。
6. Real/Generated TFR 小面板内网格和 `Input` 标签间距太紧。
7. 整张图在全局相似后继续微调收益很低，必须把这些细节变成脚本闸门。

## 本次修改

### 1. GAN/TFR 方向检查

`scene_validate.py` 和 `scene_audit.py` 现在会识别 GAN/TFR 场景中的 `Discriminator -> Generated` 主箭头，并提示应重建为 `Generated/Reconstructed TFR -> Discriminator`。

### 2. 外圈循环箭头语义检查

外圈 `loop_arrow` 新增审查：

- 建议使用 `curve_mode: "smooth"`。
- 建议加 `semantic_role: "outer_update_loop"`。
- 建议用 `label_id` / `loop_label_id` 绑定底部 update 标签。

这样能避免外圈曲线被当作装饰边框。

### 3. 虚线评价框洁净度检查

`dashed_feedback_path` 如果穿过 `dashed_region` 内部，会被标记为 `[REBUILD]`。正确做法是让路径从边界点或 `boundary_port` 离开，评价框内部只保留标题和公式。

### 4. 底部回传线束检查

对于三根以上平行竖向 dashed feedback arrows 指向同一个 Discriminator 的情况，审查器会要求改成共享 `merge_bus` / `junction_point`，并使用 `bundle_id` 组织线束。

### 5. 新增 `math_text`

新增 `math_text` 节点类型，用于 `L_adv`、`L_rec` 这类短公式。渲染器会把 `L_adv` 拆成 `L` 和较小、下移的 `adv` 片段。精确复刻时，普通 `text_block` 中出现 raw underscore loss notation 会被标记为 `[REBUILD]`。

### 6. TFR 小面板间距检查

新增 Real/Generated TFR 局部审查：

- 两个 `grid_matrix` 应保持相同尺寸和 y 位置。
- 网格下方到 `Input` 标签需要留出清晰间距。
- 不能让网格、标签和回传箭头挤在同一条局部区域里。

### 7. Smoke 示例同步

`templates/examples/gan_loop_feedback.scene.json` 已更新为：

- `math_text` loss label
- smooth outer `loop_arrow`
- semantic loop role / label binding
- clean dashed evaluation region

## 测试结果

正确 smoke scene：

```powershell
python scripts/scene_validate.py templates/examples/gan_loop_feedback.scene.json --strict
python scripts/scene_audit.py templates/examples/gan_loop_feedback.scene.json --fail-on-rebuild
```

结果：验证通过，审查无 `[REBUILD]`。

当前失败 GAN/TFR scene：

```powershell
python scripts/scene_validate.py examples/gan_tfr/gan_tfr_recreate.scene.json --strict
python scripts/scene_audit.py examples/gan_tfr/gan_tfr_recreate.scene.json --fail-on-rebuild
```

结果：审查器返回退出码 `2`，检测到 `10` 个 `[REBUILD]`，覆盖公式 raw underscore、主箭头方向、虚线评价框杂线、虚线穿字、底部回传线束等问题。

## 后续使用规则

遇到“整体已经像，但小问题怎么改都改不好”的 GAN/TFR 复刻时，不要继续全局微调。先让 `scene_audit.py --fail-on-rebuild` 通过，再进入视觉修边。
