# 2026-05-21 字体复刻与本机字体 fallback 优化

## 背景

Visio 复刻图的文字问题主要有两类：

- 源图使用多种字体，但 scene 只继承一个全局默认字体，导致整体看起来“像”，细节却不对。
- 字体本机其实有，但 scene 没选对；或者字体本机没有，Visio 静默替换，导致字宽、换行、对齐都漂移。

这类问题不能只靠后期挪文本框解决。字体会改变文字宽度和视觉重心，应该在渲染前作为风格语法处理。

## 本机字体识别

新增：

```powershell
python scripts/font_inventory.py
```

它从 Windows 字体注册表读取当前机器可见字体，并给出 role fallback 检查。本机测试结果：

- base font families: 170
- registry font entries: 284
- `Times New Roman`: installed
- `Cambria Math`: installed
- `Calibri`: installed
- `Microsoft YaHei UI`: installed
- `Aptos`: not installed
- `Helvetica`: mapped to `Arial`

## 新增字体语法

节点/样式现在可以表达三层字体意图：

```json
{
  "style": {
    "source_font_family": "Calibri",
    "font_family": "Calibri",
    "font_family_candidates": ["Calibri", "Arial", "Segoe UI"],
    "font_role": "ui_sans"
  }
}
```

- `source_font_family`: 记录源图字体，供 audit 判断“字体有但没选对”。
- `font_family`: 首选渲染字体。
- `font_family_candidates`: 首选字体不可用时的相似候选。
- `font_role`: 当无法确定字体名时的视觉类别。

支持的 role：

- `paper_serif`
- `serif`
- `ui_sans`
- `sans`
- `math`
- `mono`
- `cjk_sans`
- `cjk_serif`

## 渲染层优化

新增 `scripts/font_utils.py`，由 renderer、validate、audit 共用：

- 读取本机字体库存。
- 处理常见别名，例如 `Helvetica -> Arial`、`Times -> Times New Roman`。
- 处理 `Cambria & Cambria Math` 这类 Windows 注册表名称。
- 按字体 role 选择最接近的本机 fallback。
- 对中文文本优先选择 CJK 字体，避免 Visio 静默替换造成字宽漂移。

`scene_to_visio.py` 现在不会盲目把 `font_family` 写进 Visio，而是先解析为本机可用字体再写入 `Char.Font`。

## 审查层优化

`scene_validate.py` 新增字体 warning：

- 请求字体不存在。
- 记录了 `source_font_family`，但本机没有。
- `source_font_family` 本机存在，但实际渲染字体不同。
- 中文文本解析到非 CJK 字体。

`scene_audit.py` 新增 `Typography Review`：

- 汇总当前 scene 实际解析到的字体。
- 列出字体 fallback 和源字体不匹配问题。
- 对“源字体已安装但 scene 没用上”的情况标为 `[REBUILD]`，因为这属于可修复的复刻错误。

## 使用建议

当源图字体较复杂时，先做字体预检：

```powershell
python scripts/font_inventory.py --check "Times New Roman" --check "Cambria Math" --check "Calibri" --check "Microsoft YaHei UI"
python scripts/scene_audit.py <scene.json>
```

如果看不出精确字体名，也至少要区分：

- 论文图常见 Times/Cambria 类 serif。
- 软件或产品图常见 Calibri/Arial/Segoe 类 sans。
- 公式和符号使用 math role。
- 中文图使用 CJK role。

这不能保证 100% 复刻所有图片字体，但能明显减少“字体明明存在却没选中”和“字体不存在导致 Visio 偷偷替换”的问题。
