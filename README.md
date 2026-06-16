# Visiomaster

Visiomaster 是一个 Windows 优先的图片转可编辑 Visio 工具。它的目标不是把原图整张贴进 Visio，而是把流程图、架构图、论文模块图中的框、线、文字、图标和小模块拆成可编辑的 Visio 对象，并输出 `.vsdx`、`.png`、`.svg`。

当前版本包含一个简化 GUI：上传图片后自动处理、自动截图自检，通过后才允许下载 Visio 文件。GUI 默认禁用原图嵌入和局部图片贴片，优先生成可编辑形状、线段、文字和图标矢量部件。

## 主要功能

- 上传 PNG/JPG/JPEG/BMP/WEBP 图片。
- 自动重建为可编辑 Visio `.vsdx`。
- 同时导出预览 `.png` 和 `.svg`。
- 自动截图自检：比较源图和 Visio 导出图，失败时自动换策略重跑。
- 防贴图检查：扫描 `scene.json` 和 `.vsdx`，发现 `image_tile`、`assets`、`ForeignData`、`/media/` 等图片嵌入时禁止下载。
- 图标矢量复现：云、数据库、用户、搜索、设备、模块图标等非传统流程图形状会尽量拆成 `polygon_node` 和 `line_segment`，保持可编辑。
- 批量验收：可对文件夹内图片运行同 GUI 工作流的批量自检。

## 当前质量边界

这个项目提供的是“可编辑模块复现工作流”，不是任意图片的像素级完美复刻器。

已实现的硬性门槛：

- 不允许整张原图嵌入 Visio。
- GUI 默认不生成 raster tile。
- 生成后必须导出 PNG 截图并跑自检。
- 截图自检失败会自动尝试 `standard -> vector_trace -> vector_trace_dense`。
- 只有自检通过并且确认没有图片嵌入时，GUI 才允许下载。

对复杂论文图、小字号公式、照片、热力图、真实截图类输入，系统会尽量拆成可编辑对象，但局部细节可能是粗略矢量重建。

## 环境要求

- Windows
- Microsoft Visio 桌面版
- Python 3.10+
- Git

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

主要依赖：

- `pywin32`：调用 Microsoft Visio COM 自动化
- `Pillow`：图片读取和预览
- `opencv-python` / `numpy`：图像分析、边缘检测、自检
- `rapidocr-onnxruntime`：OCR 文字检测
- `pyinstaller`：打包 GUI EXE

## GUI 使用

直接运行源码 GUI：

```powershell
python gui_app.py
```

操作流程：

1. 点击上传图片。
2. 点击开始处理。
3. 等待系统自动生成 Visio、导出截图并自检。
4. 自检通过后下载 `.vsdx`、`.png`、`.svg`。

GUI 输出保存在：

```text
work/gui_runs/<timestamp>/
```

每次运行会包含：

- `source/original.*`：固定后的源图
- `attempt_*/<name>.scene.json`：可编辑场景描述
- `attempt_*/exports/*.vsdx|*.png|*.svg`：Visio 输出
- `attempt_*/self_check/self_check.json`：自检指标
- `attempt_*/self_check/self_check_comparison.png`：源图、结果图、差异热图
- `attempt_*/quality/quality_report.md`：质量报告

## 打包 EXE

运行：

```powershell
.\build_exe.ps1
```

生成文件：

```text
dist/VisiomasterGUI.exe
```

注意：`dist/` 和生成的 `.exe` 默认不提交到 Git 仓库。GitHub 普通 Git 仓库单文件限制为 100MB，当前 EXE 超过该限制。需要直接下载 EXE 时，请在本仓库的 GitHub Releases 中下载 `VisiomasterGUI.exe`；也可以本地运行 `build_exe.ps1` 重新打包。

## CLI 使用

从图片生成可编辑 scene：

```powershell
python scripts\image_auto_scene.py `
  --image path\to\input.png `
  --output work\demo\scene.json `
  --disable-raster-tiles `
  --mode standard `
  --overwrite
```

渲染到 Visio：

```powershell
python scripts\scene_to_visio.py work\demo\scene.json `
  --output-dir work\demo\exports `
  --basename demo
```

截图自检：

```powershell
python scripts\self_check.py `
  --source path\to\input.png `
  --replica work\demo\exports\demo.png `
  --output-json work\demo\self_check.json `
  --output-png work\demo\self_check_comparison.png
```

## 批量验收

对默认素材目录运行 GUI 同款工作流：

```powershell
python scripts\batch_workflow_check.py
```

指定目录：

```powershell
python scripts\batch_workflow_check.py `
  --source-dir "C:\path\to\images" `
  --batch-dir work\workflow_check\manual_run
```

输出：

- `summary.json`：每张图是否通过、分数、是否可下载、是否有图片嵌入
- `batch_workflow_check.log`：批量运行日志
- `self_check_contact_sheet.png`：所有自检对比图拼图

## 项目结构

```text
gui_app.py                       GUI 主程序
build_exe.ps1                    EXE 打包脚本
VisiomasterGUI.spec              PyInstaller 配置
requirements.txt                 Python 依赖
scripts/image_auto_scene.py      图片转可编辑 scene
scripts/scene_to_visio.py        scene 渲染为 Visio
scripts/self_check.py            截图自检
scripts/batch_workflow_check.py  批量 GUI 工作流验收
scripts/scene_validate.py        scene 结构校验
scripts/scene_audit.py           模块审计
references/                      scene 规范和审查说明
templates/                       组件和样式模板
tests/                           自动测试
```

## 测试

```powershell
python -m pytest tests\test_public_release_smoke.py -q
python gui_app.py --smoke
dist\VisiomasterGUI.exe --smoke
```

测试覆盖内容包括：

- 基础 scene 校验
- rounded orthogonal connector
- review bundle/checklist gate
- arrow plan gate
- 图标类元素矢量复现
- GUI smoke

## GitHub 同步

当前公开仓库：

```text
https://github.com/symyyds/Visiomaster-GUI
```

日常更新流程：

```powershell
git status
python -m pytest tests\test_public_release_smoke.py -q
git add .
git commit -m "Describe update"
git push origin main
```

后续在本项目中继续修改代码时，应同步更新 README、requirements、测试结果，并推送到 GitHub。生成目录如 `.venv/`、`build/`、`dist/`、`work/`、`exports/` 不进入 Git 仓库。

大文件说明：如果后续需要把大型构建产物直接纳入 Git 工作流，可以使用 Git LFS；当前公开发布的 EXE 使用 GitHub Releases 分发。

## License

见 [LICENSE](LICENSE)。
