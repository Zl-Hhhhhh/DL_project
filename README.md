# 对象感知的可控图像拼贴

这是 `PROJECT_PLAN.md` 第一阶段的本地主项目：固定 YOLO 实例分割、等比补边、双色候选、构图规则选择和精确重放。网页输出可选 512×512 或 1024×1024（默认），冻结的开发批次仍使用 512×512。
`CollageRL/` 与 `InstantStyle/` 是独立的外部参考仓库；主项目不导入它们。现在可在根目录 `.venv` 中运行参考仓库，基础环境和参考项目的依赖分别锁定，见 [共享环境说明](docs/SHARED_ENVIRONMENT.md)。

## 本机启动

双击根目录的 `start-ui.cmd`，然后访问 http://127.0.0.1:7860。保持启动窗口运行，关闭窗口即停止服务。

界面提供两个独立的转换模式，可通过“输出尺寸”选择 512×512 或 1024×1024 PNG。原图画布、风格化和 CollageRL 后处理均按所选尺寸运行；切换尺寸后需要重新生成，识别框也会重新计算：

- **拼接**：上传原图，选择双色配色或“保持原色”，搭配左/中/右构图。
- **风格化**：分别上传原图和风格图，调整风格强度与结构保持强度；可选填英文画面描述。使用 InstantStyle + ControlNet，保留结构的同时改变视觉风格，局部内容可能变化。

两种模式均可勾选默认关闭的 **CollageRL 材质重构**，在转换完成后重构整张结果。素材来源可选：

- **使用内置素材**（默认）：使用 CollageRL 官方 README 推荐的 UCI 报纸与杂志素材库，无需上传。完整库有 101 张页面，预选固定 8 张用于当前低内存推理。
- **自定义上传**：上传 1–8 张素材，每张不超过 4 MB。切换来源时会保留已选的自定义图片。

后处理会改变整张画面；原转换结果另存为 `base_output.png`。拼接与风格化不会串联，CollageRL 仅在两者之一完成后运行。内置素材来源、许可和准备方式见 [素材库说明](docs/MATERIALS.md)。

生成后可对比原图并下载；图片仅在本机处理，输入、素材、模型日志和结果保存在 `experiments/ui/`。低显存风格化和材质重构可能需要数分钟，页面显示当前阶段。

拼接模式在关闭 CollageRL 时保留原图背景，仅用浅米色遮住主体原来的掩码区域，再放置编辑后的主体。除旧主体掩码和新主体覆盖的位置外，其余像素与所选尺寸等比补边后的原图一致；不会填掉整个主体外接矩形。旧版命令行开发批次继续使用原来的纯色背景配置。

原图和风格图各限 12 MB、2500 万像素以内的静态 JPG/PNG/WebP。拼接模式中主体触边时会提示，可明确选择使用当前可见主体继续生成。

勾选“显示识别框”后，可在原图预览中查看置信度严格高于所选阈值的矩形框，阈值范围为 1%–100%，默认 50%。不显示类别、分数或掩码，拖动滑块即时筛选。识别框仅用于预览，不写入下载图片；此阈值不改变生成时的主体选择规则。

终端启动界面：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m object_collage.cli ui
```

端口占用时添加 `--port 7861`，并访问对应端口。

## 命令行批量生成

在 `D:\mystudy\DSAA2012\project` 中执行：

```powershell
.\.venv\Scripts\object-collage.exe doctor
.\.venv\Scripts\object-collage.exe run --run-id my_dev_run
```

运行 ID 必须未使用。默认读取 `data/dev/manifest.json` 和 `configs/dev.yaml`。
打开 `experiments/my_dev_run/index.html` 查看所有输入、掩码、候选、输出和失败；页面可离线打开，无需服务器。

单张图、指定类别和构图：

```powershell
.\.venv\Scripts\object-collage.exe run --image D:/path/to/photo.jpg --category person --composition left
```

不传 `--category` 时选择所有类别中面积最大的高置信度实例。看过 `detections.png` 后，可加 `--instance 0` 显式选中另一个检测实例。
默认拒绝主体触边；仅在确实希望保留截断对象时添加 `--allow-truncated`，该决定会记入日志。
`--device cpu` 可显式切换 CPU，不会无提示地回退。

## 重放与汇总

```powershell
.\.venv\Scripts\object-collage.exe replay experiments/my_dev_run/某个源图ID/params.json --output experiments/replay.png
.\.venv\Scripts\object-collage.exe report experiments/my_dev_run --output reports/my_dev_run.html
.\.venv\Scripts\python.exe -m pytest -q
```

重放使用保存的画布和 mask，检查输入和输出 SHA-256，无需重新调用分割模型；已存在的输出不会覆盖。
图层参数、候选 ID、模型哈希、坐标映射、版本、失败原因与分阶段计时均保留。

## 重建

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1 -Python D:/anaconda/envs/espnet_env/python.exe
.\.venv\Scripts\python.exe scripts/prepare_dev.py
```

需要 Python 3.10 和 `uv`。首次下载 PyTorch GPU wheel、YOLO 权重，以及约 253 MB COCO 标注压缩包与选定的 12 张图；不下载完整 COCO 图片集。
本机 venv 借用已有 Conda 环境的 Python 3.10 解释器，包安装在根目录 `.venv` 中。不要删除该基础解释器；迁移机器时重新建立 venv。
完整实装版本保存在 `requirements-lock.txt`，重建脚本优先使用它。基本依赖记录在 `requirements.txt`。
若存在官方 PyTorch wheel 缓存，脚本验证 SHA-256 后复用；没有缓存时会从官方索引下载。安装完成后可删除 `.cache/wheels/` 中的安装包以节省空间，不影响已安装的 `.venv`。

根目录 VS Code 配置使用主项目 `.venv`；直接打开 `CollageRL` 时仍使用其独立 `.venv`。
更换配置后需要新建终端；若已有手动解释器缓存，请用 `Python: Select Interpreter` 选择根目录 `.venv/Scripts/python.exe`。

## 目录与阶段边界

| 目录 | 用途 |
|---|---|
| `data/dev` | 冻结的开发来源、许可、类别与 dev 划分 |
| `configs` | 模型、阈值、种子、候选和调色参数 |
| `src/object_collage` | 分割、渲染、重放、日志及 CLI |
| `models` | YOLO 权重与来源/哈希记录 |
| `experiments` | 每次独立运行及失败记录 |
| `reports` | 可重建的汇总报告 |
| `docs` | 部署实测、方法边界和数据说明 |

当前提供双色效果、原色主体拼贴及固定规则选择。切片、重复拼贴、DINOv2 特征缓存、偏好标注、排序训练、LaMa 摄影修复尚未实现，不把这些写成已部署能力。
模型与公共数据的已有能力不代表本项目的研究贡献；开发集的自动约束通过也不等同于人工确认的掩码或美学质量。
下一步按计划核验掩码、选定有出处的风格示例板，再实现切片效果。

## GitHub 仓库

参考仓库的固定版本及本地适配保存在 [third_party](third_party/README.md)。模型权重、虚拟环境、缓存和运行输出通过准备脚本在本机重建。
