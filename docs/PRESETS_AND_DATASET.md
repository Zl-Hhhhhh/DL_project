# 具名预设与扩充数据集

2026 年 10 月 27 日。本轮提供确定性的颜色与构图预设，不调用扩散风格化或 CollageRL 推理。

## 界面使用

启动 `start-ui.cmd`，在「拼接 → 风格预设」中选择效果：

| 预设 | 处理方式 | 控制 |
| --- | --- | --- |
| 暖色胶片 | 柔和色彩、轻微暖色与暗部提升，保持构图 | 强度 0% 至 100%，0% 为原图画布 |
| 低饱和复古 | 降低饱和度与对比度、抬起暗部 | 强度 0% 至 100%，0% 为原图画布 |
| 主体突出 | 主体颜色与位置不变，降低背景饱和度 | 强度越高，背景越接近灰度 |
| 简洁海报 | 原色主体放置在米白背景上 | 留白程度 0% 至 100%，主体最长边从画布的 60% 降到 30%；支持居中与四个三分位置 |

暖色胶片与复古滤镜无需检测到主体；另外两项使用分割掩码。预设支持 512×512 和 1024×1024，保留原有原图拼接、主体保色和纯色拼贴入口。缩略图来自同一张训练样例，不是对上传图片的实时预览。

这些是本项目实现的基础视觉预设，尚未完成“网络热门程度”的参考整理，也没有宣称经过训练或证明更好看。界面交互参考 [Lightroom 的缩略图与 Amount 控制](https://helpx.adobe.com/lightroom/desktop/edit-photos/presets.html) 和 [Google Photos 的滤镜强度控制](https://support.google.com/photos/answer/6128850)。原有 CSS、字体、布局与键盘交互继续使用。

生成任务的 `params.json` 保存预设版本、强度、颜色参数、构图和输入／输出哈希；已有 `replay` 命令支持 `photo-presets-v1`。保色承诺针对基础输出，选择后处理后需单独评价最终结果。

## 数据集

新增数据集位于 `data/photo120`，已冻结 120 张：10 类各 12 张，每类分为 8 张训练、2 张验证、2 张测试，总划分为 80／20／20。类别为人物、猫、狗、马、椅子、公交车、碗、笔记本电脑、伞、大象；依据许可与标注条件下的候选数量确定，不依据生成效果选择。原有 12 张开发图保持独立。新增原图共 20,425,550 字节，约 19.5 MiB。

来源为 [COCO 2017](https://cocodataset.org/#download) 的官方图片与实例标注。默认从 `val2017` 来源中选择许可为 CC BY／CC BY-SA、主体不接触边缘、非 crowd、面积与框大小满足条件的照片。这里的训练／验证／测试是本项目自己的划分，与 COCO 官方名称不同。

随机种子固定为 `20261027`，选择发生在模型推理之前。原图 ID、SHA-256、64 位 dHash、类别、许可及原始 Flickr 链接均进入清单。排除原开发集 ID／哈希和近重复图像，dHash 距离不超过 4 的候选被拒绝。这是近重复筛查，不能保证发现所有同场景拍摄组。使用 COCO 预训练模型时仍需披露预训练重叠可能。

文件用途：

- `manifest.json`：冻结的原图清单和本项目划分。
- `summary.json`：图片数量、类别、划分、空间与重复筛查统计。
- `selection.json`：在推理前固定的候选及备用图片，备用项不是正式数据集。
- `ATTRIBUTION.md`：来源与许可链接；衍生图片应保留适用的署名与相同方式共享要求。
- `images/`：下载图片，已从 Git 跟踪中排除，迁移 HPC 时需要单独传输。

官方标注通过 ZIP 字节范围读取，只下载所需实例标注成员，并校验 ZIP CRC32、记录解压 JSON 的 SHA-256。无需下载整套图片或解压完整标注包。

恢复与校验：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/prepare_photo_dataset.py
```

已有清单时按原 ID 补齐和检查哈希，不重新抽样。在需要更大训练集时，可选择 COCO train2017 和新的输出目录，例如 `--coco-split train2017 --per-category 120 --output data/photo1200`；脚本会先检查许可与数量是否足够。不要覆盖第一版冻结数据。

## 固定对比

从训练部分按类别和原图 ID 固定选择 12 张，在加载检测器之前保存源图清单。每张提供原图、暖色胶片、低饱和复古、主体突出、居中海报和三分海报六个槽位，统一使用 70% 强度／留白。无法提取可靠主体时保留失败说明，不按结果好坏替换输入。

本轮结果目录为 `experiments/preset_study_20261027_512`，打开其中 `index.html` 浏览；各图保存来源、分割记录、参数、输出与重放检查。所有验证／测试图保持预留，本轮没有偏好标注或审美评分。

本轮 12 张训练图共有 72 个展示槽位：12 个原图对照、54 个实际编辑、6 个未生成方案。后者来自两张被检测为主体截断的照片。54 个实际编辑均通过参数重放和 PNG 哈希检查。界面缩略图优先使用这组样例中首个可用的人物照片，仅用于展示，不改变固定对比的选图或结果。

验证记录：48 项轻量测试通过，4 项与风格化有关的测试未运行。真实浏览器完成四个预设 × 两种尺寸的生成与下载检查，下载 PNG 的尺寸和哈希均匹配；检查了强度变化后结果失效、预设与构图的键盘切换，以及 375／320 像素宽度无横向溢出。证据保存在 `.cache/ui-verification/named-presets`。颜色与构图控制通过不等于审美质量合格，分割缺口与被遮挡主体仍会影响海报效果。

重新生成时指定新的输出目录：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/run_preset_study.py --output experiments/presets_repeat --size 512
```

## HPC：UniScheduler 与 NVIDIA A800-SXM4-80GB

2026 年 10 月 28 日更新：集群使用 UniScheduler，GPU 改为 **NVIDIA A800-SXM4-80GB**，运行环境为 **`pytorch:py310-2.1.0`**。在学校的作业模板中选择该镜像与 GPU 资源；提交命令、GPU 队列和账户参数沿用学校模板。

该环境目标为 Python 3.10、PyTorch 2.1.0，torchvision 使用官方配套的 0.16.0；官方 CUDA 构建包含 11.8 和 12.1，具体以镜像内的实际版本为准，见 [PyTorch 历史版本说明](https://pytorch.org/get-started/previous-versions/#v210)。安装入口与本机分开，防止直接使用 Windows 锁定文件将集群 PyTorch 改为 2.5.1。

在镜像环境中、项目根目录安装依赖：

```bash
python -m pip install -r requirements-hpc.txt
python -m pip check
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/hpc/check_environment.py --output .cache/hpc-environment.json
```

`configs/hpc-constraints.txt` 固定 torch 2.1.0、torchvision 0.16.0、NumPy 1.26.4 与 OpenCV 4.10.0.84。图像已提供匹配的 torch／torchvision 时会复用；安装输出若提示替换框架，应先核对镜像实际版本和 CUDA 构建。不要执行本机的 `setup.ps1` 或安装 `requirements-lock.txt` 来覆盖集群环境。

启动检查验证 Linux、Python／PyTorch／torchvision 版本、NumPy 桥接、CUDA 矩阵乘法、torchvision 的 CUDA NMS，以及 A800-SXM4-80GB 型号和可见设备的总显存规格；总显存不等于当前空闲显存。任何关键检查失败都会退出，批量任务不会继续。镜像标签在 UniScheduler 中选择，程序校验其内部可观察的实际运行环境。

从本机迁移源码、配置、脚本、`data/photo120`（含图片）和 `models/yolov8n-seg.pt` 及其 JSON 来源记录。无需传输本机 `.venv`、缓存和扩散模型权重。激活集群环境，在项目根目录、已获分配的 GPU 作业中执行：

```bash
bash scripts/hpc/run_preset_study.sh
```

该脚本先执行上述环境检查、保存报告，再校验数据并进行训练样例上的批量生成。它不是模型训练任务，也不会使用验证／测试图。可设置 `PRESET_PYTHON`、`PRESET_SIZE` 和 `PRESET_RUN_ID` 来选择解释器、尺寸与输出目录。

之后可以用同一份划分加入偏好标注和小模型训练，再考虑更大的数据集或模型。本轮只准备了可迁移处理流程，尚未连接或提交学校集群任务，也未在目标镜像或 A800 上实际运行。
