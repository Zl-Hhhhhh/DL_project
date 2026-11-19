# 本机 Ascend 环境

适用设备：Linux aarch64、Ascend 910、CANN 8.5.0。本机项目环境独立放在根目录 `.venv/`，默认只使用 `npu:3`。主项目分割采用 SAM 2.1 自动掩码；类别字段仅保留为数据来源元信息，不参与选主体。

## 建立环境

在项目根目录执行。系统 Python 3.9 不适合当前依赖；以下使用 uv 下载 Python 3.10，并在项目目录建立隔离环境：

```bash
python3 -m pip install --prefix "$PWD/.cache/bootstrap" uv==0.13.6
export UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"
export UV_CACHE_DIR="$PWD/.cache/uv"
.cache/bootstrap/bin/uv python install 3.10
.cache/bootstrap/bin/uv venv --python 3.10 .venv
source /usr/local/Ascend/cann/set_env.sh
.cache/bootstrap/bin/uv pip install --python .venv/bin/python -r requirements-ascend.txt
.cache/bootstrap/bin/uv pip install --python .venv/bin/python -e .
SAM2_BUILD_CUDA=0 .cache/bootstrap/bin/uv pip install --python .venv/bin/python -e third_party/repos/SAM2
.cache/bootstrap/bin/uv pip install --python .venv/bin/python -r requirements-references.txt
.cache/bootstrap/bin/uv pip check --python .venv/bin/python
```

SAM2 的可选 CUDA 扩展在 Ascend 环境下关闭。参考仓库的依赖与主项目共用此隔离环境。InstantStyle、DragonDiffusion 和 CollageRL 的本地入口均固定使用 NPU 3；覆盖补丁见 [外部仓库说明](../third_party/README.md)。

## 权重与数据

SAM2 的权重文件、固定来源及本机 SHA-256 记录在 `models/sam2/SOURCE.json`。默认使用 tiny，可将 `configs/dev.yaml` 中的模型路径、配置和哈希同时改成 large 版本。重新下载：

```bash
mkdir -p models/sam2
curl -fL --retry 5 -o models/sam2/sam2.1_hiera_tiny.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
curl -fL --retry 5 -o models/sam2/sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
sha256sum models/sam2/*.pt
```

哈希必须与 `models/sam2/SOURCE.json` 一致。DragonDiffusion 的 SD 1.5、IP-Adapter、辅助模型由准备脚本下载并根据发布方元数据验证，完成后生成 `models/dragondiffusion/MANIFEST.json`。InstantStyle 的 SDXL、IP-Adapter 和 ControlNet 由另一脚本下载。Hugging Face 官方站不可达时可给脚本设置可访问的 `HF_ENDPOINT`，例如下方镜像地址；下载中断后重复执行会继续已完成的分块，完整文件最后按发布方哈希核验。

```bash
HF_ENDPOINT=https://hf-mirror.com MODEL_DOWNLOAD_WORKERS=2 .venv/bin/python scripts/prepare_dragondiffusion.py
HF_ENDPOINT=https://hf-mirror.com MODEL_DOWNLOAD_WORKERS=2 .venv/bin/python scripts/prepare_dragondiffusion.py --verify-only
HF_ENDPOINT=https://hf-mirror.com MODEL_DOWNLOAD_WORKERS=2 .venv/bin/python scripts/prepare_instantstyle.py --ranged
.venv/bin/python scripts/prepare_dev.py
.venv/bin/python scripts/prepare_photo_dataset.py
.venv/bin/python scripts/prepare_style_references.py
.venv/bin/python scripts/prepare_materials.py
.venv/bin/python scripts/prepare_material_references.py
```

权重放在 `models/` 下，不放入外部源码仓库；CollageRL 自带的 agent 权重保留在 `third_party/repos/CollageRL/weights/`。内置报纸素材为独立数据集，准备方式见 [素材说明](MATERIALS.md)。

## 运行与范围

```bash
source /usr/local/Ascend/cann/set_env.sh
.venv/bin/python -m object_collage.cli doctor
.venv/bin/python -m pytest -q
.venv/bin/python -m object_collage.cli run --image data/dev/images/000000315663.jpg --run-id sam2_check
.venv/bin/python -m object_collage.cli sequence --image data/dev/images/000000283835.jpg --recipe configs/recipes/subject_color.json --run-id sequence_check
.venv/bin/python -m object_collage.cli ui --port 7860
```

`doctor` 会在 `npu:3` 上做矩阵乘法，默认配置也指向 `npu:3`。`--device cpu` 可用于排查，运行不会自动切换设备。SAM2 自动选面积合适、质量过阈值的内部掩码；若一张图有多个主体，可根据 `detections.png` 用 `--instance` 指定掩码编号。SAM2 不输出类别，`--category` 已禁用。

基础拼接、InstantStyle + ControlNet、DragonDiffusion 主体移动和 CollageRL 材质重构均已在 NPU 3 上通过实际图片的基础推理。顺序执行器保留原图、每一步的图像、掩码、参数和哈希；生成类步骤后重新运行 SAM2。功能通过不等于主体保留或审美质量合格。

本机复核：`uv pip check` 通过，62 项自动测试通过；冻结开发集 12/12 张完成 SAM2 分割并生成 96/96 个候选。网页真实 1024×1024 拼接作业成功，下载 PNG 可读。DragonDiffusion 的 17 个文件共 5,064,527,543 字节已逐项复核；InstantStyle 的 SDXL、IP-Adapter 和 ControlNet 文件已再次通过发布方哈希校验。以上成功率只代表流程与几何约束通过，主体身份和画质仍需人工检查。

风格化失败作业的报错原因为旧入口强制要求 CUDA。改为 NPU 3 后，使用同一原图与风格图经网页 API 完成 1024×1024、20 步生成，输出见 `experiments/ui/e10e3ca46ac646b6bf0a7f8f7e1b2aab/output.png`。`data/photo120/` 的 120 张原图现已全部按清单哈希校验，划分为 80/20/20。内置 UCI 材质库已装齐并验证 101 张，固定风格与材质参考分别位于 `data/style_refs/`、`data/material_refs/`。DragonDiffusion 的 NPU 3 移动试验见 `experiments/dragon_npu3_smoke1/` 和 `experiments/sequence_dragon_npu3_full1/`，CollageRL 真实 512×512 试验见 `experiments/sequence_material_npu3_smoke4/`。
