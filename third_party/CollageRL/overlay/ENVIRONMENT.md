# Windows 环境

2026-09-15 更新：已在父目录 `project/.venv`（Python 3.10.16、PyTorch 2.5.1+cu121）完成小规模 GPU 推理，`infer.ps1` 默认使用该共享环境。安装、版本和验证记录见 [共享环境说明](../docs/SHARED_ENVIRONMENT.md)。

`CollageRL/.venv` 的旧环境仍保留，下面的版本记录属于该独立环境。根目录 VS Code 使用主项目 `.venv`，直接打开本仓库时仍使用旧环境；需要共享环境时选择 `../.venv/Scripts/python.exe`。也可以通过 `infer.ps1 -Python` 显式指定其他解释器。

## 版本选择

- PyTorch 2.0.1 + CUDA 11.8、torchvision 0.15.2：用于本机 RTX 4060。
  配套版本见 [PyTorch 官方历史版本](https://docs.pytorch.org/get-started/previous-versions/)。
- NumPy 1.23.5、SciPy 1.8.0、Kornia 0.6.8、Gym 0.17.3：保留旧代码的依赖接口。
- OpenCV 的普通版及 Albumentations 依赖的 headless 版统一为 4.8.0.74，避免解析到要求新版 NumPy 的版本。
- Gym 构建使用 setuptools 65.5.0 和 wheel 0.38.4。
- `module/gpu.py` 使用 PyTorch 查询显存并自动选卡，避免旧版 NVML 绑定写死的 Windows DLL 路径。

原始 `requirements.txt` 保留用于论文环境复现；本机使用 `requirements-windows.txt`。
PyTorch wheel 包含 CUDA 运行库，无须另装完整 CUDA Toolkit。

## 使用

在 PowerShell 中：

```powershell
cd D:\mystudy\DSAA2012\project\CollageRL
.\.venv\Scripts\python.exe -m pip check
powershell -ExecutionPolicy Bypass -File .\infer.ps1 -SourceDir samples/materials/newspaper
```

请先将 JPG/PNG 素材放入 `samples/materials/newspaper/`，或用 `-SourceDir` 指定其他素材目录。
仓库自带目标图 `samples/goals/boat.jpg` 及预训练权重，但没有附带报纸素材集。
素材尺寸应足够覆盖推理窗口；默认最大窗口为 512，素材缩放比例为 0.6。
仅接受 `.jpg`、`.png`、`.JPG`、`.PNG` 后缀，与原代码一致。

可换目标图或调整每步候选素材数量：

```powershell
powershell -ExecutionPolicy Bypass -File .\infer.ps1 -Goal samples/goals/boat.jpg -SourceDir samples/materials/newspaper -SourceSampleSize 30
```

`infer.ps1` 默认每步评估 30 个候选素材（原脚本为 100），以降低显存开销。
结果写入 `samples/results/`。
`-SmokeTest` 使用 128×128 目标、64 像素窗口、2 轮，适合验证运行链路，不能用于评价最终拼贴质量。

训练入口为 `.\.venv\Scripts\python.exe -X utf8 train.py`，具体参数参考 `train.sh`。
完整训练还需要自行准备数据集；开启日志时需要配置自己的 W&B 账号。

## 重建环境

在仓库目录执行；将 Python 路径替换为可用的 Python 3.10 解释器：

```powershell
uv venv .venv --python D:/anaconda/envs/espnet_env/python.exe --no-cache
uv pip install --python .venv/Scripts/python.exe --no-cache --index-strategy unsafe-best-match --build-constraint requirements-windows.txt -r requirements-windows.txt
```

当前虚拟环境借用上述 Conda 环境的 Python 解释器，包安装在本项目内，不会修改该 Conda 环境。
如果删除或移动该 Conda 环境，需要使用其他 Python 3.10 解释器重建本项目环境。
`--no-cache` 避开本机 uv 默认缓存目录的初始化错误。

## 本机验证记录

- `pip check` 通过；全部主要依赖、训练数据加载模块导入通过。
- `infer.py --help`、`train.py --help` 均正常退出。
- RTX 4060 上 CUDA 张量计算、自动选卡及预训练权重加载通过。
- `infer.ps1 -SourceDir samples/goals -SourceSampleSize 4 -SmokeTest` 完成 18 次有效拼贴，生成 128×128 图片与可读取的 18 帧 MP4。
  此处仅借用目标图作为测试素材，不代表正式素材集的效果。
- 测试日志：`.venv/smoke-test.log`；已安装包版本快照：`.venv/installed-requirements.txt`。
- 尚未执行完整数据集训练或默认 512 分辨率的完整推理。
