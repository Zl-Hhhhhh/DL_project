# 本机运行

本仓库已适配父目录 `project/.venv` 的 Python 3.10、PyTorch 2.5.1+cu121 环境，使用本地 SDXL 与 IP-Adapter 权重。

原图和风格图双输入入口为 `run_controlnet.py --content 原图路径 --style 风格图路径 --output 输出路径`，已补齐 ControlNet 权重。支持 `--size 512` 或 `--size 1024`（默认）；网页选择“风格化”并设置“输出尺寸”即可使用此入口。“CollageRL 材质重构”为转换后的独立可选阶段，沿用所选尺寸。

在父目录执行：

```powershell
.\.venv\Scripts\python.exe -X utf8 InstantStyle/run_local.py --steps 20 --size 1024 --output experiments/instantstyle/result.png
```

用 `--image` 指定风格参考图，`--prompt` 指定文本。默认启用 CPU offload，适配本机 RTX 4060 Laptop 8 GB；不要在原始 `infer_style.py` 上直接套用全模型驻留 GPU 的默认方式。

默认输出已改为 1024×1024，已完成 512 和 1024 两种尺寸、20 步真实出图验证。完整安装、下载、锁定版本、结果和范围见 [共享环境说明](../docs/SHARED_ENVIRONMENT.md)。
