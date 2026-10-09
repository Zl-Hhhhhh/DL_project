# CollageRL 与 InstantStyle 共享环境 · 2026-09-15

统一解释器：`D:/mystudy/DSAA2012/project/.venv/Scripts/python.exe`。Python 3.10.16、PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121、NumPy 1.26.4 均保持不变。

## 安装和复现

在项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File setup-references.ps1
# 需要下载或补齐 InstantStyle 模型时：
powershell -ExecutionPolicy Bypass -File setup-references.ps1 -DownloadModels
```

`requirements-lock.txt` 保留主项目的依赖快照；`requirements-references.txt` 声明新增依赖，`requirements-references-lock.txt` 锁定完整共享环境。当前 81 个已安装发行包，原有包的版本全部保留。Gym 0.17.3 仅在隔离构建中使用旧 setuptools/wheel，约束位于 `configs/legacy-build-constraints.txt`，不降级主环境的构建工具。

关键兼容版本：Kornia 0.6.8、Albumentations 1.1.0、Gym 0.17.3、W&B 0.13.3；Diffusers 0.29.2、Transformers 4.44.2、Accelerate 0.34.2、Hugging Face Hub 0.25.2。OpenCV 普通和 headless 包均为 4.10.0.84。更新这两个 OpenCV 包时先停止使用它们的 Python 服务，避免 Windows DLL 文件占用。

## CollageRL

根目录执行小规模推理：

```powershell
powershell -ExecutionPolicy Bypass -File CollageRL/infer.ps1 -SourceDir samples/goals -SourceSampleSize 4 -SmokeTest
```

这条命令默认使用根 `.venv`。`SourceDir` 和 `Goal` 的相对路径以 CollageRL 目录为基准；正式生成应提供自己的素材目录。这里借用目标图作为测试素材，仅验证运行，不用于评价作品质量。旧版独立环境仍可用 `-Python D:/mystudy/DSAA2012/project/CollageRL/.venv/Scripts/python.exe` 指定。

实测完成 18 次有效拼贴，输出 128×128 JPG 和可完整读取的 18 帧 MP4。启动脚本退出码为 0；`infer.py --help` 和 `train.py --help` 均通过。结果位于 `CollageRL/samples/results/boat_shared_env128square_64___goalsx1.0_sensitivity3.0_cycle2_fixt9.0_mcskip0/`，日志位于 `.cache/shared-environment/collagerl-launcher.log`。没有执行完整数据集训练或默认多尺度长推理。

## InstantStyle

```powershell
.\.venv\Scripts\python.exe -X utf8 InstantStyle/run_local.py --image InstantStyle/assets/0.jpg --prompt "a cat, masterpiece, best quality" --steps 20 --size 1024 --output experiments/instantstyle/my-result.png
```

默认输出为 1024×1024，仍可通过 --size 512 指定小图。输出文件必须未存在。启动脚本使用本仓库原始 `IPAdapterXL` 和 `up_blocks.0.attentions.1` 风格层，SDXL 基础模型为官方 fp16 权重。为 8 GB 显存启用 CPU offload；CLIP 图像编码器按需移入显卡，使用后移回 CPU。图像编码器直接以 fp16 加载，降低 CPU 内存峰值。CPU offload 的用途参见 [Diffusers 文档](https://huggingface.co/docs/diffusers/v0.29.2/en/optimization/memory)。

模型保存在 `models/instantstyle/`，缓存限定在项目 `.cache/huggingface`。下载脚本锁定 SDXL revision `462165984030d82259a11f4367a4eed129e94a7b`、IP-Adapter revision `018e402774aeeddd60609b4ecdb7e298259dc729`；分段下载校验发布方 SHA-256（小文件校验 Git blob 哈希），`SOURCE.json` 保存来源。启动时使用本地模型，无需重复下载。

此入口覆盖 SDXL 风格参考生成。ControlNet 双图入口见下节；inpainting、Plus、独立 Gradio 演示与完整训练未纳入验证。结果 PNG 旁保存 JSON，记录实际解释器、参数、用时和显存。

实测成功：以仓库 `assets/0.jpg` 为参考，生成 512×512 PNG，20 步、种子 42，进程退出码为 0。含模型加载用时 111.64 秒，PyTorch 峰值已分配显存 6,083,848,192 字节（约 5.67 GiB），不是整机显存占用。已打开检查，输出为正常可读的彩色猫咪图片。此单张结果验证运行链路，不代表风格或内容保持的系统评价。

输出：[instantstyle-512.png](../experiments/shared_environment/instantstyle-512.png)；参数：[instantstyle-512.json](../experiments/shared_environment/instantstyle-512.json)；日志：`.cache/shared-environment/instantstyle-inference.log`。

1024 更新：默认尺寸已改为 1024×1024，并完成同一参考图、20 步的实际推理。本次含加载 49.75 秒，PyTorch 峰值已分配显存约 5.87 GiB，退出码为 0。输出为 [instantstyle-1024.png](../experiments/shared_environment/instantstyle-1024.png)，参数保存在同名 JSON。不同尺寸的生成构图可能变化；这是重新生成的图片。

## 原图＋风格图与网页后处理更新

已增加 `InstantStyle/run_controlnet.py`：

```powershell
.\.venv\Scripts\python.exe -X utf8 InstantStyle/run_controlnet.py --content path/to/content.jpg --style path/to/style.jpg --prompt "a cat sitting in a garden" --output experiments/stylize/result.png
```

输出支持 `--size 512` 或 `--size 1024`，默认 1024×1024、20 步；网页“输出尺寸”同样适用于拼接、风格化及 CollageRL 后处理。可用 `--style-strength` 和 `--structure-strength` 调整两种条件的强度。模型为 [官方 SDXL Canny ControlNet](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0)，revision `eb115a19a10d14909256db740ed109532ab1483c`，fp16 safetensors 的 SHA-256 为 `b2e7d3921058a442cc80430d1ec8847f42599c705e2451c95e77cf4dcf8d6c25`。运行 `scripts/prepare_instantstyle.py --ranged --controlnet-only` 可补齐，完整下载脚本也已包含它。

在 8 GB GPU 上交替卸载 ControlNet 和 UNet，避免同时常驻；CLIP 只在编码风格图时上显卡，编码完成后释放该单次任务不再需要的编码器，减少系统内存占用。实测双图风格化用时 280.18 秒，PyTorch 峰值已分配显存 6,585,965,056 字节（约 6.13 GiB）。输出与结构图保存在 `experiments/controlnet_check/`，已打开确认输出可读且内容轮廓保留、色彩和纹理发生变化。

网页提供独立的“拼接 / 风格化”模式，并将 CollageRL 作为默认关闭的后处理。后处理需要用户上传素材，使用 256/128/64 多尺度、每尺度 4 轮的小规模配置；每个 GPU 阶段使用独立子进程，阶段结束释放权重。`base_output.png` 保存原转换结果，`postprocess.json` 保存输入/输出哈希、素材数量和实际命令。拼接的 `params.json` 仍用于重放后处理前的基础结果；不声称可精确重放 CollageRL 或扩散结果。

后处理给原仓库 `infer.py` 增加了 `--output_dir`、`--no_video` 与最终 PNG，避免网页任务保存数百帧视频导致内存膨胀。CLI 原先的视频功能仍保留。

网页端已完成真实生成和下载验证：拼接任务 `e4e5222a0ef84c118eb6d053c0f6ac1b`；20 步风格化接 CollageRL 任务 `cbdda80bea014dbe83acc5cf0f4fc12e`。后者风格化含加载约 440 秒，CollageRL 完成 1253 次有效拼贴；最终 PNG 为 1024×1024，下载文件与服务器输出哈希一致，后处理输入哈希与保留的基础结果一致。桌面和手机布局已检查，浏览器无 JavaScript 错误。当前 CollageRL 配置呈现明显的碎片拼贴效果，素材和后处理配置会影响内容可辨识度。

最终编码器释放改动另经 2 步实际推理验证，退出码 0，含加载约 55 秒，峰值已分配显存约 6.13 GiB；记录位于 `experiments/controlnet_final_check/`。该短测试只验证运行和输出尺寸，不用于评价质量。当前 28 项自动测试及 `pip check` 均通过。

## 主项目兼容性（基础环境记录）

2026-09-17 双尺寸更新：网页、ControlNet CLI 和 CollageRL 参数已接入 512/1024 选项，默认 1024。33 项自动测试通过；真实浏览器拼接任务 `8e54ba1f95bc4ce894f6c184acda5ec5`（512）和 `f25bdf3fb8fd49faac1b0a55ab98c0bc`（1024）均成功，下载尺寸与哈希校验通过，切换尺寸后的识别框重新计算也已验证。按用户要求，512 风格化任务 `3dcf115b663f4765bf2b81fb3c6bb9d8` 在 11/20 步时主动停止，其后续 CollageRL 未执行；这次新接入的 512 风格化及其后处理端到端验证尚未完成。此前已完成的 1024 实测记录仍保留。

全部原有依赖版本保留，`pip check` 通过，21 项测试通过，CUDA 检查通过。`setup-references.ps1` 在当前环境重复执行通过，未使用删除环境的方式做全新重建测试。拼贴界面继续使用根目录 `.venv`，启动方式不变。
