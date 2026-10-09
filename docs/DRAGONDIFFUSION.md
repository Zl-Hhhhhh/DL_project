# DragonDiffusion 本地验证

2026-09-18：在项目根 `.venv` 中验证 512×512 主体移动，未执行 InstantStyle 风格化。此入口用于独立实验，尚未接入网页。

## 模型与环境

- 源码：[MC-E/DragonDiffusion](https://github.com/MC-E/DragonDiffusion)，commit `f9b5bdffcedfbc6337af258e64e40f3a5a836a88`，本地目录 `DragonDiffusion/`。
- 权重：`models/dragondiffusion/`，17 个模型/配置文件，共 **5,064,527,543 字节**。已重新对照发布方 SHA-256 / Git blob 哈希验证，清单见 `MANIFEST.json` 和各目录 `SOURCE.json`。
- SD 1.5 使用官方 FP16 safetensors；图像编码器、IP 适配器、EfficientSAM 和人脸关键点权重均已下载。
- Python 3.10.16、PyTorch 2.5.1+cu121、Diffusers 0.29.2、Transformers 4.44.2、Accelerate 0.34.2；GPU 为 RTX 4060 Laptop 8 GB。
- 未安装、升级或降级共享环境中的包；`pip check` 通过。

## 兼容与显存适配

`DragonDiffusion/run_local.py` 使用本地权重并强制离线，沿用官方 DDIM 反演、特征梯度引导、注意力记忆库及 D+ 采样流程。

1. 将旧版 Diffusers 导入路径适配到当前安装版本，通过现代模型加载器读取 FP16 variant。
2. 用 PyTorch SDPA 替换自注意力中的 xFormers 运算，保留官方 Q/K/V 与记忆库逻辑；无需安装与旧 PyTorch 绑定的 xFormers。数值检查覆盖前向与 Q/K/V 梯度，对照显式 softmax 注意力计算通过，记录在 `experiments/dragondiffusion_check/sdpa-check.json`。
3. UNet 与特征估计网络交替驻留 GPU；估计网络保持在 GPU 直到潜变量梯度计算完成。冻结模型参数，但保留编辑所需的潜变量梯度。
4. 图像编码器计算完成后释放；VAE 按阶段上 GPU，启用切片与分块解码；注意力记忆库存于 CPU。
5. CLI 的图像转换使用小型本地函数，不依赖 BasicSR。主体移动不需要 dlib、Lightning 或 Gradio。

官方 `app.py` 仍是上游 Gradio 入口，未做完整兼容验证；请使用下面的本地 CLI。人脸关键点权重只验证了文件完整性，未测试人脸编辑，也未验证外观替换、拖拽和粘贴等其他模式。

## 复现

在项目根目录运行，`--output` 必须指定一个尚不存在的目录：

```powershell
.\.venv\Scripts\python.exe -X utf8 DragonDiffusion/run_local.py --image experiments/dragondiffusion_check/official-move-001.png --steps 50 --dx 128 --output experiments/dragondiffusion_check/my-move
```

默认先将输入居中裁成 512×512。默认框 `[95, 215, 225, 390]` 针对本次官方示例的前景杯子；其他图片需要设置 `--box x1 y1 x2 y2`，坐标以裁切后的 512 图像为准。也可传入 `--mask`，白色代表主体、黑色代表背景。自定义掩码须与裁切后的图像对齐。`--dx` / `--dy` 控制位移，`--prompt` 描述图像，默认随机种子为 42。

输入来自[官方移动示例](https://huggingface.co/Adapter/DragonDiffusion/blob/eb744a24b896db24ad6751242a07d60cb5995986/examples/move/001.png)。默认提示词 `a photo of a cup`；保持官方引导参数：CFG 4、energy scale 500、编辑权重 4、内容权重 6、对比权重 0.2、修复权重 0.8、SDE 0.4、IP scale 0.1。

每次保存 `input.png`、`mask.png`、`output.png`、左右对照 `comparison.png` 和阶段耗时/显存 `metrics.json`。2 步仅用于检查流程；正式记录使用 50 步反演 + 50 步编辑。

## 实测记录

- `smoke-512/`：EfficientSAM 分割 + 2 步短流程通过，用时 12.676 秒。分割得到 13,779 个前景像素。此结果不用于评价画质。
- `full-512/`：复用上述掩码，向右移动 128 像素，完整 50 步，进程退出码 0。用时 **51.965 秒**，其中反演 5.294 秒、编辑 42.205 秒。
- 完整流程 PyTorch 峰值已分配显存 **3,516,766,208 字节（约 3.28 GiB）**，峰值保留显存 **3,613,392,896 字节（约 3.37 GiB）**。这不是整机显存或所有图片的上限。
- 完整输出已验证为可读的 512×512 PNG，潜变量有限、输出非空白，SHA-256 记录在 `metrics.json`；结束后 `nvidia-smi` 显示 GPU 内存使用回到 0 MiB。
- 视觉检查：128 像素位移产生明显杯口重影/变形，背景也发生变化。**运行通过不代表编辑质量合格**；扩散输出不保证非编辑区域逐像素不变，尚不能代替现有的严格掩码拼接流程。
- `move64-512/`：同一图像、掩码、种子和引导参数，仅将水平位移改为 64 像素。完整 50 步通过、退出码 0，用时 **66.928 秒**；峰值已分配显存 **3,527,405,056 字节（约 3.29 GiB）**，峰值保留显存 **3,590,324,224 字节（约 3.34 GiB）**。主体杯子较完整，但上方手部和杯子被明显改写，仍不满足保持其他区域不变的要求。此组与前一组只构成单图位移对比，不能代表普遍质量。

权重清单中的 `inference_tested: false` 是下载器的下载阶段字段；推理结论以上述独立实验记录为准。未进行 768 或 1024 分辨率测试。
