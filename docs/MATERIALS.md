# CollageRL 内置素材

界面勾选“CollageRL 材质重构”后，默认选择“使用内置素材”，也可切换“自定义上传”。后者仍要求 1–8 张图片，每张不超过 4 MB。关闭后处理时，两类素材都不参与生成。

素材采用 CollageRL 官方 README 推荐的 [Newspaper and magazine images segmentation dataset](https://archive.ics.uci.edu/dataset/306/newspaper+and+magazine+images+segmentation+dataset)，包含 101 张俄文报纸、杂志扫描页。作者为 Aleksey Vilkin 和 Ilia Safonov，DOI 为 `10.24432/C5N60V`，UCI 标示许可为 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。这是仓库推荐的独立数据集。

本地目录为 `CollageRL/samples/materials/newspaper/`。原包包含 75 张 JPG 及 26 张 BMP/TIFF 原图。为控制磁盘和推理内存占用，全部原图按比例缩小至最长边 1024，保存为质量 90 的 RGB JPEG；文件名以 `_m` 结尾的分割标注不用于拼贴，下载压缩包不作为运行依赖。`SOURCE.json` 记录来源、修改方式、下载文件 SHA-256、每张原图和处理后图片的 SHA-256。这里的下载哈希是本次下载的记录，不是发布方提供的签名。

界面预览显示实际预选的 8 张素材。选择由固定种子 42 决定，避免每次加载整个素材库；素材具体文件名和来源会写入任务目录的 `postprocess.json` 及 `result.json`。需要不同素材组合时可使用自定义上传。

在项目根目录准备或校验素材：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/prepare_materials.py
```

首次准备需要网络和 UnRAR；本机使用 `C:/Program Files/WinRAR/UnRAR.exe`，其他位置可通过 `--unrar` 指定。也可用 `--archive path/to/dataset_segmentation.rar` 复用已下载的官方 RAR，或传入 UCI 下载的 ZIP。素材已存在时脚本仅校验，不重复下载。默认下载的临时压缩包和提取文件在准备后自动清理；通过 `--archive` 指定的用户文件不会被删除。

素材准备好后，正常生成无需 UnRAR 或联网。素材库不完整时，界面提示改用自定义上传；后端不会悄悄替换成其他图片。

网页的“输出尺寸”同时作用于转换结果和材质重构，支持 512×512、1024×1024。CollageRL 本身没有统一的“强度百分比”；`scale_order` 影响处理的碎片尺度，`num_cycles` 控制每个窗口的拼贴轮数，素材图的内容和缩放也会影响外观。更多轮数用于继续细化，不等于视觉变化必然更强。目前界面中的“风格强度”属于 InstantStyle，尚未提供 CollageRL 强度滑块。若把重构结果与基础结果按比例混合，那是额外的图像合成操作。

2026-09-17 本机验证：101 张素材安装后约 25.37 MiB，全部校验通过；临时下载压缩包已清理。30 项自动测试通过，桌面及 375/320 像素宽度界面检查通过。真实浏览器任务 `3ed7aec38c294bf69d349e071f43cad7` 使用 8 张内置素材完成拼接后处理，下载为 1024×1024 PNG，下载哈希和后处理输入/输出记录一致；浏览器无 JavaScript 错误。截图和验证记录位于 `.cache/ui-verification/builtin-materials-*`。
