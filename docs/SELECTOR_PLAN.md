# 图像处理顺序选择器：数据与训练接口

已完成 100 张正式图的候选生成与人工排序，80 张原训练集、20 张原验证集均已标注。成品排序器及生成前配方评分器都已训练；见 `experiments/selector100_t15_v14_20261028/` 中的权重与报告。

## 决策输入

当前离线排序器读取**原图**、原图的 SAM2 主体掩码、候选成品图、按顺序排列的操作与参数，以及固定参考 ID。风格参考图与材质组保持不变；网络学习它们在什么图片与处理路径下值得选择，而不是修改参考图。顺序执行器在生成类步骤之后重新分割当前输出，后续处理因此使用实际中间图。未来的在线策略才会在每一步重新读取当前图并选下一步。

主体数量的自动估计只负责约束候选空间，不作人工标签。SAM2 候选可能把同一主体拆成多个部件；训练记录需要人工核对无主体、单主体或多主体。配方可写 `subject_count` 使用人工确认的数值，运行记录会注明来源；省略时使用保守的 SAM2 估计。

## 候选与标注

`data/photo120` 原始清单固定为 80 张 train、20 张 val、20 张 test。本轮使用前 20 张试验图以外的 100 张重新划分：固定种子 2012，在每类剩余 10 张中随机选 8 张 train、2 张 val，总计 80/20；该批次没有独立 test。原清单划分与新划分均保留在批次元数据。对每张图在其主体规则下生成不同处理顺序、参考 ID 和参数的候选；保存配方、逐步图片、掩码、权重与参考哈希、随机种子、失败原因。每张图仍有 12 个候选：无主体为 8 个单步、3 个两步、1 个三步；单主体为 5/4/3；多主体为 6/4/2。试验 20 张中，单步风格化 15/21 可接受，材质后接暖色处理 9/10，单独材质化 1/11；因此有主体候选固定包含两个单步风格参考、两个不同材质后接暖色处理、一个拼贴后接风格化，另保留高温度抽样的其他操作。材质重构结果按 `material_strength` 与当前图全图混合；比例为报纸 0.45、青花瓷 0.65、彩色玻璃 0.60。探索模板以温度 1.5 加权抽样并由原图 ID 固定种子复现，整批覆盖现有 16 种操作。失败候选记录为失败，不交给标注员当作审美劣例。

默认标注界面同时展示同一原图的全部成功候选，隐藏处理方法，初始位置按标注人和原图确定性打乱。标注员按从好到差的顺序点击可接受的候选，先选的排名越高；再次点击可取消，箭头可微调名次。提交时未选候选一律标为不可接受，可只选少数几张，甚至一张不选。被选候选可另标主体身份，原图主体数量仍需人工确认。排序保存在 `ranks_train.jsonl`；训练时使用所有有明确顺序或可接受性差异的候选对，两张都不可接受的不强行排序，每张原图等权、每个候选仅计一次可接受性损失。`/pairs` 保留盲法两两比较，用于平局和重复题一致性检查，记录保存在 `labels_train.jsonl`。材质重构可能覆盖整个主体；这种结果即便技术上运行成功，也应标成不可接受候选。标注员不看模型分数。

## 训练与评估

先做一个候选排序器：复用已下载的 DragonDiffusion CLIP 视觉编码器并冻结权重，在 NPU 3 上缓存原图、掩码主体图和候选成品的视觉特征；用按步骤位置排列的操作、参考 ID 和参数特征表示路径，由小型 MLP 输出偏好及可接受性分数。以 train 中同图候选的偏好和不可接受标记训练成对排序及辅助分类损失。val 只用于模型选择；这次 100 张划分没有独立 test，因此不报告独立测试结论。人工主体数与 SAM2 估计不同的图像仍保留其偏好标签，训练特征使用推理时可获得的 SAM2 估计，同时统计不一致数量。按源图划分，任何同源裁剪或生成图不得跨 split。报告偏好胜率、不可接受率、主体保留率，并与随机、固定规则、只看动作不看图片的消融基线比较。

生成前配方评分器仅使用原图、主体区域和配方特征，绝不读取候选成品；`selector-recommend` 可以先评分、拒绝全部不合格配方，或以 `--execute` 执行最高分配方并在运行失败时尝试下一项。它是完整配方选择，不是逐步观察中间结果的在线策略。仍需新的独立测试图片验证端到端质量。

## 本地执行第三阶段

在项目根目录执行。第三点固定选取 train 中每类前两张原图，共 20 张；这 20 张继续属于 train。三种主体路由均按上述比例生成 12 个候选。此试验比较全部操作、不同固定参考、少量强度与顺序，并记录失败与耗时。`--run-id` 指定可续跑批次；不读取 val/test 图片。前十张旧批次的标注保留；新版试验可用 `--source-offset 10 --limit 10` 单独生成后十张，不能与旧批次共用运行目录。

```bash
source /usr/local/Ascend/cann/set_env.sh
.venv/bin/python -m object_collage.cli selector-generate --pilot20 --run-id selector_pilot20
.venv/bin/python -m object_collage.cli selector-pilot-report --batch experiments/selector_pilot20/batch.json
.venv/bin/python -m object_collage.cli selector-label --batch experiments/selector_pilot20/batch.json --port 7862
.venv/bin/python -m object_collage.cli selector-generate --pilot20 --source-offset 10 --limit 10 --run-id selector_pilot20_tail_v4
.venv/bin/python -m object_collage.cli selector-generate --pilot20 --source-offset 14 --limit 6 --run-id selector_pilot20_tail_v7_20261028
```

2026-10-28 后六张使用 `selector_pilot20_tail_v7_20261028` 独立批次：72 个候选中 67 个成功，5 个因后续裁剪或移动的主体边界约束失败。20 张已标注，分别保存在 `selector_pilot20_12_v3_20261028`、`selector_pilot20_tail_v4_20261028`、`selector_pilot20_tail_v7_20261028`；第 11 张有旧新两次标注，以后者作试验分析。这些标签只用于调整候选设计，不进入新 100 张的训练或验证。

完整候选批次可断点续跑，train/val 标签保存在同一批次中：

```bash
source /usr/local/Ascend/cann/set_env.sh
.venv/bin/python -u -m object_collage.cli selector-generate --split selector100 --sampling-temperature 1.5 --run-id selector100_t15_v14_20261028
.venv/bin/python -m object_collage.cli selector-label --batch experiments/selector100_t15_v14_20261028/batch.json --port 7863
```

2026-10-29 起，尚未建立候选计划的 40 张来源使用 `detector_box_v1`：CPU 上的 Faster R-CNN 提供完整目标框，NPU 3 上的 SAM2 以框提示生成掩码，优先选人或动物等完整个体。检测器权重、来源和哈希配置在 `configs/dev.yaml`。此前已建立计划的 60 张继续使用 `sam2_automatic`，每张来源的 `subject_selection_mode` 写入 `batch.json`，续跑和后续重新分割均按该字段执行。无合格检测框时，SAM2 回退到自动掩码，并在 `mask_source` 中注明。

主体数量原规则对无主体、单主体、多主体的判别为 52/100；目标检测框计数规则在每类两张、共五折的验证中为 71/100，各折都选择分数至少 0.5、图像面积至少 5%、优先计人和动物的相同规则。运行时若有效 SAM2 掩码已选中主体，数量至少记为 1；这个掩码兜底未纳入仅靠检测框的离线 71/100。新图使用该规则；原 100 张的候选与标注不改写。四张人工标为无主体的图仍全部被检测器判成多主体，表明显著性判断尚未解决。逐张审计在 `route_audit.json`。

原版损失的五折中，成品排序器、生成前配方评分器、只看配方基线分别为 58/100、31/100、51/100，报告在 `cross_validation/report.json`。平衡可接受性类别后，同一五折分别为 72/100、71/100、51/100，报告在 `cross_validation_balanced/report.json`。每折验证图也用于挑选最佳轮，因此这些数字偏乐观；若预先固定训练 80 轮，三者依次为 69/100、66/100、50/100。生成前模型以五折最佳轮次中位数 49 轮在全部 100 张上训练，权重为 `recipe_ranker.pt`。当前仍没有独立测试集，原 20 张试验图参与过方案设计，不能当作无偏测试。

```bash
source /usr/local/Ascend/cann/set_env.sh
.venv/bin/python -m object_collage.cli selector-audit-routes --batch experiments/selector100_t15_v14_20261028/batch.json
.venv/bin/python -m object_collage.cli selector-cross-validate --batch experiments/selector100_t15_v14_20261028/batch.json --epochs 80
.venv/bin/python -m object_collage.cli selector-recommend --batch experiments/selector100_t15_v14_20261028/batch.json --image data/dev/images/000000283835.jpg --run-id recipe_example
# Add --execute to render the selected treatment.
```

## 已接入的固定选项

新版风格 ID：`van_gogh_starry_night`、`da_vinci_mona_lisa`、`van_gogh_sunflowers`、`cezanne_sainte_victoire`，来源与哈希见 `data/style_refs/SOURCE.json`。旧版的梵高《柏树》和葛饰北斋《神奈川冲浪里》保留，用于旧批次溯源和训练编码。新版材质 ID：`newspaper_a`、`newspaper_b`、`newspaper_c` 由已校验的 UCI 素材库按固定种子各抽取 8 页；`ceramic_blue_white` 与 `stained_glass` 分别从用户提供的两张青花瓷、两张彩色玻璃图片中各裁八张固定素材。来源文件哈希、裁剪坐标和素材哈希见 `data/material_refs/SOURCE.json`。旧版材质 ID 保留用于既有批次复现。报纸的三个 ID 只代表同一材质内部变化。

配方示例在 `configs/recipes/`。在本机执行：

```bash
source /usr/local/Ascend/cann/set_env.sh
.venv/bin/python -m object_collage.cli sequence --image data/dev/images/000000283835.jpg --recipe configs/recipes/subject_color.json --run-id example_sequence
```
