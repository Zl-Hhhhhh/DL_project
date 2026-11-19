# 外部仓库

所有外部源码统一放在 `third_party/repos/`，不纳入主项目 Git。仓库版本见 `sources.json`；CollageRL、InstantStyle 和 DragonDiffusion 的本地补丁、增补文件分别保存在这里的同名目录。SAM2 使用固定上游提交，权重来源和哈希见 `models/sam2/SOURCE.json`。

在项目根目录重建源码（已经存在的目录无需重复克隆）：

```bash
mkdir -p third_party/repos
git clone https://github.com/northadventure/CollageRL.git third_party/repos/CollageRL
git -C third_party/repos/CollageRL checkout ac9b066a398f3f352783d14cfb65332258de73a2
git -C third_party/repos/CollageRL apply "$PWD/third_party/CollageRL/local-changes.patch"
cp -a third_party/CollageRL/overlay/. third_party/repos/CollageRL/

git clone https://github.com/instantX-research/InstantStyle.git third_party/repos/InstantStyle
git -C third_party/repos/InstantStyle checkout 6b40588e263c958653353ec24eb7eb990cfa3da7
git -C third_party/repos/InstantStyle apply "$PWD/third_party/InstantStyle/local-changes.patch"
cp -a third_party/InstantStyle/overlay/. third_party/repos/InstantStyle/

git clone https://github.com/MC-E/DragonDiffusion.git third_party/repos/DragonDiffusion
git -C third_party/repos/DragonDiffusion checkout f9b5bdffcedfbc6337af258e64e40f3a5a836a88
git -C third_party/repos/DragonDiffusion apply "$PWD/third_party/DragonDiffusion/local-changes.patch"
cp -a third_party/DragonDiffusion/overlay/. third_party/repos/DragonDiffusion/

git clone https://github.com/facebookresearch/sam2.git third_party/repos/SAM2
git -C third_party/repos/SAM2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
```

环境和权重准备见 [本机 Ascend 说明](../docs/LOCAL_ASCEND.md)。参考仓库的原始代码、模型权重和数据仍遵循各自的许可。历史 CUDA/Windows 验证记录保留在 `docs/SHARED_ENVIRONMENT.md` 和 `docs/DRAGONDIFFUSION.md`。
