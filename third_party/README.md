# 参考仓库的本地适配

这里保存上游仓库的固定提交、当前 Python 修改补丁及新增辅助文件。原始克隆目录不纳入主项目 Git 仓库；模型、数据、推理结果和字节码缓存不在这些快照中。

每个目录的 `SOURCE.json` 记录仓库地址、固定提交与辅助文件清单，`local-changes.patch` 记录已跟踪 Python 源码的本地修改，`overlay/` 保存新增文件。上游代码与权重保持各自的许可条款；本目录不为它们赋予新的开源许可证。

## 新机器重建

如果目标目录已经存在且完成适配，跳过克隆和应用补丁。这些命令用于主项目的新克隆，直接在项目根目录执行：

```powershell
git clone https://github.com/northadventure/CollageRL.git CollageRL
git -C CollageRL checkout ac9b066a398f3f352783d14cfb65332258de73a2
git -C CollageRL apply --check ../third_party/CollageRL/local-changes.patch
git -C CollageRL apply ../third_party/CollageRL/local-changes.patch
Copy-Item -Path .\third_party\CollageRL\overlay\* -Destination .\CollageRL -Recurse -Force

git clone https://github.com/instantX-research/InstantStyle.git InstantStyle
git -C InstantStyle checkout 6b40588e263c958653353ec24eb7eb990cfa3da7
git -C InstantStyle apply --check ../third_party/InstantStyle/local-changes.patch
git -C InstantStyle apply ../third_party/InstantStyle/local-changes.patch
Copy-Item -Path .\third_party\InstantStyle\overlay\* -Destination .\InstantStyle -Recurse -Force

git clone https://github.com/MC-E/DragonDiffusion.git DragonDiffusion
git -C DragonDiffusion checkout f9b5bdffcedfbc6337af258e64e40f3a5a836a88
git -C DragonDiffusion apply --check ../third_party/DragonDiffusion/local-changes.patch
git -C DragonDiffusion apply ../third_party/DragonDiffusion/local-changes.patch
Copy-Item -Path .\third_party\DragonDiffusion\overlay\* -Destination .\DragonDiffusion -Recurse -Force

```

补丁只恢复代码适配，不安装环境或下载模型。后续按 [共享环境说明](../docs/SHARED_ENVIRONMENT.md)、[DragonDiffusion 说明](../docs/DRAGONDIFFUSION.md) 及现有准备脚本完成环境与资源准备。

这些补丁在导出时通过了当前工作目录的反向应用检查。参考仓库继续修改后，需重新导出补丁并同步新增文件；主项目不会自动同步被忽略的克隆目录。
