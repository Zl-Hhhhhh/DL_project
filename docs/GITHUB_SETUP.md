# 将项目上传到 GitHub

当前主项目根目录还没有初始化 Git。建议仓库名使用 `object-aware-collage`，课程期间先设为 Private。

## 1. 当前上传范围

保留主项目源码、测试、配置、依赖记录、部署说明、数据来源清单和 Proposal 的 LaTeX 源码。根目录 `.gitignore` 已排除虚拟环境、下载缓存、模型权重、运行生成的图片与文档、课程提供的 PDF，以及三个外部参考仓库目录。忽略文件只影响 Git 跟踪，不会删除本机内容。

第三方仓库的固定提交、我们修改的 Python 代码补丁和新增辅助文件保存于 `third_party/`，避免将嵌套仓库直接加入主仓库。重建方法见 [参考仓库适配说明](../third_party/README.md)。这些文件是当前适配快照，后续若修改参考仓库，需要同步更新补丁或辅助文件。

模型通过已有准备脚本下载，数据通过 `scripts/prepare_dev.py` 准备。GitHub 普通 Git 仓库阻止大于 100 MiB 的文件，详见 [官方大文件说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。

## 2. 在 GitHub 创建空仓库

登录 GitHub，点击右上角加号，再选择 New repository。填写仓库名，选择可见性；新仓库不要勾选自动创建 README、.gitignore 或 License，避免与本地文件产生首次合并。创建后复制 HTTPS 仓库地址。当前尚未为自有代码选择开源许可证，第三方代码与权重仍受各自的上游条款约束。

## 3. 本地首次提交

在 VS Code 中打开整个项目文件夹，并新建 PowerShell 终端。依次执行：

```powershell
cd D:\mystudy\DSAA2012\project
git init -b main
git add .
git status --short
```

检查暂存列表，应包含 `src/`、`tests/`、`configs/`、`docs/` 和 `third_party/`，不应出现 `.venv/`、缓存、权重或 `experiments/ui/` 中的上传图片。然后提交：

```powershell
git commit -m "Initial project import"
```

如果 Git 提示缺少作者身份，设置该仓库的姓名与提交邮箱，再重试提交。邮箱可以使用 GitHub Settings → Emails 中提供的私密提交邮箱：

```powershell
git config user.name "你的 GitHub 名称"
git config user.email "你的 GitHub 提交邮箱"
git commit -m "Initial project import"
```

## 4. 连接远程并推送

将下面地址中的用户名和仓库名替换为实际值：

```powershell
git remote add origin https://github.com/你的用户名/object-aware-collage.git
git push -u origin main
```

按 Git 弹出的登录流程认证。成功后刷新 GitHub 仓库页面，即可看到项目。官方流程见 [Adding locally hosted code to GitHub](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)。

## 5. 后续更新

修改代码后，可以在 VS Code 的 Source Control 面板查看变更、暂存、提交和推送，也可以执行：

```powershell
git add .
git diff --cached --stat
git commit -m "Describe the change"
git push
```

上传项目代码和文件使用 Repository。GitHub 的 Projects 功能用于任务看板，可以在仓库创建后另行添加。
