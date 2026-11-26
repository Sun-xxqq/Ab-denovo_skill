---
name: agent-ab-portable
description: "为 Linux 服务器检测和部署 Agent-Ab。列出已可用软件、必需缺项和官方安装教程；按授权复用或补装依赖，复检后生成本机站点配置。也可单独盘点 ProteinMPNN soluble 和 FoldX。"
---

# Agent-Ab 依赖检测与安装

1. **先检测。** 有完整发布包时，在目标服务器的项目根目录运行 `bash scripts/install.sh --check`。为完整流程准备工具时加 `--profile full`。单独使用本 Skill 时，按 [部署与复检](references/deployment.md) 调用 [preflight.py](scripts/preflight.py)。
2. **根据报告处理缺项。** 查看已通过项、必需缺项和按需工具。对必需缺项打开报告中的官方 GitHub 或上游安装说明，确认其版本、系统条件、环境和资产要求。使用已确认的实际位置追踪软件、解释器、权重和数据库。
3. **按授权安装并复检。** 在新环境中执行对应版本的官方步骤或经授权配方，直接复用可用的软件。需要登录、许可或管理员操作时，向操作者说明具体步骤。安装完成后重新检测，按实际证据记录就绪状态。
4. **生成配置。** 准备齐全后，发布包使用 `bash scripts/install.sh --setup`。独立脚本使用 [配置说明](references/deployment.md) 中的 `--write-site`。核对生成的机器路径并运行 `doctor`，已有的站点配置按项目原路径处理。
5. **验证入口。** 先跑示例 `validate`、`plan` 和 `pipeline --dry-run`，报告通过和缺项。真实设计任务按用户确认的 spec 与运行范围执行。
