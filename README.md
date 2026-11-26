# Agent-Ab Pipeline 与 Skills

此仓库包含固定 Pipeline 源码、工具执行脚本、初始化入口和两项可选 Agent Skill。面向 Linux NVIDIA GPU 服务器。

## 从新机器开始

1. 先运行 `bash scripts/install.sh --check`。报告列出已有软件、当前必需缺项，以及对应的官方教程和安装后核对项。准备完整流程时加 `--profile full`。
2. 按 [安装指南](docs/INSTALL.md) 补齐缺项，也可让有终端权限的 AI 按官方教程辅助安装。再运行 `bash scripts/install.sh --setup` 复检并生成 `configs/site.json`。
3. 用 PPIFlow 的上游 IL‑13 示例做首次 Stage 1 检查。命令和验收标准见 [安装指南](docs/INSTALL.md#运行上游示例)。

```bash
bash scripts/install.sh --check
bash scripts/install.sh --setup
.venv/bin/python scripts/make-example.py --output example-task.json
bash scripts/agent-ab validate example-task.json
bash scripts/agent-ab plan example-task.json
bash scripts/agent-ab pipeline RUN_ID --dry-run
bash scripts/agent-ab pipeline RUN_ID --gpu auto
```

用 `plan` 返回的编号替换 `RUN_ID`。实际任务完成后，用 `status RUN_ID` 和 `parts RUN_ID` 核对运行状态及结果。完整任务用经确认的 `full` spec，并按暂停提示提供骨架选择与其他输入。

有 AI 助手时，把 [AI 辅助安装示例](docs/INSTALL.md#用-ai-辅助安装) 交给它即可。安装器也支持直接从终端使用。

## 包含的文件

- `agent_ab/` 与 `scripts/` 提供 Pipeline 编排、启动器和执行适配器。
- `configs/site.template.json` 提供可迁移的站点模板。它将历史项目使用的固定靶点改为通用配置，实际任务需填写自己的输入和筛选配置。
- [固定流程 Skill](skills/agent-ab-workflow/SKILL.md) 用于运行、续跑和检查任务。
- [部署预检 Skill](skills/agent-ab-portable/SKILL.md) 用于盘点和配置依赖。

发布状态为**待独立 GPU 环境验收**。第三方工具、模型权重、数据库和既有任务产物由执行方单独管理。
