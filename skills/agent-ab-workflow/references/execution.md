# Agent-Ab 执行参考

## 连接与入口

使用操作者提供的 Agent-Ab 项目路径和服务器连接。复用当前授权的 SSH 会话或认证方式，凭据通过连接工具处理。固定流程使用 `scripts/agent-ab pipeline`，研究 Agent 的 `agent-create`、`agent` 和 `ResearchTools` 对应隔离的 Stage 1。

命令在项目目录执行。将 `RUN_ID`、`TOP_N` 和 `GPU_ID` 替换为实际值。使用 argv 列表或正确引用的 shell 参数。站点配置通常读取 `configs/site.json`；指定覆盖文件时，在子命令前传入 `--site-config PATH`。

## 新任务与续跑

新任务从用户提供或确认的 spec 开始，使用新的任务名。

```bash
cd /path/to/agent-Ab
scripts/agent-ab validate /absolute/path/task.json
scripts/agent-ab plan /absolute/path/task.json
scripts/agent-ab pipeline RUN_ID --dry-run
scripts/agent-ab pipeline RUN_ID --gpu GPU_ID
```

从 `plan` 的返回值读取实际 `run_id`。按需求选择预览或启动计算。已有任务直接读取状态和各阶段日志，确认运行状态后再继续。

```bash
scripts/agent-ab status RUN_ID
scripts/agent-ab parts RUN_ID --tail 30
scripts/agent-ab pipeline RUN_ID --gpu GPU_ID
```

已有计算仍在运行时，继续观察同一个任务。续跑沿用保存的 spec、原编号和已完成产物。原生 PPIFlow 的 `run`、`resume` 供负责该子流程时使用。

## 暂停与失败

| 看到的状态 | 处理方式 |
| --- | --- |
| 等待选择骨架 | 查看 `action_required`、排序表和允许范围。使用用户指定或已记录的 Top‑N。需补选时给出排序与缺项。 |
| FastRelax 或其他步骤等待输入 | 根据 `action_required` 区分所需文件、确认或执行条件，补齐后用原 `run_id` 继续。 |
| 计算失败 | 核对返回码、对应日志和恢复状态。确认可恢复并修正原因后，用项目支持的 `--retry-failed` 重试失败步骤。 |
| `complete_no_stage*_…` | 按实际零候选分支报告停止阶段、门槛和已有输出。 |

记录 Top‑N 后继续：

```bash
scripts/agent-ab select-backbones RUN_ID TOP_N
scripts/agent-ab pipeline RUN_ID --gpu GPU_ID
```

## 后续阶段

以保存的 spec、站点配置和实际编排状态确认覆盖范围。骨架筛选后，序列设计、FlowPacker 和预测使用任务配置选定的工具。按实际启用条件执行后续步骤。

- `generation.profile=full` 进入 Stage 2 路径。继续已有任务时遵循它保存的运行范围。
- FastRelax 和负靶筛选按该任务的配置与适用条件执行。
- 自动 IgGM 核对 `_automatic_iggm_enabled(config, spec)`，包括 `tools.iggm.automatic.enabled`、设计类型和轻链。独立 `iggm-mature` 只在用户指定独立任务、父本与采样范围时使用。
- Stage 3 核对 specificity 开关、通过的父本与 IgGM 实际输出。

## 核对与交付

读取任务的状态、编排 manifest、筛选表和 `result_paths`，按实际启用的阶段核对：

1. 记录输入、完成、通过、拒绝和未确定数量。用候选 ID 统计拒绝，避免重复计入有多个拒绝理由的候选。
2. 比对预期和实际输出，核对公开结果目录、原始 backbone 与 design ID，以及交付需要的结构和序列。中间预测、PAE 与续跑记录按项目约定留在任务的 `.agent_work/`。
3. 正负靶筛选核对候选配对、分数方向和实际 delta 门槛。Stage 1 的 4+4 继续条件与最终通过条件分别核对。
4. IgGM 按父本核对 `batch_manifest` 和实际 FASTA 数量，并单独核对后续结构预测、筛选覆盖率和失败记录。IgGM FASTA 按序列变体报告，ipSAE 和 Rosetta 按计算筛选结果报告。实验亲和力依据实验数据。

最终说明 `run_id`、实际完成或暂停的阶段、确认的候选数量、输出目录，以及恢复所需的具体输入或命令。
