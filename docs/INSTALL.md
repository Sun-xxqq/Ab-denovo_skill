# 在新的 Linux 服务器部署 Agent-Ab

## 1. 先检测

在目标 Linux 服务器上进入仓库目录。准备 Bash 和 Python 3.9–3.12；运行计算需要可用的 NVIDIA GPU 和驱动。

```bash
bash scripts/install.sh --check
# 要为完整流程做准备时使用
bash scripts/install.sh --check --profile full
```

报告写入 `.setup-report/software-inventory.md` 和 `.setup-report/software-inventory.json`。先看「本次需要处理」，再看每个缺项下的官方教程和安装后核对项。检测会纳入站点标为必需的工具；默认范围使用发布包的 PPIFlow、AbMPNN、FlowPacker、Protenix 和 ipSAE 配置。完整范围另检查 Rosetta，并在站点启用自动成熟时检查 IgGM。替换预测器或开启其他分支时，按实际任务补齐相应工具和数据。

脚本搜索 `$HOME/agent-ab-deps` 及报告规则内的常见位置。软件装在其他位置时，复制 [位置示例](../configs/locations.example.json) 到 `configs/locations.json`，填实际目录和 Python 环境后重新检测。

```bash
bash scripts/install.sh --check --deps "$HOME/agent-ab-deps" -- \
  --locations configs/locations.json
```

检测阶段只生成报告。报告里已通过检查的软件可以直接用于下一步。

## 2. 按官方教程补齐缺项

下表区分上游安装步骤和 Agent-Ab 需要找到的文件。按照目标版本的官方说明安装，对照该版本的 Python、CUDA、PyTorch 和系统要求。服务器上的旧部署未能确定所有外部源码的提交号，新取得的版本需要用本仓库的示例验收。

| 缺少的工具 | 按官方教程做什么 | 安装后核对 |
| --- | --- | --- |
| PPIFlow | 获取官方仓库的目标版本，按其环境要求运行仓库内的 `bash Install.sh`，从其检查点说明获取对应权重。 | PPIFlow 源码目录有 `pipeline.py`，`ppiflow_af3` 环境可执行，站点模板使用 `weigths/` 下的权重。 [PPIFlow 安装与检查点](https://github.com/Mingchenchen/PPIFlow#environment-and-installation) |
| AbMPNN 与 FlowPacker | 按 PPIFlow 指南准备配套工具，再按 FlowPacker 上游说明补其模型权重。 | `tools/AbMPNN/` 里的运行脚本和权重，以及 `tools/flowpacker/` 里的配置和检查点都能被检测到。 [PPIFlow 依赖说明](https://github.com/Mingchenchen/PPIFlow#external-model-requirements) · [FlowPacker 上游](https://gitlab.com/mjslee0921/flowpacker) |
| Protenix | 按上游的本地源码安装方式，在独立环境安装选定版本，例如从对应源码目录执行其文档里的 editable 安装。 | 源码目录包含 `runner/inference.py`，环境可启动，模板对应 `protenix_mini_esm_v0.5.0` 的 checkpoint 和数据目录能访问。把实际位置填入 `protenix` 的 `locations`。 [Protenix 安装和数据准备](https://github.com/bytedance/Protenix/blob/main/docs/training_inference_instructions.md#-installation) |
| ipSAE | 从官方仓库获取 `ipsae.py`，按其说明在执行该脚本的 Python 环境准备 NumPy。 | `ipsae.py` 在配置的路径，负责运行它的 Python 可以导入 NumPy。 [ipSAE 安装](https://github.com/DunbrackLab/IPSAE#installation) |
| Rosetta / PyRosetta | 按 Rosetta Commons 或 PyRosetta 官方渠道取得适用版本和授权。 | 完整 PPIFlow 路径需要模板所期望的 Rosetta 可执行文件。只有启用 PyRosetta 相关步骤时才配置它的独立环境。 [Rosetta Commons](https://rosettacommons.org/) · [PyRosetta](https://www.pyrosetta.org/) |
| IgGM | 在启用成熟步骤时，按官方 `environment.yaml` 创建环境，按其 wheel 说明配齐依赖，并从 README 给出的来源获取检查点。 | `design.py`、IgGM 解释器和站点要求的检查点都被检测到。 [IgGM 安装与检查点](https://github.com/TencentAI4S/IgGM#-getting-started) |
| AF3Score / DockQ | 启用这类分支时，分别按上游说明准备 AlphaFold 3 数据和已授权参数，并在该阶段的 Python 环境安装 DockQ。 | AF3 数据与权重目录有实际文件。用目标环境运行 `DockQ --help`，排查 NumPy 和编译扩展版本不匹配。 [AlphaFold 3 安装](https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md) · [DockQ 安装](https://github.com/wallnerlab/DockQ#installation) |

可选工具以检测报告每一项的上游链接为准。GPU 不可用时按服务器发行版的官方说明处理驱动，或交给服务器管理员。 [NVIDIA Linux 驱动指南](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/)

## 3. 生成站点配置

补齐后运行 `--setup`。它创建 Agent-Ab 自己的 `.venv`、安装锁定的 NumPy、复检外部工具并运行 `doctor`。首次通过后才保存 `configs/site.json`。已有站点配置时会继续使用它。

```bash
bash scripts/install.sh --setup --profile stage1 -- \
  --locations configs/locations.json
```

已经将官方安装步骤或团队授权安装包写成配方时，可对指定缺项执行。每项配方包含工作目录 `cwd` 和命令参数列表 `commands`，示例见 [recipes.example.json](../configs/recipes.example.json)。如不使用配方，先按上游教程安装，再运行上面的 `--setup` 即可。

```bash
bash scripts/install.sh --setup --profile full -- \
  --locations configs/locations.json \
  --recipes configs/recipes.json --apply ppiflow protenix ipsae iggm
```

自动配置会先检查指定工具，已经可用的直接复用。安装后再探测，缺项未解决时保留诊断。需要登录或许可的上游资产由操作者通过相应官方渠道获取，再把路径交给安装器。

## 用 AI 辅助安装

带终端权限的 AI 助手可以读这份仓库的 [部署 Skill](../skills/agent-ab-portable/SKILL.md)，执行同样的检查和安装流程。普通终端可以直接运行同一套脚本。下面这段可以直接交给 AI。

> 在我下载的 Agent-Ab 项目里，用 `agent-ab-portable` 帮我部署。先运行 `bash scripts/install.sh --check`，读软件盘点，告诉我哪些已有、哪些必需、具体缺什么。对必需缺项打开报告给出的官方 GitHub 教程，按要安装版本的说明列出命令和安装位置。按我授权的范围协助安装，能复用的环境直接用。遇到需要登录、许可或管理员操作的步骤，请我完成对应步骤。每补齐一项就重新检查。最后运行 `--setup`，通过后用上游示例做 `dry-run`，汇报检查结果。

AI 辅助安装需要它能访问目标终端和官方说明。项目自身的研究 Agent 模式有独立的模型服务配置，终端安装流程可直接运行。

## 运行上游示例

`make-example.py` 从本地 PPIFlow 安装读取其文档使用的 IL‑13 和 nanobody framework 结构，生成 Stage 1 示例任务。

```bash
.venv/bin/python scripts/make-example.py --output example-task.json
bash scripts/agent-ab validate example-task.json
bash scripts/agent-ab plan example-task.json
bash scripts/agent-ab pipeline RUN_ID --dry-run
bash scripts/agent-ab pipeline RUN_ID --gpu auto
bash scripts/agent-ab status RUN_ID
bash scripts/agent-ab parts RUN_ID --tail 30
```

用 `plan` 返回的编号替换 `RUN_ID`。若流程等待 Top‑N，按排序表和允许范围选择，再通过 `select-backbones RUN_ID TOP_N` 记录并继续。完整任务用实际 `full` spec 验收 Stage 2、IgGM 和 Stage 3 的已启用分支。站点模板默认关闭原项目专用的负靶筛选，按实际任务提供对应结构、链和参数。

新版已在隔离的 Linux 目录验证两种检查范围、缺项教程链接，以及用模拟配方安装后复检和复用。上一版已验证复用现有第三方工具、创建项目环境、生成站点配置及示例 `dry-run`。在第二台服务器从零装齐第三方工具后的完整 GPU 运行仍待验收。
