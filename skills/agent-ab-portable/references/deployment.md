# 第二台 Linux 服务器

将 `agent-ab-portable` 目录交给第二位操作者，放在实际使用 Codex 的机器的用户技能目录中。Python 脚本及同级 `references/tools.json` 需要在目标服务器上可读。操作者另行取得同一版本的 Agent-Ab 项目和有权使用的安装包、模型与数据库。

如果已有完整发布包，优先在它的项目根目录运行 `bash scripts/install.sh --check`，再阅读仓库的 `docs/INSTALL.md`。报告里每个缺项附带官方教程或下载入口和安装后核对项。需要配置时使用发布包的 `--setup`。

## 第一次盘点

在服务器使用 Python 3.9 或更新版本。把示例路径替换为本地路径。`--site` 可传入原版 `configs/site.json`，用来保留流程设置并识别标准工具及配套资产。

```bash
SKILL="$HOME/agent-ab-portable"
PROJECT="$HOME/agent-Ab"
SOURCE_SITE="$PROJECT/configs/site.json"

python3 "$SKILL/scripts/preflight.py" --project "$PROJECT" \
  --site "$SOURCE_SITE" --search "$HOME/tools" --search /opt \
  --sizes --report-dir "$HOME/agent-ab-inventory"
```

省略 `--verify` 时只做路径和文件检查。对于操作者认识并信任的软件，加 `--verify` 检查 Python 版本、包版本和已定义的轻量入口。加 `--profile stage1` 或 `--profile full` 时，报告标出默认发布范围的必需项。单独盘点两款软件时用 `--only proteinmpnn_soluble foldx`。点名想复用的安装、模型或环境时，通过 `--locations locations.json` 提供路径。下面是示例。

```json
{
  "ppiflow": {"path": "/srv/bio/PPIFlow", "env": "/srv/envs/ppiflow_af3"},
  "protenix": {
    "path": "/srv/bio/Protenix",
    "env": "/srv/envs/protenix",
    "weights": "/srv/models/protenix/protenix_mini_esm_v0.5.0.pt",
    "data_dir": "/srv/models/protenix/common",
    "checkpoint_dir": "/srv/models/protenix"
  },
  "proteinmpnn_soluble": {"path": "/srv/bio/MPNN", "env": "/srv/bio/MPNN/env"},
  "foldx": {"path": "/srv/bio/FoldX"}
}
```

同一个工具有多个候选安装时，优先选择缺项最少者。同等完整时按显式位置、站点配置和自动发现的先后顺序选择。Python 环境字段使用环境目录或其 `bin/python`，避免使用已解析到基础解释器的软链目标路径。

## 配置缺项

AI 根据报告逐项打开相应版本的上游 README、安装文档或发行说明，确认安装命令、目标目录和环境名称。执行经授权的步骤后重新检测，验证解释器、命令入口与权重目录。碰到上游登录、许可或管理员操作时，把具体步骤交给操作者。

用 `--apply TOOL ... --recipes recipes.json` 只配置列出的缺项。每项配方指定工作目录和直接执行的 argv 列表。先审查每条命令、来源、版本和校验和。配方的写入路径由配方本身决定，使用当前用户可写目录。逐条命令遇错即停止该项，随后重新探测。已通过检查的工具直接复用。涉及新增环境或补装依赖时，将安装位置一起放入 `--locations`，或确保 `--search` 可以发现它。

例如，已有签入的 ipSAE 副本压缩包和对应 SHA-256 清单时，可以把它部署到新的工具目录。以下示例假定压缩包内是以 `ipsae.py` 等文件为根的已审核相对路径，SHA-256 清单中记录文件名 `ipsae.tar.gz`。解包目标应是操作者专门为该工具准备的空目录。

```json
{
  "ipsae": {
    "cwd": "/srv/staging",
    "commands": [
      ["sha256sum", "--check", "ipsae.tar.gz.sha256"],
      ["mkdir", "-p", "/srv/bio/IPSAE"],
      ["tar", "--extract", "--file", "ipsae.tar.gz", "--directory", "/srv/bio/IPSAE",
       "--no-same-owner", "--no-same-permissions"]
    ]
  }
}
```

```bash
python3 "$SKILL/scripts/preflight.py" --project "$PROJECT" --site "$SOURCE_SITE" \
  --search /srv/bio --locations locations.json --recipes recipes.json \
  --apply ipsae --sizes --report-dir "$HOME/agent-ab-inventory"
```

同样可以使用固定版本的官方源码、带哈希的离线 wheelhouse，或团队授权的构建包。已有但不完整的安装先选定新的空目录作为部署目标。把包、环境和模型权重作为独立资产确认。迁移现有 Conda 环境时使用可重定位的构建方案，并在迁移后验证 Python 和依赖导入。FoldX 由第二位操作者按授权取得合适的 Linux 二进制。AF3、Rosetta/PyRosetta 的包和权重由操作者准备其有权使用的版本。

- [ProteinMPNN 官方源码与 soluble 权重说明](https://github.com/dauparas/ProteinMPNN)
- [FoldX 学术下载入口](https://foldxsuite.crg.eu/academic-license-info)
- [Protenix 官方安装说明](https://github.com/bytedance/Protenix/blob/main/docs/training_inference_instructions.md)
- [IgGM 官方源码和安装说明](https://github.com/TencentAI4S/IgGM)

## 生成站点配置并验证

```bash
python3 "$SKILL/scripts/preflight.py" --project "$PROJECT" --site "$SOURCE_SITE" \
  --search /srv/bio --locations locations.json --verify \
  --map /旧服务器/数据库目录 /本机/数据库目录 \
  --write-site "$PROJECT/configs/site.portable.json" \
  --report-dir "$HOME/agent-ab-inventory"
```

`--write-site` 创建新文件，保留来源配置已有的流程开关、阈值和任务设置。用 `--map 旧前缀 新前缀` 显式迁移没有自动映射的独立目录，如结构输入、数据库及共享权重。检查生成的 `storage.runs_root`、`off_target_screening.structure`、`default_frameworks`、`large_assets` 和工具 `required_paths`，使它们对应目标服务器与实际任务。

在项目根目录，用已通过检查的 Agent-Ab Python 环境直接调用模块，或使用该项目已有的可移植启动器。本发布包的 `scripts/agent-ab` 默认使用本地 `.venv`，PPIFlow 路径来自已加载的站点配置。

```bash
cd "$PROJECT"
export AGENT_AB_ROOT="$PROJECT"
SITE="$PROJECT/configs/site.portable.json"
AGENT_PYTHON=/本机/Agent-Ab环境/bin/python
"$AGENT_PYTHON" -m agent_ab.cli --site-config "$SITE" doctor
"$AGENT_PYTHON" -m agent_ab.cli --site-config "$SITE" plan --help
"$AGENT_PYTHON" -m agent_ab.cli --site-config "$SITE" pipeline --help
```

任务执行时沿用保存的 spec。按现有 CLI，以 `plan` 创建新任务、`pipeline RUN_ID --gpu GPU_ID` 执行、`status RUN_ID` 和 `parts RUN_ID` 查状态。继续任务时沿用原 `RUN_ID`。软件盘点和帮助检查只验证部署准备，完整流程需在目标服务器上以实际运行验证。
