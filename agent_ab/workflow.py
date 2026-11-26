from .fastrelax_naming import fastrelax_path, fastrelax_stage
import json
import os
import subprocess
from pathlib import Path

from .config import env_python
from .doctor import run_doctor
from .ppiflow import build_steps, build_task, command
from .pipeline_choices import build_workflow_plan
from .contacts import remap_framework_chains, split_complex_pdb
from .spec import load_and_validate, validate_spec
from .util import disk_free_gb, now_iso, read_json, select_gpus, write_json, write_yaml


def _runs_root(config):
    return Path(config["storage"]["runs_root"]).expanduser()


def _run_dir(config, run_id):
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("invalid run ID")
    return _runs_root(config) / run_id


def _write_status(run_dir, status, detail=None, **extra):
    path = Path(run_dir) / "status.json"
    previous = read_json(path) if path.exists() else {"history": []}
    event = {"status": status, "at": now_iso()}
    if detail:
        event["detail"] = detail
    history = list(previous.get("history", []))
    history.append(event)
    value = {"status": status, "updated_at": event["at"], "history": history}
    if detail:
        value["detail"] = detail
    value.update(extra)
    write_json(path, value)
    return value


def _stage2_task_without_rosetta(run_dir):
    """Create a Stage 2 task copy that skips PPIFlow's serial RosettaFixStep."""
    run_dir = Path(run_dir)
    source = run_dir / "ppiflow" / "task.yaml"
    output = run_dir / "ppiflow" / "task.stage2.parallel.yaml"
    lines = source.read_text(encoding="utf-8").splitlines()
    replaced = 0
    updated = []
    for line in lines:
        if line.lstrip().startswith("RosettaFixStep:"):
            indentation = line[: len(line) - len(line.lstrip())]
            updated.append(indentation + "RosettaFixStep: false")
            replaced += 1
        else:
            updated.append(line)
    if replaced != 1:
        raise RuntimeError(
            "expected exactly one RosettaFixStep switch in %s; found %s"
            % (source, replaced)
        )
    output.write_text("\n".join(updated) + "\n", encoding="utf-8")
    return output


def _scfv_screen_root(run_dir):
    """Return the single Stage 1 backbone-screen directory used by new runs."""
    return Path(run_dir) / "pipeline_output" / "stage1" / "backbone_screen"


def _pending_backbone_selection(run_dir):
    root = _scfv_screen_root(run_dir)
    request_path = root / "selection_request.json"
    selection_path = root / "backbone_selection.json"
    if request_path.exists() and not selection_path.exists():
        return read_json(request_path)
    return None


def _stage1_task_after_selection(run_dir):
    """Reuse completed backbones and resume Stage 1 at AbMPNN."""
    run_dir = Path(run_dir)
    source = run_dir / "ppiflow" / "task.yaml"
    output = run_dir / "ppiflow" / "task.stage1.after_selection.yaml"
    lines = source.read_text(encoding="utf-8").splitlines()
    replaced = 0
    updated = []
    for line in lines:
        if line.lstrip().startswith("PPIFlowStep:"):
            indentation = line[: len(line) - len(line.lstrip())]
            updated.append(indentation + "PPIFlowStep: false")
            replaced += 1
        else:
            updated.append(line)
    if replaced != 1:
        raise RuntimeError(
            "expected exactly one PPIFlowStep switch in %s; found %s"
            % (source, replaced)
        )
    output.write_text("\n".join(updated) + "\n", encoding="utf-8")
    return output


def record_backbone_selection(config, run_id, top_n):
    """Persist the user's top-N decision so Stage 1 can continue downstream."""
    from .backbone_screen import select_top_scfv_backbones

    run_dir = _run_dir(config, run_id)
    spec = read_json(run_dir / "spec.json")
    if spec.get("design_type") not in ("antibody", "nanobody"):
        raise ValueError("top-N backbone selection is only used for scFv or VHH runs")
    root = _scfv_screen_root(run_dir)
    summary_path = root / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("backbone screening summary not found: %s" % summary_path)
    summary = read_json(summary_path)
    accepted = int(summary.get("accepted_backbones", 0))
    top_n = int(top_n)
    if not 1 <= top_n <= accepted:
        raise ValueError("top-N must be between 1 and %s; received %s" % (accepted, top_n))
    selection = select_top_scfv_backbones(root, top_n)
    request_path = root / "selection_request.json"
    if request_path.exists():
        request_path.unlink()
    _write_status(
        run_dir,
        "selection_recorded",
        detail="User selected top %s of %s passing scFv backbones; ready to resume at AbMPNN"
        % (top_n, accepted),
        selected_top_n=top_n,
    )
    return {
        "run_id": run_id,
        "status": "selection_recorded",
        "selected_top_n": top_n,
        "accepted_backbones": accepted,
        "ranked_table": summary.get("sorted_csv"),
        "selection_file": str((root / "backbone_selection.json").resolve()),
        "continue_command": "scripts/agent-ab pipeline %s" % run_id,
    }


def _parallel_rosetta_command(config, run_dir, spec, stage=1):
    """Build coordinate-preserving Stage 1 scoring or Stage 2 FastRelax."""
    run_dir = Path(run_dir)
    stage = int(stage)
    if stage not in (1, 2):
        raise ValueError("FastRelax stage must be 1 or 2")
    ppiflow = config["tools"]["ppiflow"]
    script = Path(config["project_root"]) / "scripts" / "parallel_fastrelax.py"
    stage_name = "stage%d" % stage
    input_name = "fastrelax" if stage == 1 else "filtered_delta"
    argv = [
        env_python(config, ppiflow["env"]),
        str(script),
        "--input-pdbdir",
        str(fastrelax_path(run_dir / "pipeline_output" / stage_name / input_name)),
        "--output-dir",
        str(fastrelax_path(run_dir / "pipeline_output" / stage_name / "fastrelax_raw_output")),
        "--gentype",
        spec["design_type"],
        "--ppiflow-root",
        ppiflow["path"],
        "--max-workers",
        str(int(ppiflow.get("fastrelax_max_workers", ppiflow.get("rosetta_fix_max_workers", 8)))),
    ]
    if stage == 1:
        argv.extend(
            [
                "--preserve-coordinates",
                "--group-manifest",
                str(
                    run_dir
                    / "pipeline_output"
                    / "stage1"
                    / "rosetta_group_manifest.json"
                ),
                "--fixed-positions-csv",
                str(
                    run_dir
                    / "pipeline_output"
                    / "stage1"
                    / "fixed_positions.csv"
                ),
                "--mpnn-seqs-csv",
                str(
                    run_dir
                    / "pipeline_output"
                    / "stage1"
                    / "mpnn_pdbs"
                    / "mpnn_seqs.csv"
                ),
                "--backbone-root",
                str(
                    run_dir
                    / "pipeline_output"
                    / "stage1"
                    / "backbone_output"
                ),
            ]
        )
    else:
        argv.extend(
            [
                "--locked-positions-csv",
                str(
                    run_dir
                    / "pipeline_output"
                    / "stage2"
                    / "fixed_positions.csv"
                ),
            ]
        )
    return argv


def _finalize_numbered_stage2(run_dir, spec):
    """Publish the stable 00/01/02/... Stage 2 view after standalone FastRelax."""
    if not bool(spec.get("output_naming", {}).get("enabled", False)):
        return None
    from .artifacts import write_result_paths
    from .stage2_layout import finalize_stage2_layout

    layout = finalize_stage2_layout(
        run_dir,
        expected_sequences=int(
            spec.get("generation", {}).get("stage2_sequences_per_backbone", 4)
        ),
    )
    write_result_paths(run_dir)
    return layout


def run_fastrelax(config, run_id, stage, finalize_layout=True):
    """Run Stage 1 score-only interface analysis or final Stage 2 FastRelax."""
    run_dir = _run_dir(config, run_id)
    spec = read_json(run_dir / "spec.json")
    stage = int(stage)
    stage_name = "stage%d" % stage
    input_name = "fastrelax" if stage == 1 else "filtered_delta"
    input_dir = fastrelax_path(run_dir / "pipeline_output" / stage_name / input_name)
    if not input_dir.is_dir() or not list(input_dir.rglob("*.pdb")):
        raise FileNotFoundError("no accepted FastRelax inputs found: %s" % input_dir)
    output_dir = fastrelax_path(run_dir / "pipeline_output" / stage_name / "fastrelax_raw_output")
    coordinate_preserving = stage == 1
    operation = "interface scoring" if coordinate_preserving else "FastRelax"
    state_key = "stage1_interface_score" if coordinate_preserving else "stage2_fastrelax"
    log_name = "stage1.interface_score.log" if coordinate_preserving else "stage2.fastrelax.log"
    log_path = run_dir / "logs" / log_name
    argv = _parallel_rosetta_command(config, run_dir, spec, stage=stage)
    from .recovery import RecoveryState
    recovery = RecoveryState(run_dir)
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["MKL_THREADING_LAYER"] = "GNU"
    _write_status(
        run_dir,
        "running",
        detail=(
            "Rosetta interface scoring is running without coordinate changes"
            if coordinate_preserving
            else "FastRelax is running as the final step of stage2"
        ),
    )
    recovery.update(state_key, "running", detail=str(input_dir))
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            argv,
            cwd=str(run_dir),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        _write_status(
            run_dir,
            "failed",
            detail="%s %s failed; see %s" % (stage_name, operation, log_path),
            returncode=completed.returncode,
        )
        recovery.update(state_key, "failed", detail=str(log_path))
        raise RuntimeError("%s %s failed; see %s" % (stage_name, operation, log_path))
    numbered_layout = None
    if stage == 2 and finalize_layout:
        try:
            numbered_layout = _finalize_numbered_stage2(run_dir, spec)
        except Exception as exc:
            _write_status(
                run_dir,
                "failed",
                detail=(
                    "Stage 2 FastRelax completed, but numbered output layout "
                    "failed: %s" % exc
                ),
                returncode=0,
            )
            recovery.update(
                state_key,
                "failed",
                detail="FastRelax complete; numbered layout failed: %s" % exc,
            )
            raise
    _write_status(
        run_dir,
        "complete",
        detail=(
            "Stage 1 interface scoring completed with input coordinates preserved"
            if coordinate_preserving
            else (
                "FastRelax completed; Stage 2 results published in numbered "
                "00/01/02/... layout"
                if numbered_layout is not None
                else "FastRelax completed as the final step of stage2; no later stage was started"
            )
        ),
        returncode=0,
    )
    recovery.update(state_key, "complete", detail=str(output_dir))
    return {
        "run_id": run_id,
        "stage": stage,
        "status": "complete",
        "operation": operation,
        "coordinate_policy": (
            "score_only_preserve_input_coordinates"
            if coordinate_preserving
            else "fastrelax"
        ),
        "input": str(input_dir),
        "output": str(output_dir),
        "numbered_layout": numbered_layout,
        "next_command": "scripts/agent-ab pipeline %s --gpu auto" % run_id,
    }


# Backward-compatible Python API for existing launchers.
run_rosettafix = run_fastrelax


def _validation_plan(spec, config):
    tools = config["tools"]
    requested_predictors = spec["validation"]["independent_predictors"]
    requested_scores = spec["validation"]["interface_scoring"]
    workflow = build_workflow_plan(spec, config)
    return {
        "workflow": workflow,
        "primary_generation": {
            "tool": workflow["selected"]["backbone_generation"],
            "state": "configured",
            "purpose": "target-conditioned generation, sequence design, packing and AF3Score filtering",
        },
        "independent_prediction": [
            {
                "tool": name,
                "state": "post_generation_gate",
                "role": tools.get(name, {}).get("role", "unconfigured"),
            }
            for name in requested_predictors
        ],
        "interface_scoring": [
            {
                "tool": name,
                "state": "post_prediction_gate",
                "role": tools.get(name, {}).get("role", "unconfigured"),
            }
            for name in requested_scores
        ],
        "alternate_generators": ["rfdiffusion", "rfdiffusion3", "boltzgen"],
        "selection_policy": [
            "Reject candidates failing the primary PPIFlow/AF3Score filter.",
            "Require agreement from at least one independent predictor before nomination.",
            "Evaluate positive and every declared negative target with the same protocol.",
            "Rank only after interface confidence, geometry, energy and sequence-liability checks are joined by candidate ID.",
            "Treat all computational candidates as hypotheses requiring experimental validation.",
        ],
    }


def _write_report(run_dir, manifest, spec, doctor, validation_plan):
    run_dir = Path(run_dir)
    failed = [item for item in doctor["checks"] if item["required"] and not item["ok"]]
    lines = [
        "# agent-Ab 设计任务",
        "",
        "- 任务 ID：`%s`" % manifest["run_id"],
        "- 名称：%s" % spec["name"],
        "- 类型：`%s`" % spec["design_type"],
        "- 主流程：`PPIFlow`（%s）" % spec["generation"]["profile"],
        "- 骨架生成：`%s`" % spec["workflow"]["backbone_generation"],
        "- 序列设计：`%s`" % spec["workflow"]["sequence_design"],
        "- 侧链装配：`%s`" % spec["workflow"]["sidechain_packing"],
        "- 复合物预测：`%s`" % spec["workflow"]["complex_prediction"],
        "- 界面评分：`%s`" % spec["workflow"]["interface_scoring"],
        "- 第一轮骨架数：%s" % spec["generation"]["samples"],
        "- 第一轮每骨架序列数：%s"
        % spec["generation"]["num_sequences_per_backbone"],
        "- 第二轮每个 PartialFlow 骨架序列数：%s"
        % spec["generation"]["stage2_sequences_per_backbone"],
        "- 目标结构：`%s`，链 `%s`" % (spec["target"]["structure"], spec["target"]["chain"]),
        "- 热点：%s" % (", ".join(spec["target"]["hotspots"]) or "未指定"),
        "",
        "## 执行入口",
        "",
        "```bash",
        "scripts/agent-ab resume %s --gpu auto" % manifest["run_id"],
        "```",
        "",
        "## 配置产物",
        "",
        "- `spec.json`：规范化、不可歧义的任务定义",
        "- `ppiflow/task.yaml`：任务与步骤开关",
        "- `ppiflow/steps.yaml`：服务器真实路径与参数",
        "- `tool_status.json`：执行前环境快照",
        "- `validation_plan.json`：独立预测和评分门控",
        "- `workflow_plan.json`：逐阶段软件选择、环境、路径和备选项",
        "",
        "## 科学门控",
        "",
    ]
    lines.extend("%s. %s" % (index, item) for index, item in enumerate(validation_plan["selection_policy"], 1))
    lines.extend(["", "## 执行前检查", ""])
    if failed:
        lines.append("存在 %s 个必需项阻塞：" % len(failed))
        lines.append("")
        lines.extend("- `%s`：%s" % (item["name"], item["detail"]) for item in failed)
    else:
        lines.append("所有主流程必需项均已就绪。")
    lines.extend(
        [
            "",
            "> 计算筛选不能证明结合、特异性或可开发性；候选进入实验前必须完成阴性靶点交叉验证和湿实验确认。",
            "",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def create_plan(config, spec_path=None, spec=None, request_context=None):
    normalized = load_and_validate(spec_path) if spec is None else validate_spec(spec)
    free_gb = disk_free_gb(config["storage"]["runs_root"])
    minimum = float(config["storage"]["min_free_gb_plan"])
    if free_gb < minimum:
        raise RuntimeError("only %.1f GiB free; planning requires %.1f GiB" % (free_gb, minimum))
    runs_root = _runs_root(config)
    runs_root.mkdir(parents=True, exist_ok=True)
    run_id = normalized["name"]
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("job name must be a single directory name")
    run_dir = runs_root / run_id
    if run_dir.exists():
        raise FileExistsError(
            "run already exists: %s; choose a different --name" % run_id
        )
    run_dir.mkdir(parents=False)
    (run_dir / "logs").mkdir()
    (run_dir / "pipeline_output").mkdir()
    if normalized["design_type"] in ("antibody", "nanobody"):
        framework = normalized["framework"]
        target = normalized["target"]
        target_path = target["structure"]
        framework_path = framework["structure"]

        if framework["structure"] == normalized["target"]["structure"]:
            original = target["structure"]
            partners = [framework["heavy_chain"]]
            if normalized["design_type"] == "antibody":
                partners.append(framework["light_chain"])
            target_path, framework_path = split_complex_pdb(
                original,
                target["chain"],
                partners,
                run_dir / "inputs",
                variable_only=False,
            )
            default_framework = config.get("default_frameworks", {}).get(normalized["design_type"])
            if default_framework:
                framework_path = default_framework["structure"]
                framework["heavy_chain"] = default_framework["heavy_chain"]
                if normalized["design_type"] == "antibody":
                    framework["light_chain"] = default_framework["light_chain"]
                framework["automatic_framework_source"] = default_framework["structure"]
                framework["automatic_framework_policy"] = "validated_ppiflow_framework"
            target["source_complex"] = original
            framework["source_complex"] = original

        # PPIFlow's antibody and nanobody stages use canonical chains A (VH/VHH),
        # B (VL), and C (antigen).  Normalize separate input files as well as
        # pre-complexed inputs so downstream Rosetta and partial-flow stages see
        # the same chain IDs.
        original_target_chain = target["chain"]
        canonical_target_chain = "C"
        if original_target_chain != canonical_target_chain:
            target_path = remap_framework_chains(
                target_path,
                {original_target_chain: canonical_target_chain},
                run_dir / "inputs" / "target_C.pdb",
            )
            target["hotspots"] = [
                canonical_target_chain + residue[len(original_target_chain):]
                for residue in target["hotspots"]
            ]
            target["canonical_chain_mapping"] = {
                original_target_chain: canonical_target_chain
            }
        target["chain"] = canonical_target_chain

        original_heavy_chain = framework["heavy_chain"]
        framework_mapping = {original_heavy_chain: "A"}
        framework["heavy_chain"] = "A"
        if normalized["design_type"] == "antibody":
            original_light_chain = framework["light_chain"]
            framework_mapping[original_light_chain] = "B"
            framework["light_chain"] = "B"
            framework_name = "framework_HL.pdb"
        else:
            framework_name = "framework_A.pdb"
        framework_path = remap_framework_chains(
            framework_path,
            framework_mapping,
            run_dir / "inputs" / framework_name,
        )

        target["structure"] = target_path
        framework["structure"] = framework_path
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "name": normalized["name"],
        "design_type": normalized["design_type"],
        "created_at": now_iso(),
        "project_root": config["project_root"],
        "run_dir": str(run_dir),
    }
    if request_context is not None:
        manifest["request_source"] = "natural_language"
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "spec.json", normalized)
    if request_context is not None:
        write_json(run_dir / "request.json", request_context)
    doctor = run_doctor(config)
    write_json(run_dir / "tool_status.json", doctor)
    validation = _validation_plan(normalized, config)
    write_json(run_dir / "validation_plan.json", validation)
    write_json(run_dir / "workflow_plan.json", validation["workflow"])
    write_yaml(run_dir / "ppiflow" / "task.yaml", build_task(normalized, run_dir / "pipeline_output"))
    write_yaml(run_dir / "ppiflow" / "steps.yaml", build_steps(normalized, config))
    _write_status(run_dir, "planned", free_gb=round(free_gb, 2))
    _write_report(run_dir, manifest, normalized, doctor, validation)
    return {"run_id": run_id, "run_dir": str(run_dir), "report": str(run_dir / "report.md")}


def _inventory(run_dir):
    output = Path(run_dir) / "pipeline_output"
    by_suffix = {}
    files = []
    if output.exists():
        for path in output.rglob("*"):
            if path.is_file():
                suffix = path.suffix.lower() or "[none]"
                by_suffix[suffix] = by_suffix.get(suffix, 0) + 1
                if len(files) < 500:
                    files.append(str(path.relative_to(run_dir)))
    value = {"counts_by_suffix": by_suffix, "files_sample": files, "sample_truncated": len(files) == 500}
    write_json(Path(run_dir) / "artifacts.json", value)
    return value


def execute_run(config, run_id, stage=None, gpu="auto"):
    run_dir = _run_dir(config, run_id)
    if not run_dir.is_dir():
        raise FileNotFoundError("run not found: %s" % run_id)
    spec = load_and_validate(run_dir / "spec.json")
    ppiflow_only = {
        "backbone_generation": "ppiflow",
        "sequence_design": "ppiflow_mpnn",
        "sidechain_packing": "ppiflow_flowpacker",
    }
    incompatible = [
        "%s=%s" % (name, spec["workflow"][name])
        for name, expected in ppiflow_only.items()
        if spec["workflow"][name] != expected
    ]
    if incompatible:
        raise RuntimeError(
            "selected workflow cannot use the PPIFlow resume adapter yet: %s"
            % ", ".join(incompatible)
        )
    if stage not in (None, 1, 2):
        raise ValueError("stage must be 1, 2 or omitted")
    if stage == 2 and spec["generation"]["profile"] != "full":
        raise ValueError("stage 2 requires generation.profile=full")
    if stage is None and spec["generation"]["profile"] == "stage1":
        stage = 1
    preflight = run_doctor(config)
    write_json(run_dir / "tool_status.preflight.json", preflight)
    if not preflight["ready"]:
        raise RuntimeError(
            "preflight failed: %s required tool or host checks are not ready"
            % preflight["required_failures"]
        )
    current_status = read_json(run_dir / "status.json")
    if current_status.get("status") == "running":
        raise RuntimeError("run is already marked running; inspect its process and status before retrying")
    free_gb = disk_free_gb(run_dir)
    minimum = float(config["storage"]["min_free_gb_run"])
    if free_gb < minimum:
        raise RuntimeError("only %.1f GiB free; execution requires %.1f GiB" % (free_gb, minimum))
    selection_path = _scfv_screen_root(run_dir) / "backbone_selection.json"
    parallel_backbones = stage != 2 and not selection_path.exists()
    gpu_selection_mode = "auto" if gpu == "auto" else "explicit"
    if gpu == "auto":
        requested_gpu_count = int(
            config["tools"]["ppiflow"].get("backbone_parallel_gpus", 4)
        )
        gpu_ids = select_gpus(requested_gpu_count if parallel_backbones else 1)
    else:
        gpu_ids = []
        for item in str(gpu).split(","):
            gpu_id = int(item.strip())
            if gpu_id not in gpu_ids:
                gpu_ids.append(gpu_id)
        if not gpu_ids:
            raise ValueError("--gpu must be 'auto', one GPU id, or comma-separated GPU ids")
    gpu = int(gpu_ids[0])
    task_path = run_dir / "ppiflow" / "task.yaml"
    if stage == 1 and selection_path.exists():
        task_path = _stage1_task_after_selection(run_dir)
    parallel_rosetta_argv = None
    parallel_rosetta_log = None
    rosetta_fix_output = (
        fastrelax_path(run_dir / "pipeline_output" / "stage1" / "fastrelax_raw_output")
    )
    if stage == 2:
        task_path = _stage2_task_without_rosetta(run_dir)
        from .round_selection import active_round_input

        round_input = active_round_input(config, run_id, 2)
        if round_input is None and not rosetta_fix_output.exists():
            parallel_rosetta_argv = _parallel_rosetta_command(config, run_dir, spec)
            parallel_rosetta_log = fastrelax_path(run_dir / "logs" / "stage1.interface_score.log")
    else:
        round_input = None

    argv = command(
        config,
        task_path,
        run_dir / "ppiflow" / "steps.yaml",
        stage=stage,
    )
    label = "full" if stage is None else "stage%s" % stage
    log_path = run_dir / "logs" / ("ppiflow.%s.log" % label)
    backbone_gpu_ids = gpu_ids if parallel_backbones else [gpu]
    if len(backbone_gpu_ids) > 1:
        running_detail = (
            "PPIFlow %s; backbone generation split across GPUs %s; "
            "AbMPNN and FlowPacker auto-shard across idle GPUs"
            % (label, ",".join(str(item) for item in backbone_gpu_ids))
        )
    elif gpu_selection_mode == "auto" and not parallel_backbones:
        running_detail = (
            "PPIFlow %s; AbMPNN and FlowPacker auto-shard across idle GPUs"
            % label
        )
    else:
        running_detail = "PPIFlow %s on GPU %s" % (label, gpu)
    if round_input is not None:
        running_detail += (
            "; reusing round %s checkpoint for round %s"
            % (round_input["from_round"], round_input["to_round"])
        )
    _write_status(
        run_dir,
        "running",
        detail=running_detail,
        gpu=gpu,
        backbone_gpus=backbone_gpu_ids,
        abmpnn_gpu_policy=(
            "auto_idle" if gpu_selection_mode == "auto" else "explicit"
        ),
        flowpacker_gpu_policy=(
            "auto_idle" if gpu_selection_mode == "auto" else "explicit"
        ),
        parallel_gpu_policy=(
            "auto_idle" if gpu_selection_mode == "auto" else "explicit"
        ),
        round_input=(
            round_input["checkpoint_dir"] if round_input is not None else None
        ),
    )
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["AGENT_AB_GPU_SELECTION_MODE"] = gpu_selection_mode
    environment["AGENT_AB_ABMPNN_GPUS"] = ",".join(
        str(item) for item in gpu_ids
    )
    environment["AGENT_AB_FLOWPACKER_GPUS"] = ",".join(
        str(item) for item in gpu_ids
    )
    environment["AGENT_AB_PARALLEL_GPUS"] = ",".join(str(item) for item in gpu_ids)
    parallel_gpu_limit = int(
        config["tools"]["ppiflow"].get(
            "parallel_gpu_workers",
            config["tools"]["ppiflow"].get(
                "abmpnn_parallel_gpus",
                config["tools"]["ppiflow"].get("backbone_parallel_gpus", 4),
            ),
        )
    )
    environment["AGENT_AB_ABMPNN_MAX_GPUS"] = str(parallel_gpu_limit)
    environment["AGENT_AB_PARALLEL_MAX_GPUS"] = str(parallel_gpu_limit)
    environment["AGENT_AB_PPIFLOW_BACKBONE_GPUS"] = ",".join(
        str(item) for item in backbone_gpu_ids
    )
    environment["PYTHONUNBUFFERED"] = "1"
    environment["MKL_THREADING_LAYER"] = "GNU"
    environment["CHAI_DOWNLOADS_DIR"] = config["tools"]["chai"]["weights"]
    environment["FOUNDRY_CHECKPOINT_DIRS"] = config["tools"]["rfdiffusion3"]["weights"]
    if spec["generation"]["profile"] == "full" and stage != 1:
        # Upstream PPIFlow hard-codes this reader under Stage 2.  The project
        # wrapper redirects it to the real Stage 1 result directory.
        environment["AGENT_AB_ROSETTA_FIX_OUTPUT"] = str(rosetta_fix_output)
    if round_input is not None:
        environment["AGENT_AB_ROUND_INPUT_DIR"] = round_input["checkpoint_dir"]
    if parallel_rosetta_argv is not None:
        with parallel_rosetta_log.open("a", encoding="utf-8") as parallel_log:
            parallel_log.write(
                "argv=" + json.dumps(parallel_rosetta_argv, ensure_ascii=False) + "\n"
            )
            parallel_log.flush()
            parallel_completed = subprocess.run(
                parallel_rosetta_argv,
                cwd=str(run_dir),
                env=environment,
                stdout=parallel_log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if parallel_completed.returncode != 0:
            _write_status(
                run_dir,
                "failed",
                detail="parallel FastRelax exited %s; see %s"
                % (parallel_completed.returncode, parallel_rosetta_log),
                returncode=parallel_completed.returncode,
            )
            raise RuntimeError(
                "parallel FastRelax failed with exit code %s; log: %s"
                % (parallel_completed.returncode, parallel_rosetta_log)
            )
    with log_path.open("a", encoding="utf-8") as log:
        log.write("argv=" + json.dumps(argv, ensure_ascii=False) + "\n")
        log.flush()
        completed = subprocess.run(
            argv,
            cwd=str(run_dir),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    artifacts = _inventory(run_dir)
    selection_request = _pending_backbone_selection(run_dir)
    if completed.returncode == 0 and selection_request:
        _write_status(
            run_dir,
            "awaiting_user_input",
            detail=selection_request["prompt"],
            returncode=0,
            action_required=selection_request,
            artifacts=artifacts["counts_by_suffix"],
        )
        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "log": str(log_path),
            "status": "awaiting_user_input",
            "action_required": selection_request,
        }
    if completed.returncode == 0:
        _write_status(
            run_dir,
            "complete",
            detail="PPIFlow %s finished" % label,
            returncode=0,
            artifacts=artifacts["counts_by_suffix"],
        )
    else:
        _write_status(
            run_dir,
            "failed",
            detail="PPIFlow %s exited %s; see %s" % (label, completed.returncode, log_path),
            returncode=completed.returncode,
        )
        raise RuntimeError("PPIFlow failed with exit code %s; log: %s" % (completed.returncode, log_path))
    return {"run_id": run_id, "run_dir": str(run_dir), "log": str(log_path)}


def list_runs(config):
    root = _runs_root(config)
    values = []
    if not root.exists():
        return values
    for directory in sorted((item for item in root.iterdir() if item.is_dir()), reverse=True):
        status_path = directory / "status.json"
        manifest_path = directory / "manifest.json"
        if not status_path.exists() or not manifest_path.exists():
            continue
        status = read_json(status_path)
        orchestration_path = directory / "orchestration_state.json"
        orchestration = read_json(orchestration_path) if orchestration_path.exists() else {"stages": {}}
        stage_values = list(orchestration.get("stages", {}).values())
        stage_statuses = [item.get("status") for item in stage_values]
        if "running" in stage_statuses:
            display_status = "running"
        elif "awaiting_user_input" in stage_statuses:
            display_status = "awaiting_user_input"
        elif "failed" in stage_statuses:
            display_status = "failed"
        elif orchestration.get("stages", {}).get(
            "stage2_cldn18_1_offtarget_screen", {}
        ).get("status") == "complete":
            display_status = "complete"
        elif orchestration.get("stages", {}).get(
            "stage2_on_target_screen", {}
        ).get("status") == "complete":
            display_status = "complete"
        elif orchestration.get("stages", {}).get("final_functional_evaluation", {}).get("status") == "complete":
            display_status = "complete"
        elif stage_values and any(value == "complete" for value in stage_statuses):
            display_status = "in_progress"
        else:
            display_status = status.get("status")
        manifest = read_json(manifest_path)
        values.append(
            {
                "run_id": directory.name,
                "name": manifest.get("name"),
                "design_type": manifest.get("design_type"),
                "status": display_status,
                "updated_at": status.get("updated_at"),
            }
        )
    return values


def get_run(config, run_id):
    run_dir = _run_dir(config, run_id)
    if not run_dir.is_dir():
        raise FileNotFoundError("run not found: %s" % run_id)
    orchestration_path = run_dir / "orchestration_state.json"
    return {
        "manifest": read_json(run_dir / "manifest.json"),
        "status": read_json(run_dir / "status.json"),
        "spec": read_json(run_dir / "spec.json"),
        "report": str(run_dir / "report.md"),
        "orchestration": read_json(orchestration_path) if orchestration_path.exists() else {"stages": {}, "history": []},
    }


def get_run_parts(config, run_id, tail_lines=0):
    """Return the status, log and output path for every integrated part."""
    run_dir = _run_dir(config, run_id)
    if not run_dir.is_dir():
        raise FileNotFoundError("run not found: %s" % run_id)
    state_path = run_dir / "orchestration_state.json"
    state = read_json(state_path) if state_path.exists() else {"stages": {}}
    spec_path = run_dir / "spec.json"
    spec = read_json(spec_path) if spec_path.exists() else {}
    design_type = spec.get("design_type")
    screen_profile = (
        "nanobody" if design_type == "nanobody"
        else "scfv" if design_type == "antibody"
        else None
    )
    stage1_predictor = spec.get("workflow", {}).get("complex_prediction", "protenix")
    stage2_comparison = (
        spec.get("validation", {})
        .get("candidate_screening", {})
        .get("stage2_structural_comparison", {})
    )
    stage2_predictor = stage2_comparison.get("predictor", stage1_predictor)

    def predictor_output_name(predictor):
        return "%s_output" % predictor if predictor == "protenix" else predictor

    stage1_prediction_label = "stage1_%s" % stage1_predictor
    stage2_prediction_label = "stage2_%s" % stage2_predictor
    stage1_prediction_output = (
        "pipeline_output/stage1/%s" % predictor_output_name(stage1_predictor)
        if stage1_predictor == "protenix"
        else "pipeline_output/stage1/%s/predictions"
        % predictor_output_name(stage1_predictor)
    )
    stage2_prediction_output = (
        "pipeline_output/stage2/%s" % predictor_output_name(stage2_predictor)
        if stage2_predictor == "protenix"
        else "pipeline_output/stage2/%s/predictions"
        % predictor_output_name(stage2_predictor)
    )
    # Completed Stage 2 runs may use the numbered public layout.  Prefer those
    # paths for status/count reporting while retaining the runtime layout for
    # runs that are still in progress.
    numbered_stage2 = run_dir / "pipeline_output" / "stage2"
    stage2_partialflow_output = "pipeline_output/stage2/partialflow_output"
    stage2_on_target_output = "pipeline_output/stage2/design_pred_screen.csv"
    stage2_rosetta_output = "pipeline_output/stage2/fastrelax_output"
    stage2_offtarget_predictor_output = (
        "pipeline_output/stage2/protenix_offtarget_output"
    )
    stage2_offtarget_screen_output = (
        "pipeline_output/stage2/cldn18_1_offtarget_screen.csv"
    )
    if (numbered_stage2 / "01_partialflow").exists():
        stage2_partialflow_output = "pipeline_output/stage2/01_partialflow"
    numbered_predictor = numbered_stage2 / ("04_%s" % stage2_predictor)
    if numbered_predictor.exists():
        stage2_prediction_output = "pipeline_output/stage2/04_%s" % stage2_predictor
    if (numbered_stage2 / "05_on_target_screen.csv").exists():
        stage2_on_target_output = "pipeline_output/stage2/05_on_target_screen.csv"
    if (fastrelax_path(numbered_stage2 / "08_fastrelax")).exists():
        stage2_rosetta_output = "pipeline_output/stage2/08_fastrelax"
    if (numbered_stage2 / "07_cldn18_1_offtarget").exists():
        stage2_offtarget_predictor_output = (
            "pipeline_output/stage2/07_cldn18_1_offtarget"
        )
        stage2_offtarget_screen_output = (
            "pipeline_output/stage2/07_cldn18_1_offtarget/"
            "cldn18_1_offtarget_screen.csv"
        )
    definitions = [
        (
            "stage1_ppiflow",
            "ppiflow_primary",
            "logs/ppiflow.stage1.log",
            "pipeline_output/stage1/backbone_output",
            "pipeline_output/stage1/ppiflow_output",
        ),
        (
            stage1_prediction_label,
            stage1_predictor,
            "logs/adapter.%s.log" % stage1_predictor,
            stage1_prediction_output,
            "adapter_output/%s" % stage1_predictor,
        ),
        ("stage1_design_pred_screen", "ipsae", "logs/adapter.ipsae.log", "pipeline_output/stage1/design_pred_screen.csv", "adapter_output/ipsae"),
        ("stage1_interface_score", "stage1_interface_score", "logs/stage1.interface_score.log", "pipeline_output/stage1/fastrelax_output", "pipeline_output/stage1/fastrelax_raw_output"),
    ]
    offscreen = dict(config.get("off_target_screening", {}))
    offscreen.update(
        spec.get("validation", {})
        .get("candidate_screening", {})
        .get("off_target_screening", {})
    )
    definitions.extend(
        [
            ("stage2_partialflow_design", "ppiflow_iterative_redesign", "logs/ppiflow.stage2.log", stage2_partialflow_output, None),
            (
                stage2_prediction_label,
                "stage2_%s_prediction" % stage2_predictor,
                "logs/stage2.%s.log" % stage2_predictor,
                stage2_prediction_output,
                "adapter_output/stage2_%s" % stage2_predictor,
            ),
            ("stage2_on_target_screen", "stage2_on_target_screen", "logs/stage2.on_target.ipsae.log", stage2_on_target_output, None),
            ("stage2_fastrelax", "stage2_fastrelax", "logs/stage2.fastrelax.log", stage2_rosetta_output, "pipeline_output/stage2/fastrelax_raw_output"),
        ]
    )
    if offscreen.get("enabled", True):
        definitions.extend(
            [
                (
                    "stage2_cldn18_1_protenix",
                    "stage2_offtarget_protenix",
                    "logs/stage2.offtarget.protenix.log",
                    stage2_offtarget_predictor_output,
                    None,
                ),
                (
                    "stage2_cldn18_1_ipsae_delta",
                    "stage2_offtarget_ipsae",
                    "logs/stage2.offtarget.ipsae.log",
                    stage2_offtarget_screen_output,
                    None,
                ),
            ]
        )
    if screen_profile:
        definitions.insert(
            1,
            (
                "backbone_hotspot_screen",
                None,
                "logs/ppiflow.stage1.log",
                "pipeline_output/stage1/backbone_screen",
                None,
            ),
        )
    parts = []
    for label, state_key, log_rel, output_rel, legacy_output_rel in definitions:
        log_path = fastrelax_path(run_dir / log_rel) if log_rel else None
        output_path = fastrelax_path(run_dir / output_rel)
        if (
            legacy_output_rel
            and not output_path.exists()
            and fastrelax_path(run_dir / legacy_output_rel).exists()
        ):
            output_path = fastrelax_path(run_dir / legacy_output_rel)
        stage = fastrelax_stage(state.get("stages", {}), state_key) if state_key else {}
        item = {
            "part": label,
            "status": stage.get("status", "not_started"),
            "log": str(log_path) if log_path else None,
            "log_exists": bool(log_path and log_path.exists()),
            "output": str(output_path),
            "output_exists": output_path.exists(),
        }
        if stage.get("detail"):
            item["detail"] = stage["detail"]
        if label in {"stage1_interface_score", "stage2_fastrelax"} and (
            output_path.exists()
            and any(path.is_file() for path in output_path.rglob("*"))
        ):
            item["status"] = "complete"
        if label == "stage1_ppiflow" and screen_profile:
            grouped_dir = run_dir / "pipeline_output" / "stage1" / "backbone_output"
            native_dir = run_dir / "pipeline_output" / "stage1" / "ppiflow_output"
            if grouped_dir.exists():
                generated = len(list(grouped_dir.glob("*/*.pdb")))
            else:
                generated = len(list(native_dir.glob("*.pdb"))) if native_dir.exists() else 0
            expected = int(spec.get("generation", {}).get("samples", 0))
            item["backbone_generation_progress"] = {
                "completed": generated,
                "total": expected,
            }
        if label == "backbone_hotspot_screen":
            progress_path = output_path / "progress.json"
            summary_path = output_path / "summary.json"
            selection_path = output_path / "backbone_selection.json"
            progress = read_json(progress_path) if progress_path.exists() else None
            summary = read_json(summary_path) if summary_path.exists() else None
            selection = read_json(selection_path) if selection_path.exists() else None
            if selection:
                item["status"] = "complete"
            elif summary and screen_profile == "scfv":
                item["status"] = "awaiting_top_n_selection"
            elif summary:
                item["status"] = "complete"
            elif progress:
                item["status"] = progress.get("status", "running")
            else:
                item["status"] = "not_started"
            if progress:
                item["screening_progress"] = {
                    "completed": int(progress.get("completed", 0)),
                    "total": int(progress.get("total", 0)),
                    "accepted_so_far": int(progress.get("accepted_so_far", 0)),
                }
                if progress.get("current_backbone"):
                    item["current_backbone"] = progress["current_backbone"]
                if progress.get("detail"):
                    item["detail"] = progress["detail"]
            if summary:
                item["screening_result"] = {
                    "input_backbones": int(summary.get("input_backbones", 0)),
                    "accepted_backbones": int(summary.get("accepted_backbones", 0)),
                    "rejected_backbones": int(summary.get("rejected_backbones", 0)),
                }
                item["ranked_table"] = summary.get("sorted_csv")
            if selection:
                item["selected_top_n"] = int(selection.get("selected_top_n", 0))
                item["selection_file"] = str(selection_path)
            elif summary and screen_profile == "scfv":
                request_path = output_path / "selection_request.json"
                request = read_json(request_path) if request_path.exists() else {}
                item["action_required"] = request or {
                    "request_type": "scfv_backbone_top_n",
                    "prompt": "请人工复核排序表后，告诉Agent保留前多少号。",
                    "response_command": "scripts/agent-ab select-backbones %s <N>" % run_id,
                }
        candidate_paths = {
            stage1_prediction_label: (
                run_dir / ".agent_work" / "stage1" / stage1_predictor / "candidates.json",
                run_dir / "pipeline_output" / "stage1" / predictor_output_name(stage1_predictor) / "candidates.json",
                run_dir / "adapter_inputs" / stage1_predictor / "candidates.json",
            ),
            stage2_prediction_label: (
                run_dir / ".agent_work" / "stage2" / stage2_predictor / "candidates.json",
                run_dir / "pipeline_output" / "stage2" / predictor_output_name(stage2_predictor) / "candidates.json",
                run_dir / "adapter_inputs" / "stage2_comparison" / "candidates.json",
            ),
        }.get(label)
        if candidate_paths:
            candidates_path = next(
                (path for path in candidate_paths if path.exists()),
                candidate_paths[0],
            )
            candidates = read_json(candidates_path) if candidates_path.exists() else []
            completed = 0
            if output_path.exists():
                if label in (stage1_prediction_label, stage2_prediction_label) and (
                    (
                        label == stage1_prediction_label
                        and stage1_predictor == "protenix"
                    )
                    or (
                        label == stage2_prediction_label
                        and stage2_predictor == "protenix"
                    )
                ):
                    completed = len(list(output_path.glob("backbone*/*.cif")))
                else:
                    completed = len(list(output_path.rglob("metrics.json")))
            total_candidates = len(candidates)
            if not total_candidates:
                stage_name = "stage1" if label == stage1_prediction_label else "stage2"
                table_path = run_dir / "pipeline_output" / stage_name / "design_pred_screen.csv"
                if table_path.exists():
                    total_candidates = max(
                        0,
                        len(table_path.read_text(encoding="utf-8").splitlines()) - 1,
                    )
            item["candidate_progress"] = {
                "completed": completed,
                "total": total_candidates,
            }
        if tail_lines and log_path and log_path.exists():
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            item["log_tail"] = lines[-int(tail_lines):]
        parts.append(item)
    return {"run_id": run_id, "run_dir": str(run_dir), "parts": parts}
