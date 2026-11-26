"""Collapse completed Stage 1 runs into the small, user-facing layout."""

import csv
import json
import os
import re
import shutil
from pathlib import Path
from .fastrelax_naming import fastrelax_path

from .abmpnn_outputs import organize_abmpnn_output
from .backbone_outputs import organize_backbone_output
from .flowpacker_outputs import organize_flowpacker_output


BASE_FINAL_STAGE1_ENTRIES = {
    "backbone_output",
    "backbone_screen",
    "abmpnn_output",
    "flowpacker_output",
    "design_pred_screen.csv",
    "canonical_round0",
    "fastrelax_output",
    "stage1.log",
}


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass"}


def _remove(path):
    path = Path(path)
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _replace_directory(staging, output):
    staging, output = Path(staging), Path(output)
    _remove(output)
    staging.replace(output)


def _link_or_copy(source, output):
    source, output = Path(source), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(str(source), str(output))
    except OSError:
        shutil.copy2(source, output)


def _predictor_output_name(predictor):
    return "%s_output" % predictor if predictor == "protenix" else predictor


def _move_prediction(run_dir, stage1, predictor):
    output = stage1 / _predictor_output_name(predictor)
    if not output.exists():
        current = stage1 / predictor
        legacy = run_dir / "adapter_output" / predictor
        if current.exists():
            current.replace(output)
        elif legacy.exists():
            output.mkdir(parents=True)
            legacy.replace(output / "predictions")
        else:
            raise FileNotFoundError("Stage 1 %s output was not found" % predictor)
    predictions = output / "predictions"
    if not predictions.exists():
        predictions = output
    manifest = run_dir / "adapter_inputs" / predictor / "candidates.json"
    if manifest.exists() and not (output / "candidates.json").exists():
        shutil.copy2(manifest, output / "candidates.json")
    return predictions


def _design_pred_table(run_dir, stage1, predictions, predictor):
    output = stage1 / "design_pred_screen.csv"
    candidates = [
        output,
        stage1 / "screening_tables" / "round0_candidates.csv",
    ]
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise FileNotFoundError("Stage 1 design/prediction screening table was not found")
    rows = _read_csv(source)
    if not rows:
        raise RuntimeError("Stage 1 design/prediction screening table is empty")
    fields = list(rows[0])
    for row in rows:
        backbone_id = row.get("backbone_id", "")
        design_id = row.get("design_id") or row.get("sequence_id", "")
        if backbone_id.startswith("backbone") and design_id.startswith("backbone"):
            suffix = design_id[len("backbone") :]
            design = stage1 / "flowpacker_output" / backbone_id / ("flowpacker%s.pdb" % suffix)
            sequence = stage1 / "abmpnn_output" / backbone_id / ("abmpnn%s.fa" % suffix)
            if design.is_file():
                row["design_path"] = str(design.resolve())
            if sequence.is_file():
                row["sequence_path"] = str(sequence.resolve())
        value = row.get("prediction_path", "")
        if not value or Path(value).exists():
            continue
        parts = Path(value).parts
        try:
            index = parts.index(predictor)
        except ValueError:
            continue
        suffix = list(parts[index + 1 :])
        if suffix and suffix[0] == "predictions":
            suffix.pop(0)
        replacement = predictions.joinpath(*suffix)
        if replacement.exists():
            row["prediction_path"] = str(replacement.resolve())
    for field in ("design_path", "sequence_path"):
        if field not in fields:
            fields.append(field)
    _write_csv(output, rows, fields)
    return rows


def _compact_prediction_output(stage1, rows, predictor):
    """Keep eight predictions and one metrics table per backbone."""
    output = stage1 / _predictor_output_name(predictor)
    staging = stage1 / (".%s.tmp" % _predictor_output_name(predictor))
    _remove(staging)
    staging.mkdir()
    groups = set()
    for row in rows:
        backbone_id = row["backbone_id"]
        groups.add(backbone_id)
        design_id = row.get("design_id") or row["sequence_id"]
        if not design_id.startswith("backbone"):
            raise ValueError("unexpected prediction design ID: %s" % design_id)
        suffix = design_id[len("backbone"):]
        prediction_name = "pred" + suffix + ".cif"
        source = output / backbone_id / prediction_name
        if not source.is_file():
            raise FileNotFoundError("published Protenix file not found: %s" % source)
        _link_or_copy(source, staging / backbone_id / prediction_name)
        row["prediction_path"] = str(
            (output / backbone_id / prediction_name).resolve()
        )
    for backbone_id in groups:
        metrics = output / backbone_id / "metrics.json"
        if not metrics.is_file():
            raise FileNotFoundError("prediction metrics not found: %s" % metrics)
        _link_or_copy(metrics, staging / backbone_id / "metrics.json")
    fields = list(rows[0])
    if "prediction_path" not in fields:
        fields.append("prediction_path")
    _write_csv(stage1 / "design_pred_screen.csv", rows, fields)
    _write_csv(
        staging / ("%s_design_pred_screen.csv" % predictor), rows, fields
    )
    _replace_directory(staging, output)


def _offtarget_gate_by_id(stage1):
    path = Path(stage1) / "cldn18_1_offtarget_screen.csv"
    if not path.exists():
        return None
    records = {}
    for row in _read_csv(path):
        candidate_id = row.get("candidate_id")
        if not candidate_id:
            raise ValueError("off-target result has no candidate_id")
        if candidate_id in records:
            raise ValueError("duplicate off-target candidate_id: %s" % candidate_id)
        value = str(row.get("specificity_pass", "")).strip().lower()
        if value in {"1", "true", "yes", "y", "pass"}:
            records[candidate_id] = True
        elif value in {"0", "false", "no", "n", "fail"}:
            records[candidate_id] = False
        elif not value:
            records[candidate_id] = None
        else:
            raise ValueError(
                "invalid specificity_pass for %s: %s" % (candidate_id, value)
            )
    return records


def _canonical_round0(stage1, rows, off_target_gate=None):
    on_target_accepted = [row for row in rows if _truthy(row.get("filter_pass"))]
    if off_target_gate is not None:
        without_id = [row for row in on_target_accepted if not row.get("candidate_id")]
        if without_id:
            raise ValueError("accepted Stage 1 row has no candidate_id")
        missing = sorted(
            row["candidate_id"]
            for row in on_target_accepted
            if row["candidate_id"] not in off_target_gate
        )
        if missing:
            raise RuntimeError(
                "Stage 1 candidates missing from off-target results: %s"
                % ", ".join(missing)
            )
        accepted = [
            row
            for row in on_target_accepted
            if off_target_gate[row["candidate_id"]] is True
        ]
    else:
        accepted = on_target_accepted
    staging = stage1 / ".canonical_round0.tmp"
    _remove(staging)
    staging.mkdir()
    for row in accepted:
        backbone_id = row["backbone_id"]
        design_id = row.get("design_id") or row["sequence_id"]
        if not re.fullmatch(r"backbone\.0_\d+", backbone_id):
            raise ValueError("unexpected backbone ID: %s" % backbone_id)
        source = Path(row["design_path"])
        if not source.is_file():
            raise FileNotFoundError("accepted FlowPacker design not found: %s" % source)
        output = staging / backbone_id / (design_id + ".pdb")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.symlink_to(os.path.relpath(source, output.parent))
    _replace_directory(staging, stage1 / "canonical_round0")
    return accepted


def _fixed_target_rows(run_dir):
    candidates = [
        run_dir / "pipeline_output" / "stage2" / "fixed_positions.csv",
        run_dir / "pipeline_output" / "stage1" / "fixed_positions.csv",
    ]
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise FileNotFoundError("Rosetta fixed-position table was not found")
    result = {}
    for row in _read_csv(source):
        match = re.search(r"(backbone\.0_\d+)", row.get("filename", ""))
        if match:
            result[match.group(1)] = row
    return result


def _compact_rosetta(run_dir, stage1, accepted):
    legacy = fastrelax_path(stage1 / "fastrelax_raw_output")
    output = fastrelax_path(stage1 / "fastrelax_output")
    if not legacy.exists() and output.exists():
        return "complete"
    if not legacy.exists():
        return "awaiting_manual_fix" if accepted else "not_required_no_candidates"
    fixed = _fixed_target_rows(run_dir)
    staging = stage1 / ".fastrelax_output.tmp"
    _remove(staging)
    staging.mkdir()
    groups = {}
    for row in accepted:
        groups.setdefault(row["backbone_id"], []).append(row)
    for backbone_id, group_rows in sorted(groups.items()):
        group_dir = staging / backbone_id
        group_dir.mkdir()
        target = fixed.get(backbone_id, {})
        _write_csv(
            group_dir / "fixed_targets.csv",
            [
                {
                    "backbone_id": backbone_id,
                    "fixed_positions": target.get("fixed_positions", "NONE"),
                    "merge_seq": target.get("merge_seq", ""),
                }
            ],
            ["backbone_id", "fixed_positions", "merge_seq"],
        )
        for row in group_rows:
            design_id = row.get("design_id") or row["sequence_id"]
            design_dir = legacy / design_id
            source = design_dir / (design_id + "_0001.pdb")
            if not source.exists():
                matches = sorted(design_dir.glob("*_0001.pdb"))
                if not matches:
                    raise FileNotFoundError("Stage 1 score-only structure not found for %s" % design_id)
                source = matches[0]
            _link_or_copy(source, group_dir / (design_id + ".pdb"))
    _replace_directory(staging, output)
    return "complete"


def _stage_pending_rosetta_inputs(stage1, accepted):
    """Publish Stage 1 coordinate-preserving scoring inputs by backbone group."""
    output = fastrelax_path(stage1 / "fastrelax")
    staging = stage1 / ".fastrelax.tmp"
    _remove(staging)
    staging.mkdir()
    seen = set()
    for row in accepted:
        backbone_id = row["backbone_id"]
        if not re.fullmatch(r"backbone\.0_\d+", backbone_id):
            raise ValueError("unexpected Stage 1 scoring backbone ID: %s" % backbone_id)
        design_id = row.get("design_id") or row["sequence_id"]
        name = design_id + ".pdb"
        if name in seen:
            raise ValueError("duplicate Stage 1 scoring input name: %s" % name)
        seen.add(name)
        source = Path(row["design_path"])
        if not source.is_file():
            raise FileNotFoundError("Stage 1 scoring input not found: %s" % source)
        if not design_id.startswith(backbone_id + "_"):
            raise ValueError(
                "Stage 1 scoring candidate %s does not belong to group %s"
                % (design_id, backbone_id)
            )
        output_path = staging / backbone_id / name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.symlink_to(os.path.relpath(source, output_path.parent))
    _replace_directory(staging, output)
    return len(seen)


def prepare_stage1_rosetta_inputs(run_dir):
    """Create grouped score-only inputs without changing Stage 1 coordinates."""
    run_dir = Path(run_dir).resolve()
    stage1 = run_dir / "pipeline_output" / "stage1"
    source = next(
        (
            path
            for path in (
                stage1 / "design_pred_screen.csv",
                stage1 / "screening_tables" / "round0_candidates.csv",
            )
            if path.exists()
        ),
        None,
    )
    if source is None:
        raise FileNotFoundError("Stage 1 design/prediction table was not found")
    rows = _read_csv(source)
    if not rows:
        raise RuntimeError("Stage 1 design/prediction table is empty")
    accepted = _canonical_round0(stage1, rows)
    _stage_pending_rosetta_inputs(stage1, accepted)
    groups = {}
    for row in accepted:
        backbone_id = row["backbone_id"]
        design_id = row.get("design_id") or row["sequence_id"]
        groups.setdefault(backbone_id, []).append(
            {
                "candidate_id": row.get("candidate_id", design_id),
                "design_id": design_id,
                "input": str(
                    fastrelax_path(stage1 / "fastrelax") / backbone_id / (design_id + ".pdb")
                ),
            }
        )
    manifest = {
        "schema_version": 1,
        "grouping_key": "backbone_id",
        "cross_group_merge": False,
        "group_count": len(groups),
        "candidate_count": len(accepted),
        "groups": groups,
    }
    manifest_path = stage1 / "rosetta_group_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "manifest": str(manifest_path)}


def _normalize_backbone_screen(stage1):
    root = stage1 / "backbone_screen"
    summary = {}
    selection = {}
    if (root / "summary.json").exists():
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if (root / "backbone_selection.json").exists():
        selection = json.loads(
            (root / "backbone_selection.json").read_text(encoding="utf-8")
        )
    output = root / "backbone_screen.csv"
    if not output.exists():
        sources = sorted(root.glob("*.csv"))
        if not sources:
            raise FileNotFoundError("backbone screening CSV was not found")
        shutil.copy2(sources[0], output)
    rows = _read_csv(output)
    if rows:
        fields = list(rows[0])
        for row in rows:
            backbone_id = row.get("backbone_id", "")
            grouped = (
                stage1
                / "backbone_output"
                / backbone_id
                / (backbone_id + ".pdb")
            )
            if grouped.is_file() and "source_pdb" in row:
                row["source_pdb"] = str(grouped.resolve())
        _write_csv(output, rows, fields)
    for path in list(root.iterdir()):
        if path != output:
            _remove(path)
    return summary, selection


def _count_files(root, pattern):
    root = Path(root)
    return sum(1 for path in root.glob(pattern) if path.is_file()) if root.exists() else 0


def _percent(passed, total):
    return "%.2f%%" % (100.0 * passed / total) if total else "0.00%"


def _write_log(
    stage1, rows, accepted, screen_summary, selection, predictor, rosettafix_status
):
    all_groups = {row["backbone_id"] for row in rows}
    passed_groups = {row["backbone_id"] for row in accepted}
    ppiflow = _count_files(stage1 / "backbone_output", "*/*.pdb")
    sequences = _count_files(stage1 / "abmpnn_output", "*/*.fa")
    designs = _count_files(stage1 / "flowpacker_output", "*/*.pdb")
    predictor_output = _predictor_output_name(predictor)
    prediction_pattern = (
        "*/*.cif" if predictor == "protenix" else "**/metrics.json"
    )
    predictions = _count_files(stage1 / predictor_output, prediction_pattern)
    fixed_structures = _count_files(fastrelax_path(stage1 / "fastrelax_output"), "*/*.pdb")
    fixed_tables = list((fastrelax_path(stage1 / "fastrelax_output")).glob("*/fixed_targets.csv"))
    usable_fixed = 0
    for path in fixed_tables:
        table = _read_csv(path)
        if table and table[0].get("fixed_positions", "NONE") not in {"", "NONE"}:
            usable_fixed += 1
    backbone_rows = _read_csv(stage1 / "backbone_screen" / "backbone_screen.csv")
    screened = int(screen_summary.get("input_backbones", len(backbone_rows) or ppiflow))
    geometry_from_table = sum(
        _truthy(row.get("filter_pass")) for row in backbone_rows
    )
    geometry_passed = int(
        screen_summary.get("accepted_backbones", geometry_from_table or screened)
    )
    selected = int(selection.get("selected_top_n", len(all_groups)))
    expected_sequences = selected * 8
    off_target_layout = []
    if (stage1 / "offtarget_screen").exists():
        off_target_layout = [
            "cldn18_1_offtarget_screen.csv: 同一批抗体对CLDN18.1的第二轮反靶评分；复用第一轮S_on。",
            "offtarget_screen/: CLDN18.1反靶明细、状态与汇总。",
            "protenix_offtarget_output/: 未重新生成骨架的CLDN18.1复合物预测。",
        ]
    lines = [
        "Stage1 final layout",
        "",
        "步骤与通过率",
        "1. PPIFlow骨架生成: %d/%d，完成率 %s。脚本: scripts/ppiflow_pipeline_screened.py"
        % (ppiflow, screened, _percent(ppiflow, screened)),
        "2. 骨架几何筛选: %d/%d，通过率 %s；Top-N保留: %d/%d，通过率 %s。脚本: agent_ab/backbone_screen.py"
        % (
            geometry_passed,
            screened,
            _percent(geometry_passed, screened),
            selected,
            geometry_passed,
            _percent(selected, geometry_passed),
        ),
        "3. AbMPNN序列生成: %d/%d，每个骨架8条。脚本: scripts/abmpnn_adapter/protein_mpnn_run.py"
        % (sequences, expected_sequences),
        "4. FlowPacker结构生成: %d/%d，完成率 %s。脚本: PPIFlow/tools/flowpacker/sampler_pdb_antibody.py"
        % (designs, sequences, _percent(designs, sequences)),
        "5. %s结构预测: %d/%d，完成率 %s。脚本: scripts/%s_batch_adapter.py"
        % (predictor, predictions, designs, _percent(predictions, designs), predictor),
        "6. Stage 1 design/pred结构筛选: %d/%d，通过率 %s；保留骨架组: %d/%d，通过率 %s。阈值: antibody_pLDDT>=0.60, pTM>=0.60, ipTM>=0.25, RMSD<6.0A；不使用S_on/S_off/delta。"
        % (
            len(accepted),
            len(rows),
            _percent(len(accepted), len(rows)),
            len(passed_groups),
            len(all_groups),
            _percent(len(passed_groups), len(all_groups)),
        ),
        (
            "7. Rosetta界面能量评分（保持Stage 1坐标）: 等待执行；Stage 2尚未开始。"
            if rosettafix_status == "awaiting_manual_fix"
            else "7. Rosetta界面能量评分（不做FastRelax）: %d/%d，完成率 %s；有可用固定靶点的骨架组: %d/%d。脚本: scripts/parallel_fastrelax.py --preserve-coordinates"
            % (
                fixed_structures,
                len(accepted),
                _percent(fixed_structures, len(accepted)),
                usable_fixed,
                len(passed_groups),
            )
        ),
        "",
        "目录内容",
        "backbone_output/<backbone_id>/: 每个文件夹一个PPIFlow骨架PDB。",
        "backbone_screen/backbone_screen.csv: 第一次骨架筛选表。",
        "abmpnn_output/<backbone_id>/: 每个骨架8条FASTA，短名称为abmpnn.0_<骨架号>_<序列号>。",
        "flowpacker_output/<backbone_id>/: 每个骨架8个设计PDB，短名称为flowpacker.0_<骨架号>_<序列号>。",
        "%s/<backbone_id>/: 8个pred CIF及一份metrics.json；design PDB只保留在flowpacker_output。"
        % predictor_output,
        "design_pred_screen.csv: 全部design/pred指标、是否通过及路径。",
        *off_target_layout,
        "canonical_round0/<backbone_id>/: 仅通过筛选的设计结构；使用指向flowpacker_output的软链接，不重复占用空间；无通过序列的骨架不建文件夹。",
        "fastrelax_output/<backbone_id>/: 兼容目录名；PDB坐标与Stage 1输入一致，并含fixed_targets.csv。",
        "stage1.log: 本文件。",
        "",
        "已删除Stage1模型临时输入、重复副本、旧筛选中间目录、Rosetta日志/score/xml及旧canonical_outputs。",
    ]
    (stage1 / "stage1.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize_stage1_layout(run_dir):
    """Finalize a completed run and delete only redundant Stage 1 intermediates."""
    run_dir = Path(run_dir).resolve()
    stage1 = run_dir / "pipeline_output" / "stage1"
    if not stage1.is_dir():
        raise FileNotFoundError("Stage 1 directory not found: %s" % stage1)
    from .stage1_numbered_layout import existing_numbered_stage1
    numbered = existing_numbered_stage1(stage1)
    if numbered is not None:
        return numbered
    spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
    if not (stage1 / "backbone_output").exists():
        organize_backbone_output(
            stage1 / "ppiflow_output", stage1 / "backbone_output"
        )
    expected_sequences = int(spec.get("generation", {}).get("num_sequences_per_backbone", 8))
    sequence_summary = organize_abmpnn_output(
        stage1 / "abmpnn_output", expected_sequences=expected_sequences
    )
    organize_flowpacker_output(
        stage1 / "flowpacker_output",
        expected_backbones=sequence_summary["backbone_count"],
        expected_sequences=expected_sequences,
    )
    predictor = spec["workflow"]["complex_prediction"]
    predictor_output = _predictor_output_name(predictor)
    final_entries = set(BASE_FINAL_STAGE1_ENTRIES) | {predictor_output}
    for optional in (
        "fixed_positions.csv",
        "rosetta_group_manifest.json",
        "reuse_manifest.json",
    ):
        if (stage1 / optional).exists():
            final_entries.add(optional)
    off_target_entries = {
        "cldn18_1_offtarget_screen.csv",
        "offtarget_screen",
        "protenix_offtarget_output",
    }
    present_off_target = {
        name for name in off_target_entries if (stage1 / name).exists()
    }
    if present_off_target and present_off_target != off_target_entries:
        missing = sorted(off_target_entries - present_off_target)
        raise RuntimeError(
            "refusing Stage 1 cleanup; incomplete off-target outputs: %s"
            % missing
        )
    final_entries.update(present_off_target)
    predictions = _move_prediction(run_dir, stage1, predictor)
    rows = _design_pred_table(run_dir, stage1, predictions, predictor)
    off_target_gate = _offtarget_gate_by_id(stage1)
    accepted = _canonical_round0(stage1, rows, off_target_gate)
    rosettafix_status = _compact_rosetta(run_dir, stage1, accepted)
    if rosettafix_status == "awaiting_manual_fix":
        _stage_pending_rosetta_inputs(stage1, accepted)
        final_entries.add("fastrelax")
        final_entries.discard("fastrelax_output")
    elif rosettafix_status == "not_required_no_candidates":
        final_entries.discard("fastrelax_output")
    screen_summary, selection = _normalize_backbone_screen(stage1)
    if predictor == "protenix":
        _compact_prediction_output(stage1, rows, predictor)

    # Older runs may still use legacy directory names. Preserve both the actual
    # output directory and any compatibility aliases when compacting the layout.
    final_entries = {fastrelax_path(stage1 / name).name for name in final_entries}
    for path in stage1.iterdir():
        if path.is_symlink() and path.resolve().parent == stage1.resolve():
            if path.resolve().name in final_entries:
                final_entries.add(path.name)
    required = final_entries - {"stage1.log"}
    missing = sorted(name for name in required if not (stage1 / name).exists())
    if missing:
        raise RuntimeError("refusing Stage 1 cleanup; missing final outputs: %s" % missing)
    archive = stage1 / ".work" / "legacy_layout"
    for path in list(stage1.iterdir()):
        if path.name in final_entries or path.name == ".work":
            continue
        archive.mkdir(parents=True, exist_ok=True)
        destination = archive / path.name
        if destination.exists() or destination.is_symlink():
            raise RuntimeError(
                "refusing Stage 1 archive overwrite: %s" % destination
            )
        path.replace(destination)

    for legacy in (
        run_dir / "adapter_output" / "ipsae",
        run_dir / "adapter_inputs" / predictor,
        run_dir / ".agent_work" / "stage1" / predictor,
        run_dir / ".agent_work" / "stage1" / "offtarget_protenix",
    ):
        _remove(legacy)
    summary = run_dir / "pipeline_output" / "screening_summary.csv"
    if summary.exists():
        summary.unlink()
    _write_log(
        stage1,
        rows,
        accepted,
        screen_summary,
        selection,
        predictor,
        rosettafix_status,
    )
    summary = {
        "stage1": str(stage1),
        "design_pred_candidates": len(rows),
        "accepted_designs": len(accepted),
        "accepted_backbone_groups": len({row["backbone_id"] for row in accepted}),
        "fastrelax_status": rosettafix_status,
        "stage1_interface_score_status": rosettafix_status,
        "rosettafix_status": rosettafix_status,  # Legacy API field.
    }
    from .stage1_numbered_layout import publish_numbered_stage1
    return publish_numbered_stage1(stage1, predictor, summary)
