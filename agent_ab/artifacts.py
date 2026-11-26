from .fastrelax_naming import fastrelax_path
"""Canonical two-round output names and auditable result-path manifests."""

import csv
import json
import re
from pathlib import Path

from .naming import parse_backbone_id, stage1_backbone_id, stage2_backbone_id
from .util import read_json, write_json


def _prefer_current_path(current, legacy):
    current = Path(current)
    legacy = Path(legacy)
    return current if current.exists() or not legacy.exists() else legacy


def _write_csv(path, records, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return path


def _native_key(link_name, round_index):
    try:
        stage, parent_index, sample_index = parse_backbone_id(link_name)
    except ValueError:
        stem = Path(link_name).stem
        if round_index == 0:
            match = re.search(r"_(\d+)$", stem)
            if match:
                return (int(match.group(1)),)
        else:
            match = re.search(r"_(\d+)_sample(\d+)$", stem)
            if match:
                return tuple(int(value) for value in match.groups())
        raise
    expected_stage = int(round_index) + 1
    if stage != expected_stage:
        raise ValueError(
            "expected Stage %d backbone, got %s" % (expected_stage, link_name)
        )
    return (parent_index,) if stage == 1 else (parent_index, sample_index)


def _prediction_path(candidate_dir):
    for name in ("pred.cif", "predicted.cif", "pred.pdb", "predicted.pdb"):
        path = Path(candidate_dir) / name
        if path.exists():
            return path
    return None


def _jobs_by_design(manifest_path, prediction_dir):
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return {}
    result = {}
    for job in read_json(manifest_path):
        design = Path(job["pdb"])
        candidate_dir = Path(prediction_dir) / job.get(
            "output_relpath", job["candidate_id"]
        )
        result[str(design.resolve())] = {
            "job": job,
            "prediction": _prediction_path(candidate_dir),
        }
        result[design.name] = result[str(design.resolve())]
        if job.get("candidate_id"):
            result[job["candidate_id"]] = result[str(design.resolve())]
    return result


def _stage1_metrics(run_dir):
    result = {}
    run_dir = Path(run_dir)
    root = _prefer_current_path(
        run_dir / "pipeline_output" / "stage1" / "ipsae" / "results",
        run_dir / "adapter_output" / "ipsae",
    )
    if not root.exists():
        return result
    for path in sorted(root.glob("backbone_*/filter_results.json")):
        for record in read_json(path):
            for key in (
                record.get("original_source_pdb"),
                record.get("source_pdb"),
                record.get("candidate_id"),
            ):
                if key:
                    result[str(key)] = record
                    result[Path(str(key)).name] = record
    return result


def _stage2_metrics(run_dir):
    result = {}
    root = Path(run_dir) / "pipeline_output" / "stage2" / "ipsae" / "results"
    if not root.exists():
        root = (
            Path(run_dir)
            / "pipeline_output"
            / "stage2"
            / ".work"
            / "legacy_layout"
            / "ipsae"
            / "results"
        )
    records = []
    if root.exists():
        for path in sorted(root.glob("*/filter_results.json")):
            records.extend(read_json(path))
    else:
        path = Path(run_dir) / "stage2_structural_comparison" / "pred_design_comparison.json"
        if path.exists():
            records = read_json(path)
    for record in records:
        for key in (
            record.get("original_source_pdb"),
            record.get("source_pdb"),
            record.get("design_pdb"),
            record.get("candidate_id"),
        ):
            if key:
                result[str(key)] = record
                result[Path(str(key)).name] = record
    return result


def _round_locations(run_dir, spec, round_index):
    run_dir = Path(run_dir)
    stage = "stage1" if round_index == 0 else "stage2"
    if round_index == 0:
        predictor = spec["workflow"]["complex_prediction"]
        predictor_dir = {
            "opendde": "opendde_output",
            "protenix": "protenix_output",
        }.get(predictor, predictor)
        tool_root = run_dir / "pipeline_output" / "stage1" / predictor_dir
        manifest = _prefer_current_path(
            run_dir / ".agent_work" / "stage1" / predictor / "candidates.json",
            _prefer_current_path(
                tool_root / "candidates.json",
                run_dir / "adapter_inputs" / predictor / "candidates.json",
            ),
        )
        prediction_dir = _prefer_current_path(
            run_dir / ".agent_work" / "stage1" / predictor / "predictions",
            _prefer_current_path(
                tool_root / "predictions",
                run_dir / "adapter_output" / predictor,
            ),
        )
    else:
        comparison = spec["validation"]["candidate_screening"].get(
            "stage2_structural_comparison", {}
        )
        predictor = comparison.get(
            "predictor", spec["workflow"]["complex_prediction"]
        )
        predictor_dir = {
            "opendde": "opendde_output",
            "protenix": "protenix_output",
        }.get(predictor, predictor)
        tool_root = run_dir / "pipeline_output" / "stage2" / predictor_dir
        manifest = _prefer_current_path(
            run_dir / ".agent_work" / "stage2" / predictor / "candidates.json",
            _prefer_current_path(
                tool_root / "candidates.json",
                run_dir / "adapter_inputs" / "stage2_comparison" / "candidates.json",
            ),
        )
        prediction_dir = _prefer_current_path(
            run_dir / ".agent_work" / "stage2" / predictor / "predictions",
            _prefer_current_path(
                tool_root / "predictions",
                run_dir / "adapter_output" / ("stage2_%s" % predictor),
            ),
        )
    return {
        "stage": stage,
        "sequence_csv": run_dir / "pipeline_output" / stage / "mpnn_pdbs" / "mpnn_seqs.csv",
        "backbone_dir": run_dir / "pipeline_output" / stage / "mpnn_pdbs",
        "design_dir": (
            run_dir / "pipeline_output" / stage / "flowpacker_output"
        ),
        "prediction_manifest": manifest,
        "prediction_dir": prediction_dir,
        "predictor": predictor,
        "public_prediction_root": tool_root,
    }


def _native_design_path(design_root, design_stem, round_index):
    design_root = Path(design_root)
    if round_index == 0:
        match = re.search(r"(backbone\.[A-Za-z0-9-]+_\d+)_(\d+)$", design_stem)
        if match:
            filename = "flowpacker%s_%d.pdb" % (
                match.group(1)[len("backbone") :],
                int(match.group(2)),
            )
            canonical = (
                design_root
                / match.group(1)
                / filename
            )
            if canonical.exists():
                return canonical
        legacy = design_root / "run_1" / (design_stem + ".pdb")
        if legacy.exists():
            return legacy
    else:
        match = re.search(
            r"(backbone\.(?:1_\d+_\d+|0_\d+_sample\d+))_(\d+)$",
            design_stem,
        )
        if match:
            filename = "flowpacker%s_%d.pdb" % (
                match.group(1)[len("backbone") :],
                int(match.group(2)),
            )
            canonical = (
                design_root
                / match.group(1)
                / filename
            )
            if canonical.exists():
                return canonical
        legacy = design_root / "run_1" / (design_stem + ".pdb")
        if legacy.exists():
            return legacy
    return design_root / (design_stem + ".pdb")


def materialize_round_outputs(run_dir, spec, round_index):
    """Write table-only round summaries using the original backbone numbers."""
    run_dir = Path(run_dir)
    locations = _round_locations(run_dir, spec, round_index)
    sequence_csv = locations["sequence_csv"]
    if not sequence_csv.exists():
        raise FileNotFoundError("sequence index not found: %s" % sequence_csv)
    with sequence_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped = {}
    for row in rows:
        key = _native_key(row["link_name"], round_index)
        grouped.setdefault(key, []).append(row)

    output_root = run_dir / "pipeline_output" / locations["stage"] / "screening_tables"
    jobs = _jobs_by_design(
        locations["prediction_manifest"], locations["prediction_dir"]
    )
    metrics = _stage1_metrics(run_dir) if round_index == 0 else _stage2_metrics(run_dir)
    records = []
    for native_key in sorted(grouped):
        native_rows = sorted(grouped[native_key], key=lambda item: int(item["seq_idx"]))
        link_name = native_rows[0]["link_name"]
        native_backbone = locations["backbone_dir"] / link_name
        if round_index == 0:
            backbone_id = stage1_backbone_id(native_key[0])
            parent_backbone_id = ""
            refined_sample_index = ""
        else:
            backbone_id = stage2_backbone_id(*native_key)
            parent_backbone_id = stage1_backbone_id(native_key[0])
            refined_sample_index = native_key[1]
        for row in native_rows:
            sequence_index = int(row["seq_idx"])
            design_id = "%s_%d" % (backbone_id, sequence_index)
            design_stem = Path(link_name).stem + "_" + str(row["seq_idx"])
            native_design = _native_design_path(
                locations["design_dir"], design_stem, round_index
            )
            if not native_backbone.exists():
                raise FileNotFoundError("backbone PDB not found: %s" % native_backbone)
            if not native_design.exists():
                raise FileNotFoundError("FlowPacker design PDB not found: %s" % native_design)
            candidate_id = (
                "backbone_%03d_design_%03d" % (native_key[0], sequence_index)
                if round_index == 0
                else None
            )
            job_value = (
                jobs.get(str(native_design.resolve()))
                or jobs.get(native_design.name)
                or (jobs.get(candidate_id) if candidate_id else None)
            )
            prediction = job_value.get("prediction") if job_value else None
            metric = (
                metrics.get(str(native_design.resolve()))
                or metrics.get(native_design.name)
                or (
                    metrics.get(job_value["job"]["candidate_id"])
                    if job_value
                    else None
                )
                or {}
            )
            sequence_path = ""
            candidate_sequence = (
                run_dir
                / "pipeline_output"
                / locations["stage"]
                / "abmpnn_output"
                / backbone_id
                / ("abmpnn%s.fa" % design_id[len("backbone") :])
            )
            if candidate_sequence.exists():
                sequence_path = str(candidate_sequence.resolve())
            record = {
                "round": round_index,
                "candidate_id": (
                    job_value["job"].get("candidate_id", "")
                    if job_value
                    else (candidate_id or design_id)
                ),
                "backbone_id": backbone_id,
                "sequence_id": design_id,
                "design_id": design_id,
                "prediction_id": (
                    job_value["job"].get("prediction_name")
                    or job_value["job"].get("candidate_id", "")
                    if job_value
                    else ""
                ),
                "parent_backbone_id": parent_backbone_id,
                "refined_sample_index": refined_sample_index,
                "sequence_index": sequence_index,
                "filter_pass": (
                    metric.get("filter_pass", "")
                    if round_index == 0
                    else metric.get(
                        "filter_pass", metric.get("structurally_similar", "")
                    )
                ),
                "enters_partialflow": (
                    bool(metric.get("filter_pass")) if round_index == 0 else ""
                ),
                "antibody_plddt": metric.get("antibody_plddt", ""),
                "ptm": metric.get("ptm", ""),
                "iptm_global": metric.get("iptm_global", ""),
                "ipsae_antibody_to_target": metric.get(
                    "ipsae_antibody_to_target", ""
                ),
                "ipsae_target_to_antibody": metric.get(
                    "ipsae_target_to_antibody", ""
                ),
                "S_on": metric.get("S_on", ""),
                "ipsae_min": metric.get("ipsae_min", metric.get("S_on", "")),
                "design_pred_rmsd": metric.get("design_pred_rmsd", ""),
                "rejection_reasons": ";".join(metric.get("rejection_reasons", [])),
                "structurally_similar": metric.get("structurally_similar", ""),
                "backbone_path": str(native_backbone.resolve()),
                "sequence_path": sequence_path,
                "design_path": str(native_design.resolve()),
                "prediction_path": str(prediction.resolve()) if prediction else "",
            }
            records.append(record)

    fields = [
        "round",
        "candidate_id",
        "backbone_id",
        "sequence_id",
        "design_id",
        "prediction_id",
        "parent_backbone_id",
        "refined_sample_index",
        "sequence_index",
        "filter_pass",
        "enters_partialflow",
        "antibody_plddt",
        "ptm",
        "iptm_global",
        "ipsae_antibody_to_target",
        "ipsae_target_to_antibody",
        "S_on",
        "ipsae_min",
        "design_pred_rmsd",
        "rejection_reasons",
        "structurally_similar",
        "backbone_path",
        "sequence_path",
        "design_path",
        "prediction_path",
    ]
    table_dir = output_root
    all_table = _write_csv(
        table_dir / ("round%s_candidates.csv" % round_index), records, fields
    )
    public_screen_table = None
    if locations["predictor"] == "protenix":
        public_screen_table = _write_csv(
            locations["public_prediction_root"]
            / ("%s_design_pred_screen.csv" % locations["predictor"]),
            records,
            fields,
        )
    accepted_table = None
    accepted = []
    if round_index == 0:
        accepted = [record for record in records if record["enters_partialflow"]]
        accepted_table = _write_csv(
            table_dir / "partialflow_accepted_candidates.csv", accepted, fields
        )
    summary = {
        "round": round_index,
        "backbones": len(grouped),
        "sequences": len(records),
        "designs": sum(bool(record["design_path"]) for record in records),
        "predictions": sum(bool(record["prediction_path"]) for record in records),
        "accepted_for_partialflow": len(accepted) if round_index == 0 else None,
        "output_root": str(output_root.absolute()),
        "all_candidates_table": str(all_table.absolute()),
        "predictor_screen_table": (
            str(public_screen_table.absolute()) if public_screen_table else None
        ),
        "partialflow_accepted_table": (
            str(accepted_table.absolute()) if accepted_table else None
        ),
    }
    write_json(output_root / "summary.json", summary)
    return summary


def write_result_paths(run_dir):
    """Write and return the paths printed when the pipeline finishes."""
    run_dir = Path(run_dir).resolve()
    root = run_dir / "pipeline_output"
    stage2 = root / "stage2"
    numbered = (stage2 / "00_stage2_plan" / "organization_manifest.json").is_file()
    if numbered:
        round1_table = stage2 / "05_on_target_screen.csv"
        canonical_round1 = stage2 / "06_screened_candidates"
        stage2_metrics = stage2 / "05_on_target_screen.csv"
        stage2_log = stage2 / "00_stage2_plan" / "organization_manifest.json"
        offtarget_screen = (
            stage2
            / "07_cldn18_1_offtarget"
            / "cldn18_1_offtarget_screen.csv"
        )
        offtarget_summary = (
            stage2 / "07_cldn18_1_offtarget" / "offtarget_screen" / "summary.json"
        )
        stage2_rosettafix = fastrelax_path(stage2 / "08_fastrelax")
    else:
        round1_table = stage2 / "design_pred_screen.csv"
        canonical_round1 = stage2 / "canonical_round1"
        stage2_metrics = stage2 / "evaluation_metrics"
        stage2_log = stage2 / "stage2.log"
        offtarget_screen = stage2 / "cldn18_1_offtarget_screen.csv"
        offtarget_summary = stage2 / "offtarget_screen" / "summary.json"
        stage2_rosettafix = fastrelax_path(stage2 / "fastrelax_output")
    stage1 = root / "stage1"
    stage1_numbered = (
        stage1 / "00_stage1_plan" / "organization_manifest.json"
    ).is_file()
    if stage1_numbered:
        stage1_screen = stage1 / "06_design_pred_screen.csv"
        canonical_round0 = stage1 / "07_screened_candidates"
        stage1_fastrelax = fastrelax_path(stage1 / "08_interface_energy")
        backbone_screen = (
            stage1 / "02_backbone_screen" / "backbone_screen.csv"
        )
        stage1_log = stage1 / "09_stage1_summary.log"
    else:
        stage1_screen = stage1 / "design_pred_screen.csv"
        canonical_round0 = stage1 / "canonical_round0"
        stage1_fastrelax = fastrelax_path(stage1 / "fastrelax_output")
        backbone_screen = (
            stage1 / "backbone_screen" / "backbone_screen.csv"
        )
        stage1_log = stage1 / "stage1.log"

    values = {
        "run_directory": str(run_dir),
        "design_pred_screen": str(stage1_screen),
        "canonical_round0": str(canonical_round0),
        "fastrelax_output": str(stage1_fastrelax),
        "round1_comparison_table": str(round1_table),
        "canonical_round1": str(canonical_round1),
        "stage2_evaluation_metrics": str(stage2_metrics),
        "stage2_log": str(stage2_log),
        "backbone_screen_table": str(backbone_screen),
        "stage1_log": str(stage1_log),
        "offtarget_screen": str(offtarget_screen),
        "offtarget_summary": str(offtarget_summary),
        "stage2_fastrelax_output": str(stage2_rosettafix),
        "stage2_iggm_output": str(stage2 / "09_IgGM"),
        "stage3_directory": str(root / "stage3"),
        "stage3_flowpacker_designs": str(
            root / "stage3" / "02_flowpacker_sidechain_packing"
        ),
        "stage3_on_target_predictions": str(
            root / "stage3" / "03_cldn18_2_structure_prediction"
        ),
        "stage3_on_target_screen": str(
            root / "stage3" / "04_cldn18_2_on_target_screen" / "on_target_screen.csv"
        ),
        "stage3_off_target_predictions": str(
            root / "stage3" / "05_cldn18_1_structure_prediction"
        ),
        "stage3_off_target_screen": str(
            root
            / "stage3"
            / "06_cldn18_1_off_target_screen"
            / "cldn18_1_offtarget_screen.csv"
        ),
        "stage3_final_candidates": str(
            root / "stage3" / "07_final_candidates"
        ),
    }
    text = "\n".join("%s\t%s" % item for item in values.items()) + "\n"
    # Keep serialized fields readable by existing downstream clients.
    values["rosettafix_output"] = values["fastrelax_output"]
    values["stage2_rosettafix_output"] = values["stage2_fastrelax_output"]
    write_json(run_dir / "result_paths.json", values)
    (run_dir / "result_paths.txt").write_text(text, encoding="utf-8")
    values["result_paths_json"] = str(run_dir / "result_paths.json")
    values["result_paths_text"] = str(run_dir / "result_paths.txt")
    return values


def write_screening_summary(run_dir):
    """Write one compact CSV with pass counts for every available screen."""
    run_dir = Path(run_dir).resolve()
    rows = []

    def add(stage, label, unit, passed, total, criterion):
        if passed is None or total in (None, 0):
            return
        passed, total = int(passed), int(total)
        rows.append(
            {
                "step": len(rows) + 1,
                "stage": stage,
                "screen": label,
                "unit": unit,
                "passed": passed,
                "total": total,
                "pass_rate_percent": "%.2f" % (100.0 * passed / total),
                "criterion": criterion,
            }
        )

    screen_roots = [run_dir / "pipeline_output" / "stage1" / "backbone_screen"]
    screen_roots = [path for path in screen_roots if path.exists()]
    if screen_roots:
        screen_root = screen_roots[0]
        summary_path = screen_root / "summary.json"
        if summary_path.exists():
            summary = read_json(summary_path)
            add(
                "backbone_hotspot_screen",
                "hotspot geometry",
                "backbone",
                summary.get("accepted_backbones"),
                summary.get("input_backbones"),
                summary.get("pass_rule", "hotspot distance gate"),
            )
        selection_path = screen_root / "backbone_selection.json"
        if selection_path.exists():
            selection = read_json(selection_path)
            add(
                "backbone_top_n_selection",
                "Top-N selection",
                "backbone",
                selection.get("selected_top_n"),
                selection.get("available_passing_backbones"),
                selection.get("selection_order", "ranked Top-N"),
            )

    stage1_path = _prefer_current_path(
        run_dir / "pipeline_output" / "stage1" / "ipsae" / "results" / "summary.json",
        run_dir / "adapter_output" / "ipsae" / "summary.json",
    )
    stage1_groups = None
    if stage1_path.exists():
        stage1 = read_json(stage1_path)
        add(
            "stage1_pred_design_filter",
            "Stage1 design-predict loose filter",
            "candidate",
            stage1.get("accepted"),
            stage1.get("evaluated"),
            "pLDDT/pTM/ipTM/design-predict RMSD",
        )
        groups = stage1.get("groups", {})
        if isinstance(groups, dict) and groups:
            stage1_groups = sum(
                int(item.get("accepted", 0)) > 0 for item in groups.values()
            )
            add(
                "stage1_group_survival",
                "groups retaining >=1 candidate",
                "backbone group",
                stage1_groups,
                len(groups),
                "at least one Stage1 candidate passed",
            )

    partial_root = run_dir / "pipeline_output" / "stage2" / "partial_output"
    if not partial_root.exists():
        partial_root = (
            run_dir
            / "pipeline_output"
            / "stage2"
            / ".work"
            / "legacy_layout"
            / "partial_output"
        )
    partial_groups = {
        pdb.parent.name for pdb in partial_root.glob("*/sample*.pdb")
    }
    if stage1_groups is not None and partial_groups:
        add(
            "partialflow_input_gate",
            "Rosetta fixed-position gate",
            "backbone group",
            len(partial_groups),
            stage1_groups,
            "usable fixed positions and completed PartialFlow backbone",
        )

    stage2_path = (
        run_dir
        / "pipeline_output"
        / "stage2"
        / "ipsae"
        / "results"
        / "summary.json"
    )
    if not stage2_path.exists():
        stage2_path = (
            run_dir
            / "pipeline_output"
            / "stage2"
            / ".work"
            / "legacy_layout"
            / "ipsae"
            / "results"
            / "summary.json"
        )
    if stage2_path.exists():
        stage2 = read_json(stage2_path)
        add(
            "stage2_on_target_screen",
            "Stage2 on-target design/predict screen",
            "candidate",
            stage2.get("accepted"),
            stage2.get("evaluated"),
            "pLDDT/pTM/ipTM/RMSD and S_on>=0.40",
        )

    output = run_dir / "pipeline_output" / "screening_summary.csv"
    _write_csv(
        output,
        rows,
        [
            "step",
            "stage",
            "screen",
            "unit",
            "passed",
            "total",
            "pass_rate_percent",
            "criterion",
        ],
    )
    return {"path": str(output), "rows": rows}
