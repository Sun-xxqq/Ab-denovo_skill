#!/usr/bin/env python3
"""Build a compact, stage-oriented view of one agent-Ab run.

Large structures are exposed with relative symbolic links so the organized view
does not duplicate native PPIFlow or adapter outputs.  Small status and inventory
files are generated directly in the organized output directory.
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path


BASE_STAGES = (
    "01_backbone",
    "02_abmpnn",
    "03_flowpacker",
    "05_design_pred_comparison",
    "06_filtered_candidates",
)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _link(source, destination):
    source = Path(source).resolve()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source:
            return destination
        destination.unlink()
    elif destination.exists():
        raise FileExistsError("refusing to replace non-link output: %s" % destination)
    destination.symlink_to(os.path.relpath(source, destination.parent))
    return destination


def _status(root, stage, state, count, detail, **extra):
    value = {"stage": stage, "state": state, "count": int(count), "detail": detail}
    value.update(extra)
    _write_json(Path(root) / stage / "status.json", value)
    return value


def _remove_legacy_link_directory(path):
    """Remove the former screened_pass view, but never delete real artifacts."""
    path = Path(path)
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink():
        path.unlink()
        return
    for child in path.iterdir():
        if not child.is_symlink():
            raise RuntimeError("refusing to delete non-link legacy output: %s" % child)
        child.unlink()
    path.rmdir()


def _backbone_index(name):
    match = re.search(r"_(\d+)(?:\.pdb|\.fa)$", name, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _design_indices(name):
    match = re.search(r"_(\d+)_(\d+)\.pdb$", name, re.IGNORECASE)
    return tuple(int(value) for value in match.groups()) if match else None


def _add_inventory(records, stage, group, source, organized):
    records.append(
        {
            "stage": stage,
            "group": group,
            "item": Path(organized).name,
            "source": str(Path(source).resolve()),
            "organized_path": str(Path(organized).absolute()),
        }
    )


def _organize_backbones(run_dir, output_root, records, expected):
    stage = "01_backbone"
    _remove_legacy_link_directory(output_root / stage / "screened_pass")
    native = run_dir / "pipeline_output" / "stage1" / "ppiflow_output"
    pdbs = sorted(native.glob("*.pdb"))
    for pdb in pdbs:
        destination = output_root / stage / "all_pdb" / pdb.name
        _link(pdb, destination)
        _add_inventory(records, stage, "all", pdb, destination)

    screen_candidates = (
        run_dir / "pipeline_output" / "stage1" / "backbone_screen",
    )
    screen_root = next((path for path in screen_candidates if path.exists()), None)
    accepted_count = 0
    if screen_root:
        for source in sorted(screen_root.glob("*distances_sorted.*")):
            destination = output_root / stage / "screening" / source.name
            _link(source, destination)
            _add_inventory(records, stage, "screening", source, destination)
        summary = screen_root / "summary.json"
        if summary.exists():
            destination = output_root / stage / "screening" / summary.name
            _link(summary, destination)
            _add_inventory(records, stage, "screening", summary, destination)
            try:
                accepted_count = int(json.loads(summary.read_text(encoding="utf-8")).get("accepted_backbones", 0))
            except (OSError, ValueError):
                accepted_count = 0
    complete = expected is not None and len(pdbs) >= expected
    state = "complete" if complete else ("partial_interrupted" if pdbs else "not_started")
    detail = "%d/%s backbone PDBs present; %d currently pass the distance screen" % (
        len(pdbs),
        expected if expected is not None else "unknown",
        accepted_count,
    )
    return _status(
        output_root,
        stage,
        state,
        len(pdbs),
        detail,
        expected_count=expected,
        screened_pass_count=accepted_count,
    )


def _organize_abmpnn(run_dir, output_root, records):
    stage = "02_abmpnn"
    root = run_dir / "pipeline_output" / "stage1" / "abmpnn_output"
    legacy = root / "seqs"
    if legacy.is_dir():
        files = sorted(legacy.glob("*.fa"))
    else:
        files = sorted(root.glob("*/*.fa")) if root.exists() else []
    backbone_groups = set()
    for source in files:
        if source.parent != legacy:
            group = source.parent.name
        else:
            index = _backbone_index(source.name)
            group = "backbone_%03d" % index if index is not None else "unmatched"
        backbone_groups.add(group)
        destination = output_root / stage / "by_backbone" / group / source.name
        _link(source, destination)
        _add_inventory(records, stage, group, source, destination)
    sequence_csv = run_dir / "pipeline_output" / "stage1" / "mpnn_pdbs" / "mpnn_seqs.csv"
    if sequence_csv.exists():
        destination = output_root / stage / "sequence_manifest.csv"
        _link(sequence_csv, destination)
        _add_inventory(records, stage, "manifest", sequence_csv, destination)
    state = "complete" if files and sequence_csv.exists() else "not_started"
    detail = (
        "%d designed sequences across %d backbones" % (len(files), len(backbone_groups))
        if files
        else "AbMPNN did not start"
    )
    return _status(
        output_root,
        stage,
        state,
        len(files),
        detail,
        backbone_count=len(backbone_groups),
    )


def _organize_flowpacker(run_dir, output_root, records):
    stage = "03_flowpacker"
    root = run_dir / "pipeline_output" / "stage1" / "flowpacker_output"
    files = sorted(root.glob("backbone.*/*.pdb")) if root.exists() else []
    if not files and (root / "run_1").exists():
        files = sorted((root / "run_1").glob("*.pdb"))
    unmatched = 0
    for source in files:
        indices = _design_indices(source.name)
        if source.parent.name.startswith("backbone."):
            group = source.parent.name
        elif indices:
            group = "backbone_%03d" % indices[0]
        else:
            group = "unmatched"
            unmatched += 1
        destination = output_root / stage / "by_backbone" / group / source.name
        _link(source, destination)
        _add_inventory(records, stage, group, source, destination)
    state = "complete" if files else "not_started"
    detail = "%d side-chain-packed design PDBs" % len(files) if files else "FlowPacker did not start"
    return _status(output_root, stage, state, len(files), detail, unmatched=unmatched)


def _prediction_output_name(predictor):
    return "%s_output" % predictor if predictor == "protenix" else predictor


def _organize_prediction(run_dir, output_root, records, predictor):
    stage = "04_%s" % predictor
    root = (
        run_dir
        / "pipeline_output"
        / "stage1"
        / _prediction_output_name(predictor)
        / "predictions"
    )
    legacy = run_dir / "adapter_output" / predictor
    if not root.exists() and legacy.exists():
        root = legacy
    predictions = []
    if root.exists():
        predictions = sorted(root.glob("*/pred.pdb"))
        predictions += sorted(root.glob("*/pred.cif"))
    seen = set()
    converted = 0
    for source in predictions:
        candidate = source.parent.name
        if candidate in seen:
            continue
        seen.add(candidate)
        match = re.fullmatch(r"pred_(\d+)_(\d+)", candidate)
        group = "backbone_%03d" % int(match.group(1)) if match else "unmatched"
        destination = output_root / stage / "by_backbone" / group / (candidate + ".pdb")
        if source.suffix.lower() == ".cif":
            from Bio.PDB import MMCIFParser, PDBIO

            destination.parent.mkdir(parents=True, exist_ok=True)
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure(candidate, str(source))
            writer = PDBIO()
            writer.set_structure(structure)
            writer.save(str(destination))
            converted += 1
        else:
            _link(source, destination)
        _add_inventory(records, stage, group, source, destination)
    state = "complete" if seen else "not_started"
    detail = (
        "%d predicted complex structures" % len(seen)
        if seen
        else "%s did not start" % predictor
    )
    return _status(output_root, stage, state, len(seen), detail, cif_converted_to_pdb=converted)


def _comparison_rows(run_dir):
    rows = []
    root = run_dir / "adapter_output" / "ipsae"
    for path in sorted(root.glob("backbone_*/filter_results.json")) if root.exists() else []:
        try:
            values = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(values, list):
            rows.extend(value for value in values if isinstance(value, dict))
    return rows


def _organize_comparison(run_dir, output_root, records):
    stage = "05_design_pred_comparison"
    rows = _comparison_rows(run_dir)
    destination = output_root / stage / "design_pred_metrics.csv"
    if rows:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields and not isinstance(row[key], (dict, list)):
                    fields.append(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        records.append(
            {
                "stage": stage,
                "group": "all",
                "item": destination.name,
                "source": "generated from adapter_output/ipsae/backbone_*/filter_results.json",
                "organized_path": str(destination.absolute()),
            }
        )
    state = "complete" if rows else "not_started"
    detail = "%d design-pred comparison rows" % len(rows) if rows else "design-pred comparison did not start"
    return _status(output_root, stage, state, len(rows), detail)


def _organize_selected(run_dir, output_root, records):
    stage = "06_filtered_candidates"
    root = run_dir / "pipeline_output" / "stage1" / "stage1_candidates"
    files = sorted(root.glob("backbone_*/*.pdb")) if root.exists() else []
    for source in files:
        group = source.parent.name
        destination = output_root / stage / "by_backbone" / group / source.name
        _link(source, destination)
        _add_inventory(records, stage, group, source, destination)
    state = "complete" if files else "not_started"
    detail = "%d retained PDBs grouped by original backbone" % len(files) if files else "candidate filtering did not start"
    return _status(output_root, stage, state, len(files), detail)


def organize(run_dir, output_root=None):
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError("run directory not found: %s" % run_dir)
    expected = None
    predictor = "protenix"
    spec_path = run_dir / "spec.json"
    if spec_path.exists():
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        expected = int(spec["generation"]["samples"])
        predictor = spec.get("workflow", {}).get("complex_prediction", predictor)

    output_root = Path(output_root).resolve() if output_root else run_dir / "organized_output"
    output_root.mkdir(parents=True, exist_ok=True)
    stages = BASE_STAGES[:3] + ("04_%s" % predictor,) + BASE_STAGES[3:]
    for stage in stages:
        (output_root / stage).mkdir(exist_ok=True)

    records = []
    status = {}
    status["backbone"] = _organize_backbones(run_dir, output_root, records, expected)
    status["abmpnn"] = _organize_abmpnn(run_dir, output_root, records)
    status["flowpacker"] = _organize_flowpacker(run_dir, output_root, records)
    status[predictor] = _organize_prediction(
        run_dir, output_root, records, predictor
    )
    status["comparison"] = _organize_comparison(run_dir, output_root, records)
    status["filtered_candidates"] = _organize_selected(run_dir, output_root, records)

    inventory = output_root / "inventory.csv"
    with inventory.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("stage", "group", "item", "source", "organized_path"),
        )
        writer.writeheader()
        writer.writerows(records)

    readme = """# Organized output

This directory is the concise, user-facing view of the run. Large structure and
sequence files are relative symbolic links to native outputs, so they do not use
duplicate disk space. `pipeline_output` and `adapter_output` remain the raw audit
trail.

- `01_backbone`: all generated backbone PDBs and distance-screen tables; passing status stays in the table.
- `02_abmpnn`: generated sequence files, grouped by backbone.
- `03_flowpacker`: side-chain-packed design PDBs, grouped by backbone.
- `04_{predictor}`: predicted complex structures from {predictor}, grouped by backbone.
- `05_design_pred_comparison`: one combined design-vs-prediction metrics table.
- `06_filtered_candidates`: PDBs retained after the first-round filter, grouped by backbone.

Read each stage's `status.json` before interpreting an empty directory.
""".format(predictor=predictor)
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    summary = {
        "run_dir": str(run_dir),
        "organized_output": str(output_root),
        "storage_policy": "relative_symlinks_for_large_artifacts",
        "inventory": str(inventory),
        "stages": status,
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    print(json.dumps(organize(args.run_dir, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
