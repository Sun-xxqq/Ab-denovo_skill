"""Publish completed Stage 2 results in a stable, numbered layout.

Runtime directories are retained under ``stage2/.work/legacy_layout``. The
user-facing files are hard-linked when possible (copied otherwise); symbolic
links are never created.
"""

import csv
import json
import os
import re
import shutil
from pathlib import Path
from .fastrelax_naming import fastrelax_path


STATIC_PUBLIC_DIRS = (
    "00_stage2_plan",
    "01_partialflow",
    "02_abmpnn",
    "03_flowpacker",
    "06_screened_candidates",
    "07_cldn18_1_offtarget",
    "08_fastrelax",
    "09_IgGM",
)


def _truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def _remove(path):
    path = Path(path)
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _require(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _link_or_copy(source, destination):
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(str(source), str(destination))
    except OSError:
        shutil.copy2(source, destination)


def _copy_tree_files(source, destination):
    source = Path(source)
    destination = Path(destination)
    if not source.is_dir():
        return 0
    copied = 0
    for item in sorted(source.rglob("*")):
        if item.is_file():
            _link_or_copy(item, destination / item.relative_to(source))
            copied += 1
    return copied


def _materialize_archived_symlinks(stage2, archive):
    """Replace moved runtime symlinks with self-contained files in the archive."""
    converted = 0
    for link in sorted(archive.rglob("*")):
        if not link.is_symlink():
            continue
        raw_target = Path(os.readlink(str(link)))
        original_link = stage2 / link.relative_to(archive)
        if raw_target.is_absolute():
            original_target = raw_target
        else:
            original_target = (original_link.parent / raw_target).resolve()
        try:
            relative_target = original_target.relative_to(stage2)
        except ValueError:
            continue
        archived_target = archive / relative_target
        if not archived_target.is_file():
            raise FileNotFoundError(
                "archived symlink target was not preserved: %s -> %s"
                % (link, archived_target)
            )
        link.unlink()
        _link_or_copy(archived_target, link)
        converted += 1
    return converted


def _count(root, pattern):
    root = Path(root)
    return (
        sum(1 for path in root.glob(pattern) if path.is_file())
        if root.exists()
        else 0
    )


def _partialflow_id(parent_name, sample_stem):
    parent = re.fullmatch(r"backbone\.0_(\d+)", parent_name)
    sample = re.fullmatch(r"sample(\d+)", sample_stem)
    if not parent or not sample:
        raise ValueError(
            "unexpected PartialFlow name: %s/%s" % (parent_name, sample_stem)
        )
    return "backbone.1_%s_%s" % (parent.group(1), sample.group(1))


def _predictor_source(stage2, predictor):
    candidates = (
        stage2 / ("%s_output" % predictor),
        stage2 / predictor,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Stage 2 %s output was not found; checked %s"
        % (predictor, ", ".join(str(path) for path in candidates))
    )


def _screen_source(stage2):
    candidates = (
        stage2 / "ipsae" / "on_target_screen.csv",
        stage2 / "design_pred_screen.csv",
        stage2 / "screening_tables" / "round1_candidates.csv",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Stage 2 design/prediction screen was not found; checked %s"
        % ", ".join(str(path) for path in candidates)
    )


def _rewrite_screen_paths(stage2, source, destination, predictor):
    """Rewrite the public screening table to point only at numbered outputs."""
    with Path(source).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not fields:
        raise RuntimeError("screening table has no header: %s" % source)

    predictor_root = stage2 / ("04_%s" % predictor)
    for row in rows:
        old_design = Path(
            row.get("original_source_pdb") or row.get("design_path") or ""
        )
        match = re.fullmatch(
            r"flowpacker(\.1_\d+_\d+_\d+)\.pdb", old_design.name
        )
        group = old_design.parent.name
        if not match or not re.fullmatch(r"backbone\.1_\d+_\d+", group):
            raise ValueError(
                "cannot map screening row to numbered output: %s" % old_design
            )
        suffix = match.group(1)
        public_backbone = stage2 / "01_partialflow" / group / (group + ".pdb")
        public_sequence = (
            stage2 / "02_abmpnn" / group / ("abmpnn%s.fa" % suffix)
        )
        public_design = stage2 / "03_flowpacker" / group / old_design.name
        public_predictor = predictor_root / group
        mapped = {
            "original_source_pdb": public_design,
            "source_pdb": public_predictor / ("design%s.pdb" % suffix),
            "predicted_cif": public_predictor / ("pred%s.cif" % suffix),
            "prediction_path": public_predictor / ("pred%s.cif" % suffix),
            "design_path": public_design,
            "sequence_path": public_sequence,
            "backbone_path": public_backbone,
            "%s_summary" % predictor: public_predictor / "metrics.json",
        }
        for field, path in mapped.items():
            if field in row:
                row[field] = str(path)

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)
    return rows


def _copy_optional_stage_outputs(run_dir, stage2, staging):
    rosetta_target = fastrelax_path(staging / "08_fastrelax")
    for name in ("fastrelax_output", "fastrelax_raw_output", "rosettafix_output", "rosetta_fix_output"):
        source = stage2 / name
        if source.is_dir():
            _copy_tree_files(source, rosetta_target)
            break

    offtarget_target = staging / "07_cldn18_1_offtarget"
    for name in (
        "offtarget_screen",
        "filtered_delta",
        "protenix_offtarget_output",
        "opendde_offtarget_output",
    ):
        source = stage2 / name
        if source.is_dir():
            _copy_tree_files(source, offtarget_target / name)
    off_target_csv = stage2 / "cldn18_1_offtarget_screen.csv"
    if off_target_csv.is_file():
        _link_or_copy(off_target_csv, offtarget_target / off_target_csv.name)

def _build_public_layout(run_dir, stage2, staging, predictor):
    source_partial = _require(stage2 / "partial_output")
    source_abmpnn = _require(stage2 / "abmpnn_output")
    source_flowpacker = _require(stage2 / "flowpacker_output")
    source_predictor = _predictor_source(stage2, predictor)
    source_screen = _screen_source(stage2)

    for name in (*STATIC_PUBLIC_DIRS, "04_%s" % predictor):
        (staging / name).mkdir(parents=True, exist_ok=False)

    plan = staging / "00_stage2_plan"
    for name in (
        "maturation_plan.json",
        "fixed_positions.csv",
        "partialflow_mask_policy.json",
        "full_atom_fix_manifest.json",
    ):
        source = stage2 / name
        if source.is_file():
            _link_or_copy(source, plan / name)
    thresholds = stage2 / "ipsae" / "on_target_thresholds.json"
    if thresholds.is_file():
        _link_or_copy(thresholds, plan / thresholds.name)

    partial_count = 0
    for source in sorted(source_partial.glob("backbone.0_*/sample*.pdb")):
        backbone_id = _partialflow_id(source.parent.name, source.stem)
        _link_or_copy(
            source,
            staging / "01_partialflow" / backbone_id / (backbone_id + ".pdb"),
        )
        partial_count += 1

    for source in sorted(source_abmpnn.glob("backbone.1_*/*.fa")):
        _link_or_copy(
            source,
            staging / "02_abmpnn" / source.parent.name / source.name,
        )

    for source in sorted(source_flowpacker.glob("backbone.1_*/*.pdb")):
        _link_or_copy(
            source,
            staging / "03_flowpacker" / source.parent.name / source.name,
        )

    predictor_target = staging / ("04_%s" % predictor)
    for group in sorted(source_predictor.glob("backbone.1_*")):
        if not group.is_dir():
            continue
        target_group = predictor_target / group.name
        for source in sorted(group.glob("*.cif")):
            _link_or_copy(source, target_group / source.name)
        metrics = _require(group / "metrics.json")
        _link_or_copy(metrics, target_group / metrics.name)
        for design in sorted(
            (source_flowpacker / group.name).glob("flowpacker.*.pdb")
        ):
            design_name = "design" + design.name[len("flowpacker") :]
            _link_or_copy(design, target_group / design_name)

    rows = _rewrite_screen_paths(
        stage2,
        source_screen,
        staging / "05_on_target_screen.csv",
        predictor,
    )

    source_filtered = stage2 / "filtered_on_target"
    if source_filtered.is_dir():
        _copy_tree_files(source_filtered, staging / "06_screened_candidates")
    _copy_optional_stage_outputs(run_dir, stage2, staging)

    counts = {
        "partialflow_pdb": _count(staging / "01_partialflow", "*/*.pdb"),
        "abmpnn_fasta": _count(staging / "02_abmpnn", "*/*.fa"),
        "flowpacker_pdb": _count(staging / "03_flowpacker", "*/*.pdb"),
        "predictor_design_pdb": _count(predictor_target, "*/*.pdb"),
        "predictor_prediction_cif": _count(predictor_target, "*/*.cif"),
        "predictor_metrics_json": _count(predictor_target, "*/metrics.json"),
        "screened_candidate_files": _count(
            staging / "06_screened_candidates", "**/*"
        ),
        "screen_rows": len(rows),
        "screen_passed": sum(_truthy(row.get("filter_pass")) for row in rows),
    }
    expected = {
        "partialflow_pdb": partial_count,
        "abmpnn_fasta": _count(source_abmpnn, "*/*.fa"),
        "flowpacker_pdb": _count(source_flowpacker, "*/*.pdb"),
        "predictor_design_pdb": _count(source_flowpacker, "*/*.pdb"),
        "predictor_prediction_cif": _count(source_predictor, "*/*.cif"),
        "predictor_metrics_json": _count(source_predictor, "*/metrics.json"),
        "screened_candidate_files": _count(source_filtered, "**/*"),
    }
    actual = {key: counts[key] for key in expected}
    if actual != expected:
        raise RuntimeError(
            "Stage 2 numbered-layout validation failed: published=%s expected=%s"
            % (actual, expected)
        )
    return counts


def finalize_stage2_layout(run_dir, expected_sequences=4):
    """Archive runtime files and publish the permanent numbered Stage 2 layout."""
    del expected_sequences  # Counts are validated directly from runtime outputs.
    run_dir = Path(run_dir).resolve()
    stage2 = _require(run_dir / "pipeline_output" / "stage2")
    manifest_path = stage2 / "00_stage2_plan" / "organization_manifest.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
    comparison = (
        spec.get("validation", {})
        .get("candidate_screening", {})
        .get("stage2_structural_comparison", {})
    )
    predictor = comparison.get(
        "predictor",
        spec.get("workflow", {}).get("complex_prediction", "protenix"),
    )
    public_entries = {
        *STATIC_PUBLIC_DIRS,
        "04_%s" % predictor,
        "05_on_target_screen.csv",
    }
    conflicts = sorted(
        name for name in public_entries if (stage2 / name).exists()
    )
    if conflicts:
        raise RuntimeError(
            "numbered Stage 2 layout is incomplete or untracked: %s" % conflicts
        )

    staging = stage2.parent / (".%s.numbered-layout.tmp" % stage2.name)
    _remove(staging)
    staging.mkdir()
    moved = []
    try:
        counts = _build_public_layout(run_dir, stage2, staging, predictor)
        archive = stage2 / ".work" / "legacy_layout"
        if archive.exists():
            raise RuntimeError("legacy Stage 2 archive already exists: %s" % archive)
        archive.mkdir(parents=True)

        for source in sorted(stage2.iterdir()):
            if source.name == ".work" or source.name in public_entries:
                continue
            destination = archive / source.name
            source.replace(destination)
            moved.append((source, destination))

        archived_symlinks_materialized = _materialize_archived_symlinks(
            stage2, archive
        )

        try:
            for entry in sorted(staging.iterdir()):
                entry.replace(stage2 / entry.name)
        except Exception:
            for source, destination in reversed(moved):
                if destination.exists() and not source.exists():
                    destination.replace(source)
            raise

        manifest = {
            "layout": "numbered_stage2_v1",
            "predictor": predictor,
            "symbolic_links_created": False,
            "archived_symlinks_materialized": archived_symlinks_materialized,
            "legacy_archive": str(archive),
            "counts": counts,
            "published_entries": sorted(
                path.name for path in stage2.iterdir() if path.name != ".work"
            ),
            "stage2": str(stage2),
            "partialflow_backbones": counts["partialflow_pdb"],
            "sequences": counts["abmpnn_fasta"],
            "evaluated": counts["screen_rows"],
            "passed": counts["screen_passed"],
        }
        manifest_path = (
            stage2 / "00_stage2_plan" / "organization_manifest.json"
        )
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest
    finally:
        _remove(staging)
