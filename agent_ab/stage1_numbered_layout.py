"""Publish completed Stage 1 results in a stable numbered layout."""

import json
import os
import shutil
from pathlib import Path


PLAN_FILES = (
    "fixed_positions.csv",
    "rosetta_group_manifest.json",
    "reuse_manifest.json",
)


def _link_or_copy(source, destination):
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(str(source), str(destination))
    except OSError:
        shutil.copy2(str(source), str(destination))


def _count_files(root):
    root = Path(root)
    return sum(1 for path in root.rglob("*") if path.is_file()) if root.exists() else 0


def existing_numbered_stage1(stage1):
    manifest = Path(stage1) / "00_stage1_plan" / "organization_manifest.json"
    if not manifest.is_file():
        return None
    return json.loads(manifest.read_text(encoding="utf-8"))


def _materialize_symlinks(root):
    converted = 0
    root = Path(root)
    if not root.is_dir():
        return converted
    for link in sorted(root.rglob("*")):
        if not link.is_symlink():
            continue
        target = link.resolve(strict=True)
        link.unlink()
        _link_or_copy(target, link)
        converted += 1
    return converted


def _replace_json_strings(value, replacements):
    if isinstance(value, dict):
        return {
            key: _replace_json_strings(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_json_strings(item, replacements) for item in value]
    if isinstance(value, str):
        for old, new in replacements:
            value = value.replace(old, new)
        return value
    return value


def _rewrite_public_paths(stage1, replacements):
    rewritten = 0
    for path in sorted(Path(stage1).rglob("*")):
        if not path.is_file() or ".work" in path.parts:
            continue
        if path.suffix.lower() == ".json":
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            updated = _replace_json_strings(value, replacements)
            if updated != value:
                path.write_text(
                    json.dumps(updated, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                rewritten += 1
        elif path.suffix.lower() in {".csv", ".log", ".txt"}:
            try:
                value = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            updated = value
            for old, new in replacements:
                updated = updated.replace(old, new)
            if updated != value:
                path.write_text(updated, encoding="utf-8")
                rewritten += 1
    return rewritten


def publish_numbered_stage1(stage1, predictor, summary):
    """Rename the compact Stage 1 view and rewrite its embedded paths."""
    stage1 = Path(stage1).resolve()
    previous = existing_numbered_stage1(stage1)
    if previous is not None:
        return previous

    predictor_source = "%s_output" % predictor if predictor == "protenix" else predictor
    predictor_target = "05_%s" % predictor
    mapping = [
        ("backbone_output", "01_ppiflow_backbones"),
        ("backbone_screen", "02_backbone_screen"),
        ("abmpnn_output", "03_abmpnn"),
        ("flowpacker_output", "04_flowpacker"),
        (predictor_source, predictor_target),
        ("design_pred_screen.csv", "06_design_pred_screen.csv"),
        ("canonical_round0", "07_screened_candidates"),
        ("fastrelax_output", "08_interface_energy"),
        ("fastrelax", "08_interface_energy"),
        ("stage1.log", "09_stage1_summary.log"),
    ]
    active_mapping = []
    used_targets = set()
    for source_name, target_name in mapping:
        source = stage1 / source_name
        if not source.exists() and not source.is_symlink():
            continue
        if target_name in used_targets:
            raise RuntimeError(
                "multiple Stage 1 sources map to the same numbered entry: %s"
                % target_name
            )
        target = stage1 / target_name
        if target.exists() or target.is_symlink():
            raise RuntimeError("numbered Stage 1 target already exists: %s" % target)
        active_mapping.append((source_name, target_name))
        used_targets.add(target_name)

    required = {
        "01_ppiflow_backbones",
        "02_backbone_screen",
        "03_abmpnn",
        "04_flowpacker",
        predictor_target,
        "06_design_pred_screen.csv",
        "07_screened_candidates",
        "08_interface_energy",
        "09_stage1_summary.log",
    }
    missing = sorted(required - used_targets)
    if missing:
        raise RuntimeError(
            "cannot publish numbered Stage 1 layout; missing entries: %s"
            % ", ".join(missing)
        )

    converted_symlinks = _materialize_symlinks(stage1 / "canonical_round0")
    source_counts = {
        source_name: _count_files(stage1 / source_name)
        for source_name, _ in active_mapping
    }

    plan = stage1 / "00_stage1_plan"
    plan.mkdir()
    moved_plan = []
    moved_entries = []
    try:
        for name in PLAN_FILES:
            source = stage1 / name
            if source.is_file():
                destination = plan / name
                source.replace(destination)
                moved_plan.append((source, destination))

        for source_name, target_name in active_mapping:
            source = stage1 / source_name
            target = stage1 / target_name
            source.replace(target)
            moved_entries.append((source, target))

        replacements = []
        for source_name, target_name in active_mapping:
            old = str(stage1 / source_name)
            new = str(stage1 / target_name)
            replacements.append((old, new))
            replacements.append((source_name, target_name))
        for name in PLAN_FILES:
            replacements.append(
                (str(stage1 / name), str(plan / name))
            )
        rewritten_files = _rewrite_public_paths(stage1, replacements)

        published_counts = {
            target_name: _count_files(stage1 / target_name)
            for _, target_name in active_mapping
        }
        expected_counts = {
            target_name: source_counts[source_name]
            for source_name, target_name in active_mapping
        }
        if published_counts != expected_counts:
            raise RuntimeError(
                "Stage 1 numbered-layout validation failed: published=%s expected=%s"
                % (published_counts, expected_counts)
            )

        legacy_archive = stage1 / ".work" / "legacy_layout"
        result = dict(summary)
        result.update(
            {
                "layout": "numbered_stage1_v1",
                "predictor": predictor,
                "legacy_archive": str(legacy_archive),
                "symbolic_links_created": False,
                "materialized_screened_candidate_links": converted_symlinks,
                "rewritten_metadata_files": rewritten_files,
                "counts": published_counts,
                "stage1": str(stage1),
            }
        )
        result["published_entries"] = sorted(
            path.name for path in stage1.iterdir() if path.name != ".work"
        )
        manifest = plan / "organization_manifest.json"
        manifest.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return result
    except Exception:
        for source, target in reversed(moved_entries):
            if target.exists() and not source.exists():
                target.replace(source)
        for source, target in reversed(moved_plan):
            if target.exists() and not source.exists():
                target.replace(source)
        if plan.exists() and not any(plan.iterdir()):
            plan.rmdir()
        raise
