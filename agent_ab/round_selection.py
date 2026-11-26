from .fastrelax_naming import fastrelax_path
"""Create reusable, candidate-ID keyed checkpoints between design rounds."""

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

from .util import now_iso, read_json, write_json


def _run_dir(config, run_id):
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("invalid run ID")
    return Path(config["storage"]["runs_root"]) / run_id


def _truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass"}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _round_root(run_dir, round_number):
    return Path(run_dir) / "pipeline_output" / ("stage%s" % int(round_number))


def _default_table(run_dir, round_number):
    root = _round_root(run_dir, round_number)
    candidates = []
    if int(round_number) == 1:
        candidates.append(root / "cldn18_1_offtarget_screen.csv")
    candidates.append(root / "design_pred_screen.csv")
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            "no result table found for round %s; checked %s"
            % (round_number, ", ".join(map(str, candidates)))
        )
    return path


def _read_table(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    records = {}
    for row in rows:
        candidate_id = (row.get("candidate_id") or "").strip()
        if not candidate_id:
            raise ValueError("round result row has no candidate_id: %s" % path)
        if candidate_id in records:
            raise ValueError("duplicate candidate_id in round table: %s" % candidate_id)
        records[candidate_id] = row
    return records


def _default_selected(row):
    for field in ("selected", "specificity_pass", "filter_pass"):
        if field in row and str(row.get(field, "")).strip():
            return _truthy(row[field])
    return True


def _ids_from_file(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        value = read_json(path)
        if isinstance(value, dict):
            value = value.get("candidate_ids", value.get("selected", []))
        result = []
        for item in value:
            result.append(item.get("candidate_id") if isinstance(item, dict) else item)
        return [str(item).strip() for item in result if str(item).strip()]
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return [
            row["candidate_id"].strip()
            for row in rows
            if row.get("candidate_id")
            and (not str(row.get("selected", "")).strip() or _truthy(row["selected"]))
        ]
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _source_root(run_dir, round_number, source_kind):
    root = _round_root(run_dir, round_number)
    round_index = int(round_number) - 1
    choices = {
        "fastrelax": [fastrelax_path(root / "fastrelax_output"), fastrelax_path(root / "fastrelax_raw_output")],
        "canonical": [root / ("canonical_round%s" % round_index)],
    }
    source_kind = "fastrelax" if source_kind == "rosettafix" else source_kind
    kinds = [source_kind] if source_kind != "auto" else ["fastrelax", "canonical"]
    for kind in kinds:
        path = next((item for item in choices[kind] if item.is_dir()), None)
        if path is not None:
            return kind, path
    raise FileNotFoundError(
        "no %s structures found for round %s" % (source_kind, round_number)
    )


def _structure_index(root):
    result = {}
    for path in sorted(Path(root).rglob("*.pdb")):
        keys = {path.stem}
        if path.stem.endswith("_0001"):
            keys.add(path.stem[:-5])
        for key in keys:
            result.setdefault(key, []).append(path)
    return result


def _resolve_structure(index, row, candidate_id):
    keys = [row.get("design_id"), candidate_id]
    for key in keys:
        matches = index.get(str(key), []) if key else []
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            backbone_id = row.get("backbone_id")
            narrowed = [item for item in matches if backbone_id in item.parts]
            if len(narrowed) == 1:
                return narrowed[0]
            raise ValueError("ambiguous structure for candidate_id %s" % candidate_id)
    raise FileNotFoundError("no source structure for candidate_id %s" % candidate_id)


def _fixed_values(run_dir, round_number, source_root, row):
    backbone_id = row.get("backbone_id")
    candidates = []
    if backbone_id:
        candidates.extend(
            [
                Path(source_root) / backbone_id / "fixed_targets.csv",
                fastrelax_path(_round_root(run_dir, round_number)
                / "fastrelax_output")
                / backbone_id
                / "fixed_targets.csv",
            ]
        )
    for path in candidates:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            values = list(csv.DictReader(handle))
        if values:
            return (
                values[0].get("fixed_positions", "NONE"),
                values[0].get("merge_seq", ""),
                str(path.resolve()),
            )
    return "NONE", "", None


def _safe_candidate_name(candidate_id):
    if Path(candidate_id).name != candidate_id or candidate_id in (".", ".."):
        raise ValueError("unsafe candidate_id: %s" % candidate_id)
    return candidate_id + ".pdb"


def record_round_input(
    config,
    run_id,
    from_round,
    to_round,
    candidate_ids=None,
    candidate_file=None,
    source_kind="auto",
    table=None,
    replace=False,
):
    from_round, to_round = int(from_round), int(to_round)
    if from_round < 1 or to_round < 2 or from_round >= to_round:
        raise ValueError("rounds must satisfy 1 <= from_round < to_round")
    run_dir = _run_dir(config, run_id)
    if not run_dir.is_dir():
        raise FileNotFoundError("run not found: %s" % run_id)
    table = Path(table) if table else _default_table(run_dir, from_round)
    rows = _read_table(table)
    explicit = list(candidate_ids or [])
    if candidate_file:
        explicit.extend(_ids_from_file(candidate_file))
    if explicit:
        selected_ids = []
        for candidate_id in explicit:
            candidate_id = str(candidate_id).strip()
            if candidate_id and candidate_id not in selected_ids:
                selected_ids.append(candidate_id)
    else:
        selected_ids = [
            candidate_id
            for candidate_id, row in rows.items()
            if _default_selected(row)
        ]
    if not selected_ids:
        raise ValueError("round selection contains no candidates")
    missing = sorted(set(selected_ids) - set(rows))
    if missing:
        raise ValueError(
            "candidate_id values missing from source table: %s" % ", ".join(missing)
        )

    resolved_kind, source_root = _source_root(run_dir, from_round, source_kind)
    index = _structure_index(source_root)
    destination = (
        run_dir
        / "pipeline_output"
        / "round_inputs"
        / ("round_%s" % to_round)
        / ("from_round_%s" % from_round)
    )
    destination = destination.resolve()
    allowed = (run_dir / "pipeline_output" / "round_inputs").resolve()
    destination.relative_to(allowed)
    manifest_path = destination / "selection.json"
    if destination.exists():
        if not replace:
            existing = read_json(manifest_path) if manifest_path.exists() else {}
            if existing.get("requested_candidate_ids") == selected_ids:
                existing["status"] = "reused_existing_checkpoint"
                return existing
            raise FileExistsError(
                "round input already exists with a different selection; use --replace"
            )
        shutil.rmtree(destination)

    structures = destination / "structures"
    structures.mkdir(parents=True)
    records = []
    for candidate_id in selected_ids:
        row = rows[candidate_id]
        source = _resolve_structure(index, row, candidate_id).resolve()
        output_name = _safe_candidate_name(candidate_id)
        output = structures / output_name
        output.symlink_to(os.path.relpath(source, output.parent))
        fixed_positions, merge_seq, fixed_source = _fixed_values(
            run_dir, from_round, source_root, row
        )
        records.append(
            {
                "candidate_id": candidate_id,
                "backbone_id": row.get("backbone_id"),
                "design_id": row.get("design_id"),
                "filename": output_name,
                "fixed_positions": fixed_positions,
                "merge_seq": merge_seq,
                "source_structure": str(source),
                "source_sha256": _sha256(source),
                "fixed_positions_source": fixed_source,
                "source_filter_pass": row.get("filter_pass"),
                "source_specificity_pass": row.get("specificity_pass"),
                "source_status": row.get("status"),
            }
        )
    fixed_csv = destination / "fixed_positions.csv"
    fields = [
        "candidate_id",
        "backbone_id",
        "design_id",
        "filename",
        "fixed_positions",
        "merge_seq",
        "source_structure",
        "source_sha256",
        "fixed_positions_source",
        "source_filter_pass",
        "source_specificity_pass",
        "source_status",
    ]
    with fixed_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    signature_payload = [
        {"candidate_id": item["candidate_id"], "sha256": item["source_sha256"]}
        for item in records
    ]
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "status": "ready",
        "run_id": run_id,
        "from_round": from_round,
        "to_round": to_round,
        "created_at": now_iso(),
        "source_kind": resolved_kind,
        "source_root": str(source_root.resolve()),
        "source_table": str(table.resolve()),
        "pairing_key": "candidate_id",
        "requested_candidate_ids": selected_ids,
        "candidate_count": len(records),
        "selection_signature": signature,
        "checkpoint_dir": str(destination),
        "structures_dir": str(structures),
        "fixed_positions_csv": str(fixed_csv),
        "recompute_policy": "reuse_source_structures_and_fixed_positions",
        "backbone_regenerated": False,
        "fastrelax_recomputed": False,
        "rosettafix_recomputed": False,  # Legacy API field.
        "candidates": records,
        "continue_command": (
            "scripts/agent-ab resume %s --stage %s --gpu auto" % (run_id, to_round)
            if to_round in (1, 2)
            else None
        ),
    }
    write_json(manifest_path, manifest)
    active_path = destination.parent / "active.json"
    write_json(
        active_path,
        {
            "from_round": from_round,
            "to_round": to_round,
            "selection": str(manifest_path),
            "selection_signature": signature,
        },
    )
    return manifest


def active_round_input(config, run_id, to_round):
    run_dir = _run_dir(config, run_id)
    active = (
        run_dir
        / "pipeline_output"
        / "round_inputs"
        / ("round_%s" % int(to_round))
        / "active.json"
    )
    if not active.exists():
        return None
    pointer = read_json(active)
    selection = Path(pointer["selection"])
    if not selection.is_file():
        raise FileNotFoundError("active round selection is missing: %s" % selection)
    manifest = read_json(selection)
    for key in ("checkpoint_dir", "structures_dir", "fixed_positions_csv"):
        if not Path(manifest[key]).exists():
            raise FileNotFoundError("round checkpoint artifact is missing: %s" % manifest[key])
    return manifest


def list_round_inputs(config, run_id):
    run_dir = _run_dir(config, run_id)
    root = run_dir / "pipeline_output" / "round_inputs"
    if not root.exists():
        return []
    return [read_json(path) for path in sorted(root.glob("round_*/from_round_*/selection.json"))]
