"""Nanobody pre-sequence CDR-to-hotspot screening for PPIFlow backbones."""

from __future__ import annotations

import csv
import json
import math
import os
import re
from pathlib import Path


CDR_NAMES = ("CDRH1", "CDRH2", "CDRH3")
SCFV_CDR_NAMES = ("CDRH1", "CDRH2", "CDRH3", "CDRL1", "CDRL2", "CDRL3")


def _write_progress(path, value):
    """Atomically publish progress so `agent-ab parts` can safely read it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def _score_with_progress(pdbs, output_dir, profile, scorer):
    progress_path = Path(output_dir) / "progress.json"
    progress = {
        "screening_profile": profile,
        "status": "running",
        "completed": 0,
        "total": len(pdbs),
        "accepted_so_far": 0,
        "current_backbone": None,
    }
    _write_progress(progress_path, progress)
    records = []
    accepted_so_far = 0
    try:
        for index, pdb in enumerate(pdbs, 1):
            record = scorer(pdb)
            records.append(record)
            accepted_so_far += int(record["filter_pass"])
            progress.update(
                {
                    "completed": index,
                    "accepted_so_far": accepted_so_far,
                    "current_backbone": Path(pdb).name,
                }
            )
            _write_progress(progress_path, progress)
    except Exception as exc:
        progress.update({"status": "failed", "detail": str(exc)})
        _write_progress(progress_path, progress)
        raise
    progress.update({"status": "complete", "current_backbone": None})
    _write_progress(progress_path, progress)
    return records


def _pdb_residues(path, chain_id):
    residues = []
    by_key = {}
    with Path(path).open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
                continue
            if line[21].strip() != chain_id:
                continue
            try:
                residue_number = int(line[22:26])
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            insertion_code = line[26].strip()
            key = (residue_number, insertion_code)
            residue = by_key.get(key)
            if residue is None:
                residue = {
                    "key": key,
                    "label": "%s%s" % (residue_number, insertion_code),
                    "resname": line[17:20].strip(),
                    "atoms": [],
                }
                by_key[key] = residue
                residues.append(residue)
            atom_name = line[12:16].strip()
            element = line[76:78].strip().upper() if len(line) >= 78 else ""
            if not element:
                element = re.sub(r"[^A-Za-z]", "", atom_name)[:1].upper()
            residue["atoms"].append(
                {"name": atom_name, "element": element, "coord": (x, y, z)}
            )
    return residues


def _framework_segments(framework_pdb, heavy_chain):
    residues = _pdb_residues(framework_pdb, heavy_chain)
    if not residues:
        raise ValueError("heavy chain %s not found in framework %s" % (heavy_chain, framework_pdb))
    segments = [[residues[0]]]
    for previous, current in zip(residues, residues[1:]):
        previous_number = previous["key"][0]
        current_number = current["key"][0]
        if current_number != previous_number + 1:
            segments.append([])
        segments[-1].append(current)
    if len(segments) != 4:
        raise ValueError(
            "expected exactly three CDR gaps in framework heavy chain %s; found %s"
            % (heavy_chain, len(segments) - 1)
        )
    return segments


def _find_subsequence(sequence, pattern, start):
    limit = len(sequence) - len(pattern) + 1
    for index in range(start, max(start, limit)):
        if sequence[index : index + len(pattern)] == pattern:
            return index
    raise ValueError("framework segment was not found in generated backbone")


def identify_cdr_residues(
    backbone_pdb, framework_pdb, heavy_chain="A", cdr_names=CDR_NAMES
):
    """Identify variable-length CDRs by aligning four framework segments."""
    backbone = _pdb_residues(backbone_pdb, heavy_chain)
    if not backbone:
        raise ValueError("heavy chain %s not found in backbone %s" % (heavy_chain, backbone_pdb))
    segments = _framework_segments(framework_pdb, heavy_chain)
    backbone_names = [residue["resname"] for residue in backbone]
    segment_names = [[residue["resname"] for residue in segment] for segment in segments]
    if backbone_names[: len(segment_names[0])] != segment_names[0]:
        raise ValueError("generated heavy chain does not begin with framework segment 1")

    positions = [0]
    cursor = len(segment_names[0])
    for names in segment_names[1:]:
        position = _find_subsequence(backbone_names, names, cursor)
        positions.append(position)
        cursor = position + len(names)
    if positions[-1] + len(segment_names[-1]) != len(backbone_names):
        raise ValueError("generated heavy chain does not end with framework segment 4")

    cdrs = {}
    for index, name in enumerate(cdr_names):
        start = positions[index] + len(segment_names[index])
        end = positions[index + 1]
        if end <= start:
            raise ValueError("%s is empty in %s" % (name, backbone_pdb))
        cdrs[name] = backbone[start:end]
    return cdrs


def identify_scfv_cdr_residues(
    backbone_pdb, framework_pdb, heavy_chain="A", light_chain="B"
):
    """Identify all six scFv CDR loops from the paired framework gaps."""
    cdrs = identify_cdr_residues(
        backbone_pdb,
        framework_pdb,
        heavy_chain=heavy_chain,
        cdr_names=("CDRH1", "CDRH2", "CDRH3"),
    )
    cdrs.update(
        identify_cdr_residues(
            backbone_pdb,
            framework_pdb,
            heavy_chain=light_chain,
            cdr_names=("CDRL1", "CDRL2", "CDRL3"),
        )
    )
    return cdrs


def _distance(left, right):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _parse_hotspot(value):
    match = re.fullmatch(r"([A-Za-z])(-?\d+)([A-Za-z]?)", str(value).strip())
    if not match:
        raise ValueError("invalid hotspot: %r" % value)
    return match.group(1), int(match.group(2)), match.group(3)


def _hotspot_atoms(backbone_pdb, hotspots):
    parsed = [_parse_hotspot(value) for value in hotspots]
    chains = {chain for chain, _, _ in parsed}
    if len(chains) != 1:
        raise ValueError("all screening hotspots must use the same antigen chain")
    antigen_chain = next(iter(chains))
    antigen = _pdb_residues(backbone_pdb, antigen_chain)
    by_key = {residue["key"]: residue for residue in antigen}
    result = {}
    for chain, number, insertion in parsed:
        residue = by_key.get((number, insertion))
        label = "%s%s%s" % (chain, number, insertion)
        if residue is None:
            raise ValueError("hotspot %s not found in %s" % (label, backbone_pdb))
        atoms = [atom for atom in residue["atoms"] if atom["element"] != "H"]
        if not atoms:
            raise ValueError("hotspot %s has no heavy atoms in %s" % (label, backbone_pdb))
        result[label] = atoms
    return result


def score_backbone(backbone_pdb, framework_pdb, hotspots, heavy_chain="A", cutoff=8.0):
    """Score one backbone using CDR C-alpha to hotspot heavy-atom distances."""
    cdrs = identify_cdr_residues(backbone_pdb, framework_pdb, heavy_chain=heavy_chain)
    hotspot_atoms = _hotspot_atoms(backbone_pdb, hotspots)
    pair_distances = {}
    pair_details = {}
    for cdr_name, residues in cdrs.items():
        c_alpha = []
        for residue in residues:
            for atom in residue["atoms"]:
                if atom["name"] == "CA":
                    c_alpha.append((residue, atom))
                    break
        if not c_alpha:
            raise ValueError("%s has no C-alpha atoms in %s" % (cdr_name, backbone_pdb))
        for hotspot, atoms in hotspot_atoms.items():
            best = None
            for residue, cdr_atom in c_alpha:
                for hotspot_atom in atoms:
                    distance = _distance(cdr_atom["coord"], hotspot_atom["coord"])
                    if best is None or distance < best[0]:
                        best = (distance, residue["label"], hotspot_atom["name"])
            pair_distances[(cdr_name, hotspot)] = best[0]
            pair_details[(cdr_name, hotspot)] = {
                "cdr_residue": "%s%s" % (heavy_chain, best[1]),
                "hotspot_atom": best[2],
            }

    cdr_minima = {
        cdr: min(pair_distances[(cdr, hotspot)] for hotspot in hotspot_atoms)
        for cdr in CDR_NAMES
    }
    hotspot_minima = {
        hotspot: min(pair_distances[(cdr, hotspot)] for cdr in CDR_NAMES)
        for hotspot in hotspot_atoms
    }
    ordered_hotspot_minima = sorted(hotspot_minima.values())
    second_hotspot_distance = ordered_hotspot_minima[1]
    h12_distance = min(cdr_minima["CDRH1"], cdr_minima["CDRH2"])
    screen_distance = max(cdr_minima["CDRH3"], h12_distance, second_hotspot_distance)
    passed = screen_distance < float(cutoff)
    rejection_reasons = []
    if cdr_minima["CDRH3"] >= cutoff:
        rejection_reasons.append("CDRH3_min_not_below_%.1f" % cutoff)
    if h12_distance >= cutoff:
        rejection_reasons.append("both_CDRH1_CDRH2_min_not_below_%.1f" % cutoff)
    coverage = sum(distance < cutoff for distance in hotspot_minima.values())
    if coverage < 2:
        rejection_reasons.append("hotspot_coverage_below_2_at_%.1f" % cutoff)

    return {
        "backbone_id": Path(backbone_pdb).stem,
        "source_pdb": str(Path(backbone_pdb).resolve()),
        "filter_pass": passed,
        "screen_distance_A": screen_distance,
        "CDRH1_min_A": cdr_minima["CDRH1"],
        "CDRH2_min_A": cdr_minima["CDRH2"],
        "CDRH3_min_A": cdr_minima["CDRH3"],
        "H1_or_H2_min_A": h12_distance,
        "hotspot_second_min_A": second_hotspot_distance,
        "cutoff_A": float(cutoff),
        "hotspot_coverage_below_cutoff": coverage,
        "rejection_reasons": ";".join(rejection_reasons),
        "cdr_residue_ranges": {
            cdr: "%s%s-%s%s" % (heavy_chain, residues[0]["label"], heavy_chain, residues[-1]["label"])
            for cdr, residues in cdrs.items()
        },
        "cdr_minima": cdr_minima,
        "hotspot_minima": hotspot_minima,
        "pair_distances": {
            "%s__%s" % (cdr, hotspot): pair_distances[(cdr, hotspot)]
            for cdr in CDR_NAMES
            for hotspot in hotspot_atoms
        },
        "pair_details": {
            "%s__%s" % (cdr, hotspot): pair_details[(cdr, hotspot)]
            for cdr in CDR_NAMES
            for hotspot in hotspot_atoms
        },
    }


def _flatten_record(record, hotspots):
    row = {
        "rank": None,
        "backbone_id": record["backbone_id"],
        "filter_pass": record["filter_pass"],
        "screen_distance_A": record["screen_distance_A"],
        "cutoff_A": record["cutoff_A"],
        "CDRH1_min_A": record["CDRH1_min_A"],
        "CDRH2_min_A": record["CDRH2_min_A"],
        "CDRH3_min_A": record["CDRH3_min_A"],
        "H1_or_H2_min_A": record["H1_or_H2_min_A"],
        "hotspot_second_min_A": record["hotspot_second_min_A"],
        "hotspot_coverage_below_cutoff": record["hotspot_coverage_below_cutoff"],
        "CDRH1_range": record["cdr_residue_ranges"]["CDRH1"],
        "CDRH2_range": record["cdr_residue_ranges"]["CDRH2"],
        "CDRH3_range": record["cdr_residue_ranges"]["CDRH3"],
        "rejection_reasons": record["rejection_reasons"],
        "source_pdb": record["source_pdb"],
    }
    for hotspot in hotspots:
        row["%s_min_A" % hotspot] = record["hotspot_minima"][hotspot]
    for cdr in CDR_NAMES:
        for hotspot in hotspots:
            row["%s_to_%s_A" % (cdr, hotspot)] = record["pair_distances"]["%s__%s" % (cdr, hotspot)]
    return row


def screen_backbones(input_dir, output_dir, framework_pdb, hotspots, heavy_chain="A", cutoff=8.0):
    """Apply the nanobody geometry profile and write table-only results."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    pdbs = sorted(input_dir.glob("*.pdb"))
    if not pdbs:
        raise FileNotFoundError("no PDB backbones found in %s" % input_dir)
    canonical_hotspots = ["%s%s%s" % _parse_hotspot(value) for value in hotspots]
    if len(canonical_hotspots) < 2:
        raise ValueError("backbone screening requires at least two hotspots")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _score_with_progress(
        pdbs,
        output_dir,
        "nanobody_cdr_hotspot",
        lambda pdb: score_backbone(
            pdb,
            framework_pdb,
            canonical_hotspots,
            heavy_chain=heavy_chain,
            cutoff=cutoff,
        ),
    )
    records.sort(key=lambda item: (item["screen_distance_A"], item["backbone_id"]))
    rows = [_flatten_record(record, canonical_hotspots) for record in records]
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank

    csv_path = output_dir / "backbone_hotspot_distances_sorted.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    accepted = [record for record in records if record["filter_pass"]]

    summary = {
        "screening_profile": "nanobody_cdr_hotspot",
        "input_backbones": len(records),
        "accepted_backbones": len(accepted),
        "rejected_backbones": len(records) - len(accepted),
        "cutoff_A": float(cutoff),
        "heavy_chain": heavy_chain,
        "hotspots": canonical_hotspots,
        "distance_definition": "CDR C-alpha to hotspot residue heavy atoms; minimum Euclidean distance",
        "comparison": "strictly less than cutoff",
        "pass_rule": "CDRH3_min < cutoff AND min(CDRH1_min, CDRH2_min) < cutoff AND at least 2 hotspot minima < cutoff",
        "sort_key": "screen_distance_A = max(CDRH3_min, min(CDRH1_min, CDRH2_min), second-smallest hotspot minimum)",
        "sorted_csv": str(csv_path.resolve()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def score_scfv_backbone(
    backbone_pdb,
    framework_pdb,
    hotspots,
    heavy_chain="A",
    light_chain="B",
    cutoff=8.0,
):
    """Score one scFv backbone using the user-selected hotspot set."""
    cdrs = identify_scfv_cdr_residues(
        backbone_pdb,
        framework_pdb,
        heavy_chain=heavy_chain,
        light_chain=light_chain,
    )
    hotspot_atoms = _hotspot_atoms(backbone_pdb, hotspots)
    pair_distances = {}
    for cdr_name, residues in cdrs.items():
        c_alpha = []
        for residue in residues:
            atom = next((atom for atom in residue["atoms"] if atom["name"] == "CA"), None)
            if atom is not None:
                c_alpha.append(atom["coord"])
        if not c_alpha:
            raise ValueError("%s has no C-alpha atoms in %s" % (cdr_name, backbone_pdb))
        for hotspot, atoms in hotspot_atoms.items():
            pair_distances[(cdr_name, hotspot)] = min(
                _distance(cdr_coord, hotspot_atom["coord"])
                for cdr_coord in c_alpha
                for hotspot_atom in atoms
            )

    cdr_minima = {
        cdr: min(pair_distances[(cdr, hotspot)] for hotspot in hotspot_atoms)
        for cdr in SCFV_CDR_NAMES
    }
    hotspot_minima = {
        hotspot: min(pair_distances[(cdr, hotspot)] for cdr in SCFV_CDR_NAMES)
        for hotspot in hotspot_atoms
    }
    cdr3_anchor = min(cdr_minima["CDRH3"], cdr_minima["CDRL3"])
    active_cdr_count = sum(value < cutoff for value in cdr_minima.values())
    hotspot_coverage = sum(value < cutoff for value in hotspot_minima.values())
    third_cdr_distance = sorted(cdr_minima.values())[2]
    third_hotspot_distance = sorted(hotspot_minima.values())[2]
    screen_distance = max(cdr3_anchor, third_cdr_distance, third_hotspot_distance)
    passed = screen_distance < float(cutoff)
    rejection_reasons = []
    if cdr3_anchor >= cutoff:
        rejection_reasons.append("neither_CDRH3_nor_CDRL3_below_%.1f" % cutoff)
    if active_cdr_count < 3:
        rejection_reasons.append("active_CDR_count_below_3_at_%.1f" % cutoff)
    if hotspot_coverage < 3:
        rejection_reasons.append("hotspot_coverage_below_3_at_%.1f" % cutoff)

    return {
        "backbone_id": Path(backbone_pdb).stem,
        "source_pdb": str(Path(backbone_pdb).resolve()),
        "filter_pass": passed,
        "screen_distance_A": screen_distance,
        "cutoff_A": float(cutoff),
        "CDR3_anchor_min_A": cdr3_anchor,
        "third_CDR_min_A": third_cdr_distance,
        "third_hotspot_min_A": third_hotspot_distance,
        "active_CDR_count": active_cdr_count,
        "hotspot_coverage_below_cutoff": hotspot_coverage,
        "rejection_reasons": ";".join(rejection_reasons),
        "cdr_residue_ranges": {
            cdr: "%s%s-%s%s" % (
                heavy_chain if cdr.startswith("CDRH") else light_chain,
                residues[0]["label"],
                heavy_chain if cdr.startswith("CDRH") else light_chain,
                residues[-1]["label"],
            )
            for cdr, residues in cdrs.items()
        },
        "cdr_minima": cdr_minima,
        "hotspot_minima": hotspot_minima,
        "pair_distances": {
            "%s__%s" % (cdr, hotspot): pair_distances[(cdr, hotspot)]
            for cdr in SCFV_CDR_NAMES
            for hotspot in hotspot_atoms
        },
    }


def _flatten_scfv_record(record, hotspots):
    row = {
        "rank": None,
        "backbone_id": record["backbone_id"],
        "filter_pass": record["filter_pass"],
        "screen_distance_A": record["screen_distance_A"],
        "cutoff_A": record["cutoff_A"],
        "CDR3_anchor_min_A": record["CDR3_anchor_min_A"],
        "third_CDR_min_A": record["third_CDR_min_A"],
        "third_hotspot_min_A": record["third_hotspot_min_A"],
        "active_CDR_count": record["active_CDR_count"],
        "hotspot_coverage_below_cutoff": record["hotspot_coverage_below_cutoff"],
    }
    for cdr in SCFV_CDR_NAMES:
        row["%s_min_A" % cdr] = record["cdr_minima"][cdr]
        row["%s_range" % cdr] = record["cdr_residue_ranges"][cdr]
    row["rejection_reasons"] = record["rejection_reasons"]
    row["source_pdb"] = record["source_pdb"]
    for hotspot in hotspots:
        row["%s_min_A" % hotspot] = record["hotspot_minima"][hotspot]
    for cdr in SCFV_CDR_NAMES:
        for hotspot in hotspots:
            row["%s_to_%s_A" % (cdr, hotspot)] = record["pair_distances"][
                "%s__%s" % (cdr, hotspot)
            ]
    return row


def screen_scfv_backbones(
    input_dir,
    output_dir,
    framework_pdb,
    hotspots,
    heavy_chain="A",
    light_chain="B",
    cutoff=8.0,
):
    """Apply the scFv six-CDR geometry profile and write table-only results."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    pdbs = sorted(input_dir.glob("*.pdb"))
    if not pdbs:
        raise FileNotFoundError("no PDB backbones found in %s" % input_dir)
    canonical_hotspots = ["%s%s%s" % _parse_hotspot(value) for value in hotspots]
    if len(canonical_hotspots) < 3:
        raise ValueError("scFv screening requires at least three specified hotspots")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _score_with_progress(
        pdbs,
        output_dir,
        "scfv_six_cdr_hotspot",
        lambda pdb: score_scfv_backbone(
            pdb,
            framework_pdb,
            canonical_hotspots,
            heavy_chain=heavy_chain,
            light_chain=light_chain,
            cutoff=cutoff,
        ),
    )
    records.sort(key=lambda item: (item["screen_distance_A"], item["backbone_id"]))
    rows = [_flatten_scfv_record(record, canonical_hotspots) for record in records]
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank

    csv_path = output_dir / "scfv_backbone_hotspot_distances_sorted.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    accepted = [record for record in records if record["filter_pass"]]

    summary = {
        "screening_profile": "scfv_six_cdr_hotspot",
        "input_backbones": len(records),
        "accepted_backbones": len(accepted),
        "rejected_backbones": len(records) - len(accepted),
        "cutoff_A": float(cutoff),
        "comparison": "strictly less than cutoff",
        "heavy_chain": heavy_chain,
        "light_chain": light_chain,
        "hotspots": canonical_hotspots,
        "distance_definition": "CDR C-alpha to hotspot residue heavy atoms; minimum Euclidean distance",
        "pass_rule": "min(CDRH3_min, CDRL3_min) < cutoff AND at least 3 CDR minima < cutoff AND at least 3 specified-hotspot minima < cutoff",
        "sort_key": "screen_distance_A = max(CDR3 anchor minimum, third-smallest CDR minimum, third-smallest hotspot minimum)",
        "sorted_csv": str(csv_path.resolve()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def select_top_scfv_backbones(output_dir, top_n):
    """Record the reviewed top-N passing scFv or VHH backbones without copies."""
    output_dir = Path(output_dir)
    candidates = (
        ("scfv_six_cdr_hotspot", output_dir / "scfv_backbone_hotspot_distances_sorted.csv"),
        ("nanobody_cdr_hotspot", output_dir / "backbone_hotspot_distances_sorted.csv"),
    )
    selected_table = next(
        ((profile, table) for profile, table in candidates if table.is_file()),
        None,
    )
    if selected_table is None:
        raise FileNotFoundError(
            "ranked backbone table not found in %s" % output_dir
        )
    screening_profile, csv_path = selected_table
    with csv_path.open(newline="", encoding="utf-8") as handle:
        passing = [
            row
            for row in csv.DictReader(handle)
            if str(row.get("filter_pass", "")).strip().lower() == "true"
        ]
    top_n = int(top_n)
    if not 1 <= top_n <= len(passing):
        raise ValueError("top_n must be between 1 and %s" % len(passing))
    selected = passing[:top_n]
    selection = {
        "screening_profile": screening_profile,
        "available_passing_backbones": len(passing),
        "selected_top_n": top_n,
        "selection_order": "ascending screen_distance_A",
        "ranked_table": str(csv_path.resolve()),
        "selected": [
            {
                "rank": int(row["rank"]),
                "backbone_id": row["backbone_id"],
                "screen_distance_A": float(row["screen_distance_A"]),
                "source_pdb": row["source_pdb"],
            }
            for row in selected
        ],
    }
    (output_dir / "backbone_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return selection
