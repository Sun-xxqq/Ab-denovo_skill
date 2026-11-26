#!/usr/bin/env python3
"""Apply the Stage 1 filter to normalized Protenix design/pred pairs."""

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_ab.filtering import (
    DEFAULT_LOOSE_THRESHOLDS,
    DEFAULT_STAGE2_THRESHOLDS,
    passes_loose_filter,
)


def _model(path):
    parser = MMCIFParser(QUIET=True) if str(path).lower().endswith(".cif") else PDBParser(QUIET=True)
    return parser.get_structure("model", str(path))[0]


def _ca_maps(path, chains):
    result = {}
    for chain in _model(path):
        if chain.id not in chains:
            continue
        values = {}
        for residue in chain:
            if "CA" in residue:
                values[int(residue.id[1])] = np.asarray(residue["CA"].coord, dtype=float)
        result[chain.id] = values
    return result


def _fallback_cdr_positions(job, antibody_chains):
    result = {}
    lengths = {item["chain"]: len(item["sequence"]) for item in job["sequences"]}
    for index, chain in enumerate(antibody_chains):
        length = lengths.get(chain, 0)
        ranges = (
            ((26, 35), (51, 65), (97, max(97, length - 11)))
            if index == 0
            else ((24, 35), (50, 56), (89, max(89, length - 10)))
        )
        result[chain] = [position for start, end in ranges for position in range(start, min(end, length) + 1)]
    return result


def _cdr_positions(job, antibody_chains):
    supplied = job.get("designed_positions") or {}
    values = {chain: [int(value) for value in supplied.get(chain, [])] for chain in antibody_chains}
    return values if any(values.values()) else _fallback_cdr_positions(job, antibody_chains)


def _rmsd_after_alignment(design_maps, pred_maps, align_keys, measure_keys):
    align = [key for key in align_keys if key[0] in design_maps and key[0] in pred_maps and key[1] in design_maps[key[0]] and key[1] in pred_maps[key[0]]]
    measure = [key for key in measure_keys if key[0] in design_maps and key[0] in pred_maps and key[1] in design_maps[key[0]] and key[1] in pred_maps[key[0]]]
    if len(align) < 3 or not measure:
        return None
    left = np.asarray([design_maps[chain][position] for chain, position in align])
    right = np.asarray([pred_maps[chain][position] for chain, position in align])
    left_center, right_center = left.mean(axis=0), right.mean(axis=0)
    u, _, vt = np.linalg.svd((left - left_center).T @ (right - right_center))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    left_eval = np.asarray([design_maps[chain][position] for chain, position in measure])
    right_eval = np.asarray([pred_maps[chain][position] for chain, position in measure])
    transformed = (left_eval - left_center) @ rotation + right_center
    delta = transformed - right_eval
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def _rmsd_metrics(design_path, predicted_path, antibody_chains, target_chain, cdr=None):
    """Return one whole-complex C-alpha RMSD after optimal rigid alignment."""
    chains = set(antibody_chains + [target_chain])
    design, predicted = _ca_maps(design_path, chains), _ca_maps(predicted_path, chains)
    all_keys = [
        (chain, position)
        for chain in sorted(chains)
        for position in design.get(chain, {})
    ]
    return {
        "design_pred_rmsd": _rmsd_after_alignment(
            design, predicted, all_keys, all_keys
        ),
    }


def _heavy_contacts(path, antibody_chains, target_chain, cutoff=5.0):
    model = _model(path)
    antibody, target = [], []
    for chain in model:
        if chain.id not in set(antibody_chains + [target_chain]):
            continue
        for residue in chain:
            atoms = [np.asarray(atom.coord, dtype=float) for atom in residue if str(atom.element).upper() != "H"]
            if not atoms:
                continue
            item = (chain.id, int(residue.id[1]), np.asarray(atoms))
            (target if chain.id == target_chain else antibody).append(item)
    antibody_contacts, target_contacts = set(), set()
    for chain, residue_number, antibody_atoms in antibody:
        for _, target_number, target_atoms in target:
            distances = np.linalg.norm(antibody_atoms[:, None, :] - target_atoms[None, :, :], axis=2)
            if float(np.min(distances)) <= cutoff:
                antibody_contacts.add((chain, residue_number))
                target_contacts.add(target_number)
    return antibody_contacts, target_contacts


def _interface_pae(candidate_dir, predicted_path, sequences, antibody_chains, target_chain, cutoff=10.0):
    with np.load(candidate_dir / "pae_candidate.npz") as stored:
        pae = np.asarray(stored["pae"], dtype=float)
    coordinates = _ca_maps(predicted_path, set(antibody_chains + [target_chain]))
    offsets, cursor = {}, 0
    for item in sequences:
        offsets[item["chain"]] = cursor
        cursor += len(item["sequence"])
    values = []
    target_positions = sorted(coordinates.get(target_chain, {}))
    for antibody in antibody_chains:
        antibody_positions = sorted(coordinates.get(antibody, {}))
        for row, antibody_position in enumerate(antibody_positions):
            for column, target_position in enumerate(target_positions):
                if np.linalg.norm(coordinates[antibody][antibody_position] - coordinates[target_chain][target_position]) >= cutoff:
                    continue
                ai = offsets[antibody] + row
                ti = offsets[target_chain] + column
                if ai < pae.shape[0] and ti < pae.shape[1]:
                    values.extend([pae[ai, ti], pae[ti, ai]])
    return {
        "interface_pae_mean": float(np.mean(values)) if values else None,
        "interface_pae_p90": float(np.percentile(values, 90)) if values else None,
    }


def _antibody_plddt_metric(candidate_dir, job, antibody_chains):
    with np.load(candidate_dir / "plddt_candidate.npz") as stored:
        plddt = np.asarray(stored["plddt"], dtype=float).reshape(-1)
    offsets, lengths, cursor = {}, {}, 0
    for item in job["sequences"]:
        offsets[item["chain"]] = cursor
        lengths[item["chain"]] = len(item["sequence"])
        cursor += len(item["sequence"])

    indices = [
        offsets[chain] + position - 1
        for chain in antibody_chains
        for position in range(1, lengths.get(chain, 0) + 1)
        if chain in offsets
    ]
    values = [plddt[index] for index in indices if index < len(plddt)]
    return {"antibody_plddt": float(np.mean(values)) if values else None}


def _plddt_metrics(candidate_dir, job, antibody_chains, cdr, antibody_contacts):
    """Full pLDDT breakdown retained for final functional evaluation."""
    with np.load(candidate_dir / "plddt_candidate.npz") as stored:
        plddt = np.asarray(stored["plddt"], dtype=float).reshape(-1)
    offsets, lengths, cursor = {}, {}, 0
    for item in job["sequences"]:
        offsets[item["chain"]] = cursor
        lengths[item["chain"]] = len(item["sequence"])
        cursor += len(item["sequence"])

    def values(keys):
        indices = [
            offsets[chain] + position - 1
            for chain, position in keys
            if chain in offsets and 1 <= position <= lengths[chain]
        ]
        return [plddt[index] for index in indices if index < len(plddt)]

    antibody_keys = [
        (chain, position)
        for chain in antibody_chains
        for position in range(1, lengths.get(chain, 0) + 1)
    ]
    cdr_keys = [
        (chain, position)
        for chain in antibody_chains
        for position in cdr.get(chain, [])
    ]
    antibody_values = values(antibody_keys)
    cdr_values = values(cdr_keys)
    interface_values = values(antibody_contacts)
    cdr_plddt = float(np.mean(cdr_values)) if cdr_values else None
    interface_plddt = (
        float(np.mean(interface_values)) if interface_values else None
    )
    combined = (
        min(cdr_plddt, interface_plddt)
        if cdr_plddt is not None and interface_plddt is not None
        else None
    )
    return {
        "antibody_plddt": (
            float(np.mean(antibody_values)) if antibody_values else None
        ),
        "cdr_plddt": cdr_plddt,
        "interface_plddt": interface_plddt,
        "cdr_interface_plddt": combined,
    }


def _parse_ipsae(path, antibody_chains, target_chain):
    """Return the bidirectional on-target ipSAE score used by Stage 1.

    ipSAE reports one asymmetric score per ordered chain pair.  An antibody can
    contain one (VHH) or two (VH/VL) chains, so each direction is represented by
    the strongest antibody-chain contribution before applying the conservative
    minimum requested for S_on.
    """
    directional = {}
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        fields = raw.split()
        if len(fields) < 9 or fields[4] != "asym":
            continue
        try:
            directional[(fields[0], fields[1])] = float(fields[5])
        except ValueError:
            continue

    antibody_to_target = [
        directional[(chain, target_chain)]
        for chain in antibody_chains
        if (chain, target_chain) in directional
    ]
    target_to_antibody = [
        directional[(target_chain, chain)]
        for chain in antibody_chains
        if (target_chain, chain) in directional
    ]
    ab_to_target = max(antibody_to_target) if antibody_to_target else None
    target_to_ab = max(target_to_antibody) if target_to_antibody else None
    s_on = (
        min(ab_to_target, target_to_ab)
        if ab_to_target is not None and target_to_ab is not None
        else None
    )
    return {
        "ipsae_antibody_to_target": ab_to_target,
        "ipsae_target_to_antibody": target_to_ab,
        "S_on": s_on,
        # Keep the historical name so downstream reports remain readable.
        "ipsae_min": s_on,
    }


def _collision_value(metrics, candidate_dir):
    if "has_inter_chain_clashes" in metrics:
        return bool(metrics["has_inter_chain_clashes"])
    paths = sorted(candidate_dir.glob("attempt_*/scores.model_idx_0.npz"))
    if not paths:
        return None
    with np.load(paths[-1]) as stored:
        if "has_inter_chain_clashes" not in stored.files:
            return None
        return bool(np.asarray(stored["has_inter_chain_clashes"]).reshape(-1)[0])


def _hotspot_numbers(values, target_chain):
    result = set()
    for value in values:
        match = re.match(r"^%s(-?\d+)" % re.escape(target_chain), str(value).strip())
        if match:
            result.add(int(match.group(1)))
    return result


def _safe_remove(path, root, directory=False):
    path, root = Path(path).resolve(), Path(root).resolve()
    path.relative_to(root)
    if not path.exists():
        return False
    if directory:
        shutil.rmtree(path)
    else:
        path.unlink()
    return True


def _write_csv(path, records):
    fields = sorted({key for record in records for key in record if key != "rejection_reasons"}) + ["rejection_reasons"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            item = dict(record)
            item["rejection_reasons"] = ";".join(record.get("rejection_reasons", []))
            writer.writerow(item)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-dir", "--chai-dir", dest="prediction_dir", required=True
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ipsae-script", required=True)
    parser.add_argument("--ipsae-python", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selected-dir", required=True)
    parser.add_argument("--stage1-candidates-dir", required=True)
    parser.add_argument("--thresholds-json", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--delete-rejected", action="store_true")
    parser.add_argument("--hotspots", default="")
    parser.add_argument("--pae-cutoff", type=float, default=10)
    parser.add_argument("--dist-cutoff", type=float, default=10)
    parser.add_argument("--heavy-chain", default="A")
    parser.add_argument("--light-chain", default="B")
    parser.add_argument("--target-chain", default="C")
    parser.add_argument(
        "--skip-ipsae",
        action="store_true",
        help="apply structural gates only; used by Stage 1",
    )
    args = parser.parse_args()

    prediction_dir, output = Path(args.prediction_dir), Path(args.output)
    selected_dir, stage1_root = Path(args.selected_dir), Path(args.stage1_candidates_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_dir.mkdir(parents=True, exist_ok=True)
    stage1_root.mkdir(parents=True, exist_ok=True)
    thresholds = dict(DEFAULT_LOOSE_THRESHOLDS)
    configured_thresholds = json.loads(
        Path(args.thresholds_json).read_text(encoding="utf-8")
    )
    thresholds.update(
        {
            key: value
            for key, value in configured_thresholds.items()
            if key in DEFAULT_STAGE2_THRESHOLDS
        }
    )
    jobs = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    antibody_chains = [args.heavy_chain] + ([args.light_chain] if args.light_chain else [])
    records, pending_deletions = [], []

    for job in jobs:
        candidate_id = job["candidate_id"]
        candidate = prediction_dir / job.get("output_relpath", candidate_id)
        metrics_path = candidate / "metrics.json"
        predicted = candidate / "pred.cif"
        if not predicted.exists():
            predicted = candidate / "predicted.cif"
        if not metrics_path.exists() or not predicted.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        pae_path = candidate / "pae_candidate.npz"
        result_path = candidate / (
            "%s_%02d_%02d.txt"
            % (predicted.stem, int(args.pae_cutoff), int(args.dist_cutoff))
        )
        if not args.skip_ipsae and pae_path.exists() and not result_path.exists():
            with (output / (candidate_id + ".ipsae.log")).open(
                "a", encoding="utf-8"
            ) as log:
                completed = subprocess.run(
                    [
                        args.ipsae_python,
                        args.ipsae_script,
                        str(pae_path),
                        str(predicted),
                        str(args.pae_cutoff),
                        str(args.dist_cutoff),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if completed.returncode:
                metrics["ipsae_error"] = "exit_code_%s" % completed.returncode
        if not args.skip_ipsae and result_path.exists():
            metrics.update(
                _parse_ipsae(result_path, antibody_chains, args.target_chain)
            )
        else:
            metrics.update(
                {
                    "ipsae_antibody_to_target": None,
                    "ipsae_target_to_antibody": None,
                    "S_on": None,
                    "ipsae_min": None,
                }
            )
        design = candidate / "design.pdb"
        if not design.exists():
            design = Path(job["pdb"])
        metrics.update({
            "candidate_id": candidate_id,
            "backbone_id": job.get("backbone_id"),
            "design_id": job.get("design_id"),
            "source_pdb": str(design),
            "original_source_pdb": job["pdb"],
              "variant_fasta": job.get("variant_fasta"),
            "predicted_cif": str(predicted),
            "has_inter_chain_clashes": _collision_value(metrics, candidate),
        })
        if pae_path.exists():
            metrics.update(_interface_pae(candidate, predicted, job["sequences"], antibody_chains, args.target_chain))
        if (candidate / "plddt_candidate.npz").exists():
            metrics.update(_antibody_plddt_metric(candidate, job, antibody_chains))
        metrics.update(
            _rmsd_metrics(
                design,
                predicted,
                antibody_chains,
                args.target_chain,
            )
        )
        metrics["iptm_global"] = metrics.get("iptm")
        if not args.skip_ipsae and metrics.get("S_on") is None:
            metrics["filter_pass"] = None
            metrics["rejection_reasons"] = []
            metrics["status"] = "UNCERTAIN"
            metrics["uncertain_reason"] = (
                "ON_TARGET_PAE_OR_DIRECTIONAL_IPSAE_MISSING"
            )
        else:
            metrics["filter_pass"], metrics["rejection_reasons"] = passes_loose_filter(metrics, thresholds)
            metrics["status"] = "OK"
            metrics["uncertain_reason"] = ""
        records.append(metrics)

        if metrics["filter_pass"]:
            original = Path(job["pdb"])
            selected_source = Path(job.get("variant_fasta") or original)
            selected_name = (
                metrics["candidate_id"] + selected_source.suffix
                if job.get("variant_fasta")
                else selected_source.name
            )
            shutil.copy2(selected_source, selected_dir / selected_name)
            group_dir = stage1_root / metrics["backbone_id"]
            group_dir.mkdir(parents=True, exist_ok=True)
            suffix = selected_source.suffix or ".fasta"
            shutil.copy2(
                selected_source,
                group_dir / (metrics["design_id"] + suffix),
            )
        elif args.delete_rejected:
            pending_deletions.append((candidate_id, candidate, Path(job["pdb"])))

    grouped = {}
    for record in records:
        grouped.setdefault(record.get("backbone_id") or "backbone_unknown", []).append(record)
    group_index = {}
    for backbone_id, group in sorted(grouped.items()):
        group_output = output / backbone_id
        group_output.mkdir(parents=True, exist_ok=True)
        (group_output / "filter_results.json").write_text(json.dumps(group, indent=2) + "\n", encoding="utf-8")
        _write_csv(group_output / "filter_results.csv", group)
        accepted = [item for item in group if item.get("filter_pass") is True]
        rejected = [item for item in group if item.get("filter_pass") is False]
        candidate_group = stage1_root / backbone_id
        candidate_group.mkdir(parents=True, exist_ok=True)
        (candidate_group / "stage1_candidates.json").write_text(json.dumps(accepted, indent=2) + "\n", encoding="utf-8")
        (group_output / "rejected_candidates.json").write_text(json.dumps(rejected, indent=2) + "\n", encoding="utf-8")
        group_index[backbone_id] = {
            "evaluated": len(group),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "uncertain": sum(item.get("status") == "UNCERTAIN" for item in group),
            "directory": str(candidate_group),
        }

    (stage1_root / "stage1_groups.json").write_text(json.dumps(group_index, indent=2) + "\n", encoding="utf-8")
    summary = {
        "mode": "loose_filter_only", "ranking": False,
        "ipsae_screening": "skipped" if args.skip_ipsae else "enabled",
        "delete_rejected": bool(args.delete_rejected), "thresholds": thresholds,
        "evaluated": len(records),
        "accepted": sum(item.get("filter_pass") is True for item in records),
        "rejected": sum(item.get("filter_pass") is False for item in records),
        "uncertain": sum(item.get("status") == "UNCERTAIN" for item in records),
        "groups": group_index,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    deletion_log = []
    flowpacker_root = Path(args.run_dir) / "pipeline_output" / "stage1" / "flowpacker_output"
    for candidate_id, candidate, original in pending_deletions:
        entry = {"candidate_id": candidate_id, "deleted": []}
        try:
            if _safe_remove(candidate, prediction_dir, directory=True):
                entry["deleted"].append(str(candidate))
            if _safe_remove(original, flowpacker_root, directory=False):
                entry["deleted"].append(str(original))
        except (OSError, ValueError) as exc:
            entry["error"] = str(exc)
        deletion_log.append(entry)
    (output / "deletion_log.json").write_text(json.dumps(deletion_log, indent=2) + "\n", encoding="utf-8")
    if not records:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
