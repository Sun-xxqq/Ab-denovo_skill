#!/usr/bin/env python3
"""Compute Chai/ipSAE/geometry metrics for final matured candidates."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from evaluate_chai_candidates import (
    _cdr_positions,
    _collision_value,
    _heavy_contacts,
    _interface_pae,
    _parse_ipsae,
    _plddt_metrics,
    _rmsd_metrics,
)


def _hotspot_numbers(values):
    result = set()
    for value in values or []:
        digits = "".join(character for character in str(value) if character.isdigit())
        if digits:
            result.add(int(digits))
    return result


def _developability(sequence):
    # Keep this script standalone in the PPIFlow environment.
    from agent_ab.functional import sequence_developability

    return sequence_developability(sequence)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--chai-dir", required=True)
    parser.add_argument("--ipsae-script", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pae-cutoff", type=float, default=10)
    parser.add_argument("--dist-cutoff", type=float, default=10)
    args = parser.parse_args()

    jobs = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    chai_dir, output = Path(args.chai_dir), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records, failures = [], []
    for job in jobs:
        candidate_id = job["candidate_id"]
        candidate = chai_dir / job.get("output_relpath", candidate_id)
        metrics_path = candidate / "metrics.json"
        predicted = candidate / "pred.cif"
        if not predicted.exists():
            predicted = candidate / "predicted.cif"
        if not metrics_path.exists() or not predicted.exists():
            failures.append({"candidate_id": candidate_id, "error": "missing Chai metrics or prediction"})
            continue
        result_path = candidate / (
            "%s_%02d_%02d.txt"
            % (predicted.stem, int(args.pae_cutoff), int(args.dist_cutoff))
        )
        if not result_path.exists():
            log_path = output / (candidate_id + ".ipsae.log")
            with log_path.open("a", encoding="utf-8") as log:
                completed = subprocess.run(
                    [sys.executable, args.ipsae_script, str(candidate / "pae_candidate.npz"), str(predicted), str(args.pae_cutoff), str(args.dist_cutoff)],
                    stdout=log, stderr=subprocess.STDOUT, check=False,
                )
            if completed.returncode or not result_path.exists():
                failures.append({"candidate_id": candidate_id, "error": "ipSAE failed", "log": str(log_path)})
                continue

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        design = candidate / "design.pdb"
        if not design.exists():
            design = Path(job["pdb"])
        design_chains = job["design_chains"]
        target_chain = job["target_chain"]
        cdr = _cdr_positions(job, design_chains)
        design_contacts, target_contacts = _heavy_contacts(predicted, design_chains, target_chain)
        cdr_keys = {(chain, position) for chain in design_chains for position in cdr.get(chain, [])}
        hotspots = _hotspot_numbers(job.get("hotspots", []))
        metrics.update(
            {
                "candidate_id": candidate_id,
                "backbone_id": job["backbone_id"],
                "final_design_id": job["design_id"],
                "source_pdb": str(design),
                "predicted_cif": str(predicted),
                "has_inter_chain_clashes": _collision_value(metrics, candidate),
                "hotspots_total": len(hotspots),
                "hotspot_contacts": len(hotspots & target_contacts),
                "hotspot_coverage": (len(hotspots & target_contacts) / len(hotspots)) if hotspots else None,
                "cdr_contact_fraction": (len(design_contacts & cdr_keys) / len(design_contacts)) if design_contacts and cdr_keys else None,
                "interface_definition": "%s_%s" % ("".join(design_chains), target_chain),
            }
        )
        metrics.update(_rmsd_metrics(design, predicted, design_chains, target_chain, cdr))
        metrics.update(_interface_pae(candidate, predicted, job["sequences"], design_chains, target_chain))
        metrics.update(_plddt_metrics(candidate, job, design_chains, cdr, design_contacts))
        metrics["global_plddt"] = metrics.get("plddt")
        if metrics.get("antibody_plddt") is not None:
            metrics["plddt"] = metrics["antibody_plddt"]
        metrics["iptm_global"] = metrics.get("iptm")
        metrics["ipsae_min"], metrics["iptm"] = _parse_ipsae(result_path, design_chains, target_chain)
        designed_sequence = "".join(
            item["sequence"] for item in job["sequences"] if item["chain"] in design_chains
        )
        metrics.update(_developability(designed_sequence))
        records.append(metrics)

    (output / "structural_interface_metrics.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
