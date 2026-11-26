#!/usr/bin/env python3
"""Pair on-target S_on with a CLDN18.1-only Protenix/ipSAE evaluation."""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_chai_candidates import _model, _parse_ipsae


def _truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass"}


def _load_on_target_results(root):
    records = {}
    for path in sorted(Path(root).glob("*/filter_results.json")):
        values = json.loads(path.read_text(encoding="utf-8"))
        for item in values:
            candidate_id = item.get("candidate_id")
            if not candidate_id or not _truthy(item.get("filter_pass")):
                continue
            if candidate_id in records:
                raise ValueError("duplicate on-target candidate_id: %s" % candidate_id)
            records[candidate_id] = item
    return records


def _sequence_hash(items):
    encoded = json.dumps(items, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _ca_count(path, chain_id):
    model = _model(path)
    for chain in model:
        if chain.id == chain_id:
            return sum(1 for residue in chain if "CA" in residue)
    return 0


def _uncertain(record, reason, detail=None):
    record.update(
        {
            "S_off": None,
            "delta": None,
            "specificity_pass": None,
            "filter_pass": None,
            "rejection_reasons": [],
            "status": "UNCERTAIN",
            "uncertain_reason": reason,
        }
    )
    if detail:
        record["detail"] = str(detail)
    return record


def _apply_delta_gate(record, s_off):
    s_off = float(s_off)
    delta = float(record["S_on"]) - s_off
    passed = delta > 0.0
    record.update(
        {
            "S_off": s_off,
            "delta": delta,
            "specificity_pass": passed,
            "filter_pass": passed,
            "rejection_reasons": [] if passed else ["DELTA_NOT_POSITIVE"],
            "status": "OK",
            "uncertain_reason": "",
        }
    )
    return record


def _write_csv(path, records):
    preferred = [
        "candidate_id",
        "backbone_id",
        "design_id",
        "S_on",
        "ipsae_antibody_to_offtarget",
        "ipsae_offtarget_to_antibody",
        "S_off",
        "delta",
        "design_pred_rmsd",
        "antibody_pred_rmsd",
        "target_pred_rmsd",
        "specificity_pass",
        "filter_pass",
        "status",
        "uncertain_reason",
        "rejection_reasons",
        "selected_from_stage",
        "antibody_chains",
        "target_chain",
        "antibody_sequence_sha256",
        "off_target_sequence_sha256",
        "prediction_path",
        "source_pdb",
        "detail",
    ]
    present = {key for record in records for key in record}
    fields = [key for key in preferred if key in present]
    fields.extend(sorted(present - set(fields)))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            item = dict(record)
            item["rejection_reasons"] = ";".join(
                record.get("rejection_reasons", [])
            )
            writer.writerow(item)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--on-target-results", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--ipsae-script", required=True)
    parser.add_argument("--ipsae-python", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pae-cutoff", type=float, default=10)
    parser.add_argument("--dist-cutoff", type=float, default=10)
    parser.add_argument("--heavy-chain", default="A")
    parser.add_argument("--light-chain", default="")
    parser.add_argument("--target-chain", default="C")
    parser.add_argument("--workflow-stage", default="stage2")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    prediction_dir = Path(args.prediction_dir)
    on_target = _load_on_target_results(args.on_target_results)
    jobs = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    antibody_chains = [args.heavy_chain] + (
        [args.light_chain] if args.light_chain else []
    )
    failures_path = prediction_dir / "failures.json"
    failures = {}
    if failures_path.exists():
        failures = {
            item.get("candidate_id"): item.get("error", "prediction failed")
            for item in json.loads(failures_path.read_text(encoding="utf-8"))
        }

    records = []
    seen = set()
    for job in jobs:
        candidate_id = job["candidate_id"]
        if candidate_id in seen:
            raise ValueError("duplicate off-target candidate_id: %s" % candidate_id)
        seen.add(candidate_id)
        candidate = prediction_dir / job.get("output_relpath", candidate_id)
        predicted = candidate / "pred.cif"
        metrics_path = candidate / "metrics.json"
        pae_path = candidate / "pae_candidate.npz"
        target_sequence = next(
            (
                item["sequence"]
                for item in job["sequences"]
                if item["chain"] == args.target_chain
            ),
            "",
        )
        antibody_sequences = [
            {"chain": item["chain"], "sequence": item["sequence"]}
            for item in job["sequences"]
            if item["chain"] in antibody_chains
        ]
        on_record = on_target.get(candidate_id, {})
        record = {
            "candidate_id": candidate_id,
            "backbone_id": job.get("backbone_id"),
            "design_id": job.get("design_id"),
            "S_on": on_record.get("S_on"),
            "design_pred_rmsd": on_record.get("design_pred_rmsd"),
            "antibody_pred_rmsd": on_record.get("antibody_pred_rmsd"),
            "target_pred_rmsd": on_record.get("target_pred_rmsd"),
            "selected_from_stage": args.workflow_stage,
            "antibody_chains": ",".join(item["chain"] for item in antibody_sequences),
            "target_chain": args.target_chain,
            "antibody_sequence_sha256": _sequence_hash(antibody_sequences),
            "off_target_sequence_sha256": _sequence_hash(
                [{"chain": args.target_chain, "sequence": target_sequence}]
            ),
            "prediction_path": str(predicted),
            "source_pdb": job.get("pdb"),
        }
        if candidate_id not in on_target or record["S_on"] is None:
            records.append(_uncertain(record, "ON_TARGET_SCORE_MISSING"))
            continue
        try:
            record["S_on"] = float(record["S_on"])
        except (TypeError, ValueError):
            records.append(_uncertain(record, "ON_TARGET_SCORE_INVALID"))
            continue
        if candidate_id in failures or not predicted.exists() or not metrics_path.exists():
            records.append(
                _uncertain(
                    record,
                    "OFF_TARGET_PREDICTION_FAILED",
                    failures.get(candidate_id),
                )
            )
            continue
        if not pae_path.exists():
            records.append(_uncertain(record, "OFF_TARGET_PAE_MISSING"))
            continue
        try:
            observed = _ca_count(predicted, args.target_chain)
        except Exception as exc:
            records.append(
                _uncertain(record, "OFF_TARGET_ANTIGEN_STRUCTURE_ABNORMAL", exc)
            )
            continue
        if not target_sequence or observed != len(target_sequence):
            records.append(
                _uncertain(
                    record,
                    "OFF_TARGET_ANTIGEN_STRUCTURE_ABNORMAL",
                    "target CA count %s, expected %s"
                    % (observed, len(target_sequence)),
                )
            )
            continue

        result_path = candidate / (
            "%s_%02d_%02d.txt"
            % (predicted.stem, int(args.pae_cutoff), int(args.dist_cutoff))
        )
        if not result_path.exists():
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
                records.append(
                    _uncertain(
                        record,
                        "OFF_TARGET_IPSAE_FAILED",
                        "exit code %s" % completed.returncode,
                    )
                )
                continue
        if not result_path.exists():
            records.append(_uncertain(record, "OFF_TARGET_IPSAE_OUTPUT_MISSING"))
            continue
        off_target = _parse_ipsae(
            result_path, antibody_chains, args.target_chain
        )
        s_off = off_target.get("S_on")
        if s_off is None:
            records.append(
                _uncertain(record, "OFF_TARGET_DIRECTIONAL_IPSAE_MISSING")
            )
            continue
        record.update(
            {
                "ipsae_antibody_to_offtarget": off_target.get(
                    "ipsae_antibody_to_target"
                ),
                "ipsae_offtarget_to_antibody": off_target.get(
                    "ipsae_target_to_antibody"
                ),
            }
        )
        records.append(_apply_delta_gate(record, s_off))

    result_json = output / "offtarget_results.json"
    result_json.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "offtarget_results.csv", records)
    _write_csv(output.parent / "cldn18_1_offtarget_screen.csv", records)
    summary = {
        "workflow_stage": args.workflow_stage,
        "screening_round": 2,
        "target": "CLDN18.1",
        "pairing_key": "candidate_id",
        "on_target_recomputed": False,
        "backbone_regenerated": False,
        "antibody_redesigned": False,
        "filter_rule": "status == OK and delta > 0",
        "delta_threshold": 0.0,
        "delta_comparison": "strictly_greater_than",
        "evaluated": len(records),
        "ok": sum(item.get("status") == "OK" for item in records),
        "passed": sum(item.get("specificity_pass") is True for item in records),
        "rejected": sum(item.get("specificity_pass") is False for item in records),
        "uncertain": sum(item.get("status") == "UNCERTAIN" for item in records),
        "final_candidates_are_subset_of_on_target_passes": True,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
