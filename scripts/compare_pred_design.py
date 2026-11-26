#!/usr/bin/env python3
"""Compare second-round predictor structures with FlowPacker designs.

This is intentionally the terminal stage of the default workflow: it records
structural agreement and does not run affinity scoring, ranking, or selection.
"""

import argparse
import csv
import json
from pathlib import Path

from evaluate_chai_candidates import _rmsd_metrics


def _prediction(candidate):
    for name in ("pred.cif", "predicted.cif", "pred.pdb", "predicted.pdb"):
        path = candidate / name
        if path.exists():
            return path
    return None


def _write_csv(path, records):
    fields = [
        "candidate_id",
        "backbone_id",
        "design_id",
        "design_pdb",
        "predicted_structure",
        "antibody_plddt",
        "ptm",
        "iptm_global",
        "design_pred_rmsd",
        "filter_pass",
        "rejection_reasons",
        "structurally_similar",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plddt-min", type=float, default=0.60)
    parser.add_argument("--ptm-min", type=float, default=0.60)
    parser.add_argument("--iptm-min", type=float, default=0.25)
    parser.add_argument("--design-pred-rmsd-max", type=float, default=6.0)
    args = parser.parse_args()

    jobs = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    prediction_dir = Path(args.prediction_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records, failures = [], []

    for job in jobs:
        candidate_id = job["candidate_id"]
        candidate = prediction_dir / job.get("output_relpath", candidate_id)
        predicted = _prediction(candidate)
        design = candidate / "design.pdb"
        if not design.exists():
            design = Path(job["pdb"])
        if predicted is None or not design.exists():
            failures.append(
                {
                    "candidate_id": candidate_id,
                    "error": "missing prediction or FlowPacker design",
                    "candidate_dir": str(candidate),
                }
            )
            continue

        design_chains = list(job["design_chains"])
        target_chain = job["target_chain"]
        metrics = _rmsd_metrics(design, predicted, design_chains, target_chain)
        design_pred_rmsd = metrics.get("design_pred_rmsd")
        predictor_metrics_path = candidate / "metrics.json"
        predictor_metrics = (
            json.loads(predictor_metrics_path.read_text(encoding="utf-8"))
            if predictor_metrics_path.exists()
            else {}
        )
        antibody_plddt = predictor_metrics.get(
            "antibody_plddt", predictor_metrics.get("plddt")
        )
        ptm = predictor_metrics.get("ptm")
        iptm = predictor_metrics.get("iptm", predictor_metrics.get("iptm_global"))
        reasons = []
        if antibody_plddt is None or float(antibody_plddt) <= args.plddt_min:
            reasons.append("antibody_plddt_not_above_%s" % args.plddt_min)
        if ptm is None or float(ptm) <= args.ptm_min:
            reasons.append("ptm_not_above_%s" % args.ptm_min)
        if iptm is None or float(iptm) <= args.iptm_min:
            reasons.append("iptm_not_above_%s" % args.iptm_min)
        if (
            design_pred_rmsd is None
            or float(design_pred_rmsd) >= args.design_pred_rmsd_max
        ):
            reasons.append(
                "design_pred_rmsd_not_below_%s" % args.design_pred_rmsd_max
            )
        filter_pass = not reasons
        records.append(
            {
                "candidate_id": candidate_id,
                "backbone_id": job["backbone_id"],
                "design_id": job["design_id"],
                "design_pdb": str(design),
                "predicted_structure": str(predicted),
                "antibody_plddt": antibody_plddt,
                "ptm": ptm,
                "iptm_global": iptm,
                "design_pred_rmsd": design_pred_rmsd,
                "filter_pass": filter_pass,
                "rejection_reasons": ";".join(reasons),
                "structurally_similar": bool(
                    design_pred_rmsd is not None
                    and design_pred_rmsd < args.design_pred_rmsd_max
                ),
            }
        )

    (output / "pred_design_comparison.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    (output / "failures.json").write_text(
        json.dumps(failures, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(output / "pred_design_comparison.csv", records)
    summary = {
        "terminal_stage": "stage2_pred_design_comparison",
        "ranking": False,
        "evaluated": len(records),
        "structurally_similar": sum(
            1 for item in records if item["structurally_similar"]
        ),
        "passed": sum(1 for item in records if item["filter_pass"]),
        "failed": len(failures),
        "thresholds": {
            "antibody_plddt_min_exclusive": args.plddt_min,
            "ptm_min_exclusive": args.ptm_min,
            "iptm_min_exclusive": args.iptm_min,
            "design_pred_rmsd_max_exclusive": args.design_pred_rmsd_max,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
