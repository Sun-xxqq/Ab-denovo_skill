#!/usr/bin/env python3
"""Run PyRosetta InterfaceAnalyzer for every final Chai prediction."""

import argparse
import json
import traceback
from pathlib import Path


def _optional_metric(data, name):
    value = getattr(data, name, None)
    if value is None:
        return None
    try:
        return float(value[0])
    except (TypeError, IndexError):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--chai-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import pyrosetta
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

    pyrosetta.init("-mute all -ignore_unrecognized_res true")
    jobs = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    chai_dir, output = Path(args.chai_dir), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    scorefxn = pyrosetta.get_fa_scorefxn()
    metrics_path = output / "pyrosetta_interface_metrics.json"
    records = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else []
    completed_ids = {item["candidate_id"] for item in records}
    failures = []
    for job in jobs:
        if job["candidate_id"] in completed_ids:
            continue
        candidate_dir = chai_dir / job.get("output_relpath", job["candidate_id"])
        predicted = candidate_dir / "pred.cif"
        if not predicted.exists():
            predicted = candidate_dir / "predicted.cif"
        try:
            pose = pyrosetta.pose_from_file(str(predicted))
            interface = "%s_%s" % ("".join(job["design_chains"]), job["target_chain"])
            mover = InterfaceAnalyzerMover(interface, False, scorefxn, False, False, False)
            mover.apply(pose)
            data = mover.get_all_data()
            record = {
                    "candidate_id": job["candidate_id"],
                    "backbone_id": job["backbone_id"],
                    "interface_definition": interface,
                    "interface_energy": float(mover.get_interface_dG()),
                    "interface_delta_sasa": float(
                        mover.get_interface_delta_sasa()
                    ),
                }
            optional = {
                "shape_complementarity": _optional_metric(data, "sc_value"),
                "interface_hbonds": _optional_metric(data, "interface_hbonds"),
                "buried_unsatisfied_hbonds": _optional_metric(data, "delta_unsat_hbonds"),
                "packstat": _optional_metric(data, "packstat"),
            }
            record.update({key: value for key, value in optional.items() if value is not None})
            records.append(record)
            metrics_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:
            failures.append({"candidate_id": job["candidate_id"], "error": str(exc), "traceback": traceback.format_exc()})
        (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    metrics_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
