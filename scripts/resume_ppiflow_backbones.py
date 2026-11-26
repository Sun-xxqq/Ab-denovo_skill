#!/usr/bin/env python3
"""Resume only missing PPIFlow backbone samples from an interrupted run."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PPIFLOW_PYTHON_ROOT = Path(os.environ.get("AGENT_AB_PPIFLOW_ROOT") or PROJECT_ROOT / "vendor/PPIFlow") / "tools/PPIFlow"
for path in (PROJECT_ROOT, PPIFLOW_PYTHON_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent_ab.backbone_screen import screen_scfv_backbones
from organize_run_outputs import organize


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _existing_names(output_dir):
    return {path.stem for path in Path(output_dir).glob("*.pdb")}


def _write_subset(source_csv, destination, rows, fieldnames):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resume(run_dir, limit=None, screen=False):
    run_dir = Path(run_dir).resolve()
    output_dir = run_dir / "pipeline_output" / "stage1" / "ppiflow_output"
    source_csv = output_dir / "input" / (run_dir.name + "_input.csv")
    config_path = output_dir / "yaml" / ("antibody_sample_config_%s.yml" % run_dir.name)
    status_path = run_dir / "backbone_recovery_status.json"
    if not source_csv.exists() or not config_path.exists():
        raise FileNotFoundError("PPIFlow input CSV or saved inference config is missing")

    with source_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        all_rows = list(reader)
    existing_before = _existing_names(output_dir)
    missing = [row for row in all_rows if row["pdb_name"] not in existing_before]
    if limit is not None:
        missing = missing[: int(limit)]

    value = {
        "state": "running" if missing else "complete",
        "expected_backbones": len(all_rows),
        "existing_before": len(existing_before),
        "scheduled_missing": len(missing),
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    _write_json(status_path, value)

    if missing:
        subset_csv = output_dir / "input" / (run_dir.name + "_missing.csv")
        _write_subset(source_csv, subset_csv, missing, fieldnames)
        try:
            from omegaconf import OmegaConf
            from experiments.inference_antibody import Experiment

            cfg = OmegaConf.load(str(config_path))
            cfg.ppi_dataset.test_csv_path = str(subset_csv)
            cfg.ppi_dataset.samples_per_target = 1
            cfg.experiment.testing_model.save_dir = str(output_dir)
            cfg.experiment.checkpointer.dirpath = str(output_dir)
            experiment = Experiment(cfg=cfg)
            experiment.test()
        except BaseException as exc:
            value.update(
                {
                    "state": "failed",
                    "error": "%s: %s" % (type(exc).__name__, exc),
                    "existing_after": len(_existing_names(output_dir)),
                }
            )
            _write_json(status_path, value)
            raise

    existing_after = _existing_names(output_dir)
    still_missing = [row["pdb_name"] for row in all_rows if row["pdb_name"] not in existing_after]
    value.update(
        {
            "state": "complete" if not still_missing else "partial",
            "existing_after": len(existing_after),
            "still_missing": len(still_missing),
            "still_missing_names": still_missing[:100],
        }
    )
    _write_json(status_path, value)

    if screen and not still_missing:
        spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
        target = spec["target"]
        framework = spec["framework"]
        screen_scfv_backbones(
            input_dir=output_dir,
            output_dir=run_dir / "pipeline_output" / "stage1" / "backbone_screen",
            framework_pdb=framework["structure"],
            hotspots=target["hotspots"],
            heavy_chain=framework["heavy_chain"],
            light_chain=framework["light_chain"],
            cutoff=8.0,
        )
        organize(run_dir)
        value["screening_refreshed"] = True
        _write_json(status_path, value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--screen", action="store_true")
    args = parser.parse_args()
    print(json.dumps(resume(args.run_dir, args.limit, args.screen), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
