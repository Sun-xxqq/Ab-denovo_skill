#!/usr/bin/env python3
"""Launch PPIFlow antibody sampling with Agent-Ab's minimal QC module."""

import importlib
import importlib.util
import os
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PPI_FLOW_TOOL_ROOT = Path(os.environ.get("AGENT_AB_PPIFLOW_ROOT") or PROJECT_ROOT / "vendor/PPIFlow") / "tools/PPIFlow"
OVERRIDE_ROOT = PROJECT_ROOT / "agent_ab" / "ppiflow_overrides"


def _load_override(module_name, source_path):
    source_path = Path(source_path)
    if not source_path.is_file():
        raise FileNotFoundError(
            "PPIFlow override not found for %s: %s" % (module_name, source_path)
        )
    spec = importlib.util.spec_from_file_location(module_name, str(source_path))
    if spec is None or spec.loader is None:
        raise ImportError("unable to load PPIFlow override: %s" % source_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resume_shard(directory, count):
    import csv
    import json
    import tempfile
    import hashlib
    from omegaconf import OmegaConf
    from analysis.antibody_metric import filter_pdb_by_bfactor, calc_rmsd, parse_pdb_and_get_cdr
    from analysis import pdb_clash
    from models.flow_module_antibody import _count_binder_chain_breaks
    from experiments.inference_antibody import Experiment
    root = Path(directory).resolve()
    cfg = OmegaConf.load(root / "yaml/antibody_sample_config_backbone.0.yml")
    with (root / "input/backbone.0_input.csv").open() as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    selected = [row for row in rows if int(row["pdb_name"].rsplit("_", 1)[1]) < count]
    expected = {"backbone.0_%s" % i for i in range(count)}
    if {row["pdb_name"] for row in selected} != expected or len(selected) != count:
        raise ValueError("Saved input metadata does not cover requested sample IDs")
    saved = []; pending = []; hashes = {}; rejected = []
    heavy = cfg.experiment.testing_model.heavy_chain
    light = cfg.experiment.testing_model.light_chain
    for row in selected:
        pdb = root / (row["pdb_name"] + ".pdb")
        if not pdb.exists():
            pending.append(row); continue
        with tempfile.TemporaryDirectory() as temporary:
            filtered = Path(temporary) / "framework.pdb"
            filter_pdb_by_bfactor(str(pdb), str(filtered))
            rmsd = calc_rmsd(str(filtered), cfg.experiment.framework_pdb)
        clash = pdb_clash.detect_backbone_clash(str(pdb))
        breaks = _count_binder_chain_breaks(str(pdb), heavy, light)
        if not (rmsd < 1 and not clash and breaks == 0):
            quarantine = root / "interrupted_unaccepted"
            quarantine.mkdir(exist_ok=True)
            destination = quarantine / pdb.name
            if destination.exists():
                raise FileExistsError(destination)
            pdb.rename(destination)
            rejected.append(row["pdb_name"]); pending.append(row); continue
        record = {"sample": pdb.name, "pdb_path": str(pdb)}
        record.update(parse_pdb_and_get_cdr(str(pdb), heavy, light))
        record.update(rmsd_framework=rmsd, has_clash=clash, peptide_chain_breaks=breaks)
        saved.append(record)
        hashes[str(pdb)] = hashlib.sha256(pdb.read_bytes()).hexdigest()
    subset = root / "input/backbone.0_resume_missing.csv"
    with subset.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(pending)
    state = {"target": count, "reused": len(saved), "pending": len(pending), "quarantined": rejected, "status": "running"}
    (root / "resume_progress.json").write_text(json.dumps(state, indent=2) + "\n")
    print("RESUME: target=%s reused=%s pending=%s quarantined=%s" % (count, len(saved), len(pending), len(rejected)), flush=True)
    fresh = []
    if pending:
        cfg.ppi_dataset.test_csv_path = str(subset)
        cfg.ppi_dataset.samples_per_target = 1
        cfg.experiment.testing_model.save_dir = str(root)
        cfg.experiment.checkpointer.dirpath = str(root)
        Experiment(cfg=cfg).test()
        with (root / "sample_metrics.csv").open() as handle:
            fresh = list(csv.DictReader(handle))
    combined = saved + fresh
    names = [Path(row["sample"]).stem for row in combined]
    if len(names) != count or set(names) != expected:
        raise RuntimeError("Resumed generation incomplete; outputs retained")
    for name, digest in hashes.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Previously completed backbone changed: " + name)
    fields = list(combined[0])
    with (root / "sample_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows(sorted(combined, key=lambda row: int(Path(row["sample"]).stem.rsplit("_", 1)[1])))
    state.update(status="complete", completed=count)
    (root / "resume_progress.json").write_text(json.dumps(state, indent=2) + "\n")


def main():
    for path in (PPI_FLOW_TOOL_ROOT, PROJECT_ROOT):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)

    importlib.import_module("analysis")
    _load_override(
        "analysis.antibody_metric",
        OVERRIDE_ROOT / "antibody_metric.py",
    )
    importlib.import_module("models")
    _load_override(
        "models.flow_module_antibody",
        OVERRIDE_ROOT / "flow_module_antibody.py",
    )

    if "--resume-shard" in sys.argv:
        index = sys.argv.index("--resume-shard")
        directory = sys.argv[index + 1]
        count = int(sys.argv[sys.argv.index("--resume-count") + 1])
        _resume_shard(directory, count)
        return

    runpy.run_path(
        str(PPI_FLOW_TOOL_ROOT / "sample_antibody_nanobody.py"),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
