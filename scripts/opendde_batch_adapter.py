#!/usr/bin/env python3
"""Run OpenDDE on multiple GPUs and normalize outputs for Agent-Ab."""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _normalize_plddt(value):
    value = float(value)
    return value / 100.0 if value > 1.0 else value


def _residue_plddt(full_data, residue_count):
    atom_plddt = np.asarray(full_data["atom_plddt"], dtype=float).reshape(-1)
    atom_to_token = np.asarray(full_data["atom_to_token_idx"], dtype=int).reshape(-1)
    values = []
    for token_index in range(residue_count):
        selected = atom_plddt[atom_to_token == token_index]
        values.append(float(np.mean(selected)) if selected.size else np.nan)
    return np.asarray(values, dtype=float)


def _prediction_bundle(search_root, prediction_name, need_atom_confidence):
    """Return the newest complete OpenDDE result bundle, if one exists."""
    filename = "%s_summary_confidence_sample_0.json" % prediction_name
    search_root = Path(search_root)
    summaries = list(
        search_root.glob("raw/%s/seed_*/predictions/%s" % (prediction_name, filename))
    )
    summaries.extend(
        search_root.glob(
            ".workers/*/raw/%s/seed_*/predictions/%s"
            % (prediction_name, filename)
        )
    )
    summaries = sorted(
        summaries,
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for summary in summaries:
        prediction = summary.with_name("%s_sample_0.cif" % prediction_name)
        full_data = summary.with_name("%s_full_data_sample_0.json" % prediction_name)
        if not prediction.is_file():
            continue
        if need_atom_confidence and not full_data.is_file():
            continue
        return summary, prediction, full_data
    return None


def _public_paths(job, public_root):
    """Map one FlowPacker candidate into its public backbone directory."""
    source = Path(job["pdb"])
    backbone_folder = source.parent.name
    design_stem = source.stem
    if design_stem.startswith("backbone"):
        suffix = design_stem[len("backbone"):]
        prediction_stem = "pred" + suffix
        public_design_stem = "design" + suffix
    else:
        prediction_stem = "pred." + job.get("prediction_name", job["candidate_id"])
        public_design_stem = "design." + job.get(
            "prediction_name", job["candidate_id"]
        )
    public_root = Path(public_root)
    group = public_root / backbone_folder
    return (
        group,
        group / (prediction_stem + ".cif"),
        group / (public_design_stem + ".pdb"),
    )


def _public_metric_record(job, metrics, prediction_path, design_path):
    return {
        "candidate_id": job["candidate_id"],
        "prediction_id": prediction_path.stem,
        "design_id": design_path.stem,
        "prediction_cif": prediction_path.name,
        "design_pdb": design_path.name,
        "plddt": metrics.get("plddt"),
        "antibody_plddt": metrics.get("antibody_plddt"),
        "ptm": metrics.get("ptm"),
        "iptm": metrics.get("iptm"),
        "aggregate_score": metrics.get("aggregate_score"),
        "has_inter_chain_clashes": metrics.get("has_inter_chain_clashes"),
    }


def _publish_candidate(job, public_root, source_prediction, metrics):
    group, prediction_path, design_path = _public_paths(job, public_root)
    group.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_prediction, prediction_path)
    shutil.copy2(job["pdb"], design_path)

    metrics_path = group / "metrics.json"
    value = {"backbone_id": group.name, "predictions": []}
    if metrics_path.is_file():
        try:
            current = json.loads(metrics_path.read_text(encoding="utf-8"))
            if isinstance(current, dict) and isinstance(current.get("predictions"), list):
                value = current
        except (OSError, json.JSONDecodeError):
            pass
    record = _public_metric_record(job, metrics, prediction_path, design_path)
    value["predictions"] = [
        item
        for item in value["predictions"]
        if item.get("candidate_id") != job["candidate_id"]
    ]
    value["predictions"].append(record)
    value["predictions"].sort(key=lambda item: item["prediction_id"])
    _write_json(metrics_path, value)
    return prediction_path


def _required_outputs(job, output_root, need_atom_confidence):
    candidate_dir = output_root / job.get("output_relpath", job["candidate_id"])
    required = [candidate_dir / "metrics.json", candidate_dir / "pred.cif"]
    if need_atom_confidence:
        required.extend(
            [candidate_dir / "pae_candidate.npz", candidate_dir / "plddt_candidate.npz"]
        )
    return candidate_dir, required


def _remove_raw_bundle(summary_path):
    """Delete only one completed raw candidate directory after normalization."""
    candidate_root = summary_path.parents[2]
    if candidate_root.is_dir():
        shutil.rmtree(candidate_root)


def _normalize(job, search_root, output_root, public_root, target_chain, design_chains,
               need_atom_confidence):
    candidate_id = job["candidate_id"]
    prediction_name = job.get("prediction_name", candidate_id)
    bundle = _prediction_bundle(search_root, prediction_name, need_atom_confidence)
    if bundle is None:
        raise FileNotFoundError("incomplete OpenDDE result for %s" % candidate_id)
    summary_path, prediction, full_path = bundle
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    candidate_dir = output_root / job.get("output_relpath", candidate_id)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(prediction, candidate_dir / "pred.cif")
    _, published, _ = _public_paths(job, public_root)
    if not (candidate_dir / "design.pdb").exists():
        shutil.copy2(job["pdb"], candidate_dir / "design.pdb")

    chains = [item["chain"] for item in job["sequences"]]
    target = chains.index(job.get("target_chain", target_chain))
    requested_design = job.get("design_chains") or design_chains
    design_indices = [chains.index(chain) for chain in requested_design if chain in chains]
    if not design_indices:
        design_indices = [index for index in range(len(chains)) if index != target]

    chain_plddt = np.asarray(summary.get("chain_plddt", []), dtype=float)
    selected_plddt = [
        _normalize_plddt(chain_plddt[index])
        for index in design_indices
        if index < chain_plddt.size
    ]
    pair_iptm = np.asarray(summary.get("chain_pair_iptm", []), dtype=float)
    confidence = (
        {
            "pair_chains_iptm": {
                str(i): {
                    str(j): float(pair_iptm[i, j])
                    for j in range(pair_iptm.shape[1])
                }
                for i in range(pair_iptm.shape[0])
            }
        }
        if pair_iptm.ndim == 2
        else {"pair_chains_iptm": {}}
    )
    _write_json(candidate_dir / "confidence_candidate.json", confidence)

    if need_atom_confidence:
        full_data = json.loads(full_path.read_text(encoding="utf-8"))
        pae = np.asarray(full_data["token_pair_pae"], dtype=float)
        plddt = _residue_plddt(
            full_data, sum(len(item["sequence"]) for item in job["sequences"])
        )
        finite_plddt = plddt[np.isfinite(plddt)]
        if finite_plddt.size and np.max(finite_plddt) > 1.0:
            plddt = plddt / 100.0
        np.savez(candidate_dir / "pae_candidate.npz", pae=pae)
        np.savez(candidate_dir / "plddt_candidate.npz", plddt=plddt)

    metrics = {
        "candidate_id": candidate_id,
        "prediction_name": prediction_name,
        "backbone_id": job.get("backbone_id"),
        "design_id": job.get("design_id"),
        "source_pdb": str(candidate_dir / "design.pdb"),
        "original_source_pdb": job["pdb"],
        "predicted_cif": str(candidate_dir / "pred.cif"),
        "published_cif": str(published),
        "plddt": _normalize_plddt(summary["plddt"]),
        "antibody_plddt": float(np.mean(selected_plddt)) if selected_plddt else None,
        "ptm": float(summary["ptm"]),
        "iptm": float(summary["iptm"]),
        "aggregate_score": float(summary["ranking_score"]),
        "has_inter_chain_clashes": bool(summary["has_clash"]),
        "opendde_summary": str(summary_path),
    }
    _write_json(candidate_dir / "metrics.json", metrics)
    _publish_candidate(job, public_root, prediction, metrics)
    _remove_raw_bundle(summary_path)
    return metrics


def _opendde_job(job, seed):
    return {
        "name": job.get("prediction_name", job["candidate_id"]),
        "modelSeeds": [seed],
        "sequences": [
            {
                "proteinChain": {
                    "sequence": item["sequence"],
                    "count": 1,
                    "id": [item["chain"]],
                }
            }
            for item in job["sequences"]
        ],
    }


def _available_gpu_ids():
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True)
        records = []
        for line in output.splitlines():
            index, used = [part.strip() for part in line.split(",", 1)]
            records.append((int(used), int(index)))
        return [index for _, index in sorted(records)]
    except (OSError, subprocess.SubprocessError, ValueError):
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        return [int(value) for value in visible.split(",") if value.strip().isdigit()]


def _select_gpu_ids(parallel_gpus, requested_ids):
    if requested_ids:
        gpu_ids = [int(value) for value in requested_ids.split(",") if value.strip()]
    else:
        gpu_ids = _available_gpu_ids()
    if not gpu_ids:
        raise RuntimeError("no NVIDIA GPU was detected for OpenDDE")
    return gpu_ids[:max(1, min(parallel_gpus, len(gpu_ids)))]


def _build_command(args, input_path, raw_dir):
    return [
        args.opendde_bin,
        "pred",
        "-i", str(input_path),
        "-o", str(raw_dir),
        "-s", str(args.seed),
        "-n", args.model,
        "--use_msa", "false",
        "--use_template", "false",
        "--use_rna_msa", "false",
        "--sample", str(args.sample),
        "--step", str(args.step),
        "--cycle", str(args.cycle),
        "--load_checkpoint_path", args.checkpoint,
        "--need_atom_confidence", str(bool(args.need_atom_confidence)).lower(),
    ]


def _recover_completed(jobs, search_root, output, public_root, target_chain, design_chains,
                       need_atom_confidence):
    completed, pending = [], []
    for job in jobs:
        candidate_dir, required = _required_outputs(job, output, need_atom_confidence)
        if all(path.is_file() for path in required):
            metrics = json.loads(
                (candidate_dir / "metrics.json").read_text(encoding="utf-8")
            )
            _publish_candidate(job, public_root, candidate_dir / "pred.cif", metrics)
            completed.append(metrics)
            continue
        try:
            completed.append(
                _normalize(
                    job,
                    search_root,
                    output,
                    public_root,
                    target_chain,
                    design_chains,
                    need_atom_confidence,
                )
            )
        except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError, OSError):
            pending.append(job)
    return completed, pending


def _remove_empty_work_dirs(output):
    for path in (output / "raw", output / ".workers"):
        if path.exists():
            try:
                shutil.rmtree(path)
            except OSError:
                pass
    input_path = output / "opendde_input.json"
    if input_path.exists():
        input_path.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--publish-root", default="")
    parser.add_argument("--opendde-bin", required=True)
    parser.add_argument("--opendde-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--sample", type=int, default=1)
    parser.add_argument("--step", type=int, default=200)
    parser.add_argument("--cycle", type=int, default=10)
    parser.add_argument("--parallel-gpus", type=int, default=1)
    parser.add_argument("--gpu-ids", default="")
    parser.add_argument("--target-chain", required=True)
    parser.add_argument("--design-chains", default="")
    parser.add_argument("--need-atom-confidence", action="store_true")
    parser.add_argument("--candidate-id", help="Run one manifest candidate for recovery")
    args = parser.parse_args()

    if not Path(args.checkpoint).is_file():
        raise SystemExit("OpenDDE checkpoint does not exist: %s" % args.checkpoint)
    for required_path in (args.opendde_bin, args.opendde_root):
        if not Path(required_path).exists():
            raise SystemExit("OpenDDE runtime path does not exist: %s" % required_path)

    jobs = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if args.candidate_id:
        jobs = [job for job in jobs if job.get("candidate_id") == args.candidate_id]
        if not jobs:
            raise SystemExit("candidate is not present in manifest: %s" % args.candidate_id)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    public_root = (
        Path(args.publish_root)
        if args.publish_root
        else (output.parent if output.name == "predictions" else output)
    )
    public_root.mkdir(parents=True, exist_ok=True)
    design_chains = [value for value in args.design_chains.split(",") if value]

    completed, pending = _recover_completed(
        jobs,
        output,
        output,
        public_root,
        args.target_chain,
        design_chains,
        args.need_atom_confidence,
    )
    print(
        "OpenDDE resume: recovered %d/%d candidates; %d pending"
        % (len(completed), len(jobs), len(pending)),
        flush=True,
    )

    processes = []
    log_handles = []
    if pending:
        gpu_ids = _select_gpu_ids(args.parallel_gpus, args.gpu_ids)
        shards = [pending[index::len(gpu_ids)] for index in range(len(gpu_ids))]
        print("OpenDDE GPU workers: %s" % ",".join(map(str, gpu_ids)), flush=True)
        workers_root = output / ".workers"
        workers_root.mkdir(parents=True, exist_ok=True)
        for gpu_id, shard in zip(gpu_ids, shards):
            if not shard:
                continue
            worker = workers_root / ("gpu_%s" % gpu_id)
            raw_dir = worker / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            input_path = worker / "opendde_input.json"
            _write_json(input_path, [_opendde_job(job, args.seed) for job in shard])
            log_handle = (worker / "opendde.log").open("a", encoding="utf-8")
            environment = dict(os.environ)
            environment["OPENDDE_ROOT_DIR"] = args.opendde_root
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            process = subprocess.Popen(
                _build_command(args, input_path, raw_dir),
                cwd=str(worker),
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((gpu_id, process))
            log_handles.append(log_handle)
            print(
                "started GPU %s worker pid=%s candidates=%s"
                % (gpu_id, process.pid, len(shard)),
                flush=True,
            )

        remaining = list(pending)
        try:
            while any(process.poll() is None for _, process in processes):
                recovered, remaining = _recover_completed(
                    remaining,
                    output,
                    output,
                    public_root,
                    args.target_chain,
                    design_chains,
                    args.need_atom_confidence,
                )
                if recovered:
                    completed.extend(recovered)
                    print(
                        "OpenDDE progress: %d/%d complete"
                        % (len(completed), len(jobs)),
                        flush=True,
                    )
                time.sleep(3)
        finally:
            for handle in log_handles:
                handle.close()

        recovered, remaining = _recover_completed(
            remaining,
            output,
            output,
            public_root,
            args.target_chain,
            design_chains,
            args.need_atom_confidence,
        )
        completed.extend(recovered)
        failed_workers = [
            {"gpu": gpu_id, "returncode": process.returncode}
            for gpu_id, process in processes
            if process.returncode != 0
        ]
    else:
        remaining = []
        failed_workers = []

    failures = [
        {"candidate_id": job.get("candidate_id"), "error": "OpenDDE output incomplete"}
        for job in remaining
    ]
    if failed_workers:
        failures.append({"workers": failed_workers, "error": "OpenDDE worker failed"})

    completed_by_id = {item["candidate_id"]: item for item in completed}
    ordered_completed = [
        completed_by_id[job["candidate_id"]]
        for job in jobs
        if job["candidate_id"] in completed_by_id
    ]
    _write_json(output / "opendde_metrics.json", ordered_completed)
    _write_json(output / "failures.json", failures)
    if failures:
        raise SystemExit(1)
    _remove_empty_work_dirs(output)
    print("OpenDDE complete: %d/%d" % (len(ordered_completed), len(jobs)), flush=True)


if __name__ == "__main__":
    main()
