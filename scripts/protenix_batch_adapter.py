#!/usr/bin/env python3
"""Run Protenix in batches and normalize outputs for Agent-Ab evaluators."""

import argparse
import json
import os
import re
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


def _public_paths(job, public_root):
    source = Path(job["pdb"])
    backbone_folder = job.get("public_group") or source.parent.name
    design_stem = source.stem
    if job.get("public_stem"):
        prediction_stem = "pred." + str(job["public_stem"])
        public_design_stem = "design." + str(job["public_stem"])
    elif design_stem.startswith(("backbone", "flowpacker")):
        source_prefix = (
            "flowpacker" if design_stem.startswith("flowpacker") else "backbone"
        )
        suffix = design_stem[len(source_prefix):]
        prediction_stem = "pred" + suffix
        public_design_stem = "design" + suffix
    else:
        prediction_stem = "pred." + job.get("prediction_name", job["candidate_id"])
        public_design_stem = "design." + job.get(
            "prediction_name", job["candidate_id"]
        )
    public_root = Path(public_root)
    group = public_root / backbone_folder
    return group, group / (prediction_stem + ".cif"), group / (public_design_stem + ".pdb")


def _publish_candidate(job, public_root, source_prediction, metrics):
    group, prediction_path, design_path = _public_paths(job, public_root)
    group.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_prediction, prediction_path)
    publish_design = job.get("publish_design", True)
    if publish_design:
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
    record = {
        "candidate_id": job["candidate_id"],
        "prediction_id": prediction_path.stem,
        "design_id": design_path.stem,
        "prediction_cif": prediction_path.name,
        "design_pdb": design_path.name if publish_design else None,
        "plddt": metrics.get("plddt"),
        "antibody_plddt": metrics.get("antibody_plddt"),
        "ptm": metrics.get("ptm"),
        "iptm": metrics.get("iptm"),
        "aggregate_score": metrics.get("aggregate_score"),
        "has_inter_chain_clashes": metrics.get("has_inter_chain_clashes"),
        "interface_pae_mean": metrics.get("interface_pae_mean"),
        "interface_pae_p90": metrics.get("interface_pae_p90"),
    }
    value["predictions"] = [
        item
        for item in value["predictions"]
        if item.get("candidate_id") != job["candidate_id"]
    ]
    value["predictions"].append(record)
    value["predictions"].sort(key=lambda item: item["prediction_id"])
    _write_json(metrics_path, value)


def _matrix_values(matrix, chain_indices, target_index):
    array = np.asarray(matrix, dtype=float)
    values = []
    for index in chain_indices:
        if index == target_index:
            continue
        if index < array.shape[0] and target_index < array.shape[1]:
            values.append(float(array[index, target_index]))
        if target_index < array.shape[0] and index < array.shape[1]:
            values.append(float(array[target_index, index]))
    return values


def _residue_plddt(full_data, residue_count):
    atom_plddt = np.asarray(full_data["atom_plddt"], dtype=float).reshape(-1)
    atom_to_token = np.asarray(full_data["atom_to_token_idx"], dtype=int).reshape(-1)
    values = []
    for token_index in range(residue_count):
        selected = atom_plddt[atom_to_token == token_index]
        values.append(float(np.mean(selected)) if selected.size else np.nan)
    return np.asarray(values, dtype=float)


def _prediction_file(raw_dir, candidate_id, kind):
    pattern = "%s_%s_sample_0.json" % (candidate_id, kind)
    # Protenix writes <raw>/<name>/seed_*/predictions/*.  Restrict the search
    # to that candidate so recovery of thousands of jobs is linear instead of
    # recursively walking the entire raw tree once per candidate.
    matches = sorted(
        (Path(raw_dir) / candidate_id).glob("seed_*/predictions/%s" % pattern)
    )
    if len(matches) != 1:
        raise RuntimeError("expected one %s for %s, found %s" % (kind, candidate_id, len(matches)))
    return matches[0]


def _normalize(job, raw_dir, output_root, public_root, target_chain, design_chains,
               need_atom_confidence):
    candidate_id = job["candidate_id"]
    prediction_name = job.get("prediction_name", candidate_id)
    candidate_dir = output_root / job.get("output_relpath", candidate_id)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    summary_path = _prediction_file(raw_dir, prediction_name, "summary_confidence")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    prediction = summary_path.with_name("%s_sample_0.cif" % prediction_name)
    if not prediction.exists():
        raise RuntimeError("missing Protenix structure for %s" % candidate_id)
    shutil.copy2(prediction, candidate_dir / "pred.cif")
    if job.get("copy_design", True) and not (candidate_dir / "design.pdb").exists():
        shutil.copy2(job["pdb"], candidate_dir / "design.pdb")

    chains = [item["chain"] for item in job["sequences"]]
    target = chains.index(job.get("target_chain", target_chain))
    requested_design = job.get("design_chains") or design_chains
    design_indices = [chains.index(chain) for chain in requested_design if chain in chains]
    if not design_indices:
        design_indices = [index for index in range(len(chains)) if index != target]

    chain_plddt = np.asarray(summary.get("chain_plddt", []), dtype=float)
    selected_plddt = [chain_plddt[index] for index in design_indices if index < chain_plddt.size]
    pae_values = _matrix_values(summary.get("chain_pair_pae_mean", []), design_indices, target)
    pair_iptm = np.asarray(summary.get("chain_pair_iptm", []), dtype=float)
    confidence = {
        "pair_chains_iptm": {
            str(i): {str(j): float(pair_iptm[i, j]) for j in range(pair_iptm.shape[1])}
            for i in range(pair_iptm.shape[0])
        }
    } if pair_iptm.ndim == 2 else {"pair_chains_iptm": {}}
    _write_json(candidate_dir / "confidence_candidate.json", confidence)

    if need_atom_confidence:
        full_path = _prediction_file(raw_dir, prediction_name, "full_data")
        full_data = json.loads(full_path.read_text(encoding="utf-8"))
        pae = np.asarray(full_data["token_pair_pae"], dtype=float)
        plddt = _residue_plddt(
            full_data, sum(len(item["sequence"]) for item in job["sequences"])
        )
        np.savez(candidate_dir / "pae_candidate.npz", pae=pae)
        np.savez(candidate_dir / "plddt_candidate.npz", plddt=plddt)

    metrics = {
        "candidate_id": candidate_id,
        "prediction_name": prediction_name,
        "backbone_id": job.get("backbone_id"),
        "design_id": job.get("design_id"),
        "source_pdb": str(
            candidate_dir / "design.pdb"
            if (candidate_dir / "design.pdb").exists()
            else job["pdb"]
        ),
        "original_source_pdb": job["pdb"],
        "predicted_cif": str(candidate_dir / "pred.cif"),
        "plddt": float(summary["plddt"]) / 100.0,
        "antibody_plddt": float(np.mean(selected_plddt)) if selected_plddt else None,
        "ptm": float(summary["ptm"]),
        "iptm": float(summary["iptm"]),
        "aggregate_score": float(summary["ranking_score"]),
        "has_inter_chain_clashes": bool(summary["has_clash"]),
        "interface_pae_mean": float(np.mean(pae_values)) if pae_values else None,
        "interface_pae_p90": float(np.percentile(pae_values, 90)) if pae_values else None,
        "protenix_summary": str(summary_path),
    }
    _write_json(candidate_dir / "metrics.json", metrics)
    _publish_candidate(job, public_root, prediction, metrics)
    return metrics


def _idle_gpu_inventory(maximum=None):
    """Inspect every physical card and return the currently idle subset."""
    gpu_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if gpu_query.returncode:
        raise RuntimeError("cannot query GPUs with nvidia-smi")
    process_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if process_query.returncode:
        raise RuntimeError("cannot query GPU compute processes with nvidia-smi")
    processes = {}
    for line in process_query.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) >= 2 and fields[0].startswith("GPU-"):
            processes.setdefault(fields[0], []).append(int(fields[1]))

    memory_limit = int(os.environ.get("AGENT_AB_PROTENIX_IDLE_MEMORY_MIB", "1024"))
    utilization_limit = int(
        os.environ.get("AGENT_AB_PROTENIX_IDLE_UTILIZATION", "10")
    )
    maximum = int(
        maximum
        if maximum is not None
        else os.environ.get("AGENT_AB_PROTENIX_MAX_GPUS", "4")
    )
    maximum = max(1, maximum)
    inventory = []
    for line in gpu_query.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 4:
            continue
        index, uuid, memory, utilization = fields
        record = {
            "index": int(index),
            "uuid": uuid,
            "memory_used_mib": int(memory),
            "utilization_percent": int(utilization),
            "compute_pids": processes.get(uuid, []),
        }
        record["idle"] = (
            not record["compute_pids"]
            and record["memory_used_mib"] <= memory_limit
            and record["utilization_percent"] <= utilization_limit
        )
        inventory.append(record)
    idle = sorted(
        (item for item in inventory if item["idle"]),
        key=lambda item: (
            item["memory_used_mib"],
            item["utilization_percent"],
            item["index"],
        ),
    )[:maximum]
    if not idle:
        raise RuntimeError(
            "no idle GPU is available for Protenix; any compute process, "
            "more than %d MiB memory, or more than %d%% utilization marks a GPU busy"
            % (memory_limit, utilization_limit)
        )
    return inventory, idle


def _balanced_job_shards(jobs, gpu_records):
    active = list(gpu_records[: max(1, min(len(gpu_records), len(jobs)))])
    shards = [
        {"index": index, "gpu": gpu["index"], "jobs": []}
        for index, gpu in enumerate(active)
    ]
    for index, job in enumerate(jobs):
        shards[index % len(shards)]["jobs"].append(job)
    return shards


def _run_root_for_output(output):
    output = Path(output).resolve()
    for candidate in (output,) + tuple(output.parents):
        if (candidate / "spec.json").is_file():
            return candidate
    return output.parent


def _protenix_jobs(jobs):
    return [
        {
            "name": job.get("prediction_name", job["candidate_id"]),
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
        for job in jobs
    ]


def _raw_prediction_complete(raw_dir, job, need_atom_confidence):
    prediction_name = job.get("prediction_name", job["candidate_id"])
    prediction_root = Path(raw_dir) / prediction_name
    summaries = sorted(
        prediction_root.glob(
            "seed_*/predictions/%s_summary_confidence_sample_0.json"
            % prediction_name
        )
    )
    if len(summaries) != 1:
        return False
    if not summaries[0].with_name("%s_sample_0.cif" % prediction_name).is_file():
        return False
    if need_atom_confidence:
        full = sorted(
            prediction_root.glob(
                "seed_*/predictions/%s_full_data_sample_0.json"
                % prediction_name
            )
        )
        if len(full) != 1:
            return False
    return True


def _candidate_raw_root(output, job, need_atom_confidence):
    output = Path(output)
    roots = [output / "raw"] + sorted(
        output.glob("protenix_shards/shard_*/raw")
    )
    for root in roots:
        if root.is_dir() and _raw_prediction_complete(
            root, job, need_atom_confidence
        ):
            return root
    return None


def _seed_shared_embedding_cache(output, shard_root):
    """Reuse the global ESM cache while keeping per-process metadata isolated."""
    source = Path(output) / "esm_embeddings"
    destination = Path(shard_root) / "esm_embeddings"
    destination.mkdir(parents=True, exist_ok=True)
    if not source.is_dir():
        return []
    linked = []
    for model_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        # Error reports are mutable process output, not reusable model caches.
        # Keep them isolated so parallel workers cannot race on one directory.
        if model_dir.name == "esm_error":
            continue
        if not re.fullmatch(r"esm[^/]*", model_dir.name):
            continue
        target = destination / model_dir.name
        if not target.exists() and not target.is_symlink():
            target.symlink_to(model_dir.resolve(), target_is_directory=True)
        linked.append(model_dir.name)
    return linked


def _parallel_predict(args, jobs, output):
    """Run one isolated Protenix CLI process per currently idle physical GPU."""
    inventory, idle = _idle_gpu_inventory(args.max_gpus)
    shards = _balanced_job_shards(jobs, idle)
    output = Path(output)
    work_root = output / "protenix_shards"
    work_root.mkdir(parents=True, exist_ok=True)
    run_root = _run_root_for_output(output)
    stage_name = output.parents[1].name if len(output.parents) > 1 else "stage"
    logs_root = run_root / "logs" / "protenix_shards"
    logs_root.mkdir(parents=True, exist_ok=True)
    progress_path = output / "parallel_protenix_progress.json"
    processes = []
    handles = []
    records = []
    failure = None
    try:
        for shard in shards:
            shard_root = work_root / (
                "shard_%02d_gpu_%s" % (shard["index"], shard["gpu"])
            )
            shard_root.mkdir(parents=True, exist_ok=True)
            raw_dir = shard_root / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            input_path = shard_root / "protenix_input.json"
            _write_json(input_path, _protenix_jobs(shard["jobs"]))
            cache_links = _seed_shared_embedding_cache(output, shard_root)
            command = [
                args.protenix_bin,
                "pred",
                "-i",
                str(input_path),
                "-o",
                str(raw_dir),
                "-s",
                str(args.seed),
                "-n",
                args.model,
                "-e",
                "1",
                "--use_default_params",
                "true",
                "--need_atom_confidence",
                str(bool(args.need_atom_confidence)).lower(),
            ]
            log_path = logs_root / (
                "protenix.%s.shard_%02d.gpu_%s.log"
                % (stage_name, shard["index"], shard["gpu"])
            )
            log_handle = log_path.open("a", encoding="utf-8")
            log_handle.write("argv=" + json.dumps(command, ensure_ascii=False) + "\n")
            log_handle.flush()
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(shard["gpu"])
            process = subprocess.Popen(
                command,
                cwd=str(shard_root),
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            record = {
                "index": shard["index"],
                "gpu": shard["gpu"],
                "candidate_count": len(shard["jobs"]),
                "log": str(log_path),
                "raw_dir": str(raw_dir),
                "shared_embedding_caches": cache_links,
            }
            processes.append((shard, process, raw_dir))
            handles.append(log_handle)
            records.append(record)

        print(
            "[ParallelProtenix] pending=%d GPUs=%s shards=%s"
            % (
                len(jobs),
                ",".join(str(item["gpu"]) for item in records),
                ",".join(str(item["candidate_count"]) for item in records),
            ),
            flush=True,
        )
        while True:
            running = False
            failed = None
            completed_now = 0
            for shard, process, raw_dir in processes:
                returncode = process.poll()
                if returncode is None:
                    running = True
                elif returncode != 0 and failed is None:
                    failed = (shard, returncode)
                completed_now += sum(
                    _raw_prediction_complete(
                        raw_dir, job, args.need_atom_confidence
                    )
                    for job in shard["jobs"]
                )
            _write_json(
                progress_path,
                {
                    "status": "running" if running and not failed else "normalizing",
                    "total_candidates": len(jobs),
                    "completed_in_shards": completed_now,
                    "selected_gpus": [item["gpu"] for item in records],
                    "gpu_inventory": inventory,
                    "shards": records,
                },
            )
            if failed:
                shard, returncode = failed
                failure = "Protenix shard %s on GPU %s failed with exit code %s" % (
                    shard["index"],
                    shard["gpu"],
                    returncode,
                )
                break
            if not running:
                break
            time.sleep(5)
    except BaseException:
        for _, process, _ in processes:
            if process.poll() is None:
                process.terminate()
        for _, process, _ in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        raise
    finally:
        if failure:
            for _, process, _ in processes:
                if process.poll() is None:
                    process.terminate()
            for _, process, _ in processes:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
        for handle in handles:
            handle.close()
    return records, failure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--publish-root", default="")
    parser.add_argument("--protenix-bin", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--target-chain", required=True)
    parser.add_argument("--design-chains", default="")
    parser.add_argument("--need-atom-confidence", action="store_true")
    parser.add_argument("--allow-partial-failures", action="store_true")
    parser.add_argument("--candidate-id", help="Run one manifest candidate for recovery or validation")
    parser.add_argument(
        "--max-gpus",
        type=int,
        default=None,
        help="Maximum idle GPUs to use (default: AGENT_AB_PROTENIX_MAX_GPUS or 4)",
    )
    args = parser.parse_args()

    if not Path(args.checkpoint).is_file():
        raise SystemExit("Protenix checkpoint does not exist: %s" % args.checkpoint)
    jobs = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if args.candidate_id:
        jobs = [job for job in jobs if job.get("candidate_id") == args.candidate_id]
        if not jobs:
            raise SystemExit("candidate is not present in manifest: %s" % args.candidate_id)
    output = Path(args.output)
    raw_dir = output / "raw"
    output.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    public_root = (
        Path(args.publish_root)
        if args.publish_root
        else (output.parent if output.name == "predictions" else output)
    )
    public_root.mkdir(parents=True, exist_ok=True)
    design_chains = [value for value in args.design_chains.split(",") if value]

    completed_by_id = {}
    failures = []
    pending = []
    for job in jobs:
        candidate_dir = output / job.get("output_relpath", job["candidate_id"])
        required = [candidate_dir / "metrics.json", candidate_dir / "pred.cif"]
        if args.need_atom_confidence:
            required.extend([candidate_dir / "pae_candidate.npz", candidate_dir / "plddt_candidate.npz"])
        if all(path.exists() for path in required):
            metrics = json.loads(
                (candidate_dir / "metrics.json").read_text(encoding="utf-8")
            )
            _publish_candidate(job, public_root, candidate_dir / "pred.cif", metrics)
            completed_by_id[job["candidate_id"]] = metrics
        else:
            pending.append(job)

    recovered = 0
    still_pending = []
    for job in pending:
        raw_root = _candidate_raw_root(output, job, args.need_atom_confidence)
        if raw_root is None:
            still_pending.append(job)
            continue
        try:
            metrics = _normalize(
                job,
                raw_root,
                output,
                public_root,
                args.target_chain,
                design_chains,
                args.need_atom_confidence,
            )
            completed_by_id[job["candidate_id"]] = metrics
            recovered += 1
        except Exception as exc:
            still_pending.append(job)
    pending = still_pending

    records = []
    parallel_failure = None
    if pending:
        records, parallel_failure = _parallel_predict(args, pending, output)
        for job in pending:
            if job["candidate_id"] in completed_by_id:
                continue
            raw_root = _candidate_raw_root(
                output, job, args.need_atom_confidence
            )
            if raw_root is None:
                failures.append(
                    {
                        "candidate_id": job.get("candidate_id"),
                        "error": "Protenix prediction is incomplete after parallel run",
                    }
                )
                continue
            try:
                completed_by_id[job["candidate_id"]] = _normalize(
                    job,
                    raw_root,
                    output,
                    public_root,
                    args.target_chain,
                    design_chains,
                    args.need_atom_confidence,
                )
            except Exception as exc:
                failures.append(
                    {"candidate_id": job.get("candidate_id"), "error": str(exc)}
                )
    if parallel_failure:
        failures.append({"candidate_id": "__parallel__", "error": parallel_failure})

    completed = [
        completed_by_id[job["candidate_id"]]
        for job in jobs
        if job["candidate_id"] in completed_by_id
    ]
    _write_json(output / "protenix_metrics.json", completed)
    _write_json(output / "failures.json", failures)
    manifest = {
        "mode": "parallel_protenix_idle_gpu_shards",
        "status": "complete" if not failures else "partial_failure",
        "total_candidates": len(jobs),
        "completed_before": len(jobs) - len(pending) - recovered,
        "recovered_from_raw": recovered,
        "scheduled": len(pending),
        "completed": len(completed),
        "failed": len(failures),
        "selected_gpus": [item["gpu"] for item in records],
        "shards": records,
    }
    _write_json(output / "parallel_protenix_manifest.json", manifest)
    _write_json(output / "parallel_protenix_progress.json", manifest)
    if failures and not args.allow_partial_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
