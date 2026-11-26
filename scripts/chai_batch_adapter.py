#!/usr/bin/env python3
"""Checkpointed Chai-1 batch inference with full confidence export."""

import argparse
import json
import shutil
import traceback
from pathlib import Path

import numpy as np


def _install_runtime_caches():
    """Install process-local Chai caches without defeating low-memory inference.

    TorchScript components are deserialized from disk once and retained on CPU.
    Each lightweight wrapper moves its component to the active device and moves
    it back to CPU when Chai releases that phase. ESM embeddings are cached by
    exact protein sequence, so a shared target chain is embedded only once.
    """
    import torch
    import chai_lab.chai1 as chai1
    from chai_lab.data.dataset import inference_dataset
    from chai_lab.data.dataset.embeddings import esm as esm_module

    installed = getattr(chai1, "_agent_ab_runtime_caches", None)
    if installed is not None:
        return installed

    stats = {
        "model_disk_loads": 0,
        "model_reuses": 0,
        "model_cache_entries": 0,
        "esm_sequences_computed": 0,
        "esm_sequence_hits": 0,
        "esm_cache_entries": 0,
        "conformer_cache_loads": 0,
        "conformer_cache_reuses": 0,
    }

    model_cache = {}

    class CachedModuleWrapper(chai1.ModuleWrapper):
        def __del__(self):
            module = getattr(self, "jit_module", None)
            if module is not None:
                try:
                    module.to(torch.device("cpu"))
                except Exception:
                    # Interpreter shutdown or an already-torn-down CUDA runtime.
                    pass

    def cached_load_exported(comp_key, device):
        torch.jit.set_fusion_strategy([("STATIC", 0), ("DYNAMIC", 0)])
        if comp_key not in model_cache:
            local_path = chai1.chai1_component(comp_key)
            model_cache[comp_key] = torch.jit.load(local_path, map_location="cpu")
            stats["model_disk_loads"] += 1
            stats["model_cache_entries"] = len(model_cache)
        else:
            stats["model_reuses"] += 1
        return CachedModuleWrapper(model_cache[comp_key].to(device))

    chai1.load_exported = cached_load_exported

    original_esm_loader = esm_module._get_esm_contexts_for_sequences
    esm_cache = {}

    def cached_esm_loader(prot_sequences, device):
        requested = set(prot_sequences)
        missing = requested.difference(esm_cache)
        stats["esm_sequence_hits"] += len(requested) - len(missing)
        if missing:
            computed = original_esm_loader(missing, device)
            esm_cache.update(computed)
            stats["esm_sequences_computed"] += len(computed)
            stats["esm_cache_entries"] = len(esm_cache)
        return {sequence: esm_cache[sequence] for sequence in requested}

    esm_module._get_esm_contexts_for_sequences = cached_esm_loader

    original_conformer_generator = inference_dataset.RefConformerGenerator
    conformer_generator = original_conformer_generator()
    stats["conformer_cache_loads"] = 1

    def cached_conformer_generator(*args, **kwargs):
        if args or kwargs:
            return original_conformer_generator(*args, **kwargs)
        stats["conformer_cache_reuses"] += 1
        return conformer_generator

    inference_dataset.RefConformerGenerator = cached_conformer_generator

    def prewarm_esm(sequences):
        sequences = {str(sequence) for sequence in sequences if sequence}
        if sequences:
            cached_esm_loader(sequences, torch.device("cuda:0"))

    installed = (stats, prewarm_esm)
    chai1._agent_ab_runtime_caches = installed
    return installed


def _write_cache_stats(output, stats):
    (Path(output) / "runtime_cache_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _json_value(value):
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else array.tolist()


def _next_attempt(root):
    index = 1
    while (root / ("attempt_%03d" % index)).exists():
        index += 1
    path = root / ("attempt_%03d" % index)
    path.mkdir(parents=True)
    return path


def _write_fasta(path, sequences):
    with path.open("w", encoding="utf-8") as handle:
        for item in sequences:
            handle.write(">protein|%s\n%s\n" % (item["chain"], item["sequence"]))


def _finalize(job, candidate_dir, best_scores, pae, plddt):
    candidate_id = job["candidate_id"]
    predicted = candidate_dir / "pred.cif"
    if not predicted.exists() and (candidate_dir / "predicted.cif").exists():
        predicted = candidate_dir / "predicted.cif"
    design = candidate_dir / "design.pdb"
    pairwise = np.squeeze(np.asarray(best_scores.get("per_chain_pair_iptm", [])))
    confidence = {
        "pair_chains_iptm": {
            str(i): {str(j): float(pairwise[i, j]) for j in range(pairwise.shape[1])}
            for i in range(pairwise.shape[0])
        }
    }
    (candidate_dir / "confidence_candidate.json").write_text(
        json.dumps(confidence, indent=2) + "\n", encoding="utf-8"
    )
    metrics = {
        "candidate_id": candidate_id,
        "backbone_id": job.get("backbone_id"),
        "design_id": job.get("design_id"),
        "source_pdb": str(design if design.exists() else job["pdb"]),
        "original_source_pdb": job["pdb"],
        "predicted_cif": str(predicted),
        "plddt": float(np.mean(plddt)),
        "ptm": float(np.asarray(best_scores["ptm"]).reshape(-1)[0]),
        "iptm": float(np.asarray(best_scores["iptm"]).reshape(-1)[0]),
        "aggregate_score": float(np.asarray(best_scores["aggregate_score"]).reshape(-1)[0]),
        "has_inter_chain_clashes": bool(np.asarray(best_scores.get("has_inter_chain_clashes", [True])).reshape(-1)[0]),
        "chain_chain_clashes": np.asarray(best_scores.get("chain_chain_clashes", [])).squeeze().tolist(),
    }
    (candidate_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def _run_one(job, output_root, seed, num_diffn_samples):
    from chai_lab.chai1 import run_inference
    from chai_lab.ranking.rank import get_scores

    candidate_id = job["candidate_id"]
    candidate_dir = output_root / job.get("output_relpath", candidate_id)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    design_path = candidate_dir / "design.pdb"
    if not design_path.exists():
        shutil.copy2(job["pdb"], design_path)
    metrics_path = candidate_dir / "metrics.json"
    if metrics_path.exists() and ((candidate_dir / "pred.cif").exists() or (candidate_dir / "predicted.cif").exists()):
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    score_paths = sorted(candidate_dir.glob("attempt_*/scores.model_idx_0.npz"))
    if score_paths and ((candidate_dir / "pred.cif").exists() or (candidate_dir / "predicted.cif").exists()) and (candidate_dir / "pae_candidate.npz").exists() and (candidate_dir / "plddt_candidate.npz").exists():
        with np.load(score_paths[-1]) as stored:
            best_scores = {key: stored[key] for key in stored.files}
        with np.load(candidate_dir / "pae_candidate.npz") as stored:
            pae = stored["pae"]
        with np.load(candidate_dir / "plddt_candidate.npz") as stored:
            plddt = stored["plddt"]
        return _finalize(job, candidate_dir, best_scores, pae, plddt)

    fasta_path = candidate_dir / "input.fasta"
    _write_fasta(fasta_path, job["sequences"])
    attempt_dir = _next_attempt(candidate_dir)
    candidates = run_inference(
        fasta_file=fasta_path,
        output_dir=attempt_dir,
        use_esm_embeddings=True,
        use_msa_server=False,
        num_trunk_recycles=3,
        num_diffn_timesteps=200,
        num_diffn_samples=num_diffn_samples,
        num_trunk_samples=1,
        seed=seed,
        low_memory=True,
    ).sorted()

    best_scores = {key: _json_value(value) for key, value in get_scores(candidates.ranking_data[0]).items()}
    pae = candidates.pae[0].detach().cpu().numpy()
    plddt = candidates.plddt[0].detach().cpu().numpy()
    predicted = candidate_dir / "pred.cif"
    shutil.copy2(candidates.cif_paths[0], predicted)
    np.savez(candidate_dir / "pae_candidate.npz", pae=pae)
    np.savez(candidate_dir / "plddt_candidate.npz", plddt=plddt)

    return _finalize(job, candidate_dir, best_scores, pae, plddt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--num-diffn-samples", type=int, default=1)
    parser.add_argument("--disable-runtime-cache", action="store_true")
    args = parser.parse_args()
    jobs = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cache_stats = None
    if not args.disable_runtime_cache:
        cache_stats, prewarm_esm = _install_runtime_caches()
        unique_sequences = {
            item["sequence"]
            for job in jobs
            for item in job.get("sequences", [])
            if item.get("sequence")
        }
        print(
            "[chai-cache] precomputing ESM embeddings for %s unique sequences"
            % len(unique_sequences),
            flush=True,
        )
        prewarm_esm(unique_sequences)
        _write_cache_stats(output, cache_stats)
        print("[chai-cache] ESM precompute complete", flush=True)
    completed, failures = [], []
    for index, job in enumerate(jobs):
        try:
            completed.append(_run_one(job, output, args.seed + index, args.num_diffn_samples))
        except Exception as exc:
            failures.append({"candidate_id": job.get("candidate_id"), "error": str(exc), "traceback": traceback.format_exc()})
            (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
        finally:
            if cache_stats is not None:
                _write_cache_stats(output, cache_stats)
    (output / "chai_metrics.json").write_text(json.dumps(completed, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
