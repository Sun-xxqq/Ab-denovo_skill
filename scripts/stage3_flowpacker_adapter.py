#!/usr/bin/env python3
"""Pack unique Stage 3 IgGM sequences on their own IgGM input structures."""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _link_or_copy(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(str(source), str(destination))
    except OSError:
        shutil.copy2(source, destination)


def _safe(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")


def _antibody_sequence(job, heavy_chain, light_chain):
    by_chain = {item["chain"]: item["sequence"] for item in job["sequences"]}
    required = [heavy_chain] + ([light_chain] if light_chain else [])
    missing = [chain for chain in required if not by_chain.get(chain)]
    if missing:
        raise ValueError(
            "%s lacks antibody chains required by FlowPacker: %s"
            % (job["candidate_id"], ",".join(missing))
        )
    if heavy_chain != "A" or (light_chain and light_chain != "B"):
        raise ValueError(
            "the installed antibody FlowPacker requires heavy chain A and light chain B"
        )
    return {chain: by_chain[chain] for chain in required}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--publish-root", required=True)
    parser.add_argument("--sampler", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--heavy-chain", default="A")
    parser.add_argument("--light-chain", default="B")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    jobs = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    output = Path(args.output).resolve()
    publish_root = Path(args.publish_root).resolve()
    input_root = output / "input_pdbs"
    raw_root = output / "raw"
    packed_root = output / "packed"
    for path in (input_root, raw_root, packed_root, publish_root):
        path.mkdir(parents=True, exist_ok=True)

    rows, mapping, seen_tokens = [], {}, set()
    for job in jobs:
        candidate_id = job["candidate_id"]
        parent = Path(job.get("iggm_input_pdb") or job["pdb"]).resolve()
        if not parent.is_file():
            raise FileNotFoundError(
                "IgGM input PDB for %s was not found: %s" % (candidate_id, parent)
            )
        destination = packed_root / job["output_relpath"] / "design.pdb"
        public = (
            publish_root
            / job.get("public_group", job.get("iggm_parent", "candidates"))
            / (job.get("public_stem", job.get("iggm_variant", _safe(candidate_id))) + ".pdb")
        )
        if destination.is_file() and destination.stat().st_size:
            _link_or_copy(destination, public)
            continue

        token = _safe(job.get("prediction_name") or candidate_id)
        if token in seen_tokens:
            raise ValueError("duplicate Stage 3 FlowPacker input token: %s" % token)
        seen_tokens.add(token)
        link_name = token + ".pdb"
        _link_or_copy(parent, input_root / link_name)
        rows.append(
            {
                "link_name": link_name,
                "seq_idx": 1,
                "sequence_dict": repr(
                    _antibody_sequence(job, args.heavy_chain, args.light_chain)
                ),
            }
        )
        mapping[token + "_1"] = {
            "candidate_id": candidate_id,
            "destination": str(destination),
            "public": str(public),
            "iggm_input_pdb": str(parent),
        }

    csv_path = output / "flowpacker_candidates.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["link_name", "seq_idx", "sequence_dict"]
        )
        writer.writeheader()
        writer.writerows(rows)
    _write_json(output / "flowpacker_mapping.json", mapping)

    if rows:
        command = [
            sys.executable,
            str(Path(args.sampler).resolve()),
            str(Path(args.config).resolve()),
            "--save_dir",
            str(raw_root),
            "--use_gt_masks",
            "True",
            "--csv_file",
            str(csv_path),
            "--pdb_dir",
            str(input_root),
            "--ckpt",
            str(Path(args.checkpoint).resolve()),
            "--seed",
            str(args.seed),
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            raise RuntimeError(
                "Stage 3 FlowPacker failed with exit code %s" % completed.returncode
            )

    packed = []
    for code, item in mapping.items():
        candidates = [
            raw_root / "best_run" / (code + ".pdb"),
            raw_root / "run_1" / (code + ".pdb"),
        ]
        source = next((path for path in candidates if path.is_file()), None)
        if source is None:
            raise FileNotFoundError(
                "FlowPacker produced no packed PDB for %s" % item["candidate_id"]
            )
        destination = Path(item["destination"])
        public = Path(item["public"])
        _link_or_copy(source, destination)
        _link_or_copy(destination, public)
        packed.append(
            {
                "candidate_id": item["candidate_id"],
                "iggm_input_pdb": item["iggm_input_pdb"],
                "flowpacker_pdb": str(destination),
                "public_pdb": str(public),
            }
        )

    # Include already completed outputs in the completion audit.
    by_id = {item["candidate_id"]: item for item in packed}
    for job in jobs:
        if job["candidate_id"] in by_id:
            continue
        destination = packed_root / job["output_relpath"] / "design.pdb"
        if not destination.is_file():
            raise FileNotFoundError(
                "Stage 3 FlowPacker output was not found: %s" % destination
            )
        by_id[job["candidate_id"]] = {
            "candidate_id": job["candidate_id"],
            "iggm_input_pdb": str(
                Path(job.get("iggm_input_pdb") or job["pdb"]).resolve()
            ),
            "flowpacker_pdb": str(destination),
        }
    summary = {
        "status": "complete",
        "candidate_count": len(jobs),
        "packed_count": len(by_id),
        "reference_policy": "each candidate uses its own IgGM input PDB",
        "sequence_policy": "unique complete sequences from Stage 3 manifest",
        "packed_root": str(packed_root),
        "records": [by_id[key] for key in sorted(by_id)],
    }
    _write_json(output / "completion_manifest.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
