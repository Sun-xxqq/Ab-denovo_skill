"""Build and publish the Stage 3 IgGM specificity workflow."""

import csv
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

from .util import read_json, write_json


STAGE3_DIRS = (
    "00_stage3_plan",
    "01_iggm_variants",
    "02_flowpacker_sidechain_packing",
    "03_cldn18_2_structure_prediction",
    "04_cldn18_2_on_target_screen",
    "05_cldn18_1_structure_prediction",
    "06_cldn18_1_off_target_screen",
    "07_final_candidates",
)


def ensure_stage3_layout(run_dir):
    root = Path(run_dir) / "pipeline_output" / "stage3"
    for name in STAGE3_DIRS:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def _link_or_copy(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise RuntimeError("Stage 3 destination conflicts with source: %s" % destination)
        return destination
    try:
        os.link(str(source), str(destination))
    except OSError:
        shutil.copy2(source, destination)
    return destination


def _read_fasta(path):
    records, name, sequence = [], None, []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append({"chain": name, "sequence": "".join(sequence)})
                name, sequence = line[1:].split()[0], []
            elif name is None:
                raise ValueError("FASTA sequence appears before a header: %s" % path)
            else:
                sequence.append(line.upper())
    if name is not None:
        records.append({"chain": name, "sequence": "".join(sequence)})
    if not records or any(not item["sequence"] for item in records):
        raise ValueError("IgGM FASTA has no usable records: %s" % path)
    return records


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")


def _mutation_notation(origin_fasta, variant_fasta):
    origin = _read_fasta(origin_fasta)
    variant = _read_fasta(variant_fasta)
    if [item["chain"] for item in origin] != [item["chain"] for item in variant]:
        raise ValueError("origin and variant FASTA chain IDs differ")
    mutations = []
    for original, changed in zip(origin, variant):
        if len(original["sequence"]) != len(changed["sequence"]):
            raise ValueError(
                "origin and variant FASTA lengths differ for chain %s"
                % original["chain"]
            )
        mutations.extend(
            "%s%s%s" % (before, index, after)
            for index, (before, after) in enumerate(
                zip(original["sequence"], changed["sequence"]), 1
            )
            if before != after
        )
    return ";".join(mutations) if mutations else "WT"


def _sequence_sha256(records):
    """Hash the ordered chain IDs and complete sequences used by Protenix."""
    payload = json.dumps(
        [(item["chain"], item["sequence"]) for item in records],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_stage3_manifest(run_dir, spec):
    """Build one Protenix job per unique, complete IgGM sequence."""
    run_dir = Path(run_dir).resolve()
    stage3 = ensure_stage3_layout(run_dir)
    iggm = run_dir / "pipeline_output" / "stage2" / "09_IgGM"
    batch_path = iggm / "batch_manifest.json"
    if not batch_path.is_file():
        raise FileNotFoundError("IgGM batch manifest was not found: %s" % batch_path)
    batch = read_json(batch_path)
    heavy = spec["framework"]["heavy_chain"]
    light = spec["framework"].get("light_chain")
    target = spec["target"]["chain"]
    required = [heavy] + ([light] if light else []) + [target]
    raw_candidates, seen = [], set()

    for result in batch.get("results", []):
        if result.get("status") != "complete":
            continue
        label = _safe_name(result.get("label") or Path(result["output"]).parent.name)
        parent_pdb = Path(result.get("antigen") or result.get("source_pdb", ""))
        if not parent_pdb.is_file():
            raise FileNotFoundError("IgGM parent complex was not found: %s" % parent_pdb)
        origin_fasta = Path(result.get("origin_fasta", ""))
        if not origin_fasta.is_file():
            raise FileNotFoundError(
                "IgGM origin FASTA was not found: %s" % origin_fasta
            )
        candidate_dir = iggm / label / "candidates"
        if not candidate_dir.is_dir():
            configured_output = Path(result.get("output") or "")
            candidate_dir = (
                configured_output
                if configured_output.is_dir()
                else iggm / label
            )
        index_path = candidate_dir / "candidates.csv"
        if index_path.is_file():
            with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        else:
            rows = [
                {"candidate_id": path.stem, "pdb": "", "fasta": str(path)}
                for path in sorted(candidate_dir.glob("*.fasta"))
            ]
        if not rows:
            raise FileNotFoundError(
                "IgGM candidate index or FASTA variants were not found: %s"
                % candidate_dir
            )
        for row in rows:
            variant_id = _safe_name(row.get("candidate_id") or Path(row.get("fasta", "")).stem)
            source_fasta = candidate_dir / (variant_id + ".fasta")
            if not source_fasta.is_file():
                source_fasta = Path(row.get("fasta", ""))
            if not source_fasta.is_file():
                raise FileNotFoundError("IgGM variant FASTA was not found: %s" % source_fasta)
            sequences = _read_fasta(source_fasta)
            chain_ids = [item["chain"] for item in sequences]
            if chain_ids != required:
                raise ValueError(
                    "%s chains are %s; expected %s" % (source_fasta, chain_ids, required)
                )
            candidate_id = "%s__%s" % (label, variant_id)
            if candidate_id in seen:
                raise ValueError("duplicate Stage 3 candidate_id: %s" % candidate_id)
            seen.add(candidate_id)
            raw_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "label": label,
                    "variant_id": variant_id,
                    "source_fasta": source_fasta,
                    "origin_fasta": origin_fasta,
                    "parent_pdb": parent_pdb,
                    "sequences": sequences,
                }
            )
    if not raw_candidates:
        raise RuntimeError("IgGM completed without any Stage 3 FASTA variants")

    jobs, canonical_by_group_sequence, duplicates = [], {}, []
    for item in sorted(raw_candidates, key=lambda value: value["candidate_id"]):
        candidate_id = item["candidate_id"]
        sequence_sha256 = _sequence_sha256(item["sequences"])
        deduplication_key = (item["label"], sequence_sha256)
        canonical = canonical_by_group_sequence.get(deduplication_key)
        if canonical is not None:
            canonical["duplicate_candidate_ids"].append(candidate_id)
            duplicates.append(
                {
                    "duplicate_candidate_id": candidate_id,
                    "canonical_candidate_id": canonical["candidate_id"],
                    "iggm_parent": item["label"],
                    "sequence_sha256": sequence_sha256,
                    "source_fasta": str(item["source_fasta"].resolve()),
                }
            )
            stale_public = (
                stage3
                / "01_iggm_variants"
                / item["label"]
                / (item["variant_id"] + ".fasta")
            )
            if stale_public.is_file():
                stale_public.unlink()
            continue

        public_fasta = _link_or_copy(
            item["source_fasta"],
            stage3
            / "01_iggm_variants"
            / item["label"]
            / (item["variant_id"] + ".fasta"),
        )
        prediction_name = "iggm_%s_%s" % (
            _safe_name(item["label"]).replace(".", "_"),
            _safe_name(item["variant_id"]).replace(".", "_"),
        )
        job = {
            "candidate_id": candidate_id,
            "backbone_id": item["label"],
            "design_id": item["variant_id"],
            "prediction_name": prediction_name,
            "output_relpath": "%s/%s" % (item["label"], item["variant_id"]),
            "pdb": str(item["parent_pdb"].resolve()),
            "iggm_input_pdb": str(item["parent_pdb"].resolve()),
            "variant_fasta": str(public_fasta.resolve()),
            "origin_fasta": str(item["origin_fasta"].resolve()),
            "mutation": _mutation_notation(item["origin_fasta"], public_fasta),
            "sequence_sha256": sequence_sha256,
            "duplicate_candidate_ids": [],
            "iggm_parent": item["label"],
            "iggm_variant": item["variant_id"],
            "sequences": item["sequences"],
            "design_chains": [heavy] + ([light] if light else []),
            "target_chain": target,
            "hotspots": list(spec.get("target", {}).get("hotspots", [])),
            "copy_design": False,
            "publish_design": False,
            "public_group": item["label"],
            "public_stem": item["variant_id"],
        }
        jobs.append(job)
        canonical_by_group_sequence[deduplication_key] = job

    plan_path = stage3 / "00_stage3_plan" / "candidates.json"
    write_json(plan_path, jobs)
    work_path = run_dir / ".agent_work" / "stage3" / "protenix" / "candidates.json"
    write_json(work_path, jobs)
    dedup_path = stage3 / "00_stage3_plan" / "sequence_deduplication.json"
    write_json(
        dedup_path,
        {
            "strategy": "sha256 of ordered chain IDs and complete sequences",
            "scope": "within each IgGM parent group",
            "representative_rule": (
                "lexicographically smallest candidate_id within each IgGM parent"
            ),
            "input_candidate_count": len(raw_candidates),
            "unique_candidate_count": len(jobs),
            "duplicate_candidate_count": len(duplicates),
            "duplicates": duplicates,
        },
    )
    write_json(
        stage3 / "00_stage3_plan" / "stage3_plan.json",
        {
            "input": "Stage 2 09_IgGM FASTA variants",
            "input_candidate_count": len(raw_candidates),
            "candidate_count": len(jobs),
            "duplicates_removed": len(duplicates),
            "sequence_deduplication": str(dedup_path.resolve()),
            "sidechain_packing": (
                "FlowPacker on each unique sequence using its own IgGM input PDB"
            ),
            "steps": list(STAGE3_DIRS[1:]),
            "on_target": "CLDN18.2",
            "off_target": "CLDN18.1",
            "specificity_rule": "status == OK and S_on - S_off > 0",
            "threshold_auto_relaxed": False,
        },
    )
    return plan_path, jobs


def attach_stage3_flowpacker_outputs(run_dir, jobs, packed_root):
    """Replace each Protenix design reference with its packed Stage 3 PDB."""
    run_dir = Path(run_dir).resolve()
    packed_root = Path(packed_root).resolve()
    updated = []
    for original in jobs:
        job = dict(original)
        packed = packed_root / job["output_relpath"] / "design.pdb"
        if not packed.is_file() or packed.stat().st_size == 0:
            raise FileNotFoundError(
                "Stage 3 FlowPacker output was not found: %s" % packed
            )
        parent = Path(job.get("iggm_input_pdb") or job["pdb"]).resolve()
        job["iggm_input_pdb"] = str(parent)
        job["flowpacker_pdb"] = str(packed)
        job["pdb"] = str(packed)
        job["copy_design"] = True
        job["publish_design"] = True
        updated.append(job)

    stage3 = ensure_stage3_layout(run_dir)
    plan_path = stage3 / "00_stage3_plan" / "candidates.json"
    write_json(plan_path, updated)
    write_json(
        run_dir / ".agent_work" / "stage3" / "protenix" / "candidates.json",
        updated,
    )
    return plan_path, updated


def _antibody_sequences(fasta_path, design_chains):
    """Format complete antibody chains in design order for one CSV cell."""
    if not design_chains or len(set(design_chains)) != len(design_chains):
        raise ValueError("Stage 3 antibody chains must be nonempty and unique")
    records = _read_fasta(fasta_path)
    sequences = {record["chain"]: record["sequence"] for record in records}
    if len(sequences) != len(records):
        raise ValueError("Stage 3 FASTA contains duplicate chain IDs: %s" % fasta_path)
    missing = [chain for chain in design_chains if chain not in sequences]
    if missing:
        raise ValueError("Stage 3 FASTA is missing antibody chains: %s" % missing)
    return "\n".join("%s: %s" % (chain, sequences[chain]) for chain in design_chains)


def publish_stage3_results(run_dir, jobs, off_target_results):
    """Publish paired S_on/S_off/delta results and materialize passing variants."""
    run_dir = Path(run_dir).resolve()
    stage3 = ensure_stage3_layout(run_dir)
    paired = Path(off_target_results)
    records = read_json(paired / "offtarget_results.json")
    by_id = {item["candidate_id"]: item for item in jobs}
    accepted = [
        item
        for item in records
        if item.get("status") == "OK"
        and item.get("specificity_pass") is True
        and item.get("delta") is not None
        and float(item["delta"]) > 0.0
    ]
    final_root = stage3 / "07_final_candidates"
    published = []
    for record in accepted:
        job = by_id[record["candidate_id"]]
        destination = final_root / job["iggm_parent"] / job["iggm_variant"]
        _link_or_copy(job["variant_fasta"], destination / "variant.fasta")
        _link_or_copy(job["pdb"], destination / "design.pdb")
        on_prediction = (
            run_dir
            / ".agent_work"
            / "stage3"
            / "protenix"
            / "predictions"
            / job["output_relpath"]
            / "pred.cif"
        )
        off_prediction = (
            run_dir
            / ".agent_work"
            / "stage3"
            / "offtarget_protenix"
            / "predictions"
            / job["output_relpath"]
            / "pred.cif"
        )
        _link_or_copy(on_prediction, destination / "cldn18_2.cif")
        _link_or_copy(off_prediction, destination / "cldn18_1.cif")
        write_json(destination / "specificity_metrics.json", record)
        published.append(
            {
                "candidate_id": record["candidate_id"],
                "mutation": job.get("mutation")
                or _mutation_notation(job["origin_fasta"], job["variant_fasta"]),
                "S_on": record.get("S_on"),
                "S_off": record.get("S_off"),
                "delta": record.get("delta"),
                "design_pred_rmsd": record.get("design_pred_rmsd"),
                "antibody_pred_rmsd": record.get("antibody_pred_rmsd"),
                "target_pred_rmsd": record.get("target_pred_rmsd"),
                "design_pdb": str(destination / "design.pdb"),
                "directory": str(destination),
                "antibody_sequences": _antibody_sequences(
                    job["variant_fasta"], job["design_chains"]
                ),
            }
        )
    fields = [
        "candidate_id",
        "mutation",
        "S_on",
        "S_off",
        "delta",
        "design_pred_rmsd",
        "antibody_pred_rmsd",
        "target_pred_rmsd",
        "design_pdb",
        "directory",
        "antibody_sequences",
    ]
    with (final_root / "final_candidates.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(published)
    summary = {
        "stage": "stage3_iggm_specificity_screening",
        "evaluated": len(records),
        "passed": len(accepted),
        "rejected": sum(item.get("specificity_pass") is False for item in records),
        "uncertain": sum(item.get("status") == "UNCERTAIN" for item in records),
        "rule": "status == OK and delta = S_on - S_off > 0",
        "final_candidates": str(final_root),
    }
    write_json(stage3 / "00_stage3_plan" / "completion_manifest.json", summary)
    return summary
