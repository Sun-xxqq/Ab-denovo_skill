from .fastrelax_naming import fastrelax_path
import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from .adapters import build_command
from .pipeline_choices import STAGES
from .recovery import RecoveryState
from .ranking import write_ranking
from .util import read_json, select_gpu, write_json
from .workflow import _run_dir, execute_run, run_fastrelax
from .maturation import build_maturation_plan
from .functional import write_functional_assessment
from .artifacts import (
    materialize_round_outputs,
    write_result_paths,
    write_screening_summary,
)
from .stage1_layout import finalize_stage1_layout, prepare_stage1_rosetta_inputs
from .stage2_layout import finalize_stage2_layout
from .iggm import run_stage2_affinity_maturation
from .stage3 import (
    attach_stage3_flowpacker_outputs,
    build_stage3_manifest,
    publish_stage3_results,
)


AA3 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}

PREDICTION_TOOLS = {"protenix", "chai"}


def _truthy(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def _stage_tool_root(run_dir, stage, tool):
    """Keep tool-specific inputs and results inside their scientific stage."""
    if stage == "stage3":
        leaf = {
            "protenix": "03_cldn18_2_structure_prediction",
            "ipsae": "04_cldn18_2_on_target_screen",
        }.get(tool, tool)
        return Path(run_dir) / "pipeline_output" / stage / leaf
    leaf = {"protenix": "protenix_output"}.get(tool, tool)
    return Path(run_dir) / "pipeline_output" / stage / leaf


def _prediction_work_root(run_dir, stage, tool):
    """Keep predictor manifests, PAE arrays and raw files out of public output."""
    return Path(run_dir) / ".agent_work" / stage / tool


def _stage_tool_output(run_dir, stage, tool):
    if tool == "protenix":
        return _prediction_work_root(run_dir, stage, tool) / "predictions"
    leaf = "predictions" if tool in PREDICTION_TOOLS else "results"
    return _stage_tool_root(run_dir, stage, tool) / leaf


def _prefer_current_path(current, legacy):
    """Read historical runs without writing new data into the legacy layout."""
    current = Path(current)
    legacy = Path(legacy)
    return current if current.exists() or not legacy.exists() else legacy


def _finalized_round0_summary(run_dir):
    """Read the compact final table when a completed run is resumed."""
    table = (
        Path(run_dir)
        / "pipeline_output"
        / "stage1"
        / "design_pred_screen.csv"
    )
    if not table.exists():
        return None
    with table.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    accepted = [
        row
        for row in rows
        if str(row.get("filter_pass", "")).strip().lower()
        in {"1", "true", "yes", "pass"}
    ]
    return {
        "round": 0,
        "backbones": len({row.get("backbone_id") for row in rows}),
        "sequences": len(rows),
        "designs": sum(bool(row.get("design_path")) for row in rows),
        "predictions": sum(bool(row.get("prediction_path")) for row in rows),
        "accepted_for_partialflow": len(accepted),
        "output_root": str(table.parent),
        "all_candidates_table": str(table),
        "partialflow_accepted_table": None,
    }


def _finalized_round1_summary(run_dir):
    """Read compact Stage 2 outputs when a completed run is resumed."""
    table = Path(run_dir) / "pipeline_output" / "stage2" / "design_pred_screen.csv"
    if not table.exists():
        return None
    with table.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    accepted = [
        row
        for row in rows
        if str(row.get("filter_pass", "")).strip().lower()
        in {"1", "true", "yes", "pass"}
    ]
    return {
        "round": 1,
        "backbones": len({row.get("backbone_id") for row in rows}),
        "sequences": len(rows),
        "designs": sum(bool(row.get("design_path")) for row in rows),
        "predictions": sum(bool(row.get("prediction_path")) for row in rows),
        "accepted": len(accepted),
        "output_root": str(table.parent),
        "all_candidates_table": str(table),
    }


def _pdb_sequences(path):
    chains, seen = {}, set()
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                continue
            chain, residue, number = line[21:22].strip() or "A", line[17:20].strip(), line[22:27].strip()
            key = (chain, number)
            if key in seen:
                continue
            seen.add(key)
            chains.setdefault(chain, []).append(AA3.get(residue, "X"))
    return {chain: "".join(sequence) for chain, sequence in chains.items() if sequence}


def _candidate_pdbs(run_dir):
    roots = [Path(run_dir) / "pipeline_output", Path(run_dir) / "adapter_output"]
    values = []
    for root in roots:
        if root.exists():
            values.extend(path for path in root.rglob("*.pdb") if path.is_file())
    return sorted(set(values))


def _flowpacker_candidates(run_dir):
    run_dir = Path(run_dir)
    root = run_dir / "pipeline_output" / "stage1" / "flowpacker_output"
    values = sorted(root.glob("backbone.*/*.pdb")) if root.exists() else []
    legacy = root / "run_1"
    if not values and legacy.exists():
        values = sorted(legacy.glob("*.pdb"))
    return values or _candidate_pdbs(run_dir)


def _select_design_inputs(paths, screening):
    """Apply an independent limit to every backbone; never globally truncate."""
    grouped = {}
    for index, path in enumerate(paths):
        backbone_id, design_id = _design_identity(path, index)
        grouped.setdefault(backbone_id, []).append((design_id, path))
    per_backbone = int(screening.get("max_candidates_per_backbone", 1000))
    selected = []
    for backbone_id in sorted(grouped):
        for design_id, path in sorted(grouped[backbone_id])[:per_backbone]:
            selected.append((backbone_id, design_id, path))
    return selected


def _design_identity(path, fallback_index):
    match = re.search(r"_(\d+)_(\d+)$", Path(path).stem)
    if match:
        backbone_index, sequence_index = (int(value) for value in match.groups())
    else:
        backbone_index, sequence_index = fallback_index, 0
    backbone_id = "backbone_%03d" % backbone_index
    design_id = "design_%03d" % sequence_index
    return backbone_id, design_id


def _prediction_name(backbone_id, design_id):
    """Return a short stable on-disk name without changing internal IDs."""
    backbone_match = re.search(r"(\d+)$", str(backbone_id))
    design_match = re.search(r"(\d+)$", str(design_id))
    if not backbone_match or not design_match:
        raise ValueError(
            "prediction IDs must end in numbers: %s, %s"
            % (backbone_id, design_id)
        )
    return "pred_%d_%d" % (
        int(backbone_match.group(1)), int(design_match.group(1))
    )


def _designed_positions(run_dir, pdb, sequences, stage="stage1"):
    """Recover the AbMPNN-designable (CDR) positions from its fixed CSV."""
    path = Path(run_dir) / "pipeline_output" / stage / "mpnn_pdbs" / "fixed_positions.csv"
    if not path.exists():
        return {}
    source_key = re.sub(r"_\d+$", "", Path(pdb).stem).lower()
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = next((item for item in rows if str(item.get("pdb_name", "")).lower() == source_key), None)
    if not row:
        return {}
    segments = [item.strip() for item in str(row.get("motif_index", "")).strip().strip("-").split("-")]
    result = {}
    for index, item in enumerate(sequences):
        if index >= len(segments):
            break
        fixed = {int(value) for value in segments[index].split() if value.isdigit()}
        result[item["chain"]] = [position for position in range(1, len(item["sequence"]) + 1) if position not in fixed]
    return result


def _stage1_group_prefixes(run_dir):
    root = Path(run_dir) / "pipeline_output" / "stage1" / "stage1_candidates"
    result = {}
    index_path = root / "stage1_groups.json"
    groups = read_json(index_path) if index_path.exists() else {}
    for backbone_id, metadata in groups.items():
        if int(metadata.get("accepted", 0)) < 1:
            continue
        records_path = root / backbone_id / "stage1_candidates.json"
        records = read_json(records_path) if records_path.exists() else []
        prefixes = set()
        for record in records:
            source = record.get("original_source_pdb") or record.get("source_pdb")
            if not source:
                continue
            stem = Path(source).stem
            prefixes.add(re.sub(r"_\d+$", "", stem))
        result[backbone_id] = sorted(prefixes, key=len, reverse=True)
    return result


def _final_candidate_roles(spec, sequences):
    available = {item["chain"] for item in sequences}
    target = spec["target"]["chain"]
    if spec["design_type"] == "binder":
        design = [spec["generation"]["binder_chain"]]
        canonical = (["A"], "B")
    elif spec["design_type"] == "nanobody":
        design = [spec["framework"]["heavy_chain"]]
        canonical = (["A"], "C")
    else:
        design = [spec["framework"]["heavy_chain"], spec["framework"]["light_chain"]]
        canonical = (["A", "B"], "C")
    if target in available and all(chain in available for chain in design):
        return design, target
    canonical_design, canonical_target = canonical
    if canonical_target in available and all(chain in available for chain in canonical_design):
        return canonical_design, canonical_target
    raise ValueError("cannot resolve design/target chain roles from final PDB chains: %s" % sorted(available))


def _build_final_candidate_manifest(run_dir, spec, manifest_path=None):
    run_dir = Path(run_dir)
    flowpacker_root = run_dir / "pipeline_output" / "stage2" / "flowpacker_output"
    canonical_pdbs = sorted(flowpacker_root.glob("backbone.1_*/*.pdb"))
    if not canonical_pdbs:
        canonical_pdbs = sorted(flowpacker_root.glob("backbone.0_*_sample*/*.pdb"))
    legacy_root = flowpacker_root / "run_1"
    pdbs = canonical_pdbs or (
        sorted(legacy_root.glob("*.pdb")) if legacy_root.exists() else []
    )
    prefixes = _stage1_group_prefixes(run_dir)
    # Stage 2 intentionally drops Stage 1 groups for which the Rosetta energy
    # gate found no usable fixed positions.  Validate the final 4-way expansion
    # only against groups that actually produced a PartialFlow backbone; keeping
    # every Stage 1 group here incorrectly reports the gated-out groups as
    # missing final candidates.
    partial_root = run_dir / "pipeline_output" / "stage2" / "partial_output"
    partial_prefixes = {
        pdb.parent.name for pdb in partial_root.glob("*/sample*.pdb")
    }
    if partial_prefixes:
        prefixes = {
            backbone_id: values
            for backbone_id, values in prefixes.items()
            if any(prefix in partial_prefixes for prefix in values)
        }
    unmatched = []
    if canonical_pdbs:
        grouped = {}
        for pdb in pdbs:
            match = re.match(
                r"(?:backbone|flowpacker)\.(?:1_(\d+)_\d+|0_(\d+)_sample\d+)_\d+$",
                pdb.stem,
            )
            if not match:
                unmatched.append(str(pdb))
                continue
            parent_index = next(value for value in match.groups() if value is not None)
            backbone_id = "backbone_%03d" % int(parent_index)
            grouped.setdefault(backbone_id, []).append(pdb)
    else:
        grouped = {backbone_id: [] for backbone_id in prefixes}
        for pdb in pdbs:
            matches = [
                (len(prefix), backbone_id)
                for backbone_id, values in prefixes.items()
                for prefix in values
                if pdb.stem == prefix or pdb.stem.startswith(prefix + "_")
            ]
            if not matches:
                unmatched.append(str(pdb))
                continue
            grouped[max(matches)[1]].append(pdb)

    target_count = int(
        spec["validation"]["candidate_screening"]["maturation"]["final_generation"][
            "candidates_per_original_backbone_group"
        ]
    )
    incomplete = {
        backbone_id: len(values)
        for backbone_id, values in grouped.items()
        if len(values) != target_count
    }
    if incomplete or unmatched:
        raise RuntimeError(
            "final candidate set is incomplete: expected %s per group; counts=%s unmatched=%s"
            % (target_count, incomplete, len(unmatched))
        )

    jobs = []
    for backbone_id in sorted(grouped):
        for index, pdb in enumerate(sorted(grouped[backbone_id]), start=1):
            sequence_map = _pdb_sequences(pdb)
            sequences = [
                {"chain": chain, "sequence": sequence}
                for chain, sequence in sequence_map.items()
            ]
            design_chains, target_chain = _final_candidate_roles(spec, sequences)
            prediction_name = _prediction_name(backbone_id, "final_%03d" % index)
            jobs.append(
                {
                    "candidate_id": "%s_final_%03d" % (backbone_id, index),
                    "backbone_id": backbone_id,
                    "design_id": "final_%03d" % index,
                    "prediction_name": prediction_name,
                    "output_relpath": prediction_name,
                    "pdb": str(pdb),
                    "publish_design": False,
                    "sequences": sequences,
                    "design_chains": design_chains,
                    "target_chain": target_chain,
                    "hotspots": list(spec.get("target", {}).get("hotspots", [])),
                    "designed_positions": _designed_positions(
                        run_dir, pdb, sequences, stage="stage2"
                    ),
                }
            )
    path = Path(manifest_path) if manifest_path else (
        run_dir / "pipeline_output" / "final_evaluation" / "candidates.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, jobs)
    return path, jobs


def _env_python(config, env_name):
    root = Path(config["conda_root"])
    return str(root / ("bin/python" if env_name == "base" else "envs/%s/bin/python" % env_name))


def _offtarget_config(config, spec):
    result = dict(config.get("off_target_screening", {}))
    result.update(
        spec.get("validation", {})
        .get("candidate_screening", {})
        .get("off_target_screening", {})
    )
    result.setdefault("enabled", True)
    result.setdefault("name", "CLDN18.1")
    result.setdefault("source_chain", "A")
    result.setdefault("seed", 101)
    result.setdefault("pae_cutoff", 10)
    result.setdefault("dist_cutoff", 10)
    return result


def _accepted_stage_metrics(run_dir, stage):
    root = _prefer_current_path(
        _stage_tool_output(run_dir, stage, "ipsae"),
        Path(run_dir) / "adapter_output" / ("ipsae" if stage == "stage1" else "stage2_ipsae"),
    )
    records = {}
    for path in sorted(root.glob("*/filter_results.json")):
        for item in read_json(path):
            if not _truthy(item.get("filter_pass")):
                continue
            candidate_id = item.get("candidate_id")
            if not candidate_id:
                raise ValueError("accepted %s result has no candidate_id" % stage)
            if candidate_id in records:
                raise ValueError("duplicate accepted %s candidate_id: %s" % (stage, candidate_id))
            records[candidate_id] = item
    return root, records


def _build_offtarget_manifest(config, run_dir, spec, stage="stage2"):
    """Clone accepted on-target jobs and replace only the antigen sequence."""
    run_dir = Path(run_dir)
    predictor = spec["workflow"]["complex_prediction"]
    if predictor != "protenix":
        raise RuntimeError("CLDN18.1 off-target screening requires Protenix")
    source_manifest = _prefer_current_path(
        _prediction_work_root(run_dir, stage, predictor) / "candidates.json",
        _prefer_current_path(
            _stage_tool_root(run_dir, stage, predictor) / "candidates.json",
            run_dir / "adapter_inputs" / (predictor if stage == "stage1" else "stage2_comparison") / "candidates.json",
        ),
    )
    if not source_manifest.exists():
        raise FileNotFoundError("%s candidate manifest was not found" % stage)
    stage_jobs = read_json(source_manifest)
    jobs_by_id = {}
    for job in stage_jobs:
        candidate_id = job.get("candidate_id")
        if candidate_id in jobs_by_id:
            raise ValueError("duplicate %s manifest candidate_id: %s" % (stage, candidate_id))
        jobs_by_id[candidate_id] = job

    on_target_root, accepted = _accepted_stage_metrics(run_dir, stage)
    missing_jobs = sorted(set(accepted) - set(jobs_by_id))
    if missing_jobs:
        raise RuntimeError(
            "accepted candidate_id values missing from %s manifest: %s"
            % (stage, ", ".join(missing_jobs))
        )

    off_config = _offtarget_config(config, spec)
    structure = Path(off_config["structure"])
    sequences = _pdb_sequences(structure)
    source_chain = str(off_config["source_chain"])
    if source_chain not in sequences or not sequences[source_chain]:
        raise ValueError(
            "off-target antigen structure has no usable chain %s: %s"
            % (source_chain, structure)
        )
    off_sequence = sequences[source_chain]
    jobs = []
    for candidate_id in sorted(accepted):
        original = jobs_by_id[candidate_id]
        job = json.loads(json.dumps(original))
        target_chain = job.get("target_chain", spec["target"]["chain"])
        before = [dict(item) for item in job["sequences"]]
        replaced = 0
        for item in job["sequences"]:
            if item["chain"] == target_chain:
                item["sequence"] = off_sequence
                replaced += 1
        if replaced != 1:
            raise ValueError(
                "%s has %s target-chain entries; expected exactly one"
                % (candidate_id, replaced)
            )
        before_antibody = [item for item in before if item["chain"] != target_chain]
        after_antibody = [
            item for item in job["sequences"] if item["chain"] != target_chain
        ]
        if before_antibody != after_antibody:
            raise RuntimeError("antibody sequence changed for %s" % candidate_id)
        if [item["chain"] for item in before] != [
            item["chain"] for item in job["sequences"]
        ]:
            raise RuntimeError("chain representation changed for %s" % candidate_id)
        job["candidate_id"] = candidate_id
        job["off_target_name"] = off_config["name"]
        job["off_target_structure"] = str(structure)
        job["on_target_S_on"] = accepted[candidate_id].get("S_on")
        job["copy_design"] = False
        job["publish_design"] = False
        jobs.append(job)

    output = (
        run_dir
        / "pipeline_output"
        / stage
        / (
            "06_cldn18_1_off_target_screen"
            if stage == "stage3"
            else "offtarget_screen"
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "candidates.json"
    write_json(manifest_path, jobs)
    return manifest_path, jobs, on_target_root, off_config


def _run_offtarget_screen(
    config, run_dir, spec, state, gpu, stage="stage2", retry_failed=False, command_runner=None
):
    """Run A+CLDN18.1 and pair by candidate_id with the same-stage S_on."""
    run_dir = Path(run_dir)
    run_command = command_runner or _run_recoverable_command
    manifest_path, jobs, on_target_root, off_config = _build_offtarget_manifest(
        config, run_dir, spec, stage=stage
    )
    project_root = Path(config["project_root"])
    stage_root = run_dir / "pipeline_output" / stage
    if stage == "stage3":
        output = stage_root / "06_cldn18_1_off_target_screen" / "results"
        public_root = stage_root / "05_cldn18_1_structure_prediction"
    else:
        output = stage_root / "offtarget_screen"
        public_root = stage_root / "protenix_offtarget_output"
    prediction_dir = (
        run_dir / ".agent_work" / stage / "offtarget_protenix" / "predictions"
    )
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    protenix_config = config["tools"]["protenix"]
    environment["PROTENIX_ROOT_DIR"] = protenix_config.get(
        "root_dir", str(Path(protenix_config["data_dir"]).parent)
    )
    prediction_inputs = {
        "manifest": str(manifest_path),
        "publish_root": str(public_root),
        "seed": int(off_config["seed"]),
        "target_chain": spec["target"]["chain"],
        "need_atom_confidence": True,
        "allow_partial_failures": True,
    }
    prediction_argv = build_command(
        "protenix", config, prediction_inputs, prediction_dir, spec
    )
    run_command(
        state,
        "%s_offtarget_protenix" % stage,
        prediction_argv,
        config["tools"]["protenix"].get("path") or project_root,
        environment,
        run_dir / "logs" / ("%s.offtarget.protenix.log" % stage),
        retry_failed,
    )

    framework = spec.get("framework", {})
    evaluation_argv = [
        _env_python(config, config["tools"]["ppiflow"]["env"]),
        str(project_root / "scripts" / "evaluate_offtarget_candidates.py"),
        "--manifest", str(manifest_path),
        "--on-target-results", str(on_target_root),
        "--prediction-dir", str(prediction_dir),
        "--ipsae-script", str(Path(config["tools"]["ipsae"]["path"]) / "ipsae.py"),
        "--ipsae-python", _env_python(config, config["tools"]["ipsae"]["env"]),
        "--output", str(output),
        "--pae-cutoff", str(off_config["pae_cutoff"]),
        "--dist-cutoff", str(off_config["dist_cutoff"]),
        "--heavy-chain", str(framework.get("heavy_chain", "A")),
        "--light-chain", str(framework.get("light_chain", "")),
        "--target-chain", str(spec["target"]["chain"]),
        "--workflow-stage", stage,
    ]
    run_command(
        state,
        "%s_offtarget_ipsae" % stage,
        evaluation_argv,
        project_root,
        environment,
        run_dir / "logs" / ("%s.offtarget.ipsae.log" % stage),
        retry_failed,
    )
    summary = read_json(output / "summary.json")
    return {
        "manifest": str(manifest_path),
        "prediction_dir": str(prediction_dir),
        "output": str(output),
        "candidate_count": len(jobs),
        "summary": summary,
        "terminal_stage": True,
        "on_target_recomputed": False,
    }


def _run_recoverable_command(state, stage_name, argv, cwd, environment, log_path, retry_failed=False):
    if not state.should_run(stage_name, retry_failed=retry_failed):
        return
    state.update(stage_name, "running", argv=argv, log=str(log_path))
    with Path(log_path).open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            argv, cwd=str(cwd), env=environment, stdout=log,
            stderr=subprocess.STDOUT, check=False,
        )
    if completed.returncode:
        state.update(stage_name, "failed", returncode=completed.returncode, detail="see %s" % log_path)
        raise RuntimeError("%s failed with exit code %s; see %s" % (stage_name, completed.returncode, log_path))
    state.update(stage_name, "complete", returncode=0)


def _write_grouped_screen_table(results_root, output_path):
    records = []
    for path in sorted(Path(results_root).glob("*/filter_results.json")):
        records.extend(read_json(path))
    fields = sorted({key for record in records for key in record})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = dict(record)
            if isinstance(row.get("rejection_reasons"), list):
                row["rejection_reasons"] = ";".join(row["rejection_reasons"])
            writer.writerow(row)
    return records


def _prepare_stage2_rosetta_inputs(run_dir):
    """Publish only status=OK, delta>0 Stage 2 designs for final FastRelax."""
    run_dir = Path(run_dir)
    stage2 = run_dir / "pipeline_output" / "stage2"
    results_path = stage2 / "offtarget_screen" / "offtarget_results.json"
    records = read_json(results_path) if results_path.exists() else []
    accepted = [
        item
        for item in records
        if item.get("status") == "OK"
        and item.get("specificity_pass") is True
        and item.get("delta") is not None
        and float(item["delta"]) > 0.0
    ]
    output = stage2 / "filtered_delta"
    staging = stage2 / ".filtered_delta.tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    seen = set()
    for item in accepted:
        source = Path(item["source_pdb"])
        if not source.is_file():
            raise FileNotFoundError("Stage 2 FastRelax input not found: %s" % source)
        name = source.name
        if name in seen:
            raise ValueError("duplicate Stage 2 FastRelax input: %s" % name)
        seen.add(name)
        destination = staging / name
        destination.symlink_to(os.path.relpath(source, destination.parent))
    if output.exists():
        shutil.rmtree(output)
    staging.replace(output)
    return {
        "accepted": len(accepted),
        "uncertain": sum(item.get("status") == "UNCERTAIN" for item in records),
        "input_dir": str(output),
    }


def _run_stage2_structural_comparison(
    config, run_dir, spec, state, gpu, retry_failed=False
):
    """Predict Stage 2 designs, calculate S_on, and apply all on-target gates."""
    run_dir = Path(run_dir)
    comparison = spec["validation"]["candidate_screening"][
        "stage2_structural_comparison"
    ]
    predictor = comparison.get(
        "predictor", spec["workflow"]["complex_prediction"]
    )
    if predictor != "protenix":
        raise RuntimeError("Stage 2 on-target screening requires Protenix")
    tool_root = _stage_tool_root(run_dir, "stage2", predictor)
    work_root = _prediction_work_root(run_dir, "stage2", predictor)
    work_root.mkdir(parents=True, exist_ok=True)
    manifest_path, jobs = _build_final_candidate_manifest(
        run_dir, spec, manifest_path=work_root / "candidates.json"
    )

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["CHAI_DOWNLOADS_DIR"] = config["tools"]["chai"]["weights"]
    protenix_config = config["tools"]["protenix"]
    environment["PROTENIX_ROOT_DIR"] = protenix_config.get(
        "root_dir", str(Path(protenix_config["data_dir"]).parent)
    )
    project_root = Path(config["project_root"])
    prediction_dir = _stage_tool_output(run_dir, "stage2", predictor)
    prediction_inputs = {
        "manifest": str(manifest_path),
        "publish_root": str(tool_root),
        "num_diffn_samples": 1,
        "seed": int(comparison.get("seed", 101)),
        "need_atom_confidence": predictor == "protenix",
    }
    prediction_argv = build_command(
        predictor, config, prediction_inputs, prediction_dir, spec
    )
    _run_recoverable_command(
        state,
        "stage2_%s_prediction" % predictor,
        prediction_argv,
        config["tools"][predictor].get("path") or project_root,
        environment,
        run_dir / "logs" / ("stage2.%s.log" % predictor),
        retry_failed,
    )

    screening = spec["validation"]["candidate_screening"]
    thresholds = screening.get(
        "stage2_thresholds",
        {
            "antibody_plddt_min": 0.60,
            "ptm_min": 0.60,
            "iptm_global_min": 0.25,
            "design_pred_rmsd_max": 8.0,
            "design_pred_rmsd_max_inclusive": True,
            "s_on_min": 0.20,
        },
    )
    threshold_root = run_dir / "pipeline_output" / "stage2" / "ipsae"
    threshold_root.mkdir(parents=True, exist_ok=True)
    thresholds_path = threshold_root / "on_target_thresholds.json"
    write_json(thresholds_path, thresholds)
    output_dir = threshold_root / "results"
    evaluation_inputs = {
        "manifest": str(manifest_path),
        "prediction_dir": str(prediction_dir),
        "selected_dir": str(run_dir / "pipeline_output" / "stage2" / "filtered_on_target"),
        "stage1_candidates_dir": str(run_dir / "pipeline_output" / "stage2" / "on_target_candidates"),
        "delete_rejected": False,
        "thresholds_json": str(thresholds_path),
        "run_dir": str(run_dir),
        "hotspots": list(spec.get("target", {}).get("hotspots", [])),
        "pae_cutoff": 10,
        "dist_cutoff": 10,
    }
    compare_argv = build_command("ipsae", config, evaluation_inputs, output_dir, spec)
    _run_recoverable_command(
        state,
        "stage2_on_target_screen",
        compare_argv,
        project_root,
        environment,
        run_dir / "logs" / "stage2.on_target.ipsae.log",
        retry_failed,
    )
    records = _write_grouped_screen_table(
        output_dir,
        threshold_root / "on_target_screen.csv",
    )
    summary = read_json(output_dir / "summary.json")
    return {
        "manifest": str(manifest_path),
        "prediction_dir": str(prediction_dir),
        "output": str(output_dir),
        "candidate_count": len(jobs),
        "accepted": int(summary.get("accepted", 0)),
        "evaluated": len(records),
        "thresholds": thresholds,
        "zero_candidate_policy": "report_zero_keep_outputs_never_relax",
        "terminal_stage": False,
    }


def _run_final_functional_evaluation(config, run_dir, spec, state, gpu, retry_failed=False):
    run_dir = Path(run_dir)
    final_config = spec["validation"]["candidate_screening"].get(
        "final_functional_evaluation", {}
    )
    predictor = final_config.get("predictor", spec["workflow"]["complex_prediction"])
    if predictor not in ("chai", "protenix"):
        raise RuntimeError("final functional predictor must be chai or protenix")
    final_root = run_dir / "pipeline_output" / "final_evaluation"
    predictor_root = final_root / predictor
    manifest_path, jobs = _build_final_candidate_manifest(
        run_dir, spec, manifest_path=predictor_root / "candidates.json"
    )
    expected = int(spec["validation"]["candidate_screening"]["maturation"]["final_generation"]["candidates_per_original_backbone_group"])
    groups = {item["backbone_id"] for item in jobs}
    if len(jobs) != expected * len(groups):
        raise RuntimeError("final manifest count does not equal 100 candidates per active group")

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["CHAI_DOWNLOADS_DIR"] = config["tools"]["chai"]["weights"]
    protenix_config = config["tools"]["protenix"]
    environment["PROTENIX_ROOT_DIR"] = protenix_config.get(
        "root_dir", str(Path(protenix_config["data_dir"]).parent)
    )
    project_root = Path(config["project_root"])
    prediction_dir = predictor_root / "predictions"
    prediction_inputs = {
        "manifest": str(manifest_path),
        "num_diffn_samples": 1,
        "need_atom_confidence": predictor == "protenix",
    }
    prediction_argv = build_command(
        predictor, config, prediction_inputs, prediction_dir, spec
    )
    _run_recoverable_command(
        state, "final_functional_%s" % predictor, prediction_argv,
        config["tools"][predictor].get("path") or project_root, environment,
        run_dir / "logs" / ("final_functional.%s.log" % predictor), retry_failed,
    )

    metrics_dir = final_root / "ipsae" / "results"
    ipsae_argv = [
        _env_python(config, config["tools"]["ppiflow"]["env"]),
        str(project_root / "scripts" / "evaluate_final_candidates.py"),
        "--manifest", str(manifest_path),
        "--chai-dir", str(prediction_dir),
        "--ipsae-script", str(Path(config["tools"]["ipsae"]["path"]) / "ipsae.py"),
        "--output", str(metrics_dir),
    ]
    _run_recoverable_command(
        state, "final_functional_%s_ipsae" % predictor, ipsae_argv, project_root, environment,
        run_dir / "logs" / "final_functional.ipsae.log", retry_failed,
    )

    pyro_dir = final_root / "pyrosetta" / "results"
    pyro_argv = [
        _env_python(config, config["tools"]["pyrosetta"]["env"]),
        str(project_root / "scripts" / "score_interfaces_batch.py"),
        "--manifest", str(manifest_path), "--chai-dir", str(prediction_dir), "--output", str(pyro_dir),
    ]
    _run_recoverable_command(
        state, "final_functional_%s_pyrosetta" % predictor, pyro_argv, project_root, environment,
        run_dir / "logs" / "final_functional.pyrosetta.log", retry_failed,
    )

    structural = read_json(metrics_dir / "structural_interface_metrics.json")
    energy = read_json(pyro_dir / "pyrosetta_interface_metrics.json")
    energy_by_id = {item["candidate_id"]: item for item in energy}
    combined = []
    for item in structural:
        record = dict(item)
        record.update(energy_by_id.get(item["candidate_id"], {}))
        combined.append(record)
    missing_energy = [item["candidate_id"] for item in structural if item["candidate_id"] not in energy_by_id]
    if missing_energy:
        raise RuntimeError("missing PyRosetta metrics for %s final candidates" % len(missing_energy))
    gates = spec["validation"]["candidate_screening"]["final_functional_evaluation"]["hard_gates"]
    output_dir = run_dir / "final_functional_assessment"
    summary = write_functional_assessment(output_dir, combined, gates=gates)
    state.update(
        "final_functional_%s_ranking" % predictor,
        "complete",
        detail=str(output_dir / "functional_assessment.csv"),
    )
    return {"manifest": str(manifest_path), "output": str(output_dir), "summary": summary}


def _automatic_inputs(tool, run_dir, spec):
    run_dir = Path(run_dir)
    explicit = spec.get("adapter_inputs", {}).get(tool)
    if explicit:
        return dict(explicit)
    if tool == "rfdiffusion":
        return {"target": spec["target"]["structure"]}
    pdbs = _candidate_pdbs(run_dir)
    if tool == "proteinmpnn" and pdbs:
        chains = spec.get("generation", {}).get("binder_chain") or spec.get("framework", {}).get("heavy_chain", "A")
        return {"pdb": str(pdbs[0]), "chains": chains}
    if tool in ("protenix", "chai") and pdbs:
        input_dir = (
            _prediction_work_root(run_dir, "stage1", tool)
            if tool == "protenix"
            else _stage_tool_root(run_dir, "stage1", tool)
        )
        input_dir.mkdir(parents=True, exist_ok=True)
        publish_root = _stage_tool_root(run_dir, "stage1", tool)
        publish_root.mkdir(parents=True, exist_ok=True)
        screening = spec.get("validation", {}).get("candidate_screening", {})
        paths = _flowpacker_candidates(run_dir)
        selected_designs = _select_design_inputs(paths, screening)
        jobs = []
        for index, (backbone_id, design_id, pdb) in enumerate(selected_designs):
            sequence_map = _pdb_sequences(pdb)
            if sequence_map:
                sequences = [
                    {"chain": chain, "sequence": sequence}
                    for chain, sequence in sequence_map.items()
                ]
                design_chains, target_chain = _final_candidate_roles(spec, sequences)
                prediction_name = _prediction_name(backbone_id, design_id)
                jobs.append({
                    "candidate_id": "%s_%s" % (backbone_id, design_id),
                    "backbone_id": backbone_id,
                    "design_id": design_id,
                    "prediction_name": prediction_name,
                    "output_relpath": prediction_name,
                    "pdb": str(pdb),
                    "publish_design": False,
                    "sequences": sequences,
                    "design_chains": design_chains,
                    "target_chain": target_chain,
                    "designed_positions": _designed_positions(
                        run_dir,
                        pdb,
                        sequences,
                    ),
                })
        manifest = input_dir / "candidates.json"
        write_json(manifest, jobs)
        return {
            "manifest": str(manifest),
            "publish_root": str(publish_root),
            "num_diffn_samples": 1,
            "seed": 101,
            "target_chain": spec["target"]["chain"],
            "need_atom_confidence": tool == "protenix",
        }
    if tool == "pyrosetta" and pdbs:
        return {"structure": str(pdbs[0])}
    if tool == "rfdiffusion3" and pdbs:
        return {"mode": "predict", "spec": str(pdbs[0])}
    if tool == "ipsae":
        predictor = spec["workflow"]["complex_prediction"]
        prediction_manifest = _prefer_current_path(
            _prediction_work_root(run_dir, "stage1", predictor) / "candidates.json",
            _prefer_current_path(
                _stage_tool_root(run_dir, "stage1", predictor) / "candidates.json",
                run_dir / "adapter_inputs" / predictor / "candidates.json",
            ),
        )
        prediction_dir = _prefer_current_path(
            _stage_tool_output(run_dir, "stage1", predictor),
            run_dir / "adapter_output" / predictor,
        )
        if prediction_manifest.exists() and prediction_dir.exists():
            screening = spec.get("validation", {}).get("candidate_screening", {})
            threshold_dir = _stage_tool_root(run_dir, "stage1", "ipsae")
            threshold_dir.mkdir(parents=True, exist_ok=True)
            thresholds_path = threshold_dir / "loose_thresholds.json"
            write_json(
                thresholds_path,
                screening.get(
                    "stage1_thresholds",
                    {
                        "antibody_plddt_min": 0.60,
                        "ptm_min": 0.60,
                        "iptm_global_min": 0.25,
                        "design_pred_rmsd_max": 6.0,
                    },
                ),
            )
            return {
                "manifest": str(prediction_manifest),
                "prediction_dir": str(prediction_dir),
                "selected_dir": str(fastrelax_path(run_dir / "pipeline_output" / "stage1" / "fastrelax")),
                "stage1_candidates_dir": str(run_dir / "pipeline_output" / "stage1" / "stage1_candidates"),
                "mode": screening.get("mode", "loose_filter_only"),
                "delete_rejected": bool(screening.get("delete_rejected", False)),
                "thresholds_json": str(thresholds_path),
                "run_dir": str(run_dir),
                "hotspots": list(spec.get("target", {}).get("hotspots", [])),
                "pae_cutoff": 10,
                "dist_cutoff": 10,
                "skip_ipsae": True,
            }
        confidence = sorted((run_dir / "pipeline_output").rglob("*full_data*.json"))
        structures = sorted((run_dir / "pipeline_output").rglob("*.cif")) + pdbs
        if confidence and structures:
            return {"confidence": str(confidence[0]), "structure": str(structures[0]), "pae_cutoff": 10, "dist_cutoff": 10}
    return None


def _stage1_adaptive_prediction_plan(spec, jobs):
    """Split each Stage 1 backbone into the configured first and second waves."""
    screening = spec.get("validation", {}).get("candidate_screening", {})
    policy = dict(screening.get("stage1_adaptive_prediction", {}))
    if not policy.get("enabled", False):
        return None
    first_count = int(policy.get("first_batch_per_backbone", 4))
    second_count = int(policy.get("second_batch_per_backbone", 4))
    grouped = {}
    for job in jobs:
        grouped.setdefault(job["backbone_id"], []).append(job)
    first_wave = []
    second_wave = {}
    for backbone_id in sorted(grouped):
        ordered = sorted(
            grouped[backbone_id],
            key=lambda item: (
                int(re.search(r"(\d+)$", item["design_id"]).group(1)),
                item["candidate_id"],
            ),
        )
        first_wave.extend(ordered[:first_count])
        second_wave[backbone_id] = ordered[
            first_count : first_count + second_count
        ]
    if not first_wave or not any(second_wave.values()):
        return None
    return {
        "policy": policy,
        "first_wave": first_wave,
        "second_wave": second_wave,
        "backbone_count": len(grouped),
    }


def _run_stage1_adaptive_protenix(
    config,
    run_dir,
    spec,
    state,
    inputs,
    output,
    environment,
    retry_failed=False,
):
    """Run Stage 1 Protenix as a conservative 4+4 backbone-grouped schedule."""
    run_dir = Path(run_dir)
    canonical_manifest = Path(inputs["manifest"])
    all_jobs = read_json(canonical_manifest)
    plan = _stage1_adaptive_prediction_plan(spec, all_jobs)
    if plan is None:
        return None

    work_root = _prediction_work_root(
        run_dir, "stage1", "protenix"
    ) / "adaptive_prediction"
    work_root.mkdir(parents=True, exist_ok=True)
    all_manifest = work_root / "all_candidates_before_4_plus_4.json"
    write_json(all_manifest, all_jobs)
    first_manifest = work_root / "wave1_candidates.json"
    write_json(first_manifest, plan["first_wave"])

    first_inputs = dict(inputs)
    first_inputs["manifest"] = str(first_manifest)
    first_argv = build_command("protenix", config, first_inputs, output, spec)
    _run_recoverable_command(
        state,
        "stage1_protenix_wave1",
        first_argv,
        config["tools"]["protenix"].get("path") or config["project_root"],
        environment,
        run_dir / "logs" / "stage1.protenix.wave1.log",
        retry_failed,
    )

    screening = spec["validation"]["candidate_screening"]
    continuation_thresholds = dict(screening["stage1_thresholds"])
    continuation_thresholds["design_pred_rmsd_max"] = float(
        plan["policy"].get("continuation_design_pred_rmsd_max", 10.0)
    )
    # The continuation gate is strictly below 10 A; this is not a final gate.
    continuation_thresholds["design_pred_rmsd_max_inclusive"] = False
    thresholds_path = work_root / "continuation_thresholds.json"
    write_json(thresholds_path, continuation_thresholds)
    pilot_results = work_root / "wave1_screen" / "results"
    pilot_inputs = {
        "manifest": str(first_manifest),
        "prediction_dir": str(output),
        "selected_dir": str(work_root / "wave1_screen" / "selected"),
        "stage1_candidates_dir": str(
            work_root / "wave1_screen" / "candidate_groups"
        ),
        "delete_rejected": False,
        "thresholds_json": str(thresholds_path),
        "run_dir": str(run_dir),
        "hotspots": list(spec.get("target", {}).get("hotspots", [])),
        "pae_cutoff": 10,
        "dist_cutoff": 10,
        "skip_ipsae": True,
    }
    pilot_argv = build_command(
        "ipsae", config, pilot_inputs, pilot_results, spec
    )
    _run_recoverable_command(
        state,
        "stage1_protenix_wave1_continuation_gate",
        pilot_argv,
        config["project_root"],
        environment,
        run_dir / "logs" / "stage1.protenix.wave1_gate.log",
        retry_failed,
    )

    pilot_summary = read_json(pilot_results / "summary.json")
    eligible_backbones = {
        backbone_id
        for backbone_id, group in pilot_summary.get("groups", {}).items()
        if int(group.get("accepted", 0)) > 0
    }
    second_jobs = []
    for backbone_id in sorted(eligible_backbones):
        second_jobs.extend(plan["second_wave"].get(backbone_id, []))
    write_json(work_root / "wave2_candidates.json", second_jobs)
    final_jobs = list(plan["first_wave"]) + second_jobs
    write_json(canonical_manifest, final_jobs)

    second_inputs = dict(inputs)
    second_inputs["manifest"] = str(canonical_manifest)
    second_argv = build_command("protenix", config, second_inputs, output, spec)
    _run_recoverable_command(
        state,
        "stage1_protenix_wave2",
        second_argv,
        config["tools"]["protenix"].get("path") or config["project_root"],
        environment,
        run_dir / "logs" / "stage1.protenix.wave2.log",
        retry_failed,
    )

    audit = {
        "mode": "stage1_backbone_grouped_4_plus_4",
        "first_batch_per_backbone": int(
            plan["policy"].get("first_batch_per_backbone", 4)
        ),
        "second_batch_per_backbone": int(
            plan["policy"].get("second_batch_per_backbone", 4)
        ),
        "continuation_gate": {
            "design_pred_rmsd_max_exclusive": continuation_thresholds[
                "design_pred_rmsd_max"
            ],
            "other_stage1_thresholds_unchanged": True,
            "final_release_gate_unchanged": True,
        },
        "backbone_groups": plan["backbone_count"],
        "wave1_candidates": len(plan["first_wave"]),
        "continuation_backbone_groups": len(eligible_backbones),
        "wave2_candidates": len(second_jobs),
        "final_predictions": len(final_jobs),
        "random_rescue_enabled": False,
        "eligible_backbones": sorted(eligible_backbones),
        "all_candidates_manifest": str(all_manifest),
        "wave1_manifest": str(first_manifest),
        "wave2_manifest": str(work_root / "wave2_candidates.json"),
        "canonical_manifest": str(canonical_manifest),
    }
    write_json(work_root / "adaptive_prediction_manifest.json", audit)
    return audit


def _selected_external_tools(spec):
    values = []
    for stage in STAGES:
        choice = spec["workflow"][stage]
        tool = {"ppiflow_mpnn":"ppiflow", "ppiflow_flowpacker":"ppiflow", "ppiflow_af3score":"ppiflow"}.get(choice, choice)
        if tool != "ppiflow" and tool not in values:
            values.append(tool)
    return values


def _iterative_redesign_enabled(spec):
    screening = spec.get("validation", {}).get("candidate_screening", {})
    return spec.get("generation", {}).get("profile") == "full" and screening.get("iterative_redesign", True)


def _collect_metrics(run_dir):
    records = {}
    for path in Path(run_dir).rglob("*.json"):
        if path.name in {"spec.json", "manifest.json", "status.json", "orchestration_state.json", "candidate_ranking.json"}:
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        mappings = value if isinstance(value, list) else [value]
        for item in mappings:
            if not isinstance(item, dict):
                continue
            metric = {}
            for key, raw in item.items():
                lowered = str(key).lower()
                if not isinstance(raw, (int, float)):
                    continue
                if lowered == "s_on":
                    metric["s_on"] = raw
                    metric["ipsae"] = raw
                elif "ipsae" in lowered:
                    metric["ipsae"] = raw
                elif "scrmsd" in lowered:
                    metric["scrmsd"] = raw
                elif "iptm" in lowered:
                    metric["iptm"] = raw
                elif "plddt" in lowered:
                    metric["plddt"] = raw
                elif lowered == "ptm" or lowered.endswith("_ptm"):
                    metric["ptm"] = raw
                elif "dockq" in lowered:
                    metric["dockq"] = raw
                elif "interface_energy" in lowered:
                    metric["interface_energy"] = raw
                elif "pae" in lowered:
                    metric["pae"] = raw
            if metric:
                candidate_id = item.get("candidate_id") or item.get("name") or path.stem
                record = records.setdefault(candidate_id, {"candidate_id": candidate_id, "sources": []})
                record.update(metric)
                record["sources"].append(str(path.relative_to(run_dir)))
                for key in ("backbone_id", "design_id", "threshold_class", "threshold_tier", "selected_top_k", "source_pdb", "original_source_pdb", "predicted_cif"):
                    if key in item:
                        record[key] = item[key]
                if "consensus_score" in item and "all_thresholds" in path.name:
                    record["threshold_consensus_score"] = item["consensus_score"]
    return list(records.values())


def _automatic_iggm_enabled(config, spec):
    tool = config.get("tools", {}).get("iggm", {})
    automatic = tool.get("automatic", {})
    return bool(
        automatic.get("enabled", False)
        and spec.get("design_type") == "antibody"
        and spec.get("framework", {}).get("light_chain")
    )


def _automatic_stage3_enabled(config, spec):
    screening = (
        spec.get("validation", {})
        .get("candidate_screening", {})
        .get("stage3_iggm_specificity", {})
    )
    automatic = config.get("tools", {}).get("iggm", {}).get("automatic", {})
    return bool(
        _automatic_iggm_enabled(config, spec)
        and screening.get("enabled", True)
        and automatic.get("stage3_specificity_screening", True)
    )


def _run_automatic_iggm(config, run_id, spec, state, manifest, gpu, retry_failed=False):
    stage_name = "stage2_iggm_affinity_maturation"
    status = state.stage(stage_name).get("status")
    output = (
        Path(config["storage"]["runs_root"])
        / run_id
        / "pipeline_output"
        / "stage2"
        / "09_IgGM"
    )
    batch_manifest = output / "batch_manifest.json"
    if status == "complete" and batch_manifest.is_file():
        result = read_json(batch_manifest)
        manifest["steps"].append(
            {
                "stage": stage_name,
                "state": "complete",
                "automatic": True,
                "candidate_count": result.get("candidate_count", 0),
                "output": str(output),
                "reused": True,
            }
        )
        return result
    if status == "failed" and not retry_failed:
        raise RuntimeError("automatic IgGM maturation failed; resume with --retry-failed")
    if status == "running":
        raise RuntimeError("automatic IgGM maturation is already running")
    entry = {
        "stage": stage_name,
        "state": "running",
        "automatic": True,
        "input": "Stage 2 on-target and off-target passing FastRelax candidates",
    }
    manifest["steps"].append(entry)
    state.update(stage_name, "running", detail=str(output))
    try:
        result = run_stage2_affinity_maturation(config, run_id, spec, gpu=gpu)
    except Exception as exc:
        entry.update({"state": "failed", "detail": str(exc)})
        state.update(stage_name, "failed", detail=str(exc))
        write_json(Path(config["storage"]["runs_root"]) / run_id / "execution_manifest.json", manifest)
        raise
    entry.update(
        {
            "state": "complete",
            "candidate_count": result["candidate_count"],
            "output": str(output),
        }
    )
    state.update(stage_name, "complete", detail=str(output))
    return result


def _run_stage3_iggm_specificity(
    config, run_id, spec, state, manifest, gpu, retry_failed=False
):
    """Pack unique IgGM sequences, predict both targets, and pair scores."""
    stage_name = "stage3_iggm_specificity_screening"
    run_dir = _run_dir(config, run_id)
    stage3 = run_dir / "pipeline_output" / "stage3"
    completion = stage3 / "00_stage3_plan" / "completion_manifest.json"
    if state.stage(stage_name).get("status") == "complete" and completion.is_file():
        result = read_json(completion)
        manifest["steps"].append(
            {
                "stage": stage_name,
                "state": "complete",
                "automatic": True,
                "output": str(stage3),
                "reused": True,
                **result,
            }
        )
        return result
    if state.stage(stage_name).get("status") == "failed" and not retry_failed:
        raise RuntimeError("Stage 3 IgGM specificity screening failed; resume with --retry-failed")
    entry = {
        "stage": stage_name,
        "state": "running",
        "automatic": True,
        "input": "Stage 2 09_IgGM FASTA variants",
        "on_target": "CLDN18.2",
        "off_target": "CLDN18.1",
    }
    manifest["steps"].append(entry)
    state.update(stage_name, "running", detail=str(stage3))
    try:
        manifest_path, jobs = build_stage3_manifest(run_dir, spec)
        screening = spec["validation"]["candidate_screening"]
        stage3_config = screening.get("stage3_iggm_specificity", {})
        predictor = stage3_config.get("predictor", "protenix")
        if predictor != "protenix":
            raise RuntimeError("Stage 3 IgGM specificity screening requires Protenix")
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["CHAI_DOWNLOADS_DIR"] = config["tools"]["chai"]["weights"]
        protenix_config = config["tools"]["protenix"]
        environment["PROTENIX_ROOT_DIR"] = protenix_config.get(
            "root_dir", str(Path(protenix_config["data_dir"]).parent)
        )
        project_root = Path(config["project_root"])
        ppiflow_root = Path(config["tools"]["ppiflow"]["path"])
        flowpacker_root = ppiflow_root / "tools" / "flowpacker"
        flowpacker_work = run_dir / ".agent_work" / "stage3" / "flowpacker"
        flowpacker_argv = [
            _env_python(config, config["tools"]["ppiflow"]["env"]),
            str(project_root / "scripts" / "stage3_flowpacker_adapter.py"),
            "--manifest", str(manifest_path),
            "--output", str(flowpacker_work),
            "--publish-root", str(
                stage3 / "02_flowpacker_sidechain_packing"
            ),
            "--sampler", str(flowpacker_root / "sampler_pdb_antibody.py"),
            "--config", str(flowpacker_root / "config" / "inference" / "base.yaml"),
            "--checkpoint", str(flowpacker_root / "checkpoints" / "cluster.pth"),
            "--heavy-chain", str(spec["framework"]["heavy_chain"]),
            "--light-chain", str(spec["framework"].get("light_chain", "")),
            "--seed", str(int(stage3_config.get("flowpacker_seed", 42))),
        ]
        _run_recoverable_command(
            state,
            "stage3_flowpacker_sidechain_packing",
            flowpacker_argv,
            project_root,
            environment,
            run_dir / "logs" / "stage3.flowpacker.log",
            retry_failed,
        )
        manifest_path, jobs = attach_stage3_flowpacker_outputs(
            run_dir, jobs, flowpacker_work / "packed"
        )

        prediction_dir = _stage_tool_output(run_dir, "stage3", predictor)
        prediction_argv = build_command(
            predictor,
            config,
            {
                "manifest": str(manifest_path),
                "publish_root": str(_stage_tool_root(run_dir, "stage3", predictor)),
                "seed": int(stage3_config.get("seed", 101)),
                "need_atom_confidence": True,
                "allow_partial_failures": False,
            },
            prediction_dir,
            spec,
        )
        _run_recoverable_command(
            state,
            "stage3_cldn18_2_protenix",
            prediction_argv,
            config["tools"][predictor].get("path") or project_root,
            environment,
            run_dir / "logs" / "stage3.cldn18_2.protenix.log",
            retry_failed,
        )

        thresholds = dict(screening.get("stage2_thresholds", {}))
        thresholds.update(stage3_config.get("thresholds", {}))
        thresholds_path = stage3 / "00_stage3_plan" / "on_target_thresholds.json"
        write_json(thresholds_path, thresholds)
        on_target_root = _stage_tool_root(run_dir, "stage3", "ipsae")
        on_target_results = _stage_tool_output(run_dir, "stage3", "ipsae")
        compare_argv = build_command(
            "ipsae",
            config,
            {
                "manifest": str(manifest_path),
                "prediction_dir": str(prediction_dir),
                "selected_dir": str(run_dir / ".agent_work" / "stage3" / "selected_on_target"),
                "stage1_candidates_dir": str(run_dir / ".agent_work" / "stage3" / "on_target_groups"),
                "delete_rejected": False,
                "thresholds_json": str(thresholds_path),
                "run_dir": str(run_dir),
                "hotspots": list(spec.get("target", {}).get("hotspots", [])),
                "pae_cutoff": float(stage3_config.get("pae_cutoff", 10)),
                "dist_cutoff": float(stage3_config.get("dist_cutoff", 10)),
            },
            on_target_results,
            spec,
        )
        _run_recoverable_command(
            state,
            "stage3_cldn18_2_on_target_screen",
            compare_argv,
            project_root,
            environment,
            run_dir / "logs" / "stage3.cldn18_2.ipsae.log",
            retry_failed,
        )
        on_records = _write_grouped_screen_table(
            on_target_results, on_target_root / "on_target_screen.csv"
        )
        on_summary = read_json(on_target_results / "summary.json")
        if int(on_summary.get("accepted", 0)) == 0:
            result = {
                "stage": stage_name,
                "evaluated": len(on_records),
                "passed": 0,
                "status": "complete_no_stage3_on_target_candidates",
                "threshold_auto_relaxed": False,
                "output": str(stage3),
            }
            (
                stage3 / "07_final_candidates" / "final_candidates.csv"
            ).write_text(
                "candidate_id,mutation,S_on,S_off,delta,design_pred_rmsd,antibody_pred_rmsd,target_pred_rmsd,design_pdb,directory,antibody_sequences\n", encoding="utf-8"
            )
            write_json(completion, result)
        else:
            off_result = _run_offtarget_screen(
                config,
                run_dir,
                spec,
                state,
                gpu,
                stage="stage3",
                retry_failed=retry_failed,
            )
            result = publish_stage3_results(
                run_dir, jobs, Path(off_result["output"])
            )
            result["output"] = str(stage3)
        entry.update({"state": "complete", **result})
        state.update(stage_name, "complete", detail=str(stage3))
        return result
    except Exception as exc:
        entry.update({"state": "failed", "detail": str(exc)})
        state.update(stage_name, "failed", detail=str(exc))
        write_json(run_dir / "execution_manifest.json", manifest)
        raise


def run_orchestration(config, run_id, gpu="auto", max_retries=1, dry_run=False, retry_failed=False):
    run_dir = _run_dir(config, run_id)
    spec = read_json(run_dir / "spec.json")
    state = RecoveryState(run_dir)
    completed_manifest = run_dir / "execution_manifest.json"
    if (
        not dry_run
        and state.stage("stage2_fastrelax").get("status") == "complete"
        and completed_manifest.is_file()
    ):
        previous = read_json(completed_manifest)
        if previous.get("status") == "complete":
            if not _automatic_iggm_enabled(config, spec):
                return previous
            iggm_complete = (
                state.stage("stage2_iggm_affinity_maturation").get("status")
                == "complete"
            )
            stage3_enabled = _automatic_stage3_enabled(config, spec)
            stage3_complete = (
                state.stage("stage3_iggm_specificity_screening").get("status")
                == "complete"
            )
            if iggm_complete and (not stage3_enabled or stage3_complete):
                return previous
            resume_gpu = select_gpu() if gpu == "auto" else int(str(gpu).split(",", 1)[0].strip())
            previous["steps"] = [
                item
                for item in previous.get("steps", [])
                if item.get("stage")
                not in {
                    "stage2_iggm_affinity_maturation",
                    "stage3_iggm_specificity_screening",
                }
            ]
            if not iggm_complete:
                _run_automatic_iggm(
                    config,
                    run_id,
                    spec,
                    state,
                    previous,
                    resume_gpu,
                    retry_failed=retry_failed,
                )
            if stage3_enabled:
                _run_stage3_iggm_specificity(
                    config,
                    run_id,
                    spec,
                    state,
                    previous,
                    resume_gpu,
                    retry_failed=retry_failed,
                )
            previous["status"] = "complete"
            write_json(completed_manifest, previous)
            return previous
    ppiflow_gpu = gpu
    if not dry_run:
        if gpu == "auto":
            gpu = select_gpu()
        else:
            # PPIFlow accepts an explicit comma-separated shard list.  All
            # downstream adapters remain single-GPU and use its first id.
            gpu = int(str(gpu).split(",", 1)[0].strip())
    manifest = {"run_id": run_id, "dry_run": bool(dry_run), "steps": []}
    # Historical runs do not contain this opt-in field and are deliberately
    # left untouched. Newly created/validated specs materialize the field.
    canonical_outputs_enabled = bool(
        spec.get("output_naming", {}).get("enabled", False)
    )
    primary = spec["workflow"]["backbone_generation"]
    if primary == "ppiflow":
        primary_entry = {"stage": "ppiflow_primary", "adapter": "ppiflow", "state": "native"}
        manifest["steps"].append(primary_entry)
        screen_root = run_dir / "pipeline_output" / "stage1" / "backbone_screen"
        request_path = screen_root / "selection_request.json"
        selection_path = screen_root / "backbone_selection.json"
        if request_path.exists() and not selection_path.exists():
            action = read_json(request_path)
            primary_entry.update({"state": "awaiting_user_input", "action_required": action})
            manifest["status"] = "awaiting_user_input"
            manifest["action_required"] = action
            if not dry_run:
                state.update("ppiflow_primary", "awaiting_user_input", detail=action["prompt"])
            write_json(run_dir / "execution_manifest.json", manifest)
            return manifest
        if not dry_run and state.should_run("ppiflow_primary", retry_failed=retry_failed):
            state.update("ppiflow_primary", "running")
            try:
                primary_result = execute_run(
                    config,
                    run_id,
                    stage=1 if _iterative_redesign_enabled(spec) else None,
                    gpu=ppiflow_gpu,
                )
                if primary_result.get("status") == "awaiting_user_input":
                    action = primary_result["action_required"]
                    primary_entry.update(
                        {"state": "awaiting_user_input", "action_required": action}
                    )
                    manifest["status"] = "awaiting_user_input"
                    manifest["action_required"] = action
                    state.update(
                        "ppiflow_primary",
                        "awaiting_user_input",
                        detail=action["prompt"],
                    )
                    write_json(run_dir / "execution_manifest.json", manifest)
                    return manifest
                state.update("ppiflow_primary", "complete")
                primary_entry["state"] = "complete"
            except Exception as exc:
                state.update("ppiflow_primary", "failed", detail=str(exc))
                raise
    external = _selected_external_tools(spec)
    for tool in external:
        if not dry_run and not state.should_run(tool, retry_failed=retry_failed):
            manifest["steps"].append(
                {"stage": tool, "adapter": tool, "state": "complete_reused"}
            )
            continue
        inputs = _automatic_inputs(tool, run_dir, spec)
        entry = {"stage": tool, "adapter": tool, "inputs": inputs}
        if not inputs:
            entry["state"] = "waiting_for_inputs"
            manifest["steps"].append(entry)
            if not dry_run:
                state.update(tool, "failed", detail="no compatible upstream artifacts or adapter_inputs")
                raise RuntimeError("%s adapter has no compatible inputs" % tool)
            continue
        output = _stage_tool_output(run_dir, "stage1", tool)
        argv = build_command(tool, config, inputs, output, spec)
        entry.update({"state": "ready", "argv": argv, "output": str(output)})
        manifest["steps"].append(entry)
        if dry_run:
            continue
        output.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["CHAI_DOWNLOADS_DIR"] = config["tools"]["chai"]["weights"]
        environment["FOUNDRY_CHECKPOINT_DIRS"] = config["tools"]["rfdiffusion3"]["weights"]
        protenix_config = config["tools"]["protenix"]
        protenix_root = protenix_config.get("root_dir")
        if not protenix_root:
            data_dir = Path(protenix_config["data_dir"])
            protenix_root = str(data_dir.parent if data_dir.name == "common" else data_dir)
        environment["PROTENIX_ROOT_DIR"] = protenix_root
        log_path = run_dir / "logs" / ("adapter.%s.log" % tool)
        if tool == "protenix":
            try:
                state.update(
                    tool,
                    "running",
                    detail="Stage 1 adaptive 4+4 prediction",
                )
                adaptive_result = _run_stage1_adaptive_protenix(
                    config,
                    run_dir,
                    spec,
                    state,
                    inputs,
                    output,
                    environment,
                    retry_failed=retry_failed,
                )
                if adaptive_result is not None:
                    entry["adaptive_prediction"] = adaptive_result
                    entry["state"] = "complete"
                    state.update(tool, "complete", returncode=0)
                    continue
            except Exception as exc:
                state.update(tool, "failed", detail=str(exc))
                raise
        for attempt in range(int(max_retries) + 1):
            state.update(tool, "running", argv=argv, log=str(log_path))
            with log_path.open("a", encoding="utf-8") as log:
                completed = subprocess.run(argv, cwd=config["tools"][tool].get("path") or config["project_root"], env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
            if completed.returncode == 0:
                state.update(tool, "complete", returncode=0)
                break
            state.update(tool, "failed", detail="exit code %s" % completed.returncode, returncode=completed.returncode)
            if attempt >= int(max_retries):
                raise RuntimeError("%s failed after %s attempts; see %s" % (tool, attempt + 1, log_path))
    if not dry_run and canonical_outputs_enabled:
        round0_summary_path = (
            run_dir / "pipeline_output" / "stage1" / "screening_tables" / "summary.json"
        )
        finalized_summary = _finalized_round0_summary(run_dir)
        if finalized_summary is not None:
            round0_outputs = finalized_summary
        elif round0_summary_path.exists():
            round0_outputs = read_json(round0_summary_path)
        else:
            round0_outputs = materialize_round_outputs(
                run_dir, spec, round_index=0
            )
        manifest["steps"].append(
            {
                "stage": "stage1_screening_tables",
                "state": "complete",
                "summary": round0_outputs,
            }
        )
        state.update(
            "stage1_screening_tables",
            "complete",
            detail=round0_outputs["output_root"],
        )
    write_json(run_dir / "execution_manifest.json", manifest)
    screening = spec.get("validation", {}).get("candidate_screening", {})
    if screening.get("mode", "loose_filter_only") == "loose_filter_only":
        manifest["ranking"] = {"skipped": True, "reason": "filter-only mode has no candidate ranking"}
    else:
        metrics = _collect_metrics(run_dir)
        manifest["ranking"] = write_ranking(run_dir, metrics)
    off_config = _offtarget_config(config, spec)
    # Historical compatibility only. New two-stage runs defer all S_on/S_off
    # and CLDN18.1 work until Stage 2.
    if screening.get("legacy_stage1_offtarget", False) and off_config.get("enabled", True):
        off_entry = {
            "stage": "stage1_round2_cldn18_1_offtarget_screen",
            "state": "planned" if dry_run else "ready",
            "predictor": "protenix",
            "target": off_config.get("name", "CLDN18.1"),
            "seed": int(off_config.get("seed", 101)),
            "candidate_source": "Stage 1 accepted candidate_id values",
            "reuse_stage1_S_on": True,
            "backbone_regenerated": False,
            "antibody_redesign": False,
            "terminal_stage": True,
        }
        manifest["steps"].append(off_entry)
        if dry_run:
            write_json(run_dir / "execution_manifest.json", manifest)
            return manifest
        try:
            result = _run_offtarget_screen(
                config,
                run_dir,
                spec,
                state,
                gpu,
                stage="stage1",
                retry_failed=retry_failed,
            )
            off_entry.update({"state": "complete", **result})
            state.update(
                "stage1_round2_cldn18_1_offtarget_screen",
                "complete",
                detail=result["output"],
            )
        except Exception as exc:
            off_entry.update({"state": "failed", "detail": str(exc)})
            state.update(
                "stage1_round2_cldn18_1_offtarget_screen", "failed", detail=str(exc)
            )
            write_json(run_dir / "execution_manifest.json", manifest)
            raise
        if canonical_outputs_enabled:
            manifest["stage1_layout"] = finalize_stage1_layout(run_dir)
            manifest["result_paths"] = write_result_paths(run_dir)
        else:
            manifest["screening_summary"] = write_screening_summary(run_dir)
        layout = manifest.get("stage1_layout", {})
        if layout.get("rosettafix_status") == "awaiting_manual_fix":
            preparation = prepare_stage1_rosetta_inputs(run_dir)
            result = run_fastrelax(
                config, run_dir.name, stage=1, finalize_layout=False
            )
            manifest["steps"].append(
                {
                    "stage": "stage1_grouped_interface_score",
                    "state": "complete",
                    "automatic": True,
                    "coordinate_policy": "score_only_preserve_input_coordinates",
                    "grouping_key": "backbone_id",
                    "cross_group_merge": False,
                    "group_count": preparation["group_count"],
                    "candidate_count": preparation["candidate_count"],
                    **{key: value for key, value in result.items() if key != "stage"},
                }
            )
            if canonical_outputs_enabled:
                manifest["stage1_layout"] = finalize_stage1_layout(run_dir)
                manifest["result_paths"] = write_result_paths(run_dir)
        manifest["status"] = "complete"
        write_json(run_dir / "execution_manifest.json", manifest)
        return manifest
    if _iterative_redesign_enabled(spec):
        stage1_summary = _prefer_current_path(
            _stage_tool_output(run_dir, "stage1", "ipsae") / "summary.json",
            run_dir / "adapter_output" / "ipsae" / "summary.json",
        )
        stage1_accepted = -1 if dry_run else (
            int(read_json(stage1_summary).get("accepted", 0))
            if stage1_summary.exists()
            else int((_finalized_round0_summary(run_dir) or {}).get("accepted_for_partialflow", 0))
        )
        stage1_rosetta = fastrelax_path(run_dir / "pipeline_output" / "stage1" / "fastrelax_output")
        stage1_rosetta_legacy = fastrelax_path(run_dir / "pipeline_output" / "stage1" / "fastrelax_raw_output")
        if not dry_run and stage1_accepted == 0:
            manifest["status"] = "complete_no_stage1_candidates"
            manifest["zero_candidate_policy"] = "report_zero_keep_outputs_never_relax"
            write_json(run_dir / "execution_manifest.json", manifest)
            return manifest
        if not dry_run and not stage1_rosetta.exists() and not stage1_rosetta_legacy.exists():
            preparation = prepare_stage1_rosetta_inputs(run_dir)
            rosetta_entry = {
                "stage": "stage1_grouped_interface_score",
                "state": "running",
                "automatic": True,
                "coordinate_policy": "score_only_preserve_input_coordinates",
                "grouping_key": "backbone_id",
                "cross_group_merge": False,
                "group_count": preparation["group_count"],
                "candidate_count": preparation["candidate_count"],
                "group_manifest": preparation["manifest"],
            }
            manifest["steps"].append(rosetta_entry)
            try:
                rosetta_result = run_fastrelax(
                    config, run_id, stage=1, finalize_layout=False
                )
            except Exception as exc:
                rosetta_entry.update({"state": "failed", "detail": str(exc)})
                write_json(run_dir / "execution_manifest.json", manifest)
                raise
            rosetta_entry.update(
                {
                    "state": "complete",
                    **{
                        key: value
                        for key, value in rosetta_result.items()
                        if key != "stage"
                    },
                }
            )
            manifest["stage1_status"] = "complete"
            manifest["stage1_fixed_positions"] = str(
                run_dir / "pipeline_output" / "stage1" / "fixed_positions.csv"
            )
        stage_name = "ppiflow_iterative_redesign"
        entry = {"stage": stage_name, "adapter": "ppiflow", "state": "ready", "stage_number": 2}
        manifest["steps"].append(entry)
        filter_summary = _prefer_current_path(
            _stage_tool_output(run_dir, "stage1", "ipsae") / "summary.json",
            run_dir / "adapter_output" / "ipsae" / "summary.json",
        )
        if screening.get("mode", "loose_filter_only") == "loose_filter_only" and filter_summary.exists():
            accepted = int(read_json(filter_summary).get("accepted", 0))
            if accepted == 0:
                entry.update({"state": "skipped_no_candidates", "accepted_stage1": 0})
                state.update(stage_name, "complete", detail="no candidates passed the loose Stage 1 filter")
                if canonical_outputs_enabled:
                    manifest["screening_summary"] = write_screening_summary(run_dir)
                    manifest["result_paths"] = write_result_paths(run_dir)
                write_json(run_dir / "execution_manifest.json", manifest)
                return manifest
        groups_path = run_dir / "pipeline_output" / "stage1" / "stage1_candidates" / "stage1_groups.json"
        if groups_path.exists():
            groups = read_json(groups_path)
        else:
            compact_root = (
                run_dir / "pipeline_output" / "stage1" / "canonical_round0"
            )
            groups = {
                path.name: {
                    "accepted": len(list(path.glob("*.pdb"))),
                    "directory": str(path),
                }
                for path in sorted(compact_root.glob("backbone.0_*"))
                if path.is_dir()
            }
        maturation_plan = build_maturation_plan(spec, groups)
        maturation_path = run_dir / "pipeline_output" / "stage2" / "maturation_plan.json"
        maturation_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(maturation_path, maturation_plan)
        entry["maturation_plan"] = str(maturation_path)
        entry["independent_backbone_groups"] = maturation_plan["group_count"]
        if not dry_run and state.should_run(stage_name, retry_failed=retry_failed):
            state.update(stage_name, "running")
            try:
                execute_run(config, run_id, stage=2, gpu=gpu)
                state.update(stage_name, "complete")
                entry["state"] = "complete"
            except Exception as exc:
                state.update(stage_name, "failed", detail=str(exc))
                raise
        comparison_config = screening.get("stage2_structural_comparison", {})
        if comparison_config.get("enabled", True):
            comparison_entry = {
                "stage": "stage2_on_target_screen",
                "state": "planned" if dry_run else "ready",
                "predictor": comparison_config.get(
                    "predictor", spec["workflow"]["complex_prediction"]
                ),
                "ranking": False,
                "terminal_stage": False,
            }
            manifest["steps"].append(comparison_entry)
            if not dry_run:
                try:
                    result = _run_stage2_structural_comparison(
                        config,
                        run_dir,
                        spec,
                        state,
                        gpu,
                        retry_failed=retry_failed,
                    )
                    comparison_entry.update({"state": "complete", **result})
                except Exception as exc:
                    comparison_entry.update({"state": "failed", "detail": str(exc)})
                    state.update(
                        "stage2_on_target_screen", "failed", detail=str(exc)
                    )
                    write_json(run_dir / "execution_manifest.json", manifest)
                    raise
        if not dry_run and canonical_outputs_enabled:
            round1_outputs = _finalized_round1_summary(run_dir)
            if round1_outputs is None:
                round1_outputs = materialize_round_outputs(
                    run_dir, spec, round_index=1
                )
            manifest["steps"].append(
                {
                    "stage": "stage2_screening_tables",
                    "state": "complete",
                    "summary": round1_outputs,
                }
            )
            state.update(
                "stage2_screening_tables",
                "complete",
                detail=round1_outputs["output_root"],
            )
        if not dry_run and comparison_config.get("enabled", True):
            _, stage2_accepted = _accepted_stage_metrics(run_dir, "stage2")
            if not stage2_accepted:
                manifest["status"] = "complete_no_stage2_on_target_candidates"
                manifest["zero_candidate_policy"] = {
                    "S_on_min": float(
                        spec["validation"]["candidate_screening"][
                            "stage2_thresholds"
                        ]["s_on_min"]
                    ),
                    "reported_passed": 0,
                    "threshold_auto_relaxed": False,
                    "outputs_preserved_for_manual_review": True,
                }
                if canonical_outputs_enabled:
                    manifest["stage2_layout"] = finalize_stage2_layout(
                        run_dir,
                        expected_sequences=int(
                            spec["generation"]["stage2_sequences_per_backbone"]
                        ),
                    )
                    manifest["result_paths"] = write_result_paths(run_dir)
                write_json(run_dir / "execution_manifest.json", manifest)
                return manifest

            if off_config.get("enabled", True):
                off_entry = {
                    "stage": "stage2_cldn18_1_offtarget_screen",
                    "state": "ready",
                    "predictor": "protenix",
                    "target": off_config.get("name", "CLDN18.1"),
                    "seed": int(off_config.get("seed", 101)),
                    "candidate_source": "Stage 2 S_on-passing candidate_id values",
                    "reuse_stage2_S_on": True,
                    "backbone_regenerated": False,
                    "antibody_redesign": False,
                }
                manifest["steps"].append(off_entry)
                try:
                    off_result = _run_offtarget_screen(
                        config,
                        run_dir,
                        spec,
                        state,
                        gpu,
                        stage="stage2",
                        retry_failed=retry_failed,
                    )
                    off_entry.update({"state": "complete", **off_result})
                    state.update(
                        "stage2_cldn18_1_offtarget_screen",
                        "complete",
                        detail=off_result["output"],
                    )
                except Exception as exc:
                    off_entry.update({"state": "failed", "detail": str(exc)})
                    state.update(
                        "stage2_cldn18_1_offtarget_screen",
                        "failed",
                        detail=str(exc),
                    )
                    write_json(run_dir / "execution_manifest.json", manifest)
                    raise

                final_inputs = _prepare_stage2_rosetta_inputs(run_dir)
                manifest["stage2_specificity"] = final_inputs
                if final_inputs["accepted"] == 0:
                    manifest["status"] = "complete_no_stage2_specificity_candidates"
                    manifest["zero_candidate_policy"] = {
                        "reported_passed": 0,
                        "threshold_auto_relaxed": False,
                        "outputs_preserved_for_manual_review": True,
                        "uncertain": final_inputs["uncertain"],
                    }
                    write_json(run_dir / "execution_manifest.json", manifest)
                    return manifest

                stage2_rosetta = fastrelax_path(run_dir / "pipeline_output" / "stage2" / "fastrelax_output")
                stage2_rosetta_legacy = fastrelax_path(run_dir / "pipeline_output" / "stage2" / "fastrelax_raw_output")
                rosetta_status = state.stage("stage2_fastrelax").get("status")
                rosetta_complete = rosetta_status == "complete" and (
                    stage2_rosetta.is_dir() or stage2_rosetta_legacy.is_dir()
                )
                if not rosetta_complete:
                    if rosetta_status == "failed" and not retry_failed:
                        raise RuntimeError(
                            "Stage 2 FastRelax failed; resume with --retry-failed"
                        )
                    if rosetta_status == "running":
                        raise RuntimeError("Stage 2 FastRelax is already running")
                    rosetta_entry = {
                        "stage": "stage2_fastrelax",
                        "state": "running",
                        "automatic": True,
                        "candidate_count": final_inputs["accepted"],
                        "input": final_inputs["input_dir"],
                    }
                    manifest["steps"].append(rosetta_entry)
                    try:
                        result = run_fastrelax(
                            config, run_id, stage=2, finalize_layout=False
                        )
                    except Exception as exc:
                        rosetta_entry.update({"state": "failed", "detail": str(exc)})
                        write_json(run_dir / "execution_manifest.json", manifest)
                        raise
                    state = RecoveryState(run_dir)
                    rosetta_entry.update(
                        {
                            "state": "complete",
                            "output": result["output"],
                        }
                    )
                if _automatic_iggm_enabled(config, spec):
                    if (
                        canonical_outputs_enabled
                        and not screening.get(
                            "final_functional_evaluation", {}
                        ).get("enabled", False)
                    ):
                        manifest["stage2_layout"] = finalize_stage2_layout(
                            run_dir,
                            expected_sequences=int(
                                spec["generation"]["stage2_sequences_per_backbone"]
                            ),
                        )
                        manifest["result_paths"] = write_result_paths(run_dir)
                    _run_automatic_iggm(
                        config,
                        run_id,
                        spec,
                        state,
                        manifest,
                        gpu,
                        retry_failed=retry_failed,
                    )
                    if _automatic_stage3_enabled(config, spec):
                        _run_stage3_iggm_specificity(
                            config,
                            run_id,
                            spec,
                            state,
                            manifest,
                            gpu,
                            retry_failed=retry_failed,
                        )
        final_config = screening.get("final_functional_evaluation", {})
        if final_config.get("enabled", False):
            final_entry = {
                "stage": "final_functional_evaluation",
                "state": "planned" if dry_run else "ready",
                "candidates_per_backbone_group": int(
                    screening["maturation"]["final_generation"][
                        "candidates_per_original_backbone_group"
                    ]
                ),
                "ranking_scope": "within_original_backbone_group",
                "affinity_output": "relative_proxy_not_kd",
            }
            manifest["steps"].append(final_entry)
            if not dry_run:
                try:
                    result = _run_final_functional_evaluation(
                        config, run_dir, spec, state, gpu, retry_failed=retry_failed
                    )
                    final_entry.update({"state": "complete", **result})
                except Exception as exc:
                    final_entry.update({"state": "failed", "detail": str(exc)})
                    state.update("final_functional_evaluation", "failed", detail=str(exc))
                    write_json(run_dir / "execution_manifest.json", manifest)
                    raise
                state.update("final_functional_evaluation", "complete", detail=result["output"])
    if not dry_run and canonical_outputs_enabled:
        manifest["stage1_layout"] = finalize_stage1_layout(run_dir)
        stage2_root = run_dir / "pipeline_output" / "stage2"
        if stage2_root.exists():
            manifest["stage2_layout"] = finalize_stage2_layout(
                run_dir,
                expected_sequences=int(
                    spec["generation"]["stage2_sequences_per_backbone"]
                ),
            )
        manifest["result_paths"] = write_result_paths(run_dir)
    elif not dry_run:
        manifest["screening_summary"] = write_screening_summary(run_dir)
    manifest["status"] = "complete"
    write_json(run_dir / "execution_manifest.json", manifest)
    return manifest
