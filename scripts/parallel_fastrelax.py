#!/usr/bin/env python3
"""Run PPIFlow's Rosetta interface analysis in parallel.

This wrapper does not modify the installed PPIFlow source tree.  It imports the
existing RosettaFixStep, runs its isolated ``_prepare_one()`` jobs concurrently,
and then invokes the existing interface-energy extractor.  Stage 1 can use
``--preserve-coordinates`` to score the input pose without running FastRelax.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_PPIFLOW_ROOT = os.environ.get("AGENT_AB_PPIFLOW_ROOT") or str(Path(__file__).resolve().parents[1] / "vendor/PPIFlow")
AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
BACKBONE_ID = re.compile(r"backbone\.0_\d+")
STAGE2_ID = re.compile(r"^(?:backbone|flowpacker)\.1_(\d+)(?:_|$)")


def _compact_candidate_stem(path: str) -> str:
    """Drop the run prefix while preserving the stable ``backbone.*`` ID."""
    stem = Path(path).stem
    marker = "_backbone."
    marker_index = stem.rfind(marker)
    if marker_index < 0:
        return stem
    return stem[marker_index + 1 :]


def _stage2_parent_id(stem: str) -> str:
    """Map a canonical or legacy Stage 2 candidate ID to its Stage 1 parent."""
    match = STAGE2_ID.match(stem)
    if not match:
        raise ValueError("cannot resolve Stage 2 parent from candidate: %s" % stem)
    return "backbone.0_%s" % match.group(1)


def _stage_compact_inputs(
    input_pdbs: Sequence[str], directory: str
) -> Tuple[List[str], Dict[str, str]]:
    """Expose input PDBs under compact names without modifying Stage 1."""
    staged_paths: List[str] = []
    source_by_staged_name: Dict[str, str] = {}
    for input_pdb in input_pdbs:
        source = os.path.abspath(input_pdb)
        staged_name = f"{_compact_candidate_stem(source)}.pdb"
        if staged_name in source_by_staged_name:
            raise ValueError(
                f"compact PDB name collision for {staged_name}: "
                f"{source_by_staged_name[staged_name]} and {source}"
            )
        staged_path = os.path.join(directory, staged_name)
        os.symlink(source, staged_path)
        staged_paths.append(staged_path)
        source_by_staged_name[staged_name] = source
    return staged_paths, source_by_staged_name


def _group_inputs(
    input_pdbdir: str,
    input_pdbs: Sequence[str],
    manifest_path: str | None,
) -> Dict[str, List[str]]:
    """Validate that every candidate belongs to exactly one backbone group."""
    root = Path(input_pdbdir).absolute()
    groups: Dict[str, List[str]] = {}
    for raw in input_pdbs:
        path = Path(raw).absolute()
        relative = path.relative_to(root)
        name_match = BACKBONE_ID.match(path.stem)
        if name_match is None:
            raise ValueError(f"candidate name has no backbone group: {path.name}")
        name_group = name_match.group(0)
        directory_group = relative.parts[0] if len(relative.parts) > 1 else name_group
        if not BACKBONE_ID.fullmatch(directory_group):
            raise ValueError(f"invalid FastRelax group directory: {directory_group}")
        if directory_group != name_group:
            raise ValueError(
                f"candidate {path.name} is under {directory_group}, "
                f"but its filename belongs to {name_group}"
            )
        groups.setdefault(name_group, []).append(str(path))

    if manifest_path:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if manifest.get("cross_group_merge") is not False:
            raise ValueError("FastRelax group manifest must forbid cross-group merge")
        declared = {
            group: {str(item["design_id"]) for item in items}
            for group, items in manifest.get("groups", {}).items()
        }
        actual = {
            group: {Path(item).stem for item in items}
            for group, items in groups.items()
        }
        if declared != actual:
            raise ValueError("FastRelax group manifest does not match grouped inputs")
    return groups


def _pdb_sequences(path: str) -> Dict[str, str]:
    chains: Dict[str, List[str]] = {}
    seen = set()
    with open(path, "r", errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                continue
            chain = line[21:22].strip()
            residue_key = (chain, line[22:27])
            if not chain or residue_key in seen:
                continue
            seen.add(residue_key)
            chains.setdefault(chain, []).append(AA3.get(line[17:20].strip(), "X"))
    return {chain: "".join(values) for chain, values in chains.items()}


def _ensure_mpnn_inputs(
    mpnn_seqs_csv: str,
    groups: Dict[str, List[str]],
    backbone_root: str | None,
) -> str:
    """Restore the minimal Stage 1 sequence/backbone index needed downstream."""
    output = Path(mpnn_seqs_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        temporary = output.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["link_name", "sequence_dict", "seq_idx"]
            )
            writer.writeheader()
            for group, candidates in sorted(groups.items()):
                for candidate in sorted(candidates):
                    stem = Path(candidate).stem
                    suffix = stem[len(group) + 1 :]
                    if not suffix.isdigit():
                        raise ValueError(f"cannot read sequence index from {stem}")
                    writer.writerow(
                        {
                            "link_name": group + ".pdb",
                            "sequence_dict": repr(_pdb_sequences(candidate)),
                            "seq_idx": int(suffix),
                        }
                    )
        temporary.replace(output)

    for group, candidates in sorted(groups.items()):
        destination = output.parent / (group + ".pdb")
        if destination.exists():
            continue
        source = (
            Path(backbone_root) / group / (group + ".pdb")
            if backbone_root
            else Path(candidates[0])
        )
        if not source.is_file():
            raise FileNotFoundError(f"backbone source missing for {group}: {source}")
        destination.symlink_to(os.path.relpath(source, destination.parent))
    return str(output)


def _build_fixed_positions(
    *,
    ppiflow_root: str,
    rosetta_output_dir: str,
    output_csv: str,
    gentype: str,
    mpnn_seqs_csv: str,
    energy_threshold: float,
    groups: Dict[str, List[str]],
) -> str:
    root = os.path.abspath(ppiflow_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from pipeline import _build_fixed_positions_csv

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    _build_fixed_positions_csv(
        rosetta_fix_output_dir=rosetta_output_dir,
        output_csv=str(output),
        gentype=gentype,
        mpnn_seqs_csv=mpnn_seqs_csv,
        energy_threshold=energy_threshold,
    )
    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    observed = [Path(row.get("filename", "")).stem for row in rows]
    if len(observed) != len(set(observed)):
        raise ValueError("fixed-position output contains duplicate backbone groups")
    if set(observed) != set(groups):
        raise ValueError(
            "fixed-position groups do not match FastRelax input groups: "
            f"expected={sorted(groups)} observed={sorted(observed)}"
        )
    return str(output)


def _restore_interface_csv_provenance(
    interface_outputs: Dict[str, str],
    source_by_staged_name: Dict[str, str],
) -> None:
    """Keep original candidate IDs while pointing at compact Rosetta paths."""
    for output_dir in interface_outputs.values():
        csv_path = os.path.join(output_dir, "residue_energy.csv")
        if not os.path.isfile(csv_path):
            continue
        with open(csv_path, newline="") as src:
            reader = csv.DictReader(src)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
        if "pdbpath" not in fieldnames:
            continue
        for row in rows:
            staged_name = os.path.basename(row.get("pdbpath", ""))
            original = source_by_staged_name.get(staged_name)
            if original is None:
                continue
            row["pdbpath"] = original
            if "pdbname" in fieldnames:
                # PPIFlow calls os.path.splitext() on pdbname.  Keep the .pdb
                # suffix so compact IDs such as ``backbone.0_132_3`` are not
                # truncated to ``backbone`` at the first dot.
                row["pdbname"] = Path(original).name
        temporary_path = csv_path + ".tmp"
        with open(temporary_path, "w", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, csv_path)


def _parse_chain_pair(value: str) -> Tuple[str, str, str]:
    """Parse BINDER:TARGET[:OUTPUT_LABEL], for example A:B:A_C."""
    parts = [part.strip() for part in value.split(":")]
    if len(parts) not in (2, 3) or not all(parts):
        raise argparse.ArgumentTypeError(
            "chain pair must be BINDER:TARGET or BINDER:TARGET:OUTPUT_LABEL"
        )
    binder, target = parts[:2]
    label = parts[2] if len(parts) == 3 else f"{binder}_{target}"
    return binder, target, label


def _pdb_chain_ids(path: str) -> set[str]:
    chains: set[str] = set()
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if line.startswith("ATOM") and len(line) > 21:
                chain = line[21].strip()
                if chain:
                    chains.add(chain)
    return chains


def _default_chain_pairs(
    gentype: str, input_pdbs: Sequence[str]
) -> List[Tuple[str, str, str]]:
    """Resolve actual chains while retaining PPIFlow's canonical output labels."""
    chain_sets = [_pdb_chain_ids(path) for path in input_pdbs]
    common = set.intersection(*chain_sets)
    if gentype == "binder":
        design_chains, labels = ("A",), ("A_B",)
    elif gentype == "nanobody":
        design_chains, labels = ("A",), ("A_C",)
    elif gentype == "antibody":
        design_chains, labels = ("A", "B"), ("A_C", "B_C")
    else:
        raise ValueError(f"unsupported gentype: {gentype}")

    missing_design = [chain for chain in design_chains if chain not in common]
    if missing_design:
        raise ValueError(
            f"design chains missing from one or more input PDBs: {missing_design}; "
            f"common chains={sorted(common)}"
        )

    target_candidates = sorted(common.difference(design_chains))
    if not target_candidates:
        raise ValueError(
            f"cannot infer target chain; common chains={sorted(common)} "
            f"design chains={list(design_chains)}"
        )
    target = "C" if "C" in target_candidates else target_candidates[0]
    return [
        (binder, target, label)
        for binder, label in zip(design_chains, labels)
    ]


def _load_rosetta_fix_step(ppiflow_root: str):
    root = os.path.abspath(ppiflow_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from steps.rosetta_fix_step import RosettaFixStep

    return RosettaFixStep


def _has_residue_energy_output(path: str) -> bool:
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    with open(path, "r", errors="replace") as fh:
        return any(line.startswith("ResResE") for line in fh)


def _coordinate_policy_path(step, input_pdb: str) -> Path:
    return Path(step._build_paths(os.path.abspath(input_pdb))["work_dir"]) / "coordinate_policy.json"


def _fixed_residue_policy_path(step, input_pdb: str) -> Path:
    return Path(step._build_paths(os.path.abspath(input_pdb))["work_dir"]) / "fixed_residue_policy.json"


def _is_complete(
    step,
    input_pdb: str,
    preserve_coordinates: bool = False,
    locked_positions: str | None = None,
) -> bool:
    paths = step._build_paths(os.path.abspath(input_pdb))
    output_pdb = os.path.join(
        paths["work_dir"], f'{paths["pdb_name"]}_0001.pdb'
    )
    complete = (
        os.path.isfile(output_pdb)
        and os.path.getsize(output_pdb) > 0
        and _has_residue_energy_output(paths["rosetta_out"])
    )
    if not complete:
        return False
    if locked_positions:
        policy_path = _fixed_residue_policy_path(step, input_pdb)
        if not policy_path.is_file():
            return False
        try:
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return (
            policy.get("policy") == "restore_exact_fixed_residue_atoms_after_fastrelax"
            and policy.get("fixed_positions") == locked_positions
        )
    if not preserve_coordinates:
        return complete
    policy_path = _coordinate_policy_path(step, input_pdb)
    if not policy_path.is_file():
        return False
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return policy.get("policy") == "score_only_preserve_input_coordinates"


def _completed_result(step, input_pdb: str) -> Dict[str, object]:
    input_pdb = os.path.abspath(input_pdb)
    paths = step._build_paths(input_pdb)
    return {
        "gentype": step.gentype,
        "input_pdb": input_pdb,
        "pdb_name": paths["pdb_name"],
        "work_dir": paths["work_dir"],
        "local_pdb": paths["local_pdb"],
        "update_xml": paths["update_xml"],
        "rosetta_out": paths["rosetta_out"],
        "status": "skipped_complete",
    }


def _score_only_xml(path: str) -> None:
    """Score the input pose without minimization and retain pair-energy output."""
    xml_path = Path(path)
    text = xml_path.read_text(encoding="utf-8")
    updated, count = re.subn(
        r'<Add\s+mover_name=["\']fast_relax["\']\s*/>',
        '<Add mover_name="score_only"/>',
        text,
    )
    if count != 1:
        raise ValueError(
            "expected exactly one FastRelax protocol call in %s; found %d"
            % (xml_path, count)
        )
    updated, scorefxn_count = re.subn(
        r"<SCOREFXNS>\s*</SCOREFXNS>",
        '<SCOREFXNS>\n                <ScoreFunction name="score_only_ref2015" weights="ref2015"/>\n        </SCOREFXNS>',
        updated,
        count=1,
    )
    if scorefxn_count != 1:
        raise ValueError("failed to add the score-only scorefunction in %s" % xml_path)
    updated, mover_count = re.subn(
        r"</MOVERS>",
        '             <ScoreMover name="score_only" scorefxn="score_only_ref2015"/>\n        </MOVERS>',
        updated,
        count=1,
    )
    if mover_count != 1:
        raise ValueError("failed to add the score-only Rosetta mover in %s" % xml_path)
    xml_path.write_text(updated, encoding="utf-8")


def _pdb_atom_coordinates(path: str) -> Dict[Tuple[str, str, str, str, str], Tuple[float, float, float]]:
    coordinates = {}
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM  ") or len(line) < 54:
                continue
            key = (
                line[21],
                line[22:26],
                line[26],
                line[12:16],
                line[16],
            )
            coordinates[key] = (
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            )
    return coordinates


def _verify_coordinates_unchanged(input_pdb: str, output_pdb: str) -> Dict[str, object]:
    before = _pdb_atom_coordinates(input_pdb)
    after = _pdb_atom_coordinates(output_pdb)
    missing = sorted(set(before).difference(after))
    if missing:
        raise RuntimeError(
            "score-only Rosetta output lost %d input atoms; first=%s"
            % (len(missing), missing[0])
        )
    max_displacement = 0.0
    moved = None
    for key, xyz_before in before.items():
        xyz_after = after[key]
        displacement = sum(
            (xyz_after[index] - xyz_before[index]) ** 2 for index in range(3)
        ) ** 0.5
        if displacement > max_displacement:
            max_displacement = displacement
            moved = key
    if max_displacement > 0.0015:
        raise RuntimeError(
            "score-only Rosetta changed coordinates: max displacement %.6f A at %s"
            % (max_displacement, moved)
        )
    return {
        "policy": "score_only_preserve_input_coordinates",
        "input_atom_count": len(before),
        "output_atom_count": len(after),
        "max_input_atom_displacement_angstrom": max_displacement,
    }


def _run_score_only(step, input_pdb: str) -> Dict[str, object]:
    input_pdb = os.path.abspath(input_pdb)
    paths = step._build_paths(input_pdb)
    step._prepare_workdir(input_pdb, paths)
    step._generate_update_xml(paths["local_pdb"], paths["update_xml"])
    _score_only_xml(paths["update_xml"])
    step._run_rosetta(
        work_dir=paths["work_dir"],
        local_pdb=paths["local_pdb"],
        rosetta_out_path=paths["rosetta_out"],
    )
    output_pdb = os.path.join(
        paths["work_dir"], "%s_0001.pdb" % paths["pdb_name"]
    )
    policy = _verify_coordinates_unchanged(paths["local_pdb"], output_pdb)
    _coordinate_policy_path(step, input_pdb).write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "gentype": step.gentype,
        "input_pdb": input_pdb,
        "pdb_name": paths["pdb_name"],
        "work_dir": paths["work_dir"],
        "local_pdb": paths["local_pdb"],
        "update_xml": paths["update_xml"],
        "rosetta_out": paths["rosetta_out"],
        "coordinate_policy": policy,
    }


def _restore_locked_positions(
    step,
    input_pdb: str,
    fixed_positions: str,
) -> Dict[str, object]:
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from agent_ab.fixed_residue_atoms import restore_full_atom_fixed_residues

    paths = step._build_paths(os.path.abspath(input_pdb))
    output_pdb = os.path.join(
        paths["work_dir"], "%s_0001.pdb" % paths["pdb_name"]
    )
    alignment_chain = "B" if step.gentype == "binder" else "C"
    report = restore_full_atom_fixed_residues(
        reference_pdb=input_pdb,
        mobile_pdb=output_pdb,
        fixed_positions=fixed_positions,
        alignment_chains=[alignment_chain],
    )
    policy = {
        "policy": "restore_exact_fixed_residue_atoms_after_fastrelax",
        "fixed_positions": fixed_positions,
        "alignment_chain": alignment_chain,
        **report,
    }
    _fixed_residue_policy_path(step, input_pdb).write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return policy


def _run_one(
    step,
    input_pdb: str,
    resume: bool,
    preserve_coordinates: bool = False,
    locked_positions: str | None = None,
) -> Dict[str, object]:
    if resume and not step.overwrite and _is_complete(
        step,
        input_pdb,
        preserve_coordinates=preserve_coordinates,
        locked_positions=locked_positions,
    ):
        return _completed_result(step, input_pdb)

    result = (
        _run_score_only(step, input_pdb)
        if preserve_coordinates
        else step._prepare_one(input_pdb)
    )
    if locked_positions:
        result["fixed_residue_policy"] = _restore_locked_positions(
            step,
            input_pdb,
            locked_positions,
        )
    result["status"] = "completed"
    return result


def _run_parallel(
    step,
    input_pdbs: Sequence[str],
    max_workers: int,
    resume: bool,
    preserve_coordinates: bool = False,
    locked_positions_by_input: Dict[str, str] | None = None,
) -> List[Dict[str, object]]:
    worker_count = min(max_workers, len(input_pdbs))
    print(
        f"[ParallelFastRelax] candidates={len(input_pdbs)} "
        f"workers={worker_count} resume={resume}",
        flush=True,
    )

    result_by_input: Dict[str, Dict[str, object]] = {}
    failures: List[Tuple[str, Exception]] = []
    progress_interval = max(1, len(input_pdbs) // 20)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_input = {
            executor.submit(
                _run_one,
                step,
                input_pdb,
                resume,
                preserve_coordinates,
                (locked_positions_by_input or {}).get(input_pdb),
            ): input_pdb
            for input_pdb in input_pdbs
        }
        for completed, future in enumerate(as_completed(future_to_input), start=1):
            input_pdb = future_to_input[future]
            try:
                result_by_input[input_pdb] = future.result()
            except Exception as exc:
                failures.append((input_pdb, exc))
                print(
                    f"[ParallelFastRelax] Failed: {input_pdb}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            if completed % progress_interval == 0 or completed == len(input_pdbs):
                print(
                    f"[ParallelFastRelax] progress={completed}/{len(input_pdbs)}",
                    flush=True,
                )

    if failures:
        details = "; ".join(
            f"{Path(path).name}: {exc}" for path, exc in failures
        )
        raise RuntimeError(
            f"Rosetta failed for {len(failures)}/{len(input_pdbs)} "
            f"candidates: {details}"
        )

    return [result_by_input[input_pdb] for input_pdb in input_pdbs]


def _run_interface_energy(
    *,
    ppiflow_root: str,
    python: str,
    input_pdbdir: str,
    rosetta_dir: str,
    chain_pairs: Iterable[Tuple[str, str, str]],
    interface_dist: float,
    plot_toggle: bool,
) -> Dict[str, str]:
    script = os.path.join(
        os.path.abspath(ppiflow_root),
        "steps",
        "rosetta",
        "1get_interface_energy.py",
    )
    if not os.path.isfile(script):
        raise FileNotFoundError(f"interface-energy script not found: {script}")

    outputs: Dict[str, str] = {}
    for binder, target, output_label in chain_pairs:
        output_dir = os.path.join(
            os.path.abspath(rosetta_dir), f"interface_energy_{output_label}"
        )
        os.makedirs(output_dir, exist_ok=True)
        cmd = [
            python,
            script,
            "--input_pdbdir",
            os.path.abspath(input_pdbdir),
            "--rosetta_dir",
            os.path.abspath(rosetta_dir),
            "--binder_id",
            binder,
            "--target_id",
            target,
            "--output_dir",
            output_dir,
            "--interface_dist",
            str(interface_dist),
            "--plot_toggle",
            str(plot_toggle),
        ]
        print(
            f"[ParallelFastRelax] Interface energy {binder}:{target} "
            f"-> interface_energy_{output_label}",
            flush=True,
        )
        # The upstream extractor calls ``plt.show()`` unconditionally after
        # saving its summary plot.  When a pipeline was launched from
        # ``ssh -X``, the inherited DISPLAY makes that call wait forever for a
        # GUI window to be closed.  Run the extractor with a headless backend
        # so batch jobs terminate after writing their outputs.
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        env.pop("DISPLAY", None)
        completed = subprocess.run(
            cmd,
            check=False,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if completed.returncode != 0:
            diagnostic_tail = completed.stdout[-4000:]
            raise RuntimeError(
                f"interface-energy extractor failed with exit code "
                f"{completed.returncode}:\n{diagnostic_tail}"
            )
        print(
            f"[ParallelFastRelax] Interface energy {binder}:{target} complete",
            flush=True,
        )
        outputs[output_label] = output_dir
    return outputs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the installed PPIFlow FastRelax per-candidate work in parallel "
            "without modifying the installed PPIFlow files."
        )
    )
    parser.add_argument("--input-pdbdir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--group-manifest",
        help="candidate-to-backbone manifest; cross-group merging must be false",
    )
    parser.add_argument(
        "--fixed-positions-csv",
        help="write one fixed-position row per original backbone group",
    )
    parser.add_argument("--mpnn-seqs-csv")
    parser.add_argument("--backbone-root")
    parser.add_argument("--energy-threshold", type=float, default=-5.0)
    parser.add_argument(
        "--gentype", required=True, choices=("binder", "nanobody", "antibody")
    )
    parser.add_argument("--ppiflow-root", default=DEFAULT_PPIFLOW_ROOT)
    parser.add_argument("--rosetta-bin")
    parser.add_argument("--native-xml")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--interface-dist", type=float, default=12.0)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--chain-pair",
        action="append",
        type=_parse_chain_pair,
        help=(
            "actual binder:target chains and optional downstream output label; "
            "repeat as needed, e.g. A:B:A_C"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-resume", action="store_true", help="rerun candidates even if complete"
    )
    parser.add_argument("--skip-interface-energy", action="store_true")
    parser.add_argument(
        "--preserve-coordinates",
        action="store_true",
        help=(
            "score the original input pose and extract interface energies "
            "without invoking FastRelax; verifies every input atom coordinate"
        ),
    )
    parser.add_argument(
        "--locked-positions-csv",
        help=(
            "Stage 2 fixed_positions.csv; after FastRelax, restore every listed "
            "residue from the input PDB and verify the restoration marker"
        ),
    )
    parser.add_argument("--plot-toggle", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show candidate/completion counts without running Rosetta",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least 1")

    input_pdbdir = os.path.abspath(args.input_pdbdir)
    original_input_pdbs = sorted(
        str(path.absolute()) for path in Path(input_pdbdir).rglob("*.pdb")
    )
    if not original_input_pdbs:
        raise ValueError(f"no PDB files found in: {input_pdbdir}")
    grouped_mode = bool(args.group_manifest or args.fixed_positions_csv)
    groups = (
        _group_inputs(input_pdbdir, original_input_pdbs, args.group_manifest)
        if grouped_mode
        else {"all_candidates": list(original_input_pdbs)}
    )
    locked_positions_by_stem = None
    if args.locked_positions_csv:
        with open(args.locked_positions_csv, "r", encoding="utf-8", newline="") as handle:
            locked_by_parent = {
                Path(row["filename"]).stem: (row.get("fixed_positions") or "").strip()
                for row in csv.DictReader(handle)
            }
        locked_positions_by_stem = {}
        for input_pdb in original_input_pdbs:
            stem = _compact_candidate_stem(input_pdb)
            parent = _stage2_parent_id(stem)
            fixed_positions = locked_by_parent.get(parent, "")
            if not fixed_positions or fixed_positions.upper() == "NONE":
                raise ValueError(
                    "Stage 2 candidate %s has no locked positions for parent %s"
                    % (stem, parent)
                )
            locked_positions_by_stem[stem] = fixed_positions

    root = os.path.abspath(args.ppiflow_root)
    rosetta_bin = args.rosetta_bin or os.path.join(
        root,
        "rosetta.binary.linux.release-371",
        "main",
        "source",
        "bin",
        "rosetta_scripts.static.linuxgccrelease",
    )
    with tempfile.TemporaryDirectory(prefix="agent-ab-rosetta-inputs-") as staged_dir:
        input_pdbs, source_by_staged_name = _stage_compact_inputs(
            original_input_pdbs, staged_dir
        )
        locked_positions_by_input = (
            {
                path: locked_positions_by_stem[Path(path).stem]
                for path in input_pdbs
            }
            if locked_positions_by_stem is not None
            else None
        )
        RosettaFixStep = _load_rosetta_fix_step(root)
        step = RosettaFixStep(
            gentype=args.gentype,
            output_dir=os.path.abspath(args.output_dir),
            rosetta_bin=rosetta_bin,
            native_xml=args.native_xml,
            python=args.python,
            interface_dist=args.interface_dist,
            plot_toggle=args.plot_toggle,
            overwrite=args.overwrite,
        )

        resume = not args.no_resume
        completed_before = sum(
            _is_complete(
                step,
                path,
                preserve_coordinates=args.preserve_coordinates,
                locked_positions=(locked_positions_by_input or {}).get(path),
            )
            for path in input_pdbs
        )
        print(
            json.dumps(
                {
                    "input_pdbdir": input_pdbdir,
                    "output_dir": os.path.abspath(args.output_dir),
                    "candidates": len(input_pdbs),
                    "backbone_groups": len(groups),
                    "grouping_key": "backbone_id" if grouped_mode else None,
                    "cross_group_merge": False if grouped_mode else None,
                    "already_complete": completed_before,
                    "would_run": len(input_pdbs) - completed_before if resume else len(input_pdbs),
                    "max_workers": min(args.max_workers, len(input_pdbs)),
                    "coordinate_policy": (
                        "score_only_preserve_input_coordinates"
                        if args.preserve_coordinates
                        else "fastrelax"
                    ),
                },
                indent=2,
            ),
            flush=True,
        )
        if args.dry_run:
            return 0

        results = _run_parallel(
            step,
            input_pdbs,
            args.max_workers,
            resume,
            preserve_coordinates=args.preserve_coordinates,
            locked_positions_by_input=locked_positions_by_input,
        )
        chain_pairs = args.chain_pair or _default_chain_pairs(args.gentype, input_pdbs)
        print(
            "[ParallelFastRelax] chain_pairs="
            + json.dumps(chain_pairs, ensure_ascii=False),
            flush=True,
        )
        interface_outputs = {}
        if not args.skip_interface_energy:
            interface_outputs = _run_interface_energy(
                ppiflow_root=root,
                python=args.python,
                input_pdbdir=staged_dir,
                rosetta_dir=os.path.abspath(args.output_dir),
                chain_pairs=chain_pairs,
                interface_dist=args.interface_dist,
                plot_toggle=args.plot_toggle,
            )
            _restore_interface_csv_provenance(
                interface_outputs, source_by_staged_name
            )

    fixed_positions_csv = None
    if args.fixed_positions_csv:
        if args.skip_interface_energy:
            raise ValueError(
                "--fixed-positions-csv cannot be used with --skip-interface-energy"
            )
        if not args.mpnn_seqs_csv:
            raise ValueError(
                "--mpnn-seqs-csv is required with --fixed-positions-csv"
            )
        mpnn_seqs_csv = _ensure_mpnn_inputs(
            args.mpnn_seqs_csv, groups, args.backbone_root
        )
        fixed_positions_csv = _build_fixed_positions(
            ppiflow_root=root,
            rosetta_output_dir=os.path.abspath(args.output_dir),
            output_csv=args.fixed_positions_csv,
            gentype=args.gentype,
            mpnn_seqs_csv=mpnn_seqs_csv,
            energy_threshold=args.energy_threshold,
            groups=groups,
        )

    group_summary = {
        "schema_version": 1,
        "grouping_key": "backbone_id" if grouped_mode else None,
        "cross_group_merge": False if grouped_mode else None,
        "group_count": len(groups),
        "candidate_count": len(original_input_pdbs),
        "coordinate_policy": (
            "score_only_preserve_input_coordinates"
            if args.preserve_coordinates
            else "fastrelax"
        ),
        "fixed_residue_policy": (
            "restore_exact_fixed_residue_atoms_after_fastrelax"
            if locked_positions_by_stem is not None
            else None
        ),
        "fixed_positions_csv": fixed_positions_csv,
        "groups": {
            group: [Path(path).stem for path in paths]
            for group, paths in sorted(groups.items())
        },
    }
    group_summary_path = Path(args.output_dir) / "group_manifest.json"
    group_summary_path.write_text(
        json.dumps(group_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "num_pdbs": len(input_pdbs),
        "completed": sum(item["status"] == "completed" for item in results),
        "skipped_complete": sum(
            item["status"] == "skipped_complete" for item in results
        ),
        "interface_outputs": interface_outputs,
        "group_manifest": str(group_summary_path),
        "fixed_positions_csv": fixed_positions_csv,
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
