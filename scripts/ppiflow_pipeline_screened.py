#!/usr/bin/env python3
"""PPIFlow scheduler wrapper that screens nanobody backbones before AbMPNN."""

import argparse
import ast
import csv
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PPI_FLOW_ROOT = Path(os.environ.get("AGENT_AB_PPIFLOW_ROOT") or PROJECT_ROOT / "vendor/PPIFlow")
PPI_FLOW_ANTIBODY_LAUNCHER = (
    PROJECT_ROOT / "scripts" / "ppiflow_sample_antibody_minimal.py"
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PPI_FLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(PPI_FLOW_ROOT))

from agent_ab.backbone_screen import (
    screen_backbones,
    screen_scfv_backbones,
    select_top_scfv_backbones,
)
from agent_ab.backbone_outputs import organize_backbone_output
from agent_ab.abmpnn_outputs import organize_abmpnn_output
from agent_ab.flowpacker_outputs import organize_flowpacker_output
from agent_ab.naming import canonical_backbone_id, parse_backbone_id, stage2_backbone_id


class BackboneSelectionRequired(RuntimeError):
    """Stop Stage 1 cleanly while the agent asks the user for a top-N choice."""

    def __init__(self, request):
        self.request = request
        super().__init__(request["prompt"])


def _screening_hotspots(task, drop_nanobody_auxiliary=False):
    """Map requested antigen hotspots to PPIFlow's canonical antigen chain C."""
    raw = task.get("backbone_screen_hotspots", task.get("specified_hotspots", ""))
    hotspots = []
    for value in (item.strip() for item in raw.split(",")):
        if not value:
            continue
        match = re.fullmatch(r"[A-Za-z](-?\d+)([A-Za-z]?)", value)
        if not match:
            raise ValueError("invalid screening hotspot: %r" % value)
        residue_number = int(match.group(1))
        if drop_nanobody_auxiliary and residue_number in (40, 42):
            continue
        hotspots.append("C%s%s" % (residue_number, match.group(2)))
    return hotspots


def _load_pipeline_module():
    """Import PPIFlow under its real, child-process-importable module name."""
    module = importlib.import_module("pipeline")
    expected = (PPI_FLOW_ROOT / "pipeline.py").resolve()
    loaded = Path(module.__file__).resolve()
    if loaded != expected:
        raise ImportError(
            "loaded unexpected pipeline module: %s (expected %s)"
            % (loaded, expected)
        )
    if not PPI_FLOW_ANTIBODY_LAUNCHER.is_file():
        raise FileNotFoundError(
            "PPIFlow antibody launcher not found: %s"
            % PPI_FLOW_ANTIBODY_LAUNCHER
        )
    ppiflow_step_module = importlib.import_module("steps.ppiflow_step")
    ppiflow_step_module._ANTIBODY_SCRIPT = str(PPI_FLOW_ANTIBODY_LAUNCHER)
    # Keep steps.partial_step._ANTIBODY_SCRIPT unchanged so Stage 2 uses
    # PPIFlow/tools/PPIFlow/sample_antibody_nanobody_partial.py,
    # exactly as documented by the official PPIFlow tutorial.
    return module


class _PathWithRosettaFixOverride:
    """Delegate os.path except for PPIFlow's hard-coded FastRelax directory."""

    def __init__(self, original, output_dir):
        self._original = original
        self._output_dir = str(output_dir)

    def join(self, *parts):
        if parts and str(parts[-1]) == "rosetta_fix_output":
            return self._output_dir
        return self._original.join(*parts)

    def __getattr__(self, name):
        return getattr(self._original, name)


class _OsWithRosettaFixOverride:
    def __init__(self, original, output_dir):
        self._original = original
        self.path = _PathWithRosettaFixOverride(original.path, output_dir)

    def __getattr__(self, name):
        return getattr(self._original, name)


def _route_rosetta_fix_output(native, output_dir):
    """Make native Stage 2 consume the real FastRelax results stored in Stage 1."""
    if output_dir:
        native.os = _OsWithRosettaFixOverride(native.os, output_dir)


def _remove_disabled_empty_output(pipeline, step_name, directory_name, log):
    """Remove an empty native output directory when its pipeline step is disabled."""
    if pipeline._enabled(step_name):
        return False
    output_dir = Path(pipeline.stage2_dir) / directory_name
    if not output_dir.is_dir():
        return False
    try:
        output_dir.rmdir()
    except OSError:
        log.info(
            "Preserving non-empty output directory for disabled %s: %s",
            step_name,
            output_dir,
        )
        return False
    log.info("Removed empty output directory for disabled %s: %s", step_name, output_dir)
    return True


def _parallel_gpu_ids():
    """Return unique physical GPU ids requested by the Agent runner."""
    raw = os.environ.get("AGENT_AB_PPIFLOW_BACKBONE_GPUS", "")
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        gpu = int(item)
        if gpu not in values:
            values.append(gpu)
    return values


def _balanced_shards(total, gpu_ids):
    """Split global sample ids into contiguous, balanced GPU shards."""
    active = list(gpu_ids[: max(1, min(len(gpu_ids), total))])
    base, remainder = divmod(total, len(active))
    shards = []
    offset = 0
    for index, gpu in enumerate(active):
        count = base + (1 if index < remainder else 0)
        shards.append(
            {
                "index": index,
                "gpu": gpu,
                "offset": offset,
                "count": count,
            }
        )
        offset += count
    return shards


def _parse_gpu_id_list(raw):
    values = []
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        gpu = int(item)
        if gpu not in values:
            values.append(gpu)
    return values


def _idle_physical_gpu_ids(profile="ABMPNN"):
    """Return genuinely idle physical GPUs without taking over other jobs."""
    profile = str(profile).strip().upper()
    if not re.fullmatch(r"[A-Z0-9_]+", profile):
        raise ValueError("invalid GPU policy profile: %r" % profile)
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
    busy_uuids = set()
    for line in process_query.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if fields and fields[0].startswith("GPU-"):
            busy_uuids.add(fields[0])

    memory_limit = int(
        os.environ.get("AGENT_AB_%s_IDLE_MEMORY_MIB" % profile, "1024")
    )
    utilization_limit = int(
        os.environ.get("AGENT_AB_%s_IDLE_UTILIZATION" % profile, "10")
    )
    maximum = max(
        1, int(os.environ.get("AGENT_AB_%s_MAX_GPUS" % profile, "4"))
    )
    choices = []
    for line in gpu_query.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 4:
            continue
        index, uuid, memory, utilization = fields
        index = int(index)
        memory = int(memory)
        utilization = int(utilization)
        if (
            uuid not in busy_uuids
            and memory <= memory_limit
            and utilization <= utilization_limit
        ):
            choices.append((memory, utilization, index))
    if not choices:
        print("[IdleGPUQuery] profile=%s self_pid=%s gpu=%r compute=%r" % (profile, os.getpid(), gpu_query.stdout, process_query.stdout), flush=True)
    return [item[2] for item in sorted(choices)[:maximum]]


def _abmpnn_gpu_ids():
    """Rediscover idle cards immediately before every AbMPNN stage.

    The top-level pipeline resolves ``--gpu auto`` to one coordinator GPU when
    it enters the nested PPIFlow Stage 2 process.  That internal handoff looks
    like an explicit single-GPU request, so trusting its inherited list would
    accidentally serialize AbMPNN.  AbMPNN therefore follows the same launch-
    time idle-card policy as PartialFlow and FlowPacker.
    """
    values = _idle_physical_gpu_ids()
    if not values:
        raise RuntimeError(
            "no idle GPU is available for AbMPNN; another compute process, "
            "more than 1024 MiB memory, or more than 10% utilization marks a GPU busy"
        )
    return values


def _flowpacker_gpu_ids():
    """Discover idle cards for FlowPacker immediately before it starts."""
    values = _idle_physical_gpu_ids(profile="FLOWPACKER")
    if not values:
        raise RuntimeError(
            "no idle GPU is available for FlowPacker; any compute process, "
            "more than 1024 MiB memory, or more than 10% utilization marks a GPU busy"
        )
    return values


def _partialflow_gpu_ids():
    """Discover idle cards for Stage 2 PartialFlow at launch time."""
    values = _idle_physical_gpu_ids(profile="PARTIALFLOW")
    if not values:
        raise RuntimeError(
            "no idle GPU is available for PartialFlow; any compute process, "
            "more than 1024 MiB memory, or more than 10% utilization marks a GPU busy"
        )
    return values


def _valid_abmpnn_fasta(path, expected_sequences):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_records = int(expected_sequences) + 1
    if len(lines) != expected_records * 2:
        return False
    return all(
        lines[index].startswith(">") and not lines[index + 1].startswith(">")
        for index in range(0, len(lines), 2)
    )


def _read_single_fasta_sequence(path):
    sequence = "".join(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith(">")
    )
    if not sequence:
        raise ValueError("empty organized AbMPNN FASTA: %s" % path)
    return sequence


def _restore_native_abmpnn_fastas(output_dir, pdbs, expected_sequences):
    """Rebuild native multi-record FASTAs after canonical output compaction."""
    output_dir = Path(output_dir)
    native_seqs = output_dir / "seqs"
    native_seqs.mkdir(parents=True, exist_ok=True)
    restored = 0
    for pdb in pdbs:
        destination = native_seqs / (pdb.stem + ".fa")
        if _valid_abmpnn_fasta(destination, expected_sequences):
            continue
        organized_dir = output_dir / pdb.stem
        if not organized_dir.is_dir() or not pdb.stem.startswith("backbone"):
            continue
        prefix = "abmpnn" + pdb.stem[len("backbone") :]
        sources = [
            organized_dir / ("%s_%d.fa" % (prefix, index))
            for index in range(1, int(expected_sequences) + 1)
        ]
        if not all(source.is_file() for source in sources):
            continue
        sequences = [_read_single_fasta_sequence(source) for source in sources]
        records = [("original_reconstructed", sequences[0])] + [
            ("sample=%d" % index, sequence)
            for index, sequence in enumerate(sequences, 1)
        ]
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(
            "".join(">%s\n%s\n" % record for record in records),
            encoding="utf-8",
        )
        if not _valid_abmpnn_fasta(temporary, expected_sequences):
            temporary.unlink()
            raise RuntimeError("failed to reconstruct AbMPNN FASTA: %s" % destination)
        os.replace(str(temporary), str(destination))
        restored += 1
    return restored


def _balanced_file_shards(paths, gpu_ids):
    active = list(gpu_ids[: max(1, min(len(gpu_ids), len(paths)))])
    shards = [{"index": index, "gpu": gpu, "paths": []} for index, gpu in enumerate(active)]
    for index, path in enumerate(paths):
        shards[index % len(shards)]["paths"].append(path)
    return shards


def _write_json_atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(str(temporary), str(path))


def _run_root_for_output(output_dir):
    output_dir = Path(output_dir).resolve()
    for candidate in (output_dir,) + tuple(output_dir.parents):
        if (candidate / "spec.json").is_file():
            return candidate
    return output_dir.parent


def _partialflow_sample_count(output_subdir, expected_samples):
    output_subdir = Path(output_subdir)
    return sum(
        (output_subdir / ("sample%d.pdb" % index)).is_file()
        and (output_subdir / ("sample%d.pdb" % index)).stat().st_size > 0
        for index in range(int(expected_samples))
    )


def _partialflow_group_complete(output_subdir, expected_samples):
    """Use the official PPIFlow completion contract: all requested samples exist."""
    return _partialflow_sample_count(output_subdir, expected_samples) == int(
        expected_samples
    )


def _parallel_partialflow_step(original_class):
    """Resume complete parent groups and run one PartialFlow process per idle GPU."""

    class ParallelPartialStep:
        def __init__(self, *args, **kwargs):
            if args:
                raise TypeError(
                    "parallel PartialFlow wrapper requires keyword arguments"
                )
            self.kwargs = dict(kwargs)
            self.serial_step = original_class(**kwargs)

        def run(self, check=True):
            output_dir = Path(self.kwargs["output_dir"]).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            pdbs = [
                Path(path).resolve()
                for path in self.serial_step._get_pdb_files()
            ]
            if not pdbs:
                raise FileNotFoundError(
                    "no PartialFlow parent PDBs found: %s"
                    % self.kwargs["pdb_dir"]
                )
            expected = int(self.serial_step.samples_per_target)
            if expected < 1:
                raise ValueError("PartialFlow samples_per_target must be positive")
            def output_subdir(path):
                return Path(self.serial_step._get_output_subdir(str(path)))

            completed = [
                path
                for path in pdbs
                if _partialflow_group_complete(output_subdir(path), expected)
            ]
            completed_set = set(completed)
            pending = [path for path in pdbs if path not in completed_set]
            existing_samples = sum(
                _partialflow_sample_count(output_subdir(path), expected)
                for path in pdbs
            )
            progress_path = output_dir / "parallel_partialflow_progress.json"
            manifest_path = output_dir / "parallel_partialflow_manifest.json"
            if not pending:
                manifest = {
                    "mode": "parallel_partialflow_idle_gpu_parent_shards",
                    "status": "complete",
                    "total_parent_groups": len(pdbs),
                    "samples_per_parent": expected,
                    "total_samples": len(pdbs) * expected,
                    "completed_parent_groups_before": len(completed),
                    "completed_samples_before": existing_samples,
                    "processed_parent_groups": 0,
                    "selected_gpus": [],
                    "shards": [],
                }
                _write_json_atomic(manifest_path, manifest)
                _write_json_atomic(progress_path, manifest)
                print(
                    "[ParallelPartialFlow] All %d parent groups are already complete"
                    % len(pdbs),
                    flush=True,
                )
                return []

            gpu_ids = _partialflow_gpu_ids()
            shards = _balanced_file_shards(pending, gpu_ids)
            run_root = _run_root_for_output(output_dir)
            logs_root = run_root / "logs" / "partialflow_shards"
            logs_root.mkdir(parents=True, exist_ok=True)
            stage_name = output_dir.parent.name
            workers = []
            handles = []
            shard_records = []
            status = "running"

            def launch(worker):
                path = worker["shard"]["paths"][worker["cursor"]]
                target = output_subdir(path)
                target.mkdir(parents=True, exist_ok=True)
                command = self.serial_step.build_cmd(str(path))
                handle = worker["handle"]
                handle.write(
                    "argv=" + json.dumps(command, ensure_ascii=False) + "\n"
                )
                handle.flush()
                environment = dict(os.environ)
                environment["CUDA_VISIBLE_DEVICES"] = str(
                    worker["shard"]["gpu"]
                )
                worker["current"] = path
                worker["record"]["current_parent"] = path.stem
                worker["process"] = subprocess.Popen(
                    command,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )

            try:
                for shard in shards:
                    log_path = logs_root / (
                        "%s.shard_%02d.gpu_%s.log"
                        % (stage_name, shard["index"], shard["gpu"])
                    )
                    handle = log_path.open("a", encoding="utf-8")
                    record = {
                        "index": shard["index"],
                        "gpu": shard["gpu"],
                        "parent_count": len(shard["paths"]),
                        "parents": [path.stem for path in shard["paths"]],
                        "completed_parent_groups": 0,
                        "current_parent": None,
                        "log": str(log_path),
                    }
                    worker = {
                        "shard": shard,
                        "cursor": 0,
                        "process": None,
                        "current": None,
                        "handle": handle,
                        "record": record,
                    }
                    handles.append(handle)
                    workers.append(worker)
                    shard_records.append(record)
                    launch(worker)

                print(
                    "[ParallelPartialFlow] total_parents=%d total_samples=%d "
                    "completed_parents_before=%d existing_samples=%d pending_parents=%d "
                    "GPUs=%s parent_shards=%s"
                    % (
                        len(pdbs),
                        len(pdbs) * expected,
                        len(completed),
                        existing_samples,
                        len(pending),
                        ",".join(str(item["gpu"]) for item in shard_records),
                        ",".join(
                            str(item["parent_count"]) for item in shard_records
                        ),
                    ),
                    flush=True,
                )
                while True:
                    failed = None
                    running = False
                    for worker in workers:
                        process = worker["process"]
                        if process is None:
                            continue
                        returncode = process.poll()
                        if returncode is None:
                            running = True
                            continue
                        path = worker["current"]
                        if returncode != 0:
                            failed = (worker, returncode)
                            break
                        if not _partialflow_group_complete(
                            output_subdir(path), expected
                        ):
                            failed = (worker, "incomplete output")
                            break
                        worker["record"]["completed_parent_groups"] += 1
                        worker["cursor"] += 1
                        if worker["cursor"] < len(worker["shard"]["paths"]):
                            launch(worker)
                            running = True
                        else:
                            worker["process"] = None
                            worker["current"] = None
                            worker["record"]["current_parent"] = None

                    current_samples = sum(
                        _partialflow_sample_count(output_subdir(path), expected)
                        for path in pdbs
                    )
                    _write_json_atomic(
                        progress_path,
                        {
                            "status": "running" if running and not failed else "validating",
                            "total_parent_groups": len(pdbs),
                            "samples_per_parent": expected,
                            "total_samples": len(pdbs) * expected,
                            "completed_parent_groups_before": len(completed),
                            "completed_samples_before": existing_samples,
                            "pending_parent_groups_at_start": len(pending),
                            "completed_samples_total": current_samples,
                            "selected_gpus": [
                                item["gpu"] for item in shard_records
                            ],
                            "shards": shard_records,
                        },
                    )
                    if failed:
                        worker, returncode = failed
                        raise RuntimeError(
                            "PartialFlow shard %s on GPU %s failed for %s (%s); see %s"
                            % (
                                worker["shard"]["index"],
                                worker["shard"]["gpu"],
                                worker["current"].name,
                                returncode,
                                worker["record"]["log"],
                            )
                        )
                    if not running:
                        break
                    time.sleep(2)

                incomplete = [
                    path.name
                    for path in pdbs
                    if not _partialflow_group_complete(output_subdir(path), expected)
                ]
                if incomplete:
                    raise RuntimeError(
                        "PartialFlow output is incomplete for %d parent groups; first=%s"
                        % (len(incomplete), incomplete[0])
                    )
                status = "complete"
                manifest = {
                    "mode": "parallel_partialflow_idle_gpu_parent_shards",
                    "status": status,
                    "total_parent_groups": len(pdbs),
                    "samples_per_parent": expected,
                    "total_samples": len(pdbs) * expected,
                    "completed_parent_groups_before": len(completed),
                    "completed_samples_before": existing_samples,
                    "processed_parent_groups": len(pending),
                    "selected_gpus": [item["gpu"] for item in shard_records],
                    "shards": shard_records,
                }
                _write_json_atomic(manifest_path, manifest)
                _write_json_atomic(progress_path, manifest)
            except BaseException:
                status = "failed"
                for worker in workers:
                    process = worker["process"]
                    if process is not None and process.poll() is None:
                        process.terminate()
                for worker in workers:
                    process = worker["process"]
                    if process is None:
                        continue
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                _write_json_atomic(
                    progress_path,
                    {
                        "status": status,
                        "total_parent_groups": len(pdbs),
                        "samples_per_parent": expected,
                        "total_samples": len(pdbs) * expected,
                        "completed_parent_groups_before": len(completed),
                        "completed_samples_before": existing_samples,
                        "pending_parent_groups_at_start": len(pending),
                        "selected_gpus": [item["gpu"] for item in shard_records],
                        "shards": shard_records,
                    },
                )
                raise
            finally:
                for handle in handles:
                    handle.close()

            return [
                subprocess.CompletedProcess(
                    args=["parallel_partialflow_idle_gpu_parent_shards"],
                    returncode=0,
                )
                for _ in shards
            ]

    return ParallelPartialStep


def _parallel_abmpnn_step(original_class):
    """Resume valid FASTAs and distribute the remaining PDBs over idle GPUs."""

    class ParallelAbMPNNStep:
        def __init__(self, *args, **kwargs):
            if args:
                raise TypeError("parallel AbMPNN wrapper requires keyword arguments")
            self.kwargs = dict(kwargs)
            self.serial_step = original_class(**kwargs)

        def run(self, check=True):
            output_dir = Path(self.kwargs["output_dir"]).resolve()
            input_dir = Path(self.kwargs["folder_with_pdbs"]).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)

            fix_csv_cmd = self.serial_step._build_fix_csv_cmd()
            print(
                "[ParallelAbMPNN] Running make_fix_csv: %s"
                % " ".join(fix_csv_cmd),
                flush=True,
            )
            fix_csv_result = subprocess.run(fix_csv_cmd, check=check)
            if self.serial_step.fix_positions:
                self.serial_step._merge_fix_positions_into_csv()

            pdbs = sorted(input_dir.glob("*.pdb"), key=lambda path: path.name)
            if not pdbs:
                raise FileNotFoundError("no AbMPNN input PDBs found: %s" % input_dir)
            expected = int(self.serial_step.num_seq_per_target)
            final_seqs = output_dir / "seqs"
            final_seqs.mkdir(parents=True, exist_ok=True)
            restored = _restore_native_abmpnn_fastas(output_dir, pdbs, expected)
            if restored:
                print(
                    "[ParallelAbMPNN] Restored %d native FASTAs from canonical output"
                    % restored,
                    flush=True,
                )
            completed = [
                path
                for path in pdbs
                if _valid_abmpnn_fasta(final_seqs / (path.stem + ".fa"), expected)
            ]
            completed_set = set(completed)
            pending = [path for path in pdbs if path not in completed_set]
            if not pending:
                print(
                    "[ParallelAbMPNN] All %d backbones already have %d valid sequences"
                    % (len(pdbs), expected),
                    flush=True,
                )
                return {
                    "make_fix_csv": fix_csv_result,
                    "abmpnn": subprocess.CompletedProcess(
                        args=["parallel_abmpnn_resume"], returncode=0
                    ),
                }

            gpu_ids = _abmpnn_gpu_ids()
            shards = _balanced_file_shards(pending, gpu_ids)
            run_root = _run_root_for_output(output_dir)
            logs_root = run_root / "logs" / "abmpnn_shards"
            logs_root.mkdir(parents=True, exist_ok=True)
            job_key = "%s_%s" % (output_dir.parent.name, output_dir.name)
            work_root = output_dir.parent / (".%s.abmpnn_shards" % output_dir.name)
            if work_root.exists():
                shutil.rmtree(str(work_root))
            work_root.mkdir(parents=True)
            progress_path = output_dir / "parallel_abmpnn_progress.json"
            manifest_path = output_dir / "parallel_abmpnn_manifest.json"

            fixed_positions_path = Path(
                self.serial_step._fixed_positions_csv_path()
            )
            with fixed_positions_path.open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                reader = csv.DictReader(handle)
                fixed_fields = list(reader.fieldnames or [])
                fixed_rows = {row["pdb_name"]: row for row in reader}
            expected_names = {path.stem for path in pdbs}
            missing_fixed = sorted(expected_names - set(fixed_rows))
            if missing_fixed:
                raise RuntimeError(
                    "fixed_positions.csv is missing %d AbMPNN inputs; first=%s"
                    % (len(missing_fixed), missing_fixed[0])
                )

            processes = []
            handles = []
            shard_records = []
            status = "running"
            try:
                for shard in shards:
                    shard_root = work_root / ("shard_%02d_gpu_%s" % (shard["index"], shard["gpu"]))
                    shard_input = shard_root / "inputs"
                    shard_output = shard_root / "output"
                    shard_input.mkdir(parents=True)
                    shard_output.mkdir(parents=True)
                    names = {path.stem for path in shard["paths"]}
                    for source in shard["paths"]:
                        (shard_input / source.name).symlink_to(source)
                    with (shard_input / "fixed_positions.csv").open(
                        "w", encoding="utf-8", newline=""
                    ) as handle:
                        writer = csv.DictWriter(handle, fieldnames=fixed_fields)
                        writer.writeheader()
                        for name in sorted(names):
                            writer.writerow(fixed_rows[name])
                    shard_kwargs = dict(self.kwargs)
                    shard_kwargs.update(
                        {
                            "folder_with_pdbs": str(shard_input),
                            "output_dir": str(shard_output),
                        }
                    )
                    command = original_class(**shard_kwargs)._build_abmpnn_cmd()
                    log_path = logs_root / (
                        "%s.shard_%02d.gpu_%s.log"
                        % (job_key, shard["index"], shard["gpu"])
                    )
                    log_handle = log_path.open("a", encoding="utf-8")
                    log_handle.write("argv=" + json.dumps(command, ensure_ascii=False) + "\n")
                    log_handle.flush()
                    environment = dict(os.environ)
                    environment["CUDA_VISIBLE_DEVICES"] = str(shard["gpu"])
                    process = subprocess.Popen(
                        command,
                        env=environment,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                    )
                    record = {
                        "index": shard["index"],
                        "gpu": shard["gpu"],
                        "count": len(shard["paths"]),
                        "log": str(log_path),
                    }
                    processes.append((shard, process, shard_output))
                    handles.append(log_handle)
                    shard_records.append(record)

                print(
                    "[ParallelAbMPNN] total=%d completed_before=%d pending=%d GPUs=%s shards=%s"
                    % (
                        len(pdbs),
                        len(completed),
                        len(pending),
                        ",".join(str(item) for item in gpu_ids[: len(shards)]),
                        ",".join(str(item["count"]) for item in shard_records),
                    ),
                    flush=True,
                )
                while True:
                    failed = None
                    running = False
                    completed_now = 0
                    for shard, process, shard_output in processes:
                        returncode = process.poll()
                        if returncode is None:
                            running = True
                        elif returncode != 0 and failed is None:
                            failed = (shard, returncode)
                        completed_now += sum(
                            _valid_abmpnn_fasta(path, expected)
                            for path in (shard_output / "seqs").glob("*.fa")
                        )
                    _write_json_atomic(
                        progress_path,
                        {
                            "status": "running" if running and not failed else "merging",
                            "total_backbones": len(pdbs),
                            "completed_before": len(completed),
                            "pending_at_start": len(pending),
                            "completed_in_shards": completed_now,
                            "selected_gpus": gpu_ids[: len(shards)],
                            "shards": shard_records,
                        },
                    )
                    if failed:
                        shard, returncode = failed
                        raise RuntimeError(
                            "AbMPNN shard %s on GPU %s failed with exit code %s; see %s"
                            % (
                                shard["index"],
                                shard["gpu"],
                                returncode,
                                shard_records[shard["index"]]["log"],
                            )
                        )
                    if not running:
                        break
                    time.sleep(2)

                for shard, _, shard_output in processes:
                    for source_pdb in shard["paths"]:
                        source = shard_output / "seqs" / (source_pdb.stem + ".fa")
                        if not _valid_abmpnn_fasta(source, expected):
                            raise RuntimeError(
                                "AbMPNN shard produced an incomplete FASTA: %s" % source
                            )
                        destination = final_seqs / source.name
                        temporary = destination.with_name(destination.name + ".tmp")
                        shutil.copy2(str(source), str(temporary))
                        os.replace(str(temporary), str(destination))

                invalid = [
                    path.name
                    for path in pdbs
                    if not _valid_abmpnn_fasta(
                        final_seqs / (path.stem + ".fa"), expected
                    )
                ]
                if invalid:
                    raise RuntimeError(
                        "AbMPNN merge is incomplete for %d backbones; first=%s"
                        % (len(invalid), invalid[0])
                    )
                status = "complete"
                manifest = {
                    "mode": "parallel_abmpnn_idle_gpu_shards",
                    "total_backbones": len(pdbs),
                    "completed_before": len(completed),
                    "processed": len(pending),
                    "sequences_per_backbone": expected,
                    "total_sequences": len(pdbs) * expected,
                    "selected_gpus": gpu_ids[: len(shards)],
                    "shards": shard_records,
                }
                _write_json_atomic(manifest_path, manifest)
                _write_json_atomic(progress_path, {"status": "complete", **manifest})
            except BaseException:
                status = "failed"
                for _, process, _ in processes:
                    if process.poll() is None:
                        process.terminate()
                for _, process, _ in processes:
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                _write_json_atomic(
                    progress_path,
                    {
                        "status": status,
                        "total_backbones": len(pdbs),
                        "completed_before": len(completed),
                        "pending_at_start": len(pending),
                        "selected_gpus": gpu_ids[: len(shards)],
                        "shards": shard_records,
                    },
                )
                raise
            finally:
                for handle in handles:
                    handle.close()
                if status == "complete" and work_root.exists():
                    shutil.rmtree(str(work_root))

            return {
                "make_fix_csv": fix_csv_result,
                "abmpnn": subprocess.CompletedProcess(
                    args=["parallel_abmpnn_idle_gpu_shards"], returncode=0
                ),
            }

    return ParallelAbMPNNStep


def _flowpacker_output_name(row):
    link_name = Path((row.get("link_name") or "").strip())
    sequence_index = str(row.get("seq_idx") or "").strip()
    if not link_name.name or not sequence_index:
        raise ValueError("FlowPacker CSV row is missing link_name or seq_idx")
    return "%s_%s.pdb" % (link_name.stem, sequence_index)


def _flowpacker_sample_count(config_path):
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    count = int((config.get("sample") or {}).get("n_samples", 1))
    if count < 1:
        raise ValueError("FlowPacker sample.n_samples must be positive")
    return count


def _flowpacker_row_complete(output_dir, row, sample_count):
    name = _flowpacker_output_name(row)
    return all(
        (Path(output_dir) / ("run_%d" % sample) / name).is_file()
        and (Path(output_dir) / ("run_%d" % sample) / name).stat().st_size > 0
        for sample in range(1, sample_count + 1)
    )


def _copy_atomic(source, destination):
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copy2(str(source), str(temporary))
    os.replace(str(temporary), str(destination))


def _merge_flowpacker_shard_outputs(work_root, output_dir):
    """Recover completed shard files without trusting a shared write directory."""
    work_root = Path(work_root)
    output_dir = Path(output_dir)
    copied = 0
    if not work_root.is_dir():
        return copied
    for shard_output in sorted(work_root.glob("shard_*/output")):
        for run_dir in sorted(shard_output.glob("run_*")):
            for source in sorted(run_dir.glob("*.pdb")):
                destination = output_dir / run_dir.name / source.name
                if not destination.is_file() or destination.stat().st_size == 0:
                    _copy_atomic(source, destination)
                    copied += 1
        after_pdbs = shard_output / "after_pdbs"
        if after_pdbs.is_dir():
            for source in sorted(after_pdbs.glob("*.pdb")):
                destination = output_dir / "after_pdbs" / source.name
                if not destination.is_file() or destination.stat().st_size == 0:
                    _copy_atomic(source, destination)
        after_batches = shard_output / "after_pdbs_batch"
        if after_batches.is_dir():
            for source in sorted(after_batches.rglob("*.pdb")):
                destination = output_dir / "after_pdbs_batch" / source.relative_to(
                    after_batches
                )
                if not destination.is_file() or destination.stat().st_size == 0:
                    _copy_atomic(source, destination)
    return copied


def _balanced_flowpacker_shards(rows, gpu_ids):
    """Keep every backbone's pending designs together while balancing backbones."""
    groups = {}
    order = []
    for row in rows:
        key = (row.get("link_name") or "").strip()
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)
    active = list(gpu_ids[: max(1, min(len(gpu_ids), len(order)))])
    shards = [
        {"index": index, "gpu": gpu, "backbones": [], "rows": []}
        for index, gpu in enumerate(active)
    ]
    for index, key in enumerate(order):
        shard = shards[index % len(shards)]
        shard["backbones"].append(key)
        shard["rows"].extend(groups[key])
    return shards


def _parallel_flowpacker_step(original_class):
    """Resume FlowPacker and run one isolated, balanced shard per idle GPU."""

    class ParallelFlowpackerStep:
        def __init__(self, *args, **kwargs):
            if args:
                raise TypeError("parallel FlowPacker wrapper requires keyword arguments")
            self.kwargs = dict(kwargs)

        def run(self, check=True):
            output_dir = Path(self.kwargs["output_dir"]).resolve()
            csv_path = Path(self.kwargs["csv_path"]).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            run_root = _run_root_for_output(output_dir)
            stage_name = output_dir.parent.name
            work_root = (
                run_root / ".agent_work" / stage_name / "flowpacker_shards"
            )
            logs_root = run_root / "logs" / "flowpacker_shards"
            logs_root.mkdir(parents=True, exist_ok=True)
            sample_count = _flowpacker_sample_count(self.kwargs["config_path"])

            recovered = _merge_flowpacker_shard_outputs(work_root, output_dir)
            if work_root.exists():
                shutil.rmtree(str(work_root))
            work_root.mkdir(parents=True)

            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = list(reader.fieldnames or [])
                rows = list(reader)
            for required in ("link_name", "sequence_dict", "seq_idx"):
                if required not in fieldnames:
                    raise ValueError("FlowPacker CSV is missing column: %s" % required)
            if not rows:
                raise ValueError("FlowPacker CSV contains no designs: %s" % csv_path)

            completed = [
                row
                for row in rows
                if _flowpacker_row_complete(output_dir, row, sample_count)
            ]
            completed_ids = {
                ((row.get("link_name") or "").strip(), str(row.get("seq_idx") or "").strip())
                for row in completed
            }
            pending = [
                row
                for row in rows
                if (
                    (row.get("link_name") or "").strip(),
                    str(row.get("seq_idx") or "").strip(),
                )
                not in completed_ids
            ]
            progress_path = output_dir / "parallel_flowpacker_progress.json"
            manifest_path = output_dir / "parallel_flowpacker_manifest.json"
            if not pending:
                manifest = {
                    "mode": "parallel_flowpacker_idle_gpu_shards",
                    "status": "complete",
                    "total_backbones": len(
                        {(row.get("link_name") or "").strip() for row in rows}
                    ),
                    "total_designs": len(rows),
                    "completed_before": len(completed),
                    "processed": 0,
                    "recovered_from_shards": recovered,
                    "samples_per_design": sample_count,
                    "selected_gpus": [],
                    "shards": [],
                }
                _write_json_atomic(manifest_path, manifest)
                _write_json_atomic(progress_path, manifest)
                shutil.rmtree(str(work_root))
                print(
                    "[ParallelFlowPacker] All %d designs are already complete"
                    % len(rows),
                    flush=True,
                )
                return subprocess.CompletedProcess(
                    args=["parallel_flowpacker_resume"], returncode=0
                )

            gpu_ids = _flowpacker_gpu_ids()
            shards = _balanced_flowpacker_shards(pending, gpu_ids)
            processes = []
            handles = []
            shard_records = []
            status = "running"
            try:
                for shard in shards:
                    shard_root = work_root / (
                        "shard_%02d_gpu_%s" % (shard["index"], shard["gpu"])
                    )
                    shard_output = shard_root / "output"
                    shard_root.mkdir(parents=True, exist_ok=True)
                    shard_csv = shard_root / "input.csv"
                    with shard_csv.open("w", encoding="utf-8", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(shard["rows"])
                    shard_kwargs = dict(self.kwargs)
                    shard_kwargs.update(
                        {"csv_path": str(shard_csv), "output_dir": str(shard_output)}
                    )
                    command = original_class(**shard_kwargs).build_cmd()
                    log_path = logs_root / (
                        "%s.shard_%02d.gpu_%s.log"
                        % (stage_name, shard["index"], shard["gpu"])
                    )
                    log_handle = log_path.open("a", encoding="utf-8")
                    log_handle.write("argv=" + json.dumps(command, ensure_ascii=False) + "\n")
                    log_handle.flush()
                    environment = dict(os.environ)
                    environment["CUDA_VISIBLE_DEVICES"] = str(shard["gpu"])
                    environment["PATH"] = (
                        str(Path(command[1]).resolve().parent)
                        + os.pathsep
                        + environment.get("PATH", "")
                    )
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
                        "backbone_count": len(shard["backbones"]),
                        "design_count": len(shard["rows"]),
                        "log": str(log_path),
                    }
                    processes.append((shard, process, shard_output))
                    handles.append(log_handle)
                    shard_records.append(record)

                print(
                    "[ParallelFlowPacker] total_backbones=%d total_designs=%d "
                    "completed_before=%d pending=%d GPUs=%s backbone_shards=%s "
                    "design_shards=%s"
                    % (
                        len({(row.get("link_name") or "").strip() for row in rows}),
                        len(rows),
                        len(completed),
                        len(pending),
                        ",".join(str(item["gpu"]) for item in shard_records),
                        ",".join(str(item["backbone_count"]) for item in shard_records),
                        ",".join(str(item["design_count"]) for item in shard_records),
                    ),
                    flush=True,
                )
                while True:
                    failed = None
                    running = False
                    finished_now = 0
                    for shard, process, shard_output in processes:
                        returncode = process.poll()
                        if returncode is None:
                            running = True
                        elif returncode != 0 and failed is None:
                            failed = (shard, returncode)
                        finished_now += sum(
                            _flowpacker_row_complete(
                                shard_output, row, sample_count
                            )
                            for row in shard["rows"]
                        )
                    _write_json_atomic(
                        progress_path,
                        {
                            "status": "running" if running and not failed else "merging",
                            "total_backbones": len(
                                {(row.get("link_name") or "").strip() for row in rows}
                            ),
                            "total_designs": len(rows),
                            "completed_before": len(completed),
                            "pending_at_start": len(pending),
                            "completed_in_shards": finished_now,
                            "recovered_from_shards": recovered,
                            "selected_gpus": [item["gpu"] for item in shard_records],
                            "shards": shard_records,
                        },
                    )
                    if failed:
                        shard, returncode = failed
                        raise RuntimeError(
                            "FlowPacker shard %s on GPU %s failed with exit code %s; see %s"
                            % (
                                shard["index"],
                                shard["gpu"],
                                returncode,
                                shard_records[shard["index"]]["log"],
                            )
                        )
                    if not running:
                        break
                    time.sleep(2)

                merged = _merge_flowpacker_shard_outputs(work_root, output_dir)
                incomplete = [
                    _flowpacker_output_name(row)
                    for row in rows
                    if not _flowpacker_row_complete(output_dir, row, sample_count)
                ]
                if incomplete:
                    raise RuntimeError(
                        "FlowPacker merge is incomplete for %d designs; first=%s"
                        % (len(incomplete), incomplete[0])
                    )
                status = "complete"
                manifest = {
                    "mode": "parallel_flowpacker_idle_gpu_shards",
                    "status": status,
                    "total_backbones": len(
                        {(row.get("link_name") or "").strip() for row in rows}
                    ),
                    "total_designs": len(rows),
                    "completed_before": len(completed),
                    "processed": len(pending),
                    "recovered_from_shards": recovered,
                    "merged_from_current_shards": merged,
                    "samples_per_design": sample_count,
                    "selected_gpus": [item["gpu"] for item in shard_records],
                    "shards": shard_records,
                }
                _write_json_atomic(manifest_path, manifest)
                _write_json_atomic(progress_path, manifest)
            except BaseException:
                status = "failed"
                for _, process, _ in processes:
                    if process.poll() is None:
                        process.terminate()
                for _, process, _ in processes:
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                _write_json_atomic(
                    progress_path,
                    {
                        "status": status,
                        "total_designs": len(rows),
                        "completed_before": len(completed),
                        "pending_at_start": len(pending),
                        "recovered_from_shards": recovered,
                        "selected_gpus": [item["gpu"] for item in shard_records],
                        "shards": shard_records,
                    },
                )
                raise
            finally:
                for handle in handles:
                    handle.close()
                if status == "complete" and work_root.exists():
                    shutil.rmtree(str(work_root))

            return subprocess.CompletedProcess(
                args=["parallel_flowpacker_idle_gpu_shards"], returncode=0
            )

    return ParallelFlowpackerStep


def _write_shard_config(source, destination, seed):
    """Preserve 100 sampling steps while giving every worker a unique seed."""
    with Path(source).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    sampling = config.get("interpolant", {}).get("sampling", {})
    steps = int(sampling.get("num_timesteps", 0))
    if steps != 100:
        raise RuntimeError(
            "parallel PPIFlow backbone generation requires num_timesteps=100; "
            "found %s in %s" % (steps, source)
        )
    config.setdefault("experiment", {})["seed"] = int(seed)
    config.setdefault("ppi_dataset", {})["seed"] = int(seed)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return steps


def _local_sample_id(filename, shard_name):
    match = re.fullmatch(re.escape(shard_name) + r"_(\d+)\.pdb", Path(filename).name)
    if not match:
        raise RuntimeError(
            "unexpected PPIFlow shard filename %r (expected %s_<id>.pdb)"
            % (Path(filename).name, shard_name)
        )
    return int(match.group(1))


def _merge_parallel_shards(output_dir, original_name, shards):
    """Validate and merge shard PDBs/metrics into one globally numbered set."""
    output_dir = Path(output_dir)
    merged_dir = output_dir.parent / ".ppiflow_backbone_merged"
    if merged_dir.exists():
        shutil.rmtree(str(merged_dir))
    merged_dir.mkdir(parents=True)

    expected_global_ids = {item["offset"] + i for item in shards for i in range(item["count"])}
    observed_global_ids = set()
    metric_rows = []
    metric_fields = None

    for shard in shards:
        shard_dir = Path(shard["output_dir"])
        shard_name = shard["name"]
        pdb_files = sorted(shard_dir.glob("*.pdb"))
        local_ids = {_local_sample_id(path.name, shard_name) for path in pdb_files}
        expected_local_ids = set(range(shard["count"]))
        if local_ids != expected_local_ids:
            raise RuntimeError(
                "PPIFlow shard %s on GPU %s is incomplete: expected local ids %s, got %s"
                % (
                    shard["index"],
                    shard["gpu"],
                    sorted(expected_local_ids),
                    sorted(local_ids),
                )
            )
        for pdb_path in pdb_files:
            local_id = _local_sample_id(pdb_path.name, shard_name)
            global_id = shard["offset"] + local_id
            if global_id in observed_global_ids:
                raise RuntimeError("duplicate global PPIFlow sample id %s" % global_id)
            observed_global_ids.add(global_id)
            destination = merged_dir / ("backbone.0_%s.pdb" % global_id)
            shutil.copy2(str(pdb_path), str(destination))

        metrics_path = shard_dir / "sample_metrics.csv"
        if not metrics_path.is_file():
            raise FileNotFoundError("PPIFlow shard metrics not found: %s" % metrics_path)
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            if metric_fields is None:
                metric_fields = fields
            elif fields != metric_fields:
                raise RuntimeError("PPIFlow shard metric columns do not match")
            shard_metric_ids = set()
            for row in reader:
                local_id = _local_sample_id(row.get("sample", ""), shard_name)
                if local_id in shard_metric_ids:
                    raise RuntimeError(
                        "duplicate metric row for shard %s local id %s"
                        % (shard["index"], local_id)
                    )
                shard_metric_ids.add(local_id)
                global_id = shard["offset"] + local_id
                final_name = "backbone.0_%s.pdb" % global_id
                row["sample"] = final_name
                if "pdb_path" in row:
                    row["pdb_path"] = str((output_dir / final_name).resolve())
                metric_rows.append((global_id, row))
            if shard_metric_ids != expected_local_ids:
                raise RuntimeError(
                    "PPIFlow shard %s metric ids do not match its PDB ids"
                    % shard["index"]
                )

    if observed_global_ids != expected_global_ids:
        raise RuntimeError(
            "merged PPIFlow ids are incomplete: expected %s, got %s"
            % (sorted(expected_global_ids), sorted(observed_global_ids))
        )
    if len(metric_rows) != len(expected_global_ids):
        raise RuntimeError(
            "merged PPIFlow metric row count %s does not match PDB count %s"
            % (len(metric_rows), len(expected_global_ids))
        )

    metrics_output = merged_dir / "sample_metrics.csv"
    with metrics_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        writer.writeheader()
        for _, row in sorted(metric_rows, key=lambda item: item[0]):
            writer.writerow(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    for old_pdb in output_dir.glob(original_name + "_*.pdb"):
        old_pdb.unlink()
    for old_pdb in output_dir.glob("backbone.0_*.pdb"):
        old_pdb.unlink()
    old_metrics = output_dir / "sample_metrics.csv"
    if old_metrics.exists():
        old_metrics.unlink()
    for merged_file in merged_dir.iterdir():
        shutil.move(str(merged_file), str(output_dir / merged_file.name))
    shutil.rmtree(str(merged_dir))
    return len(observed_global_ids)


def _parallel_ppiflow_step(original_class):
    """Wrap PPIFlow's generation step with safe multi-GPU sharding."""

    class ParallelPPIFlowStep:
        def __init__(self, *args, **kwargs):
            if args:
                raise TypeError("parallel PPIFlow wrapper requires keyword arguments")
            self.kwargs = dict(kwargs)
            self.serial_step = original_class(**kwargs)

        def run(self, check=True):
            gpu_ids = _parallel_gpu_ids()
            total = int(self.kwargs.get("samples_per_target", 100))
            config_path = self.kwargs.get("config")
            if len(gpu_ids) < 2 or total < 2 or not config_path:
                return self.serial_step.run(check=check)

            output_dir = Path(self.kwargs["output_dir"])
            original_name = self.kwargs["name"]
            shards_root = output_dir.parent / ".ppiflow_backbone_shards"
            logs_root = output_dir.parents[2] / "logs" / "ppiflow_backbone_shards"
            shards_root.mkdir(parents=True, exist_ok=True)
            logs_root.mkdir(parents=True, exist_ok=True)

            with Path(config_path).open("r", encoding="utf-8") as handle:
                base_config = yaml.safe_load(handle)
            base_seed = int(base_config.get("experiment", {}).get("seed", 123))
            resume_plan_path = output_dir.parents[2] / "ppiflow" / "backbone_resume_plan.json"
            resume_plan = json.loads(resume_plan_path.read_text()) if resume_plan_path.exists() else None
            shards = _balanced_shards(total, gpu_ids)
            if resume_plan is not None:
                if resume_plan["requested_backbones"] != total:
                    raise ValueError("Resume plan target differs from task target")
                shards = [dict(item) for item in resume_plan["shards"]]
                if sum(item["count"] for item in shards) != total:
                    raise ValueError("Resume shard counts differ from task target")
                gpu_ids = [item["gpu"] for item in shards]
            processes = []
            log_handles = []
            try:
                for shard in shards:
                    shard_name = "backbone.0"
                    shard_dir = shards_root / ("shard_%02d_gpu_%s" % (shard["index"], shard["gpu"]))
                    if resume_plan is not None:
                        if shard_dir.resolve() != Path(shard["output_dir"]).resolve() or not shard_dir.is_dir():
                            raise ValueError("Resume shard directory mismatch")
                    else:
                        if shard_dir.exists():
                            shutil.rmtree(str(shard_dir))
                        shard_dir.mkdir(parents=True)
                    seed = shard.get("seed", base_seed + shard["offset"])
                    shard_config = logs_root / ("shard_%02d.config.yaml" % shard["index"])
                    steps = _write_shard_config(config_path, shard_config, seed)
                    shard.update(
                        {
                            "name": shard_name,
                            "output_dir": str(shard_dir),
                            "seed": seed,
                            "num_timesteps": steps,
                            "log": str(logs_root / ("shard_%02d.gpu_%s.log" % (shard["index"], shard["gpu"]))),
                        }
                    )
                    shard_kwargs = dict(self.kwargs)
                    shard_kwargs.update(
                        {
                            "name": shard_name,
                            "output_dir": str(shard_dir),
                            "samples_per_target": shard["count"],
                            "config": str(shard_config),
                        }
                    )
                    command = original_class(**shard_kwargs).build_cmd()
                    if resume_plan is not None:
                        command.extend(["--resume-shard", str(shard_dir), "--resume-count", str(shard["count"])])
                    environment = dict(os.environ)
                    environment["CUDA_VISIBLE_DEVICES"] = str(shard["gpu"])
                    environment["PYTHONHASHSEED"] = str(seed)
                    log_handle = Path(shard["log"]).open("a", encoding="utf-8")
                    log_handle.write("argv=" + json.dumps(command, ensure_ascii=False) + "\n")
                    log_handle.flush()
                    process = subprocess.Popen(
                        command,
                        env=environment,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                    )
                    processes.append((shard, process))
                    log_handles.append(log_handle)

                failed = None
                while True:
                    running = False
                    for shard, process in processes:
                        returncode = process.poll()
                        if returncode is None:
                            running = True
                        elif returncode != 0 and failed is None:
                            failed = (shard, returncode)
                    if failed or not running:
                        break
                    time.sleep(1)
                if failed:
                    for _, process in processes:
                        if process.poll() is None:
                            process.terminate()
                    for _, process in processes:
                        process.wait()
                    shard, returncode = failed
                    raise RuntimeError(
                        "PPIFlow backbone shard %s on GPU %s failed with exit code %s; see %s"
                        % (shard["index"], shard["gpu"], returncode, shard["log"])
                    )
                for _, process in processes:
                    process.wait()
            finally:
                for handle in log_handles:
                    handle.close()

            merged_count = _merge_parallel_shards(output_dir, original_name, shards)
            manifest = {
                "mode": "parallel_ppiflow_backbone_generation",
                "requested_backbones": total,
                "merged_backbones": merged_count,
                "gpu_ids": gpu_ids[: len(shards)],
                "num_timesteps": 100,
                "shards": shards,
            }
            (output_dir / "parallel_generation_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            shutil.rmtree(str(shards_root))
            return subprocess.CompletedProcess(
                args=["parallel_ppiflow_backbone_generation"], returncode=0
            )

    return ParallelPPIFlowStep


def _request_backbone_top_n(output_root, accepted_count, profile_name):
    """Resolve a saved selection or require a user Top-N choice."""
    output_root = Path(output_root)
    selection_path = output_root / "backbone_selection.json"
    if selection_path.is_file():
        saved = json.loads(selection_path.read_text(encoding="utf-8"))
        top_n = int(saved["selected_top_n"])
        if 1 <= top_n <= accepted_count:
            return top_n
        raise ValueError(
            "saved %s top-N selection %s is outside the current passing range 1..%s; "
            "remove %s and rerun"
            % (profile_name, top_n, accepted_count, selection_path)
        )
    table_name = (
        "backbone_hotspot_distances_sorted.csv"
        if profile_name == "vhh"
        else "scfv_backbone_hotspot_distances_sorted.csv"
    )
    table = output_root / table_name
    run_id = output_root.parents[2].name
    display_name = "VHH" if profile_name == "vhh" else "scFv"
    prompt = (
        "%s骨架距离筛选和排序已完成，共有%s个骨架通过初筛。"
        "请人工复核排序表后，告诉我保留前多少号（输入1到%s之间的整数），"
        "再继续AbMPNN、FlowPacker、复合物结构预测和后续筛选。"
        % (display_name, accepted_count, accepted_count)
    )
    request = {
        "state": "awaiting_user_input",
        "request_type": "%s_backbone_top_n" % profile_name,
        "run_id": run_id,
        "prompt": prompt,
        "ranked_table": str(table.resolve()),
        "accepted_backbones": int(accepted_count),
        "minimum_top_n": 1,
        "maximum_top_n": int(accepted_count),
        "response_command": (
            "scripts/agent-ab select-backbones %s <N> --continue --gpu auto"
            % run_id
        ),
        "continue_command": "scripts/agent-ab pipeline %s" % run_id,
    }
    request_path = output_root / "selection_request.json"
    request_path.write_text(
        json.dumps(request, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    raise BackboneSelectionRequired(request)

def _split_structure_and_seq_idx_preserving_dots(pdbname):
    """Split the final sequence suffix without treating dots as extensions."""
    name = os.path.basename(str(pdbname))
    for suffix in (".mmcif", ".pdb", ".cif"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break

    structure_name, separator, seq_idx_text = name.rpartition("_")
    if separator:
        try:
            return structure_name, int(seq_idx_text)
        except ValueError:
            pass
    return name, None


def _stable_backbone_id(structure_name):
    """Return the run-independent ``backbone.<batch>_<id>`` suffix."""
    try:
        stage, _, _ = parse_backbone_id(structure_name)
    except ValueError:
        return os.path.basename(str(structure_name)).lower()
    if stage != 1:
        raise ValueError("expected a Stage 1 backbone id: %s" % structure_name)
    return canonical_backbone_id(structure_name)


def _load_mpnn_templates_with_backbone_aliases(native_loader):
    """Add stable aliases for MPNN rows whose link names include a run prefix."""

    def load_with_aliases(mpnn_seqs_csv):
        sequence_rows, first_templates = native_loader(mpnn_seqs_csv)

        for mapping_name, mapping in (
            ("sequence rows", sequence_rows),
            ("first templates", first_templates),
        ):
            for source_name, value in list(mapping.items()):
                alias = _stable_backbone_id(source_name)
                if alias == source_name:
                    continue
                existing = mapping.get(alias)
                if existing is not None and existing != value:
                    raise RuntimeError(
                        "conflicting %s for normalized backbone id %s "
                        "(source %s)"
                        % (mapping_name, alias, source_name)
                    )
                mapping[alias] = value

        return sequence_rows, first_templates

    return load_with_aliases


def _require_merge_sequences(native_builder):
    """Fail closed when energetic fixed positions lack sequence identities."""

    def build_and_validate(*args, **kwargs):
        output_csv = kwargs.get("output_csv")
        if output_csv is None and len(args) >= 2:
            output_csv = args[1]

        result = native_builder(*args, **kwargs)
        missing = []
        with Path(output_csv).open(newline="") as handle:
            for row in csv.DictReader(handle):
                fixed_positions = (row.get("fixed_positions") or "").strip()
                if not fixed_positions or fixed_positions.upper() == "NONE":
                    continue
                merge_seq_text = (row.get("merge_seq") or "").strip()
                try:
                    merge_seq = ast.literal_eval(merge_seq_text)
                except (SyntaxError, ValueError):
                    merge_seq = None
                if not isinstance(merge_seq, dict) or not any(merge_seq.values()):
                    missing.append((row.get("filename") or "<unknown>").strip())

        if missing:
            raise RuntimeError(
                "fixed residues have no merged sequence identity for: %s"
                % ", ".join(missing)
            )
        return result

    return build_and_validate


def _collect_partial_outputs_with_stage2_names(native_collector):
    """Canonicalize Stage 2 PDB/CSV names as ``backbone.1_<parent>_<sample>``."""

    def collect_with_stage2_names(*args, **kwargs):
        csv_path = Path(native_collector(*args, **kwargs))
        rows = []
        renamed = {}
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            for row in reader:
                old_name = (row.get("link_name") or "").strip()
                stage, parent_index, sample_index = parse_backbone_id(old_name)
                if stage != 2:
                    raise ValueError("expected a Stage 2 backbone id: %s" % old_name)
                new_name = stage2_backbone_id(parent_index, sample_index) + ".pdb"
                old_path = csv_path.parent / old_name
                new_path = csv_path.parent / new_name
                # The collector emits one CSV row per designed sequence, so the
                # same partial-backbone link appears repeatedly (normally four
                # times). Rename each physical link only once while rewriting
                # every sequence row. The existence checks also make recovery
                # idempotent after an interrupted earlier rename.
                if old_name not in renamed and old_path != new_path:
                    old_exists = old_path.exists() or old_path.is_symlink()
                    new_exists = new_path.exists() or new_path.is_symlink()
                    if old_exists:
                        if new_exists:
                            new_path.unlink()
                        old_path.replace(new_path)
                    elif not new_exists:
                        # An older interrupted rename may have removed both the
                        # legacy link and its new name. The immutable Partial
                        # Flow PDB is still present, so reconstruct the canonical
                        # link instead of rerunning backbone or sequence design.
                        partial_pdb = (
                            csv_path.parent.parent
                            / "partial_output"
                            / ("backbone.0_%d" % parent_index)
                            / ("sample%d.pdb" % sample_index)
                        )
                        if not partial_pdb.is_file():
                            raise FileNotFoundError(
                                "missing Stage 2 backbone link and Partial Flow source: "
                                "%s; %s" % (old_path, partial_pdb)
                            )
                        new_path.symlink_to(partial_pdb.resolve())
                    renamed[old_name] = new_name
                row["link_name"] = new_name
                rows.append(row)

        temporary = csv_path.with_name(csv_path.name + ".tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(csv_path)
        return str(csv_path)

    return collect_with_stage2_names


def _official_partial_inputs_with_fixed_positions_only(native_link, logger):
    """Run PPIFlow's official merge_seq linker, then gate groups without keys."""

    def link_fixed_only(fixed_positions_csv, mpnn_pdbs_dir, before_partial_pdbs_dir):
        native_link(
            fixed_positions_csv=fixed_positions_csv,
            mpnn_pdbs_dir=mpnn_pdbs_dir,
            before_partial_pdbs_dir=before_partial_pdbs_dir,
        )

        eligible = set()
        excluded = set()
        with Path(fixed_positions_csv).open(newline="") as handle:
            for row in csv.DictReader(handle):
                filename = os.path.basename((row.get("filename") or "").strip())
                fixed_positions = (row.get("fixed_positions") or "").strip()
                if not filename:
                    continue
                if fixed_positions and fixed_positions.upper() != "NONE":
                    eligible.add(filename)
                else:
                    excluded.add(filename)

        output_dir = Path(before_partial_pdbs_dir)
        present = {path.name for path in output_dir.glob("*.pdb")}
        missing = sorted(eligible - present)
        if missing:
            raise FileNotFoundError(
                "official PPIFlow linker did not create %d eligible Stage2 inputs; first=%s"
                % (len(missing), missing[0])
            )

        removed = 0
        for pdb_path in output_dir.glob("*.pdb"):
            if pdb_path.name not in eligible:
                pdb_path.unlink()
                removed += 1

        logger.info(
            "Official PPIFlow Stage2 input gate: eligible=%d excluded_without_fixed=%d removed=%d",
            len(eligible),
            len(excluded),
            removed,
        )

    return link_fixed_only



def _agent_selected_sources(src_dir, identities):
    """Resolve explicit IDs only inside this batch's backbone directory."""
    root = Path(src_dir).resolve()
    if not isinstance(identities, list) or not identities or len(identities) != len(set(identities)):
        raise ValueError("agent selection must contain unique backbone IDs")
    sources = []
    for identity in identities:
        if not isinstance(identity, str) or not re.fullmatch(r"backbone[.]0_\d+", identity):
            raise ValueError("invalid original backbone ID")
        path = (root / (identity + ".pdb")).resolve()
        if path.parent != root or not path.is_file():
            raise ValueError("selected backbone is absent or outside the batch")
        sources.append(str(path))
    return sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--steps", required=True)
    parser.add_argument("--stage", type=int, choices=(1, 2), default=None)
    parser.add_argument("--agent-backbone-seed", type=int, default=None)
    args = parser.parse_args()

    native = _load_pipeline_module()
    round_input_dir = os.environ.get("AGENT_AB_ROUND_INPUT_DIR")
    round_input_pdbs = None
    if round_input_dir:
        round_input_dir = Path(round_input_dir).resolve()
        round_input_pdbs = round_input_dir / "structures"
        round_input_fixed = round_input_dir / "fixed_positions.csv"
        if not round_input_pdbs.is_dir() or not round_input_fixed.is_file():
            raise FileNotFoundError(
                "incomplete AGENT_AB_ROUND_INPUT_DIR: %s" % round_input_dir
            )

        def reuse_fixed_positions(output_csv, **_ignored):
            output_csv = Path(output_csv)
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(round_input_fixed, output_csv)
            native.log.info(
                "Reused fixed positions checkpoint: %s -> %s",
                round_input_fixed,
                output_csv,
            )
            return str(output_csv)

        native._build_fixed_positions_csv = reuse_fixed_positions
    _route_rosetta_fix_output(
        native, os.environ.get("AGENT_AB_ROSETTA_FIX_OUTPUT")
    )
    native._split_structure_and_seq_idx = (
        _split_structure_and_seq_idx_preserving_dots
    )
    native._load_mpnn_sequence_templates = (
        _load_mpnn_templates_with_backbone_aliases(
            native._load_mpnn_sequence_templates
        )
    )
    native._build_fixed_positions_csv = _require_merge_sequences(
        native._build_fixed_positions_csv
    )
    native._collect_partial_mpnn_pdb_and_seqs = (
        _collect_partial_outputs_with_stage2_names(
            native._collect_partial_mpnn_pdb_and_seqs
        )
    )
    # Use the official PPIFlow linker to apply merge_seq to each group scaffold.
    # The thin gate only removes groups for which no key residue was selected.
    native._link_before_partial_pdbs_from_fixed_positions = (
        _official_partial_inputs_with_fixed_positions_only(
            native._link_before_partial_pdbs_from_fixed_positions,
            native.log,
        )
    )
    native.PPIFlowStep = _parallel_ppiflow_step(native.PPIFlowStep)
    native.PartialStep = _parallel_partialflow_step(native.PartialStep)
    native.AbMPNNStep = _parallel_abmpnn_step(native.AbMPNNStep)
    native.FlowpackerStep = _parallel_flowpacker_step(native.FlowpackerStep)
    pipeline = native.Pipeline(task_yaml=args.task, steps_yaml=args.steps)
    if args.agent_backbone_seed is not None:
        cfg = pipeline._cfg("PPIFlowStep")
        seeded = Path(args.steps).with_name("agent_backbone_seed_%s.yaml" % args.agent_backbone_seed)
        _write_shard_config(cfg["config"], seeded, args.agent_backbone_seed)
        cfg["config"] = str(seeded)
    original_collect = native._collect_pdb_symlinks
    original_collect_sequences = native._collect_pdb_and_seqs
    task = pipeline.task

    def collect_sequences_and_organize(mpnn_out_dir, stage_dir):
        """Build the FlowPacker CSV, then compact FASTAs before the next step."""
        csv_path = original_collect_sequences(mpnn_out_dir, stage_dir)
        root = Path(mpnn_out_dir)
        if root.name == "abmpnn_output":
            stage_name = Path(stage_dir).name
            config_name = (
                "AbMPNNStep_stage1"
                if stage_name == "stage1"
                else "AbMPNNStep_stage2"
            )
            expected = int(pipeline._cfg(config_name)["num_seq_per_target"])
            organize_abmpnn_output(root, expected_sequences=expected)
        return csv_path

    native._collect_pdb_and_seqs = collect_sequences_and_organize

    def collect_sources(sources, des_dir, prefix):
        """Link selected backbones without embedding the run name in filenames."""
        destination = Path(des_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for existing in destination.glob("*.pdb"):
            if existing.is_symlink() or existing.is_file():
                existing.unlink()
        for value in sources:
            source = Path(value).resolve()
            if not source.is_file():
                raise FileNotFoundError("selected backbone not found: %s" % source)
            target = destination / source.name.lower()
            target.symlink_to(source)

    def passing_sources(table_path):
        with Path(table_path).open(newline="", encoding="utf-8") as handle:
            return [
                row["source_pdb"]
                for row in csv.DictReader(handle)
                if row.get("filter_pass") == "True"
            ]

    def screened_collect(src_dir, des_dir, prefix):
        is_stage1_backbone_handoff = (
            Path(src_dir).resolve() == Path(pipeline.stage1_dir, "ppiflow_output").resolve()
            and Path(des_dir).resolve() == Path(pipeline.stage1_dir, "mpnn_pdbs").resolve()
        )
        if not is_stage1_backbone_handoff:
            return original_collect(src_dir, des_dir, prefix)
        agent_selection = os.environ.get("AGENT_AB_BACKBONE_SELECTION_FILE")
        if agent_selection:
            selection = json.loads(Path(agent_selection).read_text(encoding="utf-8"))
            identities = selection["backbone_ids"]
            sources = _agent_selected_sources(src_dir, identities)
            return collect_sources(sources, des_dir, prefix)
        if pipeline.gentype not in ("nanobody", "antibody"):
            return collect_sources(
                sorted(Path(src_dir).glob("*.pdb")), des_dir, prefix
            )
        organize_backbone_output(
            src_dir, Path(pipeline.stage1_dir) / "backbone_output"
        )
        is_nanobody = pipeline.gentype == "nanobody"
        hotspots = _screening_hotspots(task, drop_nanobody_auxiliary=is_nanobody)
        profile_name = "nanobody" if is_nanobody else "scfv"
        output_root = (
            Path(pipeline.output_base_dir)
            / "stage1"
            / "backbone_screen"
        )
        selection_path = output_root / "backbone_selection.json"
        if not is_nanobody and selection_path.is_file():
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            native.log.info(
                "Reusing reviewed scFv selection: top %d",
                int(selection["selected_top_n"]),
            )
            return collect_sources(
                [item["source_pdb"] for item in selection["selected"]],
                des_dir,
                prefix,
            )
        common = {
            "input_dir": src_dir,
            "output_dir": output_root,
            "framework_pdb": task["framework_pdb"],
            "hotspots": hotspots,
            "heavy_chain": task.get("heavy_chain", "A"),
            "cutoff": float(task.get("backbone_screen_cutoff", 8.0)),
        }
        if is_nanobody:
            summary = screen_backbones(**common)
        else:
            summary = screen_scfv_backbones(
                light_chain=task.get("light_chain", "B"), **common
            )
        native.log.info(
            "Backbone hotspot screen: accepted %d/%d; results=%s",
            summary["accepted_backbones"],
            summary["input_backbones"],
            output_root,
        )
        if summary["accepted_backbones"] == 0:
            raise RuntimeError(
                "no %s backbones passed hotspot screen; see %s"
                % (profile_name, output_root)
            )
        top_n = _request_backbone_top_n(
            output_root,
            summary["accepted_backbones"],
            profile_name,
        )
        selection = select_top_scfv_backbones(output_root, top_n)
        sources = [item["source_pdb"] for item in selection["selected"]]
        native.log.info(
            "%s user selection: top %d/%d passing backbones",
            "VHH" if is_nanobody else "scFv",
            top_n,
            summary["accepted_backbones"],
        )
        return collect_sources(sources, des_dir, prefix)

    native._collect_pdb_symlinks = screened_collect
    try:
        pipeline.run(stage=args.stage)
    except BackboneSelectionRequired as exc:
        # The handoff directory is only an internal AbMPNN staging area.  The
        # pipeline creates it before the scFv Top-N gate, so remove it while
        # the run is paused instead of exposing an empty output directory.
        mpnn_staging = Path(pipeline.stage1_dir) / "mpnn_pdbs"
        if mpnn_staging.is_dir() and not any(mpnn_staging.iterdir()):
            mpnn_staging.rmdir()
        print(json.dumps(exc.request, ensure_ascii=False, indent=2))
        return 0
    finally:
        _remove_disabled_empty_output(
            pipeline,
            "DockQStep",
            "dockq_output",
            native.log,
        )
    if args.stage in (None, 2) and pipeline._enabled("PartialStep"):
        policy_path = Path(pipeline.stage2_dir) / "partialflow_mask_policy.json"
        policy_path.write_text(
            json.dumps(
                {
                    "policy": "official_ppiflow_partial_flow",
                    "key_residue_merge": "group Stage1 designs; keep the lowest-energy amino acid at each residue",
                    "parent_policy": "apply merge_seq to the original Stage1 group scaffold",
                    "fixed": "fixed_positions are passed to the official PPIFlow PartialStep",
                    "sampling": "official sample_antibody_nanobody_partial.py",
                    "flowpacker_policy": "official FlowPacker output without Agent-Ab atom restoration",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if (
        args.stage in (None, 1)
        and pipeline.gentype in ("antibody", "nanobody")
        and pipeline._enabled("AbMPNNStep_stage1")
    ):
        expected = int(pipeline._cfg("AbMPNNStep_stage1")["num_seq_per_target"])
        summary = organize_abmpnn_output(
            Path(pipeline.stage1_dir) / "abmpnn_output",
            expected_sequences=expected,
        )
        native.log.info(
            "Canonical AbMPNN output: %d backbones, %d sequences, changed=%s",
            summary["backbone_count"],
            summary["sequence_count"],
            summary["changed"],
        )
        if pipeline._enabled("FlowpackerStep_stage1"):
            packed = organize_flowpacker_output(
                Path(pipeline.stage1_dir) / "flowpacker_output",
                expected_backbones=summary["backbone_count"],
                expected_sequences=expected,
            )
            native.log.info(
                "Canonical FlowPacker output: %d backbones, %d designs, changed=%s",
                packed["backbone_count"],
                packed["design_count"],
                packed["changed"],
            )
    if (
        args.stage in (None, 2)
        and pipeline.gentype in ("antibody", "nanobody")
        and pipeline._enabled("AbMPNNStep_stage2")
    ):
        expected = int(pipeline._cfg("AbMPNNStep_stage2")["num_seq_per_target"])
        summary = organize_abmpnn_output(
            Path(pipeline.stage2_dir) / "abmpnn_output",
            expected_sequences=expected,
        )
        native.log.info(
            "Canonical Stage2 AbMPNN output: %d PartialFlow backbones, "
            "%d sequences, changed=%s",
            summary["backbone_count"],
            summary["sequence_count"],
            summary["changed"],
        )
        if pipeline._enabled("FlowpackerStep_stage2"):
            packed = organize_flowpacker_output(
                Path(pipeline.stage2_dir) / "flowpacker_output",
                expected_backbones=summary["backbone_count"],
                expected_sequences=expected,
            )
            native.log.info(
                "Canonical Stage2 FlowPacker output: %d PartialFlow backbones, "
                "%d designs, changed=%s",
                packed["backbone_count"],
                packed["design_count"],
                packed["changed"],
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
