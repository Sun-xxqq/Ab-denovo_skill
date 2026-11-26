from pathlib import Path, PurePosixPath

from .config import env_python, tool
from .util import project_root as installed_project_root
from .maturation import resolve_maturation_strategy


def _step_flags(spec):
    design_type = spec["design_type"]
    profile = spec["generation"]["profile"]
    antibody_like = design_type in ("antibody", "nanobody")
    full = profile == "full"
    # AF3Score needs a separately licensed AlphaFold3 parameter directory.
    # Only enable it when the user explicitly selected PPIFlow's AF3Score as
    # the complex predictor.  Other predictors (for example Protenix or Chai)
    # run through their own adapters after PPIFlow has generated/packed the
    # candidate structures.
    internal_af3 = spec.get("workflow", {}).get("complex_prediction") == "ppiflow_af3score"
    return {
        "PPIFlowStep": True,
        "MPNNStep_stage1": not antibody_like,
        "AbMPNNStep_stage1": antibody_like,
        "FlowpackerStep_stage1": True,
        "AF3scoreStep_stage1": internal_af3,
        "FilterStep_stage1": internal_af3,
        "RosettaFixStep": full,
        "PartialStep": full,
        "MPNNStep_stage2": full and not antibody_like,
        "AbMPNNStep_stage2": full and antibody_like,
        "FlowpackerStep_stage2": full,
        "AF3scoreStep_stage2": full and internal_af3,
        "FilterStep_stage2": full and internal_af3,
        "ReFoldStep": full and internal_af3,
        "DockQStep": full and internal_af3,
        "RosettaRelaxStep": full and internal_af3,
        "RankStep": full and internal_af3,
        "ReportStep": full and internal_af3,
    }


def build_task(spec, output_dir):
    target = spec["target"]
    generation = spec["generation"]
    design_type = spec["design_type"]
    task = {
        "name": spec["name"],
        "gentype": design_type,
        "specified_hotspots": ",".join(target["hotspots"]),
        "samples_per_target": generation["samples"],
        "output_base_dir": str(Path(output_dir).resolve()),
    }
    if design_type == "binder":
        task.update(
            {
                "input_pdb": target["structure"],
                "target_chain": target["chain"],
                "sample_hotspot_rate_min": float(generation.get("sample_hotspot_rate_min", 0.2)),
                "sample_hotspot_rate_max": float(generation.get("sample_hotspot_rate_max", 0.5)),
                "samples_min_length": generation["length"]["min"],
                "samples_max_length": generation["length"]["max"],
            }
        )
        # PPIFlow accepts target-only inputs.  Include binder_chain only when
        # the input PDB already contains a binder/framework chain.
        if target.get("binder_chain"):
            task["binder_chain"] = target["binder_chain"]
    else:
        framework = spec["framework"]
        task.update(
            {
                "antigen_pdb": target["structure"],
                "antigen_chain": target["chain"],
                "framework_pdb": framework["structure"],
                "heavy_chain": framework["heavy_chain"],
                "cdr_length": generation["cdr_lengths"],
            }
        )
        if design_type == "antibody":
            task["light_chain"] = framework["light_chain"]
    return {"task": task, "steps": _step_flags(spec)}


def _filters(spec, stage):
    design_type = spec["design_type"]
    target_chain = spec["target"]["chain"]
    if design_type == "binder":
        binder_chain = spec["generation"]["binder_chain"]
        return {
            "iptm": "> 0.70" if stage == 1 else "> 0.80",
            "chain_%s_ptm" % binder_chain: "> 0.70",
        }
    framework = spec["framework"]
    heavy = framework["heavy_chain"]
    values = {
        "iptm_%s_%s" % (heavy, target_chain): "> 0.30" if stage == 1 else "> 0.50",
        "chain_%s_ptm" % heavy: "> 0.70",
    }
    if design_type == "antibody":
        values["chain_%s_ptm" % framework["light_chain"]] = "> 0.70"
    return values


def build_steps(spec, config):
    ppiflow = tool(config, "ppiflow")
    root = PurePosixPath(ppiflow["path"])
    python = env_python(config, ppiflow["env"])
    design_type = spec["design_type"]
    generation = spec["generation"]
    weight = root / "weigths" / (design_type + ".ckpt")
    inference_config = root / "tools" / "PPIFlow" / "configs" / (
        "inference_binder.yaml" if design_type == "binder" else "inference_nanobody.yaml"
    )
    partial_inference_config = root / "tools" / "PPIFlow" / "configs" / (
        "inference_binder_partial.yaml"
        if design_type == "binder"
        else "inference_nanobody.yaml"
    )
    sequence_common = {
        "num_seq_per_target": generation["num_sequences_per_backbone"],
        "sampling_temp": generation["sampling_temperature"],
        "batch_size": 1,
        "omit_AAs": generation["omit_amino_acids"],
    }
    screening = spec.get("validation", {}).get("candidate_screening", {})
    maturation = resolve_maturation_strategy(spec)
    round2 = maturation["round2_sequence_design"]
    sequence_stage2 = dict(sequence_common)
    sequence_stage2["num_seq_per_target"] = int(round2["sequences_per_refined_backbone"])
    sequence_stage2["sampling_temp"] = float(round2["sampling_temperature"])
    steps = {
        "python": python,
        "PPIFlowStep": {"model_weights": str(weight), "config": str(inference_config)},
    }
    if design_type == "binder":
        binder_chain = generation["binder_chain"]
        mpnn = dict(sequence_common)
        mpnn.update(
            {
                "protocol": "base",
                "chains_to_design": binder_chain,
                "use_soluble_model": True,
            }
        )
        steps["MPNNStep_stage1"] = dict(mpnn)
        mpnn_stage2 = dict(mpnn)
        mpnn_stage2["num_seq_per_target"] = sequence_stage2["num_seq_per_target"]
        mpnn_stage2["sampling_temp"] = sequence_stage2["sampling_temp"]
        mpnn_stage2["use_soluble_model"] = round2["backend"] == "proteinmpnn_soluble"
        steps["MPNNStep_stage2"] = mpnn_stage2
    else:
        chains = [spec["framework"]["heavy_chain"]]
        if design_type == "antibody":
            chains.append(spec["framework"]["light_chain"])
        abmpnn = dict(sequence_common)
        abmpnn["chain_list"] = " ".join(chains)
        abmpnn["pipeline_script_dir"] = str(
            PurePosixPath(config.get("project_root", installed_project_root().as_posix()))
            / "scripts"
            / "abmpnn_adapter"
        )
        steps["AbMPNNStep_stage1"] = dict(abmpnn)
        abmpnn_stage2 = dict(abmpnn)
        abmpnn_stage2["num_seq_per_target"] = sequence_stage2["num_seq_per_target"]
        abmpnn_stage2["sampling_temp"] = sequence_stage2["sampling_temp"]
        steps["AbMPNNStep_stage2"] = abmpnn_stage2

    flowpacker = {
        "config_path": str(root / "tools" / "flowpacker" / "config" / "inference" / "base.yaml"),
        "use_gt_masks": True,
    }
    af3score = {
        "db_dir": str(root / "tools" / "AF3Score" / "af3_data"),
        "model_dir": str(root / "tools" / "AF3Score" / "Alphafold3params"),
        "num_workers": 2,
    }
    steps.update(
        {
            "FlowpackerStep_stage1": dict(flowpacker),
            "FlowpackerStep_stage2": dict(flowpacker),
            "AF3scoreStep_stage1": dict(af3score),
            "AF3scoreStep_stage2": dict(af3score),
            "FilterStep_stage1": {"filters": _filters(spec, 1)},
            "FilterStep_stage2": {"filters": _filters(spec, 2)},
            "RosettaFixStep": {
                "rosetta_bin": str(root / "rosetta.binary.linux.release-371" / "main" / "source" / "bin" / "rosetta_scripts.static.linuxgccrelease"),
                "interface_dist": 12.0,
                "native_xml": None,
                "plot_toggle": False,
                "overwrite": True,
            },
            "ReFoldStep": {
                "db_dir": af3score["db_dir"],
                "model_dir": af3score["model_dir"],
                "hmmer_path": str(PurePosixPath(python).parent),
                "run_MSA": True,
                "remove_template": True,
                "run_AF3_inference": False,
                "seed_num": "5",
                "cuda_bin": "/usr/local/cuda-11.4/bin",
                "cuda_lib": "/usr/local/cuda-11.4/lib64",
                "xla_client_mem_fraction": "0.90",
            },
            "DockQStep": {
                "dockq_exe": str(PurePosixPath(python).parent / "DockQ"),
                "dockq_args": ["--short"],
                "recursive": False,
            },
            "RosettaRelaxStep": {"batch_idx": "0", "dump_pdb": False},
            "RankStep": {"dockq_threshold": 0.49, "output_csv_name": "ranked_designs.csv"},
        }
    )
    partial = {
        "model_weights": str(weight),
        "config": str(partial_inference_config),
        "samples_per_target": int(maturation["partial_flow"]["refined_backbones_per_seed"]),
        "start_t": float(maturation["partial_flow"]["start_t"]),
        # The shared PPIFlow pipeline reads this value while converting
        # Rosetta residue energies into fixed interface positions.
        "energy_threshold": float(
            maturation["interface_rotamer_enrichment"]["energy_threshold_reu"]
        ),
    }
    if design_type == "binder":
        partial.update(
            {
                "target_chain": spec["target"]["chain"],
                "binder_chain": generation["binder_chain"],
                "sample_hotspot_rate_min": 0.2,
                "sample_hotspot_rate_max": 0.5,
            }
        )
    else:
        partial.update(
            {
                "antigen_chain": spec["target"]["chain"],
                "heavy_chain": spec["framework"]["heavy_chain"],
                "retry_Limit": 10,
            }
        )
        if design_type == "antibody":
            partial["light_chain"] = spec["framework"]["light_chain"]
    steps["PartialStep"] = partial
    return steps


def command(config, task_path, steps_path, stage=None):
    ppiflow = tool(config, "ppiflow")
    project_root = PurePosixPath(config.get("project_root", installed_project_root().as_posix()))
    values = [
        env_python(config, ppiflow["env"]),
        str(project_root / "scripts" / "ppiflow_pipeline_screened.py"),
        "--task",
        str(task_path),
        "--steps",
        str(steps_path),
    ]
    if stage is not None:
        values.extend(["--stage", str(stage)])
    return values
