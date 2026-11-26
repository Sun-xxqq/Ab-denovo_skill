from pathlib import Path, PurePosixPath

from .config import env_python


def _bin(config, env_name, executable):
    root = PurePosixPath(config["conda_root"])
    base = root / "bin" if env_name == "base" else root / "envs" / env_name / "bin"
    return str(base / executable)


def build_command(tool_name, config, inputs, output_dir, spec):
    """Build argv only; no shell interpolation is ever used."""
    tool = config["tools"][tool_name]
    output_dir = str(Path(output_dir))
    if tool_name == "rfdiffusion":
        generation = spec["generation"]
        length = generation.get("length", {"min": 50, "max": 80})
        target = spec["target"]
        contig = "[%s/0 %s-%s]" % (target["chain"], length["min"], length["max"])
        return [
            env_python(config, tool["env"]),
            str(PurePosixPath(tool["path"]) / "scripts" / "run_inference.py"),
            "inference.input_pdb=%s" % target["structure"],
            "inference.output_prefix=%s/design" % output_dir,
            "inference.num_designs=%s" % generation["samples"],
            "contigmap.contigs=%s" % contig,
            "ppi.hotspot_res=[%s]" % ",".join(target["hotspots"]),
        ]
    if tool_name == "proteinmpnn":
        return [
            env_python(config, tool["env"]),
            str(PurePosixPath(tool["path"]) / "protein_mpnn_run.py"),
            "--pdb_path", str(inputs["pdb"]),
            "--pdb_path_chains", str(inputs["chains"]),
            "--out_folder", output_dir,
            "--num_seq_per_target", str(spec["generation"]["num_sequences_per_backbone"]),
            "--sampling_temp", str(spec["generation"]["sampling_temperature"]),
            "--omit_AAs", str(spec["generation"]["omit_amino_acids"]),
        ]
    if tool_name == "boltzgen":
        protocol = {"binder": "protein-anything", "nanobody": "nanobody-anything", "antibody": "antibody-anything"}[spec["design_type"]]
        return [
            _bin(config, tool["env"], "boltzgen"), "run", str(inputs["design_spec"]),
            "--protocol", protocol, "--output", output_dir,
            "--num_designs", str(spec["generation"]["samples"]),
            "--inverse_fold_num_sequences", str(spec["generation"]["num_sequences_per_backbone"]),
            "--cache", tool["cache_dir"], "--reuse",
        ]
    if tool_name == "rfdiffusion3":
        executable = "rf3" if inputs.get("mode") == "predict" else "rfd3"
        action = "fold" if executable == "rf3" else "design"
        return [
            _bin(config, tool["env"], executable), action,
            "inputs=%s" % inputs["spec"], "out_dir=%s" % output_dir,
            "skip_existing=True", "prevalidate_inputs=True",
        ]
    if tool_name == "protenix":
        framework = spec.get("framework", {})
        if spec["design_type"] == "binder":
            design_chains = [spec["generation"]["binder_chain"]]
        elif spec["design_type"] == "nanobody":
            design_chains = [framework.get("heavy_chain", "A")]
        else:
            design_chains = [
                framework.get("heavy_chain", "A"), framework.get("light_chain", "B")
            ]
        command = [
            env_python(config, tool["env"]),
            str(PurePosixPath(config["project_root"]) / "scripts" / "protenix_batch_adapter.py"),
            "--manifest", str(inputs["manifest"]),
            "--output", output_dir,
            "--publish-root", str(inputs.get("publish_root", Path(output_dir).parent)),
            "--protenix-bin", _bin(config, tool["env"], "protenix"),
            "--model", tool["model"],
            "--checkpoint", tool["weights"],
            "--seed", str(inputs.get("seed", 101)),
            "--target-chain", str(inputs.get("target_chain", spec["target"]["chain"])),
            "--design-chains", ",".join(inputs.get("design_chains", design_chains)),
        ]
        if inputs.get("need_atom_confidence"):
            command.append("--need-atom-confidence")
        if inputs.get("allow_partial_failures"):
            command.append("--allow-partial-failures")
        return command
    if tool_name == "chai":
        return [
            env_python(config, tool["env"]),
            str(PurePosixPath(config["project_root"]) / "scripts" / "chai_batch_adapter.py"),
            "--manifest", str(inputs["manifest"]),
            "--output", output_dir,
            "--num-diffn-samples", str(inputs.get("num_diffn_samples", 1)),
        ]
    if tool_name == "ipsae":
        if inputs.get("prediction_dir"):
            framework = spec.get("framework", {})
            command = [
                env_python(config, config["tools"]["ppiflow"]["env"]),
                str(PurePosixPath(config["project_root"]) / "scripts" / "evaluate_chai_candidates.py"),
                "--prediction-dir", str(inputs["prediction_dir"]),
                "--manifest", str(inputs["manifest"]),
                "--ipsae-script", str(PurePosixPath(tool["path"]) / "ipsae.py"),
                "--ipsae-python", env_python(config, tool["env"]),
                "--output", output_dir,
                "--selected-dir", str(inputs["selected_dir"]),
                "--stage1-candidates-dir", str(inputs["stage1_candidates_dir"]),
                "--pae-cutoff", str(inputs.get("pae_cutoff", 10)),
                "--dist-cutoff", str(inputs.get("dist_cutoff", 10)),
                "--heavy-chain", str(framework.get("heavy_chain", "A")),
                "--light-chain", str(framework.get("light_chain", "")),
                "--target-chain", str(spec["target"]["chain"]),
                "--thresholds-json", str(inputs["thresholds_json"]),
                "--run-dir", str(inputs["run_dir"]),
                "--hotspots", ",".join(inputs.get("hotspots", [])),
            ]
            if inputs.get("delete_rejected"):
                command.append("--delete-rejected")
            if inputs.get("skip_ipsae"):
                command.append("--skip-ipsae")
            return command
        return [
            env_python(config, tool["env"]),
            str(PurePosixPath(tool["path"]) / "ipsae.py"),
            str(inputs["confidence"]), str(inputs["structure"]),
            str(inputs.get("pae_cutoff", 10)), str(inputs.get("dist_cutoff", 10)),
        ]
    if tool_name == "pyrosetta":
        return [
            env_python(config, tool["env"]),
            str(PurePosixPath(config["project_root"]) / "scripts" / "score_interface.py"),
            "--input", str(inputs["structure"]), "--output", output_dir,
        ]
    raise ValueError("no executable adapter for tool: %s" % tool_name)
