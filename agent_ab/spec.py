from pathlib import Path

from .util import read_json
from .pipeline_choices import normalize_workflow
from .contacts import infer_complex_roles, interface_hotspots
from .filtering import DEFAULT_STAGE1_THRESHOLDS, DEFAULT_STAGE2_THRESHOLDS
from .maturation import resolve_maturation_strategy


DESIGN_TYPES = ("binder", "nanobody", "antibody")
PROFILES = ("stage1", "full")


def _required(mapping, key, context):
    value = mapping.get(key)
    if value is None or value == "":
        raise ValueError("%s.%s is required" % (context, key))
    return value


def _absolute_existing(path, label, require_inputs):
    raw = str(path).strip()
    value = Path(raw).expanduser()
    # Specs describe Linux server paths even when unit tests run from a
    # Windows control workstation.
    if not (raw.startswith("/") or value.is_absolute()):
        raise ValueError("%s must be an absolute server path" % label)
    if require_inputs and not value.is_file():
        raise ValueError("%s does not exist: %s" % (label, value))
    return raw if raw.startswith("/") else str(value)


def load_and_validate(path, require_inputs=True):
    return validate_spec(read_json(path), require_inputs=require_inputs)


def validate_spec(raw, require_inputs=True):
    if not isinstance(raw, dict):
        raise ValueError("job spec must be a JSON object")
    spec = dict(raw)
    spec.setdefault("schema_version", 1)
    if spec["schema_version"] != 1:
        raise ValueError("unsupported schema_version: %s" % spec["schema_version"])
    name = str(_required(spec, "name", "job")).strip()
    design_type = str(_required(spec, "design_type", "job")).lower()
    if design_type not in DESIGN_TYPES:
        raise ValueError("design_type must be one of: %s" % ", ".join(DESIGN_TYPES))

    target = dict(_required(spec, "target", "job"))
    target["structure"] = _absolute_existing(
        _required(target, "structure", "target"), "target.structure", require_inputs
    )
    inferred_roles = {}
    if require_inputs and target.get("structure"):
        inferred_roles = infer_complex_roles(target["structure"], design_type)
    if not target.get("chain") and inferred_roles.get("target_chain"):
        target["chain"] = inferred_roles["target_chain"]
    target["chain"] = str(_required(target, "chain", "target")).strip()
    if len(target["chain"]) != 1:
        raise ValueError("target.chain must be one PDB chain ID")
    hotspots = target.get("hotspots", [])
    if not isinstance(hotspots, list):
        raise ValueError("target.hotspots must be a list such as [\"C10\", \"C13\"]")
    target["hotspots"] = [str(item).strip() for item in hotspots if str(item).strip()]
    for residue in target["hotspots"]:
        if not residue.startswith(target["chain"]):
            raise ValueError("hotspot %s must start with target chain %s" % (residue, target["chain"]))

    generation = dict(spec.get("generation", {}))
    generation.setdefault("backend", "ppiflow")
    if generation["backend"] != "ppiflow":
        raise ValueError("automatic execution currently supports generation.backend=ppiflow")
    generation.setdefault("profile", "stage1")
    if generation["profile"] not in PROFILES:
        raise ValueError("generation.profile must be stage1 or full")
    generation.setdefault("samples", 10)
    generation["samples"] = int(generation["samples"])
    if not 1 <= generation["samples"] <= 20000:
        raise ValueError("generation.samples must be between 1 and 20000")
    generation.setdefault("num_sequences_per_backbone", 8)
    generation["num_sequences_per_backbone"] = int(generation["num_sequences_per_backbone"])
    if not 1 <= generation["num_sequences_per_backbone"] <= 1000:
        raise ValueError("generation.num_sequences_per_backbone must be between 1 and 1000")
    generation.setdefault("stage2_sequences_per_backbone", 4)
    generation["stage2_sequences_per_backbone"] = int(
        generation["stage2_sequences_per_backbone"]
    )
    if not 1 <= generation["stage2_sequences_per_backbone"] <= 1000:
        raise ValueError(
            "generation.stage2_sequences_per_backbone must be between 1 and 1000"
        )
    generation.setdefault("sampling_temperature", 0.1)
    generation["sampling_temperature"] = float(generation["sampling_temperature"])
    generation.setdefault("omit_amino_acids", "C")

    if design_type == "binder":
        if not target["hotspots"] and not generation.get("allow_no_hotspots", False):
            raise ValueError("binder design requires target.hotspots unless allow_no_hotspots is true")
        length = dict(_required(generation, "length", "generation"))
        length["min"] = int(_required(length, "min", "generation.length"))
        length["max"] = int(_required(length, "max", "generation.length"))
        if not 20 <= length["min"] <= length["max"] <= 300:
            raise ValueError("binder length must satisfy 20 <= min <= max <= 300")
        generation["length"] = length
        generation.setdefault("binder_chain", "A")
    else:
        framework = dict(spec.get("framework") or {})
        if inferred_roles.get("heavy_chain"):
            framework.setdefault("structure", target["structure"])
            framework.setdefault("heavy_chain", inferred_roles["heavy_chain"])
            if design_type == "antibody":
                framework.setdefault("light_chain", inferred_roles["light_chain"])
        if not framework:
            framework = dict(_required(spec, "framework", "job"))
        framework["structure"] = _absolute_existing(
            _required(framework, "structure", "framework"),
            "framework.structure",
            require_inputs,
        )
        framework.setdefault("heavy_chain", "A")
        if len(str(framework["heavy_chain"])) != 1:
            raise ValueError("framework.heavy_chain must be one PDB chain ID")
        if design_type == "antibody":
            framework.setdefault("light_chain", "B")
            if len(str(framework["light_chain"])) != 1:
                raise ValueError("framework.light_chain must be one PDB chain ID")
        generation.setdefault(
            "cdr_lengths",
            "CDRH1,8-8,CDRH2,8-8,CDRH3,9-21"
            if design_type == "nanobody"
            else "CDRH1,8-8,CDRH2,8-8,CDRH3,10-20,CDRL1,6-9,CDRL2,3-3,CDRL3,9-11",
        )
        spec["framework"] = framework

        if not target["hotspots"] and require_inputs and framework["structure"] == target["structure"]:
            partner_chains = [framework["heavy_chain"]]
            if design_type == "antibody":
                partner_chains.append(framework["light_chain"])
            target["hotspots"] = interface_hotspots(
                target["structure"], target["chain"], partner_chains
            )
            if target["hotspots"]:
                target["hotspot_source"] = {
                    "method": "heavy_atom_contacts",
                    "partner_chains": partner_chains,
                    "distance_cutoff_angstrom": 5.0,
                    "max_residues": 6,
                }

    workflow = normalize_workflow(spec.get("workflow"))
    output_naming = dict(spec.get("output_naming", {}))
    output_naming.setdefault("enabled", True)
    output_naming.setdefault("scheme", "two_round_v1")
    output_naming["enabled"] = bool(output_naming["enabled"])
    validation = dict(spec.get("validation", {}))
    validation.setdefault("independent_predictors", ["protenix", "chai"])
    validation.setdefault("interface_scoring", ["ipsae", "pyrosetta"])
    validation.setdefault("negative_targets", [])
    screening = dict(validation.get("candidate_screening", {}))
    screening.setdefault("mode", "loose_filter_only")
    # Stage 1 is backbone-grouped: by default every sequence/FlowPacker design
    # is sent to the predictor instead of truncating the first N files globally.
    screening.setdefault("max_candidates_per_backbone", generation["num_sequences_per_backbone"])
    # Keep predictor outputs by default so failed candidates remain available
    # for threshold diagnosis and later re-screening.
    screening.setdefault("delete_rejected", False)
    legacy_thresholds = dict(screening.get("loose_thresholds", {}))
    stage1_thresholds = dict(DEFAULT_STAGE1_THRESHOLDS)
    stage1_thresholds.update(
        {key: value for key, value in legacy_thresholds.items() if key in stage1_thresholds}
    )
    stage1_thresholds.update(screening.get("stage1_thresholds", {}))
    stage2_thresholds = dict(DEFAULT_STAGE2_THRESHOLDS)
    stage2_thresholds.update(
        {key: value for key, value in legacy_thresholds.items() if key in stage1_thresholds}
    )
    stage2_thresholds.update(screening.get("stage2_thresholds", {}))
    screening["stage1_thresholds"] = stage1_thresholds
    screening["stage2_thresholds"] = stage2_thresholds
    screening["loose_thresholds"] = dict(stage1_thresholds)
    adaptive_prediction = dict(screening.get("stage1_adaptive_prediction", {}))
    adaptive_prediction.setdefault("enabled", True)
    adaptive_prediction.setdefault("first_batch_per_backbone", 4)
    adaptive_prediction.setdefault("second_batch_per_backbone", 4)
    adaptive_prediction.setdefault("continuation_design_pred_rmsd_max", 10.0)
    adaptive_prediction["enabled"] = bool(adaptive_prediction["enabled"])
    for key in ("first_batch_per_backbone", "second_batch_per_backbone"):
        adaptive_prediction[key] = int(adaptive_prediction[key])
        if adaptive_prediction[key] < 1:
            raise ValueError(
                "validation.candidate_screening.stage1_adaptive_prediction.%s "
                "must be at least 1" % key
            )
    adaptive_prediction["continuation_design_pred_rmsd_max"] = float(
        adaptive_prediction["continuation_design_pred_rmsd_max"]
    )
    if adaptive_prediction["continuation_design_pred_rmsd_max"] <= 0:
        raise ValueError(
            "validation.candidate_screening.stage1_adaptive_prediction."
            "continuation_design_pred_rmsd_max must be positive"
        )
    screening["stage1_adaptive_prediction"] = adaptive_prediction
    screening.setdefault("iterative_redesign", True)
    if "off_target_screening" in screening:
        off_target = dict(screening["off_target_screening"])
        off_target.setdefault("enabled", True)
        off_target.setdefault("name", "CLDN18.1")
        off_target.setdefault("source_chain", "A")
        off_target["enabled"] = bool(off_target["enabled"])
        if off_target["enabled"]:
            off_target["structure"] = _absolute_existing(
                _required(
                    off_target,
                    "structure",
                    "validation.candidate_screening.off_target_screening",
                ),
                "validation.candidate_screening.off_target_screening.structure",
                require_inputs,
            )
            off_target["source_chain"] = str(off_target["source_chain"]).strip().upper()
            if len(off_target["source_chain"]) != 1:
                raise ValueError(
                    "validation.candidate_screening.off_target_screening.source_chain "
                    "must be one PDB chain ID"
                )
        screening["off_target_screening"] = off_target
    stage2_comparison = dict(screening.get("stage2_structural_comparison", {}))
    stage2_comparison.setdefault("enabled", True)
    stage2_comparison.setdefault("predictor", workflow["complex_prediction"])
    stage2_comparison.setdefault("seed", 101)
    stage2_comparison["enabled"] = bool(stage2_comparison["enabled"])
    screening["stage2_structural_comparison"] = stage2_comparison
    stage3_specificity = dict(screening.get("stage3_iggm_specificity", {}))
    stage3_specificity.setdefault("enabled", True)
    stage3_specificity.setdefault("sidechain_packing", "flowpacker")
    stage3_specificity.setdefault("flowpacker_seed", 42)
    stage3_specificity.setdefault("predictor", "protenix")
    stage3_specificity.setdefault("seed", 101)
    stage3_specificity.setdefault("pae_cutoff", 10)
    stage3_specificity.setdefault("dist_cutoff", 10)
    stage3_specificity.setdefault("thresholds", dict(stage2_thresholds))
    stage3_specificity["enabled"] = bool(stage3_specificity["enabled"])
    if stage3_specificity["predictor"] != "protenix":
        raise ValueError(
            "validation.candidate_screening.stage3_iggm_specificity.predictor "
            "must be protenix"
        )
    if stage3_specificity["sidechain_packing"] != "flowpacker":
        raise ValueError(
            "validation.candidate_screening.stage3_iggm_specificity."
            "sidechain_packing must be flowpacker"
        )
    screening["stage3_iggm_specificity"] = stage3_specificity
    final_evaluation = dict(screening.get("final_functional_evaluation", {}))
    # The default two-round workflow terminates at the Stage 2 pred/design
    # structural comparison.  Expensive interface-energy/ranking work is an
    # explicit opt-in extension.
    final_evaluation.setdefault("enabled", False)
    final_evaluation.setdefault("predictor", "protenix")
    final_evaluation.setdefault("interface_confidence", "ipsae")
    final_evaluation.setdefault("interface_energy", "pyrosetta")
    final_evaluation.setdefault("rank_within_backbone_group_only", True)
    final_evaluation.setdefault("affinity_output", "proxy_score_not_kd")
    final_evaluation.setdefault(
        "hard_gates",
        {
            "plddt_min": 0.80,
            "ptm_min": 0.70,
            "iptm_min": 0.50,
            "ipsae_min": 0.61,
            "interface_pae_mean_max": 10.0,
            "interface_energy_max": 0.0,
            "interface_delta_sasa_min": 400.0 if design_type == "binder" else 600.0,
            "hotspot_coverage_min": 0.50,
            "cdr_contact_fraction_min": None if design_type == "binder" else 0.70,
            "developability_risk_max": 3,
        },
    )
    screening["final_functional_evaluation"] = final_evaluation
    numeric_keys = ["max_candidates_per_backbone"]
    if screening["mode"] != "loose_filter_only":
        screening.setdefault("per_backbone_top_k", screening.get("top_k", generation["num_sequences_per_backbone"]))
        screening.setdefault("allow_below_standard_fill", False)
        numeric_keys.append("per_backbone_top_k")
    for key in numeric_keys:
        screening[key] = int(screening[key])
        if screening[key] < 1:
            raise ValueError("validation.candidate_screening.%s must be at least 1" % key)
    if "allow_below_standard_fill" in screening:
        screening["allow_below_standard_fill"] = bool(screening["allow_below_standard_fill"])
    screening["delete_rejected"] = bool(screening["delete_rejected"])
    validation["candidate_screening"] = screening
    spec.update(
        {
            "name": name,
            "design_type": design_type,
            "target": target,
            "generation": generation,
            "validation": validation,
            "workflow": workflow,
            "output_naming": output_naming,
        }
    )
    # Materialize the design-class-specific Stage 2 policy in every validated
    # spec so dry-runs and saved jobs expose the exact scientific assumptions.
    screening["maturation"] = resolve_maturation_strategy(spec)
    return spec
