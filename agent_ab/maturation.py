"""Design-class-specific PPIFlow interface-rotamer maturation policies.

The defaults follow Appendix B of the PPIFlow material where it specifies a
value.  Fields marked as an extension are conservative adaptations for design
classes for which the material does not provide a separate protocol.
"""

from copy import deepcopy


STRATEGY_VERSION = "ppiflow_appendix_b_interface_rotamer_v1"


def _deep_update(base, override):
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def _base_strategy():
    return {
        "strategy_version": STRATEGY_VERSION,
        "input_selection": "retained_stage1_candidates_only",
        "grouping": {
            "scope": "per_original_backbone",
            "cross_group_merge": False,
            "position_aggregation": "union_below_energy_threshold",
            "duplicate_position_energy": "minimum_across_group_sequences",
            "rotamer_identity": "lowest_energy_sequence_residue",
        },
        "interface_rotamer_enrichment": {
            "energy_threshold_reu": -5.0,
        },
        "partial_flow": {
            "start_t": 0.6,
            "refined_backbones_per_seed": 1,
        },
        "round2_sequence_design": {
            "sampling_temperature": 0.1,
            "sequences_per_refined_backbone": 4,
        },
        "final_generation": {
            "candidates_per_original_backbone_group": 4,
            "allocation": "1_partial_backbone_x_4_sequences",
            "require_exact_count_before_structural_comparison": True,
        },
        "parameter_provenance": {},
    }


def _default_strategy(spec):
    design_type = spec["design_type"]
    target_chain = spec["target"]["chain"]
    strategy = _base_strategy()

    if design_type == "binder":
        binder_chain = spec["generation"]["binder_chain"]
        strategy.update(
            {
                "design_class": "binder",
                "source_basis": "PPIFlow Appendix B.1",
                "interface_pairs": [
                    {"design_chain": binder_chain, "target_chain": target_chain}
                ],
                "constraints": {
                    "fixed_during_partial_flow": "interface_rotamer_enriched_residues",
                },
            }
        )
        strategy["round2_sequence_design"].update(
            {
                "backend": "proteinmpnn_soluble",
                "design_chains": [binder_chain],
                "count_source": "initial task generation parameters",
            }
        )
        strategy["parameter_provenance"].update(
            {
                "energy_threshold_reu": "PPIFlow Appendix B.1",
                "partial_flow.start_t": "PPIFlow Appendix B.1",
                "round2_sequence_design.sampling_temperature": "PPIFlow Appendix B.1",
                "final_generation": "initial task generation parameters; Appendix B.1 does not specify this count",
            }
        )
        return strategy

    framework = spec["framework"]
    heavy = framework["heavy_chain"]
    pairs = [{"design_chain": heavy, "target_chain": target_chain}]
    design_chains = [heavy]
    constraints = {
        "sequence_redesign_scope": "cdr_only",
        "preserve_framework": True,
        "fixed_during_partial_flow": "framework_and_interface_rotamer_enriched_residues",
    }
    source_basis = "PPIFlow Appendix B.2"
    design_class = "vhh"

    if design_type == "antibody":
        light = framework["light_chain"]
        pairs.append({"design_chain": light, "target_chain": target_chain})
        design_chains.append(light)
        constraints.update(
            {
                "preserve_vh_vl_geometry": True,
                "exclude_internal_vh_vl_interface_from_antigen_enrichment": True,
            }
        )
        design_class = spec.get("generation", {}).get("antibody_format", "paired_vh_vl")
        source_basis = "paired VH/VL extension of PPIFlow Appendix B.2"

    strategy.update(
        {
            "design_class": design_class,
            "source_basis": source_basis,
            "interface_pairs": pairs,
            "constraints": constraints,
        }
    )
    strategy["round2_sequence_design"].update(
        {
            "backend": "abmpnn",
            "design_chains": design_chains,
            "count_source": "initial task generation parameters",
        }
    )
    strategy["parameter_provenance"].update(
        {
            "energy_threshold_reu": "PPIFlow Appendix B.2",
            "partial_flow.start_t": "0.6 adaptation from Appendix B.1; Appendix B.2 does not state t",
            "final_generation": "initial task generation parameters; may override Appendix B.2 sampling count",
            "round2_sequence_design.sampling_temperature": "PPIFlow Appendix B.2",
        }
    )
    return strategy


def resolve_maturation_strategy(spec):
    """Return validated defaults plus nested and legacy job overrides."""
    strategy = _default_strategy(spec)
    screening = spec.get("validation", {}).get("candidate_screening", {})
    generation = spec.get("generation", {})
    if "stage2_sequences_per_backbone" in generation:
        strategy["round2_sequence_design"]["sequences_per_refined_backbone"] = int(
            generation["stage2_sequences_per_backbone"]
        )
    overrides = screening.get("maturation", {})
    _deep_update(strategy, overrides)

    # Unless the caller supplied an explicit total, derive it from the two
    # independently configurable Stage 2 counts.
    final_overrides = overrides.get("final_generation", {})
    refined = int(strategy["partial_flow"]["refined_backbones_per_seed"])
    sequences = int(
        strategy["round2_sequence_design"]["sequences_per_refined_backbone"]
    )
    if "candidates_per_original_backbone_group" not in final_overrides:
        strategy["final_generation"][
            "candidates_per_original_backbone_group"
        ] = refined * sequences
    if "allocation" not in final_overrides:
        strategy["final_generation"]["allocation"] = (
            "%s_partial_backbones_x_%s_sequences" % (refined, sequences)
        )

    # Old NLU versions wrote 1x1 counts into every job, even when the user did
    # not request them.  Applying those fields automatically would silently
    # disable the Appendix-B VHH/antibody 8x4 policy in all existing runs.
    # They are therefore honored only behind an explicit compatibility flag;
    # new overrides belong under candidate_screening.maturation.
    use_legacy_counts = bool(screening.get("use_legacy_redesign_counts", False))
    if use_legacy_counts and "redesign_backbones_per_seed" in screening:
        strategy["partial_flow"]["refined_backbones_per_seed"] = int(
            screening["redesign_backbones_per_seed"]
        )
    if use_legacy_counts and "redesign_sequences_per_backbone" in screening:
        strategy["round2_sequence_design"]["sequences_per_refined_backbone"] = int(
            screening["redesign_sequences_per_backbone"]
        )

    energy = float(strategy["interface_rotamer_enrichment"]["energy_threshold_reu"])
    start_t = float(strategy["partial_flow"]["start_t"])
    refined = int(strategy["partial_flow"]["refined_backbones_per_seed"])
    sequences = int(
        strategy["round2_sequence_design"]["sequences_per_refined_backbone"]
    )
    temperature = float(strategy["round2_sequence_design"]["sampling_temperature"])
    final_count = int(strategy["final_generation"]["candidates_per_original_backbone_group"])
    if energy >= 0:
        raise ValueError("maturation energy_threshold_reu must be negative")
    if not 0 < start_t <= 1:
        raise ValueError("maturation partial_flow.start_t must be in (0, 1]")
    if refined < 1 or sequences < 1:
        raise ValueError("maturation backbone and sequence counts must be at least 1")
    if temperature <= 0:
        raise ValueError("maturation sampling_temperature must be positive")
    if refined * sequences != final_count:
        raise ValueError(
            "maturation refined_backbones_per_seed * sequences_per_refined_backbone "
            "must equal final candidates_per_original_backbone_group"
        )
    if strategy["grouping"].get("cross_group_merge"):
        raise ValueError("maturation cannot merge candidates across original backbone groups")

    strategy["interface_rotamer_enrichment"]["energy_threshold_reu"] = energy
    strategy["partial_flow"].update(
        {"start_t": start_t, "refined_backbones_per_seed": refined}
    )
    strategy["round2_sequence_design"].update(
        {
            "sampling_temperature": temperature,
            "sequences_per_refined_backbone": sequences,
        }
    )
    return strategy


def build_maturation_plan(spec, stage1_groups=None):
    """Create an auditable run plan without flattening backbone groups."""
    groups = []
    for backbone_id, record in sorted((stage1_groups or {}).items()):
        groups.append(
            {
                "backbone_id": backbone_id,
                "accepted_stage1_candidates": int(record.get("accepted", 0)),
                "input_directory": record.get("directory"),
                "processing": "independent",
            }
        )
    return {
        "strategy": resolve_maturation_strategy(spec),
        "groups": groups,
        "group_count": len(groups),
    }
