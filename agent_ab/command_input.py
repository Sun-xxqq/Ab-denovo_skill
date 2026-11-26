"""Translate an explicit PPIFlow-style command into an Agent-Ab job spec.

This module deliberately does not infer intent from prose.  The caller selects
the design type first and every scientific input is supplied as a named flag.
"""

import argparse
import shlex


SUPPORTED_DESIGN_TYPES = ("binder", "nanobody", "antibody")


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def _parser(design_type):
    parser = _ArgumentParser(add_help=False, allow_abbrev=False)
    if design_type == "binder":
        parser.add_argument("--input_pdb", required=True)
        parser.add_argument("--target_chain", required=True)
        parser.add_argument("--binder_chain", default="A")
        parser.add_argument("--specified_hotspots", required=True)
        parser.add_argument("--samples_min_length", type=int, required=True)
        parser.add_argument("--samples_max_length", type=int, required=True)
    else:
        parser.add_argument("--antigen_pdb", required=True)
        parser.add_argument("--framework_pdb", required=True)
        parser.add_argument("--antigen_chain", required=True)
        parser.add_argument(
            "--offtarget_pdb",
            "--off-target-pdb",
            dest="offtarget_pdb",
            help="absolute PDB path for the negative/off-target antigen",
        )
        parser.add_argument(
            "--offtarget_chain",
            "--off-target-chain",
            dest="offtarget_chain",
            default="A",
            help="source chain to read from --offtarget_pdb (default: A)",
        )
        parser.add_argument("--heavy_chain", required=True)
        if design_type == "antibody":
            parser.add_argument("--light_chain", required=True)
        parser.add_argument("--specified_hotspots", required=True)
        parser.add_argument("--cdr_length")

    parser.add_argument(
        "--samples_per_target",
        "--num_backbones",
        dest="samples_per_target",
        type=int,
        default=5,
    )
    parser.add_argument("--name", required=True)

    # Native runtime locations are accepted so a command copied from the
    # PPIFlow README can be pasted unchanged. Agent-Ab obtains the authoritative
    # locations from configs/site.json instead of trusting user-provided paths.
    parser.add_argument("--config")
    parser.add_argument("--model_weights")
    parser.add_argument("--output_dir")
    parser.add_argument("--retry_Limit")

    # Agent-Ab orchestration controls, optional in addition to native flags.
    parser.add_argument("--profile", choices=("stage1", "full"), default="full")
    parser.add_argument(
        "--num_sequences_per_backbone",
        "--stage1_sequences_per_backbone",
        dest="num_sequences_per_backbone",
        type=int,
        default=8,
    )
    parser.add_argument("--partial_backbones_per_group", type=int, default=1)
    parser.add_argument("--stage2_sequences_per_backbone", type=int, default=4)
    parser.add_argument("--redesign_candidates_per_group", type=int)
    parser.add_argument(
        "--complex_prediction",
        "--complex-prediction",
        choices=("protenix", "chai"),
        default="protenix",
        help="complex-structure prediction backend for both screening rounds",
    )
    return parser


def _tokens(command):
    if isinstance(command, str):
        value = command.replace("\\\n", " ").strip()
        tokens = shlex.split(value)
    else:
        tokens = list(command)
    if tokens[:1] == ["--"]:
        tokens = tokens[1:]
    if tokens and tokens[0] in ("python", "python3"):
        tokens = tokens[1:]
    if tokens and tokens[0].endswith(
        ("sample_binder.py", "sample_antibody_nanobody.py")
    ):
        tokens = tokens[1:]
    return tokens


def _hotspots(value):
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def parse_ppiflow_command(design_type, command):
    design_type = str(design_type).strip().lower()
    if design_type not in SUPPORTED_DESIGN_TYPES:
        raise ValueError(
            "design_type must be one of: %s" % ", ".join(SUPPORTED_DESIGN_TYPES)
        )
    tokens = _tokens(command)
    if not tokens:
        raise ValueError("PPIFlow command cannot be empty")
    parser = _parser(design_type)
    try:
        args, unknown = parser.parse_known_args(tokens)
    except ValueError as exc:
        raise ValueError("invalid %s PPIFlow command: %s" % (design_type, exc)) from exc
    if unknown:
        raise ValueError("unsupported PPIFlow arguments: %s" % " ".join(unknown))

    generation = {
        "backend": "ppiflow",
        "profile": args.profile,
        "samples": args.samples_per_target,
        "num_sequences_per_backbone": args.num_sequences_per_backbone,
        "stage2_sequences_per_backbone": args.stage2_sequences_per_backbone,
    }
    if design_type == "binder":
        target_structure = args.input_pdb
        target_chain = args.target_chain
        generation.update(
            {
                "binder_chain": args.binder_chain,
                "length": {
                    "min": args.samples_min_length,
                    "max": args.samples_max_length,
                },
            }
        )
    else:
        target_structure = args.antigen_pdb
        target_chain = args.antigen_chain
        generation["cdr_lengths"] = args.cdr_length or (
            "CDRH1,8-8,CDRH2,8-8,CDRH3,9-21"
            if design_type == "nanobody"
            else "CDRH1,8-8,CDRH2,8-8,CDRH3,10-20,CDRL1,6-9,CDRL2,3-3,CDRL3,9-11"
        )

    screening = {
        "max_candidates_per_backbone": args.num_sequences_per_backbone,
        "mode": "loose_filter_only",
        "delete_rejected": False,
        "iterative_redesign": True,
        "stage2_structural_comparison": {
            "enabled": True,
            "predictor": args.complex_prediction,
        },
        "stage3_iggm_specificity": {
            "enabled": True,
            "sidechain_packing": "flowpacker",
            "flowpacker_seed": 42,
            "predictor": "protenix",
        },
        # This workflow intentionally stops after the second-round
        # prediction/design comparison.  The old affinity/ranking tail remains
        # available to hand-authored specs by explicitly enabling it.
        "final_functional_evaluation": {"enabled": False},
    }
    partial_backbones = args.partial_backbones_per_group
    stage2_sequences = args.stage2_sequences_per_backbone
    if args.redesign_candidates_per_group is not None:
        # Preserve the previous flag's meaning: N total candidates, allocated
        # as N PartialFlow backbones x one sequence.  New commands should use
        # the two explicit count flags above.
        partial_backbones = args.redesign_candidates_per_group
        stage2_sequences = 1
        generation["stage2_sequences_per_backbone"] = stage2_sequences
    final_count = partial_backbones * stage2_sequences
    screening["maturation"] = {
        "partial_flow": {"refined_backbones_per_seed": partial_backbones},
        "round2_sequence_design": {
            "sequences_per_refined_backbone": stage2_sequences
        },
        "final_generation": {
            "candidates_per_original_backbone_group": final_count,
            "allocation": "%s_partial_backbones_x_%s_sequences"
            % (partial_backbones, stage2_sequences),
            "require_exact_count_before_structural_comparison": True,
        },
    }

    spec = {
        "schema_version": 1,
        "name": args.name,
        "design_type": design_type,
        "target": {
            "structure": target_structure,
            "chain": target_chain.upper(),
            "hotspots": _hotspots(args.specified_hotspots),
        },
        "generation": generation,
        "validation": {
            "independent_predictors": list(
                dict.fromkeys([args.complex_prediction, "chai"])
            ),
            "interface_scoring": ["ipsae", "pyrosetta"],
            "negative_targets": [],
            "candidate_screening": screening,
        },
        "workflow": {"complex_prediction": args.complex_prediction},
    }
    if design_type != "binder":
        spec["framework"] = {
            "structure": args.framework_pdb,
            "heavy_chain": args.heavy_chain.upper(),
        }
        if design_type == "antibody":
            spec["framework"]["light_chain"] = args.light_chain.upper()
        if args.offtarget_pdb:
            screening["off_target_screening"] = {
                "enabled": True,
                "name": "CLDN18.1",
                "structure": args.offtarget_pdb,
                "source_chain": args.offtarget_chain.upper(),
            }

    return {
        "parser": "ppiflow_argv_v1",
        "design_type_selected": design_type,
        "original_command": " ".join(tokens),
        "ignored_runtime_arguments": {
            key: value
            for key, value in {
                "config": args.config,
                "model_weights": args.model_weights,
                "output_dir": args.output_dir,
            }.items()
            if value
        },
        "spec": spec,
    }
