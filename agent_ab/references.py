"""PPIFlow-native command references for explicit Agent-Ab input."""


REFERENCES = {
    "binder": {
        "supported": True,
        "required": [
            "target PDB absolute path",
            "target chain ID",
            "target hotspot residues",
            "binder length range",
            "number of initial backbones",
            "Stage 1 sequences per backbone",
            "PartialFlow backbones per retained group",
            "Stage 2 sequences per PartialFlow backbone",
        ],
        "notes": [
            "The output binder chain defaults to A.",
            "Hotspots must use target-chain PDB numbering, for example B119.",
            "The default full workflow stops after the Stage 2 pred/design structural comparison.",
        ],
        "template": (
            "python sample_binder.py --input_pdb /absolute/path/target.pdb "
            "--target_chain B --binder_chain A --config /path/to/configs/inference_binder.yaml "
            "--specified_hotspots B119,B141,B200 --samples_min_length 75 "
            "--samples_max_length 76 --num_backbones 5 "
            "--stage1_sequences_per_backbone 8 --partial_backbones_per_group 10 "
            "--stage2_sequences_per_backbone 10 --profile full "
            "--model_weights /path/to/model/binder.ckpt "
            "--output_dir /path/to/output/binder_test --name TARGET_NAME"
        ),
    },
    "nanobody": {
        "supported": True,
        "required": [
            "antigen PDB absolute path",
            "antigen chain ID",
            "VHH framework PDB absolute path and heavy-chain ID, unless supplied in one complex PDB",
            "hotspots, or a pre-complexed PDB from which contacts can be inferred",
            "initial backbone, Stage 1 sequence, PartialFlow backbone, and Stage 2 sequence counts",
        ],
        "notes": [
            "The default CDR ranges follow the PPIFlow VHH example: CDRH1 8, CDRH2 8, CDRH3 9-21.",
            "Framework residue numbering must be compatible with PPIFlow/IMGT CDR definitions.",
        ],
        "template": (
            "python sample_antibody_nanobody.py --antigen_pdb /absolute/path/antigen.pdb "
            "--framework_pdb /absolute/path/vhh_framework.pdb --antigen_chain C "
            "--heavy_chain A --specified_hotspots C101,C135,C171,C198 "
            "--cdr_length CDRH1,8-8,CDRH2,8-8,CDRH3,9-21 --num_backbones 5 "
            "--stage1_sequences_per_backbone 8 --partial_backbones_per_group 10 "
            "--stage2_sequences_per_backbone 10 --profile full "
            "--config /path/to/configs/inference_nanobody.yaml "
            "--model_weights /path/to/model/nanobody.ckpt "
            "--output_dir /path/to/output/nanobody_test --name TARGET_NAME"
        ),
    },
    "antibody": {
        "supported": True,
        "required": [
            "antigen PDB absolute path and antigen chain ID",
            "paired VH/VL framework PDB and heavy/light chain IDs, unless supplied in one complex PDB",
            "off-target antigen PDB absolute path and source-chain ID for CLDN18.1 specificity screening",
            "hotspots, or a pre-complexed PDB from which contacts can be inferred",
            "initial backbone, Stage 1 sequence, PartialFlow backbone, and Stage 2 sequence counts",
        ],
        "notes": [
            "scFv uses this antibody route.",
            "The framework must contain both VH and VL, use compatible IMGT numbering, and follow the PPIFlow framework/CDR-loop preparation convention.",
            "The final enrichment excludes the internal VH-VL interface and evaluates VH-antigen plus VL-antigen contacts.",
        ],
        "template": (
            "python sample_antibody_nanobody.py --antigen_pdb /absolute/path/antigen.pdb "
            "--framework_pdb /absolute/path/antibody_framework.pdb --antigen_chain C "
            "--offtarget_pdb /absolute/path/CLDN18.1.pdb --offtarget_chain A "
            "--heavy_chain A --light_chain B --specified_hotspots C11,C14,C15,C101,C107,C108 "
            "--cdr_length CDRH1,8-8,CDRH2,8-8,CDRH3,10-20,CDRL1,6-9,CDRL2,3-3,CDRL3,9-11 "
            "--num_backbones 5 --stage1_sequences_per_backbone 8 "
            "--partial_backbones_per_group 10 --stage2_sequences_per_backbone 10 "
            "--profile full --config /path/to/configs/inference_nanobody.yaml "
            "--model_weights /path/to/model/antibody.ckpt "
            "--output_dir /path/to/output/antibody_test --name TARGET_NAME"
        ),
    },
    "monomer": {
        "supported": False,
        "required": ["length subset", "samples per length"],
        "notes": ["PPIFlow supports this natively, but it is not yet connected to the integrated Agent schema."],
        "template": "Use PPIFlow sample_monomer.py directly until the Agent adapter is implemented.",
    },
    "motif": {
        "supported": False,
        "required": ["motif metadata CSV", "motif name", "length and contig definition"],
        "notes": ["PPIFlow supports motif scaffolding natively, but the integrated Agent does not yet validate motif CSV/contig inputs."],
        "template": "Use PPIFlow sample_monomer.py with inference_scaffolding.yaml until the Agent adapter is implemented.",
    },
}


ALIASES = {"vhh": "nanobody", "scfv": "antibody", "motif_scaffolding": "motif"}


def input_reference(design_type=None):
    if design_type is None:
        return {name: {"supported": value["supported"], "template": value["template"]} for name, value in REFERENCES.items()}
    key = ALIASES.get(str(design_type).lower(), str(design_type).lower())
    if key not in REFERENCES:
        raise ValueError("unknown reference type: %s" % design_type)
    return {"design_type": key, **REFERENCES[key]}


def shell_example(design_type):
    return input_reference(design_type)["template"]
