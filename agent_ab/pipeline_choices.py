STAGES = (
    "backbone_generation",
    "sequence_design",
    "sidechain_packing",
    "complex_prediction",
    "interface_scoring",
)

TOOL_FOR_CHOICE = {
    "ppiflow": "ppiflow",
    "rfdiffusion": "rfdiffusion",
    "rfdiffusion3": "rfdiffusion3",
    "boltzgen": "boltzgen",
    "ppiflow_mpnn": "ppiflow",
    "proteinmpnn": "proteinmpnn",
    "ppiflow_flowpacker": "ppiflow",
    "pyrosetta": "pyrosetta",
    "protenix": "protenix",
    "chai": "chai",
    "ppiflow_af3score": "ppiflow",
    "ipsae": "ipsae",
}


def normalize_workflow(raw, catalog=None):
    """Validate per-stage software selections and apply site defaults."""
    raw = dict(raw or {})
    if catalog is None:
        catalog = {
            "backbone_generation": {"default": "ppiflow", "choices": ["ppiflow", "rfdiffusion", "rfdiffusion3", "boltzgen"]},
            "sequence_design": {"default": "ppiflow_mpnn", "choices": ["ppiflow_mpnn", "proteinmpnn"]},
            "sidechain_packing": {"default": "ppiflow_flowpacker", "choices": ["ppiflow_flowpacker", "pyrosetta"]},
            "complex_prediction": {"default": "protenix", "choices": ["protenix", "chai", "rfdiffusion3", "ppiflow_af3score"]},
            "interface_scoring": {"default": "ipsae", "choices": ["ipsae", "pyrosetta"]},
        }
    unknown = sorted(set(raw) - set(STAGES))
    if unknown:
        raise ValueError("unknown workflow stages: %s" % ", ".join(unknown))
    result = {}
    for stage in STAGES:
        definition = catalog[stage]
        selected = str(raw.get(stage, definition["default"])).lower()
        if selected not in definition["choices"]:
            raise ValueError(
                "workflow.%s must be one of: %s"
                % (stage, ", ".join(definition["choices"]))
            )
        result[stage] = selected
    return result


def build_workflow_plan(spec, config):
    selected = normalize_workflow(spec.get("workflow"), config.get("workflow_catalog"))
    stages = []
    for index, stage in enumerate(STAGES, start=1):
        choice = selected[stage]
        tool_name = TOOL_FOR_CHOICE[choice]
        tool = config["tools"][tool_name]
        stages.append(
            {
                "order": index,
                "stage": stage,
                "selected": choice,
                "tool": tool_name,
                "available_choices": config["workflow_catalog"][stage]["choices"],
                "path": tool.get("path"),
                "environment": tool.get("env"),
                "model": tool.get("model"),
                "weights": tool.get("weights") or tool.get("weights_dir"),
                "cache_dir": tool.get("cache_dir"),
                "execution_state": (
                    "automated_in_ppiflow"
                    if choice.startswith("ppiflow")
                    else "executable_adapter"
                ),
            }
        )
    return {
        "selected": selected,
        "stages": stages,
        "execution_boundary": (
            "PPIFlow runs through its native pipeline. Other selections run through "
            "tool-specific adapters with checkpointed state, retries and consensus ranking."
        ),
    }
