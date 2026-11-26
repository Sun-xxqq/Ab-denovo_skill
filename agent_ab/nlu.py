"""Restricted natural-language understanding for antibody design requests.

The parser only extracts fields from a small, auditable domain vocabulary.  It
never produces shell text and never executes a tool.  Parsed values still pass
through :mod:`agent_ab.spec` before a run can be materialized.
"""

import re
from pathlib import PurePosixPath


PATH_PATTERN = re.compile(
    r"(?P<path>/[^\s，,。；;：:'\"<>]+?\.(?:pdb|cif|mmcif))(?=$|[\s，,。；;：:'\"<>])",
    re.IGNORECASE,
)
RESIDUE_PATTERN = re.compile(r"(?<![A-Za-z0-9])([A-Za-z][0-9]+[A-Za-z]?)(?![A-Za-z0-9])")


DESIGN_PATTERNS = (
    (
        "nanobody",
        re.compile(r"纳米抗体|单域抗体|nano[- ]?bod(?:y|ies)|(?<![A-Za-z0-9])VHH(?![A-Za-z0-9])", re.IGNORECASE),
    ),
    (
        "binder",
        re.compile(r"蛋白(?:质)?结合(?:物|蛋白)|结合蛋白|(?<![A-Za-z0-9])binder(?![A-Za-z0-9])|mini[- ]?protein", re.IGNORECASE),
    ),
    (
        "antibody",
        re.compile(r"全长抗体|单链抗体|抗体|(?<![A-Za-z0-9])antibod(?:y|ies)(?![A-Za-z0-9])|(?<![A-Za-z0-9])scFv(?![A-Za-z0-9])", re.IGNORECASE),
    ),
)


def _first_match(text, patterns, flags=re.IGNORECASE):
    for pattern in patterns:
        match = re.search(pattern, text, flags=flags)
        if match:
            return match.group(1).strip()
    return None


def _design_type(text):
    for name, pattern in DESIGN_PATTERNS:
        if pattern.search(text):
            return name
    return None


def _paths(text):
    values = []
    for match in PATH_PATTERN.finditer(text):
        path = match.group("path")
        prefix = text[max(0, match.start() - 42) : match.start()].lower()
        if re.search(r"(?:框架|framework|scaffold)[^/]{0,28}$", prefix, flags=re.IGNORECASE):
            role = "framework"
        elif re.search(
            r"(?:阴性靶点|阴性对照|脱靶|negative\s+target|off[- ]?target)[^/]{0,28}$",
            prefix,
            flags=re.IGNORECASE,
        ):
            role = "negative"
        elif re.search(r"(?:抗原|靶点|target|antigen)[^/]{0,28}$", prefix, flags=re.IGNORECASE):
            role = "target"
        else:
            role = "unlabeled"
        values.append({"path": path, "role": role, "start": match.start()})
    return values


def _assign_paths(text):
    values = _paths(text)
    target = next((item["path"] for item in values if item["role"] == "target"), None)
    framework = next((item["path"] for item in values if item["role"] == "framework"), None)
    negatives = [item["path"] for item in values if item["role"] == "negative"]
    unlabeled = [item["path"] for item in values if item["role"] == "unlabeled"]
    if target is None and unlabeled:
        target = unlabeled.pop(0)
    if framework is None and unlabeled:
        framework = unlabeled.pop(0)
    return target, framework, negatives


def _chain(text, kind):
    definitions = {
        "target": (
            r"(?:\u76ee\u6807)\s*(?:\u7684)?\s*(?:\u94fe|chain(?:\s*id)?)\s*[:=]?\s*([A-Za-z0-9])",
            r"(?:抗原|靶点)\s*(?:的)?\s*(?:链|chain(?:\s*id)?)\s*[:=：为是]?\s*([A-Za-z0-9])",
            r"(?:antigen|target)\s*chain(?:\s*id)?\s*[:=]?\s*([A-Za-z0-9])",
        ),
        "heavy": (
            r"(?:重链|heavy\s*chain|H\s*chain)\s*(?:id)?\s*[:=：为是]?\s*([A-Za-z0-9])",
        ),
        "light": (
            r"(?:轻链|light\s*chain|L\s*chain)\s*(?:id)?\s*[:=：为是]?\s*([A-Za-z0-9])",
        ),
        "binder": (
            r"(?:输出\s*)?(?:binder|结合蛋白)\s*(?:链|chain)\s*[:=：为是]?\s*([A-Za-z0-9])",
        ),
        "input_binder": (
            r"(?:输入|已有|模板)\s*(?:binder|结合蛋白)\s*(?:链|chain)\s*[:=：为是]?\s*([A-Za-z0-9])",
        ),
    }
    value = _first_match(text, definitions[kind])
    return value.upper() if value else None


def _hotspots(text):
    marker = re.search(r"热点(?:残基)?|表位(?:残基)?|hotspots?|epitope(?:\s+residues?)?", text, re.IGNORECASE)
    if not marker:
        return []
    segment = text[marker.end() : marker.end() + 180]
    segment = re.split(
        r"(?:框架|framework|生成|采样|每个骨架|stage|阶段|GPU|长度|阴性靶点|negative\s+target)",
        segment,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    values = []
    for match in RESIDUE_PATTERN.finditer(segment):
        value = match.group(1).upper()
        if value not in values:
            values.append(value)
    return values


def _integer(text, patterns):
    value = _first_match(text, patterns)
    return int(value) if value is not None else None


def _samples(text):
    return _integer(
        text,
        (
            r"(?:generate|create|sample)\s+([0-9]+)\s+(?:initial\s+)?backbones?",
            r"([0-9]+)\s+(?:initial\s+)?backbones?",
            r"(?:生成|采样|骨架数|样本数|samples?)\s*[:=：为是]?\s*([0-9]+)\s*(?:个|条)?(?:骨架|样本|结构|designs?)?",
            r"([0-9]+)\s*(?:个|条)\s*(?:骨架|样本|结构|designs?)",
        ),
    )


def _sequences_per_backbone(text):
    return _integer(
        text,
        (
            r"([0-9]+)\s+sequences?\s+per\s+(?:initial\s+)?backbone",
            r"(?:每个|每条)\s*骨架(?:设计|生成|采样)?\s*([0-9]+)\s*(?:个|条)?\s*(?:序列|sequence)",
            r"(?:num_sequences_per_backbone|sequences?\s+per\s+backbone)\s*[:=]\s*([0-9]+)",
        ),
    )


def _redesign_candidates_per_group(text):
    """Extract an explicit total Stage 2 candidate count for each group."""
    return _integer(
        text,
        (
            r"(?:redesign_candidates_per_group|stage2_candidates_per_group)\s*[:=]\s*([0-9]+)",
            r"(?:\u7b5b\u9009\u540e|stage\s*2|partial\s*flow)[^0-9]{0,80}"
            r"(?:\u6bcf\u7ec4|\u6bcf\u4e2a\u539f\u59cb\u9aa8\u67b6\u7ec4|per\s+(?:original\s+)?backbone\s+group)"
            r"[^0-9]{0,30}(?:\u91cd\u65b0)?(?:\u751f\u6210|generate)?\s*([0-9]+)",
        ),
    )


def _binder_length(text):
    patterns = (
        r"(?:binder|结合蛋白)?\s*长度(?:范围)?\s*[:=：为是]?\s*([0-9]+)\s*(?:-|–|—|~|至|到)\s*([0-9]+)\s*(?:aa|残基)?",
        r"([0-9]+)\s*(?:-|–|—|~|至|到)\s*([0-9]+)\s*(?:aa|个残基)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return {"min": int(match.group(1)), "max": int(match.group(2))}
    return None


def _profile(text):
    if re.search(r"完整(?:流程|两阶段)?|全流程|两阶段|stage\s*1\s*(?:and|\+|和)\s*2|\bfull\b", text, re.IGNORECASE):
        return "full"
    if re.search(r"第一阶段|阶段\s*1|stage\s*1|小规模|先测试", text, re.IGNORECASE):
        return "stage1"
    return None


def _gpu(text):
    value = _integer(text, (r"\bGPU\s*[:=#-]?\s*([0-9]+)",))
    return value if value is not None else "auto"


def _requested_action(text):
    if re.search(r"不要运行|不运行|只(?:做)?(?:规划|计划|解析)|dry[- ]?run|plan\s+only", text, re.IGNORECASE):
        return "plan"
    if re.search(r"确认运行|立即运行|直接运行|开始运行|启动计算|run\s+now|execute\s+now", text, re.IGNORECASE):
        return "run"
    return "plan"


def _explicit_name(text):
    return _first_match(
        text,
        (
            r"(?:任务名|任务名称|name)\s*[:=：为是]\s*([A-Za-z0-9_.\-\u4e00-\u9fff]{1,64})",
        ),
    )


def _derive_name(target_path, design_type):
    target = PurePosixPath(target_path).stem if target_path else "antibody"
    return "%s-%s" % (target, design_type or "design")


def missing_fields(spec):
    missing = []
    design_type = spec.get("design_type")
    if design_type not in ("binder", "nanobody", "antibody"):
        missing.append("design_type")
    target = spec.get("target", {})
    if not target.get("structure"):
        missing.append("target.structure")
    # Chain/framework fields may be inferred later from conventional H/L antibody
    # chains when a local complex PDB was supplied.
    complex_path = target.get("structure")
    inferred = {}
    if complex_path and design_type in ("antibody", "nanobody"):
        try:
            from .contacts import infer_complex_roles
            inferred = infer_complex_roles(complex_path, design_type)
        except (OSError, ValueError):
            inferred = {}
    if not target.get("chain") and not inferred.get("target_chain"):
        missing.append("target.chain")
    if design_type == "binder" and not target.get("hotspots"):
        missing.append("target.hotspots")
    if design_type in ("nanobody", "antibody"):
        framework = spec.get("framework", {})
        if not framework.get("structure") and not inferred.get("heavy_chain"):
            missing.append("framework.structure")
        if not framework.get("heavy_chain") and not inferred.get("heavy_chain"):
            missing.append("framework.heavy_chain")
        if design_type == "antibody" and not framework.get("light_chain") and not inferred.get("light_chain"):
            missing.append("framework.light_chain")
    return missing


def set_field(spec, field, value):
    """Set one prompt-completed field on a partially parsed spec."""
    if field == "design_type":
        design_type = str(value).strip().lower()
        spec[field] = design_type
        if design_type == "binder":
            spec.setdefault("generation", {}).setdefault("binder_chain", "A")
            spec["generation"].setdefault("length", {"min": 50, "max": 80})
        elif design_type in ("nanobody", "antibody"):
            spec.setdefault("framework", {})
        return
    section, key = field.split(".", 1)
    spec.setdefault(section, {})
    if field == "target.hotspots":
        spec[section][key] = [item.upper() for item in RESIDUE_PATTERN.findall(str(value))]
    else:
        spec[section][key] = str(value).strip()


def parse_request(text):
    text = " ".join(str(text).strip().split())
    if not text:
        raise ValueError("request cannot be empty")
    design_type = _design_type(text)
    target_path, framework_path, negative_paths = _assign_paths(text)
    target_chain = _chain(text, "target")
    hotspots = _hotspots(text)
    profile = _profile(text) or "full"
    sequences_per_backbone = _sequences_per_backbone(text) or 8
    redesign_candidates_per_group = _redesign_candidates_per_group(text)
    default_samples = (100 + sequences_per_backbone - 1) // sequences_per_backbone if profile == "full" else 10
    samples = _samples(text) or default_samples
    spec = {
        "schema_version": 1,
        "name": _explicit_name(text) or _derive_name(target_path, design_type),
        "target": {
            "structure": target_path,
            "chain": target_chain,
            "hotspots": hotspots,
        },
        "generation": {
            "backend": "ppiflow",
            "profile": profile,
            "samples": samples,
            "num_sequences_per_backbone": sequences_per_backbone,
        },
        "validation": {
            "independent_predictors": ["protenix", "chai"],
            "interface_scoring": ["ipsae", "pyrosetta"],
            "negative_targets": negative_paths,
            "candidate_screening": {
                "max_candidates_per_backbone": sequences_per_backbone,
                "mode": "loose_filter_only",
                "delete_rejected": False,
                "iterative_redesign": True,
            },
        },
    }
    if design_type:
        spec["design_type"] = design_type
    if redesign_candidates_per_group is not None:
        screening = spec["validation"]["candidate_screening"]
        screening["maturation"] = {
            "partial_flow": {
                "refined_backbones_per_seed": redesign_candidates_per_group,
            },
            "round2_sequence_design": {
                "sequences_per_refined_backbone": 1,
            },
            "final_generation": {
                "candidates_per_original_backbone_group": redesign_candidates_per_group,
                "allocation": "%s_refined_backbones_x_1_sequence"
                % redesign_candidates_per_group,
                "require_exact_count_before_functional_evaluation": True,
            },
        }
    heavy_chain = _chain(text, "heavy")
    light_chain = _chain(text, "light")
    if framework_path or heavy_chain or light_chain:
        spec["framework"] = {
            "structure": framework_path,
            "heavy_chain": heavy_chain,
        }
        if light_chain:
            spec["framework"]["light_chain"] = light_chain
    assumptions = []
    if _profile(text) is None:
        assumptions.append("未识别到计算档位，默认使用 stage1")
    if _samples(text) is None:
        assumptions.append("未识别到骨架数，默认生成 10 个")
    if _sequences_per_backbone(text) is None:
        assumptions.append("未识别到每骨架序列数，默认使用 8 条")

    if design_type == "binder":
        spec["generation"]["binder_chain"] = _chain(text, "binder") or "A"
        spec["generation"]["length"] = _binder_length(text) or {"min": 50, "max": 80}
        input_binder = _chain(text, "input_binder")
        if input_binder:
            spec["target"]["binder_chain"] = input_binder
        if _binder_length(text) is None:
            assumptions.append("未识别到 binder 长度，默认使用 50-80 aa")
    elif design_type in ("nanobody", "antibody"):
        spec.setdefault("framework", {})
        spec["framework"].setdefault("structure", framework_path)
        spec["framework"].setdefault("heavy_chain", heavy_chain)
        if design_type == "antibody":
            spec["framework"].setdefault("light_chain", light_chain)
            spec["generation"]["antibody_format"] = (
                "scfv" if re.search(r"(?<![A-Za-z0-9])scFv(?![A-Za-z0-9])", text, re.IGNORECASE)
                else "paired_vh_vl"
            )

    if target_path and design_type in ("nanobody", "antibody"):
        try:
            from .contacts import infer_complex_roles, interface_hotspots
            inferred = infer_complex_roles(target_path, design_type)
            if inferred.get("target_chain"):
                spec["target"]["chain"] = spec["target"].get("chain") or inferred["target_chain"]
                spec.setdefault("framework", {})
                spec["framework"]["structure"] = spec["framework"].get("structure") or target_path
                spec["framework"]["heavy_chain"] = spec["framework"].get("heavy_chain") or inferred["heavy_chain"]
                partners = [spec["framework"]["heavy_chain"]]
                if design_type == "antibody":
                    spec["framework"]["light_chain"] = spec["framework"].get("light_chain") or inferred["light_chain"]
                    partners.append(spec["framework"]["light_chain"])
                if not spec["target"]["hotspots"]:
                    spec["target"]["hotspots"] = interface_hotspots(
                        target_path, spec["target"]["chain"], partners
                    )
                    if spec["target"]["hotspots"]:
                        spec["target"]["hotspot_source"] = "auto_heavy_atom_contacts_5A"
                        assumptions.append("已从复合物PDB自动识别链角色并计算界面热点")
        except (OSError, ValueError):
            pass

    warnings = []
    if design_type in ("nanobody", "antibody") and not spec["target"]["hotspots"]:
        warnings.append("未提供热点残基；允许规划，但建议在真实靶点任务中明确表位")
    result = {
        "parser": "restricted_antibody_nlu_v1",
        "original_request": text,
        "requested_action": _requested_action(text),
        "gpu": _gpu(text),
        "spec": spec,
        "missing": missing_fields(spec),
        "assumptions": assumptions,
        "warnings": warnings,
    }
    return result


PROMPTS = {
    "design_type": "未识别设计类型，请输入 binder/nanobody/antibody",
    "target.structure": "请输入抗原 PDB 绝对路径",
    "target.chain": "请输入抗原链 ID",
    "target.hotspots": "请输入热点残基，逗号分隔（如 C10,C13,C14）",
    "framework.structure": "请输入抗体框架 PDB 绝对路径",
    "framework.heavy_chain": "请输入框架重链 ID",
    "framework.light_chain": "请输入框架轻链 ID",
}
