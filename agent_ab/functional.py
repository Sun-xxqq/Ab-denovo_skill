"""Final, per-backbone functional assessment for matured candidates.

The affinity value produced here is deliberately named an affinity *proxy*.
It combines structure confidence and interface energetics and is not a Kd/Ka
prediction or a replacement for SPR/BLI measurements.
"""

import csv
import json
import math
import re
from pathlib import Path


DEFAULT_WEIGHTS = {
    "ipsae_min": 0.15,
    "iptm": 0.10,
    "interface_energy": 0.15,
    "interface_delta_sasa": 0.08,
    "shape_complementarity": 0.07,
    "interface_hbonds": 0.05,
    "buried_unsatisfied_hbonds": 0.05,
    "plddt": 0.10,
    "ptm": 0.05,
    "interface_pae_mean": 0.05,
    "hotspot_coverage": 0.05,
    "cdr_contact_fraction": 0.05,
    "developability_risk": 0.05,
}

LOWER_IS_BETTER = {
    "interface_energy",
    "interface_pae_mean",
    "developability_risk",
    "cdr_scrmsd",
    "pose_rmsd",
    "buried_unsatisfied_hbonds",
}

AFFINITY_PROXY_METRICS = {
    "ipsae_min",
    "iptm",
    "interface_energy",
    "interface_delta_sasa",
    "shape_complementarity",
    "interface_hbonds",
    "buried_unsatisfied_hbonds",
}


def sequence_developability(sequence):
    """Return transparent sequence-liability counts, not a black-box score."""
    sequence = str(sequence or "").upper()
    liabilities = {
        "extra_cysteines": max(0, sequence.count("C") - 2),
        "n_glycosylation_motifs": len(re.findall(r"N[^P][ST]", sequence)),
        "deamidation_motifs": len(re.findall(r"N[GS]", sequence)),
        "isomerization_motifs": len(re.findall(r"D[GS]", sequence)),
        "hydrophobic_runs": len(re.findall(r"[AILMFWVY]{4,}", sequence)),
    }
    liabilities["developability_risk"] = sum(liabilities.values())
    liabilities["approx_net_charge"] = (
        sequence.count("K") + sequence.count("R") + 0.1 * sequence.count("H")
        - sequence.count("D") - sequence.count("E")
    )
    return liabilities


def functional_gate_reasons(record, gates):
    reasons = []
    if record.get("has_inter_chain_clashes") is not False:
        reasons.append("inter_chain_clashes_or_missing")
    checks = (
        ("plddt", "plddt_min", "min"),
        ("ptm", "ptm_min", "min"),
        ("iptm", "iptm_min", "min"),
        ("ipsae_min", "ipsae_min", "min"),
        ("interface_pae_mean", "interface_pae_mean_max", "max"),
        ("interface_energy", "interface_energy_max", "max"),
        ("interface_delta_sasa", "interface_delta_sasa_min", "min"),
        ("developability_risk", "developability_risk_max", "max"),
    )
    for metric, threshold_name, direction in checks:
        limit = gates.get(threshold_name)
        if limit is None:
            continue
        value = record.get(metric)
        if value is None:
            reasons.append("missing_%s" % metric)
        elif direction == "min" and float(value) < float(limit):
            reasons.append("%s_below_%s" % (metric, limit))
        elif direction == "max" and float(value) > float(limit):
            reasons.append("%s_above_%s" % (metric, limit))

    if int(record.get("hotspots_total") or 0) > 0:
        coverage = record.get("hotspot_coverage")
        if coverage is None:
            reasons.append("missing_hotspot_coverage")
        elif float(coverage) < float(gates.get("hotspot_coverage_min", 0.5)):
            reasons.append("hotspot_coverage_below_%s" % gates.get("hotspot_coverage_min", 0.5))
    cdr_limit = gates.get("cdr_contact_fraction_min")
    if cdr_limit is not None:
        value = record.get("cdr_contact_fraction")
        if value is None:
            reasons.append("missing_cdr_contact_fraction")
        elif float(value) < float(cdr_limit):
            reasons.append("cdr_contact_fraction_below_%s" % cdr_limit)
    return reasons


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _normalized(records, metric):
    values = [_number(item.get(metric)) for item in records]
    usable = [value for value in values if value is not None]
    if not usable:
        return [None] * len(values)
    low, high = min(usable), max(usable)
    if low == high:
        return [1.0 if value is not None else None for value in values]
    result = []
    for value in values:
        if value is None:
            result.append(None)
        else:
            score = (value - low) / (high - low)
            result.append(1.0 - score if metric in LOWER_IS_BETTER else score)
    return result


def _weighted_score(index, normalized, weights, selected=None):
    numerator = denominator = 0.0
    for metric, weight in weights.items():
        if selected is not None and metric not in selected:
            continue
        value = normalized[metric][index]
        if value is not None:
            numerator += value * float(weight)
            denominator += float(weight)
    return round(numerator / denominator, 6) if denominator else None


def assess_functional_candidates(records, gates, weights=None):
    """Gate and rank candidates independently inside each backbone group."""
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    groups = {}
    for record in records:
        groups.setdefault(record.get("backbone_id") or "backbone_unknown", []).append(dict(record))
    output = []
    for backbone_id in sorted(groups):
        group = groups[backbone_id]
        normalized = {metric: _normalized(group, metric) for metric in weights}
        for index, item in enumerate(group):
            reasons = functional_gate_reasons(item, gates)
            item["functional_pass"] = not reasons
            item["functional_rejection_reasons"] = reasons
            item["functional_score"] = _weighted_score(index, normalized, weights)
            item["affinity_proxy_score"] = _weighted_score(
                index, normalized, weights, selected=AFFINITY_PROXY_METRICS
            )
            item["affinity_proxy_interpretation"] = "relative computational proxy; not Kd/Ka"
        group.sort(
            key=lambda item: (
                bool(item["functional_pass"]),
                item["functional_score"] is not None,
                item["functional_score"] or -1,
            ),
            reverse=True,
        )
        passing_rank = 0
        for rank, item in enumerate(group, start=1):
            item["backbone_id"] = backbone_id
            item["group_rank_all"] = rank
            if item["functional_pass"]:
                passing_rank += 1
                item["group_rank_passing"] = passing_rank
            else:
                item["group_rank_passing"] = None
            output.append(item)
    return output


def write_functional_assessment(output_dir, records, gates, weights=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = assess_functional_candidates(records, gates=gates, weights=weights)
    json_path = output_dir / "functional_assessment.json"
    csv_path = output_dir / "functional_assessment.csv"
    json_path.write_text(json.dumps(ranked, indent=2) + "\n", encoding="utf-8")
    fields = sorted({key for item in ranked for key in item})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ranked)
    passing = [item for item in ranked if item["functional_pass"]]
    (output_dir / "passing_candidates.json").write_text(
        json.dumps(passing, indent=2) + "\n", encoding="utf-8"
    )
    passing_csv = output_dir / "passing_candidates.csv"
    passing_fields = sorted({key for item in passing for key in item})
    with passing_csv.open("w", encoding="utf-8", newline="") as handle:
        if passing_fields:
            writer = csv.DictWriter(handle, fieldnames=passing_fields)
            writer.writeheader()
            writer.writerows(passing)
    summary = {
        "evaluated": len(ranked),
        "passed": sum(1 for item in ranked if item["functional_pass"]),
        "rejected": sum(1 for item in ranked if not item["functional_pass"]),
        "groups": {},
        "affinity_note": "affinity_proxy_score is not an experimental or predicted Kd/Ka",
    }
    for item in ranked:
        group = summary["groups"].setdefault(
            item["backbone_id"], {"evaluated": 0, "passed": 0, "top_passing_candidate_ids": []}
        )
        group["evaluated"] += 1
        group["passed"] += int(item["functional_pass"])
        if item["functional_pass"] and len(group["top_passing_candidate_ids"]) < 10:
            group["top_passing_candidate_ids"].append(item["candidate_id"])
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
