import csv
import json
from pathlib import Path


DEFAULT_WEIGHTS = {
    "plddt": 0.15,
    "ptm": 0.10,
    "scrmsd": 0.15,
    "pae": 0.15,
    "ipsae": 0.25,
    "iptm": 0.20,
}
LOWER_IS_BETTER = {"scrmsd", "interface_energy", "pae", "clash_score"}


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize(values, lower=False):
    usable = [value for value in values if value is not None]
    if not usable:
        return [None] * len(values)
    low, high = min(usable), max(usable)
    if high == low:
        return [1.0 if value is not None else None for value in values]
    result = []
    for value in values:
        if value is None:
            result.append(None)
        else:
            score = (value - low) / (high - low)
            result.append(1.0 - score if lower else score)
    return result


def rank_candidates(records, weights=None):
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    records = [dict(item) for item in records]
    normalized = {}
    for metric in weights:
        normalized[metric] = _normalize(
            [_number(item.get(metric)) for item in records],
            lower=metric in LOWER_IS_BETTER,
        )
    for index, item in enumerate(records):
        numerator = 0.0
        denominator = 0.0
        evidence = 0
        for metric, weight in weights.items():
            value = normalized[metric][index]
            if value is not None:
                numerator += value * float(weight)
                denominator += float(weight)
                evidence += 1
        preset = _number(item.get("threshold_consensus_score"))
        item["consensus_score"] = preset if preset is not None else (round(numerator / denominator, 6) if denominator else None)
        item["evidence_metrics"] = evidence
    records.sort(
        key=lambda item: (
            int(item.get("threshold_tier", 0)),
            item["consensus_score"] is not None,
            item["consensus_score"] or -1,
            item["evidence_metrics"],
        ),
        reverse=True,
    )
    for rank, item in enumerate(records, start=1):
        item["rank"] = rank
    return records


def rank_candidates_by_backbone(records, weights=None):
    """Produce independent rankings without comparing different backbones."""
    groups = {}
    for item in records:
        groups.setdefault(item.get("backbone_id") or "backbone_unknown", []).append(item)
    ranked = []
    for backbone_id in sorted(groups):
        group = rank_candidates(groups[backbone_id], weights=weights)
        for item in group:
            item["backbone_id"] = backbone_id
            item["backbone_rank"] = item.pop("rank")
        ranked.extend(group)
    return ranked


def write_ranking(run_dir, records, weights=None):
    run_dir = Path(run_dir)
    ranked = (
        rank_candidates_by_backbone(records, weights=weights)
        if any(item.get("backbone_id") for item in records)
        else rank_candidates(records, weights=weights)
    )
    json_path = run_dir / "candidate_ranking.json"
    json_path.write_text(json.dumps(ranked, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = run_dir / "candidate_ranking.csv"
    fields = sorted({key for item in ranked for key in item})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ranked)
    return {"count": len(ranked), "json": str(json_path), "csv": str(csv_path)}
