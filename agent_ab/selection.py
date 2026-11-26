"""Backbone-aware Stage 1 candidate selection."""

from collections import defaultdict


def select_grouped_candidates(records, per_backbone_top_k, allow_below_standard=False):
    """Rank and select independently inside every backbone group.

    Standard/stringent candidates are eligible by default.  Below-standard
    candidates remain in reports but are only used when explicitly requested.
    There is deliberately no cross-backbone pool or global Top-K.
    """
    values = [dict(item) for item in records]
    groups = defaultdict(list)
    for item in values:
        groups[item.get("backbone_id") or "backbone_unknown"].append(item)

    for backbone_id in sorted(groups):
        group = sorted(
            groups[backbone_id],
            key=lambda item: (int(item.get("threshold_tier", 0)), float(item.get("consensus_score", 0.0))),
            reverse=True,
        )
        for rank, item in enumerate(group, 1):
            item["backbone_rank"] = rank
            item["passes_thresholds"] = int(item.get("threshold_tier", 0)) > 0
            item["selected_within_backbone"] = False
        eligible = group if allow_below_standard else [item for item in group if item["passes_thresholds"]]
        for selection_rank, item in enumerate(eligible[:int(per_backbone_top_k)], 1):
            item["selected_within_backbone"] = True
            item["selected_stage1"] = True
            item["selected_top_k"] = True  # compatibility with older readers
            item["selection_rank"] = selection_rank
        for item in group:
            item.setdefault("selected_stage1", False)
            item.setdefault("selected_top_k", False)
            item.setdefault("selection_rank", None)

    values.sort(key=lambda item: (item.get("backbone_id") or "backbone_unknown", int(item["backbone_rank"])))
    return values


def summarize_groups(records):
    groups = defaultdict(lambda: {"evaluated": 0, "passed_thresholds": 0, "selected": 0})
    for item in records:
        summary = groups[item.get("backbone_id") or "backbone_unknown"]
        summary["evaluated"] += 1
        summary["passed_thresholds"] += int(bool(item.get("passes_thresholds")))
        summary["selected"] += int(bool(item.get("selected_stage1")))
    return dict(sorted(groups.items()))
