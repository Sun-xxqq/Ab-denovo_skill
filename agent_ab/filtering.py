"""Stage-specific Protenix design/pred filters."""


DEFAULT_STAGE1_THRESHOLDS = {
    "antibody_plddt_min": 0.60,
    "ptm_min": 0.60,
    "iptm_global_min": 0.25,
    "design_pred_rmsd_max": 6.0,
}

DEFAULT_STAGE2_THRESHOLDS = {
    **DEFAULT_STAGE1_THRESHOLDS,
    "design_pred_rmsd_max": 8.0,
    "design_pred_rmsd_max_inclusive": True,
    "s_on_min": 0.20,
}

# Compatibility name used by the Stage 1 evaluator and historical callers.
DEFAULT_LOOSE_THRESHOLDS = DEFAULT_STAGE1_THRESHOLDS


def loose_filter_reasons(record, thresholds=None):
    limits = dict(DEFAULT_LOOSE_THRESHOLDS)
    limits.update(thresholds or {})
    reasons = []
    rmsd_direction = (
        "max_inclusive"
        if limits.get("design_pred_rmsd_max_inclusive", False)
        else "max_exclusive"
    )
    checks = [
        ("antibody_plddt", "antibody_plddt_min", "min"),
        ("ptm", "ptm_min", "min"),
        ("iptm_global", "iptm_global_min", "min"),
        ("design_pred_rmsd", "design_pred_rmsd_max", rmsd_direction),
    ]
    if "s_on_min" in limits:
        checks.append(("S_on", "s_on_min", "min"))
    for metric, threshold, direction in checks:
        value = record.get(metric)
        if value is None:
            reasons.append("missing_%s" % metric)
        elif direction == "min" and float(value) < float(limits[threshold]):
            reasons.append("%s_below_%s" % (metric, limits[threshold]))
        elif direction == "max_exclusive" and float(value) >= float(limits[threshold]):
            reasons.append("%s_not_below_%s" % (metric, limits[threshold]))
        elif direction == "max_inclusive" and float(value) > float(limits[threshold]):
            reasons.append("%s_above_%s" % (metric, limits[threshold]))
    return reasons


def passes_loose_filter(record, thresholds=None):
    reasons = loose_filter_reasons(record, thresholds=thresholds)
    return not reasons, reasons
