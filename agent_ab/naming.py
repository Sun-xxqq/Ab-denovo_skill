"""Stable, task-independent backbone identifiers shared by both stages."""

import os
import re


_STAGE1 = re.compile(r"(?:^|_)(?:backbone|flowpacker)\.0_(\d+)$", re.IGNORECASE)
_STAGE2 = re.compile(r"(?:^|_)(?:backbone|flowpacker)\.1_(\d+)_(\d+)$", re.IGNORECASE)
_LEGACY_STAGE2 = re.compile(
    r"(?:^|_)(?:backbone|flowpacker)\.0_(\d+)_sample(\d+)$",
    re.IGNORECASE,
)


def _stem(value):
    name = os.path.basename(str(value))
    for suffix in (".mmcif", ".pdb", ".cif", ".fasta", ".fa"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


def stage1_backbone_id(index):
    return "backbone.0_%d" % int(index)


def stage2_backbone_id(parent_index, sample_index):
    return "backbone.1_%d_%d" % (int(parent_index), int(sample_index))


def parse_backbone_id(value):
    """Return ``(stage, parent_index, sample_index_or_None)``."""
    stem = _stem(value)
    match = _STAGE2.search(stem)
    if match:
        return 2, int(match.group(1)), int(match.group(2))
    match = _LEGACY_STAGE2.search(stem)
    if match:
        return 2, int(match.group(1)), int(match.group(2))
    match = _STAGE1.search(stem)
    if match:
        return 1, int(match.group(1)), None
    raise ValueError("cannot parse backbone identifier from %s" % value)


def canonical_backbone_id(value):
    stage, parent_index, sample_index = parse_backbone_id(value)
    if stage == 1:
        return stage1_backbone_id(parent_index)
    return stage2_backbone_id(parent_index, sample_index)
