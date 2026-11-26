"""Canonical FastRelax names with read compatibility for existing runs."""

from pathlib import Path


LEGACY_NAMES = {
    "fastrelax": ("rosettafix",),
    "fastrelax_output": ("rosettafix_output",),
    "fastrelax_raw_output": ("rosetta_fix_output",),
    "08_fastrelax": ("07_fastrelax", "08_rosettafix", "07_rosettafix"),
    "07_fastrelax": ("07_rosettafix",),
    "stage1.fastrelax.log": ("stage1.rosetta_fix.log",),
    "stage2.fastrelax.log": ("stage2.rosetta_fix.log",),
}


def fastrelax_path(path):
    """Prefer the canonical path; use an existing legacy path without rerunning."""
    path = Path(path)
    if path.exists():
        return path
    for name in LEGACY_NAMES.get(path.name, ()):
        legacy = path.with_name(name)
        if legacy.exists():
            return legacy
    return path


def canonical_stage_name(name):
    return name.replace("_rosetta_fix", "_fastrelax")


def fastrelax_stage(stages, name):
    canonical = canonical_stage_name(name)
    return stages.get(canonical, stages.get(canonical.replace("_fastrelax", "_rosetta_fix"), {}))
