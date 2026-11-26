"""Grouped, compact user-facing layout for generated backbone structures."""

import os
import shutil
from pathlib import Path


def organize_backbone_output(native_dir, output_dir):
    """Publish one directory per backbone without duplicating structure data."""
    native = Path(native_dir)
    output = Path(output_dir)
    if not native.is_dir():
        raise FileNotFoundError("backbone source directory not found: %s" % native)
    sources = sorted(native.glob("backbone.0_*.pdb"))
    if not sources:
        sources = sorted(native.glob("*/backbone.0_*.pdb"))
    unexpected = [
        path
        for path in sources
        if path.parent != native and path.stem != path.parent.name
    ]
    if unexpected:
        raise ValueError("unexpected grouped backbone path: %s" % unexpected[0])
    if not sources:
        raise ValueError("no backbone PDB files found in %s" % native)

    staging = output.with_name(output.name + ".organizing")
    legacy = output.with_name(output.name + ".legacy")
    if staging.exists() or legacy.exists():
        raise FileExistsError(
            "refusing to overwrite an earlier backbone migration: %s or %s"
            % (staging, legacy)
        )
    staging.mkdir(parents=True)
    try:
        for source in sources:
            group = staging / source.stem
            group.mkdir()
            destination = group / source.name
            try:
                os.link(str(source), str(destination))
            except OSError:
                shutil.copy2(source, destination)
        if output.exists():
            os.replace(str(output), str(legacy))
        try:
            os.replace(str(staging), str(output))
        except Exception:
            if legacy.exists():
                os.replace(str(legacy), str(output))
            raise
        if legacy.exists():
            shutil.rmtree(str(legacy))
    except Exception:
        if staging.exists():
            shutil.rmtree(str(staging))
        raise
    return {
        "backbone_count": len(sources),
        "output": str(output),
        "changed": True,
    }
