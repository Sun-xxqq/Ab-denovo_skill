"""Canonical, compact on-disk layout for completed FlowPacker outputs."""

import hashlib
import os
import re
import shutil
from pathlib import Path


_DESIGN_SUFFIX = re.compile(
    r"((?:backbone|flowpacker)\.(?:0_\d+(?:_sample\d+)?|1_\d+_\d+))_(\d+)$",
    re.IGNORECASE,
)


def _design_identity(path):
    match = _DESIGN_SUFFIX.search(Path(path).stem)
    if not match:
        raise ValueError("cannot derive canonical FlowPacker name from %s" % path)
    backbone = match.group(1)
    if backbone.lower().startswith("flowpacker"):
        backbone = "backbone" + backbone[len("flowpacker") :]
    return backbone, int(match.group(2))


def _design_name(backbone, index):
    if not str(backbone).lower().startswith("backbone"):
        raise ValueError("unexpected canonical backbone name: %s" % backbone)
    return "flowpacker%s_%d" % (str(backbone)[len("backbone") :], int(index))


def _migrate_organized_names(root):
    """Upgrade the earlier backbone-prefixed compact layout in place."""
    changed = False
    for backbone_dir in sorted(path for path in Path(root).iterdir() if path.is_dir()):
        for old in sorted(backbone_dir.glob(backbone_dir.name + "_*.pdb")):
            match = re.fullmatch(re.escape(backbone_dir.name) + r"_(\d+)\.pdb", old.name)
            if not match:
                continue
            new = backbone_dir / (_design_name(backbone_dir.name, int(match.group(1))) + ".pdb")
            if new.exists():
                raise FileExistsError("refusing to overwrite FlowPacker design: %s" % new)
            old.replace(new)
            changed = True
    return changed


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_organized(root, expected_backbones=None, expected_sequences=None):
    root = Path(root)
    unexpected = [path for path in root.iterdir() if not path.is_dir()]
    if unexpected:
        raise ValueError(
            "canonical FlowPacker root contains unexpected files: %s"
            % ", ".join(path.name for path in unexpected)
        )
    backbone_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not backbone_dirs:
        raise ValueError("no FlowPacker backbone directories found in %s" % root)
    if expected_backbones is not None and len(backbone_dirs) != int(expected_backbones):
        raise ValueError(
            "%s contains %d backbones; expected %d"
            % (root, len(backbone_dirs), int(expected_backbones))
        )

    design_count = 0
    for backbone_dir in backbone_dirs:
        def sequence_index(path):
            prefix = "flowpacker" + backbone_dir.name[len("backbone") :]
            match = re.fullmatch(re.escape(prefix) + r"_(\d+)\.pdb", path.name)
            return int(match.group(1)) if match else float("inf")

        files = sorted(backbone_dir.glob("*.pdb"), key=sequence_index)
        if expected_sequences is not None and len(files) != int(expected_sequences):
            raise ValueError(
                "%s contains %d designs; expected %d"
                % (backbone_dir, len(files), int(expected_sequences))
            )
        for index, path in enumerate(files, 1):
            expected_name = _design_name(backbone_dir.name, index) + ".pdb"
            if path.name != expected_name:
                raise ValueError(
                    "unexpected FlowPacker filename %s; expected %s"
                    % (path.name, expected_name)
                )
            if path.stat().st_size == 0:
                raise ValueError("empty FlowPacker PDB: %s" % path)
        design_count += len(files)
    return {
        "backbone_count": len(backbone_dirs),
        "design_count": design_count,
        "designs_per_backbone": expected_sequences,
    }


def organize_flowpacker_output(
    output_dir, expected_backbones=None, expected_sequences=None
):
    """Keep only final packed PDBs, grouped to match canonical AbMPNN output.

    Native layout::

        flowpacker_output/after_pdbs_batch/...
        flowpacker_output/after_pdbs/...
        flowpacker_output/run_1/<run>_backbone.0_0_1.pdb
        flowpacker_output/output_dict.pth

    Canonical layout::

        flowpacker_output/backbone.0_0/flowpacker.0_0_1.pdb
        ...

    The native directory is replaced only after every final PDB has been linked
    or copied into a staging directory, hash-checked, and structurally validated.
    Temporary inputs, the duplicate flat copy, ``run_1``, and internal metrics
    are deliberately omitted from the canonical result.
    """
    root = Path(output_dir)
    if not root.is_dir():
        raise FileNotFoundError("FlowPacker output directory not found: %s" % root)

    native = root / "run_1"
    if not native.is_dir():
        changed = _migrate_organized_names(root)
        result = _validate_organized(root, expected_backbones, expected_sequences)
        result["changed"] = changed
        return result

    sources = sorted(native.glob("*.pdb"))
    if not sources:
        raise ValueError("no final FlowPacker PDBs found in %s" % native)

    staging = root.with_name(root.name + ".organizing")
    legacy = root.with_name(root.name + ".legacy")
    if staging.exists() or legacy.exists():
        raise FileExistsError(
            "refusing to overwrite an earlier migration directory: %s or %s"
            % (staging, legacy)
        )

    try:
        staging.mkdir()
        seen = set()
        for source in sources:
            backbone, sequence_index = _design_identity(source)
            identity = (backbone, sequence_index)
            if identity in seen:
                raise ValueError("duplicate FlowPacker design: %s_%d" % identity)
            seen.add(identity)
            destination_dir = staging / backbone
            destination_dir.mkdir(exist_ok=True)
            destination = destination_dir / (_design_name(*identity) + ".pdb")
            try:
                os.link(str(source), str(destination))
            except OSError:
                shutil.copy2(str(source), str(destination))
            if _sha256(source) != _sha256(destination):
                raise RuntimeError("FlowPacker PDB hash changed during migration: %s" % source)

        result = _validate_organized(
            staging,
            expected_backbones=expected_backbones,
            expected_sequences=expected_sequences,
        )
        os.replace(str(root), str(legacy))
        try:
            os.replace(str(staging), str(root))
        except Exception:
            os.replace(str(legacy), str(root))
            raise
        shutil.rmtree(str(legacy))
        result["changed"] = True
        return result
    except Exception:
        if staging.exists():
            shutil.rmtree(str(staging))
        raise
