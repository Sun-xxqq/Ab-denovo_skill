"""Canonical, compact on-disk layout for AbMPNN sequence outputs."""

import os
import re
import shutil
from pathlib import Path

from .naming import canonical_backbone_id, parse_backbone_id, stage2_backbone_id

_BACKBONE_SUFFIX = re.compile(
    r"(backbone\.(?:0_\d+(?:_sample\d+)?|1_\d+_\d+))$", re.IGNORECASE
)


def _read_fasta(path):
    records = []
    header = None
    sequence_parts = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(sequence_parts)))
                header = line[1:]
                sequence_parts = []
            elif header is None:
                raise ValueError("FASTA sequence appears before its header: %s" % path)
            else:
                sequence_parts.append(line)
    if header is not None:
        records.append((header, "".join(sequence_parts)))
    if not records or any(not sequence for _, sequence in records):
        raise ValueError("empty or invalid FASTA: %s" % path)
    return records


def _backbone_name(path):
    stem = Path(path).stem
    match = _BACKBONE_SUFFIX.search(stem)
    if match:
        return canonical_backbone_id(match.group(1))
    parent_match = re.search(
        r"(backbone\.[A-Za-z0-9-]+_\d+)$",
        Path(path).parent.parent.name,
        re.IGNORECASE,
    )
    sample_match = re.fullmatch(r"sample(\d+)", stem, re.IGNORECASE)
    if parent_match and sample_match:
        _, parent_index, _ = parse_backbone_id(parent_match.group(1))
        return stage2_backbone_id(parent_index, int(sample_match.group(1)))
    return stem


def _sequence_name(backbone, index):
    if not str(backbone).lower().startswith("backbone"):
        raise ValueError("unexpected canonical backbone name: %s" % backbone)
    return "abmpnn%s_%d" % (str(backbone)[len("backbone") :], int(index))


def _migrate_organized_names(root):
    """Upgrade the earlier backbone-prefixed compact layout in place."""
    changed = False
    for backbone_dir in sorted(path for path in Path(root).iterdir() if path.is_dir()):
        for old in sorted(backbone_dir.glob(backbone_dir.name + "_*.fa")):
            match = re.fullmatch(re.escape(backbone_dir.name) + r"_(\d+)\.fa", old.name)
            if not match:
                continue
            new_name = _sequence_name(backbone_dir.name, int(match.group(1)))
            new = backbone_dir / (new_name + ".fa")
            if new.exists():
                raise FileExistsError("refusing to overwrite AbMPNN sequence: %s" % new)
            records = _read_fasta(old)
            if len(records) != 1:
                raise ValueError("organized FASTA must contain one sequence: %s" % old)
            new.write_text(">%s\n%s\n" % (new_name, records[0][1]), encoding="utf-8")
            old.unlink()
            changed = True
    return changed


def _validate_organized(root, expected_sequences):
    backbone_dirs = sorted(path for path in Path(root).iterdir() if path.is_dir())
    if not backbone_dirs:
        raise ValueError("no backbone directories found in %s" % root)
    sequence_count = 0
    for backbone_dir in backbone_dirs:
        def sequence_index(path):
            prefix = "abmpnn" + backbone_dir.name[len("backbone") :]
            match = re.fullmatch(re.escape(prefix) + r"_(\d+)\.fa", path.name)
            return int(match.group(1)) if match else float("inf")

        files = sorted(backbone_dir.glob("*.fa"), key=sequence_index)
        if expected_sequences is not None and len(files) != int(expected_sequences):
            raise ValueError(
                "%s contains %d sequences; expected %d"
                % (backbone_dir, len(files), int(expected_sequences))
            )
        for index, path in enumerate(files, 1):
            expected_name = _sequence_name(backbone_dir.name, index)
            if path.stem != expected_name:
                raise ValueError(
                    "unexpected sequence filename %s; expected %s.fa"
                    % (path.name, expected_name)
                )
            records = _read_fasta(path)
            if len(records) != 1 or records[0][0] != expected_name:
                raise ValueError("unexpected FASTA header in %s" % path)
        sequence_count += len(files)
    return {
        "backbone_count": len(backbone_dirs),
        "sequence_count": sequence_count,
        "sequences_per_backbone": expected_sequences,
    }


def organize_abmpnn_output(output_dir, expected_sequences=None):
    """Split AbMPNN's monolithic FASTAs into one compact directory per backbone.

    Native layout::

        abmpnn_output/seqs/<run>_backbone.0_0.fa

    Canonical layout::

        abmpnn_output/backbone.0_0/abmpnn.0_0_1.fa
        ...

    The native FASTA's first record is the input/template sequence.  Only the
    sampled design records are emitted.  Replacement is atomic at directory
    level and the original output remains untouched if validation fails.
    """
    root = Path(output_dir)
    if not root.is_dir():
        raise FileNotFoundError("AbMPNN output directory not found: %s" % root)

    native_seqs = root / "seqs"
    nested_sources = sorted(
        list(root.glob("*/seqs/*.fa")) + list(root.glob("*/seqs/*.fasta"))
    )
    if not native_seqs.is_dir() and not nested_sources:
        changed = _migrate_organized_names(root)
        result = _validate_organized(root, expected_sequences)
        result["changed"] = changed
        return result

    sources = nested_sources or sorted(
        list(native_seqs.glob("*.fa")) + list(native_seqs.glob("*.fasta"))
    )
    if not sources:
        raise ValueError("no AbMPNN FASTA files found in %s" % native_seqs)

    staging = root.with_name(root.name + ".organizing")
    legacy = root.with_name(root.name + ".legacy")
    if staging.exists() or legacy.exists():
        raise FileExistsError(
            "refusing to overwrite an earlier migration directory: %s or %s"
            % (staging, legacy)
        )

    try:
        staging.mkdir()
        seen_backbones = set()
        for source in sources:
            backbone = _backbone_name(source)
            if backbone in seen_backbones:
                raise ValueError("duplicate AbMPNN backbone name: %s" % backbone)
            seen_backbones.add(backbone)
            records = _read_fasta(source)
            designed = records[1:]
            if expected_sequences is not None and len(designed) != int(expected_sequences):
                raise ValueError(
                    "%s contains %d designed sequences; expected %d"
                    % (source, len(designed), int(expected_sequences))
                )
            if not designed:
                raise ValueError("no designed sequences found in %s" % source)
            backbone_dir = staging / backbone
            backbone_dir.mkdir()
            for index, (_, sequence) in enumerate(designed, 1):
                sequence_name = _sequence_name(backbone, index)
                (backbone_dir / (sequence_name + ".fa")).write_text(
                    ">%s\n%s\n" % (sequence_name, sequence), encoding="utf-8"
                )

        result = _validate_organized(staging, expected_sequences)
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
