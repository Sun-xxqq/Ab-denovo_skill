"""Preserve full-atom fixed residues across backbone and side-chain stages."""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

import numpy as np


_ATOM_PREFIXES = ("ATOM  ", "HETATM")
_POSITION = re.compile(r"^([A-Za-z])(-?\d+)([A-Za-z]?)$")
_EXPECTED_HEAVY_ATOMS = {
    "ALA": {"N", "CA", "C", "O", "CB"},
    "ARG": {"N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"},
    "ASN": {"N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"},
    "ASP": {"N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"},
    "CYS": {"N", "CA", "C", "O", "CB", "SG"},
    "GLN": {"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"},
    "GLU": {"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"},
    "GLY": {"N", "CA", "C", "O"},
    "HIS": {"N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"},
    "ILE": {"N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"},
    "LEU": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"},
    "LYS": {"N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"},
    "MET": {"N", "CA", "C", "O", "CB", "CG", "SD", "CE"},
    "PHE": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "PRO": {"N", "CA", "C", "O", "CB", "CG", "CD"},
    "SER": {"N", "CA", "C", "O", "CB", "OG"},
    "THR": {"N", "CA", "C", "O", "CB", "OG1", "CG2"},
    "TRP": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "TYR": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"},
    "VAL": {"N", "CA", "C", "O", "CB", "CG1", "CG2"},
}


def parse_fixed_positions(value):
    """Return ``(chain, residue number, insertion code)`` values from a spec."""
    positions = set()
    text = str(value or "").strip()
    if not text or text.upper() == "NONE":
        return positions
    for token in (item.strip() for item in text.split(",")):
        if not token:
            continue
        match = _POSITION.fullmatch(token)
        if not match:
            raise ValueError("invalid fixed residue position: %r" % token)
        positions.add(
            (match.group(1), int(match.group(2)), match.group(3).strip())
        )
    return positions


def _atom_record(line):
    if not line.startswith(_ATOM_PREFIXES) or len(line) < 54:
        return None
    try:
        return {
            "residue": (line[21].strip(), int(line[22:26]), line[26].strip()),
            "atom": line[12:16].strip(),
            "altloc": line[16].strip(),
            "coordinate": np.asarray(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=float,
            ),
        }
    except (TypeError, ValueError):
        return None


def _read_atoms(path):
    atoms = {}
    residue_lines = {}
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines(
        keepends=True
    )
    for line in lines:
        record = _atom_record(line)
        if record is None:
            continue
        key = (
            *record["residue"],
            record["atom"],
            record["altloc"],
        )
        atoms[key] = record["coordinate"]
        residue_lines.setdefault(record["residue"], []).append(line)
    return lines, atoms, residue_lines


def _alignment_keys(reference, mobile, alignment_chains=None):
    chains = set(alignment_chains or [])
    common = set(reference).intersection(mobile)
    keys = [
        key
        for key in common
        if key[3] == "CA"
        and key[4] in ("", "A")
        and (not chains or key[0] in chains)
    ]
    if len(keys) < 3 and chains:
        keys = [
            key
            for key in common
            if key[3] == "CA" and key[4] in ("", "A")
        ]
    if len(keys) < 3:
        raise RuntimeError("fewer than three common CA atoms for rigid alignment")
    return sorted(keys)


def _rigid_transform(reference, mobile, keys):
    moving = np.asarray([mobile[key] for key in keys], dtype=float)
    target = np.asarray([reference[key] for key in keys], dtype=float)
    moving_center = moving.mean(axis=0)
    target_center = target.mean(axis=0)
    left, _, right_t = np.linalg.svd(
        (moving - moving_center).T @ (target - target_center)
    )
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_t
    aligned = (moving - moving_center) @ rotation + target_center
    rmsd = math.sqrt(float(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    return rotation, moving_center, target_center, rmsd


def _transform_line(line, rotation, moving_center, target_center):
    record = _atom_record(line)
    if record is None:
        return line
    coordinate = (
        (record["coordinate"] - moving_center) @ rotation + target_center
    )
    return "%s%8.3f%8.3f%8.3f%s" % (
        line[:30],
        coordinate[0],
        coordinate[1],
        coordinate[2],
        line[54:],
    )


def _renumber(lines):
    output = []
    serial = 1
    for line in lines:
        if line.startswith(_ATOM_PREFIXES) or line.startswith("TER"):
            if len(line) < 11:
                line = line.rstrip("\n").ljust(11) + "\n"
            line = "%s%5d%s" % (line[:6], serial, line[11:])
            serial += 1
        output.append(line)
    return output


def fixed_atom_inventory(path, fixed_positions):
    """Return atom names present for every requested residue."""
    positions = parse_fixed_positions(fixed_positions)
    _, _, residue_lines = _read_atoms(path)
    result = {}
    for residue in sorted(positions):
        result["%s%s%s" % residue] = sorted(
            {
                _atom_record(line)["atom"]
                for line in residue_lines.get(residue, [])
                if _atom_record(line) is not None
            }
        )
    return result


def validate_full_atom_fixed_residues(path, fixed_positions):
    """Fail when a requested fixed residue lacks canonical heavy atoms."""
    positions = parse_fixed_positions(fixed_positions)
    _, _, residue_lines = _read_atoms(path)
    result = {}
    for residue in sorted(positions):
        lines = residue_lines.get(residue, [])
        if not lines:
            raise RuntimeError(
                "fixed residue %s%s%s is absent from parent %s"
                % (*residue, path)
            )
        residue_name = lines[0][17:20].strip().upper()
        canonical_name = "HIS" if residue_name in {"HID", "HIE", "HIP"} else residue_name
        expected = _EXPECTED_HEAVY_ATOMS.get(canonical_name)
        if expected is None:
            raise RuntimeError(
                "unsupported fixed residue %s at %s%s%s in %s"
                % (residue_name, *residue, path)
            )
        observed = {
            _atom_record(line)["atom"]
            for line in lines
            if _atom_record(line) is not None
            and not _atom_record(line)["atom"].startswith("H")
        }
        missing = sorted(expected - observed)
        if missing:
            raise RuntimeError(
                "parent %s is not full atom at %s%s%s %s; missing %s"
                % (path, *residue, residue_name, ",".join(missing))
            )
        result["%s%s%s" % residue] = {
            "residue_name": residue_name,
            "heavy_atoms": sorted(observed),
        }
    return result


def restore_full_atom_fixed_residues(
    reference_pdb,
    mobile_pdb,
    fixed_positions,
    alignment_chains=None,
):
    """Align ``mobile_pdb`` to the parent and restore fixed residues verbatim.

    The entire mobile structure is first moved into the reference coordinate
    frame. Every ATOM record belonging to an explicitly fixed residue is then
    replaced by the reference record, including all available side-chain atoms.
    """
    reference_pdb = Path(reference_pdb).resolve()
    mobile_pdb = Path(mobile_pdb).resolve()
    positions = parse_fixed_positions(fixed_positions)
    if not positions:
        return {
            "reference_pdb": str(reference_pdb),
            "output_pdb": str(mobile_pdb),
            "fixed_positions": [],
            "fixed_atom_count": 0,
            "alignment_rmsd": None,
        }

    reference_lines, reference_atoms, reference_residues = _read_atoms(
        reference_pdb
    )
    mobile_lines, mobile_atoms, _ = _read_atoms(mobile_pdb)
    missing = sorted(position for position in positions if position not in reference_residues)
    if missing:
        raise RuntimeError(
            "fixed residues missing from full-atom parent %s: %s"
            % (reference_pdb, missing)
        )

    keys = _alignment_keys(reference_atoms, mobile_atoms, alignment_chains)
    rotation, moving_center, target_center, alignment_rmsd = _rigid_transform(
        reference_atoms, mobile_atoms, keys
    )

    restored = []
    emitted = set()
    observed = set()
    for line in mobile_lines:
        record = _atom_record(line)
        if record is None:
            restored.append(line)
            continue
        residue = record["residue"]
        if residue not in positions:
            restored.append(
                _transform_line(line, rotation, moving_center, target_center)
            )
            continue
        observed.add(residue)
        if residue not in emitted:
            restored.extend(reference_residues[residue])
            emitted.add(residue)

    absent_from_mobile = sorted(positions - observed)
    if absent_from_mobile:
        raise RuntimeError(
            "fixed residues missing from generated PDB %s: %s"
            % (mobile_pdb, absent_from_mobile)
        )

    temporary = mobile_pdb.with_name(mobile_pdb.name + ".full_atom_fix.tmp")
    temporary.write_text(
        "".join(_renumber(restored)), encoding="utf-8"
    )
    os.replace(temporary, mobile_pdb)

    inventory = fixed_atom_inventory(reference_pdb, fixed_positions)
    return {
        "reference_pdb": str(reference_pdb),
        "output_pdb": str(mobile_pdb),
        "fixed_positions": sorted(inventory),
        "fixed_atoms": inventory,
        "fixed_atom_count": sum(len(value) for value in inventory.values()),
        "alignment_atom_count": len(keys),
        "alignment_rmsd": alignment_rmsd,
    }
