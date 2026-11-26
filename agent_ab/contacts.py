from collections import Counter
from pathlib import Path


def pdb_chains(path):
    values = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith(("ATOM  ", "HETATM")):
                chain = line[21:22].strip()
                if chain and chain not in values:
                    values.append(chain)
    return values


def infer_complex_roles(path, design_type):
    """Infer conventional antibody-complex chain roles without guessing arbitrary chains."""
    chains = pdb_chains(path)
    result = {"chains": chains}
    if design_type == "antibody" and "H" in chains and "L" in chains:
        antigens = [chain for chain in chains if chain not in ("H", "L")]
        if len(antigens) == 1:
            result.update({"target_chain": antigens[0], "heavy_chain": "H", "light_chain": "L"})
    elif design_type == "nanobody" and "H" in chains and "L" not in chains:
        antigens = [chain for chain in chains if chain != "H"]
        if len(antigens) == 1:
            result.update({"target_chain": antigens[0], "heavy_chain": "H"})
    return result


def interface_hotspots(path, target_chain, partner_chains, cutoff=5.0, limit=6):
    """Return target residues with the most heavy-atom contacts to partner chains."""
    target, partners = [], []
    cutoff_squared = float(cutoff) ** 2
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            element = line[76:78].strip().upper()
            if element == "H" or line[12:16].strip().startswith("H"):
                continue
            chain = line[21:22].strip()
            if chain != target_chain and chain not in partner_chains:
                continue
            try:
                coordinate = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            residue = (chain, line[22:26].strip() + line[26:27].strip(), line[17:20].strip())
            (target if chain == target_chain else partners).append((residue, coordinate))
    counts = Counter()
    for residue, left in target:
        for _, right in partners:
            if sum((left[index] - right[index]) ** 2 for index in range(3)) <= cutoff_squared:
                counts[residue] += 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0][1]))[: int(limit)]
    return ["%s%s" % (residue[0], residue[1]) for residue, _ in ranked]


def split_complex_pdb(path, target_chain, partner_chains, output_dir, variable_only=False):
    """Materialize target-only and framework-only PDBs for tools requiring separation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / ("target_%s.pdb" % target_chain)
    framework_path = output_dir / ("framework_%s.pdb" % "".join(partner_chains))
    target_lines, framework_lines = [], []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM", "ANISOU")):
                continue
            chain = line[21:22].strip()
            if chain == target_chain:
                target_lines.append(line.rstrip("\n"))
            elif chain in partner_chains:
                keep = True
                if variable_only:
                    try:
                        residue_number = int(line[22:26].strip())
                        cutoff = 128 if chain == "H" else 120 if chain == "L" else 130
                        cdr_ranges = {
                            "H": ((26, 33), (51, 58), (97, 108)),
                            "L": ((27, 37), (55, 57), (94, 102)),
                        }
                        in_cdr = any(
                            start <= residue_number <= end
                            for start, end in cdr_ranges.get(chain, ())
                        )
                        keep = residue_number <= cutoff and not in_cdr
                    except ValueError:
                        keep = True
                if keep:
                    framework_lines.append(line.rstrip("\n"))
    if not target_lines:
        raise ValueError("no atoms found for inferred target chain %s" % target_chain)
    if not framework_lines:
        raise ValueError("no atoms found for inferred framework chains %s" % ",".join(partner_chains))
    target_path.write_text("\n".join(target_lines + ["TER", "END", ""]), encoding="utf-8")
    framework_path.write_text("\n".join(framework_lines + ["TER", "END", ""]), encoding="utf-8")
    return str(target_path), str(framework_path)


def remap_framework_chains(path, chain_mapping, output_path):
    """Copy a validated framework while assigning chain IDs unique to the target."""
    lines = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM", "ANISOU")):
                continue
            source = line[21:22].strip()
            if source not in chain_mapping:
                continue
            lines.append(line[:21] + chain_mapping[source] + line[22:].rstrip("\n"))
    if not lines:
        raise ValueError("default framework contains none of the configured chains")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines + ["TER", "END", ""]), encoding="utf-8")
    return str(output_path)
