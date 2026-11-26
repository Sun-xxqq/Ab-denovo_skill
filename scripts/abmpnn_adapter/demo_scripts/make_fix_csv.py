#!/usr/bin/env python
import csv
import glob
import os
import sys


def row(path, max_chains=None):
    chains, positions, seen = [], {}, set()
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            chain = line[21:22].strip()
            if not chain:
                continue
            if chain not in chains:
                chains.append(chain)
            try:
                residue, bfactor = int(line[22:26]), float(line[60:66])
            except ValueError:
                continue
            key = (chain, residue)
            if abs(bfactor - 4.0) < 0.001 and key not in seen:
                positions.setdefault(chain, []).append(residue)
                seen.add(key)
    if max_chains is not None:
        chains = chains[:max_chains]
    motif = "-".join(" ".join(str(value) for value in positions.get(chain, [])) for chain in chains)
    return [os.path.splitext(os.path.basename(path))[0], motif]


def main(directory):
    normalized = os.path.abspath(directory).replace("\\", "/")
    # PPIFlow's Stage 2 fix-position merger currently understands the two
    # antibody segments only.  Stage 1 remains fully chain-aware (A/B/C).
    max_chains = 2 if "/stage2/partial_output/" in normalized + "/" else None
    values = [row(path, max_chains=max_chains) for path in sorted(glob.glob(os.path.join(directory, "*.pdb")))]
    output = os.path.join(directory, "fixed_positions.csv")
    with open(output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pdb_name", "motif_index"])
        writer.writerows(values)
    print("Saved chain-aware fixed positions to %s (max_chains=%s)" % (output, max_chains))


if __name__ == "__main__":
    main(sys.argv[1])
