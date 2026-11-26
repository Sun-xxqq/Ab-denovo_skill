#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Score a two-part protein interface with PyRosetta")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--jump", type=int, default=1)
    args = parser.parse_args()

    import pyrosetta
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

    pyrosetta.init("-mute all -ignore_unrecognized_res true")
    pose = pyrosetta.pose_from_file(args.input)
    mover = InterfaceAnalyzerMover(args.jump, False, pyrosetta.get_fa_scorefxn(), False, False, False)
    mover.apply(pose)
    data = mover.get_all_data()
    result = {
        "candidate_id": Path(args.input).stem,
        "structure": str(Path(args.input).resolve()),
        "interface_energy": float(data.dG[0]),
        "interface_delta_sasa": float(data.dSASA[0]),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "interface_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
