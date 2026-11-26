#!/usr/bin/env python3
"""Create a new Stage 1 spec using the IL-13 example bundled with PPIFlow."""

import argparse
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=PROJECT / "configs/site.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    site = json.loads(args.site.read_text(encoding="utf-8"))
    upstream_examples = Path(site["tools"]["ppiflow"]["path"]) / "example/target_and_framework_pdb"
    target = upstream_examples / "1IJZ_IL13.pdb"
    framework = upstream_examples / "7eow_nanobody_framework.pdb"
    for structure in (target, framework):
        if not structure.is_file():
            parser.error(f"PPIFlow example file is missing: {structure}")

    spec = {
        "schema_version": 1,
        "name": "upstream-il13-stage1",
        "design_type": "nanobody",
        "target": {"structure": str(target.resolve()), "chain": "C", "hotspots": ["C10", "C13", "C14", "C100", "C106", "C107"]},
        "framework": {"structure": str(framework.resolve()), "heavy_chain": "A"},
        "generation": {"profile": "stage1", "samples": 4},
    }
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(spec, output, indent=2)
        output.write("\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
