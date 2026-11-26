#!/usr/bin/env python
import os
from pathlib import Path
import runpy
import sys


UPSTREAM = str(Path(os.environ.get("AGENT_AB_PPIFLOW_ROOT") or Path(__file__).resolve().parents[2] / "vendor/PPIFlow") / "tools/AbMPNN")
if UPSTREAM not in sys.path:
    sys.path.insert(0, UPSTREAM)
runpy.run_path(UPSTREAM + "/protein_mpnn_run.py", run_name="__main__")
