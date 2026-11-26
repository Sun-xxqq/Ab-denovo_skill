#!/usr/bin/env python3
"""Mark an orphaned Agent-Ab Stage 2 run as safely retryable."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent_ab.recovery import RecoveryState
from agent_ab.workflow import _write_status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    detail = (
        "stale running reconciled after Rosetta completed with no active "
        "pipeline process"
    )
    RecoveryState(run_dir).update(
        "ppiflow_iterative_redesign", "failed", detail=detail
    )
    _write_status(run_dir, "failed", detail=detail)
    print(f"reconciled: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
