#!/usr/bin/env python3
"""Legacy command alias; new callers should use parallel_fastrelax.py."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("parallel_fastrelax.py")), run_name="__main__")
