import os
import subprocess
from pathlib import Path

from .config import env_python
from .util import disk_free_gb, project_root


def _check(name, path, required=True, executable=False):
    path = Path(path)
    ok = path.exists() and (not executable or os.access(str(path), os.X_OK))
    return {
        "name": name,
        "path": str(path),
        "required": bool(required),
        "ok": bool(ok),
        "detail": "ok" if ok else ("missing or not executable" if executable else "missing"),
    }


def run_doctor(config):
    checks = []
    for name, item in sorted(config["tools"].items()):
        required = item.get("required", False)
        if item.get("path"):
            checks.append(_check("%s.code" % name, item["path"], required=required))
        if item.get("env"):
            checks.append(
                _check(
                    "%s.python" % name,
                    env_python(config, item["env"]),
                    required=required,
                    executable=True,
                )
            )
        for index, path in enumerate(item.get("required_paths", []), start=1):
            checks.append(_check("%s.asset.%s" % (name, index), path, required=required))

    for name, path in sorted(config.get("large_assets", {}).items()):
        checks.append(_check("asset.%s" % name, path, required=False))

    runs_root = Path(config["storage"]["runs_root"])
    free_gb = disk_free_gb(runs_root)
    threshold = float(config["storage"]["min_free_gb_plan"])
    checks.append(
        {
            "name": "storage.runs",
            "path": str(runs_root),
            "required": True,
            "ok": free_gb >= threshold,
            "detail": "%.1f GiB free; plan threshold %.1f GiB" % (free_gb, threshold),
        }
    )
    gpu_lines = []
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if completed.returncode == 0:
            gpu_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        pass
    checks.append(
        {
            "name": "gpu",
            "path": "nvidia-smi",
            "required": True,
            "ok": bool(gpu_lines),
            "detail": "; ".join(gpu_lines) if gpu_lines else "no GPU detected",
        }
    )
    blocked = [item for item in checks if item["required"] and not item["ok"]]
    return {
        "project_root": str(project_root()),
        "ready": not blocked,
        "checks": checks,
        "required_failures": len(blocked),
    }

