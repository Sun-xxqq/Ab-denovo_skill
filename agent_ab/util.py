import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def project_root():
    override = os.environ.get("AGENT_AB_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def slug(value):
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip()).strip("-._")
    return value.lower()[:48] or "design"


def new_run_id(name):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return "%s_%s" % (stamp, slug(name))


def disk_free_gb(path):
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(str(probe)).free / (1024 ** 3)


def select_gpus(limit=1):
    """Return up to *limit* GPU ids ordered by memory use and utilization."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        choices = []
        for line in completed.stdout.splitlines():
            index, memory, utilization = [int(item.strip()) for item in line.split(",")]
            choices.append((memory, utilization, index))
        if completed.returncode == 0 and choices:
            return [item[2] for item in sorted(choices)[: max(1, int(limit))]]
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return [0]


def select_gpu():
    return select_gpus(limit=1)[0]


def _yaml_scalar(value):
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def yaml_text(value, indent=0):
    """Serialize the JSON-compatible subset used by PPIFlow configs."""
    prefix = " " * indent
    lines = []
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append("%s%s:" % (prefix, key))
                lines.extend(yaml_text(item, indent + 2))
            else:
                lines.append("%s%s: %s" % (prefix, key, _yaml_scalar(item)))
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(prefix + "-")
                lines.extend(yaml_text(item, indent + 2))
            else:
                lines.append("%s- %s" % (prefix, _yaml_scalar(item)))
    else:
        lines.append(prefix + _yaml_scalar(value))
    return lines


def write_yaml(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(yaml_text(value)) + "\n", encoding="utf-8")
