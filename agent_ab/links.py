import os
from pathlib import Path

from .util import project_root


def _ensure_link(link, target):
    link = Path(link)
    target = Path(target)
    if not target.exists():
        return {"link": str(link), "target": str(target), "state": "target_missing"}
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return {"link": str(link), "target": str(target), "state": "unchanged"}
        return {"link": str(link), "target": str(target), "state": "conflict"}
    if link.exists():
        return {"link": str(link), "target": str(target), "state": "conflict"}
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(target), str(link), target_is_directory=target.is_dir())
    return {"link": str(link), "target": str(target), "state": "created"}


def initialize_links(config, include_home_compat=False):
    root = project_root()
    results = []
    for name, item in sorted(config["tools"].items()):
        if item.get("path"):
            results.append(_ensure_link(root / "links" / "software" / name, item["path"]))
        env = item.get("env")
        if env:
            env_target = Path(config["conda_root"]) if env == "base" else Path(config["conda_root"]) / "envs" / env
            results.append(
                _ensure_link(root / "links" / "envs" / env, env_target)
            )
    for name, target in sorted(config.get("large_assets", {}).items()):
        results.append(_ensure_link(root / "links" / "assets" / name, target))
    if include_home_compat:
        for name, target in sorted(config.get("home_compat_links", {}).items()):
            results.append(_ensure_link(Path.home() / name, target))
    return results
