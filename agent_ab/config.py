from pathlib import Path, PurePosixPath

from .util import project_root, read_json


def load_site_config(path=None):
    path = Path(path) if path else project_root() / "configs" / "site.json"
    config = read_json(path)
    config["_config_path"] = str(path.resolve())
    return config


def tool(config, name):
    try:
        return config["tools"][name]
    except KeyError:
        raise KeyError("tool is not configured: %s" % name)


def env_python(config, env_name):
    env_path = PurePosixPath(str(env_name))
    if env_path.is_absolute():
        if env_path.name == "python":
            return str(env_path)
        return str(env_path / "bin" / "python")
    if env_name == "base":
        return str(PurePosixPath(config["conda_root"]) / "bin" / "python")
    return str(PurePosixPath(config["conda_root"]) / "envs" / env_name / "bin" / "python")
