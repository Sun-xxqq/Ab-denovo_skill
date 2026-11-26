#!/usr/bin/env python3
"""Inventory Agent-Ab installations and configure explicitly supplied dependencies."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile


CATALOG = Path(__file__).resolve().parents[1] / "references/tools.json"
AUXILIARY = {"proteinmpnn_soluble", "foldx"}
GPU_INSTALL_GUIDE = "https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def probe(argv, cwd=None, timeout=20):
    try:
        result = subprocess.run(
            [str(arg) for arg in argv], cwd=cwd, timeout=timeout,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            text=True, encoding="utf-8", errors="replace",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, str(exc)
    return result.returncode, result.stdout.strip()


def path_value(value):
    return Path(os.path.expandvars(value)).expanduser().absolute() if value else None


def is_executable(path):
    return bool(path and path.is_file() and os.access(path, os.X_OK))


def directory_index(roots, depth):
    index = {}
    for root in dict.fromkeys(roots):
        if not root or not root.is_dir():
            continue
        for base, dirs, _ in os.walk(root, followlinks=False):
            directory = Path(base)
            index.setdefault(directory.name.casefold(), []).append(directory)
            dirs[:] = [
                name for name in sorted(dirs)
                if not name.startswith(".") and name not in {"node_modules", "__pycache__", "site-packages"}
            ] if len(directory.relative_to(root).parts) < depth else []
            for name in dirs:
                if (directory / name).is_symlink():
                    index.setdefault(name.casefold(), []).append(directory / name)
    return index


def conda_roots(project, site, search):
    active_conda = path_value(os.environ.get("CONDA_EXE"))
    return list(dict.fromkeys(filter(None, [
        path_value(site.get("conda_root")),
        active_conda.parent.parent if active_conda else None,
        *search, project / "links/envs/base",
        *(Path.home() / name for name in ("miniforge3", "miniconda3", "anaconda3", "mambaforge")),
        *(Path("/opt") / name for name in ("conda", "miniforge3", "miniconda3", "anaconda3")),
    ])))


def python_for(env, root, conda, index):
    if not env:
        return None
    candidates = []
    if root:
        candidates.extend((root / "env/bin/python", root / ".venv/bin/python"))
    marker = Path(env).expanduser()
    if marker.is_absolute():
        candidates.insert(0, marker if marker.name == "python" else marker / "bin/python")
    else:
        if env == "base":
            candidates.extend(prefix / "bin/python" for prefix in conda)
            candidates.append(Path(sys.executable))
        else:
            candidates.extend(prefix / "envs" / env / "bin/python" for prefix in conda)
            candidates.extend(directory / "bin/python" for directory in index.get(env.casefold(), []))
    return next((p for p in candidates if is_executable(p)), None)


def asset_issue(path):
    if path.is_file():
        return None if path.stat().st_size else "空文件"
    if path.is_dir():
        try:
            return None if next(path.iterdir(), None) else "空目录"
        except PermissionError:
            return "无法读取"
    return "缺失"


def installed(tool, hinted, site_tool, conda, index, parent):
    env = hinted.get("env") or site_tool.get("env") or tool.get("env")
    default_env = tool.get("env")
    python = python_for(env, None, conda, index) or python_for(default_env, None, conda, index)
    roots = [path_value(hinted.get("path")), path_value(site_tool.get("path"))]
    if tool.get("parent") and parent and parent["path"]:
        roots.append(Path(parent["path"]) / tool["subdir"])
    if tool.get("env_root") and python:
        roots.append(python.parent.parent)
    roots.extend(directory for name in tool.get("names", []) for directory in index.get(name.casefold(), []))
    if tool.get("binary") and "/" not in tool.get("entry", ""):
        binary = path_value(shutil.which(tool["entry"]))
        if binary:
            roots.append(binary.parent)

    old_root = path_value(site_tool.get("path"))
    old_python = None
    if site_tool.get("env"):
        old_python = python_for(site_tool["env"], None, conda, index)
    candidates = []
    for root in dict.fromkeys(filter(None, roots)):
        if not root.is_dir():
            continue
        entry = root / tool["entry"] if tool.get("entry") else None
        local_python = python_for(env, root, conda, index) or python
        assets = [root / item for item in tool.get("assets", [])]
        absent = []
        asset_overrides = []
        for key in tool.get("asset_keys", []):
            asset = path_value(hinted.get(key) or site_tool.get(key))
            if asset:
                assets.append(asset)
                configured = path_value(site_tool.get(key))
                if configured and hinted.get(key):
                    asset_overrides.append((configured, asset))
            else:
                absent.append(key)
        asset_overrides.sort(key=lambda mapping: -len(mapping[0].parts))
        for item in site_tool.get("required_paths", []):
            asset = path_value(item)
            for configured, selected in asset_overrides:
                if asset.is_relative_to(configured):
                    asset = selected / asset.relative_to(configured)
                    break
            else:
                if old_root and asset.is_relative_to(old_root):
                    asset = root / asset.relative_to(old_root)
                if old_python and local_python and asset.is_relative_to(old_python.parent.parent):
                    asset = local_python.parent.parent / asset.relative_to(old_python.parent.parent)
            assets.append(asset)
        assets = list(dict.fromkeys(assets))
        asset_info = [{"path": str(asset), "issue": asset_issue(asset)} for asset in assets]
        absent.extend(f"{asset['path']}（{asset['issue']}）" for asset in asset_info if asset["issue"])
        if entry and not (is_executable(entry) if tool.get("binary") else entry.is_file()):
            absent.append(str(entry))
        if env and not local_python:
            absent.append("Python 环境 " + env)
        candidates.append((root, entry, local_python, asset_info, absent))

    root, entry, python, assets, absent = min(
        candidates, key=lambda item: len(item[-1]),
        default=(None, None, python, [], ["安装目录"]),
    )
    if root is None and env and python is None:
        absent.append("Python 环境 " + env)
    launcher = root / tool["launcher"] if root and tool.get("launcher") else None
    if launcher and not is_executable(launcher):
        launcher = None
    return {
        "label": tool["label"], "role": tool["role"],
        "path": str(root) if root else None,
        "real_path": str(root.resolve()) if root else None,
        "python": str(python) if python else None,
        "entry": str(entry) if entry else None,
        "launcher": str(launcher) if launcher else None,
        "assets": assets,
        "model": site_tool.get("model") or tool.get("model"),
        "weight_note": tool.get("weight_note"),
        "source": tool.get("source"),
        "install_guide": tool.get("install_guide"),
        "configured_required": site_tool.get("required", False),
        "missing": absent,
        "verification": "缺项" if absent else "路径与所列资产存在",
    }


def verify(tool, item):
    if not item["path"]:
        return
    python = item["python"]
    root = Path(item["path"])
    source_commit = root / "SOURCE_COMMIT"
    if source_commit.is_file() and source_commit.stat().st_size < 256:
        item["source_commit"] = source_commit.read_text(errors="replace").strip()
    elif (root / ".git").exists():
        code, output = probe(["git", "-C", root, "rev-parse", "HEAD"])
        if code == 0:
            item["source_commit"] = output
    if python:
        code, output = probe([python, "-V"])
        if code == 0:
            item["python_version"] = output
        else:
            item["verification"] = "Python 启动失败"
            item["missing"].append("可用 Python 解释器")
        if tool.get("package"):
            query = "from importlib.metadata import version; import sys; print(version(sys.argv[1]))"
            code, output = probe([python, "-c", query, tool["package"]])
            if code == 0:
                item["version"] = output
    if item["missing"]:
        return

    if tool.get("import"):
        argv = [python, "-c", "import " + tool["import"]]
    elif tool.get("verify"):
        if item["launcher"]:
            argv = [item["launcher"], *tool["verify"]]
        else:
            argv = [item["entry"], *tool["verify"]]
            if not tool.get("binary"):
                argv.insert(0, python)
    else:
        return
    with tempfile.TemporaryDirectory(prefix="agent-ab-probe-") as tmp:
        code, output = probe(argv, cwd=tmp)
    version = re.search(r"FoldX\s+([\d.]+)", output, re.IGNORECASE)
    if version:
        item["version"] = "FoldX " + version.group(1)
    flag = tool.get("flag")
    if flag and item["launcher"]:
        launcher_text = Path(item["launcher"]).read_text(errors="replace")
        item["launcher_flag"] = flag if flag in launcher_text else None
    if code == 0 and (not flag or flag in output):
        item["verification"] = "命令探测通过"
    elif version:
        item["verification"] = "启动并输出版本，版本探测退出码 " + str(code)
    else:
        item["verification"] = "命令探测失败，退出码 " + str(code)
        item["probe_detail"] = "\n".join(output.splitlines()[-5:])
        item["missing"].append("通过命令探测的安装")


def size_bytes(path):
    code, value = probe(["du", "-s", "-B1", "--", path], timeout=55)
    amount = value.split(maxsplit=1)[0] if value else ""
    return int(amount) if code == 0 and amount.isdecimal() else None


def human_bytes(value):
    if value is None:
        return "未测得"
    unit = "B"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            break
        value /= 1024
    return f"{value:.1f} {unit}"


def deployment_requirements(profile, site, catalog):
    if profile is None:
        return []
    required = ["ppiflow", "abmpnn", "flowpacker", "protenix", "ipsae"]
    configured = site.get("tools", {})
    if profile == "full":
        required.append("rosetta")
        if configured.get("iggm", {}).get("automatic", {}).get("enabled"):
            required.append("iggm")
    required.extend(name for name, item in configured.items() if name in catalog and item.get("required"))
    return list(dict.fromkeys(required))


def markdown(report):
    scope = f"服务器 `{report['host']}` · 项目 `{report['project']}`"
    if report.get("inventory_date"):
        scope += " · " + report["inventory_date"]
    lines = ["# 配置信息", "", scope, "",
             f"- 系统：{report['platform']}",
             "- GPU：" + (" / ".join(report["gpus"]) or "未探测到"),
             "- 统计口径：安装目录按实际目标测量，目录内部软链不重复计入。共享环境和嵌套工具的占用不可相加。",
             "- 验证范围：路径、Python 版本及明确配置的轻量命令。"]
    if not report["gpus"]:
        lines.append("- GPU 获取帮助：[NVIDIA Linux 驱动安装说明](" + GPU_INSTALL_GUIDE + ")")
    if report.get("profile"):
        current = report["tools"]
        labels = (current[name]["label"] for name in report["required_missing"])
        lines.extend(["", "## 本次需要处理", "",
                      "- 检查范围：`" + report["profile"] + "`（发布包默认流程和站点必需项）",
                      "- 必需工具缺项：" + ("、".join(labels) or "无")])
        if report["required_unchecked"]:
            lines.append("- 未检查的必需工具：" + "、".join(report["required_unchecked"]))
        lines.append("- 按下文逐项补齐所需文件；标为按需的工具在任务启用时再准备。")
    required = set(report.get("required_tools", []))
    for name, item in report["tools"].items():
        lines.extend(["", f"## {item['label']}", "", "- 用途：" + item["role"]])
        if report.get("profile"):
            lines.append("- 本次部署：" + ("必需" if name in required else "按需"))
        fields = (
            ("安装入口目录", item["path"]),
            ("实际安装目录", item["real_path"]),
            ("对应环境", item["python"] or ("独立二进制" if item["entry"] else None)),
            ("Python 版本", item.get("python_version")),
            ("启动入口", item["launcher"] or item["entry"]),
            ("实际文件入口", item["entry"] if item["launcher"] else None),
            ("启动参数", item.get("launcher_flag")),
            ("版本", item.get("version")),
            ("源码提交", item.get("source_commit")),
            ("模型", item["model"]),
        )
        lines.extend(f"- {label}：`{value}`" for label, value in fields if value)
        if item["assets"]:
            lines.append("- 权重与配套数据：")
            lines.extend("  - `" + asset["path"] + "`" + (f"（{asset['issue']}）" if asset["issue"] else "")
                         for asset in item["assets"])
        elif item["weight_note"]:
            lines.append("- 权重：" + item["weight_note"])
        if "disk_bytes" in item:
            lines.append("- 实际占用：" + human_bytes(item["disk_bytes"]))
        if "env_disk_bytes" in item:
            lines.append("- 独立环境占用：" + human_bytes(item["env_disk_bytes"]))
        lines.append("- 验证状态：" + item["verification"])
        if item["missing"]:
            lines.append("- 缺项：" + "、".join(item["missing"]))
            if item.get("install_guide"):
                lines.append("- 安装后核对：" + item["install_guide"])
            if item.get("source"):
                lines.append("- 安装与获取：[官方教程或下载入口](" + item["source"] + ")")
        if item.get("probe_detail"):
            lines.append("- 诊断：" + item["probe_detail"].replace("\n", " / "))
    if report["legacy_paths"]:
        lines.extend(["", "## 项目中的旧绝对路径", ""])
        lines.extend(f"- `{path}` · {', '.join(names)}" for path, names in report["legacy_paths"].items())
    if report.get("setup"):
        lines.extend(["", "## 配置结果", ""])
        lines.extend("- " + result for result in report["setup"])
    return "\n".join(lines) + "\n"


def legacy_paths(project):
    result = {}
    scripts = project / "scripts"
    files = [scripts / "agent-ab", *scripts.glob("**/*.py")]
    pattern = r"/data[0-9]+/(?:anaconda3/bin/python|[\w.-]+/PPIFlow)"
    for source in sorted(p for p in files if p.is_file() and not p.is_symlink()):
        for match in sorted(set(re.findall(pattern, source.read_text(errors="replace")))):
            if not Path(match).exists():
                result.setdefault(match, []).append(str(source.relative_to(project)))
    return result


def rewrite_paths(obj, replacements):
    if isinstance(obj, dict):
        return {key: rewrite_paths(value, replacements) for key, value in obj.items()}
    if isinstance(obj, list):
        return [rewrite_paths(value, replacements) for value in obj]
    if isinstance(obj, str) and obj.startswith("/"):
        path = Path(obj)
        for source, destination in replacements:
            if path.is_relative_to(source):
                return str(destination / path.relative_to(source))
    return obj


def write_site(args, old, tools, conda, locations, catalog):
    if not old:
        raise ValueError("--write-site 需要 --site 指定一份完整站点配置")
    replacements = [(path_value(src), path_value(dst)) for src, dst in args.map]
    replacements.extend(
        (path_value(item["path"]), Path(tools[name]["path"]))
        for name, item in old["tools"].items()
        if item.get("path") and name in tools and tools[name]["path"]
    )
    for name, hint in locations.items():
        configured = old["tools"].get(name, {})
        for key in catalog.get(name, {}).get("asset_keys", []):
            if hint.get(key) and configured.get(key):
                replacements.append((path_value(configured[key]), path_value(hint[key])))
    conda_root = next((p for p in conda if is_executable(p / "bin/conda")), None)
    if conda_root and old.get("conda_root"):
        replacements.append((Path(old["conda_root"]), conda_root))
    if old.get("project_root"):
        replacements.append((Path(old["project_root"]), args.project))
    replacements = sorted(replacements, key=lambda entry: -len(entry[0].parts))
    result = rewrite_paths(old, replacements)
    result.update(host=socket.gethostname(), project_root=str(args.project))
    for name, item in result["tools"].items():
        tool = tools.get(name)
        if tool and tool["path"] and "path" in item:
            item["path"] = tool["path"]
        if tool and tool["python"]:
            item["env"] = tool["python"]
        for key in catalog.get(name, {}).get("asset_keys", []):
            if locations.get(name, {}).get(key):
                item[key] = str(path_value(locations[name][key]))
    with args.write_site.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")


def run_recipes(args, recipes, scan, conda, tools):
    if not args.apply:
        return conda, tools, [], False
    if not args.recipes:
        raise ValueError("--apply 需要 --recipes")
    events = []
    failed = False
    for name in args.apply:
        if name not in tools:
            raise ValueError("未知工具 " + name)
        if not tools[name]["missing"]:
            events.append(name + " 已找到，复用现有安装")
            continue
        recipe = recipes.get(name)
        if not recipe:
            events.append(name + " 缺少经授权的安装配方")
            continue
        cwd = path_value(recipe["cwd"])
        commands = recipe["commands"]
        if not cwd or not all(
            isinstance(argv, list) and argv and all(isinstance(arg, str) for arg in argv)
            for argv in commands
        ) or not commands:
            raise ValueError("配方需提供 cwd 和非空 argv 列表 " + name)
        cwd.mkdir(parents=True, exist_ok=True)
        for argv in commands:
            code, output = probe(argv, cwd=cwd, timeout=3600)
            if code != 0:
                events.append(f"{name} 配置失败，退出码 {code} · " + " / ".join(output.splitlines()[-4:]))
                failed = True
                break
        else:
            events.append(name + " 配方执行完成，见下方复检状态")
        conda, tools = scan()
    return conda, tools, events, failed


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project", type=path_value, required=True)
    parser.add_argument("--site", type=Path)
    parser.add_argument("--locations", type=Path, help="JSON 形式的工具路径覆盖")
    parser.add_argument("--search", type=path_value, action="append", default=[])
    parser.add_argument("--depth", type=int, default=3, choices=range(0, 7))
    parser.add_argument("--only", nargs="+", help="按工具 ID 限定盘点范围")
    parser.add_argument("--profile", choices=("stage1", "full"), help="显示该发布范围的必需工具")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--sizes", action="store_true")
    parser.add_argument("--recipes", type=Path)
    parser.add_argument("--apply", nargs="+", default=[], metavar="TOOL")
    parser.add_argument("--map", nargs=2, action="append", default=[], metavar=("OLD", "NEW"))
    parser.add_argument("--write-site", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args()

    catalog = read_json(CATALOG)
    selected = set(args.only or catalog)
    if selected - catalog.keys() or set(args.apply) - selected:
        parser.error("--only 或 --apply 含未知或未选择的工具")
    site_path = args.site or args.project / "configs/site.json"
    site = read_json(site_path) if site_path.is_file() else {}
    required = deployment_requirements(args.profile, site, catalog)
    replacements = [(path_value(src), path_value(dst)) for src, dst in args.map]
    checked_site = rewrite_paths(site, replacements)
    locations = read_json(args.locations) if args.locations else {}
    recipes = read_json(args.recipes) if args.recipes else {}
    if args.apply and not args.verify:
        args.verify = True

    def scan():
        search_roots = [*args.search, args.project / "tools", args.project / "links/software",
                        args.project.parent / "tools", Path.home() / "tools", Path.home() / "src", Path("/opt")]
        conda = conda_roots(args.project, checked_site, args.search)
        index = directory_index(search_roots + [prefix / "envs" for prefix in conda], args.depth)
        results = {}
        for name, item in catalog.items():
            if name not in selected and not any(
                catalog[other].get("parent") == name for other in selected
            ):
                continue
            parent = results.get(item.get("parent"))
            hint = locations.get(name, {})
            configured = checked_site.get("tools", {}).get(name, {})
            if name not in AUXILIARY and not configured and item.get("parent") and item.get("env"):
                configured = {"env": locations.get(item["parent"], {}).get("env")
                              or checked_site.get("tools", {}).get(item["parent"], {}).get("env")}
            results[name] = installed(item, hint, configured, conda, index, parent)
        results = {name: data for name, data in results.items() if name in selected}
        if args.verify:
            for name, data in results.items():
                verify(catalog[name], data)
        return conda, results

    conda, tools = scan()
    conda, tools, setup, recipe_failed = run_recipes(args, recipes, scan, conda, tools)
    if args.sizes:
        roots = {}
        for name, data in tools.items():
            if data["real_path"]:
                root = Path(data["real_path"])
                roots[(name, "disk_bytes")] = root
                if data["python"]:
                    env = Path(data["python"]).parent.parent.resolve()
                    if not env.is_relative_to(root):
                        roots[(name, "env_disk_bytes")] = env
        unique = list(dict.fromkeys(roots.values()))
        with ThreadPoolExecutor(max_workers=5) as executor:
            sizes = dict(zip(unique, executor.map(size_bytes, unique)))
        for (name, field), root in roots.items():
            tools[name][field] = sizes[root]

    if args.write_site:
        if args.only:
            parser.error("--write-site 需要完整盘点，请移除 --only")
        write_site(args, site, tools, conda, locations, catalog)
        setup.append("已生成站点配置 " + str(args.write_site))
    code, gpu = probe(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"])
    report = {
        "host": socket.gethostname(), "project": str(args.project),
        "platform": platform.platform(), "gpus": gpu.splitlines() if code == 0 else [],
        "tools": tools, "legacy_paths": legacy_paths(args.project), "setup": setup,
        "profile": args.profile, "required_tools": required,
        "required_missing": {name: tools[name]["missing"] for name in required if name in tools and tools[name]["missing"]},
        "required_unchecked": [name for name in required if name not in tools],
    }
    data = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    text = markdown(report)
    if args.report_dir:
        args.report_dir.mkdir(parents=True, exist_ok=True)
        (args.report_dir / "software-inventory.json").write_text(data, encoding="utf-8")
        (args.report_dir / "software-inventory.md").write_text(text, encoding="utf-8")
    print(data if args.format == "json" else text, end="")
    return 1 if recipe_failed or any(tools[name]["missing"] for name in args.apply) else 0


if __name__ == "__main__":
    raise SystemExit(main())
