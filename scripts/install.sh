#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
deps="$HOME/agent-ab-deps"
profile=stage1
mode=check

usage() {
    cat <<'HELP'
Usage: bash scripts/install.sh [--check|--setup] [--deps DIRECTORY] [--profile stage1|full] [-- PREFLIGHT_ARGS]
--check (default) inventories dependencies and prints install sources.
--setup creates the Agent-Ab environment, applies selected missing-tool recipes,
generates configs/site.json on first setup, and checks readiness.
HELP
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --deps) deps="${2:?--deps requires a directory}"; shift 2 ;;
        --profile) profile="${2:?--profile requires stage1 or full}"; shift 2 ;;
        --check) mode=check; shift ;;
        --setup) mode=setup; shift ;;
        --help|-h) usage; exit 0 ;;
        --) shift; break ;;
        *) usage >&2; exit 2 ;;
    esac
done
case "$profile" in
    stage1|full) ;;
    *) printf 'Unknown profile: %s\n' "$profile" >&2; exit 2 ;;
esac

base_python="${PYTHON3:-python3}"
if ! command -v "$base_python" >/dev/null 2>&1 ||
   ! "$base_python" -c 'import sys; sys.exit(not ((3, 9) <= sys.version_info[:2] <= (3, 12)))'; then
    printf '%s\n' '需要 Python 3.9–3.12。安装后可用 PYTHON3=/path/to/python 指定解释器。' \
        '官方说明 https://docs.python.org/3/using/unix.html' >&2
    exit 2
fi
deps="$("$base_python" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$deps")"
report="$ROOT/.setup-report"
site="$ROOT/configs/site.json"
for arg in "$@"; do
    case "$arg" in
        --profile|--profile=*|--report-dir|--report-dir=*|--write-site|--write-site=*)
            printf '%s is managed by install.sh.\n' "$arg" >&2
            exit 2 ;;
        --apply|--apply=*)
            if [[ "$mode" == check ]]; then
                printf '%s requires --setup.\n' "$arg" >&2
                exit 2
            fi ;;
    esac
done
if [[ "$mode" == check ]]; then
    if [[ ! -f "$site" ]]; then site="$ROOT/configs/site.template.json"; fi
    exec "$base_python" "$ROOT/skills/agent-ab-portable/scripts/preflight.py" \
        --project "$ROOT" --site "$site" --search "$deps" --search "$deps/conda" \
        --map /opt/agent-ab/deps "$deps" --profile "$profile" --verify --report-dir "$report" "$@"
fi
mkdir -p "$deps"
python="$ROOT/.venv/bin/python"
if [[ ! -x "$python" ]]; then
    "$base_python" -m venv "$ROOT/.venv"
fi
if ! "$python" -c 'import numpy; assert numpy.__version__ == "1.26.4"' 2>/dev/null; then
    "$python" -m pip install --only-binary=:all: -r "$ROOT/requirements.txt"
fi

mkdir -p "$report"
candidate="$site"
site_args=(--site "$site")
if [[ ! -f "$site" ]]; then
    staging="$(mktemp -d "$report/site.XXXXXX")"
    candidate="$staging/site.json"
    trap 'rm -f -- "$candidate"; rmdir -- "$staging"' EXIT
    site_args=(--site "$ROOT/configs/site.template.json" --write-site "$candidate")
fi
"$python" "$ROOT/skills/agent-ab-portable/scripts/preflight.py" \
    --project "$ROOT" "${site_args[@]}" --search "$deps" --search "$deps/conda" \
    --map /opt/agent-ab/deps "$deps" --profile "$profile" --verify --report-dir "$report" "$@"

"$python" - "$report/software-inventory.json" <<'CHECK'
import json
from pathlib import Path
import sys

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for name, missing in report["required_missing"].items():
    tool = report["tools"][name]
    print(tool["label"] + " 缺少 " + "、".join(missing), file=sys.stderr)
    if tool.get("source"):
        print("  安装教程 " + tool["source"], file=sys.stderr)
if report["required_unchecked"]:
    print("未检查的必需工具 " + "、".join(report["required_unchecked"]), file=sys.stderr)
if report["required_missing"] or report["required_unchecked"]:
    raise SystemExit("See .setup-report/software-inventory.md and docs/INSTALL.md")
CHECK

AGENT_AB_PYTHON="$python" bash "$ROOT/scripts/agent-ab" --site-config "$candidate" doctor
if [[ "$candidate" != "$site" ]]; then
    ln -- "$candidate" "$site"
    printf 'Created %s\n' "$site"
fi
