import argparse
import json
import os
import sys

from . import __version__
from .config import load_site_config
from .doctor import run_doctor
from .links import initialize_links
from .command_input import SUPPORTED_DESIGN_TYPES, parse_ppiflow_command
from .spec import load_and_validate, validate_spec
from .workflow import (
    create_plan,
    execute_run,
    get_run,
    get_run_parts,
    list_runs,
    record_backbone_selection,
    run_fastrelax,
)
from .orchestrator import run_orchestration
from .round_selection import list_round_inputs, record_round_input
from .references import input_reference, shell_example
from .iggm import run_affinity_maturation


def _print_json(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _print_result_paths(paths):
    if not paths:
        return
    print("\nRESULT PATHS")
    for name, path in paths.items():
        if name in {"rosettafix_output", "stage2_rosettafix_output"}:
            continue
        print("%-32s %s" % (name, path))


def _print_pipeline_summary(result):
    summary = result.get("screening_summary") or {}
    rows = summary.get("rows") or []
    print("\nSCREENING SUMMARY")
    for row in rows:
        print(
            "%-38s %s/%s (%s%%)"
            % (
                row["screen"],
                row["passed"],
                row["total"],
                row["pass_rate_percent"],
            )
        )
    if summary.get("path"):
        print("summary_csv                            %s" % summary["path"])
    result_paths = result.get("result_paths") or {}
    if result_paths.get("run_directory"):
        print("run_directory                          %s" % result_paths["run_directory"])


def _prompt_for_top_n(config, run_id, action):
    """Read a reviewed Top-N only from an attached interactive terminal."""
    minimum = int(action.get("minimum_top_n", 1))
    maximum = int(action["maximum_top_n"])
    print("\n" + action["prompt"])
    print("排序表: %s" % action["ranked_table"])
    while True:
        value = input("请输入要保留的 Top-N（%d-%d）: " % (minimum, maximum)).strip()
        try:
            top_n = int(value)
        except ValueError:
            print("请输入整数。")
            continue
        if not minimum <= top_n <= maximum:
            print("请输入 %d 到 %d 之间的整数。" % (minimum, maximum))
            continue
        selection = record_backbone_selection(config, run_id, top_n)
        print(
            "已记录 Top-%d；将跳过已完成的骨架生成，从 AbMPNN 自动继续。"
            % selection["selected_top_n"]
        )
        return selection


def _ask(label, default=None):
    suffix = " [%s]" % default if default not in (None, "") else ""
    value = input("%s%s: " % (label, suffix)).strip()
    return value if value else default


def _select_design_type():
    print("请选择设计类型：")
    print("  1. binder    蛋白结合物")
    print("  2. nanobody  纳米抗体/VHH")
    print("  3. antibody  VH/VL 抗体或 scFv")
    aliases = {"1": "binder", "2": "nanobody", "3": "antibody"}
    value = _ask("输入序号或类型名称", "nanobody").strip().lower()
    value = aliases.get(value, value)
    if value not in SUPPORTED_DESIGN_TYPES:
        raise ValueError("设计类型必须是 binder、nanobody 或 antibody")
    return value


def _command_input(design_type=None, command=None):
    design_type = design_type or _select_design_type()
    reference = shell_example(design_type)
    print("\nPPIFlow 参数格式参考（请替换路径、链和热点）：\n")
    print(reference)
    print()
    if command:
        supplied = command
    else:
        supplied = _ask("请粘贴一行 PPIFlow 命令")
    return parse_ppiflow_command(design_type, supplied)


def build_parser():
    parser = argparse.ArgumentParser(prog="agent-ab", description="Antibody design workflow agent")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--site-config", help="override configs/site.json")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="check software, environments, assets, GPU and storage")
    doctor.add_argument("--json", action="store_true")

    links = sub.add_parser("links", help="create lightweight symbolic links; never copy software or weights")
    links.add_argument("--home-compat", action="store_true", help="also create ~/common and ~/checkpoint")

    validate = sub.add_parser("validate", help="validate a design spec without creating a run")
    validate.add_argument("spec")

    plan = sub.add_parser("plan", help="materialize a reproducible run without GPU compute")
    plan.add_argument("spec")

    run = sub.add_parser("run", help="create and execute a PPIFlow run")
    run.add_argument("spec")
    run.add_argument("--stage", type=int, choices=[1, 2])
    run.add_argument("--gpu", default="auto")

    resume = sub.add_parser("resume", help="execute or resume a materialized run")
    resume.add_argument("run_id")
    resume.add_argument("--stage", type=int, choices=[1, 2])
    resume.add_argument("--gpu", default="auto")

    rosettafix = sub.add_parser(
        "fastrelax", aliases=["rosettafix"], help="run FastRelax as the final step of Stage 1 or Stage 2"
    )
    rosettafix.add_argument("run_id")
    rosettafix.add_argument("--stage", type=int, choices=[1, 2], required=True)

    select_backbones = sub.add_parser(
        "select-backbones",
        help="record the user's reviewed scFv top-N choice before AbMPNN",
    )
    select_backbones.add_argument("run_id")
    select_backbones.add_argument("top_n", type=int)
    select_backbones.add_argument(
        "--continue",
        dest="continue_pipeline",
        action="store_true",
        help="continue the remaining pipeline immediately after recording Top-N",
    )
    select_backbones.add_argument("--gpu", default="auto")
    select_backbones.add_argument("--max-retries", type=int, default=1)
    select_backbones.add_argument("--retry-failed", action="store_true")

    select_round = sub.add_parser(
        "select-round-input",
        help="reuse selected results from one round as a later round checkpoint",
    )
    select_round.add_argument("run_id")
    select_round.add_argument("--from-round", type=int, required=True)
    select_round.add_argument("--to-round", type=int, required=True)
    select_round.add_argument(
        "--candidate-id",
        action="append",
        default=[],
        help="candidate_id to reuse; repeat as needed (default: all passing rows)",
    )
    select_round.add_argument(
        "--candidate-file",
        help="JSON, CSV or one-candidate_id-per-line selection file",
    )
    select_round.add_argument(
        "--source-kind",
        choices=["auto", "fastrelax", "rosettafix", "canonical"],
        default="auto",
    )
    select_round.add_argument("--table", help="override the source round CSV")
    select_round.add_argument("--replace", action="store_true")

    round_inputs = sub.add_parser(
        "round-inputs", help="list reusable round checkpoints for one run"
    )
    round_inputs.add_argument("run_id")

    pipeline = sub.add_parser("pipeline", help="run all selected adapters with recovery and ranking")
    pipeline.add_argument("run_id")
    pipeline.add_argument("--gpu", default="auto")
    pipeline.add_argument("--max-retries", type=int, default=1)
    pipeline.add_argument("--dry-run", action="store_true")
    pipeline.add_argument("--retry-failed", action="store_true")

    iggm = sub.add_parser(
        "iggm-mature",
        help="generate affinity-maturation variants after structural screening",
    )
    iggm.add_argument("run_id")
    iggm.add_argument(
        "--label",
        help="candidate label used as a separate output subdirectory",
    )
    iggm.add_argument("--fasta", help="IgGM masked FASTA; X marks mutable antibody residues")
    iggm.add_argument("--fasta-origin", help="unmasked parent antibody/antigen FASTA")
    iggm.add_argument("--antigen", help="antigen or parent complex PDB for explicit FASTA mode")
    iggm.add_argument("--complex-pdb", help="parent complex PDB to convert into IgGM input")
    iggm.add_argument("--heavy-chain")
    iggm.add_argument("--light-chain")
    iggm.add_argument("--antigen-chain")
    iggm.add_argument(
        "--design-residues",
        help="PDB-numbered antibody residues, e.g. H:31,H:33-35,L:50",
    )
    iggm.add_argument("--epitope", nargs="+", type=int, help="1-based antigen sequence positions")
    iggm.add_argument("--num-samples", type=int, default=1)
    iggm.add_argument("--steps", type=int, default=10)
    iggm.add_argument("--chunk-size", type=int, default=64)
    iggm.add_argument("--temperature", type=float, default=1.0)
    iggm.add_argument("--max-antigen-size", type=int, default=384)
    iggm.add_argument("--gpu", default="auto")
    iggm.add_argument("--relax", action="store_true")
    iggm.add_argument("--dry-run", action="store_true")

    create = sub.add_parser(
        "create",
        help="select a design type and create a plan from explicit PPIFlow arguments",
    )
    create.add_argument("design_type", nargs="?", choices=SUPPORTED_DESIGN_TYPES)
    create.add_argument(
        "ppiflow_command",
        nargs=argparse.REMAINDER,
        help="PPIFlow arguments after --; omit them for interactive paste",
    )

    sub.add_parser("wizard", help="alias for interactive create")
    sub.add_parser("list", help="list runs")
    status = sub.add_parser("status", help="show one run")
    status.add_argument("run_id")
    parts = sub.add_parser("parts", help="show every pipeline part with its log and output")
    parts.add_argument("run_id")
    parts.add_argument("--tail", type=int, default=0, help="include the last N lines of each existing log")
    examples = sub.add_parser("examples", help="show PPIFlow command input by design type")
    examples.add_argument("design_type", nargs="?", choices=["binder", "nanobody", "vhh", "antibody", "scfv", "monomer", "motif"])
    agent_create = sub.add_parser("agent-create", help="create an isolated model-directed Stage 1 campaign")
    agent_create.add_argument("spec")
    agent_create.add_argument("--goal")
    agent_run = sub.add_parser("agent", help="let a configured model choose research actions from observations")
    agent_run.add_argument("run_id")
    agent_run.add_argument("--gpu", default="auto")
    agent_run.add_argument("--agent-config")
    agent_run.add_argument("--plan-only", action="store_true", help="request one decision without executing it")
    sub.add_parser("agent-tools", help="show the actual research tool catalog")
    agent_audit = sub.add_parser("agent-audit", help="read existing screening evidence without running models or compute")
    agent_audit.add_argument("run_id")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = load_site_config(args.site_config)
    ppiflow = config.get("tools", {}).get("ppiflow", {})
    if ppiflow.get("path"):
        os.environ["AGENT_AB_PPIFLOW_ROOT"] = str(ppiflow["path"])
    try:
        if args.command == "agent-create":
            from .research_agent import create_campaign
            _print_json(create_campaign(config, args.spec, goal=args.goal))
            return 0
        if args.command == "agent":
            from .research_agent import run_campaign
            result = run_campaign(config, args.run_id, gpu=args.gpu, agent_config=args.agent_config, plan_only=args.plan_only)
            _print_json(result)
            return 0 if result["status"] in {"achieved", "stopped", "planned", "needs_input"} else 2
        if args.command == "agent-tools":
            from .agent_tools import TOOLS
            _print_json(TOOLS)
            return 0
        if args.command == "agent-audit":
            from .agent_observations import audit_existing_run
            from .workflow import _run_dir
            _print_json(audit_existing_run(_run_dir(config, args.run_id)))
            return 0
        if args.command == "doctor":
            result = run_doctor(config)
            if args.json:
                _print_json(result)
            else:
                for item in result["checks"]:
                    marker = "OK" if item["ok"] else ("FAIL" if item["required"] else "WARN")
                    print("%-4s %-30s %s" % (marker, item["name"], item["detail"]))
                print("\nREADY" if result["ready"] else "\nBLOCKED: %s required failures" % result["required_failures"])
            return 0 if result["ready"] else 2
        if args.command == "links":
            results = initialize_links(config, include_home_compat=args.home_compat)
            for item in results:
                print("%-14s %s -> %s" % (item["state"], item["link"], item["target"]))
            return 2 if any(item["state"] == "conflict" for item in results) else 0
        if args.command == "validate":
            _print_json(load_and_validate(args.spec))
            return 0
        if args.command == "plan":
            _print_json(create_plan(config, spec_path=args.spec))
            return 0
        if args.command == "run":
            planned = create_plan(config, spec_path=args.spec)
            _print_json(execute_run(config, planned["run_id"], stage=args.stage, gpu=args.gpu))
            return 0
        if args.command == "resume":
            _print_json(execute_run(config, args.run_id, stage=args.stage, gpu=args.gpu))
            return 0
        if args.command in {"fastrelax", "rosettafix"}:
            _print_json(run_fastrelax(config, args.run_id, args.stage))
            return 0
        if args.command == "select-backbones":
            selection = record_backbone_selection(config, args.run_id, args.top_n)
            _print_json(selection)
            if args.continue_pipeline:
                print("\nTop-N 已记录，开始继续后续流程。")
                result = run_orchestration(
                    config,
                    args.run_id,
                    gpu=args.gpu,
                    max_retries=args.max_retries,
                    retry_failed=args.retry_failed,
                )
                _print_pipeline_summary(result)
            return 0
        if args.command == "select-round-input":
            result = record_round_input(
                config,
                args.run_id,
                args.from_round,
                args.to_round,
                candidate_ids=args.candidate_id,
                candidate_file=args.candidate_file,
                source_kind=args.source_kind,
                table=args.table,
                replace=args.replace,
            )
            _print_json(result)
            return 0
        if args.command == "round-inputs":
            _print_json(list_round_inputs(config, args.run_id))
            return 0
        if args.command == "pipeline":
            while True:
                result = run_orchestration(
                    config,
                    args.run_id,
                    gpu=args.gpu,
                    max_retries=args.max_retries,
                    dry_run=args.dry_run,
                    retry_failed=args.retry_failed,
                )
                if args.dry_run:
                    _print_json(result)
                    return 0
                if result.get("status") != "awaiting_user_input":
                    _print_pipeline_summary(result)
                    return 0
                action = result["action_required"]
                if action.get("type") in {"stage1_rosettafix", "stage2_rosettafix"}:
                    print("\n" + action["prompt"])
                    print("候选目录: %s" % action["candidate_directory"])
                    print("FastRelax输入: %s" % action["rosettafix_input"])
                    print("继续命令: %s" % action["continue_command"])
                    return 0
                if not sys.stdin.isatty():
                    print("\n" + action["prompt"])
                    print("排序表: %s" % action["ranked_table"])
                    print("当前为非交互运行，已安全暂停。")
                    print("选择命令: %s" % action["response_command"])
                    print("继续命令: %s" % action["continue_command"])
                    return 0
                _prompt_for_top_n(config, args.run_id, action)
        if args.command == "iggm-mature":
            _print_json(
                run_affinity_maturation(
                    config,
                    args.run_id,
                    masked_fasta=args.fasta,
                    origin_fasta=args.fasta_origin,
                    antigen=args.antigen,
                    complex_pdb=args.complex_pdb,
                    heavy_chain=args.heavy_chain,
                    light_chain=args.light_chain,
                    antigen_chain=args.antigen_chain,
                    design_residues=args.design_residues,
                    epitope=args.epitope,
                    num_samples=args.num_samples,
                    steps=args.steps,
                    chunk_size=args.chunk_size,
                    temperature=args.temperature,
                    max_antigen_size=args.max_antigen_size,
                    gpu=args.gpu,
                    relax=args.relax,
                    dry_run=args.dry_run,
                    label=args.label,
                )
            )
            return 0
        if args.command == "create":
            result = _command_input(args.design_type, args.ppiflow_command)
            result["spec"] = validate_spec(result["spec"])
            _print_json(
                create_plan(config, spec=result["spec"], request_context=result)
            )
            return 0
        if args.command == "wizard":
            result = _command_input()
            result["spec"] = validate_spec(result["spec"])
            _print_json(
                create_plan(config, spec=result["spec"], request_context=result)
            )
            return 0
        if args.command == "list":
            runs = list_runs(config)
            if not runs:
                print("No runs.")
            for item in runs:
                print("{run_id}\t{status}\t{design_type}\t{name}".format(**item))
            return 0
        if args.command == "status":
            _print_json(get_run(config, args.run_id))
            return 0
        if args.command == "parts":
            _print_json(get_run_parts(config, args.run_id, tail_lines=args.tail))
            return 0
        if args.command == "examples":
            value = input_reference(args.design_type)
            if args.design_type:
                value["copy_paste_command"] = shell_example(args.design_type)
            _print_json(value)
            return 0
    except (
        ValueError,
        KeyError,
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
    ) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
