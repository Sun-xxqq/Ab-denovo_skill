from .fastrelax_naming import fastrelax_path
import csv
import os
import re
import subprocess
from pathlib import Path

from .config import env_python
from .util import now_iso, read_json, select_gpu, write_json


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}


def _run_dir(config, run_id):
    path = Path(config["storage"]["runs_root"]) / run_id
    if not path.is_dir():
        raise FileNotFoundError("run not found: %s" % run_id)
    return path


def _parse_fasta(path):
    records = []
    name = None
    chunks = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(chunks)))
                name = line[1:].split()[0]
                chunks = []
            else:
                if name is None:
                    raise ValueError("FASTA sequence appears before its header: %s" % path)
                chunks.append(line.upper())
    if name is not None:
        records.append((name, "".join(chunks)))
    if not records:
        raise ValueError("FASTA has no records: %s" % path)
    return records


def _validate_fastas(masked_fasta, origin_fasta):
    masked = _parse_fasta(masked_fasta)
    origin = _parse_fasta(origin_fasta)
    if len(masked) not in (2, 3):
        raise ValueError("IgGM affinity maturation requires H+A or H+L+A FASTA records")
    if [item[0] for item in masked] != [item[0] for item in origin]:
        raise ValueError("masked and origin FASTA chain IDs differ")
    for (chain, masked_seq), (_, origin_seq) in zip(masked, origin):
        if len(masked_seq) != len(origin_seq):
            raise ValueError("masked and origin FASTA lengths differ for chain %s" % chain)
        if "X" in origin_seq:
            raise ValueError("origin FASTA must not contain X: chain %s" % chain)
    if "X" in masked[-1][1]:
        raise ValueError("the antigen chain (last FASTA record) cannot contain X")
    if not any("X" in sequence for _, sequence in masked[:-1]):
        raise ValueError("masked FASTA has no antibody design positions marked with X")
    return masked


def _pdb_chain_residues(path):
    chains = {}
    seen = set()
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")) or line[12:16].strip() != "CA":
                continue
            altloc = line[16:17]
            if altloc not in (" ", "A"):
                continue
            chain = line[21:22].strip()
            residue = (line[22:26].strip(), line[26:27].strip())
            key = (chain, residue)
            if not chain or key in seen:
                continue
            aa = AA3_TO_1.get(line[17:20].strip().upper())
            if not aa:
                continue
            seen.add(key)
            chains.setdefault(chain, []).append((residue, aa))
    return chains


def _parse_design_residues(value):
    requested = set()
    for token in re.split(r"[\s,]+", str(value or "").strip()):
        if not token:
            continue
        match = re.fullmatch(
            r"([^:]+):(-?\d+)([A-Za-z]?)(?:-(-?\d+)([A-Za-z]?))?",
            token,
        )
        if not match:
            # PPIFlow fixed_positions.csv uses compact PDB residue IDs such as
            # A101,B30.  IgGM's public interface uses A:101,B:30.  Accept the
            # compact single-character chain form at this integration boundary.
            match = re.fullmatch(
                r"([A-Za-z0-9])(-?\d+)([A-Za-z]?)(?:-(-?\d+)([A-Za-z]?))?",
                token,
            )
        if not match:
            raise ValueError(
                "invalid design residue %r; use CHAIN:31, CHAIN:31-35, "
                "or compact PDB form A31" % token
            )
        chain, start, start_icode, end, end_icode = match.groups()
        if end is None:
            requested.add((chain, str(int(start)), start_icode.upper()))
            continue
        if start_icode or end_icode:
            raise ValueError("insertion codes are only supported for single residues: %s" % token)
        first, last = int(start), int(end)
        if last < first:
            raise ValueError("design residue range is reversed: %s" % token)
        for number in range(first, last + 1):
            requested.add((chain, str(number), ""))
    if not requested:
        raise ValueError("at least one design residue is required")
    return requested


def _between_motifs(residues, left, right, name):
    sequence = "".join(item[1] for item in residues)
    try:
        start = sequence.index(left) + len(left)
        end = sequence.index(right, start)
    except ValueError:
        raise ValueError("cannot identify %s from sequence motifs %s...%s" % (name, left, right))
    if end <= start:
        raise ValueError("empty CDR inferred for %s" % name)
    return residues[start:end]


def _format_design_residues(residues):
    by_chain = {}
    for chain, number, icode in residues:
        if icode:
            by_chain.setdefault(chain, []).append((int(number), icode))
        else:
            by_chain.setdefault(chain, []).append((int(number), ""))
    tokens = []
    for chain in sorted(by_chain):
        plain = sorted(number for number, icode in by_chain[chain] if not icode)
        inserted = sorted((number, icode) for number, icode in by_chain[chain] if icode)
        if plain:
            start = previous = plain[0]
            for number in plain[1:] + [None]:
                if number is not None and number == previous + 1:
                    previous = number
                    continue
                tokens.append(
                    "%s:%s" % (chain, start)
                    if start == previous
                    else "%s:%s-%s" % (chain, start, previous)
                )
                if number is not None:
                    start = previous = number
        tokens.extend("%s:%s%s" % (chain, number, icode) for number, icode in inserted)
    return ",".join(tokens)


def infer_cdr_design_residues(complex_pdb, heavy_chain, light_chain, fixed_positions=None):
    """Infer the six generated CDR segments and preserve Stage 2 fixed residues."""
    chains = _pdb_chain_residues(complex_pdb)
    if heavy_chain not in chains or light_chain not in chains:
        raise ValueError("IgGM automatic maturation requires both heavy and light chains")
    cdrs = []
    for residues, left, right, name in (
        (chains[heavy_chain], "CKAS", "IHW", "CDR-H1"),
        (chains[heavy_chain], "LEWMGV", "DYNQ", "CDR-H2"),
        (chains[heavy_chain], "YYC", "WGQ", "CDR-H3"),
        (chains[light_chain], "CRSS", "LHW", "CDR-L1"),
        (chains[light_chain], "LLI", "SNRF", "CDR-L2"),
        (chains[light_chain], "YYC", "FGC", "CDR-L3"),
    ):
        cdrs.extend((heavy_chain if name.startswith("CDR-H") else light_chain, item[0][0], item[0][1]) for item in _between_motifs(residues, left, right, name))
    fixed = _parse_design_residues(fixed_positions) if fixed_positions else set()
    mutable = set(cdrs) - fixed
    if not mutable:
        raise ValueError("no mutable CDR residues remain after preserving Stage 2 fixed positions")
    return _format_design_residues(mutable)


def _stage2_rosetta_candidates(run_dir):
    stage2 = Path(run_dir) / "pipeline_output" / "stage2"
    roots = [
        fastrelax_path(stage2 / "08_fastrelax"),
        fastrelax_path(stage2 / "fastrelax_output"),
        fastrelax_path(stage2 / "fastrelax_raw_output"),
    ]
    candidates = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*/*_0001.pdb")):
            candidates.setdefault(path.parent.name, path)
        for path in sorted(root.glob("*_0001.pdb")):
            candidates.setdefault(path.stem[:-5], path)
    return candidates


def _stage2_fixed_positions(run_dir, candidate_label):
    match = re.match(r"flowpacker\.1_([^_]+)_", candidate_label)
    if not match:
        return None
    parent = match.group(1)
    stage2 = Path(run_dir) / "pipeline_output" / "stage2"
    tables = [
        stage2 / "00_stage2_plan" / "fixed_positions.csv",
        stage2 / "fixed_positions.csv",
    ]
    for table in tables:
        if not table.is_file():
            continue
        with table.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if Path(row.get("filename", "")).stem == "backbone.0_%s" % parent:
                    value = str(row.get("fixed_positions", "")).strip()
                    return None if not value or value.upper() == "NONE" else value
    return None


def run_stage2_affinity_maturation(config, run_id, spec, gpu="auto"):
    """Automatically mature every Stage 2 candidate that passed both target screens."""
    run_dir = _run_dir(config, run_id)
    candidates = _stage2_rosetta_candidates(run_dir)
    if not candidates:
        raise FileNotFoundError("no completed Stage 2 FastRelax candidates found")
    tool = config["tools"]["iggm"]
    automatic = tool.get("automatic", {})
    heavy_chain = spec["framework"]["heavy_chain"]
    light_chain = spec["framework"].get("light_chain")
    antigen_chain = spec["target"]["chain"]
    if not light_chain:
        raise ValueError("automatic IgGM affinity maturation requires a paired VH/VL design")
    epitope = []
    for hotspot in spec.get("target", {}).get("hotspots", []):
        match = re.fullmatch(r"([^\d-]+)(-?\d+)", str(hotspot).strip())
        if match and match.group(1) == antigen_chain:
            epitope.append(int(match.group(2)))
    results = []
    for label, pdb in sorted(candidates.items()):
        output_root = _iggm_output_root(run_dir) / label
        manifest_path = output_root / "manifest.json"
        if manifest_path.is_file():
            previous = read_json(manifest_path)
            if previous.get("status") == "complete":
                results.append(previous)
                continue
        fixed = _stage2_fixed_positions(run_dir, label)
        design_residues = infer_cdr_design_residues(
            pdb,
            heavy_chain,
            light_chain,
            fixed_positions=fixed if automatic.get("preserve_stage2_fixed_positions", True) else None,
        )
        results.append(
            run_affinity_maturation(
                config,
                run_id,
                complex_pdb=pdb,
                heavy_chain=heavy_chain,
                light_chain=light_chain,
                antigen_chain=antigen_chain,
                design_residues=design_residues,
                epitope=epitope or None,
                num_samples=int(automatic.get("num_samples", 1)),
                steps=int(automatic.get("steps", 10)),
                chunk_size=int(automatic.get("chunk_size", 16)),
                temperature=float(automatic.get("temperature", 1.0)),
                max_antigen_size=int(automatic.get("max_antigen_size", 384)),
                gpu=gpu,
                relax=False,
                dry_run=False,
                label=label,
            )
        )
    batch = {
        "stage": "stage2_iggm_affinity_maturation",
        "run_id": run_id,
        "created_at": now_iso(),
        "candidate_count": len(results),
        "results": results,
        "scientific_output": "generated variants; not an experimental affinity measurement",
    }
    write_json(
        _iggm_output_root(run_dir) / "batch_manifest.json",
        batch,
    )
    return batch


def prepare_inputs(complex_pdb, output_dir, heavy_chain, antigen_chain,
                   design_residues, light_chain=None):
    complex_pdb = Path(complex_pdb).resolve()
    if not complex_pdb.is_file():
        raise FileNotFoundError("complex PDB not found: %s" % complex_pdb)
    chains = _pdb_chain_residues(complex_pdb)
    ordered = [heavy_chain] + ([light_chain] if light_chain else []) + [antigen_chain]
    if len(set(ordered)) != len(ordered):
        raise ValueError("heavy, light and antigen chain IDs must be distinct")
    missing_chains = [chain for chain in ordered if chain not in chains]
    if missing_chains:
        raise ValueError("PDB is missing requested chains: %s" % ", ".join(missing_chains))
    requested = _parse_design_residues(design_residues)
    antibody_chains = set(ordered[:-1])
    wrong = sorted({chain for chain, _, _ in requested if chain not in antibody_chains})
    if wrong:
        raise ValueError("design residues must belong to antibody chains: %s" % ", ".join(wrong))

    found = set()
    origin_records = []
    masked_records = []
    chain_residue_map = {}
    for chain in ordered:
        sequence = "".join(item[1] for item in chains[chain])
        masked = list(sequence)
        if chain in antibody_chains:
            for index, ((number, icode), _) in enumerate(chains[chain]):
                key = (chain, str(int(number)), icode.upper())
                if key in requested:
                    masked[index] = "X"
                    found.add(key)
        chain_residue_map[chain] = [
            {
                "pdb_residue": "%s%s" % (number, icode),
                "sequence_index_1based": index + 1,
                "masked": masked[index] == "X",
            }
            for index, ((number, icode), _) in enumerate(chains[chain])
        ]
        origin_records.append((chain, sequence))
        masked_records.append((chain, "".join(masked)))
    missing = sorted(requested - found)
    if missing:
        labels = ["%s:%s%s" % item for item in missing]
        raise ValueError("design residues not found in PDB: %s" % ", ".join(labels))

    output_dir = Path(output_dir)
    input_dir = output_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    masked_path = input_dir / "masked.fasta"
    origin_path = input_dir / "origin.fasta"
    for path, records in ((masked_path, masked_records), (origin_path, origin_records)):
        with path.open("w", encoding="utf-8") as handle:
            for chain, sequence in records:
                handle.write(">%s\n%s\n" % (chain, sequence))
    mapping = {
        "source_pdb": str(complex_pdb),
        "heavy_chain": heavy_chain,
        "light_chain": light_chain,
        "antigen_chain": antigen_chain,
        "designed_pdb_residues": ["%s:%s%s" % item for item in sorted(found)],
        "chain_residue_map": chain_residue_map,
        "masked_fasta": str(masked_path),
        "origin_fasta": str(origin_path),
    }
    write_json(input_dir / "chain_mapping.json", mapping)
    return mapping


def _write_candidate_index(output_dir):
    candidates = {}
    for artifact_type, pattern in (("pdb", "*.pdb"), ("fasta", "*.fasta")):
        for path in sorted(Path(output_dir).glob(pattern)):
            row = candidates.setdefault(
                path.stem,
                {"candidate_id": path.stem, "pdb": "", "fasta": ""},
            )
            row[artifact_type] = str(path.resolve())
    rows = [candidates[candidate_id] for candidate_id in sorted(candidates)]
    index_path = Path(output_dir) / "candidates.csv"
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["candidate_id", "pdb", "fasta"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return str(index_path), len(rows)


def run_affinity_maturation(config, run_id, masked_fasta=None, origin_fasta=None,
                            antigen=None, complex_pdb=None, heavy_chain=None,
                            light_chain=None, antigen_chain=None,
                            design_residues=None, epitope=None, num_samples=1,
                            steps=10, chunk_size=64, temperature=1.0,
                            max_antigen_size=384, gpu="auto", relax=False,
                            dry_run=False, label=None):
    run_dir = _run_dir(config, run_id)
    tool = config["tools"].get("iggm")
    if not tool:
        raise KeyError("tool is not configured: iggm")
    tool_root = Path(tool["path"])
    design_script = tool_root / "design.py"
    python = Path(env_python(config, tool["env"]))
    if not design_script.is_file():
        raise FileNotFoundError("IgGM design.py not found: %s" % design_script)
    if not python.is_file():
        raise FileNotFoundError("IgGM Python not found: %s" % python)

    output_root = _iggm_output_root(run_dir)
    if label:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", str(label)):
            raise ValueError("label must contain only letters, digits, dot, underscore or hyphen")
        output_root = output_root / str(label)
    candidate_dir = output_root / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    if complex_pdb:
        if masked_fasta or origin_fasta or antigen:
            raise ValueError("use either --complex-pdb preparation mode or explicit FASTA mode")
        if not heavy_chain or not antigen_chain or not design_residues:
            raise ValueError("--complex-pdb requires --heavy-chain, --antigen-chain and --design-residues")
        prepared = prepare_inputs(
            complex_pdb, output_root, heavy_chain, antigen_chain,
            design_residues, light_chain=light_chain,
        )
        masked_fasta = prepared["masked_fasta"]
        origin_fasta = prepared["origin_fasta"]
        antigen = prepared["source_pdb"]
    elif not (masked_fasta and origin_fasta and antigen):
        raise ValueError("explicit mode requires --fasta, --fasta-origin and --antigen")

    for path_label, path in (("masked FASTA", masked_fasta), ("origin FASTA", origin_fasta), ("antigen PDB", antigen)):
        if not Path(path).is_file():
            raise FileNotFoundError("%s not found: %s" % (path_label, path))
    _validate_fastas(masked_fasta, origin_fasta)
    if int(num_samples) < 1:
        raise ValueError("num_samples must be positive")
    gpu_id = select_gpu() if gpu == "auto" else int(str(gpu).split(",", 1)[0])
    argv = [
        str(python), str(design_script),
        "--fasta", str(Path(masked_fasta).resolve()),
        "--fasta_origin", str(Path(origin_fasta).resolve()),
        "--antigen", str(Path(antigen).resolve()),
        "--output", str(candidate_dir),
        "--run_task", "affinity_maturation",
        "--num_samples", str(int(num_samples)),
        "--steps", str(int(steps)),
        "--chunk_size", str(int(chunk_size)),
        "--temperature", str(float(temperature)),
        "--max_antigen_size", str(int(max_antigen_size)),
    ]
    if epitope:
        argv.extend(["--epitope"] + [str(int(item)) for item in epitope])
    if relax:
        argv.append("--relax")
    source_commit_path = tool_root / "SOURCE_COMMIT"
    source_commit = (
        source_commit_path.read_text(encoding="utf-8").strip()
        if source_commit_path.is_file()
        else None
    )
    manifest = {
        "tool": "IgGM",
        "role": "postscreen_affinity_maturation_variant_generator",
        "label": label,
        "scientific_output": "generated variants; not an experimental affinity measurement",
        "run_id": run_id,
        "created_at": now_iso(),
        "source_commit": source_commit,
        "weights_checksums": str(tool_root / "WEIGHTS.sha256"),
        "masked_fasta": str(Path(masked_fasta).resolve()),
        "origin_fasta": str(Path(origin_fasta).resolve()),
        "antigen": str(Path(antigen).resolve()),
        "epitope": [int(item) for item in (epitope or [])],
        "gpu": gpu_id,
        "argv": argv,
        "output": str(candidate_dir),
        "status": "planned" if dry_run else "running",
    }
    manifest_path = output_root / "manifest.json"
    write_json(manifest_path, manifest)
    if dry_run:
        return manifest

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_name = "iggm_affinity_maturation%s.log" % (".%s" % label if label else "")
    log_path = run_dir / "logs" / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            argv, cwd=str(tool_root), env=environment,
            stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    manifest["returncode"] = completed.returncode
    manifest["log"] = str(log_path)
    if completed.returncode:
        manifest["status"] = "failed"
        write_json(manifest_path, manifest)
        raise RuntimeError("IgGM failed with exit code %s; see %s" % (completed.returncode, log_path))
    index_path, count = _write_candidate_index(candidate_dir)
    manifest.update({"status": "complete", "candidate_count": count, "candidate_index": index_path})
    write_json(manifest_path, manifest)
    return manifest
def _iggm_output_root(run_dir):
    """Canonical numbered Stage 2 output for IgGM variants."""
    return Path(run_dir) / "pipeline_output" / "stage2" / "09_IgGM"


