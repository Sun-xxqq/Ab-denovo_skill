"""Evidence readers: no model inference, no mutations to historical runs."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def pdb_inventory(path):
    path = Path(path)
    chains, missing, alternate = {}, [], set()
    invalid_atoms = 0
    with path.open(encoding='utf-8', errors='replace') as handle:
        for line in handle:
            if line.startswith('REMARK 465'):
                missing.append(line.strip())
            if not line.startswith('ATOM  '):
                continue
            try:
                coords = [float(line[a:b]) for a, b in [(30, 38), (38, 46), (46, 54)]]
                if not all(math.isfinite(x) for x in coords):
                    raise ValueError('non-finite coordinate')
                chain = line[21]
                number = int(line[22:26])
                insertion = line[26].strip()
                atom = line[12:16].strip()
            except (ValueError, IndexError):
                invalid_atoms += 1
                continue
            key = '%s%s' % (number, insertion)
            residue = chains.setdefault(chain, {}).setdefault(key, {'atoms': set(), 'resname': line[17:20].strip()})
            residue['atoms'].add(atom)
            if line[16].strip():
                alternate.add(chain + key)
    return {
        'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'chains': {c: {
            'residue_count': len(rs), 'residue_labels': list(rs),
            'missing_CA': [k for k, r in rs.items() if 'CA' not in r['atoms']],
        } for c, rs in chains.items()},
        'missing_residue_records': missing[:30], 'alternate_location_residues': len(alternate),
        'invalid_atom_records': invalid_atoms,
        'interpretation': 'Input integrity only; does not establish experimental accuracy or a preferred conformation.',
    }


def inspect_spec(spec):
    target = pdb_inventory(spec['target']['structure'])
    chain = spec['target']['chain']
    chain_info = target['chains'].get(chain, {})
    residues = set(chain_info.get('residue_labels', []))
    missing_hotspots = [h for h in spec['target']['hotspots'] if h[0] != chain or h[1:] not in residues]
    issues = []
    if not chain_info:
        issues.append('target_chain_missing')
    if target['invalid_atom_records']:
        issues.append('invalid_target_atom_records')
    if chain_info.get('missing_CA'):
        issues.append('target_residues_missing_CA')
    if missing_hotspots:
        issues.append('hotspot_residues_missing')
    if spec['design_type'] == 'nanobody' and len(spec['target']['hotspots']) < 2:
        issues.append('nanobody_geometry_screen_requires_two_hotspots')
    if spec['design_type'] == 'antibody' and len(spec['target']['hotspots']) < 3:
        issues.append('antibody_geometry_screen_requires_three_hotspots')
    framework = None
    if spec.get('framework'):
        framework = pdb_inventory(spec['framework']['structure'])
        names = [spec['framework']['heavy_chain']]
        if spec['design_type'] == 'antibody':
            names.append(spec['framework']['light_chain'])
        for name in names:
            if name not in framework['chains']:
                issues.append('framework_chain_missing_' + name)
        if framework['invalid_atom_records']:
            issues.append('invalid_framework_atom_records')
    return {'status': 'needs_input' if issues else 'ready', 'target': target, 'framework': framework,
            'hotspots': spec['target']['hotspots'], 'missing_hotspots': missing_hotspots,
            'issues': issues, 'conformation_assessment': 'not_assessed'}


def summarize_records(records, pass_field='filter_pass'):
    def truth(value):
        return value is True or str(value).lower() in {'true', '1', 'pass', 'yes'}
    passed = [r for r in records if truth(r.get(pass_field))]
    rejected = [r for r in records if r.get(pass_field) is False or str(r.get(pass_field)).lower() in {'false', '0'}]
    reasons = Counter()
    for r in rejected:
        raw = r.get('rejection_reasons') or []
        reasons.update(raw.split(';') if isinstance(raw, str) else raw)
    reasons.pop('', None)
    metric_names = ['antibody_plddt', 'ptm', 'iptm_global', 'S_on', 'design_pred_rmsd',
                    'screen_distance_A', 'CDR3_anchor_min_A', 'third_CDR_min_A',
                    'third_hotspot_min_A', 'active_CDR_count', 'hotspot_coverage_below_cutoff']
    evidence = [{k: r[k] for k in ['candidate_id', 'backbone_id', pass_field, 'rejection_reasons'] + metric_names if k in r}
                for r in records[:24]]
    return {'candidate_evidence': evidence, 'candidate_evidence_truncated': len(records) > 24,
            'evaluated': len(records), 'passed': len(passed), 'rejected': len(rejected),
            'uncertain': len(records) - len(passed) - len(rejected),
            'rejection_reason_counts': dict(reasons), 'reason_counts_overlap': True,
            'passing_ids': [r.get('candidate_id') or r.get('backbone_id') for r in passed]}


def read_screen(root):
    root = Path(root)
    records = []
    for path in sorted(root.glob('*/filter_results.json')):
        records.extend(json.loads(path.read_text(encoding='utf-8-sig')))
    if not records:
        raise FileNotFoundError('no filter_results.json under %s' % root)
    return records


def sequence_report(root):
    chains = []
    for path in sorted(set(Path(root).rglob('*.fasta')) | set(Path(root).rglob('*.fa'))):
        seq = ''
        for line in path.read_text().splitlines():
            if line.startswith('>'):
                if seq:
                    chains.append(seq)
                seq = ''
            else:
                seq += line.strip()
        if seq:
            chains.append(seq)
    # This is a transparent sequence warning, not a calibrated aggregation predictor.
    hydrophobic = set('VILMFWY')
    warnings = []
    for i, seq in enumerate(chains):
        run = longest = 0
        for aa in seq:
            run = run + 1 if aa in hydrophobic else 0
            longest = max(longest, run)
        if longest >= 6:
            warnings.append({'record': i, 'longest_hydrophobic_run': longest})
    return {'fasta_records': len(chains), 'unique_full_records': len(set(chains)),
            'hydrophobic_run_warnings': warnings[:20],
            'aggregation_risk': 'not_measured',
            'interpretation': 'Exact sequence duplication and hydrophobic-run heuristic only; no validated aggregation score.'}


def audit_existing_run(run_dir):
    """Read available screening tables without rescoring or rewriting the run."""
    run_dir = Path(run_dir)
    output = {}
    candidates = [run_dir / 'pipeline_output' / 'stage1' / 'design_pred_screen.csv']
    stage3 = run_dir / 'pipeline_output' / 'stage3'
    candidates += sorted(stage3.glob('*on_target_screen/on_target_screen.csv'))
    for path in candidates:
        if path.is_file():
            with path.open(encoding='utf-8-sig', newline='') as handle:
                output[str(path)] = summarize_records(list(csv.DictReader(handle)))
    for path in sorted(stage3.glob('00_stage3_plan/*manifest.json')):
        output[str(path)] = json.loads(path.read_text(encoding='utf-8-sig'))
    return {'run_directory': str(run_dir.resolve()), 'evidence': output, 'read_only': True}
