"""Four research tools backed by existing Agent-Ab execution code."""
from copy import deepcopy
import csv
import json
import os
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

from .agent_observations import inspect_spec, read_screen, sequence_report, summarize_records
from .agent_policy import reserve_budget, validate_action
from .util import read_json, write_json, write_yaml, now_iso, select_gpu, disk_free_gb


def _schema(properties, required):
    return {'type': 'object', 'properties': dict(properties, rationale={
        'type': 'string', 'description': 'Brief evidence-based reason for this action; distinguish hypotheses from findings.'}),
        'required': required + ['rationale'], 'additionalProperties': False}

TOOLS = [
    {'name': 'inspect_target', 'description': 'Inspect target/framework PDB integrity and specified hotspot presence. Returns evidence and missing inputs; does not infer structure reliability or search external databases.',
     'inputSchema': _schema({'hotspot_set': {'type': 'integer', 'minimum': 0}, 'framework_choice': {'type': 'integer', 'minimum': 0}}, [])},
    {'name': 'generate_and_screen_backbones', 'description': 'Generate ONLY backbones with PPIFlow, then apply the existing geometry screen. Stops before sequence design. Can change seed, allowed hotspot set, or allowed framework; returns passing backbone IDs and rejection evidence. Generates an isolated attempt.',
     'inputSchema': _schema({'samples': {'type': 'integer', 'minimum': 1, 'maximum': 20000},
         'hotspot_set': {'type': 'integer', 'minimum': 0}, 'framework_choice': {'type': 'integer', 'minimum': 0},
         'seed': {'type': 'integer', 'minimum': 1, 'maximum': 2147483647}}, ['samples', 'seed'])},
    {'name': 'design_and_pack_sequences', 'description': 'Run existing AbMPNN/ProteinMPNN and FlowPacker ONLY on selected passing backbones. Each call creates a separate sequence batch. Preserves original backbone IDs and standard fixed-position handling. Returns exact duplication and sequence warnings; aggregation risk remains unmeasured.',
     'inputSchema': _schema({'attempt_id': {'type': 'string'}, 'backbone_ids': {'type': 'array', 'items': {'type': 'string'}},
         'sequences_per_backbone': {'type': 'integer', 'minimum': 1, 'maximum': 1000},
         'temperature': {'type': 'number', 'minimum': 0.001, 'maximum': 1.0}},
         ['attempt_id', 'backbone_ids', 'sequences_per_backbone', 'temperature'])},
    {'name': 'validate_complexes', 'description': 'Predict a packed sequence batch with Protenix and apply unchanged Stage 1 gates, computing ipSAE. When configured, also performs matching-seed off-target prediction and strict delta > 0. Does not run PartialFlow, FastRelax, or IgGM automatically. Returns computed pass counts, reasons and evidence; these are not experimental affinity.',
     'inputSchema': _schema({'attempt_id': {'type': 'string'}, 'batch_id': {'type': 'string'},
         'seed': {'type': 'integer', 'minimum': 1, 'maximum': 2147483647}}, ['attempt_id', 'batch_id', 'seed'])},
    {'name': 'finish', 'description': 'End this research campaign with achieved, stopped, or needs_input. achieved is accepted only if recorded validation results reach min_final_candidates. Give a concise explanation grounded in tool results.',
     'inputSchema': _schema({'outcome': {'type': 'string', 'enum': ['achieved', 'stopped', 'needs_input']}}, ['outcome'])},
]
TOOL_MAP = {x['name']: x for x in TOOLS}


class ResearchTools:
    def __init__(self, config, run_dir, state, policy, gpu='auto', clock=time.monotonic):
        self.config = config
        self.root = Path(run_dir)
        self.spec = read_json(self.root / 'spec.json')
        self.state = state
        self.policy = policy
        self.gpu = gpu
        self.clock = clock
        self.started = clock()
        self.work = self.root / '.agent_work' / 'agent'
        self.work.mkdir(parents=True, exist_ok=True)

    def save(self):
        write_json(self.work / 'state.json', self.state)

    def call(self, name, arguments):
        if name not in TOOL_MAP:
            raise ValueError('unknown tool: %s' % name)
        validate_action(name, arguments, TOOL_MAP[name]['inputSchema'])
        if not arguments['rationale'].strip():
            raise ValueError('rationale must not be empty')
        if name != 'inspect_target' and not self.state.get('inspection_ready') and name != 'finish':
            raise ValueError('inspect_target must establish valid inputs before compute')
        values = {k: v for k, v in arguments.items() if k != 'rationale'}
        return getattr(self, name)(**values)

    def _variant(self, hotspot_set=0, framework_choice=0):
        spec = deepcopy(self.spec)
        try:
            spec['target']['hotspots'] = list(self.policy['hotspot_sets'][hotspot_set])
            if spec.get('framework'):
                spec['framework'] = deepcopy(self.policy['framework_choices'][framework_choice])
        except IndexError:
            raise ValueError('hotspot_set/framework_choice is outside allowed choices')
        # Alternatives are validated through the same task schema before any compute.
        from .spec import validate_spec
        return validate_spec(spec)

    def inspect_target(self, hotspot_set=0, framework_choice=0):
        result = inspect_spec(self._variant(hotspot_set, framework_choice))
        self.state['inspection_ready'] = result['status'] == 'ready'
        self.state['inspection'] = result
        self.save()
        write_json(self.work / 'target_inspection.json', result)
        return result

    def _child_config(self):
        config = deepcopy(self.config)
        config['storage']['runs_root'] = str(self.work / 'attempts')
        return config

    def _new_child(self, spec, label):
        from .workflow import create_plan
        spec = deepcopy(spec)
        spec['name'] = self.root.name + '.' + label
        # Child names are unique; no changes to source campaign thresholds or old runs.
        return Path(create_plan(self._child_config(), spec=spec)['run_dir'])

    def _command(self, argv, cwd, label, environment=None):
        elapsed = self.state.get('elapsed_seconds', 0) + self.clock() - self.started
        remaining = self.policy['budget']['max_wall_seconds'] - elapsed
        if remaining <= 0:
            raise RuntimeError('campaign wall-time budget exhausted')
        if disk_free_gb(self.root) < float(self.config['storage']['min_free_gb_run']):
            raise RuntimeError('insufficient free disk for compute')
        timeout = min(remaining, self.policy['budget']['command_timeout_seconds'])
        env = dict(os.environ)
        gpu = select_gpu() if self.gpu == 'auto' else int(str(self.gpu).split(',')[0])
        env.update({'CUDA_VISIBLE_DEVICES': str(gpu), 'AGENT_AB_GPU_SELECTION_MODE': 'explicit',
                    'AGENT_AB_ABMPNN_GPUS': str(gpu), 'AGENT_AB_FLOWPACKER_GPUS': str(gpu),
                    'AGENT_AB_PARALLEL_GPUS': str(gpu), 'AGENT_AB_PPIFLOW_GPUS': str(gpu)})
        if environment:
            env.update(environment)
        log = self.work / 'logs' / (label + '.log')
        log.parent.mkdir(exist_ok=True)
        write_json(self.work / 'commands' / (label + '.json'), {'argv': argv, 'cwd': str(cwd), 'log': str(log), 'gpu': gpu, 'timeout_seconds': timeout})
        with log.open('a', encoding='utf-8') as handle:
            process = subprocess.Popen(argv, cwd=str(cwd), env=env, stdout=handle,
                stderr=subprocess.STDOUT, start_new_session=(os.name == 'posix'))
            self.state['active_process'] = {'pid': process.pid, 'host': socket.gethostname(), 'label': label}
            self.save()
            try:
                process.wait(timeout=timeout)
            except BaseException as failure:
                # Stop the complete Linux job group, including GPU model subprocesses.
                if os.name == 'posix':
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=10)
                    except (ProcessLookupError, subprocess.TimeoutExpired):
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                else:
                    process.kill()
                    process.wait()
                if isinstance(failure, subprocess.TimeoutExpired):
                    raise RuntimeError('compute command exceeded time budget; process group terminated') from None
                raise
            finally:
                self.state.pop('active_process', None)
                self.save()
        if process.returncode:
            raise RuntimeError('%s exited %s; evidence: %s' % (label, process.returncode, log))
        return str(log)

    def _ppiflow(self, child, enabled, label, seed=None, selection=None):
        from .ppiflow import build_task, build_steps, command
        spec = read_json(child / 'spec.json')
        required_workflow = {'backbone_generation': 'ppiflow', 'sequence_design': 'ppiflow_mpnn', 'sidechain_packing': 'ppiflow_flowpacker'}
        for key, value in required_workflow.items():
            if spec['workflow'][key] != value:
                raise ValueError('research tools currently require workflow.%s=%s' % (key, value))
        task = build_task(spec, child / 'pipeline_output')
        task['steps'] = {name: name in enabled for name in task['steps']}
        task_path = child / 'ppiflow' / ('task.agent.' + label + '.yaml')
        steps_path = child / 'ppiflow' / ('steps.agent.' + label + '.yaml')
        write_yaml(task_path, task)
        write_yaml(steps_path, build_steps(spec, self.config))
        argv = command(self.config, task_path, steps_path, stage=1)
        if seed is not None:
            argv += ['--agent-backbone-seed', str(seed)]
        env = {'AGENT_AB_BACKBONE_SELECTION_FILE': str(selection)} if selection else None
        return self._command(argv, child, label, env)

    def _attempt(self, attempt_id):
        try:
            return self.state['attempts'][attempt_id]
        except KeyError:
            raise ValueError('unknown attempt_id')

    def generate_and_screen_backbones(self, samples, seed, hotspot_set=0, framework_choice=0):
        attempts = self.state.setdefault('attempts', {})
        if len(attempts) >= self.policy['budget']['max_attempts']:
            raise ValueError('attempt budget exhausted')
        spec = self._variant(hotspot_set, framework_choice)
        inspected = inspect_spec(spec)
        if inspected['status'] != 'ready':
            raise ValueError('selected inputs need repair: %s' % inspected['issues'])
        if spec.get('framework'):
            from .backbone_screen import _framework_segments
            for chain in [spec['framework']['heavy_chain']] + ([spec['framework']['light_chain']] if spec['design_type'] == 'antibody' else []):
                _framework_segments(spec['framework']['structure'], chain)
        fingerprint = [samples, seed, hotspot_set, framework_choice]
        if any(x['parameters'] == fingerprint for x in attempts.values()):
            raise ValueError('identical generation already attempted; change strategy or seed')
        reserve_budget(self.state, self.policy, backbones=samples)
        spec['generation']['samples'] = samples
        attempt_id = 'attempt_%02d' % (len(attempts) + 1)
        child = self._new_child(spec, attempt_id)
        attempt = {'root': str(child), 'parameters': fingerprint, 'status': 'running', 'batches': {}, 'validations': []}
        attempts[attempt_id] = attempt
        self.save()
        try:
            log = self._ppiflow(child, {'PPIFlowStep'}, attempt_id + '.backbones', seed=seed)
            spec = read_json(child / 'spec.json')
            raw = child / 'pipeline_output' / 'stage1' / 'ppiflow_output'
            screen = child / 'pipeline_output' / 'stage1' / 'backbone_screen'
            from .backbone_outputs import organize_backbone_output
            organize_backbone_output(raw, child / 'pipeline_output' / 'stage1' / 'backbone_output')
            if spec['design_type'] in ('nanobody', 'antibody'):
                from .backbone_screen import screen_backbones, screen_scfv_backbones
                from scripts.ppiflow_pipeline_screened import _screening_hotspots
                screened_hotspots = _screening_hotspots({'specified_hotspots': ','.join(spec['target']['hotspots'])}, drop_nanobody_auxiliary=spec['design_type'] == 'nanobody')
                kwargs = dict(input_dir=raw, output_dir=screen, framework_pdb=spec['framework']['structure'],
                              hotspots=screened_hotspots, heavy_chain=spec['framework']['heavy_chain'])
                summary = screen_backbones(**kwargs) if spec['design_type'] == 'nanobody' else screen_scfv_backbones(light_chain=spec['framework']['light_chain'], **kwargs)
                with Path(summary['sorted_csv']).open(newline='', encoding='utf-8') as handle:
                    records = list(csv.DictReader(handle))
            else:
                # Binder screening here measures hotspot contact, not antibody CDR rules.
                from .backbone_screen import _pdb_residues, _hotspot_atoms, _distance
                records = []
                for pdb in sorted(raw.glob('*.pdb')):
                    design = [a for r in _pdb_residues(pdb, spec['generation']['binder_chain']) for a in r['atoms'] if a['element'] != 'H']
                    hotspots = _hotspot_atoms(pdb, spec['target']['hotspots'])
                    contacts = {h: min((_distance(a['coord'], b['coord']) for a in design for b in atoms), default=float('inf')) for h, atoms in hotspots.items()}
                    passed = bool(design) and any(x < 8.0 for x in contacts.values())
                    records.append({'backbone_id': pdb.stem, 'source_pdb': str(pdb.resolve()), 'filter_pass': passed,
                                    'rejection_reasons': '' if passed else 'no_binder_hotspot_contact_below_8A'})
                screen.mkdir(parents=True, exist_ok=True)
                write_json(screen / 'binder_geometry_records.json', records)
            report = summarize_records(records)
            passing = {r['backbone_id']: r['source_pdb'] for r in records if str(r['filter_pass']).lower() == 'true'}
            if report['evaluated'] != samples:
                raise RuntimeError('backbone count mismatch: requested %s, observed %s' % (samples, report['evaluated']))
            attempt.update(status='complete', passing_backbones=passing, report=report)
            result = dict(report, attempt_id=attempt_id, evidence_directory=str(screen), log=log,
                          hotspot_set=hotspot_set, framework_choice=framework_choice, seed=seed,
                          interpretation='Geometry/contact screening only; does not prove binding.')
            write_json(self.work / (attempt_id + '.backbone_report.json'), result)
            return result
        except Exception as exc:
            attempt.update(status='failed', error=str(exc))
            raise
        finally:
            self.save()

    def design_and_pack_sequences(self, attempt_id, backbone_ids, sequences_per_backbone, temperature):
        attempt = self._attempt(attempt_id)
        if attempt['status'] != 'complete':
            raise ValueError('backbone attempt did not complete')
        if set(backbone_ids) - set(attempt['passing_backbones']):
            raise ValueError('selected backbone IDs must come from passing geometry results')
        signature = [sorted(backbone_ids), sequences_per_backbone, temperature]
        if any(b['parameters'] == signature for b in attempt['batches'].values()):
            raise ValueError('identical sequence batch already attempted')
        count = len(backbone_ids) * sequences_per_backbone
        reserve_budget(self.state, self.policy, sequences=count)
        spec = read_json(Path(attempt['root']) / 'spec.json')
        spec['generation']['num_sequences_per_backbone'] = sequences_per_backbone
        spec['generation']['sampling_temperature'] = temperature
        spec['validation']['candidate_screening']['max_candidates_per_backbone'] = sequences_per_backbone
        batch_id = 'batch_%02d' % (len(attempt['batches']) + 1)
        child = self._new_child(spec, attempt_id + '.' + batch_id)
        native = child / 'pipeline_output' / 'stage1' / 'ppiflow_output'
        native.mkdir(parents=True, exist_ok=True)
        for identity in backbone_ids:
            source = Path(attempt['passing_backbones'][identity])
            shutil.copy2(source, native / source.name)
        selection = child / '.agent_work' / 'agent_selected_backbones.json'
        write_json(selection, {'backbone_ids': backbone_ids})
        batch = {'root': str(child), 'parameters': signature, 'status': 'running'}
        attempt['batches'][batch_id] = batch
        self.save()
        try:
            seq_step = 'MPNNStep_stage1' if spec['design_type'] == 'binder' else 'AbMPNNStep_stage1'
            log = self._ppiflow(child, {seq_step, 'FlowpackerStep_stage1'}, attempt_id + '.' + batch_id + '.sequences', selection=selection)
            packed = child / 'pipeline_output' / 'stage1' / 'flowpacker_output'
            paths = sorted(packed.glob('backbone.*/*.pdb')) or sorted((packed / 'run_1').glob('*.pdb'))
            if len(paths) != count:
                raise RuntimeError('packed design count mismatch: expected %s, observed %s' % (count, len(paths)))
            batch.update(status='complete', packed_designs=len(paths))
            fasta_root = child / 'pipeline_output' / 'stage1' / ('mpnn_output' if spec['design_type'] == 'binder' else 'abmpnn_output')
            return {'attempt_id': attempt_id, 'batch_id': batch_id, 'packed_designs': len(paths),
                    'backbone_ids': backbone_ids, 'sequence_report': sequence_report(fasta_root),
                    'evidence_directory': str(packed), 'log': log}
        except Exception as exc:
            batch.update(status='failed', error=str(exc))
            raise
        finally:
            self.save()

    def validate_complexes(self, attempt_id, batch_id, seed):
        attempt = self._attempt(attempt_id)
        if batch_id not in attempt['batches']:
            raise ValueError('unknown batch_id')
        batch = attempt['batches'][batch_id]
        if batch['status'] != 'complete':
            raise ValueError('sequence/packing batch must complete first')
        if any(v['batch_id'] == batch_id and v['seed'] == seed for v in attempt['validations']):
            raise ValueError('this batch/seed has already been validated or attempted')
        spec = read_json(Path(batch['root']) / 'spec.json')
        if spec['workflow']['complex_prediction'] != 'protenix':
            raise ValueError('research validation currently supports Protenix')
        from .orchestrator import _offtarget_config
        off = _offtarget_config(self.config, spec)
        off_enabled = bool(off.get('enabled', False))
        reserve_budget(self.state, self.policy, predictions=batch['packed_designs'] * (2 if off_enabled else 1))
        spec['validation']['candidate_screening']['off_target_screening'] = dict(off, seed=seed)
        check_id = 'validation_%02d' % (len(attempt['validations']) + 1)
        child = self._new_child(spec, attempt_id + '.' + batch_id + '.' + check_id)
        source_stage = Path(batch['root']) / 'pipeline_output' / 'stage1'
        stage = child / 'pipeline_output' / 'stage1'
        stage.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_stage / 'flowpacker_output', stage / 'flowpacker_output', symlinks=False)
        fixed = source_stage / 'mpnn_pdbs' / 'fixed_positions.csv'
        if fixed.is_file():
            (stage / 'mpnn_pdbs').mkdir(exist_ok=True)
            shutil.copy2(fixed, stage / 'mpnn_pdbs' / fixed.name)
        check = {'batch_id': batch_id, 'seed': seed, 'root': str(child), 'status': 'running'}
        attempt['validations'].append(check)
        self.save()
        try:
            from .adapters import build_command
            from .orchestrator import _automatic_inputs, _stage_tool_output, _run_offtarget_screen
            from .recovery import RecoveryState
            inputs = _automatic_inputs('protenix', child, spec)
            if not inputs:
                raise ValueError('no packed structures available for validation')
            inputs['seed'] = seed
            prediction = _stage_tool_output(child, 'stage1', 'protenix')
            protenix = self.config['tools']['protenix']
            env = {'PROTENIX_ROOT_DIR': protenix.get('root_dir', str(Path(protenix['data_dir']).parent))}
            log = self._command(build_command('protenix', self.config, inputs, prediction, spec), protenix['path'], attempt_id + '.' + check_id + '.prediction', env)
            score_inputs = _automatic_inputs('ipsae', child, spec)
            if not score_inputs:
                raise RuntimeError('prediction artifacts not available for screening')
            score_inputs['skip_ipsae'] = False
            score_inputs['delete_rejected'] = False
            results = _stage_tool_output(child, 'stage1', 'ipsae')
            self._command(build_command('ipsae', self.config, score_inputs, results, spec), self.config['project_root'], attempt_id + '.' + check_id + '.screen', env)
            records = read_screen(results)
            report = summarize_records(records)
            if len(records) != batch['packed_designs']:
                raise RuntimeError('validation coverage mismatch')
            final_count = report['passed']
            off_report = {'enabled': off_enabled, 'evaluated': 0}
            if off_enabled and final_count:
                # Route all legacy helper subprocesses through the same budgeted runner.
                # The helper already preserves antibody sequence and delta semantics.
                calls = [0]
                def bounded_runner(state, stage_name, argv, cwd, environment, log_path, retry_failed=False):
                    calls[0] += 1
                    self._command(argv, cwd, attempt_id + '.' + check_id + '.offtarget_%s' % calls[0], environment)
                    state.update(stage_name, 'complete', returncode=0)
                off_result = _run_offtarget_screen(self.config, child, spec, RecoveryState(child),
                    select_gpu() if self.gpu == 'auto' else int(str(self.gpu).split(',')[0]),
                    stage='stage1', command_runner=bounded_runner)
                off_report = dict(off_result['summary'], enabled=True, evidence_directory=off_result['output'])
                final_count = int(off_report['passed'])
            check.update(status='complete', final_candidates=final_count, on_target=report, off_target=off_report)
            result = {'attempt_id': attempt_id, 'batch_id': batch_id, 'validation_id': check_id, 'seed': seed,
                      'on_target': report, 'off_target': off_report, 'final_candidates': final_count,
                      'evidence_directory': str(stage), 'log': log,
                      'interpretation': 'Computational screening only; no measured affinity or selectivity.'}
            write_json(self.work / (attempt_id + '.' + check_id + '.validation_report.json'), result)
            return result
        except Exception as exc:
            check.update(status='failed', error=str(exc))
            raise
        finally:
            self.save()

    def finish(self, outcome):
        best = max([v.get('final_candidates', 0) for a in self.state.get('attempts', {}).values()
                    for v in a.get('validations', []) if v['status'] == 'complete'] or [0])
        if outcome == 'achieved' and best < self.policy['min_final_candidates']:
            raise ValueError('cannot claim achieved without sufficient validated candidates')
        self.state['status'] = outcome
        self.save()
        return {'status': outcome, 'best_validated_candidate_count': best, 'evidence_directory': str(self.work)}
