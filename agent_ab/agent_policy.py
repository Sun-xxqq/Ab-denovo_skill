"""Validated bounds for model-directed research; scientific gates are immutable."""
from copy import deepcopy
import math

DEFAULT_BUDGET = {
    'max_actions': 16, 'max_attempts': 3, 'max_backbones': 60,
    'max_sequences': 240, 'max_predictions': 480,
    'max_wall_seconds': 14400, 'command_timeout_seconds': 7200,
}

def positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError('%s must be a positive integer' % label)
    return value

def agent_policy(spec):
    raw = deepcopy(spec.get('agent', {}))
    if not isinstance(raw, dict):
        raise ValueError('agent must be an object')
    allowed = {'goal', 'budget', 'hotspot_sets', 'framework_choices', 'min_final_candidates'}
    if set(raw) - allowed:
        raise ValueError('unknown agent policy fields: %s' % sorted(set(raw) - allowed))
    raw.setdefault('goal', 'Find structurally supported candidates under the existing scientific gates.')
    if not isinstance(raw['goal'], str) or not raw['goal'].strip():
        raise ValueError('agent.goal must be non-empty text')
    budget = dict(DEFAULT_BUDGET)
    overrides = raw.get('budget', {})
    if not isinstance(overrides, dict) or set(overrides) - set(budget):
        raise ValueError('unknown agent budget field')
    budget.update(overrides)
    for key, value in budget.items():
        positive_int(value, 'agent.budget.%s' % key)
    raw['budget'] = budget
    original = list(spec['target']['hotspots'])
    sets = raw.get('hotspot_sets') or [original]
    if not isinstance(sets, list) or any(not isinstance(x, list) for x in sets):
        raise ValueError('agent.hotspot_sets must be a list of residue lists')
    chain = spec['target']['chain']
    mapping = spec['target'].get('canonical_chain_mapping', {})
    for group in sets:
        for i, residue in enumerate(group):
            if not isinstance(residue, str) or len(residue) < 2:
                raise ValueError('invalid allowed hotspot')
            group[i] = mapping.get(residue[0], residue[0]) + residue[1:]
            if not group[i].startswith(chain):
                raise ValueError('allowed hotspots must use the target chain')
    unique = [original]
    for group in sets:
        if group not in unique:
            unique.append(group)
    raw['hotspot_sets'] = unique
    choices = raw.get('framework_choices') or ([deepcopy(spec['framework'])] if spec.get('framework') else [])
    if not isinstance(choices, list) or any(not isinstance(x, dict) for x in choices):
        raise ValueError('agent.framework_choices must be framework objects')
    if spec.get('framework') and spec['framework'] not in choices:
        choices.insert(0, deepcopy(spec['framework']))
    raw['framework_choices'] = choices
    raw.setdefault('min_final_candidates', 1)
    positive_int(raw['min_final_candidates'], 'agent.min_final_candidates')
    return raw

def validate_action(name, arguments, schema):
    if not isinstance(arguments, dict):
        raise ValueError('tool arguments must be an object')
    properties = schema['properties']
    if set(arguments) - set(properties):
        raise ValueError('unknown arguments for %s: %s' % (name, sorted(set(arguments) - set(properties))))
    missing = set(schema.get('required', [])) - set(arguments)
    if missing:
        raise ValueError('missing arguments: %s' % sorted(missing))
    for key, value in arguments.items():
        rule = properties[key]
        kind = rule['type']
        if kind == 'integer' and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError('%s must be integer' % key)
        if kind == 'number' and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError('%s must be finite number' % key)
        if kind == 'string' and not isinstance(value, str):
            raise ValueError('%s must be text' % key)
        if kind == 'array':
            if not isinstance(value, list) or not value or any(not isinstance(x, str) for x in value):
                raise ValueError('%s must be a non-empty string list' % key)
            if len(value) != len(set(value)):
                raise ValueError('%s contains duplicates' % key)
        if 'minimum' in rule and value < rule['minimum']:
            raise ValueError('%s below minimum' % key)
        if 'maximum' in rule and value > rule['maximum']:
            raise ValueError('%s above maximum' % key)
        if 'enum' in rule and value not in rule['enum']:
            raise ValueError('%s is not an allowed value' % key)

def reserve_budget(state, policy, **cost):
    used = state.setdefault('used', {'backbones': 0, 'sequences': 0, 'predictions': 0})
    for kind, count in cost.items():
        if not isinstance(count, int) or count < 0:
            raise ValueError('invalid budget cost')
        if used[kind] + count > policy['budget']['max_' + kind]:
            raise ValueError('%s budget exceeded (%s + %s > %s)' % (
                kind, used[kind], count, policy['budget']['max_' + kind]))
    for kind, count in cost.items():
        used[kind] += count
