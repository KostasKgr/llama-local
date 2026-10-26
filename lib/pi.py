"""Register pinned profiles in PI's models.json without changing client defaults."""
import copy
import difflib
import json
import os
from pathlib import Path
from . import core, jsonc
from .opencode import write_config

PROVIDER = 'llama-local'


def default_config():
    return str(Path(os.environ.get('PI_CODING_AGENT_DIR', '~/.pi/agent')) / 'models.json')


def model_config(profile, output_tokens=None):
    metadata = profile['opencode']
    output = metadata['output_tokens'] if output_tokens is None else output_tokens
    if type(output) is not int or not 0 < output <= profile['context']:
        raise ValueError('Output tokens must be positive and fit within context')
    reasoning = metadata.get('reasoning', False)
    model = {
        'id': profile['alias'], 'name': metadata['name'],
        'input': metadata['modalities']['input'],
        'contextWindow': profile['context'], 'maxTokens': output,
        'reasoning': reasoning,
        'cost': {'input': 0, 'output': 0, 'cacheRead': 0, 'cacheWrite': 0},
        'compat': {
            'supportsStore': False, 'supportsDeveloperRole': False,
            'supportsReasoningEffort': False, 'maxTokensField': 'max_tokens',
            'thinkingFormat': 'chat-template',
            'chatTemplateKwargs': {'enable_thinking': {'$var': 'thinking.enabled'}},
        },
    }
    if reasoning:
        # The endpoint has a boolean switch, not measured effort presets.
        model['thinkingLevelMap'] = {
            'minimal': None, 'low': None, 'medium': None, 'high': 'high',
            'xhigh': None, 'max': None,
        }
    return model


def merge(source, profile, output_tokens=None, update=False):
    # PI reads strict JSON. The span parser also rejects duplicate object keys.
    existing = jsonc.parse(source).value
    json.loads(source)
    providers = existing.get('providers', {})
    if not isinstance(providers, dict):
        raise ValueError('Expected object at providers')
    provider = providers.get(PROVIDER, {})
    if not isinstance(provider, dict):
        raise ValueError(f'Expected object at providers.{PROVIDER}')
    models = copy.deepcopy(provider.get('models', []))
    if not isinstance(models, list):
        raise ValueError('Expected models array')
    seen = set()
    for model in models:
        if not isinstance(model, dict) or not isinstance(model.get('id'), str) or not model['id']:
            raise ValueError('Every model needs a nonempty string id')
        if model['id'] in seen:
            raise ValueError(f'Duplicate model id: {model["id"]}')
        seen.add(model['id'])
    desired = model_config(profile, output_tokens)
    current = next((m for m in models if m['id'] == profile['alias']), None)
    changes, conflicts = [], []

    def combine(current, wanted, path):
        for key, value in wanted.items():
            target = path + [key]
            if key not in current:
                current[key] = copy.deepcopy(value)
            elif isinstance(value, dict):
                if not isinstance(current[key], dict):
                    raise ValueError(f'Expected object at {".".join(target)}')
                combine(current[key], value, target)
            elif current[key] != value:
                conflicts.append('.'.join(target))
                current[key] = copy.deepcopy(value)

    if current is None:
        models.append(desired)
    else:
        # Per-model transport overrides take precedence over provider settings.
        # Refuse to leave the selected alias on a different endpoint or protocol.
        if 'api' in current:
            desired['api'] = 'openai-completions'
        if 'baseUrl' in current:
            desired['baseUrl'] = f'http://127.0.0.1:{profile["port"]}/v1'
        combine(current, desired, ['models', profile['alias']])
    wanted = {
        'baseUrl': f'http://127.0.0.1:{profile["port"]}/v1',
        'api': 'openai-completions',
    }
    for key, value in wanted.items():
        if key in provider and provider[key] != value:
            conflicts.append(key)
        if provider.get(key) != value:
            changes.append((key, value))
    # Preserve custom authentication, including environment references.
    if 'apiKey' not in provider:
        changes.append(('apiKey', 'llama-local'))
    if provider.get('models') != models:
        changes.append(('models', models))
    if conflicts and not update:
        raise ValueError('Conflicting settings: ' + ', '.join(conflicts) + '; use --update to replace those fields')
    for key, value in changes:
        source = jsonc.set_value(source, ['providers', PROVIDER, key], value)
    return source


def add(name, config, dry_run=False, update=False, context=None, port=None, output_tokens=None):
    profile = core.load(name, {'context': context, 'port': port})
    path = Path(config).expanduser().resolve()
    original = path.read_bytes() if path.exists() else None
    source = original.decode('utf-8') if original is not None else '{}\n'
    updated = merge(source, profile, output_tokens, update)
    if updated == source:
        print(f'Already registered: {PROVIDER}/{profile["alias"]}')
        return
    print(''.join(difflib.unified_diff(
        source.splitlines(keepends=True), updated.splitlines(keepends=True),
        fromfile=str(path), tofile=str(path))), end='')
    if dry_run:
        return
    backup = write_config(path, original, updated)
    print(f'Registered: {PROVIDER}/{profile["alias"]}')
    if backup:
        print(f'Backup: {backup}')
