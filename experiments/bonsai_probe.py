"""API checks for Bonsai's embedded draft head and long-context recall."""
import json
import time
import uuid

from lib import probes


def draft_counts(response):
    timings = response.get('timings', {})
    drafted = timings.get('draft_n', 0)
    accepted = timings.get('draft_n_accepted', 0)
    if not 0 < accepted <= drafted:
        raise ValueError(f'Expected active MTP with accepted tokens, got {timings}')
    return {'drafted': drafted, 'accepted': accepted}


def validate_api(base, alias, spec):
    tool = {'type': 'function', 'function': {
        'name': 'lookup_temperature', 'description': 'Look up the temperature for a city.',
        'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}},
                       'required': ['city']}}}
    common = {'model': alias, 'temperature': 0, 'reasoning_effort': 'none'}
    response = probes.request(base, '/v1/chat/completions', {
        **common, 'max_tokens': 512, 'tools': [tool], 'messages': [
            {'role': 'user', 'content': 'Use lookup_temperature for Athens.'}]})
    calls = response['choices'][0]['message'].get('tool_calls') or []
    if (len(calls) != 1 or calls[0]['function']['name'] != 'lookup_temperature'
            or 'athens' not in json.loads(calls[0]['function']['arguments']).get('city', '').lower()):
        raise ValueError(f'Unexpected Bonsai tool call: {calls}')
    result = {'tool_call': response}
    if spec == 'draft-mtp':
        response = probes.request(base, '/v1/chat/completions', {
            **common, 'max_tokens': 256, 'messages': [{'role': 'user', 'content':
                'Count from 1 through 50, using digits separated by commas.'}]})
        result['mtp'] = draft_counts(response)
    return result


def validate_context(base, backend, profile):
    """Exercise the configured context through the public proxy with output space reserved."""
    available = profile['context'] - profile['opencode']['output_tokens']
    if available <= 4096:
        raise ValueError('Long-context probe needs more room for its prompt')
    early, late = (uuid.uuid4().hex[:8].upper() for _ in range(2))
    filler = '\nThe landscape contains roads, rivers, fields, and hills.'

    def count(content):
        return len(probes.request(backend, '/tokenize', {'content': content})['tokens'])

    # Budget for template overhead and measure the completed prompt at the API.
    repeats = (available - 256) * 128 // count(filler * 128)
    prompt = ('Remember the two audit codes in this document.\n'
              + filler * (repeats // 20) + f'\nEARLY audit code: {early}\n'
              + filler * (repeats - 2 * (repeats // 20)) + f'\nLATE audit code: {late}\n'
              + filler * (repeats // 20)
              + '\nReturn the EARLY and LATE audit codes in order, followed by '
                'the integers from 1 to 30 separated by commas.')
    started = time.monotonic()
    response = probes.request(base, '/v1/chat/completions', {
        'model': profile['alias'], 'messages': [{'role': 'user', 'content': prompt}],
        'reasoning_effort': 'xhigh', 'temperature': 0,
        'max_tokens': profile['opencode']['output_tokens']})
    choice = response['choices'][0]
    content = (choice['message']['content'] or '').upper()
    usage = response['usage']
    if not available - 1024 <= usage['prompt_tokens'] <= available:
        raise ValueError(f'Expected prompt near {available} tokens, got {usage}')
    if choice['finish_reason'] != 'stop' or early not in content or late not in content:
        raise ValueError(f'Long-context recall failed for {early} and {late}: {content[:500]}')
    return {'elapsed_s': round(time.monotonic() - started, 2), 'usage': usage,
            'timings': response['timings'], 'mtp': draft_counts(response),
            'expected_codes': [early, late], 'answer': choice['message']['content']}
