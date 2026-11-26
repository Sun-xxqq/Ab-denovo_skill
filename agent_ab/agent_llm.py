"""One model-selected function per decision; no deterministic fallback."""
import json
import os
from pathlib import Path
import urllib.error
import urllib.request
from urllib.parse import urlparse

from .util import read_json

class ModelUnavailable(RuntimeError):
    pass

class ToolModel:
    def __init__(self, path=None, environ=None):
        env = os.environ if environ is None else environ
        config = read_json(path) if path else {}
        self.base_url = env.get('AGENT_AB_LLM_BASE_URL') or config.get('base_url')
        self.model = env.get('AGENT_AB_LLM_MODEL') or config.get('model')
        key_env = config.get('api_key_env', 'AGENT_AB_LLM_API_KEY')
        self.key = env.get(key_env, '')
        self.timeout = config.get('timeout_seconds', 90)
        self.max_tokens = config.get('max_tokens', 1600)
        self.require_key = config.get('require_api_key', True)
        if not self.base_url or not self.model or (self.require_key and not self.key):
            raise ModelUnavailable('Configure AGENT_AB_LLM_BASE_URL, AGENT_AB_LLM_MODEL and AGENT_AB_LLM_API_KEY; or supply --agent-config. No GPU compute started.')
        parsed = urlparse(self.base_url)
        if parsed.scheme not in ('http', 'https') or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('base_url must be an HTTP(S) API base without credentials/query/fragment')
        for label, value in [('timeout_seconds', self.timeout), ('max_tokens', self.max_tokens)]:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError('%s must be positive integer' % label)

    def decide(self, messages, catalog):
        payload = {'model': self.model, 'messages': messages,
                   'tools': [{'type': 'function', 'function': {'name': t['name'], 'description': t['description'], 'parameters': t['inputSchema']}} for t in catalog],
                   'tool_choice': 'required', 'parallel_tool_calls': False, 'max_tokens': self.max_tokens}
        headers = {'Content-Type': 'application/json'}
        if self.key:
            headers['Authorization'] = 'Bearer ' + self.key
        request = urllib.request.Request(self.base_url.rstrip('/') + '/chat/completions',
                                         data=json.dumps(payload).encode('utf-8'), headers=headers, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read(4 * 1024 * 1024 + 1)
            if len(raw) > 4 * 1024 * 1024:
                raise RuntimeError('model response exceeded size bound')
            data = json.loads(raw)
        except urllib.error.HTTPError as exc:
            raise ModelUnavailable('model API returned HTTP %s; response body withheld to avoid exposing credentials' % exc.code) from None
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ModelUnavailable('model API connection failed (%s)' % type(exc).__name__) from None
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ModelUnavailable('model API returned malformed JSON') from None
        try:
            calls = data['choices'][0]['message']['tool_calls']
            if len(calls) != 1 or calls[0]['type'] != 'function':
                raise ValueError('exactly one function call required')
            call = calls[0]['function']
            arguments = json.loads(call['arguments'])
            if not isinstance(arguments, dict):
                raise ValueError('arguments must be an object')
            return {'tool': call['name'], 'arguments': arguments, 'usage': data.get('usage', {}), 'model': data.get('model', self.model)}
        except (KeyError, IndexError, TypeError, ValueError):
            raise ModelUnavailable('model response must contain exactly one valid function call; no automatic workflow fallback') from None
