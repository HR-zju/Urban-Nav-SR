"""Strict JSON configuration and explicit, trusted adapter factories."""
from __future__ import annotations

import importlib
import math


def validate_config(config):
    """Validate a deployment config without importing plugins or contacting hardware.

    Args:
        config: JSON-compatible mapping supplied by the deployment owner.
    Returns:
        The validated mapping, unchanged.
    Raises:
        ValueError: Unknown keys, invalid periods or incomplete live bindings.
    """
    if not isinstance(config, dict):
        raise ValueError('configuration must be an object')
    allowed = {'mode', 'source', 'inference', 'controller', 'ownership', 'control_hz',
               'max_plan_age_sec', 'close_timeout_sec'}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f'unknown configuration keys: {sorted(unknown)}')
    if config.get('mode', 'shadow') not in ('shadow', 'live'):
        raise ValueError('mode must be shadow or live')
    for key, default in [('control_hz', 20.), ('max_plan_age_sec', .75), ('close_timeout_sec', 5.)]:
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{key} must be a finite number')
        if value < 0 or (key != 'close_timeout_sec' and value == 0):
            raise ValueError(f'{key} out of range')
    required = ['source', 'inference']
    if config.get('mode', 'shadow') == 'live':
        required += ['controller', 'ownership']
    elif 'controller' in config or 'ownership' in config:
        raise ValueError('shadow mode must not bind physical controller/ownership factories')
    for key in required:
        spec = config.get(key)
        if not isinstance(spec, dict) or set(spec) - {'factory', 'options'}:
            raise ValueError(f'{key} needs factory and optional options')
        factory = spec.get('factory', '')
        if not isinstance(factory, str) or factory.count(':') != 1:
            raise ValueError(f'{key}.factory must be module:callable')
        if not all(part and all(segment.isidentifier() for segment in part.split('.'))
                   for part in factory.split(':')):
            raise ValueError(f'{key}.factory has an invalid import path')
        if not isinstance(spec.get('options', {}), dict):
            raise ValueError(f'{key}.options must be an object')
    return config


def load_adapter(spec, record):
    """Construct one explicitly configured trusted plugin.

    Args:
        spec: Validated factory specification; factory receives options and record.
        record: Shared nonblocking telemetry callback.
    Returns:
        Plugin object implementing the selected source/transport/controller boundary.
    Raises:
        ImportError: Configured adapter module is unavailable.
        Exception: Original plugin construction failure.
    """
    module, attribute = spec['factory'].split(':')
    factory = importlib.import_module(module)
    for component in attribute.split('.'):
        factory = getattr(factory, component)
    return factory(options=dict(spec.get('options', {})), record=record)


def http_client_factory(*, options, record):
    """Build the existing HTTP transport without contacting the server.

    Args:
        options: HttpInferenceClient keyword arguments (url and optional timeout).
        record: Shared telemetry callback, unused by this no-retry transport.
    Returns:
        HTTP inference client, health checked later by the session.
    """
    from .transport import HttpInferenceClient
    return HttpInferenceClient(**options)
