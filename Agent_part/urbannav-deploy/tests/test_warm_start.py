"""Only the new startup boundary, with no model or hardware dependencies."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from urbannav_deploy.serve_warm import warm_service


def test_startup_forward_once_and_query_state_cleared():
    events, calls = [], []

    class Service:
        scheduled_query = None
        query_cache = {}

        def predict(self, request):
            calls.append(request)
            self.scheduled_query = request['row']
            self.query_cache['condition'] = 'warmup-only'
            return {'request_id': request['request_id']}

    service = Service()
    request = {'request_id': 'startup-only', 'row': {'observation_timestamp_sec': 1.}}
    result = warm_service(service, request, lambda phase, **fields: events.append(phase))
    assert calls == [request]
    assert result['request_id'] == 'startup-only'
    assert service.scheduled_query is None
    assert service.query_cache == {}
    assert events == ['warming', 'warmed']


def test_failed_forward_propagates_without_ready_or_retry():
    events, calls = [], []

    class Service:
        def predict(self, request):
            calls.append(request)
            raise RuntimeError('native warmup failed')

    with pytest.raises(RuntimeError, match='native warmup failed'):
        warm_service(Service(), {'request_id': 'startup-only'},
                     lambda phase, **fields: events.append(phase))
    assert len(calls) == 1
    assert events == ['warming']
