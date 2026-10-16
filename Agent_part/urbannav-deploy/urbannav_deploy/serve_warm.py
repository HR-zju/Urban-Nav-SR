"""Expose the frozen r10 HTTP backend only after its first forward completes.

This entrypoint belongs to the deployment agent. Model implementations remain in
the maintained external ms-swift environment; no model source is copied here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import threading
import time


def warm_service(service, request, record):
    """Run one explicit startup input before exposing the HTTP listener.

    Args:
        service: Loaded r10 backend with predict and scheduled_query state.
        request: Recorded valid startup request with a dedicated unique ID.
        record: Lifecycle event writer.
    Returns:
        Warmup prediction, persisted by the underlying model service.
    Raises:
        Exception: Original inference failure; the caller must not open its port.
    """
    record('warming', request_id=request['request_id'])
    started = time.monotonic()
    prediction = service.predict(request)
    # 预热输入只负责首次执行初始化。清除其调度状态，让第一个业务观测建立自己的 query。
    # 图像/视觉缓存仍按原有内容身份生效，不改变模型合同或误用预热轨迹进行控制。
    service.scheduled_query = None
    if service.query_cache is not None:
        service.query_cache.clear()
    record('warmed', request_id=request['request_id'], seconds=time.monotonic() - started)
    return prediction


def main():
    """Load, warm once, then serve on loopback; preserve original failures on exit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--warmup-request', required=True, type=Path)
    parser.add_argument('--lifecycle', required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    request = json.loads(args.warmup_request.read_text())

    from swift.sidewalk_nav.cli.serve_policy import make_server
    from swift.sidewalk_nav.runtime.inference_service import R10InferenceService

    with args.lifecycle.open('x', buffering=1) as events:
        def record(phase, **fields):
            events.write(json.dumps({'phase': phase, 'unix_sec': time.time(), **fields}) + '\n')

        service = server = None
        try:
            record('loading')
            service = R10InferenceService(config)
            if service.vision_cache is not None:
                from .per_image_vision import PerImageVision
                service.vision_cache.visual = PerImageVision(service.vision_cache.visual).eval()
                record('vision_policy', batch_unit='one_image', reason='cache_history_independent_encoding')
            warm_service(service, request, record)
            server = make_server(service, int(config['port']))

            def stop(signum, _frame):
                record('stop_signal', signal=signum)
                threading.Thread(target=server.shutdown, daemon=True).start()

            for signum in (signal.SIGTERM, signal.SIGINT):
                signal.signal(signum, stop)
            record('ready', port=server.server_port, **service.health())
            server.serve_forever(poll_interval=.2)
        except BaseException as exc:
            record('failed', error=repr(exc))
            raise
        finally:
            if server is not None:
                server.server_close()
            if service is not None:
                service.close()
            record('closed')


if __name__ == '__main__':
    main()
