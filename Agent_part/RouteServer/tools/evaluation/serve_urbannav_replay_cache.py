#!/usr/bin/env python3
"""Serve UrbanNav replay caches from one shared process.

The normal sharded prediction runner loads ``poi_cache.jsonl`` and
``route_cache.jsonl`` inside every worker process.  For the final 15K split,
``route_cache.jsonl`` is large enough that many model workers can exhaust local
memory.  This service loads the replay cache once and exposes deterministic
cache-key lookups over localhost HTTP.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from typing import Any
from urllib import parse as urlparse


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcp_server.core.replay_map_backend import ReplayMapBackend  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poi-cache", required=True, type=Path, help="Frozen poi_cache.jsonl.")
    parser.add_argument("--route-cache", required=True, type=Path, help="Frozen route_cache.jsonl.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=18180, help="Bind port.")
    return parser.parse_args()


class ReplayCacheHandler(BaseHTTPRequestHandler):
    """HTTP handler backed by a process-global replay cache."""

    backend: ReplayMapBackend

    def do_GET(self) -> None:  # noqa: N802
        """Handle GET requests.

        Returns:
            None. A JSON response is written to the socket.
        """

        parsed = urlparse.urlparse(self.path)
        if parsed.path == "/healthz":
            self._write_json(
                {
                    "ok": True,
                    "poi_count": len(self.backend.poi_by_key),
                    "route_count": len(self.backend.route_by_key),
                }
            )
            return

        query = urlparse.parse_qs(parsed.query)
        cache_key = str((query.get("cache_key") or [""])[0])
        if not cache_key:
            self._write_json({"error": "missing_cache_key"}, status=400)
            return

        try:
            if parsed.path == "/poi":
                self._write_json(self.backend.search_destinations(cache_key=cache_key).to_dict())
                return
            if parsed.path == "/route":
                self._write_json(self.backend.plan_route(cache_key=cache_key).to_dict())
                return
        except KeyError:
            self._write_json({"error": "cache_key_not_found", "cache_key": cache_key}, status=404)
            return

        self._write_json({"error": "not_found", "path": parsed.path}, status=404)

    def log_message(self, fmt: str, *args: Any) -> None:
        """Suppress per-request access logs.

        Args:
            fmt: Log format.
            *args: Log arguments.

        Returns:
            None.
        """

        return

    def _write_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
        """Write one JSON response.

        Args:
            payload: JSON payload.
            status: HTTP status.

        Returns:
            None.
        """

        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    """Run the replay cache service.

    Returns:
        None.
    """

    args = parse_args()
    print(
        json.dumps(
            {
                "event": "replay_cache_loading",
                "poi_cache": str(args.poi_cache),
                "route_cache": str(args.route_cache),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    ReplayCacheHandler.backend = ReplayMapBackend(
        poi_cache_path=args.poi_cache,
        route_cache_path=args.route_cache,
    )
    server = ThreadingHTTPServer((args.host, int(args.port)), ReplayCacheHandler)
    print(
        json.dumps(
            {
                "event": "replay_cache_ready",
                "host": args.host,
                "port": int(args.port),
                "poi_count": len(ReplayCacheHandler.backend.poi_by_key),
                "route_count": len(ReplayCacheHandler.backend.route_by_key),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
