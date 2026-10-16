"""Small replaceable inference transport; never retries an uncertain prediction."""
from __future__ import annotations
import base64
import hashlib
import json
import re
from typing import Protocol
from urllib.request import Request, urlopen


def frame_reference(data: bytes, frame_id: str) -> str:
    """Bind immutable compressed bytes to an acquisition identity.

    Args:
        data: Encoded camera/GUI image bytes.
        frame_id: Source frame ID, for example its integer acquisition timestamp.
    Returns:
        Content-addressed reference used in the observation's images list.
    Raises:
        ValueError: Frame ID contains unsupported characters.
    """
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', frame_id):
        raise ValueError('frame_id must contain letters, digits, dots, underscores or hyphens')
    return f'frame:sha256:{hashlib.sha256(data).hexdigest()}:{frame_id}'


class InferenceClient(Protocol):
    """Model transport boundary; implementations return the same prediction contract."""
    def health(self) -> dict:
        """Return model readiness and immutable identity without inference."""
        ...

    def predict(self, request_id: str, row: dict, *, frames: dict[str, bytes] | None = None) -> dict:
        """Execute one uniquely identified observation, with no automatic retries."""
        ...


class HttpInferenceClient:
    """Connect replay/live acquisition to a loopback or trusted-tunnel model service.

    Args:
        url: Base HTTP address of the existing r10 endpoint.
        timeout: Per-request seconds; timeout does not imply remote work stopped.
    """
    def __init__(self, url: str, timeout: float = 120):
        self.url = url.rstrip('/')
        self.timeout = timeout

    def health(self) -> dict:
        """Return live service readiness; raises on network/HTTP errors."""
        with urlopen(self.url + '/healthz', timeout=self.timeout) as response:
            return json.load(response)

    def predict(self, request_id: str, row: dict, *, frames: dict[str, bytes] | None = None) -> dict:
        """POST one observation with optional new compressed image bytes.

        Args:
            request_id: Unique inference request identity.
            row: Target-free observation with source timestamps and image references.
            frames: New frame references mapped to compressed bytes; resident frames
                can be omitted. Missing resident images surface as the server error.
        Returns:
            Model prediction, without automatic retry on HTTP/network failure.
        """
        payload = {'request_id': request_id, 'row': row}
        if frames is not None:
            payload['frames'] = {key: base64.b64encode(value).decode('ascii') for key, value in frames.items()}
        data = json.dumps(payload).encode()
        request = Request(self.url + '/v1/predict', data=data,
                          headers={'Content-Type': 'application/json'}, method='POST')
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def retrieve(self, request_id: str) -> dict:
        """Read a durable prior response after an interrupted client, without POST."""
        from urllib.parse import quote
        with urlopen(self.url + '/v1/requests/' + quote(request_id, safe=''),
                     timeout=self.timeout) as response:
            return json.load(response)
