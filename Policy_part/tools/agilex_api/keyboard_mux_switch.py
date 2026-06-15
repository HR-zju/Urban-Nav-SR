#!/usr/bin/env python3
"""Keyboard switch for AgileX mux velocity source.

Keys:
  a -> automatic, enable bridge and select /navflow_cmd_vel
  j -> joystick/manual, select /run_management/virtual_joy
  q -> quit
"""

import argparse
import base64
import fcntl
import hashlib
import json
import os
import select
import socket
import ssl
import struct
import sys
import termios
import time
import tty
import uuid
from urllib.parse import urlparse


DEFAULT_URI = "ws://192.168.1.102:9090"
DEFAULT_INTERFACE = "wlP1p1s0"
AUTO_TOPIC = "/navflow_cmd_vel"
AUTO_TYPE = "geometry_msgs/Twist"
MANUAL_TOPIC = "/run_management/virtual_joy"
ADD_SERVICE = "/mux_vel/add"
SELECT_SERVICE = "/mux_vel/select"
SELECTED_TOPIC = "/mux_vel/selected"
HEARTBEAT_INTERVAL = 2.0
DEFAULT_BRIDGE_ENABLE_FILE = "/tmp/navflow_cmd_vel_bridge_enabled"


def ipv4_for_interface(interface: str) -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        request = struct.pack("256s", interface[:15].encode("utf-8"))
        response = fcntl.ioctl(sock.fileno(), 0x8915, request)  # SIOCGIFADDR
        return socket.inet_ntoa(response[20:24])
    finally:
        sock.close()


def resolve_source_ip(interface: str | None, source_ip: str | None) -> str | None:
    if source_ip:
        return source_ip
    if interface:
        return ipv4_for_interface(interface)
    return None


class StdlibWebSocket:
    def __init__(
        self,
        uri: str,
        timeout: float = 5.0,
        source_ip: str | None = None,
        interface: str | None = None,
    ):
        parsed = urlparse(uri)
        if parsed.scheme not in ("ws", "wss"):
            raise ValueError(f"Unsupported WebSocket URI scheme: {parsed.scheme}")
        self.host = parsed.hostname
        if self.host is None:
            raise ValueError(f"Invalid WebSocket URI: {uri}")
        self.port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        self.path = parsed.path or "/"
        if parsed.query:
            self.path += f"?{parsed.query}"

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            if interface:
                sock.setsockopt(
                    socket.SOL_SOCKET,
                    socket.SO_BINDTODEVICE,
                    interface.encode("utf-8") + b"\0",
                )
            if source_ip:
                sock.bind((source_ip, 0))
            sock.connect((self.host, self.port))
        except Exception:
            sock.close()
            raise
        if parsed.scheme == "wss":
            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=self.host)
        sock.settimeout(timeout)
        self.sock = sock
        self._handshake()

    def _handshake(self) -> None:
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET {self.path} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        )
        self.sock.sendall(request.encode("ascii"))

        response = b""
        while b"\r\n\r\n" not in response:
            chunk = self.sock.recv(4096)
            if not chunk:
                break
            response += chunk

        header = response.decode("latin1", errors="replace")
        status_line = header.split("\r\n", 1)[0] if header else "empty response"
        if " 101 " not in status_line:
            raise RuntimeError(f"WebSocket handshake failed: {status_line}")

        accept_src = (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")
        expected_accept = base64.b64encode(hashlib.sha1(accept_src).digest()).decode("ascii")
        if f"Sec-WebSocket-Accept: {expected_accept}" not in header:
            raise RuntimeError("WebSocket handshake failed: invalid Sec-WebSocket-Accept")

    def send_json(self, data: dict) -> None:
        self.send_text(json.dumps(data))

    def send_text(self, text: str) -> None:
        self._send_frame(0x1, text.encode("utf-8"))

    def send_pong(self, payload: bytes = b"") -> None:
        self._send_frame(0xA, payload)

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        mask = os.urandom(4)
        first_byte = 0x80 | opcode
        length = len(payload)
        if length < 126:
            header = struct.pack("!BB", first_byte, 0x80 | length)
        elif length < 65536:
            header = struct.pack("!BBH", first_byte, 0x80 | 126, length)
        else:
            header = struct.pack("!BBQ", first_byte, 0x80 | 127, length)
        masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        self.sock.sendall(header + mask + masked)

    def recv_json(self, timeout: float = 5.0) -> dict:
        text = self.recv_text(timeout=timeout)
        return json.loads(text)

    def recv_text(self, timeout: float = 5.0) -> str:
        self.sock.settimeout(timeout)
        while True:
            opcode, payload = self._recv_frame()
            if opcode == 0x1:
                return payload.decode("utf-8")
            if opcode == 0x8:
                raise RuntimeError("WebSocket closed by server")
            if opcode == 0x9:
                self.send_pong(payload)
                continue
            if opcode == 0xA:
                continue

    def _recv_exact(self, size: int) -> bytes:
        chunks = []
        remaining = size
        while remaining:
            chunk = self.sock.recv(remaining)
            if not chunk:
                raise RuntimeError("WebSocket socket closed")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _recv_frame(self) -> tuple[int, bytes]:
        first, second = self._recv_exact(2)
        opcode = first & 0x0F
        masked = bool(second & 0x80)
        length = second & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._recv_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._recv_exact(8))[0]
        mask = self._recv_exact(4) if masked else b""
        payload = self._recv_exact(length) if length else b""
        if masked:
            payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        return opcode, payload

    def close(self) -> None:
        try:
            self._send_frame(0x8, b"")
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


class TerminalRawMode:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)


def call_service(ws: StdlibWebSocket, service: str, args: dict, request_id: str) -> dict:
    ws.send_json(
        {
            "op": "call_service",
            "service": service,
            "id": request_id,
            "args": args,
        }
    )
    while True:
        data = ws.recv_json(timeout=5.0)
        if data.get("op") == "service_response" and data.get("id") == request_id:
            return data


def select_source(ws: StdlibWebSocket, topic: str) -> None:
    response = call_service(ws, SELECT_SERVICE, {"topic": topic}, f"select:{topic}:{time.time():.3f}")
    result = response.get("result")
    values = response.get("values")
    if result:
        prev_topic = values.get("prev_topic") if isinstance(values, dict) else values
        print(f"selected {topic} (previous: {prev_topic})")
    else:
        print(f"failed to select {topic}: {values}")


def add_source(ws: StdlibWebSocket, topic: str) -> None:
    response = call_service(ws, ADD_SERVICE, {"topic": topic}, f"add:{topic}:{time.time():.3f}")
    if response.get("result"):
        print(f"mux source available: {topic}")
    else:
        values = response.get("values")
        print(f"mux add {topic} returned: {values}")


def publish_zero_twist(ws: StdlibWebSocket, topic: str = AUTO_TOPIC) -> None:
    ws.send_json(
        {
            "op": "advertise",
            "topic": topic,
            "type": AUTO_TYPE,
        }
    )
    ws.send_json(
        {
            "op": "publish",
            "topic": topic,
            "msg": {
                "linear": {"x": 0.0, "y": 0.0, "z": 0.0},
                "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
            },
        }
    )
    print(f"published zero Twist to {topic}")


def select_source_with_retry(
    ws: StdlibWebSocket,
    uri: str,
    topic: str,
    source_ip: str | None,
    interface: str | None,
) -> StdlibWebSocket:
    for attempt in range(2):
        try:
            select_source(ws, topic)
            return reconnect(uri, ws, quiet=True, source_ip=source_ip, interface=interface)
        except Exception as exc:  # noqa: BLE001
            print(f"select {topic} failed: {exc}")
            ws = reconnect(uri, ws, quiet=False, source_ip=source_ip, interface=interface)
            if attempt == 0:
                print("retrying select once...")
    return ws


def set_bridge_enabled(enable_file: str, enabled: bool) -> None:
    if enabled:
        with open(enable_file, "w", encoding="utf-8") as file:
            file.write("1\n")
        print(f"bridge enabled: {enable_file}")
        return

    try:
        os.remove(enable_file)
    except FileNotFoundError:
        pass
    print(f"bridge disabled: {enable_file}")


def switch_manual(
    ws: StdlibWebSocket,
    uri: str,
    enable_file: str,
    source_ip: str | None,
    interface: str | None,
) -> StdlibWebSocket:
    try:
        publish_zero_twist(ws)
    except Exception as exc:  # noqa: BLE001
        print(f"publish zero {AUTO_TOPIC} failed: {exc}")
        ws = reconnect(uri, ws, quiet=False, source_ip=source_ip, interface=interface)
    ws = select_source_with_retry(ws, uri, MANUAL_TOPIC, source_ip, interface)
    set_bridge_enabled(enable_file, False)
    return ws


def switch_auto(
    ws: StdlibWebSocket,
    uri: str,
    enable_file: str,
    source_ip: str | None,
    interface: str | None,
) -> StdlibWebSocket:
    try:
        add_source(ws, AUTO_TOPIC)
    except Exception as exc:  # noqa: BLE001
        print(f"mux add {AUTO_TOPIC} failed: {exc}")
        ws = reconnect(uri, ws, quiet=False, source_ip=source_ip, interface=interface)
    set_bridge_enabled(enable_file, True)
    return select_source_with_retry(ws, uri, AUTO_TOPIC, source_ip, interface)


def subscribe_selected(ws: StdlibWebSocket) -> None:
    ws.send_json({"op": "subscribe", "topic": SELECTED_TOPIC, "type": "std_msgs/String"})


def poll_selected(ws: StdlibWebSocket, timeout: float = 0.1) -> str | None:
    try:
        data = ws.recv_json(timeout=timeout)
    except (socket.timeout, TimeoutError):
        return None
    if data.get("op") == "publish" and data.get("topic") == SELECTED_TOPIC:
        msg = data.get("msg")
        if isinstance(msg, dict):
            return msg.get("data")
    return None


def connect(
    uri: str,
    source_ip: str | None = None,
    interface: str | None = None,
) -> StdlibWebSocket:
    ws = StdlibWebSocket(uri, source_ip=source_ip, interface=interface)
    subscribe_selected(ws)
    print(f"Connected: {uri}")
    if source_ip:
        print(f"WebSocket source binding: {source_ip}")
    if interface:
        print(f"WebSocket device binding: {interface}")
    return ws


def reconnect(
    uri: str,
    old_ws: StdlibWebSocket | None = None,
    quiet: bool = False,
    source_ip: str | None = None,
    interface: str | None = None,
) -> StdlibWebSocket:
    if old_ws is not None:
        old_ws.close()
    if not quiet:
        print("reconnecting...")
    time.sleep(0.3)
    return connect(uri, source_ip=source_ip, interface=interface)


def send_heartbeat(ws: StdlibWebSocket, client_id: str) -> None:
    ws.send_json(
        {
            "op": "ping",
            "id": client_id,
            "timeStamp": int(time.time() * 1000),
        }
    )


def print_help() -> None:
    print()
    print("AgileX mux keyboard switch")
    print(f"  a: AUTO   -> enable bridge, mux {AUTO_TOPIC}")
    print(f"  j: MANUAL -> disable bridge, mux {MANUAL_TOPIC}")
    print("  q: quit")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyboard switch for AgileX mux source.")
    parser.add_argument("--uri", default=DEFAULT_URI, help=f"AgileX WebSocket URI, default: {DEFAULT_URI}")
    parser.add_argument(
        "--interface",
        default=None,
        help=f"Bind WebSocket TCP source address to this interface's IPv4 address, e.g. {DEFAULT_INTERFACE}",
    )
    parser.add_argument(
        "--source-ip",
        default=None,
        help="Bind WebSocket TCP source address to this explicit local IPv4 address.",
    )
    parser.add_argument(
        "--bridge-enable-file",
        default=DEFAULT_BRIDGE_ENABLE_FILE,
        help=f"Bridge runtime enable file, default: {DEFAULT_BRIDGE_ENABLE_FILE}",
    )
    args = parser.parse_args()

    source_ip = resolve_source_ip(args.interface, args.source_ip)
    ws = connect(args.uri, source_ip=source_ip, interface=args.interface)
    client_id = f"keyboard_mux_{uuid.uuid4().hex[:8]}"
    last_heartbeat = 0.0
    try:
        print_help()
        print("defaulting to MANUAL mode...")
        ws = switch_manual(ws, args.uri, args.bridge_enable_file, source_ip, args.interface)

        with TerminalRawMode():
            while True:
                now = time.monotonic()
                if now - last_heartbeat >= HEARTBEAT_INTERVAL:
                    try:
                        send_heartbeat(ws, client_id)
                        last_heartbeat = now
                    except Exception as exc:  # noqa: BLE001
                        print(f"heartbeat failed: {exc}; reconnecting...")
                        ws.close()
                        time.sleep(1.0)
                        ws = connect(args.uri, source_ip=source_ip, interface=args.interface)
                        last_heartbeat = 0.0
                        continue

                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                if readable:
                    key = sys.stdin.read(1).lower()
                    if key == "a":
                        ws = switch_auto(ws, args.uri, args.bridge_enable_file, source_ip, args.interface)
                        last_heartbeat = 0.0
                        continue
                    if key == "j":
                        ws = switch_manual(ws, args.uri, args.bridge_enable_file, source_ip, args.interface)
                        last_heartbeat = 0.0
                        continue
                    elif key == "q":
                        print("quit")
                        break
                    elif key in ("h", "?"):
                        print_help()

                try:
                    selected = poll_selected(ws, timeout=0.01)
                except Exception as exc:  # noqa: BLE001
                    print(f"connection closed: {exc}; reconnecting...")
                    ws.close()
                    time.sleep(1.0)
                    ws = connect(args.uri, source_ip=source_ip, interface=args.interface)
                    last_heartbeat = 0.0
                    continue
                if selected:
                    print(f"current selected: {selected}")
    finally:
        ws.close()


if __name__ == "__main__":
    main()
