#!/usr/bin/env python3
import asyncio
import base64
import glob
import inspect
import json
import math
import os
import socket
import threading
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, NavSatStatus

try:
    import serial
except ImportError:  # pragma: no cover - handled at runtime on the robot
    serial = None

try:
    import websockets
except ModuleNotFoundError:
    websockets = None


class GpsNtripNode(Node):
    def __init__(self):
        super().__init__('gps_ntrip_node')

        self.declare_parameter('port', 'auto')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('ask_gga', 'GPGGA 1\r\n')
        self.declare_parameter('frame_id', 'gps')
        self.declare_parameter('fix_topic', '/gps/fix')
        self.declare_parameter('ntrip_server', '120.253.239.161')
        self.declare_parameter('ntrip_port', 8002)
        self.declare_parameter('ntrip_username', 'cqnf3478')
        self.declare_parameter('ntrip_password', 'sdh83667')
        self.declare_parameter('ntrip_mountpoint', 'RTCM33_GRCEJ')
        self.declare_parameter('serial_timeout', 0.0)
        self.declare_parameter('connect_ntrip_on_start', True)
        self.declare_parameter('use_agilex_time', True)
        self.declare_parameter('agilex_time_uri', 'ws://192.168.1.102:9090')
        self.declare_parameter('agilex_time_topic', '/system_timestamp')

        self.port = self._resolve_serial_port(str(self.get_parameter('port').value))
        self.baudrate = int(self.get_parameter('baudrate').value)
        self.ask_gga = self.get_parameter('ask_gga').value
        self.frame_id = self.get_parameter('frame_id').value
        self.use_agilex_time = bool(self.get_parameter('use_agilex_time').value)
        self.agilex_time_uri = str(self.get_parameter('agilex_time_uri').value)
        self.agilex_time_topic = str(self.get_parameter('agilex_time_topic').value)

        self.serial_port = None
        self.ntrip_socket = None
        self.ntrip_connected = False
        self.gga_rx_flag = False
        self.gga_rx_data = ''
        self.serial_buffer = bytearray()
        self.last_decode_warn = 0.0
        self.agilex_stamp_lock = threading.Lock()
        self.latest_agilex_stamp = None
        self.agilex_time_stop = threading.Event()
        self.agilex_time_thread = None
        self.agilex_time_receive_count = 0

        self.fix_pub = self.create_publisher(
            NavSatFix,
            self.get_parameter('fix_topic').value,
            10,
        )

        self._start_agilex_time_thread()
        self._open_serial()
        if self.get_parameter('connect_ntrip_on_start').value:
            self._connect_ntrip()

        self.serial_timer = self.create_timer(0.01, self._process_io)
        self.gps_gga_timer = self.create_timer(3.0, self._send_gpgga_to_gps)
        self.ntrip_gga_timer = self.create_timer(1.0, self._send_gpgga_to_ntrip)

    def destroy_node(self):
        self.agilex_time_stop.set()
        if self.agilex_time_thread is not None and self.agilex_time_thread.is_alive():
            self.agilex_time_thread.join(timeout=2.0)
        if self.serial_port is not None and self.serial_port.is_open:
            self.serial_port.close()
        if self.ntrip_socket is not None:
            self.ntrip_socket.close()
        super().destroy_node()

    def _start_agilex_time_thread(self):
        if not self.use_agilex_time:
            return
        if websockets is None:
            self.get_logger().warn(
                'websockets is not available; /gps/fix will use the ROS2 node clock.'
            )
            return
        self.agilex_time_thread = threading.Thread(
            target=self._run_agilex_time_subscriber,
            daemon=True,
        )
        self.agilex_time_thread.start()

    def _run_agilex_time_subscriber(self):
        while not self.agilex_time_stop.is_set():
            try:
                asyncio.run(self._subscribe_agilex_system_timestamp())
            except Exception as exc:  # noqa: BLE001
                if not self.agilex_time_stop.is_set():
                    self.get_logger().warn(f'AgileX system timestamp subscriber error: {exc}')
            self.agilex_time_stop.wait(2.0)

    async def _subscribe_agilex_system_timestamp(self):
        request = {
            'op': 'subscribe',
            'topic': self.agilex_time_topic,
        }
        connect_kwargs = {}
        if 'proxy' in inspect.signature(websockets.connect).parameters:
            connect_kwargs['proxy'] = None

        async with websockets.connect(self.agilex_time_uri, ping_interval=None, **connect_kwargs) as websocket:
            await websocket.send(json.dumps(request))
            self.get_logger().info(
                f'Subscribed AgileX system timestamp: {self.agilex_time_topic} '
                f'from {self.agilex_time_uri}'
            )
            while not self.agilex_time_stop.is_set():
                try:
                    raw = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if data.get('op') != 'publish' or data.get('topic') != self.agilex_time_topic:
                    continue

                msg = data.get('msg') if isinstance(data.get('msg'), dict) else {}
                stamp = msg.get('stamp') if isinstance(msg.get('stamp'), dict) else {}
                sec = stamp.get('secs', stamp.get('sec'))
                nanosec = stamp.get('nsecs', stamp.get('nanosec'))
                if sec is None or nanosec is None:
                    continue
                try:
                    sec = int(sec)
                    nanosec = int(nanosec)
                except (TypeError, ValueError):
                    continue
                if nanosec < 0:
                    continue
                sec += nanosec // 1000000000
                nanosec = nanosec % 1000000000
                with self.agilex_stamp_lock:
                    self.latest_agilex_stamp = (sec, nanosec)
                    self.agilex_time_receive_count += 1
                    count = self.agilex_time_receive_count
                if count == 1 or count % 100 == 0:
                    self.get_logger().info(
                        f'Received AgileX system timestamp #{count}: '
                        f'sec={sec}, nanosec={nanosec}'
                    )

    def _gps_header_stamp(self):
        if self.use_agilex_time:
            with self.agilex_stamp_lock:
                stamp = self.latest_agilex_stamp
            if stamp is not None:
                msg = self.get_clock().now().to_msg()
                msg.sec = int(stamp[0])
                msg.nanosec = int(stamp[1])
                return msg
        return self.get_clock().now().to_msg()

    def _open_serial(self):
        if serial is None:
            raise RuntimeError(
                'pyserial is required to run gps_ntrip_node; install it with `pip install pyserial`.'
            )

        timeout = float(self.get_parameter('serial_timeout').value)
        self.serial_port = serial.Serial(self.port, self.baudrate, timeout=timeout)
        self.serial_port.write(b'UNLOG\r\n')
        time.sleep(1.0)
        self.serial_port.write(self.ask_gga.encode('utf-8'))
        self.get_logger().info(f'Opened GPS serial port {self.port} at {self.baudrate}')

    def _resolve_serial_port(self, requested_port: str) -> str:
        requested_port = (requested_port or '').strip()
        if requested_port and requested_port.lower() not in ('auto', 'none'):
            return requested_port

        candidates = []
        for pattern in (
            '/dev/serial/by-id/*',
            '/dev/serial/by-path/*',
            '/dev/ttyUSB*',
            '/dev/ttyACM*',
        ):
            for path in sorted(glob.glob(pattern)):
                if not os.path.exists(path):
                    continue
                real_path = os.path.realpath(path)
                if any(real_path == item[1] for item in candidates):
                    continue
                candidates.append((path, real_path))

        if not candidates:
            hint = self._usb_serial_hint()
            message = (
                'No GPS serial port found. Scanned /dev/serial/by-id, '
                '/dev/serial/by-path, /dev/ttyUSB*, and /dev/ttyACM*. '
                'Check the USB cable/port, serial adapter driver, or pass '
                'an explicit port with --ros-args -p port:=/dev/ttyXXX.'
            )
            if hint:
                message += f' {hint}'
            raise RuntimeError(message)

        def score(item):
            path, real_path = item
            text = f'{path} {real_path}'.lower()
            if path.startswith('/dev/serial/by-id/'):
                stable = 100
            elif path.startswith('/dev/serial/by-path/'):
                stable = 80
            else:
                stable = 0
            gps_name = 50 if any(
                token in text
                for token in ('gps', 'gnss', 'ublox', 'u-blox', 'cp210', 'ftdi', 'ch340', 'ch341')
            ) else 0
            acm = 10 if real_path.startswith('/dev/ttyACM') else 0
            return stable + gps_name + acm

        candidates.sort(key=lambda item: (-score(item), item[0]))
        selected_path, selected_real_path = candidates[0]
        candidate_text = ', '.join(
            path if path == real_path else f'{path}->{real_path}'
            for path, real_path in candidates
        )
        if selected_path != selected_real_path:
            self.get_logger().info(
                f'Auto-selected GPS serial port {selected_path} ({selected_real_path}); '
                f'candidates: {candidate_text}'
            )
        else:
            self.get_logger().info(
                f'Auto-selected GPS serial port {selected_path}; candidates: {candidate_text}'
            )
        return selected_path

    @staticmethod
    def _usb_serial_hint() -> str:
        try:
            for device in glob.glob('/sys/bus/usb/devices/*'):
                vendor_path = os.path.join(device, 'idVendor')
                product_path = os.path.join(device, 'idProduct')
                if not os.path.exists(vendor_path) or not os.path.exists(product_path):
                    continue
                with open(vendor_path, 'r', encoding='utf-8') as f:
                    vendor = f.read().strip().lower()
                with open(product_path, 'r', encoding='utf-8') as f:
                    product = f.read().strip().lower()
                if vendor == '1a86' and product == '7523':
                    return (
                        'Detected a CH340/CH341 USB serial adapter (1a86:7523), '
                        'but no tty device exists; the kernel likely needs the ch341 driver.'
                    )
        except Exception:
            return ''
        return ''

    def _connect_ntrip(self):
        server = self.get_parameter('ntrip_server').value
        port = int(self.get_parameter('ntrip_port').value)
        username = self.get_parameter('ntrip_username').value
        password = self.get_parameter('ntrip_password').value
        mountpoint = self.get_parameter('ntrip_mountpoint').value

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((server, port))

            auth_base64 = base64.b64encode(f'{username}:{password}'.encode('utf-8')).decode('utf-8')
            request = (
                f'GET /{mountpoint} HTTP/1.0\r\n'
                'User-Agent: NTRIP ntrip_client\r\n'
                'Accept:*/*\r\n'
                'Connection:close\r\n'
                f'Authorization: Basic {auth_base64}\r\n'
                '\r\n'
            )
            sock.sendall(request.encode('utf-8'))
            sock.settimeout(0.0)
            self.ntrip_socket = sock
            self.get_logger().info(f'Connecting to NTRIP server {server}:{port}/{mountpoint}')
        except Exception as exc:
            self.get_logger().error(f'Failed to connect to NTRIP server: {exc}')
            if self.ntrip_socket is not None:
                self.ntrip_socket.close()
            self.ntrip_socket = None

    def _send_gpgga_to_gps(self):
        if self.gga_rx_flag:
            self.gga_rx_flag = False
            return
        if self.serial_port is None or not self.serial_port.is_open:
            return
        try:
            self.serial_port.write(self.ask_gga.encode('utf-8'))
        except Exception as exc:
            self.get_logger().error(f'Failed to send GPGGA request to GPS: {exc}')

    def _send_gpgga_to_ntrip(self):
        if self.ntrip_socket is None or not self.gga_rx_data:
            return
        try:
            self.ntrip_socket.sendall(self.gga_rx_data.encode('utf-8') + b'\r\n')
        except Exception:
            return

    def _process_io(self):
        self._read_serial()
        self._read_ntrip()

    def _read_serial(self):
        if self.serial_port is None or not self.serial_port.is_open:
            return

        try:
            waiting = self.serial_port.in_waiting
            if waiting <= 0:
                return
            self.serial_buffer.extend(self.serial_port.read(waiting))
        except Exception as exc:
            self.get_logger().error(f'Error reading GPS serial port: {exc}')
            return

        while True:
            start_index = self.serial_buffer.find(b'$')
            if start_index < 0:
                if len(self.serial_buffer) > 4096:
                    self.serial_buffer.clear()
                return
            if start_index > 0:
                del self.serial_buffer[:start_index]

            end_index = self.serial_buffer.find(b'\r\n')
            if end_index < 0:
                end_index = self.serial_buffer.find(b'\n')
            if end_index < 0:
                return

            packet = self.serial_buffer[:end_index + 1]
            del self.serial_buffer[:end_index + 1]
            try:
                self._parse_gps(packet.decode('utf-8', errors='strict'))
            except UnicodeDecodeError:
                now = time.monotonic()
                if now - self.last_decode_warn > 5.0:
                    self.get_logger().warn('Failed to decode GPS data, check serial format')
                    self.last_decode_warn = now

    def _read_ntrip(self):
        if self.ntrip_socket is None or self.serial_port is None:
            return

        try:
            rtcm_data = self.ntrip_socket.recv(4096)
        except BlockingIOError:
            return
        except socket.timeout:
            return
        except Exception as exc:
            self.get_logger().error(f'Error receiving RTCM data: {exc}')
            return

        if not rtcm_data:
            return
        if b'ICY 200 OK' in rtcm_data or b'HTTP/1.1 200' in rtcm_data or b'HTTP/1.0 200' in rtcm_data:
            if not self.ntrip_connected:
                self.ntrip_connected = True
                self.get_logger().info('Connected to NTRIP server.')
            self._send_gpgga_to_ntrip()
        try:
            self.serial_port.write(rtcm_data)
        except Exception as exc:
            self.get_logger().error(f'Error writing RTCM data to GPS serial port: {exc}')

    def _parse_gps(self, data):
        for line in data.replace('\r', '\n').split('\n'):
            sentence = line.strip()
            if not sentence or ('$GPGGA' not in sentence and '$GNGGA' not in sentence):
                continue

            fields = sentence.split(',')
            if len(fields) < 14:
                continue

            try:
                fix_quality = int(fields[6])
                if fix_quality < 0:
                    continue
                self.gga_rx_flag = True
                self.gga_rx_data = sentence
                self._publish_navsatfix_from_gga(fields)
            except (ValueError, IndexError) as exc:
                self.get_logger().warn(f'Failed to parse GGA sentence: {exc}')

    def _publish_navsatfix_from_gga(self, fields):
        fix_msg = NavSatFix()
        fix_msg.header.stamp = self._gps_header_stamp()
        fix_msg.header.frame_id = self.frame_id

        fix_quality = int(fields[6])
        sat_count = int(fields[7]) if fields[7] else 0
        hdop = float(fields[8]) if fields[8] else math.nan
        altitude = float(fields[9]) if fields[9] else math.nan
        geoid_sep = float(fields[11]) if len(fields) > 11 and fields[11] else 0.0

        fix_msg.latitude = dm_to_decimal(fields[2], fields[3], is_latitude=True)
        fix_msg.longitude = dm_to_decimal(fields[4], fields[5], is_latitude=False)
        fix_msg.altitude = altitude + geoid_sep if not math.isnan(altitude) else math.nan

        if fix_quality <= 0:
            fix_msg.status.status = NavSatStatus.STATUS_NO_FIX
        elif fix_quality == 2:
            fix_msg.status.status = NavSatStatus.STATUS_SBAS_FIX
        elif fix_quality in (4, 5):
            fix_msg.status.status = NavSatStatus.STATUS_GBAS_FIX
        else:
            fix_msg.status.status = NavSatStatus.STATUS_FIX
        fix_msg.status.service = NavSatStatus.SERVICE_GPS

        if not math.isnan(hdop):
            sigma_h = hdop * 1.5
            fix_msg.position_covariance[0] = sigma_h * sigma_h
            fix_msg.position_covariance[4] = sigma_h * sigma_h
            fix_msg.position_covariance[8] = -1.0
            fix_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_APPROXIMATED
        else:
            fix_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN

        self.fix_pub.publish(fix_msg)
        self.get_logger().info(
            'Published /gps/fix lat=%.8f lon=%.8f alt=%.3f q=%d sats=%d'
            % (fix_msg.latitude, fix_msg.longitude, fix_msg.altitude, fix_quality, sat_count)
        )


def dm_to_decimal(dm_str, hemisphere, is_latitude):
    if not dm_str:
        raise ValueError('Empty degree-minute field')

    dm = float(dm_str)
    degrees = int(dm / 100.0)
    minutes = dm - degrees * 100.0
    decimal = degrees + minutes / 60.0

    if hemisphere in ('S', 'W'):
        decimal = -decimal
    if is_latitude and not -90.0 <= decimal <= 90.0:
        raise ValueError('Latitude out of range after conversion')
    if not is_latitude and not -180.0 <= decimal <= 180.0:
        raise ValueError('Longitude out of range after conversion')
    return decimal


def main(args=None):
    rclpy.init(args=args)
    node = GpsNtripNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
