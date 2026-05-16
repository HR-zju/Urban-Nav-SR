#!/usr/bin/env python3
import base64
import math
import socket
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, NavSatStatus

try:
    import serial
except ImportError:  # pragma: no cover - handled at runtime on the robot
    serial = None


class GpsNtripNode(Node):
    def __init__(self):
        super().__init__('gps_ntrip_node')

        self.declare_parameter('port', '/dev/ttyUSB0')
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

        self.port = self.get_parameter('port').value
        self.baudrate = int(self.get_parameter('baudrate').value)
        self.ask_gga = self.get_parameter('ask_gga').value
        self.frame_id = self.get_parameter('frame_id').value

        self.serial_port = None
        self.ntrip_socket = None
        self.ntrip_connected = False
        self.gga_rx_flag = False
        self.gga_rx_data = ''
        self.serial_buffer = bytearray()
        self.last_decode_warn = 0.0

        self.fix_pub = self.create_publisher(
            NavSatFix,
            self.get_parameter('fix_topic').value,
            10,
        )

        self._open_serial()
        if self.get_parameter('connect_ntrip_on_start').value:
            self._connect_ntrip()

        self.serial_timer = self.create_timer(0.01, self._process_io)
        self.gps_gga_timer = self.create_timer(3.0, self._send_gpgga_to_gps)
        self.ntrip_gga_timer = self.create_timer(1.0, self._send_gpgga_to_ntrip)

    def destroy_node(self):
        if self.serial_port is not None and self.serial_port.is_open:
            self.serial_port.close()
        if self.ntrip_socket is not None:
            self.ntrip_socket.close()
        super().destroy_node()

    def _open_serial(self):
        if serial is None:
            raise RuntimeError('python3-serial is required to run gps_ntrip_node')

        timeout = float(self.get_parameter('serial_timeout').value)
        self.serial_port = serial.Serial(self.port, self.baudrate, timeout=timeout)
        self.serial_port.write(b'UNLOG\r\n')
        time.sleep(1.0)
        self.serial_port.write(self.ask_gga.encode('utf-8'))
        self.get_logger().info(f'Opened GPS serial port {self.port} at {self.baudrate}')

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
        fix_msg.header.stamp = self.get_clock().now().to_msg()
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
