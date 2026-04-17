#!/usr/bin/env python
# coding=utf-8
import rospy
import serial
import socket
import time
import base64
from std_msgs.msg import String
import time
from datetime import datetime
from sensor_msgs.msg import NavSatFix, NavSatStatus


# 串口通信相关
port = "/dev/ttyUSB0"
baudrate = 115200
AskGGA = "GPGGA 1\r\n"

# NTRIP客户端相关
ntrip_server = "120.253.239.161"
ntrip_port = 8002
ntrip_username = "cqnf3478"
ntrip_password = "sdh83667"
ntrip_mountpoint = "RTCM33_GRCEJ"

sp = None
ntrip_socket = None
ntrip_is_connect = False
gga_rx_flag = False
gga_rx_data = ""
fix_pub = None


def dm_to_decimal(dm_str, hemisphere, is_latitude):
    """Convert NMEA degree-minute string to decimal degrees.

    NMEA GGA latitude is DDMM.MMMMM, longitude is DDDMM.MMMMM.
    """
    if not dm_str:
        raise ValueError("Empty degree-minute field")

    dm = float(dm_str)
    degrees = int(dm / 100.0)
    minutes = dm - degrees * 100.0
    decimal = degrees + minutes / 60.0

    if hemisphere in ("S", "W"):
        decimal = -decimal

    # Basic range checks to catch malformed input early.
    if is_latitude and (decimal < -90.0 or decimal > 90.0):
        raise ValueError("Latitude out of range after conversion")
    if (not is_latitude) and (decimal < -180.0 or decimal > 180.0):
        raise ValueError("Longitude out of range after conversion")

    return decimal


def publish_navsatfix_from_gga(fields):
    global fix_pub
    if fix_pub is None:
        return

    fix_msg = NavSatFix()
    fix_msg.header.stamp = rospy.Time.now()
    fix_msg.header.frame_id = "gps"

    fix_quality = int(fields[6])
    sat_count = int(fields[7]) if fields[7] else 0
    hdop = float(fields[8]) if fields[8] else float("nan")
    altitude = float(fields[9]) if fields[9] else float("nan")
    geoid_sep = float(fields[11]) if len(fields) > 11 and fields[11] else 0.0

    # Convert DDMM.MMMMM / DDDMM.MMMMM into decimal degrees.
    latitude = dm_to_decimal(fields[2], fields[3], is_latitude=True)
    longitude = dm_to_decimal(fields[4], fields[5], is_latitude=False)

    fix_msg.latitude = latitude
    fix_msg.longitude = longitude
    # NMEA GGA altitude is above mean sea level. Convert to ellipsoid height.
    fix_msg.altitude = altitude + geoid_sep if not (altitude != altitude) else float("nan")

    if fix_quality <= 0:
        fix_msg.status.status = NavSatStatus.STATUS_NO_FIX
    elif fix_quality == 2:
        fix_msg.status.status = NavSatStatus.STATUS_SBAS
    elif fix_quality in (4, 5):
        fix_msg.status.status = NavSatStatus.STATUS_GBAS
    else:
        fix_msg.status.status = NavSatStatus.STATUS_FIX

    # GN/GP GGA commonly carries GPS, but keep service broad for downstream consumers.
    fix_msg.status.service = NavSatStatus.SERVICE_GPS

    # Approximate horizontal covariance from HDOP (2D), keep vertical unknown.
    # Using 1.5m nominal UERE -> sigma_h ~= HDOP * 1.5.
    if hdop == hdop:
        sigma_h = hdop * 1.5
        fix_msg.position_covariance[0] = sigma_h * sigma_h
        fix_msg.position_covariance[4] = sigma_h * sigma_h
        fix_msg.position_covariance[8] = -1.0
        fix_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_APPROXIMATED
    else:
        fix_msg.position_covariance[0] = 0.0
        fix_msg.position_covariance[4] = 0.0
        fix_msg.position_covariance[8] = 0.0
        fix_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN

    fix_pub.publish(fix_msg)
    rospy.loginfo(
        "Published /gps/fix lat=%.8f lon=%.8f alt=%.3f q=%d sats=%d",
        fix_msg.latitude,
        fix_msg.longitude,
        fix_msg.altitude,
        fix_quality,
        sat_count,
    )

def send_gpgga_to_gps(event):
    global sp
    global gga_rx_flag
    if gga_rx_flag is True:
        gga_rx_flag =False
        return
    if sp is not None and sp.is_open:
        try:
            sp.write(AskGGA.encode('utf-8'))
        except Exception as e:
            rospy.logerr("Failed to send GPGGA to GPS: %s", e)

def send_gpgga_to_ntrip(event):
    global ntrip_socket
    if ntrip_socket is None:
        return
    # rospy.loginfo("Sent GPGGA:%s",gga_rx_data)
    try:
        # ntrip_socket.connect((ntrip_server, ntrip_port))
        ntrip_socket.send(gga_rx_data.encode('utf-8') + b"\r\n")
    except (Exception):
        return

def parse_gps(data):
    global gga_rx_flag, gga_rx_data
    # gga_rx_data = "$GPGGA,070000.000,3016.1700,N,12007.4880,E,1,10,1.0,20.0,M,0.0,M,,*66"
    # gga_rx_flag = True
    # return
    lines = data.replace("\r", "\n").split("\n")
    for line in lines:
        sentence = line.strip()
        if not sentence:
            continue
        if "$GPGGA" not in sentence and "$GNGGA" not in sentence:
            continue

        fields = sentence.split(',')
        if len(fields) < 14:
            continue

        try:
            fix_quality = int(fields[6])
            if fix_quality < 0:
                continue

            gga_rx_flag = True
            gga_rx_data = sentence
            publish_navsatfix_from_gga(fields)
        except (ValueError, IndexError) as e:
            rospy.logwarn("Failed to parse GGA sentence: %s", e)
            continue

def gps_ntrip_node():
    global sp, ntrip_socket, ntrip_is_connect,gga_rx_flag,gga_rx_data, fix_pub
    rospy.init_node('gps_ntrip_node', anonymous=True)
    fix_pub = rospy.Publisher('/gps/fix', NavSatFix, queue_size=10)

    # 配置串口
    sp = serial.Serial(port, baudrate, timeout=1)
    sp.write("UNLOG\r\n".encode('utf-8'))
    time.sleep(1)
    sp.write(AskGGA.encode('utf-8'))
    rate = rospy.Rate(100)  
    buffer = bytearray() 
    while not rospy.is_shutdown():
        # 从串口读取GPS数据
        if sp.in_waiting > 0:
            data = sp.read(sp.in_waiting)
            # print("收到串口数据: ", data)
            # 确保 data 是 bytes 类型
            if isinstance(data, str):
                data = data.encode('utf-8')
            data = bytearray(data)
            buffer.extend(data)
        else:
            if len(buffer) > 0:
                try:
                    # 将缓存数据解码为字符串
                    data_str = buffer.decode('utf-8')
                    parse_gps(data_str)
                    buffer = bytearray()   # 清空缓存
                except UnicodeDecodeError:
                    rospy.logwarn("Failed to decode data,Check data format\r\n")
                    buffer = bytearray()   # 清空缓存以避免累积错误数据
        if gga_rx_flag is True: 
            gga_rx_flag = False
            # break
        rate.sleep()

    # rospy.loginfo("GGA: %s", gga_rx_data)
    # 配置NTRIP客户端
    try:
        ntrip_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ntrip_socket.connect((ntrip_server, ntrip_port))

        # 构造认证字符串
        auth_str = ntrip_username + ":" + ntrip_password
        auth_bytes = auth_str.encode('utf-8')
        auth_base64 = base64.b64encode(auth_bytes).decode('utf-8')

        print("Connecting to NTRIP server %s:%d with mountpoint %s" % (ntrip_server, ntrip_port, ntrip_mountpoint))
        # 发送HTTP请求
        request = (
            "GET /%s HTTP/1.0\r\n"
            "User-Agent: NTRIP ntrip_client\r\n"
            "Accept:*/*\r\n"
            "Connection:close\r\n"
            "Authorization: Basic %s\r\n"
            "\r\n"
        ) % (ntrip_mountpoint, auth_base64)
        ntrip_socket.send(request.encode('utf-8'))
        print("Sent NTRIP request:\n%s" % request)
    except Exception as e:
        rospy.logerr("Failed to connect to NTRIP server: %s", e)
        return

    rospy.Timer(rospy.Duration(3), send_gpgga_to_gps)
    rospy.Timer(rospy.Duration(1), send_gpgga_to_ntrip)
    rate = rospy.Rate(100)  # 100hz
    buffer = bytearray()
    while not rospy.is_shutdown():
        # 从串口读取GPS数据
        if sp.in_waiting > 0:
            data = sp.read(sp.in_waiting)
            # 确保 data 是 bytes 类型
            if isinstance(data, str):
                data = data.encode('utf-8')
            data = bytearray(data)
            buffer.extend(data)
            # 查找 $ 和 \r\n 的位置
            start_index = buffer.find(b'$')
            end_index = buffer.find(b'\r\n')

            if start_index != -1 and end_index != -1 and start_index < end_index:
                # 截取从 $ 到 \r\n 的数据段
                gps_data = buffer[start_index:end_index + 2]  # 包括 \r\n
                # 解码数据
                decoded_data = gps_data.decode('utf-8', errors='ignore')
                # print("Extracted GPS data:", decoded_data)
                # 解析 GPS 数据
                parse_gps(decoded_data)
                # 清空缓冲区
                buffer = bytearray() 

        # 从NTRIP获取RTCM数据并发送到串口
        try:
            if ntrip_socket is not None:
                rtcm_data = ntrip_socket.recv(4096)
                if rtcm_data:
                    # rospy.loginfo("Rx RTCM: %s", rtcm_data)
                    if b"ICY 200 OK" in rtcm_data:
                        ntrip_is_connect = True
                        send_gpgga_to_ntrip(0)
                        rospy.loginfo("Connected to NTRIP server.")
                    sp.write(rtcm_data)
                    # rospy.loginfo("Sent RTCM data to serial port.%d",len(rtcm_data))

        except Exception as e:
            rospy.logerr("Error receiving RTCM data: %s", e)

        rate.sleep()

    if sp is not None and sp.is_open:
        sp.close()
    if ntrip_socket is not None:
        ntrip_socket.close()

if __name__ == '__main__':
    try:
        gps_ntrip_node()
    except rospy.ROSInterruptException:
        pass