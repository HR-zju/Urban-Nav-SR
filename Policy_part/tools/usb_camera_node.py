#!/usr/bin/env python3
import cv2
import fcntl
import glob
import os
import rclpy
import struct
import time
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


V4L2_CID_GAIN = 0x00980913
V4L2_CID_BACKLIGHT_COMPENSATION = 0x0098091C
V4L2_CID_EXPOSURE_AUTO = 0x009A0901
V4L2_CID_EXPOSURE_AUTO_PRIORITY = 0x009A0903
VIDIOC_S_CTRL = 0xC008561C


class UsbCameraNode(Node):
    def __init__(self):
        super().__init__("usb_camera_node")

        self.publisher_ = self.create_publisher(Image, "/image_raw", 10)
        self.bridge = CvBridge()
        self.publish_count = 0
        self.last_stats_time = time.monotonic()
        self.last_publish_count = 0

        self.device_path = self.select_camera_device()
        self.capture = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        self.capture.set(cv2.CAP_PROP_GAIN, 56)

        if not self.capture.isOpened():
            self.get_logger().error(
                f"Cannot open USB camera at {self.device_path}. "
                "Check connection or permissions."
            )
            return

        self.configure_camera_controls()
        self.log_camera_settings()

        self.get_logger().info(f"USB camera started from {self.device_path}, publishing /image_raw")
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)
        self.stats_timer = self.create_timer(2.0, self.stats_callback)

    def select_camera_device(self) -> str:
        env_device = os.environ.get("USB_CAMERA_DEVICE")
        if env_device:
            return env_device

        candidates = []
        candidates.extend(sorted(glob.glob("/dev/v4l/by-id/*USB_Camera*video-index0")))
        candidates.extend(sorted(glob.glob("/dev/v4l/by-path/*video-index0")))
        candidates.extend(sorted(glob.glob("/dev/video*")))

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return "/dev/video0"

    def set_v4l2_control(self, control_id: int, value: int, name: str):
        try:
            fd = os.open(self.device_path, os.O_RDWR | os.O_NONBLOCK)
            try:
                fcntl.ioctl(fd, VIDIOC_S_CTRL, struct.pack("I i", control_id, value))
                self.get_logger().info(f"Set {name}={value}")
            finally:
                os.close(fd)
        except OSError as exc:
            self.get_logger().warn(f"Failed to set {name}={value}: {exc}")

    def configure_camera_controls(self):
        # UVC auto exposure remains enabled. Gain is lowered slightly to reduce noise.
        self.set_v4l2_control(V4L2_CID_EXPOSURE_AUTO, 3, "auto_exposure")
        self.set_v4l2_control(V4L2_CID_EXPOSURE_AUTO_PRIORITY, 0, "exposure_dynamic_framerate")
        self.set_v4l2_control(V4L2_CID_GAIN, 56, "gain")
        # This camera does not expose a true center-weighted metering control.
        # Backlight compensation is the closest hardware-side option available.
        self.set_v4l2_control(V4L2_CID_BACKLIGHT_COMPENSATION, 1, "backlight_compensation")

    def log_camera_settings(self):
        fourcc = int(self.capture.get(cv2.CAP_PROP_FOURCC))
        fourcc_text = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))
        self.get_logger().info(
            "Camera settings: "
            f"{self.capture.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
            f"{self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}, "
            f"fps={self.capture.get(cv2.CAP_PROP_FPS):.1f}, "
            f"fourcc={fourcc_text}, "
            f"auto_exposure={self.capture.get(cv2.CAP_PROP_AUTO_EXPOSURE)}, "
            f"exposure={self.capture.get(cv2.CAP_PROP_EXPOSURE)}, "
            f"gain={self.capture.get(cv2.CAP_PROP_GAIN)}"
        )

    def timer_callback(self):
        ret, image = self.capture.read()
        if not ret:
            self.get_logger().warn("Failed to receive frame")
            return

        try:
            image_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            image_msg.header.stamp = self.get_clock().now().to_msg()
            image_msg.header.frame_id = "camera_link"
            self.publisher_.publish(image_msg)
            self.publish_count += 1
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"Image conversion failed: {e}")

    def stats_callback(self):
        now = time.monotonic()
        elapsed = now - self.last_stats_time
        if elapsed <= 0.0:
            return

        publish_delta = self.publish_count - self.last_publish_count
        self.get_logger().info(f"Camera loop fps: publish={publish_delta / elapsed:.1f}")

        self.last_stats_time = now
        self.last_publish_count = self.publish_count

    def destroy_node(self):
        if self.capture.isOpened():
            self.capture.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = UsbCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
