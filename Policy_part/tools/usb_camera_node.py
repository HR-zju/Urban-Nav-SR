#!/usr/bin/env python3
import cv2
import fcntl
import glob
import os
import rclpy
import struct
import subprocess
import time
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


V4L2_CID_GAIN = 0x00980913
V4L2_CID_BACKLIGHT_COMPENSATION = 0x0098091C
V4L2_CID_EXPOSURE_AUTO = 0x009A0901
V4L2_CID_EXPOSURE_AUTO_PRIORITY = 0x009A0903
VIDIOC_S_CTRL = 0xC008561C
DEFAULT_USB_CAMERA_BY_ID_GLOB = "/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0"
DEFAULT_USB_CAMERA_ID_MODEL = "USB_Camera"
DEFAULT_USB_CAMERA_ID_VENDOR_ID = "0bda"
DEFAULT_USB_CAMERA_ID_MODEL_ID = "3035"
DEFAULT_USB_CAMERA_ID_SERIAL = "Generic_USB_Camera_200901010001"
DEFAULT_USB_CAMERA_ID_PATH = "platform-3610000.usb-usb-0:2.4:1.0"


class UsbCameraNode(Node):
    def __init__(self):
        super().__init__("usb_camera_node")

        self.publisher_ = self.create_publisher(Image, "/image_raw", 10)
        self.bridge = CvBridge()
        self.publish_count = 0
        self.last_stats_time = time.monotonic()
        self.last_publish_count = 0

        try:
            self.device_path = self.select_camera_device()
        except RuntimeError as exc:
            self.get_logger().error(str(exc))
            raise SystemExit(1) from None

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
            raise SystemExit(1)

        self.configure_camera_controls()
        self.log_camera_settings()

        self.get_logger().info(f"USB camera started from {self.device_path}, publishing /image_raw")
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)
        self.stats_timer = self.create_timer(2.0, self.stats_callback)

    def select_camera_device(self) -> str:
        env_device = os.environ.get("USB_CAMERA_DEVICE")
        if env_device:
            if not os.path.exists(env_device):
                raise RuntimeError(f"USB_CAMERA_DEVICE does not exist: {env_device}")
            if not self.is_target_usb_camera(env_device):
                raise RuntimeError(
                    "USB_CAMERA_DEVICE is not the configured USB camera capture device; "
                    "refusing to open it to avoid taking Insight/other cameras. "
                    f"Device: {env_device}\n"
                    f"Detected properties:\n{self.format_device_properties(env_device)}"
                )
            return env_device

        by_id_glob = os.environ.get(
            "USB_CAMERA_BY_ID_GLOB",
            DEFAULT_USB_CAMERA_BY_ID_GLOB,
        )
        usb_camera_candidates = self.existing_video_devices(by_id_glob)
        usb_camera_candidates = [
            candidate
            for candidate in usb_camera_candidates
            if self.is_target_usb_camera(candidate)
        ]
        if len(usb_camera_candidates) == 1:
            return usb_camera_candidates[0]
        if len(usb_camera_candidates) > 1:
            raise RuntimeError(
                "Multiple USB_Camera devices found; refusing to auto-select one. "
                "Set USB_CAMERA_DEVICE to a stable device path, preferably one of:\n"
                f"{self.format_device_list(usb_camera_candidates)}"
            )

        raise RuntimeError(
            "No configured USB camera capture device found. "
            "This node will not fall back to /dev/video0 or generic /dev/v4l/by-path "
            "because those can open Insight or another camera when multiple cameras are connected. "
            "Set USB_CAMERA_DEVICE to the intended USB camera path, or adjust "
            "USB_CAMERA_BY_ID_GLOB / USB_CAMERA_ID_MODEL / USB_CAMERA_ID_VENDOR_ID / "
            "USB_CAMERA_ID_MODEL_ID / USB_CAMERA_ID_SERIAL / USB_CAMERA_ID_PATH if "
            "the USB camera has a different ID or is plugged into a different port.\n"
            f"Detected video devices:\n{self.format_detected_video_devices()}"
        )

    def existing_video_devices(self, pattern: str) -> list[str]:
        devices = []
        seen_real_paths = set()
        for candidate in sorted(glob.glob(pattern)):
            if not os.path.exists(candidate):
                continue
            real_path = os.path.realpath(candidate)
            if real_path in seen_real_paths:
                continue
            seen_real_paths.add(real_path)
            devices.append(candidate)
        return devices

    def format_device_list(self, devices: list[str]) -> str:
        lines = []
        for device in devices:
            real_path = os.path.realpath(device)
            if real_path != device:
                lines.append(f"  {device} -> {real_path}")
            else:
                lines.append(f"  {device}")
        return "\n".join(lines)

    def is_target_usb_camera(self, device: str) -> bool:
        properties = self.video_device_properties(device)
        if not properties:
            return False

        capabilities = properties.get("ID_V4L_CAPABILITIES", "")
        if ":capture:" not in capabilities:
            return False

        expected_model = os.environ.get("USB_CAMERA_ID_MODEL", DEFAULT_USB_CAMERA_ID_MODEL)
        actual_model = properties.get("ID_MODEL", "")
        if self.normalized_udev_value(actual_model) != self.normalized_udev_value(
            expected_model
        ):
            return False

        expected_vendor_id = os.environ.get(
            "USB_CAMERA_ID_VENDOR_ID",
            DEFAULT_USB_CAMERA_ID_VENDOR_ID,
        )
        actual_vendor_id = properties.get("ID_VENDOR_ID", "")
        if expected_vendor_id and actual_vendor_id.lower() != expected_vendor_id.lower():
            return False

        expected_model_id = os.environ.get(
            "USB_CAMERA_ID_MODEL_ID",
            DEFAULT_USB_CAMERA_ID_MODEL_ID,
        )
        actual_model_id = properties.get("ID_MODEL_ID", "")
        if expected_model_id and actual_model_id.lower() != expected_model_id.lower():
            return False

        expected_serial = os.environ.get(
            "USB_CAMERA_ID_SERIAL",
            DEFAULT_USB_CAMERA_ID_SERIAL,
        )
        actual_serial = properties.get("ID_SERIAL", "")
        if expected_serial and actual_serial != expected_serial:
            return False

        expected_path = os.environ.get(
            "USB_CAMERA_ID_PATH",
            DEFAULT_USB_CAMERA_ID_PATH,
        )
        actual_path = properties.get("ID_PATH", "")
        if expected_path and actual_path != expected_path:
            return False

        return True

    def video_device_properties(self, device: str) -> dict[str, str]:
        try:
            output = subprocess.check_output(
                ["udevadm", "info", "-q", "property", "-n", device],
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return {}

        properties = {}
        for line in output.splitlines():
            key, separator, value = line.partition("=")
            if separator:
                properties[key] = value
        return properties

    def normalized_udev_value(self, value: str) -> str:
        return value.strip().replace(" ", "_").lower()

    def format_device_properties(self, device: str) -> str:
        properties = self.video_device_properties(device)
        if not properties:
            return "  <udev properties unavailable>"

        keys = [
            "DEVNAME",
            "ID_VENDOR",
            "ID_VENDOR_ID",
            "ID_MODEL",
            "ID_MODEL_ID",
            "ID_SERIAL",
            "ID_PATH",
            "ID_V4L_CAPABILITIES",
            "DEVLINKS",
        ]
        return "\n".join(
            f"  {key}={properties[key]}" for key in keys if key in properties
        )

    def format_detected_video_devices(self) -> str:
        devices = self.existing_video_devices("/dev/video*")
        if not devices:
            return "  <none>"

        lines = []
        for device in devices:
            properties = self.video_device_properties(device)
            if not properties:
                lines.append(f"  {device}: <udev properties unavailable>")
                continue

            model = properties.get("ID_MODEL", "unknown")
            model_id = properties.get("ID_MODEL_ID", "unknown")
            vendor = properties.get("ID_VENDOR", "unknown")
            vendor_id = properties.get("ID_VENDOR_ID", "unknown")
            capabilities = properties.get("ID_V4L_CAPABILITIES", "unknown")
            devlinks = properties.get("DEVLINKS", "")
            lines.append(
                f"  {device}: vendor={vendor}({vendor_id}), model={model}({model_id}), "
                f"capabilities={capabilities}"
            )
            if devlinks:
                lines.append(f"    links={devlinks}")
        return "\n".join(lines)

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
    node = None
    try:
        node = UsbCameraNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
