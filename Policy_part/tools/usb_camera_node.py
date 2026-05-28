#!/usr/bin/env python3
import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class UsbCameraNode(Node):
    def __init__(self):
        super().__init__("usb_camera_node")

        self.publisher_ = self.create_publisher(Image, "/image_raw", 10)
        self.bridge = CvBridge()

        self.capture = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FPS, 30)

        if not self.capture.isOpened():
            self.get_logger().error(
                "Cannot open USB camera. Check connection or permissions, "
                "for example: sudo chmod 666 /dev/video0"
            )
            return

        self.get_logger().info("USB camera started, publishing /image_raw")
        self.timer = self.create_timer(0.1, self.timer_callback)

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
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"Image conversion failed: {e}")

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
        rclpy.shutdown()


if __name__ == "__main__":
    main()
