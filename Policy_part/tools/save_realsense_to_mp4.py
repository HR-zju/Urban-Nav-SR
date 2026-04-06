#!/usr/bin/env python3
"""
Subscribe to a ROS2 compressed image topic and save a 30Hz MP4 video to the
`front/` folder. The script initializes the VideoWriter on the first received
frame so it adapts to the camera resolution.

Usage examples:
  python tools/save_realsense_to_mp4.py
  python tools/save_realsense_to_mp4.py --topic /camera/camera/color/image_raw/compressed --fps 30 --duration 60

"""
import argparse
import os
import time
import threading
import signal

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

import cv2
import numpy as np


class RealsenseRecorder(Node):
    def __init__(self, topic, fps, out_path, max_seconds=None):
        super().__init__('realsense_recorder')
        self.topic = topic
        self.fps = float(fps)
        self.out_path = out_path
        if max_seconds is not None:
            self.max_seconds = float(max_seconds)
        else:
            self.max_seconds = None

        self._writer = None
        self._frame_count = 0
        self._start_time = None
        self._lock = threading.Lock()
        self._running = True

        qos_profile = rclpy.qos.QoSProfile(depth=10)
        self.sub = self.create_subscription(
            CompressedImage, self.topic, self._image_callback, qos_profile
        )
        self.get_logger().info(
            f"Subscribed to {self.topic}, writing to {self.out_path}"
        )

    def _init_writer(self, height, width):
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        # FourCC for mp4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            self.out_path, fourcc, self.fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(
                f"Failed to open VideoWriter for {self.out_path}"
            )
        self._writer = writer
        self.get_logger().info(
            f"VideoWriter initialized: {width}x{height} @ {self.fps} fps"
        )

    def _image_callback(self, msg: CompressedImage):
        # Decode compressed image message to BGR numpy image
        try:
            image_data = np.frombuffer(msg.data, dtype=np.uint8)
            img_bgr = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            if img_bgr is None:
                self.get_logger().warning(
                    "Failed to decode compressed image, skipping frame"
                )
                return
        except Exception as e:
            self.get_logger().error(
                f"Exception decoding compressed image: {e}"
            )
            return

        h, w = img_bgr.shape[:2]

        with self._lock:
            if self._writer is None:
                # Initialize writer on first frame
                try:
                    self._init_writer(height=h, width=w)
                    self._start_time = time.time()
                except Exception as e:
                    self.get_logger().error(
                        f"Failed to initialize writer: {e}"
                    )
                    rclpy.shutdown()
                    return

            # Write frame
            try:
                # OpenCV expects BGR; msg is BGR already from imdecode
                self._writer.write(img_bgr)
                self._frame_count += 1
            except Exception as e:
                self.get_logger().error(f"Error writing frame: {e}")

            # If duration limit is set, check elapsed time
            if self.max_seconds is not None and self._start_time is not None:
                elapsed = time.time() - self._start_time
                if elapsed >= self.max_seconds:
                    self.get_logger().info(
                        f"Reached duration limit: {elapsed:.2f}s, stopping"
                    )
                    # Trigger shutdown
                    self._running = False
                    # Release writer in separate thread to avoid deadlock
                    t = threading.Thread(
                        target=self._close_writer_and_shutdown, daemon=True
                    )
                    t.start()

    def _close_writer_and_shutdown(self):
        with self._lock:
            if self._writer is not None:
                try:
                    self._writer.release()
                    self.get_logger().info(
                        f"Saved video to {self.out_path} "
                        f"({self._frame_count} frames)"
                    )
                except Exception as e:
                    self.get_logger().error(f"Error closing writer: {e}")
                self._writer = None
        # Allow logger messages to flush, then shutdown rclpy
        time.sleep(0.1)
        try:
            rclpy.shutdown()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--topic',
        type=str,
        default='/camera/camera/color/image_raw/compressed',
    )
    parser.add_argument(
        '--fps', type=float, default=30.0, help='Desired output fps'
    )
    parser.add_argument(
        '--out',
        type=str,
        default=None,
        help='Output mp4 path (overrides timestamped default)',
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=None,
        help='Max seconds to record (optional)',
    )
    args = parser.parse_args()

    # Determine output path
    base_folder = os.path.join(os.path.dirname(__file__), '..', 'front')
    base_folder = os.path.abspath(base_folder)
    os.makedirs(base_folder, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    default_name = f'realsense_{timestamp}.mp4'
    if args.out is not None:
        out_path = args.out
    else:
        out_path = os.path.join(base_folder, default_name)

    rclpy.init()
    node = RealsenseRecorder(
        args.topic, args.fps, out_path, max_seconds=args.duration
    )

    # handle SIGINT gracefully to ensure writer is released
    def sigint_handler(signum, frame):
        # Avoid closing the writer inside the signal handler because the
        # callback thread may hold locks; instead request rclpy shutdown and
        # let the main thread run finalization (which will close the writer).
        try:
            node.get_logger().info('SIGINT received, requesting shutdown...')
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass

    signal.signal(signal.SIGINT, sigint_handler)

    try:
        # spin in the main thread until rclpy.shutdown() is called
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt, shutting down')
    finally:
        # make sure writer is closed
        node._close_writer_and_shutdown()
        try:
            node.destroy_node()
        except Exception:
            pass


if __name__ == '__main__':
    main()
