import argparse
import os
import time
from tools.utils import msg_to_pil
import atexit
from nav_msgs.msg import Odometry
import math

# ROS 2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Joy
import threading
import datetime
import shutil

# IMAGE_TOPIC = "/usb_cam/image_raw"
IMAGE_TOPIC = "/camera/camera/color/image_raw/compressed"
TOPOMAP_IMAGES_DIR = "./topomaps/images"


def remove_files_in_dir(dir_path: str):
    for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path, f)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


class CreateTopomapNode(Node):
    def __init__(self, args):
        super().__init__("create_topomap")
        self.args = args
        self.obs_img = None
        self.start_time = float("inf")
        self.i = 0
        self.lock = threading.Lock()


        # 创建订阅者和发布者
        self.image_sub = self.create_subscription(
            Image, IMAGE_TOPIC, self.callback_obs, 10
        )
        self.joy_sub = self.create_subscription(
            Joy, "joy", self.callback_joy, 10
        )

        # 创建定时器
        self.timer = self.create_timer(args.dt, self.timer_callback)

        # 创建拓扑图目录（可以通过命令行参数覆盖默认目录）
        images_base = getattr(args, "images_dir", TOPOMAP_IMAGES_DIR)
        self.topomap_name_dir = os.path.join(images_base, args.dir)
        if not os.path.isdir(self.topomap_name_dir):
            os.makedirs(self.topomap_name_dir)
        else:
            self.get_logger().info(f"{self.topomap_name_dir} already exists. Removing previous images...")
            remove_files_in_dir(self.topomap_name_dir)


        # 初始化变量
        self.last_position = None
        self.total_distance = 0.0
        self.poses = []  # 存储所有位姿
        self.distance_threshold = self.args.distance_threshold  # 保存触发阈值
        
        # 创建订阅者
        self.odom_sub = self.create_subscription(
            Odometry,
            '/mfla/dr_odom',
            self.odom_callback,
            100
        )
        
        # 注册退出时的处理函数
        atexit.register(self.save_poses_to_file)
        
        self.get_logger().info(f"Odom Recorder started. Will save poses when displacement exceeds {self.distance_threshold}m or on exit.")


    def callback_obs(self, msg: Image):
        with self.lock:
            self.obs_img = msg_to_pil(msg)

    def callback_joy(self, msg: Joy):
        if msg.buttons[0]:
            self.get_logger().info("Shutdown requested via Joy message.")
            rclpy.shutdown()

    def odom_callback(self, msg):
        # 提取当前位置
        current_position = msg.pose.pose.position
        
        # 如果是第一个位置，只记录不计算
        if self.last_position is None:
            self.last_position = current_position
            self.poses.append(msg.pose.pose)
            return
        
        # 计算与上一个位置的距离
        dx = current_position.x - self.last_position.x
        dy = current_position.y - self.last_position.y
        dz = current_position.z - self.last_position.z
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        
        self.total_distance += distance
        
        if self.total_distance >= self.distance_threshold:
            self.poses.append(msg.pose.pose)
            self.total_distance = 0.0
            self.get_logger().info(f"Displacement exceeded {self.distance_threshold}m. Pose saved. Total poses recorded: {len(self.poses)}")
        
        # 更新上一个位置
        self.last_position = current_position

    def timer_callback(self):
        # pass
        with self.lock:
            if self.obs_img is not None:
                self.obs_img.save(os.path.join(self.topomap_name_dir, f"{self.i}.png"))
                self.get_logger().info(f"Published image {self.i}")
                self.i += 1
                self.start_time = time.time()
                self.obs_img = None

            if time.time() - self.start_time > 2 * self.args.dt:
                self.get_logger().info(f"Topic {IMAGE_TOPIC} not publishing anymore. Shutting down...")
                rclpy.shutdown()
    
    # def save_poses_to_file(self):
    #     if not self.poses:
    #         self.get_logger().info("No poses to save.")
    #         return
        
    #     filename = "/home/mayukai/code/CityWalker/odom_poses.txt"
    #     try:
    #         with open(filename, 'w') as f:
    #             f.write("x,y,z,qw,qx,qy,qz\n")  # 写入标题行
    #             for pose in self.poses:
    #                 # breakpoint()
    #                 f.write(f"{pose.position.x},{pose.position.y},{pose.position.z},{pose.orientation.x},{pose.orientation.y},{pose.orientation.z},{pose.orientation.w}\n")
    #         self.get_logger().info(f"Successfully saved {len(self.poses)} poses to {filename}")
    #     except Exception as e:
    #         self.get_logger().error(f"Failed to save poses: {str(e)}")

    def save_poses_to_file(self):
        if not self.poses:
            self.get_logger().info("No poses to save.")
            return
        
        # 使用命令行参数指定输出目录和文件名（避免硬编码路径）
        output_base = getattr(self.args, "output_dir", "./")
        poses_basename = getattr(self.args, "poses_filename", "odom_poses.txt")

        # 原始文件名（在 output_base 下）
        original_filename = os.path.join(output_base, poses_basename)

        # 带时间戳的文件名，放在 output_base/logs/
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logs_dir = os.path.join(output_base, "logs")
        if not os.path.isdir(logs_dir):
            try:
                os.makedirs(logs_dir, exist_ok=True)
            except Exception:
                pass
        timestamped_filename = os.path.join(logs_dir, f"odom_{timestamp}.txt")
        
        try:
            # 写入原始文件
            with open(original_filename, 'w') as f:
                # header matches the order written below: x,y,z,qx,qy,qz,qw
                f.write("x,y,z,qx,qy,qz,qw\n")  # 写入标题行
                for pose in self.poses:
                    f.write(f"{pose.position.x},{pose.position.y},{pose.position.z},{pose.orientation.x},{pose.orientation.y},{pose.orientation.z},{pose.orientation.w}\n")
            
            # 写入带时间戳的文件
            with open(timestamped_filename, 'w') as f:
                f.write("x,y,z,qx,qy,qz,qw\n")  # 写入标题行
                for pose in self.poses:
                    f.write(f"{pose.position.x},{pose.position.y},{pose.position.z},{pose.orientation.x},{pose.orientation.y},{pose.orientation.z},{pose.orientation.w}\n")
            
            self.get_logger().info(f"Successfully saved {len(self.poses)} poses to both {original_filename} and {timestamped_filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to save poses: {str(e)}")

    


def main(args):
    rclpy.init()
    node = CreateTopomapNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_poses_to_file()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Code to generate topomaps from the {IMAGE_TOPIC} topic"
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="topomap",
        type=str,
        help="path to topological map images in ../topomaps/images directory (default: topomap)",
    )
    parser.add_argument(
        "--dt",
        "-t",
        default=1.0,
        type=float,
        help=f"time between images sampled from the {IMAGE_TOPIC} topic (default: 1.0)",
    )
    parser.add_argument('--distance-threshold', type=float, default=3.0,
                    help='Distance threshold in meters to trigger pose saving (default: 3.0)')

    parser.add_argument(
        "--images-dir",
        type=str,
        default=TOPOMAP_IMAGES_DIR,
        help=f"Base directory to save topomap images (default: {TOPOMAP_IMAGES_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default='./topomaps',
        help="Base output directory for pose files and logs (default: current directory)",
    )
    parser.add_argument(
        "--poses-filename",
        type=str,
        default='odom_poses.txt',
        help="Filename for the main poses file (default: odom_poses.txt)",
    )
    args = parser.parse_args()

    main(args)
