# test.py

import pytorch_lightning as pl
import argparse
import yaml
import os
# from scipy.interpolate import interp1d

# from pl_modules.citywalk_datamodule import CityWalkDataModule
# from pl_modules.teleop_datamodule import TeleopDataModule
# from pl_modules.citywalker_module import CityWalkerModule
# from pl_modules.citywalker_feat_module import CityWalkerFeatModule
import torch
import glob
import datetime
# ros2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from nav_msgs.msg import Odometry
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import threading
import queue
from collections import deque
import copy
import cv2
from cv_bridge import CvBridge
import torchvision.transforms.functional as TF
from std_msgs.msg import Bool, Float32MultiArray
from scipy.spatial.transform import Rotation as R
# from unitree_sdk2py.core.channel import ChannelFactoryInitialize
# from unitree_sdk2py.go2.video.video_client import VideoClient

from PIL import Image as PILImage
import numpy as np
import math
import matplotlib.pyplot as plt


torch.set_float32_matmul_precision('medium')
pl.seed_everything(42, workers=True)

import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0+PTX"
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
import importlib
import yaml
import logging
from torch.utils.data import DataLoader
import time

# from e2enav.models.base_model import BaseModel

torch.set_float32_matmul_precision('medium')


def show_tensor(img_tensor):
    # 将张量转换为 numpy 数组，并调整通道顺序为 HWC
    img_np = img_tensor.permute(1, 2, 0).numpy()
    
    # 如果张量是 float 类型且值范围在 [0, 1]，转换为 [0, 255]
    if img_np.dtype == np.float32 or img_np.dtype == np.float64:
        img_np = (img_np * 255).astype(np.uint8)
    
    # 如果是 BGR 转 RGB（根据张量来源决定是否需要）
    # img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # 显示图像
    # cv2.imshow('Image', img_np)
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()


def load_target_positions_from_file(filename="odom_poses.txt"):
    """从保存的txt文件中加载所有目标位置"""
    target_positions = [[0., 0., 0., 0., 0., 0., 1.]]
    return target_positions
    try:
        with open(filename, 'r') as f:
            # 跳过标题行
            next(f)
            # 读取所有行
            for line in f:
                line = line.strip()
                if line:  # 确保不是空行
                    parts = line.split(',')
                    if len(parts) >= 2:  # 至少需要x,y坐标
                        x = float(parts[0])
                        y = float(parts[1])
                        z = float(parts[2])
                        qx = float(parts[3])
                        qy = float(parts[4])
                        qz = float(parts[5])
                        qw = float(parts[6])

                        
                        target_positions.append([x, y, z, qx, qy, qz, qw])
            
            if not target_positions:
                raise ValueError("No valid target positions found in the file")
            
            return target_positions
            
    except FileNotFoundError:
        raise FileNotFoundError(f"Target position file not found: {filename}")
    except Exception as e:
        raise ValueError(f"Error reading target positions: {str(e)}")

def convert_fisheye_to_pinhole(K, dist, device, fisheye_tensor,
                                  output_size = None,
                                  fov_x= None,
                                  fov_y= None,
                                  fov_deg= None,
                                  crop_center = (0.0, 0.0)):
    """
    Convert fisheye batch tensor to pinhole images
    
    Args:
        fisheye_tensor: Input fisheye images (N, C, H, W) in [0, 1] range
        output_size: Output size (H, W). If None, same as input
        fov_x: Horizontal field of view (degrees). Takes priority over fov_deg
        fov_y: Vertical field of view (degrees). Takes priority over fov_deg
        fov_deg: Symmetric FOV for both axes (degrees). Default 60.0
        crop_center: Crop center in fisheye space (x, y) in [-1, 1]
        
    Returns:
        Pinhole images (N, C, H_out, W_out)
    """
    N, C, H, W = fisheye_tensor.shape
    
    # Set output size
    if output_size is None:
        H_out, W_out = H, W
    else:
        H_out, W_out = output_size
    
    # Handle FOV parameters - priority: fov_x/fov_y > fov_deg > default
    if fov_x is None and fov_y is None and fov_deg is None:
        fov_deg = 60.0  # Default
    
    if fov_x is not None:
        fov_x_rad = np.deg2rad(fov_x)
    else:
        fov_deg_val = fov_deg if fov_deg is not None else 60.0
        fov_x_rad = np.deg2rad(fov_deg_val)
        
    if fov_y is not None:
        fov_y_rad = np.deg2rad(fov_y)
    else:
        fov_deg_val = fov_deg if fov_deg is not None else 60.0
        fov_y_rad = np.deg2rad(fov_deg_val)
        
    # print(f"Converting {N} fisheye images: {H}×{W} → {H_out}×{W_out}")
    # print(f"FOV: horizontal={np.rad2deg(fov_x_rad):.1f}°, vertical={np.rad2deg(fov_y_rad):.1f}°")
    # print(f"Crop center: {crop_center}")
    
    # Calculate output camera intrinsics with different fx, fy
    fx = W_out / (2 * np.tan(fov_x_rad / 2))
    fy = H_out / (2 * np.tan(fov_y_rad / 2))
    cx, cy = W_out / 2, H_out / 2
    
    # Adjust for crop center
    crop_x, crop_y = crop_center
    cx += crop_x * fx
    cy += crop_y * fy
    
    # Create output pixel grid
    u_out = torch.arange(W_out, dtype=torch.float32, device=device)
    v_out = torch.arange(H_out, dtype=torch.float32, device=device)
    v_grid_out, u_grid_out = torch.meshgrid(v_out, u_out, indexing='ij')
    
    # Convert output pixels to normalized coordinates (pinhole)
    x_norm = (u_grid_out - cx) / fx
    y_norm = (v_grid_out - cy) / fy
    
    # Convert to 3D ray direction (pinhole projection)
    ray_x = x_norm
    ray_y = y_norm
    ray_z = torch.ones_like(x_norm)
    
    # Normalize rays
    ray_length = torch.sqrt(ray_x**2 + ray_y**2 + ray_z**2)
    ray_x /= ray_length
    ray_y /= ray_length
    ray_z /= ray_length
    
    # Convert 3D rays to fisheye coordinates
    # Calculate theta (angle from optical axis) and phi (azimuth)
    theta = torch.acos(torch.clamp(ray_z, -1.0 + 1e-7, 1.0 - 1e-7))
    phi = torch.atan2(ray_y, ray_x)
    
    # Apply fisheye distortion model (inverse projection)
    k1, k2, p1, p2, k3 = dist[0], dist[1], dist[2], dist[3], dist[4]
    
    # Iterative solution for undistorted radius
    r_u = theta.clone()  # Initial guess
    for _ in range(10):
        # Forward distortion model
        r_u2 = r_u * r_u
        r_u4 = r_u2 * r_u2
        r_u6 = r_u4 * r_u2
        
        # Radial distortion
        radial_distortion = 1 + k1*r_u2 + k2*r_u4 + k3*r_u6
        r_d = r_u * radial_distortion
        
        # Error and correction
        error = r_d - theta
        # Derivative for Newton-Raphson
        dr_d_dr_u = radial_distortion + r_u * (2*k1*r_u + 4*k2*r_u**3 + 6*k3*r_u**5)
        
        # Update
        r_u = r_u - error / (dr_d_dr_u + 1e-8)
    
    # Convert to fisheye image coordinates
    x_fish = r_u * torch.cos(phi)
    y_fish = r_u * torch.sin(phi)
    
    # Apply tangential distortion
    r2 = x_fish**2 + y_fish**2
    tangential_x = 2*p1*x_fish*y_fish + p2*(r2 + 2*x_fish**2)
    tangential_y = p1*(r2 + 2*y_fish**2) + 2*p2*x_fish*y_fish
    
    x_fish += tangential_x
    y_fish += tangential_y
    
    # Convert to pixel coordinates in fisheye image
    fx_fish, fy_fish = K[0, 0], K[1, 1]
    cx_fish, cy_fish = K[0, 2], K[1, 2]
    
    u_fish = x_fish * fx_fish + cx_fish
    v_fish = y_fish * fy_fish + cy_fish
    
    # Create sampling grid for grid_sample (normalized to [-1, 1])
    grid_u = 2.0 * u_fish / (W - 1) - 1.0
    grid_v = 2.0 * v_fish / (H - 1) - 1.0
    
    # Validity mask
    valid_mask = (u_fish >= 0) & (u_fish < W) & (v_fish >= 0) & (v_fish < H)
    
    # Set invalid pixels to sample from center (reduces black regions)
    grid_u = torch.where(valid_mask, grid_u, torch.zeros_like(grid_u))
    grid_v = torch.where(valid_mask, grid_v, torch.zeros_like(grid_v))
    
    grid = torch.stack([grid_u, grid_v], dim=-1).unsqueeze(0)  # (1, H_out, W_out, 2)
    
    # Batch sampling from fisheye images
    # print("Batch sampling fisheye images...")
    
    # Expand grid to match batch size: (1, H_out, W_out, 2) -> (N, H_out, W_out, 2)
    grid_batch = grid.expand(N, -1, -1, -1)  # (N, H_out, W_out, 2)
    
    # Batch grid sample - process all images at once
    pinhole_images = torch.nn.functional.grid_sample(
        fisheye_tensor,     # (N, C, H, W)
        grid_batch,         # (N, H_out, W_out, 2)
        mode='bilinear', 
        padding_mode='border', 
        align_corners=True
    )  # (N, C, H_out, W_out)
    
    # Apply valid mask to all images at once
    valid_mask_expanded = valid_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H_out, W_out)
    pinhole_images = pinhole_images * valid_mask_expanded.float()  # Broadcasting
    
    print(f"Batch processed {N} images")
    
    return pinhole_images
def gen_sineembed_for_position(pos_tensor, hidden_dim=256, scale_scale=1.0):
    import math
    """Mostly copy-paste from https://github.com/IDEA-opensource/DAB-DETR/
    """
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    x_embed = pos_tensor[..., 0] * scale * scale_scale
    y_embed = pos_tensor[..., 1] * scale * scale_scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos = torch.cat((pos_y, pos_x), dim=-1)
    return pos

def msg_to_pil(msg: Image) -> PILImage.Image:
    # 根据encoding处理不同格式
    if msg.encoding == 'rgb8':
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3)
        return PILImage.fromarray(img, 'RGB')
    elif msg.encoding == 'rgba8':
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 4)
        return PILImage.fromarray(img, 'RGBA')
    elif msg.encoding == 'bgr8':
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3)
        # BGR转RGB
        img = img[:, :, ::-1]
        return PILImage.fromarray(img, 'RGB')
    elif msg.encoding == 'bgra8':
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 4)
        # BGRA转RGBA
        img = img[:, :, [2, 1, 0, 3]]
        return PILImage.fromarray(img, 'RGBA')
    elif msg.encoding == 'mono8':
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width)
        return PILImage.fromarray(img, 'L')
    else:
        raise ValueError(f"Unsupported encoding: {msg.encoding}")


class ImagePoseSyncNode(Node):
    def __init__(self, distance_threshold=1.):
        super().__init__('image_pose_sync_node')
        self.bridge = CvBridge()  
        self.distance_threshold = distance_threshold

        # 初始化队列和缓冲区
        self.image_queue = [] # queue.Queue()
        self.image_batch = deque(maxlen=21)  # []
        self.image_batch_old = deque(maxlen=21)  # []
        self.waypoint_pub = self.create_publisher(Float32MultiArray, 'waypoint', 1)
        self.goal_pub = self.create_publisher(Bool, "/topoplan/reached_goal", 1)


        self.pose_buffer = deque(maxlen=20)  # 保留最近的20个位姿用于查找
        self.localpose_buffer = deque(maxlen=20)  # 保留最近的20个位姿用于查找

        self.lock = threading.Lock()
        self.last_image_time = 0.
        self.last_position = None
        self.last_localposition = None
        self.poses = []        
        self.total_distance = 0.

        # Unitree Go2 front camera via SDK2 (same as camera_opencv_ros2.py)
        # self.client = VideoClient()
        # self.client.SetTimeout(10.0)
        # self.client.Init()
        
        # Timer pull frames; will be throttled to ~6Hz in callback
        # self.timer = self.create_timer(0.1, self.image_callback)

        # RealSense pinhole camera image subscription (compressed image)
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/camera/camera/color/image_raw/compressed',
            self.image_callback,
            10
        )
        
        # self.enable = False
        self.enable_2 = False
        # self.autonomy_sub = self.create_subscription(
        #             Bool, '/joystick_services/autonomy', self.globale_auto_callback, 10)
        self.autonomy_sub2 = self.create_subscription(
                    Bool, '/cmd_vel/autonomy_enable', self.button_b_callback, 10)

        self.localodom_sub = self.create_subscription(
            Odometry,
            '/mfla/dr_odom',
            self.localodom_callback,
            20  # 适当调整队列大小
        )
        
        # 订阅里程计话题
        self.odom_sub = self.create_subscription(
            Odometry,
            '/mfla/dr_odom',
            self.odom_callback,
            100  # 适当调整队列大小
        )
        
        # 启动处理线程
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        self.get_logger().info("Image and Pose sync node initialized")

    # def callback_disable(self, msg: Bool):
    #     # self.flag = False
    #     # print(msg)
    #     pass
    # def globale_auto_callback(self, msg: Bool):
    #     enable = False
    #     data = msg.data
    #     print(data)
    #     pass

    def localodom_callback(self, msg):
        """里程计回调函数，存储位姿和时间戳"""
        # self.localpose_buffer.append((msg.header.stamp, msg.pose))
        self.localodom_record(msg)

        try:
            with self.lock:
                # 存储时间戳和位姿
                self.localpose_buffer.append((msg.header.stamp, msg.pose))
        #         dx = target_xy[0][0]-  msg.pose.pose.position.x
        #         dy = target_xy[0][1]-  msg.pose.pose.position.y
        #         pose_diff = math.sqrt(dx**2+dy**2)
        #         # self.get_logger().info(f"pose_diff = {pose_diff}")

        #         if pose_diff < self.distance_threshold:
        #             if len(target_xy) > 1:
        #                 del target_xy[0]  # 删除索引0的元素
        #             else:
        #                 reached_goal_msg = Bool()
        #                 reached_goal_msg.data = bool(True)
        #                 self.goal_pub.publish(reached_goal_msg)
        #                 self.get_logger().info("Go2 arrived. exit.")
        #                 node.destroy_node()
        #                 rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Error in odom callback: {str(e)}")

    def localodom_record(self, msg):
        # 提取当前位置
        current_position = msg.pose.pose.position
        
        # 如果是第一个位置，只记录不计算
        if self.last_localposition is None:
            self.last_localposition = np.array(
                [
                    current_position.x,
                    current_position.y,
                    current_position.z,
                    msg.pose.pose.orientation.w,
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                ]
            )
            self.poses.append(msg.pose.pose)
            return
        
        # 计算与上一个位置的距离
        # dx = current_position.x - self.last_localposition.x
        # dy = current_position.y - self.last_localposition.y
        # dz = current_position.z - self.last_localposition.z
        # distance = math.sqrt(dx**2 + dy**2 + dz**2)
        
        # 累加总距离
        # self.total_distance += distance
        
        # # 如果移动超过阈值距离，保存位姿并重置计数器
        # self.poses.append(msg.pose.pose)
        # if self.total_distance >= 0.1:
        #     self.total_distance = 0.0
        #     self.get_logger().info(f"Displacement exceeded {self.distance_threshold}m. Pose saved. Total poses recorded: {len(self.poses)}")
        
        # 更新上一个位置
        self.last_localposition = np.array(
                [
                    current_position.x,
                    current_position.y,
                    current_position.z,
                    msg.pose.pose.orientation.w,
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                ]
            )

    def button_b_callback(self, msg: Bool):
        self.enable_2 = False
        try:
            data = msg.data
            self.enable_2 = data
        except:
            pass
        pass

    def odom_record(self, msg):
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
        
        # 累加总距离
        self.total_distance += distance
        
        # 如果移动超过阈值距离，保存位姿并重置计数器
        if self.total_distance >= 0.1:
            self.poses.append(msg.pose.pose)
            self.total_distance = 0.0
            self.get_logger().info(f"Displacement exceeded {self.distance_threshold}m. Pose saved. Total poses recorded: {len(self.poses)}")
        
        # 更新上一个位置
        self.last_position = current_position

    # def image_callback(self):
    #     """图像回调函数，从Unitree VideoClient获取图像并处理"""
    #     code, data = self.client.GetImageSample()
    #     if code != 0:
    #         self.get_logger().error(f"Get image sample error. code: {code}")
    #         return

    #     image_data = np.frombuffer(bytes(data), dtype=np.uint8)
    #     image_bgr = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    #     if image_bgr is None:
    #         self.get_logger().warning("Failed to decode image sample")
    #         return
        
    #     # Get current ROS time using node's clock
    #     current_ros_time_msg = self.get_clock().now().to_msg()
    #     current_time = (current_ros_time_msg.nanosec + current_ros_time_msg.sec * 1e9) * 1e-9
    #     time_since_last = (current_time - self.last_image_time)  # 转换为秒
        
    #     if time_since_last >= 1.0 / 6.0: # control the image frequence is about 6 hz
    #         try:
    #             # Convert BGR(OpenCV) -> RGB
    #             rgb_data = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                
    #             # Find closest pose using ROS timestamp
    #             pose = self.find_closest_pose(current_ros_time_msg)
    #             localpose = self.find_closest_localpose(current_ros_time_msg)
    #             if not pose:
    #                 self.get_logger().warning(f"no pose yet")
    #                 return
                
    #             # Convert to PIL Image
    #             pil_img = PILImage.fromarray(rgb_data, 'RGB')

            
    #             # 2. 目标分辨率
    #             desired_height = 1080 #1080 // 2
    #             desired_width = 1920 #1920 // 2
    #             # 3. 先计算缩放比例，用 cv2.resize 进行缩放
    #             W, H = pil_img.size
    #             # breakpoint()
    #             scale_H, scale_W = desired_height / H, desired_width / W
    #             new_H, new_W = int(H * scale_H), int(W * scale_W)
    #             # breakpoint()
    #             if (new_H, new_W) != (H, W):
    #                 pil_img = pil_img.resize((new_W, new_H), PILImage.BILINEAR)
    #             img_np = np.array(pil_img)  # PIL Image → numpy
    #             # breakpoint()
    #             img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
    #             # 5. 如果缩放后仍不满足尺寸，进行 padding 或 center_crop
    #             _, H_resized, W_resized = img_tensor.shape
    #             pad_height = desired_height - H_resized
    #             pad_width = desired_width - W_resized
                
    #             if pad_height > 0 or pad_width > 0:  # 需要 padding
    #                 pad_top = pad_height // 2
    #                 pad_bottom = pad_height - pad_top
    #                 pad_left = pad_width // 2
    #                 pad_right = pad_width - pad_left
                    
    #                 img_tensor = TF.pad(
    #                     img_tensor,
    #                     (pad_left, pad_top, pad_right, pad_bottom),
    #                 )
                
    #             elif pad_height < 0 or pad_width < 0:  # 需要裁剪
    #                 img_tensor = TF.center_crop(img_tensor, (desired_height, desired_width))
                
    #             # 6. 检查最终尺寸是否正确
    #             assert img_tensor.shape[1] == desired_height and img_tensor.shape[2] == desired_width, \
    #                 f"Final image shape {img_tensor.shape} != ({desired_height}, {desired_width})"
               
    #             ## show image
    #             # breakpoint()
    #             show_tensor(img_tensor)

    #             with self.lock:
    #                 self.image_batch.append((img_tensor, pose, localpose))
    #             self.last_image_time = current_time

    #         except Exception as e:
    #             self.get_logger().error(f"Error in image callback: {str(e)}")

    def image_callback(self, msg: CompressedImage):
        """图像回调函数，从RealSense压缩图像话题获取图像并处理"""
        try:
            # Decode compressed image data
            image_data = np.frombuffer(msg.data, dtype=np.uint8)
            image_bgr = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            if image_bgr is None:
                self.get_logger().warning("Failed to decode compressed image")
                return

            # Use message timestamp for sync and rate limiting
            image_stamp = msg.header.stamp
            current_time = (image_stamp.nanosec + image_stamp.sec * 1e9) * 1e-9
            time_since_last = current_time - self.last_image_time

            # Control the image frequency to about 6 Hz
            if time_since_last < 1.0 / 6.0:
                return

            # Convert BGR(OpenCV) -> RGB
            rgb_data = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # Find closest pose using image timestamp
            pose = self.find_closest_pose(image_stamp)
            localpose = self.find_closest_localpose(image_stamp)
            if not pose:
                self.get_logger().warning("No pose yet for current image")
                return

            # Convert to PIL Image
            pil_img = PILImage.fromarray(rgb_data, 'RGB')

            # 2. 目标分辨率
            desired_height = 1080
            desired_width = 1920
            # 3. 先计算缩放比例，用 PIL resize 进行缩放
            W, H = pil_img.size
            scale_H, scale_W = desired_height / H, desired_width / W
            new_H, new_W = int(H * scale_H), int(W * scale_W)
            if (new_H, new_W) != (H, W):
                pil_img = pil_img.resize((new_W, new_H), PILImage.BILINEAR)
            img_np = np.array(pil_img)  # PIL Image → numpy
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

            # 5. 如果缩放后仍不满足尺寸，进行 padding 或 center_crop
            _, H_resized, W_resized = img_tensor.shape
            pad_height = desired_height - H_resized
            pad_width = desired_width - W_resized

            if pad_height > 0 or pad_width > 0:  # 需要 padding
                pad_top = pad_height // 2
                pad_bottom = pad_height - pad_top
                pad_left = pad_width // 2
                pad_right = pad_width - pad_left

                img_tensor = TF.pad(
                    img_tensor,
                    (pad_left, pad_top, pad_right, pad_bottom),
                )

            elif pad_height < 0 or pad_width < 0:  # 需要裁剪
                img_tensor = TF.center_crop(img_tensor, (desired_height, desired_width))

            # 6. 检查最终尺寸是否正确
            assert img_tensor.shape[1] == desired_height and img_tensor.shape[2] == desired_width, \
                f"Final image shape {img_tensor.shape} != ({desired_height}, {desired_width})"

            # show image if needed
            show_tensor(img_tensor)

            with self.lock:
                self.image_batch.append((img_tensor, pose, localpose))
            self.last_image_time = current_time

        except Exception as e:
            self.get_logger().error(f"Error in image callback: {str(e)}")
    
    def save_poses_to_file(self):
        global timestamp
        if not self.poses:
            self.get_logger().info("No poses to save.")
            return
        
        filename = "robot_poses.txt"
        timestamped_filename = f"/home/zjy/code/myk/fundation_navigation_model_go2/fundation_navigation_model_go2/logs/navflow{timestamp}.txt"

        try:
            with open(filename, 'w') as f:
                f.write("x,y,z,qw,qx,qy,qz\n")  # 写入标题行
                for pose in self.poses:
                    # breakpoint()
                    f.write(f"{pose.position.x},{pose.position.y},{pose.position.z},{pose.orientation.x},{pose.orientation.y},{pose.orientation.z},{pose.orientation.w}\n")
            self.get_logger().info(f"Successfully saved {len(self.poses)} poses to {filename}")
            with open(timestamped_filename, 'w') as f:
                f.write("x,y,z,qw,qx,qy,qz\n") 
                for pose in self.poses:
                    f.write(f"{pose.position.x},{pose.position.y},{pose.position.z},{pose.orientation.x},{pose.orientation.y},{pose.orientation.z},{pose.orientation.w}\n")
            self.get_logger().info(f"Successfully saved {len(self.poses)} poses to {timestamped_filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to save poses: {str(e)}")

    def odom_callback(self, msg):
        """里程计回调函数，存储位姿和时间戳"""
        self.odom_record(msg)

        try:
            with self.lock:
                # 存储时间戳和位姿
                self.pose_buffer.append((msg.header.stamp, msg.pose))
                dx = target_xy[0][0]-  msg.pose.pose.position.x
                dy = target_xy[0][1]-  msg.pose.pose.position.y
                pose_diff = math.sqrt(dx**2+dy**2)
                # self.get_logger().info(f"pose_diff = {pose_diff}")

                # if pose_diff < self.distance_threshold:
                #     if len(target_xy) > 1:
                #         del target_xy[0]  # 删除索引0的元素
                #     else:
                #         reached_goal_msg = Bool()
                #         reached_goal_msg.data = bool(True)
                #         self.goal_pub.publish(reached_goal_msg)
                #         self.get_logger().info("Go2 arrived. exit.")
                #         node.destroy_node()
                #         rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Error in odom callback: {str(e)}")
    def find_closest_localpose(self, image_stamp):
        """查找与图像时间戳最接近的位姿"""
        with self.lock:
            if not self.localpose_buffer:
                return None
            
            image_nsec = image_stamp.nanosec + image_stamp.sec * 1e9

            # 简单的线性搜索找到时间最接近的位姿
            closest_pose = None
            # min_diff = float('inf')
            
            for pose_stamp, pose in self.localpose_buffer:
                # breakpoint()
                pose_nsec = pose_stamp.nanosec + pose_stamp.sec * 1e9
                time_diff = pose_nsec - image_nsec

                if abs(time_diff) < 1e8 or time_diff > 0:
                    closest_pose = pose
                    break
                elif not self.localpose_buffer:
                    # 如果时间差过大，删除旧数据（避免未来重复计算）
                    self.localpose_buffer.popleft()
                    
            # return closest_pose
            return closest_pose if closest_pose is not None else self.localpose_buffer[-1][1]
    def find_closest_pose(self, image_stamp):
        """查找与图像时间戳最接近的位姿"""
        with self.lock:
            if not self.pose_buffer:
                return None
            
            image_nsec = image_stamp.nanosec + image_stamp.sec * 1e9

            # 简单的线性搜索找到时间最接近的位姿
            closest_pose = None
            # min_diff = float('inf')
            
            for pose_stamp, pose in self.pose_buffer:
                # breakpoint()
                pose_nsec = pose_stamp.nanosec + pose_stamp.sec * 1e9
                time_diff = pose_nsec - image_nsec

                if abs(time_diff) < 1e8 or time_diff > 0:
                    closest_pose = pose
                    break
                elif not self.pose_buffer:
                    # 如果时间差过大，删除旧数据（避免未来重复计算）
                    self.pose_buffer.popleft()
                    
            # return closest_pose
            return closest_pose if closest_pose is not None else self.pose_buffer[-1][1]

    
    def process_frames(self):
        """处理线程，从队列中取出图像并查找对应位姿"""
        
        while rclpy.ok():
            # 5 frames
            process_flag = False
            if len(self.image_batch) == 21:
                # breakpoint()
                    # self.image_batch has updated
                with self.lock:
                    process_flag = True
            #         if self.image_batch_old != self.image_batch[-1][-1]:
            #             self.image_batch_old = copy.deepcopy(self.image_batch[-1][-1])
            #             process_flag = True
            if process_flag:
                # print('===============')
                # breakpoint()
                self.process_batch(self.image_batch)
                        
    def pose_to_matrix(self, pose):
        position = pose[:3]
        rotation = R.from_quat(pose[3:])
        matrix = np.eye(4)
        matrix[:3, :3] = rotation.as_matrix()
        matrix[:3, 3] = position
        return matrix
    
    def poses_to_matrices(self, poses):
        positions = poses[:, :3]
        quats = poses[:, 3:]
        rotations = R.from_quat(quats)
        matrices = np.tile(np.eye(4), (poses.shape[0], 1, 1))
        matrices[:, :3, :3] = rotations.as_matrix()
        matrices[:, :3, 3] = positions
        return matrices
    
    def transform_poses(self, poses, current_pose_array):
        current_pose_matrix = self.pose_to_matrix(current_pose_array)
        current_pose_inv = np.linalg.inv(current_pose_matrix)
        pose_matrices = self.poses_to_matrices(poses)
        transformed_matrices = np.matmul(current_pose_inv[np.newaxis, :, :], pose_matrices)
        positions = transformed_matrices[:, :3, 3]
        return positions
               
    
    def transform_target_pose(self, target_pose, current_pose_array):
        current_pose_matrix = self.pose_to_matrix(current_pose_array)
        current_pose_inv = np.linalg.inv(current_pose_matrix)
        target_pose_matrix = self.pose_to_matrix(target_pose)
        transformed_target_matrix = np.matmul(current_pose_inv, target_pose_matrix)
        target_position = transformed_target_matrix[:3, 3]
        # breakpoint()
        return target_position
    
    @staticmethod
    def resample_waypoints_equal_distance(waypoints, num_points=None):
        """
        对路径进行等间距采样，保持斜率
        
        Args:
            waypoints: torch.Tensor, shape [batch, num_waypoints, 2]
            num_points: int, 采样后的点数，如果为None则保持原点数
        
        Returns:
            resampled: torch.Tensor, shape [batch, num_points, 2]
        """
        batch_size, num_wp, _ = waypoints.shape
        if num_points is None:
            num_points = num_wp
        
        device = waypoints.device
        resampled = []
        
        for b in range(batch_size):
            wp = waypoints[b].cpu().numpy()  # [num_wp, 2]
            
            # 计算累积距离
            diffs = np.diff(wp, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            cumulative_dist = np.concatenate([[0], np.cumsum(distances)])
            total_length = cumulative_dist[-1]
            
            if total_length < 1e-6:  # 如果路径太短，直接返回
                resampled.append(wp)
                continue
            
            # 等间距采样点
            if num_points > 1:
                sample_distances = np.linspace(0, total_length, num_points)
            else:
                sample_distances = [total_length]
            
            # 使用线性插值
            resampled_wp = np.zeros((num_points, 2))
            for i, s_dist in enumerate(sample_distances):
                # 找到 s_dist 所在的区间
                idx = np.searchsorted(cumulative_dist, s_dist, side='right') - 1
                idx = np.clip(idx, 0, len(cumulative_dist) - 2)
                
                # 线性插值
                if idx < len(cumulative_dist) - 1:
                    t = (s_dist - cumulative_dist[idx]) / (cumulative_dist[idx + 1] - cumulative_dist[idx] + 1e-8)
                    t = np.clip(t, 0, 1)
                    resampled_wp[i] = wp[idx] + t * (wp[idx + 1] - wp[idx])
                else:
                    resampled_wp[i] = wp[-1]
            
            resampled.append(resampled_wp)
        
        return torch.tensor(np.array(resampled), device=device, dtype=waypoints.dtype)

    def process_batch(self, batch):
        global image_count
        """Process a collected batch of images and poses for ONNX inference."""
        self.get_logger().info(f"Processing batch of {len(batch)} frames")

        # --- STEP 1: Create a snapshot of the current batch ---
        # Use list() to convert the deque to a static list so that subsequent
        # ONNX inference (which can take ~250 ms) is not affected by new
        # frames being pushed into self.image_batch by ROS callbacks.
        with self.lock:
            current_batch_snapshot = list(batch)

        if not current_batch_snapshot:
            self.get_logger().warning("Empty batch snapshot, skip processing")
            return

        # Anchor all saved data (image, poses, etc.) to the latest frame
        # in the snapshot. This represents the current time for inference.
        snap_image, snap_pose, snap_localpose = current_batch_snapshot[-1]

        # --- STEP 2: Prepare tensors and poses from the snapshot ---
        image_list = []
        pose_list = []
        for image, pose, localpose in current_batch_snapshot:
            image_list.append(image)
            pose_list.append([
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z,
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w,
            ])

        # Record the local pose snapshot aligned with the last image in
        # the batch; this will be saved for later visualization/analysis.
        last_localpose_snapshot = [
            snap_localpose.pose.position.x,
            snap_localpose.pose.position.y,
            snap_localpose.pose.position.z,
            snap_localpose.pose.orientation.w,
            snap_localpose.pose.orientation.x,
            snap_localpose.pose.orientation.y,
            snap_localpose.pose.orientation.z,
        ]

        input_poses = np.array(pose_list)
        current_pose = input_poses[-1]
        # Transform poses into the current pose frame and then into
        # the robot->camera coordinate system.
        post_odom_temp = self.transform_poses(input_poses, current_pose)[:, [0, 1]]
        target_diff_temp = self.transform_target_pose(
            np.array(target_xy[0]), current_pose
        )[np.newaxis, [0, 1]]

        # robot -> camera
        post_odom = post_odom_temp.copy()
        target_diff = target_diff_temp.copy()
        post_odom[:, 0] = -post_odom_temp[:, 1]
        post_odom[:, 1] = post_odom_temp[:, 0]
        target_diff[:, 0] = -target_diff_temp[:, 1]
        target_diff[:, 1] = target_diff_temp[:, 0]

        self.get_logger().info(
            f"delta (x,y) = {target_diff[0][0], target_diff[0][1]}"
        )

        transformed_input_positions = np.concatenate([post_odom, target_diff], axis=0)

        # Original step scale computation (kept for consistency).
        step_scale = torch.tensor(
            np.linalg.norm(np.diff(post_odom, axis=0), axis=1).mean()
        )
        step_scale = torch.clamp(step_scale, min=1e-2)
        step_scale = torch.tensor(1)

        transformed_input_positions /= 0.25

        # --- STEP 3: Build tensors for ONNX model input ---
        video_frames = torch.stack(image_list, dim=0).unsqueeze(0).to(device)
        input_positions = (
            torch.tensor(transformed_input_positions).unsqueeze(0).to(device)
        )

        video_frames = video_frames.to(device)
        video_frames_raw = video_frames.clone()
        video_frames = torch.nn.functional.interpolate(
            video_frames[0], (288, 512)
        ).unsqueeze(0)

        # For pinhole camera (e.g., RealSense), we do not need fisheye-to-pinhole
        # undistortion. The fisheye conversion is kept commented out here.
        # video_frames = convert_fisheye_to_pinhole(
        #     K=K_global,
        #     dist=dist_global,
        #     device=video_frames.device,
        #     fisheye_tensor=video_frames[0],
        #     fov_x=math.degrees(2 * math.atan(1920 / (2 * 790))),
        #     fov_y=math.degrees(2 * math.atan(1080 / (2 * 790)))
        # ).unsqueeze(0)

        input_positions = input_positions.to(torch.float32).to(device)

        # Build goal and past trajectory features from poses.
        goal = input_positions[:, -1, :]
        goal = torch.cat(
            [goal[:, 1:2], -goal[:, 0:1]], dim=1
        ).reshape(1, 2)
        dis = torch.norm(goal, dim=1)
        goal[..., 0] = goal[..., 0].clamp(0.1, 500.0)
        goal[..., 1] = goal[..., 1].clamp(-50.0, 50.0)
        if torch.abs(torch.arctan2(goal[0, 1], goal[0, 0])) < np.pi / 9:
            goal[..., 1] = 0.0

        past_traj = torch.cat(
            [
                input_positions[:, :-2, 1:2],
                -input_positions[:, :-2, 0:1],
            ],
            dim=2,
        ).reshape(1, -1, 2) * 0.25
        past_traj = torch.cat([past_traj], dim=1)

        goal = torch.stack(
            [
                dis.clamp(0.1, 200.0) / 200.0,
                torch.cos(torch.arctan2(goal[:, 1], goal[:, 0])),
                torch.sin(torch.arctan2(goal[:, 1], goal[:, 0])),
            ],
            dim=-1,
        )

        past_mask = torch.ones([1, 21]).float().to(device)
        video_frames = torch.cat([video_frames], dim=1)

        input = video_frames[0]
        with torch.no_grad():
            metric_spacing = np.array([[0.0, 0.51, -0.26, 0.26]]).reshape(1, 4).astype(
                np.float32
            )
            embodiment_id = np.array([[0.0]]).reshape(1, 1).astype(np.int64)
            t0 = time.perf_counter()
            output = session.run(
                None,
                {
                    "observation": input.unsqueeze(0).cpu().numpy(),
                    "embodiment_id": embodiment_id,
                    "metric_spacing": metric_spacing,
                },
            )
            t1 = time.perf_counter()
            infer_dt = max(t1 - t0, 1e-8)
            infer_fps = 1.0 / infer_dt
            print(
                f"[perf] onnx inference: {infer_dt*1000:.2f} ms, "
                f"fps={infer_fps:.2f}"
            )

            traj_raw = output[0][:, 1:]
            score_raw = output[1][:, 1:]
            traj_raw = torch.from_numpy(traj_raw)
            score_raw = torch.from_numpy(score_raw)

            traj, score = trajectory_nms(traj_raw, score_raw)
            traj = traj[0]
            score = score[0]
            score = torch.softmax(score, dim=0)
            traj_final = traj[torch.argmax(score)]
            wp_pred = traj_final[..., :2].unsqueeze(0)

            wp_pred = torch.cat(
                [
                    torch.zeros_like(wp_pred[..., 0:1, :]),
                    wp_pred[..., :-1, :],
                ],
                dim=-2,
            )
            wp_pred[..., 0] = wp_pred[..., 0].clamp(0.0, 20.0)

        wp_pred = wp_pred.detach()[..., :2]

        wp_pred = torch.cat(
            [
                -wp_pred[:, :, 1:2],
                wp_pred[:, :, 0:1],
            ],
            dim=-1,
        )

        waypoint_msg = Float32MultiArray()
        wp_pred = wp_pred.detach().cpu().numpy()
        wp_pred_show = wp_pred

        # --- STEP 4: Save snapshot image and associated data ---
        img_np = snap_image.permute(1, 2, 0).detach().cpu().numpy()
        if img_np.dtype == np.float32 or img_np.dtype == np.float64:
            img_np = (img_np * 255).astype(np.uint8)

        enable_state = np.array(self.enable_2)
        folder_path = f"vis/navflow{timestamp}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        output_path = os.path.join(folder_path, f"sample_{image_count}.png")
        cv2.imwrite(output_path, img_np[..., ::-1])

        # Save pose snapshots so that the trajectory coordinates are fully
        # aligned with the image used for inference.
        pose_snapshot_np = np.array(
            [
                snap_pose.pose.position.x,
                snap_pose.pose.position.y,
                snap_pose.pose.position.z,
                snap_pose.pose.orientation.w,
                snap_pose.pose.orientation.x,
                snap_pose.pose.orientation.y,
                snap_pose.pose.orientation.z,
            ]
        )

        np.save(
            os.path.join(folder_path, f"sample_{image_count}_traj.npy"),
            traj_raw.numpy(),
        )
        np.save(
            os.path.join(folder_path, f"sample_{image_count}_score.npy"),
            score_raw.numpy(),
        )
        np.save(
            os.path.join(folder_path, f"sample_{image_count}_auto.npy"),
            enable_state,
        )
        np.save(
            os.path.join(folder_path, f"sample_{image_count}_pose.npy"),
            pose_snapshot_np,
        )
        np.save(
            os.path.join(folder_path, f"sample_{image_count}_pose_aligned.npy"),
            np.array(last_localpose_snapshot),
        )

        image_count += 1

        # --- STEP 5: Publish waypoint message ---
        wp_pred = wp_pred.tolist()

        self.get_logger().info(
            f"waypoint (x,y) = {wp_pred[0][-1][1], -wp_pred[0][-1][0]}"
        )
        if wp_pred[0][-1][1] > 0.1:
            tmp_list = []
            for i in range(0, len(wp_pred[0]), 2):
                tmp_list += [wp_pred[0][i][1]]
                tmp_list += [-wp_pred[0][i][0]]
            waypoint_msg.data = tmp_list
        else:
            tmp_list = []
            for i in range(0, len(wp_pred[0]), 2):
                tmp_list += [0.0]
                tmp_list += [0.0]
            waypoint_msg.data = tmp_list

        self.waypoint_pub.publish(waypoint_msg)
        reached_goal = False
        reached_goal_msg = Bool()
        reached_goal_msg.data = bool(reached_goal)
        self.goal_pub.publish(reached_goal_msg)


class DictNamespace(argparse.Namespace):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, DictNamespace(**value))
            else:
                setattr(self, key, value)


def parse_args():
    parser = argparse.ArgumentParser(description='Test UrbanNav model')
    # parser.add_argument('--config', type=str, default='mix_data_350.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint. If not provided, the latest checkpoint will be used.')
    parser.add_argument('--save_predictions', action='store_true', help='Whether to save predictions')
    parser.add_argument('--distance_threshold', type=float, default=10, help='defult=1m')

    args = parser.parse_args()
    return args


def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = DictNamespace(**cfg_dict)
    return cfg


def find_latest_checkpoint(checkpoint_dir):
    """
    Finds the latest checkpoint in the given directory based on modification time.
    
    Args:
        checkpoint_dir (str): Path to the directory containing checkpoints.
    
    Returns:
        str: Path to the latest checkpoint file.
    
    Raises:
        FileNotFoundError: If no checkpoint files are found in the directory.
    """
    checkpoint_pattern = os.path.join(checkpoint_dir, '*.ckpt')
    checkpoint_files = glob.glob(checkpoint_pattern)
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in directory: {checkpoint_dir}")
    
    # Sort checkpoints by modification time (latest first)
    checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_checkpoint = checkpoint_files[0]
    return latest_checkpoint


# def main():
#     args = parse_args()
#     cfg = load_config(args.config)
#     breakpoint()

#     rclpy.init(args=None)
#     node = ImagePoseSyncNode(args.distance_threshold)
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()
def load_yaml_config(filepath):
    with open(filepath, "r") as file:
        return yaml.safe_load(file)

def get_class_from_str(class_path):
    """
    Dynamically loads a class given its full import path as a string.
    Example: 'e2enav.datasets.sim_dataset.BaseSimDataset'
    """
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


import onnxruntime as ort
import numpy as np
import torch
def trajectory_nms(trajectory_raw, score_raw, max_trajectories=6, distance_threshold=2.0):
    """
    Apply NMS to trajectories based on endpoint distance
    Args:
        trajectory_raw: [B, N, T, 3] - batch, num_trajectories, time, (x,y,z)
        score_raw: [B, N] - confidence scores
        max_trajectories: int - number of trajectories to keep
        distance_threshold: float - minimum distance between trajectory endpoints
    Returns:
        selected_trajectories: [B, max_trajectories, T, 3]
        selected_scores: [B, max_trajectories]
        selected_indices: [B, max_trajectories]
    """
    B, N, T, _ = trajectory_raw.shape
    selected_trajectories = []
    selected_scores = []
    selected_indices = []
    for b in range(B):
        # Get trajectories and scores for this batch
        trajs = trajectory_raw[b]  # [N, T, 3]
        scores = score_raw[b]      # [N]
        # Sort by score (descending)
        sorted_indices = torch.argsort(scores, descending=True)
        # NMS selection
        keep_indices = []
        for i in sorted_indices:
            if len(keep_indices) >= max_trajectories:
                break
            # Check if this trajectory is too close to any kept trajectory
            should_keep = True
            for kept_idx in keep_indices:
                # Calculate distance between trajectory endpoints
                endpoint_dist = torch.norm(trajs[i][-1, :2] - trajs[kept_idx][-1, :2])
                if endpoint_dist < distance_threshold:
                    should_keep = False
                    break
            if should_keep:
                keep_indices.append(i)
        # Pad if we don't have enough trajectories
        while len(keep_indices) < max_trajectories:
            keep_indices.append(keep_indices[-1] if keep_indices else 0)
        keep_indices = torch.tensor(keep_indices[:max_trajectories])
        selected_trajectories.append(trajs[keep_indices])
        selected_scores.append(scores[keep_indices])
        selected_indices.append(keep_indices)
    return (torch.stack(selected_trajectories),
            torch.stack(selected_scores))
 
if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    image_count = 0

    try:
        target_xy = load_target_positions_from_file()
        print(f"Loaded target position from file: {target_xy}")
    except Exception as e:
        print(f"Error loading target position, using default: {str(e)}")
        target_xy = [[3, 0]] 

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args = parse_args()
    # config = load_yaml_config(args.config)

    # model_cls = get_class_from_str(config['model']['model_path'])
    # load ckpt
    # ckpt_path = 'ours_350.ckpt'
    # checkpoint = torch.load(ckpt_path, map_location='cpu')

    # # Handle different checkpoint formats
    # if 'state_dict' in checkpoint:
    #     state_dict = checkpoint['state_dict']
    # elif 'model_state_dict' in checkpoint:
    #     state_dict = checkpoint['model_state_dict']
    # else:
    #     state_dict = checkpoint
        
    # model = model_cls(config=config['model'])
    # model.load_state_dict(state_dict, strict=True)

    # # additional_layer = torch.load('agents.pt', map_location='cpu')

    # model = model.to(device)
    # model.eval()

    # session = ort.InferenceSession("onnx_ckpts/s1_ours_coco_preference_goal_less.onnx", providers=["CUDAExecutionProvider"])
    # session = ort.InferenceSession("onnx_ckpts/s1_ours_coco_preference_original_goal_less.onnx", providers=["CUDAExecutionProvider"])
    # session = ort.InferenceSession("onnx_ckpts/s1_ours_coco_sft_goal_less.onnx", providers=["CUDAExecutionProvider"])
    session = ort.InferenceSession("onnx_ckpts/s1_ours_coco_sft_stage2_goal_less.onnx", providers=["CUDAExecutionProvider"])
    

    K_global = np.array([
        [790 * 512 / 1920, 0.0, 512 / 2.],
        [0.0, 790 * 288 / 1080, 288 / 2.],
        [0.0, 0.0, 1.0]
    ])
    dist_global = np.array([-0.03054249, 0.00974967, -0.00679687, 0.00238449, 0.0]) 
    # converter = FisheyeToPinholeConverter(K, dist, device='cuda')

    # ros
    rclpy.init(args=None)
    # Fixed network interface (sys.argv[1]) for Unitree SDK channel init
    # ChannelFactoryInitialize(0, "wlp0s20f3")
    node = ImagePoseSyncNode(args.distance_threshold)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_poses_to_file()
        node.destroy_node()
        rclpy.shutdown()

#  python test_ros2.py --config config/finetune.yaml --checkpoint ckpt/finetune.ckpt --distance_threshold 0.5
