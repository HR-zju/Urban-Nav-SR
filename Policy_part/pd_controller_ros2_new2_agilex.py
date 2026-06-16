import numpy as np
import yaml
from typing import Tuple

# ROS 2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Bool

from topic_names import (WAYPOINT_TOPIC, REACHED_GOAL_TOPIC)
from ros_data import ROSData
# from utils import clip_angle


# CONSTS
CONFIG_PATH = "config/robot.yaml"
with open(CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
VEL_TOPIC = robot_config["vel_navi_topic"]
DT = 1. #0.1 #1/robot_config["frame_rate"] * 4
RATE = 10
EPS = 1e-8
WAYPOINT_TIMEOUT = 1 # seconds # TODO: tune this
FLIP_ANG_VEL = np.pi/4
TURN_KP = 1.2
MIN_TURN_W = 0.18
TURN_DEADBAND_ANGLE = 0.05
MAX_DELTA_W = 0.12
TURN_SPEED_DECAY = 0.8

# GLOBALS
vel_msg = Twist()
# waypoint = ROSData(WAYPOINT_TIMEOUT, self.timer_callback)  # ~30Hz
waypoint = ROSData(WAYPOINT_TIMEOUT, name="waypoint")
 
reached_goal = False
reverse_mode = False
current_yaw = None

def clip_angle(theta) -> float:
    """Clip angle to [-pi, pi]"""
    theta %= 2 * np.pi
    if -np.pi < theta < np.pi:
        return theta
    return theta - 2 * np.pi
      

def pure_pursuit_control(waypoints, wheelbase=0.456, lookahead_dist=1.0):
    """
    waypoints: [(x, y, v), ...] in ego frame
    """
    # find lookahead point
    target = None
    waypoints = np.array(waypoints).reshape(-1, 2)
    for wp in waypoints:
        dist = np.hypot(wp[0], wp[1])
        if dist >= lookahead_dist:
            target = wp
            break

    if target is None:
        return 0.0

    x, y = target[0], target[1]

    # curvature
    curvature = 2 * y / (x**2 + y**2 + 1e-6)

    # steering angle
    steer = np.arctan(wheelbase * curvature)
    return steer


def pd_controller(waypoint, current_v, current_w) -> Tuple[float, float]:
    """PD controller for the robot.

    Select a lookahead point based on physical distance and compute
    linear/angular velocity from that point, with smoothing.
    """

    waypoints = np.asarray(waypoint, dtype=float).reshape(-1, 2)
    if waypoints.shape[0] == 0:
        return 0.0, 0.0

    # --- choose lookahead point based on distance ---
    dists = np.hypot(waypoints[:, 0], waypoints[:, 1])

    target_idx = None
    for i, dist in enumerate(dists):
        if 1.0 <= dist <= 1.5:
            target_idx = i
            break

    # if no point falls into [1.5, 2.0] m, fall back to the endpoint
    if target_idx is None:
        target_idx = len(waypoints) - 1

    dx, dy = waypoints[target_idx]

    # --- base control from lookahead point ---
    v = dx / DT  # / 2 changed here
    heading_error = np.arctan2(dy, dx)
    w = TURN_KP * heading_error / DT

    # --- simple progress metric along x for stop condition ---
    if waypoints.shape[0] > 1:
        dx_steps = waypoints[1:, 0] - waypoints[:-1, 0]
        dxmean = float(np.mean(dx_steps))
    else:
        dxmean = 0.0

    max_v = current_v + 0.4
    min_v = current_v - 0.5
    # angular velocity smoothing
    w = np.clip(w, current_w - MAX_DELTA_W, current_w + MAX_DELTA_W)
    if abs(heading_error) > TURN_DEADBAND_ANGLE and abs(w) < MIN_TURN_W:
        w = np.sign(heading_error) * MIN_TURN_W

    # acceleration smoothing and limits on linear velocity
    v = np.clip(v, min_v, max_v)
    v = np.clip(v, 0., MAX_V)

    # stop when almost at the end and turning is small
    if dxmean < 0.05 and np.abs(w) < 0.01:
        v *= 0.0

    # penalize linear velocity when angular velocity is large
    if np.abs(w) >= MIN_TURN_W:
        v *= np.exp(-TURN_SPEED_DECAY * np.abs(w))

    w = np.clip(w, -MAX_W, MAX_W)

    return float(v), float(w)


def estimate_curvature(waypoints, idx, eps=1e-6):
    """
    Estimate curvature using three consecutive waypoints.
    """
    p1 = np.array(waypoints[max(0, idx - 1)])
    p2 = np.array(waypoints[idx])
    p3 = np.array(waypoints[min(len(waypoints) - 1, idx + 1)])

    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p3 - p1)

    if a * b * c < eps:
        return 0.0

    # twice triangle area = |cross|
    area2 = np.cross(p2 - p1, p3 - p1)
    curvature = 2 * area2 / (a * b * c)

    return curvature


def waypoints_to_vw(
    waypoints,
    current_v,
    current_w,
    max_v=MAX_V,
    max_w=MAX_W,
    eps=1e-3,
):
    """
    waypoints: (N, 2) array, ego-frame spatial waypoints
    returns: (v, w)
    """

    waypoints = np.asarray(waypoints).reshape(-1, 2)

    # --- 1. speed-dependent lookahead (distance-based) ---
    lookahead_dist = max(2., 1. * current_v)

    idx = next(
        (i for i, wp in enumerate(waypoints)
         if np.hypot(wp[0], wp[1]) >= lookahead_dist),
        len(waypoints) - 1
    )
    idx = len(waypoints) - 2

    dx, dy = waypoints[idx]
    dist = np.hypot(dx, dy)

    # --- 2. curvature from geometry ---
    curvature = estimate_curvature(waypoints, idx)

    # --- 3. speed planning (NO dt) ---
    # slow down when curvature is large
    v_ref = min(max_v, dist)
    v_ref *= np.exp(-2.0 * abs(curvature))

    # --- 4. angular velocity ---
    w_ref = v_ref * curvature / 0.3
    #print(w_ref)
    # --- 5. rate limits (smoothness / safety) ---
    v = np.clip(v_ref, current_v - 1.5, current_v + 0.3)
    #w = np.clip(w_ref, current_w - 0.5, current_w + 0.5)
    w = w_ref
    # --- 6. absolute limits ---
    v = np.clip(v, 0.01, max_v)
    w = np.clip(w, -max_w, max_w)

    return v, w


class PDController(Node):
    def __init__(self):
        super().__init__("pd_controller")
        self.waypoint_sub = self.create_subscription(Float32MultiArray, WAYPOINT_TOPIC, self.callback_drive, 10)
        self.reached_goal_sub = self.create_subscription(Bool, REACHED_GOAL_TOPIC, self.callback_reached_goal, 10)
        self.vel_pub = self.create_publisher(Twist, VEL_TOPIC, 10)

        self.timer = self.create_timer(1.0 / RATE, self.timer_callback)
        self.vel_msg = Twist()
        self.reached_goal = False
        self.waypoint = ROSData(WAYPOINT_TIMEOUT, name="waypoint")
        self.reverse_mode = False
        self.current_v = 0
        self.current_w = 0

    def callback_drive(self, waypoint_msg: Float32MultiArray):
        """Callback function for the waypoint subscriber"""
        self.get_logger().info("Setting waypoint")
        self.waypoint.set(waypoint_msg.data)
        
    def callback_reached_goal(self, reached_goal_msg: Bool):
        """Callback function for the reached goal subscriber"""
        self.reached_goal = reached_goal_msg.data

    def timer_callback(self):
        v, w = 0.0, 0.0
        if self.reached_goal:
            self.current_v = 0.0
            self.current_w = 0.0
            self.vel_msg = Twist()
            self.get_logger().info("Reached goal. Publishing zero cmd_vel.")
        elif self.waypoint.is_valid(verbose=True):
            v, w = pd_controller(self.waypoint.get(), self.current_v, self.current_w)
            self.current_v = v
            self.current_w = w

            if self.reverse_mode:
                v *= -1
            self.vel_msg.linear.x = v
            self.vel_msg.angular.z = w
            self.get_logger().info(f"Publishing new vel: {v}, {w}")
        else:
            self.current_v = 0.0
            self.current_w = 0.0
            self.vel_msg = Twist()
            self.get_logger().warn("No valid waypoint. Publishing zero cmd_vel.")

        self.vel_pub.publish(self.vel_msg)
        print('vel_msg = {}'.format(self.vel_msg))
        print(f'v, w = {v}, {w}')


def main(args=None):
    rclpy.init(args=args)
    pd_controller = PDController()
    rclpy.spin(pd_controller)
    pd_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
