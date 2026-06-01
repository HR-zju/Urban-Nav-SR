"""2D pose EKF for fusing IMU angular rate, GPS positions, and optional yaw.

State: [x, y, theta] in meters and radians. Positions are local ENU meters
relative to an initial reference lat/lng using equirectangular approximation.

This EKF uses a minimal process model: theta is propagated by angular rate
inputs (omega_z), while x/y are corrected by GPS measurements. Process noise
on x/y should be large if no odometry translational input is available.
"""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass


EARTH_RADIUS_M = 6371000.0


def latlng_to_local_meters(lat: float, lng: float, lat0: float, lng0: float) -> tuple[float, float]:
    # Equirectangular projection around lat0
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lng - lng0)
    x = EARTH_RADIUS_M * dlon * math.cos(math.radians(lat0))
    y = EARTH_RADIUS_M * dlat
    return x, y


def local_meters_to_latlng(x: float, y: float, lat0: float, lng0: float) -> tuple[float, float]:
    dlat = y / EARTH_RADIUS_M
    dlon = x / (EARTH_RADIUS_M * math.cos(math.radians(lat0)))
    lat = lat0 + math.degrees(dlat)
    lng = lng0 + math.degrees(dlon)
    return lat, lng


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass
class PoseEKF:
    lat0: float
    lng0: float
    x: float
    y: float
    theta: float
    P: np.ndarray

    @classmethod
    def init_from_latlng_heading(
        cls,
        lat: float,
        lng: float,
        heading_deg: float,
        pos_var: float = 25.0,
        theta_var: float = 0.1,
    ):
        x, y = latlng_to_local_meters(lat, lng, lat, lng)
        theta = math.radians(heading_deg)
        P = np.diag([pos_var, pos_var, theta_var])
        return cls(lat0=lat, lng0=lng, x=x, y=y, theta=theta, P=P)

    def predict(self, omega_z: float, dt: float, q_pos: float = 1e-3, q_theta: float = 1e-2) -> None:
        """Predict state using IMU angular rate.

        q_pos and q_theta are continuous-time-like noise strengths.
        """
        self.theta = _wrap_to_pi(self.theta + float(omega_z) * float(dt))
        dt = max(float(dt), 1e-3)
        Q = np.diag([q_pos * dt, q_pos * dt, q_theta * dt])
        self.P = self.P + Q

    def update_gps(self, lat: float, lng: float, r_xy: float = 25.0) -> None:
        # Convert GPS to local meters in same frame
        z_x, z_y = latlng_to_local_meters(lat, lng, self.lat0, self.lng0)
        # Measurement model z = H x + v where H = [[1,0,0],[0,1,0]]
        H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        z = np.array([z_x, z_y])
        x_state = np.array([self.x, self.y, self.theta])
        y_resid = z - H.dot(x_state)
        S = H.dot(self.P).dot(H.T) + np.diag([r_xy, r_xy])
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        x_new = x_state + K.dot(y_resid)
        I = np.eye(3)
        self.P = (I - K.dot(H)).dot(self.P)
        self.x = float(x_new[0])
        self.y = float(x_new[1])
        # theta is unchanged by GPS position update

    def update_gps_and_yaw(
        self,
        lat: float,
        lng: float,
        yaw_deg: float | None = None,
        r_xy: float = 25.0,
        r_theta: float = 0.5,
    ) -> None:
        """Update with GPS x/y and optional yaw observation (degrees).

        yaw_deg may be None to indicate only position update.
        """
        z_x, z_y = latlng_to_local_meters(lat, lng, self.lat0, self.lng0)
        if yaw_deg is None:
            return self.update_gps(lat, lng, r_xy=r_xy)

        z_theta = math.radians(float(yaw_deg))
        # Measurement model z = [x, y, theta]
        H = np.eye(3)
        z = np.array([z_x, z_y, z_theta])
        x_state = np.array([self.x, self.y, self.theta])

        # Residual: position straightforward, theta needs wrapping
        y_resid = z - H.dot(x_state)
        y_resid[2] = (y_resid[2] + math.pi) % (2 * math.pi) - math.pi

        S = H.dot(self.P).dot(H.T) + np.diag([r_xy, r_xy, r_theta])
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        x_new = x_state + K.dot(y_resid)
        I = np.eye(3)
        self.P = (I - K.dot(H)).dot(self.P)
        self.x = float(x_new[0])
        self.y = float(x_new[1])
        self.theta = float((x_new[2] + math.pi) % (2 * math.pi) - math.pi)

    def get_latlng(self) -> tuple[float, float]:
        return local_meters_to_latlng(self.x, self.y, self.lat0, self.lng0)

    def get_heading_deg(self) -> float:
        deg = math.degrees(self.theta) % 360.0
        return deg
