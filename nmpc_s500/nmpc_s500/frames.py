"""Frame conversion utilities for MAVROS ENU/FLU and NMPC NED/FRD.

MAVROS publishes local pose and velocity in ENU world coordinates with a
ROS base_link body convention (FLU: forward, left, up). The upstream NMPC
model uses NED world coordinates with an aircraft body convention
(FRD: forward, right, down).
"""

import numpy as np
from scipy.spatial.transform import Rotation
from typing import Tuple


ENU_TO_NED = np.array([
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
])

FLU_TO_FRD = np.array([
    [1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
])


def enu_to_ned_position(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert ENU position to NED position. This transform is self-inverse."""
    return y, x, -z


def ned_to_enu_position(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert NED position to ENU position. This transform is self-inverse."""
    return y, x, -z


def enu_to_ned_velocity(vx: float, vy: float, vz: float) -> Tuple[float, float, float]:
    """Convert ENU velocity to NED velocity."""
    return vy, vx, -vz


def enu_body_to_ned_body_rates(p: float, q: float, r: float) -> Tuple[float, float, float]:
    """Convert FLU body rates to FRD body rates. This transform is self-inverse."""
    return p, -q, -r


def flu_to_frd_quaternion(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float, float]:
    """Convert an ENU/FLU orientation quaternion to the NED/FRD convention.

    ROS/MAVROS pose quaternions represent body FLU orientation in the ENU world.
    The NMPC model expects the same physical orientation represented as body FRD
    orientation in the NED world. In rotation-matrix form:

        R_ned_frd = R_enu_to_ned @ R_enu_flu @ R_frd_to_flu

    where ``R_frd_to_flu`` equals ``FLU_TO_FRD`` because the body-axis flip is
    self-inverse.
    """
    r_enu_flu = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    r_ned_frd = ENU_TO_NED @ r_enu_flu @ FLU_TO_FRD
    q_ned_frd = Rotation.from_matrix(r_ned_frd).as_quat()
    return tuple(q_ned_frd)


def frd_to_flu_quaternion(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float, float]:
    """Convert a NED/FRD orientation quaternion back to the ENU/FLU convention."""
    r_ned_frd = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    r_enu_flu = ENU_TO_NED @ r_ned_frd @ FLU_TO_FRD
    q_enu_flu = Rotation.from_matrix(r_enu_flu).as_quat()
    return tuple(q_enu_flu)


def quaternion_to_euler_zyx(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert quaternion to [roll, pitch, yaw] matching the NMPC model convention.

    The upstream NMPC model (acados_model.py) builds the world-to-body
    rotation as:
        R_world_body = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    This corresponds to scipy's intrinsic ZYX rotation, which is
    mathematically equivalent to extrinsic xyz. We use as_euler('xyz')
    because it returns [roll, pitch, yaw] directly in the desired order.

    IMPORTANT: as_euler('zyx') would return [yaw, pitch, roll] for a
    DIFFERENT decomposition (extrinsic zyx = intrinsic xyz) that does
    NOT match the model's rotation formula when yaw and roll/pitch are
    simultaneously non-zero.
    """
    return Rotation.from_quat([qx, qy, qz, qw]).as_euler('xyz', degrees=False)
