"""NMPC controller for S500 quadrotor with ROS 1 and MAVROS."""

__version__ = '0.1.0'
__author__ = 'nhat'
__license__ = 'MIT'

from .platform_config import PlatformConfig, load_platform_config
from .frames import quaternion_to_euler_zyx

__all__ = [
    'PlatformConfig',
    'load_platform_config',
    'quaternion_to_euler_zyx',
]
