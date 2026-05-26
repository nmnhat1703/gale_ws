"""Integration checks for NMPC state-vector assembly."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from nmpc_s500.nmpc_node import NmpcNode, POSE_VEL_SYNC_THRESHOLD_S


def _pose_msg(x, y, z, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
    return SimpleNamespace(
        pose=SimpleNamespace(
            position=SimpleNamespace(x=x, y=y, z=z),
            orientation=SimpleNamespace(x=qx, y=qy, z=qz, w=qw),
        )
    )


def _velocity_msg(vx, vy, vz):
    return SimpleNamespace(
        twist=SimpleNamespace(
            linear=SimpleNamespace(x=vx, y=vy, z=vz),
        )
    )


class TestStateAssembly(unittest.TestCase):
    """Exercise NmpcNode._build_state_vector without a running ROS master."""

    def _make_node(self):
        def get_param(name, default=None):
            values = {
                "~platform": "sim_iris",
                "~control_rate_hz": 50,
                "~trajectory": "hover",
                "~hover_position": [0.0, 0.0, 1.5],
                "~hover_yaw": 0.0,
                "~enable_offboard_on_start": False,
            }
            return values.get(name, default)

        platform_cfg = SimpleNamespace(
            name="sim_iris",
            mass_kg=1.5,
            max_thrust_n=27.0,
            hover_thrust_normalised=0.5,
        )

        patches = [
            patch("nmpc_s500.nmpc_node.rospy.init_node"),
            patch("nmpc_s500.nmpc_node.rospy.get_param", side_effect=get_param),
            patch("nmpc_s500.nmpc_node.rospy.loginfo"),
            patch("nmpc_s500.nmpc_node.rospy.logwarn_throttle"),
            patch("nmpc_s500.nmpc_node.rospy.logerr"),
            patch("nmpc_s500.nmpc_node.rospy.Publisher", return_value=MagicMock()),
            patch("nmpc_s500.nmpc_node.rospy.Subscriber", return_value=MagicMock()),
            patch("nmpc_s500.nmpc_node.rospy.wait_for_service"),
            patch("nmpc_s500.nmpc_node.rospy.ServiceProxy", return_value=MagicMock()),
            patch("nmpc_s500.nmpc_node.rospy.Duration", side_effect=lambda value: value),
            patch("nmpc_s500.nmpc_node.rospy.Timer", return_value=MagicMock()),
            patch("nmpc_s500.nmpc_node.rospy.on_shutdown"),
            patch("nmpc_s500.nmpc_node.load_platform_config", return_value=platform_cfg),
            patch("nmpc_s500.nmpc_node.create_solver", return_value=MagicMock()),
        ]

        with ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            return NmpcNode()

    def test_pure_east_position_and_velocity_convert_to_ned_y(self):
        """Pure ENU east maps to NED +y; FLU/ENU identity means NED yaw=pi/2."""
        node = self._make_node()
        node.pose = _pose_msg(1.0, 0.0, 0.0)
        node.velocity = _velocity_msg(2.0, 0.0, 0.0)
        node.pose_timestamp = 10.0
        node.vel_timestamp = 10.0

        state = node._build_state_vector()

        np.testing.assert_allclose(
            state,
            [0.0, 1.0, -0.0, 0.0, 2.0, -0.0, 0.0, 0.0, np.pi / 2],
            atol=1e-9,
        )

    def test_pure_up_position_and_velocity_convert_to_ned_down_negative(self):
        """Pure ENU up maps to NED -z; FLU/ENU identity means NED yaw=pi/2."""
        node = self._make_node()
        node.pose = _pose_msg(0.0, 0.0, 1.5)
        node.velocity = _velocity_msg(0.0, 0.0, 0.5)
        node.pose_timestamp = 10.0
        node.vel_timestamp = 10.0

        state = node._build_state_vector()

        np.testing.assert_allclose(
            state,
            [0.0, 0.0, -1.5, 0.0, 0.0, -0.5, 0.0, 0.0, np.pi / 2],
            atol=1e-9,
        )

    def test_drone_facing_north_level_produces_zero_yaw(self):
        """Drone facing north in ENU/FLU should have yaw=0 in NED/FRD state."""
        node = self._make_node()
        q = Rotation.from_euler('z', np.pi / 2, degrees=False).as_quat()
        node.pose = _pose_msg(0.0, 0.0, 0.0, qx=q[0], qy=q[1], qz=q[2], qw=q[3])
        node.velocity = _velocity_msg(0.0, 0.0, 0.0)
        node.pose_timestamp = 10.0
        node.vel_timestamp = 10.0

        state = node._build_state_vector()

        np.testing.assert_allclose(state[6:9], [0.0, 0.0, 0.0], atol=1e-9)

    def test_drone_east_rolled_right_produces_positive_roll(self):
        """Drone facing east in ENU/FLU, rolled +30° right.

        Identity in FLU/ENU = facing east (yaw=+pi/2 in NED/FRD).
        Roll right is +30° about FLU+x. In FRD/NED solver frame
        this is roll=+pi/6, pitch=0, yaw=+pi/2.
        """
        node = self._make_node()
        q = Rotation.from_euler('x', np.pi / 6, degrees=False).as_quat()
        node.pose = _pose_msg(0.0, 0.0, 0.0, qx=q[0], qy=q[1], qz=q[2], qw=q[3])
        node.velocity = _velocity_msg(0.0, 0.0, 0.0)
        node.pose_timestamp = 10.0
        node.vel_timestamp = 10.0

        state = node._build_state_vector()

        np.testing.assert_allclose(state[6:9], [np.pi / 6, 0.0, np.pi / 2], atol=1e-9)

    def test_drone_east_pitched_up_produces_positive_pitch(self):
        """Drone facing east in ENU/FLU, pitched 20° nose-up.

        Nose-up in FLU is NEGATIVE rotation about FLU+y (since +y is
        left and right-hand rule about +y rotates forward toward down).
        In FRD/NED solver frame this is roll=0, pitch=+pi/9, yaw=+pi/2.
        """
        node = self._make_node()
        q = Rotation.from_euler('y', -np.pi / 9, degrees=False).as_quat()
        node.pose = _pose_msg(0.0, 0.0, 0.0, qx=q[0], qy=q[1], qz=q[2], qw=q[3])
        node.velocity = _velocity_msg(0.0, 0.0, 0.0)
        node.pose_timestamp = 10.0
        node.vel_timestamp = 10.0

        state = node._build_state_vector()

        np.testing.assert_allclose(state[6:9], [0.0, np.pi / 9, np.pi / 2], atol=1e-9)

    def test_hover_position_param_is_stored_in_ned(self):
        node = self._make_node()

        np.testing.assert_allclose(
            node.hover_position_ned,
            np.array([0.0, 0.0, -1.5]),
            atol=1e-9,
        )

    def test_hover_yaw_param_is_stored_in_ned(self):
        """Default ENU hover_yaw=0 faces east, which is NED yaw=pi/2."""
        node = self._make_node()

        np.testing.assert_allclose(node.hover_yaw_ned, np.pi / 2, atol=1e-9)

    def test_publish_attitude_target_converts_solver_frd_rates_to_mavros_flu(self):
        node = self._make_node()
        solver_u = np.array([node.platform_cfg.mass_kg * 9.81, 0.0, 0.5, 0.0])

        with patch("nmpc_s500.nmpc_node.rospy.Time.now", return_value=0.0):
            node._publish_attitude_target(solver_u)

        msg = node.attitude_pub.publish.call_args[0][0]
        self.assertAlmostEqual(msg.body_rate.x, 0.0)
        self.assertAlmostEqual(msg.body_rate.y, -0.5)
        self.assertAlmostEqual(msg.body_rate.z, -0.0)

    def test_timestamp_guard_still_returns_none_for_unsynced_pose_velocity(self):
        node = self._make_node()
        node.pose = _pose_msg(1.0, 0.0, 0.0)
        node.velocity = _velocity_msg(2.0, 0.0, 0.0)
        node.pose_timestamp = 10.0
        node.vel_timestamp = 10.0 + POSE_VEL_SYNC_THRESHOLD_S + 0.001

        with patch("nmpc_s500.nmpc_node.rospy.logwarn_throttle"):
            self.assertIsNone(node._build_state_vector())


if __name__ == "__main__":
    unittest.main(verbosity=2)
