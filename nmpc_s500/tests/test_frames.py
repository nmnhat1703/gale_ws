"""Unit tests for frame conversion utilities.

Tests the quaternion → ZYX Euler angle conversion using scipy.spatial.transform.Rotation.
"""

import unittest
import numpy as np
from scipy.spatial.transform import Rotation

# Import the function under test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from nmpc_s500.frames import (
    enu_body_to_ned_body_rates,
    enu_to_ned_position,
    enu_to_ned_velocity,
    flu_to_frd_quaternion,
    frd_to_flu_quaternion,
    ned_to_enu_position,
    quaternion_to_euler_zyx,
)


class TestFrameTransforms(unittest.TestCase):
    """Test ENU/FLU to NED/FRD frame conversions."""

    def test_position_round_trip(self):
        """ENU -> NED -> ENU position should recover the original vector."""
        p_ned = enu_to_ned_position(1.0, 2.0, 3.0)
        p_enu = ned_to_enu_position(*p_ned)
        np.testing.assert_allclose(p_enu, (1.0, 2.0, 3.0), atol=1e-9)

    def test_position_formula(self):
        """ENU position (x, y, z) maps to NED (y, x, -z)."""
        self.assertEqual(enu_to_ned_position(1.0, 2.0, 3.0), (2.0, 1.0, -3.0))

    def test_velocity_formula(self):
        """ENU velocity (vx, vy, vz) maps to NED (vy, vx, -vz)."""
        self.assertEqual(enu_to_ned_velocity(1.0, 2.0, 3.0), (2.0, 1.0, -3.0))

    def test_body_rate_round_trip(self):
        """FLU -> FRD body-rate conversion is self-inverse."""
        rates_ned = enu_body_to_ned_body_rates(0.1, 0.2, 0.3)
        rates_enu = enu_body_to_ned_body_rates(*rates_ned)
        np.testing.assert_allclose(rates_enu, (0.1, 0.2, 0.3), atol=1e-9)

    def test_body_rate_formula(self):
        """FLU body rates (p, q, r) map to FRD (p, -q, -r)."""
        self.assertEqual(enu_body_to_ned_body_rates(0.1, 0.2, 0.3), (0.1, -0.2, -0.3))

    def test_quaternion_identity_mapping(self):
        """ENU/FLU identity maps to a 90-degree yaw in NED/FRD convention."""
        qx, qy, qz, qw = flu_to_frd_quaternion(0.0, 0.0, 0.0, 1.0)
        self.assertAlmostEqual(qx, 0.0, places=9)
        self.assertAlmostEqual(qy, 0.0, places=9)
        self.assertAlmostEqual(abs(qz), np.sqrt(0.5), places=9)
        self.assertAlmostEqual(qw, np.sqrt(0.5), places=9)

    def test_quaternion_round_trip_with_inverse(self):
        """FLU -> FRD -> FLU quaternion conversion should recover the input."""
        q_in = Rotation.from_euler('zyx', [0.2, -0.1, 0.3], degrees=False).as_quat()
        q_frd = flu_to_frd_quaternion(*q_in)
        q_out = np.array(frd_to_flu_quaternion(*q_frd))

        if np.dot(q_in, q_out) < 0.0:
            q_out = -q_out
        np.testing.assert_allclose(q_out, q_in, atol=1e-9)

    def test_quaternion_identity_euler_in_ned_frd(self):
        """Converted ENU/FLU identity should decode as yaw=90deg in NED/FRD."""
        q_ned_frd = flu_to_frd_quaternion(0.0, 0.0, 0.0, 1.0)
        rpy = quaternion_to_euler_zyx(*q_ned_frd)
        np.testing.assert_allclose(rpy, [0.0, 0.0, np.pi / 2], atol=1e-9)


class TestQuaternionToEulerZYX(unittest.TestCase):
    """Test quaternion to ZYX Euler angle conversion."""

    def test_identity_quaternion(self):
        """Identity quaternion should return zero roll, pitch, yaw."""
        # Identity: qx=0, qy=0, qz=0, qw=1
        rpy = quaternion_to_euler_zyx(0.0, 0.0, 0.0, 1.0)
        np.testing.assert_allclose(rpy, [0.0, 0.0, 0.0], atol=1e-10)

    def test_pure_yaw_90_degrees(self):
        """Pure yaw rotation by 90 degrees (π/2 radians)."""
        # Create a quaternion for 90° rotation around Z (yaw) using Rotation
        rot = Rotation.from_euler('zyx', [np.pi / 2, 0.0, 0.0], degrees=False)
        qx, qy, qz, qw = rot.as_quat()

        rpy = quaternion_to_euler_zyx(qx, qy, qz, qw)

        # Expected: [0, 0, π/2]
        np.testing.assert_allclose(rpy, [0.0, 0.0, np.pi / 2], atol=1e-10)

    def test_pure_roll_45_degrees(self):
        """Pure roll rotation by 45 degrees (π/4 radians)."""
        # Create a quaternion for 45° rotation around X (roll) using Rotation
        rot = Rotation.from_euler('zyx', [0.0, 0.0, np.pi / 4], degrees=False)
        qx, qy, qz, qw = rot.as_quat()

        rpy = quaternion_to_euler_zyx(qx, qy, qz, qw)

        # Expected: [π/4, 0, 0]
        np.testing.assert_allclose(rpy, [np.pi / 4, 0.0, 0.0], atol=1e-10)

    def test_pure_pitch_30_degrees(self):
        """Pure pitch rotation by 30 degrees (π/6 radians)."""
        # Create a quaternion for 30° rotation around Y (pitch) using Rotation
        rot = Rotation.from_euler('zyx', [0.0, np.pi / 6, 0.0], degrees=False)
        qx, qy, qz, qw = rot.as_quat()

        rpy = quaternion_to_euler_zyx(qx, qy, qz, qw)

        # Expected: [0, π/6, 0]
        np.testing.assert_allclose(rpy, [0.0, np.pi / 6, 0.0], atol=1e-10)

    def test_roundtrip_rpy_to_quat_to_rpy(self):
        """Round-trip conversion: RPY → quaternion → RPY should be identity."""
        # Test several RPY combinations
        test_cases = [
            [0.1, 0.2, 0.3],           # Small angles
            [np.pi / 6, np.pi / 4, np.pi / 3],  # 30°, 45°, 60°
            [-0.15, 0.05, -0.2],       # Negative angles
            [np.pi / 2, 0.0, 0.0],     # Large roll
            [0.0, np.pi / 3, 0.0],     # Large pitch
            [0.0, 0.0, np.pi / 2],     # Large yaw
        ]

        for rpy_input in test_cases:
            with self.subTest(rpy_input=rpy_input):
                # Convert RPY -> quaternion. scipy 'zyx' expects [yaw, pitch, roll].
                roll, pitch, yaw = rpy_input
                rot = Rotation.from_euler('zyx', [yaw, pitch, roll], degrees=False)
                qx, qy, qz, qw = rot.as_quat()

                # Convert quaternion → RPY
                rpy_output = quaternion_to_euler_zyx(qx, qy, qz, qw)

                # Should match to within 1e-9
                np.testing.assert_allclose(
                    rpy_output, rpy_input,
                    atol=1e-9,
                    err_msg=f"Round-trip failed for input {rpy_input}"
                )

    def test_combined_rotation(self):
        """Test combined rotation (all three angles non-zero)."""
        # Create a rotation with all three components
        rpy_input = [0.15, 0.25, 0.35]
        roll, pitch, yaw = rpy_input
        rot = Rotation.from_euler('zyx', [yaw, pitch, roll], degrees=False)
        qx, qy, qz, qw = rot.as_quat()

        rpy_output = quaternion_to_euler_zyx(qx, qy, qz, qw)

        np.testing.assert_allclose(rpy_output, rpy_input, atol=1e-10)

    def test_equivalence_with_scipy_rotation(self):
        """Verify that our function reverses scipy's [yaw, pitch, roll] output."""
        # Generate a random quaternion
        rot = Rotation.random(random_state=42)
        qx, qy, qz, qw = rot.as_quat()

        # Our function
        rpy_ours = quaternion_to_euler_zyx(qx, qy, qz, qw)

        # Scipy's function
        rpy_scipy = rot.as_euler('zyx', degrees=False)

        # scipy returns [yaw, pitch, roll]; our function returns [roll, pitch, yaw].
        np.testing.assert_allclose(rpy_ours, [rpy_scipy[2], rpy_scipy[1], rpy_scipy[0]], atol=1e-14)

    def test_array_input_via_numpy(self):
        """Test that the function works with numpy array input."""
        # Create a quaternion and convert to array form
        rot = Rotation.from_euler('zyx', [0.1, 0.2, 0.3], degrees=False)
        qx, qy, qz, qw = rot.as_quat()

        # Call with individual float args
        rpy1 = quaternion_to_euler_zyx(qx, qy, qz, qw)

        # Call with array unpacking
        quat_array = np.array([qx, qy, qz, qw])
        rpy2 = quaternion_to_euler_zyx(*quat_array)

        np.testing.assert_allclose(rpy1, rpy2, atol=1e-14)


if __name__ == "__main__":
    unittest.main(verbosity=2)
