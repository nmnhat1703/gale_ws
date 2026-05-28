"""ROS 1 NMPC controller node for S500 quadrotor with MAVROS.

This node implements NMPC control using the acados solver, publishing
attitude setpoints via /mavros/setpoint_raw/attitude.

Frame convention: ENU (East-North-Up), matching MAVROS.
State: [x, y, z, vx, vy, vz, roll, pitch, yaw]
Input: [thrust_N, p, q, r] from solver → normalized thrust [0,1] + body rates
"""

import rospy
import sys
import os
import numpy as np
import time
from enum import Enum
from collections import deque

from mavros_msgs.msg import State, AttitudeTarget
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, TwistStamped
from mavros_msgs.srv import CommandBool, SetMode
from std_msgs.msg import String

from nmpc_s500.platform_config import load_platform_config
from nmpc_s500.solver_setup import create_solver
# TODO: rename to flip_body_rates_frd_flu - name says world frame, function operates on body frame.
from nmpc_s500.frames import (
    enu_body_to_ned_body_rates,
    enu_to_ned_position,
    enu_to_ned_velocity,
    enu_yaw_to_ned_yaw,
    flu_to_frd_quaternion,
    ned_to_enu_position,
    quaternion_to_euler_zyx,
    wrap_to_pi,
)


# ============================================================================
# CONSTANTS
# ============================================================================

# AttitudeTarget type_mask: ignore attitude quaternion, USE body rates, USE thrust
# Bit 7 (0x80): ignore attitude quaternion
# Bits 0-2 (0x07): body rates (don't set = use rates)
# Bit 6 (0x40): thrust (don't set = use thrust)
# Therefore: 0x80 = 128 decimal
ATTITUDE_TARGET_USE_RATES_AND_THRUST = 0x80  # 128: ignore quat, use rates and thrust

# Time constants
POSE_VEL_SYNC_THRESHOLD_S = 0.050  # 50 ms max timestamp difference
STATE_TIMEOUT_S = 0.500  # 500 ms without update = error
SOLVER_WARMUP_ITERATIONS = 10
SETPOINT_STREAM_DURATION_S = 2.5
ARM_WAIT_TIMEOUT_S = 2.0
OFFBOARD_WAIT_TIMEOUT_S = 2.0
MAX_CONSECUTIVE_SOLVER_FAILURES = 10
TAKEOFF_SETTLE_TIMEOUT_S = 15.0
TAKEOFF_SETTLE_HOLD_S = 1.0
TAKEOFF_VERTICAL_VELOCITY_TOLERANCE_MPS = 0.15
CRUISE_SPEED = 0.5
MIN_RAMP_S = 4.0
MAX_RAMP_S = 15.0
RAMP_TARGET_POS_TOL_M = 0.30
RAMP_TARGET_VEL_TOL_MPS = 0.30
LAND_TARGET_ENU = (0.0, 0.0, 1.0)
LAND_TARGET_YAW_ENU = 0.0
LAND_HOLD_SEC = 2.0
WAYPOINT_HOLD_SEC = 2.0
WAYPOINTS_ENU_RPY = np.array([
    [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [2.0, -1.0, 2.0, 0.0, 0.0, np.pi / 2.0],
    [0.0, 0.0, 2.0, 0.0, 0.0, np.pi],
], dtype=float)

# Default horizon parameters
DEFAULT_HORIZON = 2.0
DEFAULT_NUM_STEPS = 50

ACADOS_STATUS_LABELS = {
    -1: "EXCEPTION",
    0: "OK",
    1: "MAX_ITER",
    2: "NAN",
    3: "QP_FAIL",
    4: "QP_MAX_ITER",
}


class FlightPhase(Enum):
    """Flight state machine states."""
    INIT = 1
    WAIT_FOR_FCU = 2
    STREAM_SETPOINTS = 3
    REQUEST_OFFBOARD = 4
    RUN = 5
    IDLE = 6
    TAKEOFF_PX4 = 7


class TrajectoryMode(Enum):
    """Active NMPC reference mode."""
    HOVER = 1
    RAMP_TO_CIRCLE = 2
    CIRCLE = 3
    RAMP_TO_HOVER = 4
    LAND_APPROACH = 5
    LAND_HOLD = 6
    LAND_DESCEND = 7
    DONE = 8
    WAYPOINT_RAMP = 9
    WAYPOINT_HOLD = 10
    RAMP_TO_EIGHTLOOP = 11
    EIGHTLOOP = 12


class NmpcNode:
    """ROS 1 NMPC controller node for S500 quadrotor."""

    def __init__(self):
        """Initialize ROS node, load parameters, set up subscribers/publishers."""
        rospy.init_node('nmpc_node', anonymous=False)
        rospy.loginfo("[NMPC] Initializing node...")

        # ===== Load Parameters =====
        platform_name = rospy.get_param('~platform', 'sim_iris')
        self.control_rate_hz = rospy.get_param('~control_rate_hz', 50)
        self.trajectory_type = rospy.get_param('~trajectory', 'hover')
        # ROS param is ENU (user convention: z up = positive). Convert to NED
        # at capture so all downstream solver code sees solver-frame values.
        hover_pos_param = rospy.get_param('~hover_position', [0.0, 0.0, 1.5])
        hover_pos_enu = np.array(hover_pos_param, dtype=float)
        self.hover_position_ned = np.array(
            enu_to_ned_position(hover_pos_enu[0], hover_pos_enu[1], hover_pos_enu[2]),
            dtype=float,
        )
        # ROS param is ENU yaw (0 = east). Convert at capture so
        # self.hover_yaw_ned is always solver-frame NED yaw, wrapped to [-pi, pi].
        hover_yaw_enu = rospy.get_param('~hover_yaw', 0.0)
        self.hover_yaw_ned = enu_yaw_to_ned_yaw(hover_yaw_enu)
        self.enable_offboard_on_start = rospy.get_param('~enable_offboard_on_start', False)
        circle_center_param = rospy.get_param('~circle_center', [0.0, 0.0, 2.0])
        circle_center_enu = np.array(circle_center_param, dtype=float)
        self.circle_center_ned = np.array(
            enu_to_ned_position(circle_center_enu[0], circle_center_enu[1], circle_center_enu[2]),
            dtype=float,
        )
        self.circle_radius_m = float(rospy.get_param('~circle_radius_m', 3.0))
        self.circle_period_sec = float(rospy.get_param('~circle_period_sec', 30.0))
        self.circle_hover_sec = float(rospy.get_param('~circle_hover_sec', 5.0))
        self.circle_ramp_sec = float(rospy.get_param('~circle_ramp_sec', 8.0))
        self.circle_direction = float(rospy.get_param('~circle_direction', 1.0))
        eightloop_center_param = rospy.get_param('~eightloop_center', [0.0, 0.0, 2.0])
        eightloop_center_enu = np.array(eightloop_center_param, dtype=float)
        self.eightloop_center_ned = np.array(
            enu_to_ned_position(
                eightloop_center_enu[0],
                eightloop_center_enu[1],
                eightloop_center_enu[2],
            ),
            dtype=float,
        )
        self.eightloop_length_m = float(rospy.get_param('~eightloop_length_m', 3.5))
        self.eightloop_width_m = float(rospy.get_param('~eightloop_width_m', 2.0))
        self.eightloop_period_sec = float(rospy.get_param('~eightloop_period_sec', 30.0))
        self.eightloop_ramp_sec = float(rospy.get_param('~eightloop_ramp_sec', 8.0))
        self.eightloop_direction = float(rospy.get_param('~eightloop_direction', 1.0))
        # WARNING: operator MUST keep ~takeoff_alt equal to PX4 MIS_TAKEOFF_ALT.
        # Current FC value was observed at 0.5 m and must be raised to 2.5 m before flight.
        self.takeoff_alt = float(rospy.get_param('~takeoff_alt', 2.5))
        # Diagnostic logging period in seconds. Default 1.0 (1 Hz human-readable).
        # Drop to ~0.2 for SITL tuning to get 5 Hz time-series.
        self.diag_log_period_sec = rospy.get_param('~diag_log_period_sec', 1.0)

        rospy.loginfo(f"[NMPC] Platform: {platform_name}")
        rospy.loginfo(f"[NMPC] Control rate: {self.control_rate_hz} Hz")
        rospy.loginfo(
            f"[NMPC] Hover position (ENU param): {hover_pos_enu} -> NED: {self.hover_position_ned}"
        )
        rospy.loginfo(
            f"[NMPC] Hover yaw (ENU param): {hover_yaw_enu:+.3f} rad "
            f"({np.degrees(hover_yaw_enu):+.1f} deg) -> NED: {self.hover_yaw_ned:+.3f} rad "
            f"({np.degrees(self.hover_yaw_ned):+.1f} deg)"
        )
        rospy.loginfo(f"[NMPC] Diagnostic log period: {self.diag_log_period_sec:.2f} s")
        rospy.loginfo(f"[NMPC] Enable OFFBOARD on start: {self.enable_offboard_on_start}")
        rospy.logwarn(
            f"[NMPC] PX4 takeoff handoff expects ~takeoff_alt={self.takeoff_alt:.2f} m "
            "to match MIS_TAKEOFF_ALT on the flight controller"
        )
        if self.trajectory_type == 'circle':
            rospy.loginfo(
                f"[NMPC] Circle trajectory: center ENU={circle_center_enu}, "
                f"center NED={self.circle_center_ned}, radius={self.circle_radius_m:.2f} m, "
                f"period={self.circle_period_sec:.1f} s, hover={self.circle_hover_sec:.1f} s, "
                f"ramp={self.circle_ramp_sec:.1f} s"
            )
        rospy.loginfo(
            f"[NMPC] Eightloop trajectory: center ENU={eightloop_center_enu}, "
            f"center NED={self.eightloop_center_ned}, length={self.eightloop_length_m:.2f} m, "
            f"width={self.eightloop_width_m:.2f} m, period={self.eightloop_period_sec:.1f} s, "
            f"ramp={self.eightloop_ramp_sec:.1f} s"
        )

        # ===== Load Platform Config =====
        try:
            self.platform_cfg = load_platform_config(platform_name)
            rospy.loginfo(f"[NMPC] Loaded platform: {self.platform_cfg.name}")
            rospy.loginfo(f"[NMPC]   mass: {self.platform_cfg.mass_kg} kg")
            rospy.loginfo(f"[NMPC]   max_thrust: {self.platform_cfg.max_thrust_n} N")
        except Exception as e:
            rospy.logerr(f"[NMPC] Failed to load platform config: {e}")
            raise

        # ===== Load NMPC Solver =====
        try:
            rospy.loginfo("[NMPC] Loading acados solver...")
            self.solver = create_solver(
                platform_config=self.platform_cfg,
                horizon=DEFAULT_HORIZON,
                num_steps=DEFAULT_NUM_STEPS,
                generate_c_code=False,  # Use pre-generated code
            )
            self.nx = 9  # state dimension
            self.nu = 4  # input dimension
            rospy.loginfo("[NMPC] Solver loaded successfully")
        except Exception as e:
            rospy.logerr(f"[NMPC] Failed to load solver: {e}")
            raise

        # ===== State Storage =====
        self.pose = None  # PoseStamped
        self.pose_timestamp = None
        self.velocity = None  # TwistStamped
        self.vel_timestamp = None
        self.mavros_state = None  # State message
        self.state_timestamp = None

        # ===== Control State =====
        self.phase = FlightPhase.INIT
        self.last_control_input = np.array([
            self.platform_cfg.mass_kg * 9.81, 0.0, 0.0, 0.0
        ])
        self.consecutive_solver_failures = 0
        self.solver_warmup_done = False
        self._arm_requested = False  # BUG 2: track arm request state
        self._offboard_requested = False
        self._requested_command = None
        self._takeoff_phase = None
        self._takeoff_start_time = None
        self._takeoff_settle_since = None
        self._takeoff_timeout_logged = False
        self._takeoff_mode_requested = False
        self._takeoff_handoff_pose_enu = None
        self.trajectory_mode = TrajectoryMode.HOVER
        self._ramp_t_start = None
        self._ramp_duration = None
        self._ramp_start_pos = None
        self._ramp_start_vel = None
        self._ramp_start_yaw = None
        self._ramp_entry_pos = None
        self._ramp_entry_vel = None
        self._ramp_entry_yaw = None
        self._circle_start_time = None
        self._eightloop_start_time = None
        self._land_hold_start = None
        self._land_descend_requested = False
        self._waypoint_index = None
        self._waypoint_hold_ref = None
        self._waypoint_hold_start = None
        self.arm_service = None
        self.set_mode_service = None

        # Timing for state machine
        self.phase_enter_time = time.time()
        self.setpoint_stream_start_time = None
        self.run_start_time = None

        # Running statistics
        self.solver_times = deque(maxlen=100)
        self._run_cycle_count = 0  # BUG 5: track cycles for logging

        # ===== Publishers =====
        self.attitude_pub = rospy.Publisher(
            '/mavros/setpoint_raw/attitude',
            AttitudeTarget,
            queue_size=1
        )
        self.position_pub = rospy.Publisher(
            '/mavros/setpoint_position/local',
            PoseStamped,
            queue_size=1
        )
        self.reference_traj_pub = rospy.Publisher(
            '/nmpc/reference_trajectory',
            PoseArray,
            queue_size=1
        )

        # ===== Subscribers =====
        rospy.Subscriber('/mavros/state', State, self._state_cb, queue_size=1)
        rospy.Subscriber(
            '/mavros/local_position/pose',
            PoseStamped,
            self._pose_cb,
            queue_size=1
        )
        rospy.Subscriber(
            '/mavros/local_position/velocity_local',
            TwistStamped,
            self._vel_cb,
            queue_size=1
        )
        rospy.Subscriber('/nmpc/command', String, self._command_cb, queue_size=1)

        # ===== Service Clients =====
        rospy.wait_for_service('/mavros/cmd/arming')
        self.arm_service = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        rospy.wait_for_service('/mavros/set_mode')
        self.set_mode_service = rospy.ServiceProxy('/mavros/set_mode', SetMode)

        # ===== Control Loop Timer =====
        dt = 1.0 / self.control_rate_hz
        self.control_timer = rospy.Timer(
            rospy.Duration(dt),
            self._control_loop,
            oneshot=False
        )

        rospy.on_shutdown(self._on_shutdown)

        rospy.loginfo("[NMPC] Node initialized successfully!")

    def _state_cb(self, msg):
        """Store latest FCU state."""
        self.mavros_state = msg
        self.state_timestamp = time.time()

    def _pose_cb(self, msg):
        """Store latest pose (ENU from MAVROS)."""
        self.pose = msg
        self.pose_timestamp = time.time()

    def _vel_cb(self, msg):
        """Store latest velocity (ENU from MAVROS)."""
        self.velocity = msg
        self.vel_timestamp = time.time()

    def _command_cb(self, msg):
        """Store the latest external command for later processing in the control loop."""
        command = msg.data.strip()
        self._requested_command = command
        rospy.loginfo(f"[NMPC] Command received: {command}")

    def _build_state_vector(self):
        """Build 9D state vector from pose and velocity.

        Returns:
            state (np.ndarray): [x, y, z, vx, vy, vz, roll, pitch, yaw]
            None if state is not ready
        """
        if (self.pose is None or self.velocity is None or
            self.pose_timestamp is None or self.vel_timestamp is None):
            return None

        # Check timestamp sync
        ts_diff = abs(self.pose_timestamp - self.vel_timestamp)
        if ts_diff > POSE_VEL_SYNC_THRESHOLD_S:
            rospy.logwarn_throttle(
                5.0,
                f"[NMPC] Pose/velocity timestamps out of sync: {ts_diff:.3f} s"
            )
            return None

        # Extract position (ENU from MAVROS) and convert to NED for solver
        p = np.array(enu_to_ned_position(
            self.pose.pose.position.x,
            self.pose.pose.position.y,
            self.pose.pose.position.z,
        ))

        # Extract velocity (ENU from MAVROS) and convert to NED for solver
        v = np.array(enu_to_ned_velocity(
            self.velocity.twist.linear.x,
            self.velocity.twist.linear.y,
            self.velocity.twist.linear.z,
        ))

        # Extract quaternion (FLU body relative to ENU world, from MAVROS).
        # Convert to FRD body relative to NED world for the solver, then to
        # ZYX Euler angles [roll, pitch, yaw].
        q = self.pose.pose.orientation
        q_frd = flu_to_frd_quaternion(q.x, q.y, q.z, q.w)
        euler = quaternion_to_euler_zyx(*q_frd)

        # Build state vector
        x = np.hstack((p, v, euler))
        return x

    @staticmethod
    def _smoothstep(s):
        """Cubic smoothstep and derivative for s in [0, 1]."""
        s_clamped = np.clip(s, 0.0, 1.0)
        value = s_clamped * s_clamped * (3.0 - 2.0 * s_clamped)
        deriv = 6.0 * s_clamped * (1.0 - s_clamped)
        return value, deriv

    def _circle_reference_at(self, elapsed):
        """Return one 9D circle reference in solver/NED frame."""
        if elapsed < self.circle_hover_sec:
            pos_ned = self.circle_center_ned
            vel_ned = np.zeros(3)
            yaw_ned = self.hover_yaw_ned
        else:
            t_circle = elapsed - self.circle_hover_sec
            omega = self.circle_direction * 2.0 * np.pi / self.circle_period_sec
            theta = omega * t_circle

            if self.circle_ramp_sec > 0.0 and t_circle < self.circle_ramp_sec:
                ramp, ramp_deriv = self._smoothstep(t_circle / self.circle_ramp_sec)
                radius = self.circle_radius_m * ramp
                radius_dot = self.circle_radius_m * ramp_deriv / self.circle_ramp_sec
            else:
                radius = self.circle_radius_m
                radius_dot = 0.0

            center_enu = np.array([
                self.circle_center_ned[1],
                self.circle_center_ned[0],
                -self.circle_center_ned[2],
            ])
            radial_enu = np.array([np.cos(theta), np.sin(theta), 0.0])
            tangent_enu = np.array([-np.sin(theta), np.cos(theta), 0.0])
            pos_enu = center_enu + radius * radial_enu
            vel_enu = radius_dot * radial_enu + radius * omega * tangent_enu

            pos_ned = np.array(enu_to_ned_position(pos_enu[0], pos_enu[1], pos_enu[2]))
            vel_ned = np.array(enu_to_ned_velocity(vel_enu[0], vel_enu[1], vel_enu[2]))
            yaw_ned = wrap_to_pi(self.hover_yaw_ned - theta - self.circle_direction * np.pi / 2.0)

        return np.hstack((
            pos_ned,
            vel_ned,
            [0.0, 0.0, yaw_ned],
        ))

    def _eightloop_reference_at(self, elapsed):
        """Return one 9D figure-eight reference in solver/NED frame."""
        period = max(self.eightloop_period_sec, 1.0 / max(self.control_rate_hz, 1.0))
        omega = self.eightloop_direction * 2.0 * np.pi / period
        theta = omega * elapsed

        if self.eightloop_ramp_sec > 0.0 and elapsed < self.eightloop_ramp_sec:
            amp, amp_deriv = self._smoothstep(elapsed / self.eightloop_ramp_sec)
            amp_dot = amp_deriv / self.eightloop_ramp_sec
        else:
            amp = 1.0
            amp_dot = 0.0

        half_length = 0.5 * self.eightloop_length_m
        half_width = 0.5 * self.eightloop_width_m
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_2theta = np.sin(2.0 * theta)
        cos_2theta = np.cos(2.0 * theta)

        center_enu = np.array([
            self.eightloop_center_ned[1],
            self.eightloop_center_ned[0],
            -self.eightloop_center_ned[2],
        ])
        pos_enu = center_enu + np.array([
            half_length * amp * sin_theta,
            half_width * amp * sin_2theta,
            0.0,
        ])
        vel_enu = np.array([
            half_length * (amp_dot * sin_theta + amp * omega * cos_theta),
            half_width * (amp_dot * sin_2theta + amp * 2.0 * omega * cos_2theta),
            0.0,
        ])
        tangent_enu = np.array([
            half_length * omega * cos_theta,
            half_width * 2.0 * omega * cos_2theta,
        ])
        if np.linalg.norm(tangent_enu) > 1e-6:
            yaw_enu = np.arctan2(tangent_enu[1], tangent_enu[0])
            yaw_ned = enu_yaw_to_ned_yaw(yaw_enu)
        else:
            yaw_ned = self.hover_yaw_ned

        pos_ned = np.array(enu_to_ned_position(pos_enu[0], pos_enu[1], pos_enu[2]))
        vel_ned = np.array(enu_to_ned_velocity(vel_enu[0], vel_enu[1], vel_enu[2]))
        return np.hstack((
            pos_ned,
            vel_ned,
            [0.0, 0.0, yaw_ned],
        ))

    def _begin_ramp(self, target_pos_ned, target_vel_ned, target_yaw_ned):
        """Snapshot current state and prepare a pose-aware ramp to a NED target."""
        x_now = self._build_state_vector()
        if x_now is None:
            rospy.logwarn("[NMPC] Ramp start unavailable: state unavailable")
            return False

        self._ramp_start_pos = np.array(x_now[0:3], dtype=float)
        self._ramp_start_vel = np.array(x_now[3:6], dtype=float)
        self._ramp_start_yaw = float(x_now[8])
        self._ramp_entry_pos = np.array(target_pos_ned, dtype=float)
        self._ramp_entry_vel = np.array(target_vel_ned, dtype=float)
        self._ramp_entry_yaw = float(target_yaw_ned)

        dist = float(np.linalg.norm(self._ramp_entry_pos - self._ramp_start_pos))
        self._ramp_duration = float(np.clip(dist / CRUISE_SPEED, MIN_RAMP_S, MAX_RAMP_S))
        self._ramp_t_start = time.time()
        self._circle_start_time = None
        self._eightloop_start_time = None
        rospy.loginfo(f"[NMPC] Ramp: dist={dist:.2f} m duration={self._ramp_duration:.1f} s")
        return True

    def _begin_ramp_to_circle(self):
        """Prepare a pose-aware ramp to the circle generator's entry state."""
        entry = self._circle_reference_at(0.0)
        return self._begin_ramp(entry[0:3], np.zeros(3), entry[8])

    def _begin_ramp_to_eightloop(self):
        """Prepare a pose-aware ramp to the figure-eight generator's entry state."""
        entry = self._eightloop_reference_at(0.0)
        return self._begin_ramp(entry[0:3], entry[3:6], entry[8])

    def _waypoint_reference(self, index):
        """Return one waypoint reference from the editable ENU/RPY matrix."""
        waypoint = WAYPOINTS_ENU_RPY[index]
        pos_ned = np.array(enu_to_ned_position(waypoint[0], waypoint[1], waypoint[2]), dtype=float)
        yaw_ned = enu_yaw_to_ned_yaw(waypoint[5])
        return np.hstack((pos_ned, [0.0, 0.0, 0.0], [waypoint[3], waypoint[4], yaw_ned]))

    def _begin_waypoint_ramp(self, index):
        """Prepare a pose-aware ramp to one configured waypoint."""
        ref = self._waypoint_reference(index)
        if not self._begin_ramp(ref[0:3], ref[3:6], ref[8]):
            return False
        self._waypoint_index = index
        self._waypoint_hold_ref = ref
        self._waypoint_hold_start = None
        return True

    def _ramp_target_reached(self):
        """Return True when actual state is close enough to ramp target."""
        x_now = self._build_state_vector()
        if x_now is None:
            rospy.logwarn_throttle(2.0, "[NMPC] Ramp gate: waiting for valid state")
            return False

        pos_err = float(np.linalg.norm(self._ramp_entry_pos - x_now[0:3]))
        vel_err = float(np.linalg.norm(self._ramp_entry_vel - x_now[3:6]))
        ready = (
            pos_err <= RAMP_TARGET_POS_TOL_M
            and vel_err <= RAMP_TARGET_VEL_TOL_MPS
        )

        if not ready:
            rospy.loginfo_throttle(
                1.0,
                f"[NMPC] Ramp gate: holding endpoint, pos_err={pos_err:.2f} m "
                f"(tol {RAMP_TARGET_POS_TOL_M:.2f}), vel_err={vel_err:.2f} m/s "
                f"(tol {RAMP_TARGET_VEL_TOL_MPS:.2f})"
            )
        return ready

    def _ramp_reference_at(self, elapsed):
        """Return one 9D ramp-to-circle reference in solver/NED frame."""
        duration = max(self._ramp_duration, 1.0 / max(self.control_rate_hz, 1.0))
        s = np.clip(elapsed / duration, 0.0, 1.0)
        ramp, ramp_deriv = self._smoothstep(s)

        pos_delta = self._ramp_entry_pos - self._ramp_start_pos
        pos = self._ramp_start_pos + ramp * pos_delta
        vel = (
            self._ramp_start_vel
            + ramp * (self._ramp_entry_vel - self._ramp_start_vel)
            + ramp_deriv / duration * pos_delta
        )
        yaw_delta = wrap_to_pi(self._ramp_entry_yaw - self._ramp_start_yaw)
        yaw = wrap_to_pi(self._ramp_start_yaw + ramp * yaw_delta)
        return np.hstack((pos, vel, [0.0, 0.0, yaw]))

    def _build_reference_trajectory(self, start_elapsed=None):
        """Build the NMPC reference trajectory for the full prediction horizon."""
        dt = DEFAULT_HORIZON / DEFAULT_NUM_STEPS

        if self.trajectory_mode in (
            TrajectoryMode.RAMP_TO_CIRCLE,
            TrajectoryMode.RAMP_TO_EIGHTLOOP,
            TrajectoryMode.RAMP_TO_HOVER,
            TrajectoryMode.LAND_APPROACH,
            TrajectoryMode.WAYPOINT_RAMP,
        ):
            ramp_elapsed = time.time() - self._ramp_t_start
            return np.vstack([
                self._ramp_reference_at(ramp_elapsed + i * dt)
                for i in range(DEFAULT_NUM_STEPS)
            ])

        if self.trajectory_mode == TrajectoryMode.CIRCLE:
            circle_elapsed = time.time() - self._circle_start_time
            return np.vstack([
                self._circle_reference_at(circle_elapsed + i * dt)
                for i in range(DEFAULT_NUM_STEPS)
            ])

        if self.trajectory_mode == TrajectoryMode.EIGHTLOOP:
            eightloop_elapsed = time.time() - self._eightloop_start_time
            return np.vstack([
                self._eightloop_reference_at(eightloop_elapsed + i * dt)
                for i in range(DEFAULT_NUM_STEPS)
            ])

        if self.trajectory_mode == TrajectoryMode.LAND_HOLD:
            land_pos_ned = np.array(enu_to_ned_position(*LAND_TARGET_ENU), dtype=float)
            land_yaw_ned = enu_yaw_to_ned_yaw(LAND_TARGET_YAW_ENU)
            ref_per_stage = np.hstack((
                land_pos_ned,
                [0.0, 0.0, 0.0],
                [0.0, 0.0, land_yaw_ned],
            ))
            return np.tile(ref_per_stage, (DEFAULT_NUM_STEPS, 1))

        if self.trajectory_mode == TrajectoryMode.WAYPOINT_HOLD:
            return np.tile(self._waypoint_hold_ref, (DEFAULT_NUM_STEPS, 1))

        ref_per_stage = np.hstack((
            self.hover_position_ned,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, self.hover_yaw_ned],
        ))
        return np.tile(ref_per_stage, (DEFAULT_NUM_STEPS, 1))

    def _solve(self, x_now, x_ref):
        """Solve NMPC problem.

        Args:
            x_now (np.ndarray): Current state (9D)
            x_ref (np.ndarray): Reference trajectory (N, 9)

        Returns:
            u_optimal (np.ndarray): Optimal control (4D) or None if solve failed
            solve_time (float): Time to solve in seconds
            status (int): acados solver status code
        """
        t0 = time.time()

        try:
            u, _, status = self.solver.solve_mpc_control(
                x_now,
                x_ref,
                self.last_control_input,
                nx=self.nx,
                nu=self.nu,
                verbose=False,
            )
            t1 = time.time()
            solve_time = t1 - t0
            self.solver_times.append(solve_time)

            # Return first control input
            u_optimal = u[0, :] if status == 0 else None
            return u_optimal, solve_time, status
        except Exception as e:
            rospy.logerr(f"[NMPC] Solver exception: {e}")
            return None, 0.0, -1

    def _handle_solver_status(self, status):
        """Log and interpret solver status code.

        Returns:
            use_solution (bool): Whether to use the solution
        """
        if status == 0:
            # Success
            return True
        elif status == 1:
            rospy.logwarn_throttle(5.0, "[NMPC] Solver: max iterations reached")
            return True  # Solution may still be reasonable
        elif status == 2:
            rospy.logwarn_throttle(5.0, "[NMPC] Solver: NaN in solution")
            return False
        elif status == 3:
            rospy.logwarn_throttle(5.0, "[NMPC] Solver: QP failure")
            return False
        elif status == 4:
            rospy.logwarn_throttle(5.0, "[NMPC] Solver: QP max iterations")
            return True
        else:
            rospy.logerr(f"[NMPC] Solver: unknown status {status}")
            return False

    def _publish_attitude_target(self, u):
        """Publish attitude setpoint with body rates and thrust.

        Input u is in solver/FRD frame. Body-rate conversion to MAVROS FLU
        happens inside this method immediately before publishing.

        Args:
            u (np.ndarray): [thrust_N, p, q, r] in solver/FRD frame
        """
        msg = AttitudeTarget()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"

        # Type mask: ignore quaternion, use body rates and thrust
        msg.type_mask = ATTITUDE_TARGET_USE_RATES_AND_THRUST

        # Quaternion (ignored by type_mask, set to identity)
        msg.orientation.w = 1.0
        msg.orientation.x = 0.0
        msg.orientation.y = 0.0
        msg.orientation.z = 0.0

        # Body rates: solver outputs FRD (p, q, r); mavros expects FLU.
        # The transform (p, q, r) -> (p, -q, -r) is self-inverse.
        # mavros internally converts FLU -> FRD before sending
        # SET_ATTITUDE_TARGET to PX4 (see setpoint_raw.cpp::attitude_cb).
        p_flu, q_flu, r_flu = enu_body_to_ned_body_rates(u[1], u[2], u[3])
        msg.body_rate.x = p_flu
        msg.body_rate.y = q_flu
        msg.body_rate.z = r_flu

        # Thrust: normalize [0, max_thrust_n] N to [0, 1]
        thrust_normalised = np.clip(
            u[0] / self.platform_cfg.max_thrust_n,
            0.0,
            1.0
        )
        msg.thrust = thrust_normalised

        self.attitude_pub.publish(msg)

    def _publish_safe_hover(self):
        """Publish level hover command (safe fallback)."""
        msg = AttitudeTarget()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        msg.type_mask = ATTITUDE_TARGET_USE_RATES_AND_THRUST

        # Level attitude (identity quaternion)
        msg.orientation.w = 1.0
        msg.orientation.x = 0.0
        msg.orientation.y = 0.0
        msg.orientation.z = 0.0

        # Zero body rates
        msg.body_rate.x = 0.0
        msg.body_rate.y = 0.0
        msg.body_rate.z = 0.0

        # Hover thrust (normalised)
        msg.thrust = self.platform_cfg.hover_thrust_normalised

        self.attitude_pub.publish(msg)

    def _publish_current_position_keepalive(self):
        """Publish a local-position proof-of-life setpoint at the current pose."""
        if self.pose is None:
            rospy.logwarn_throttle(2.0, "[NMPC] Position keepalive skipped: no local pose")
            return

        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.pose.header.frame_id or "map"
        msg.pose.position.x = self.pose.pose.position.x
        msg.pose.position.y = self.pose.pose.position.y
        msg.pose.position.z = self.pose.pose.position.z
        msg.pose.orientation = self.pose.pose.orientation
        self.position_pub.publish(msg)

    def _publish_reference_trajectory(self, x_ref):
        """Publish the active NMPC horizon reference as ENU poses for RViz/rosbag."""
        msg = PoseArray()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"

        for ref in x_ref:
            x_enu, y_enu, z_enu = ned_to_enu_position(ref[0], ref[1], ref[2])
            yaw_enu = enu_yaw_to_ned_yaw(ref[8])
            half_yaw = 0.5 * yaw_enu

            pose = Pose()
            pose.position.x = float(x_enu)
            pose.position.y = float(y_enu)
            pose.position.z = float(z_enu)
            pose.orientation.z = float(np.sin(half_yaw))
            pose.orientation.w = float(np.cos(half_yaw))
            msg.poses.append(pose)

        self.reference_traj_pub.publish(msg)

    def _solve_and_publish_reference(self, x_now, x_ref, label):
        """Solve NMPC for a supplied reference and publish the resulting command."""
        self._publish_reference_trajectory(x_ref)
        u_optimal, solve_time, status = self._solve(x_now, x_ref)

        self._run_cycle_count += 1
        if self._run_cycle_count % 250 == 0 and len(self.solver_times) > 0:
            mean_time = np.mean(self.solver_times)
            max_time = np.max(self.solver_times)
            rospy.loginfo(
                f"[NMPC] Solver: mean {mean_time*1000:.1f} ms, "
                f"max {max_time*1000:.1f} ms (last 100 cycles)"
            )

        use_solution = self._handle_solver_status(status)
        publish_solution = use_solution and u_optimal is not None
        u_to_publish = u_optimal if publish_solution else self.last_control_input

        if publish_solution:
            self.last_control_input = u_optimal
            self.consecutive_solver_failures = 0
        else:
            self.consecutive_solver_failures += 1
            rospy.logwarn_throttle(
                5.0,
                f"[NMPC] Using last control (failure #{self.consecutive_solver_failures})"
            )

        p_flu, q_flu, r_flu = enu_body_to_ned_body_rates(
            u_to_publish[1], u_to_publish[2], u_to_publish[3]
        )
        status_str = ACADOS_STATUS_LABELS.get(status, f"UNKNOWN({status})")
        rospy.loginfo_throttle(
            self.diag_log_period_sec,
            f"[NMPC] {label} diag: "
            f"status={status_str} use_solution={publish_solution} "
            f"failures={self.consecutive_solver_failures} "
            f"solve_ms={solve_time * 1000.0:.1f} "
            f"pos_ned_m=[{x_now[0]:+.2f},{x_now[1]:+.2f},{x_now[2]:+.2f}] "
            f"ref_ned_m=[{x_ref[0, 0]:+.2f},{x_ref[0, 1]:+.2f},{x_ref[0, 2]:+.2f}] "
            f"vel_ned_mps=[{x_now[3]:+.2f},{x_now[4]:+.2f},{x_now[5]:+.2f}] "
            f"rpy_ned_deg=[{np.degrees(x_now[6]):+.1f},"
            f"{np.degrees(x_now[7]):+.1f},{np.degrees(x_now[8]):+.1f}] "
            f"u_frd=[T={u_to_publish[0]:+.2f}N,"
            f"p={np.degrees(u_to_publish[1]):+.1f},"
            f"q={np.degrees(u_to_publish[2]):+.1f},"
            f"r={np.degrees(u_to_publish[3]):+.1f}]degps "
            f"rates_flu_degps=[{np.degrees(p_flu):+.1f},"
            f"{np.degrees(q_flu):+.1f},{np.degrees(r_flu):+.1f}]"
        )

        self._publish_attitude_target(u_to_publish)

        if (
            not publish_solution
            and self.consecutive_solver_failures >= MAX_CONSECUTIVE_SOLVER_FAILURES
        ):
            rospy.logerr("[NMPC] Too many solver failures, switching to safe hover")
            self._publish_safe_hover()

        return publish_solution, solve_time, status

    def _phase_init(self):
        """INIT state: wait for MAVROS and gather initial pose/velocity."""
        if self.mavros_state is None:
            rospy.loginfo_throttle(2.0, "[NMPC] INIT: waiting for /mavros/state...")
            return FlightPhase.INIT

        if not self.mavros_state.connected:
            rospy.loginfo_throttle(2.0, "[NMPC] INIT: FCU not connected")
            return FlightPhase.INIT

        rospy.loginfo("[NMPC] INIT: FCU connected, moving to WAIT_FOR_FCU")
        return FlightPhase.WAIT_FOR_FCU

    def _phase_wait_for_fcu(self):
        """WAIT_FOR_FCU: ensure pose/velocity are publishing and warm up solver."""
        if (self.pose is None or self.velocity is None or
            self._build_state_vector() is None):
            rospy.loginfo_throttle(2.0, "[NMPC] WAIT_FOR_FCU: waiting for pose/velocity...")
            return FlightPhase.WAIT_FOR_FCU

        # Warmup solver on first entry
        if not self.solver_warmup_done:
            rospy.loginfo("[NMPC] WAIT_FOR_FCU: warming up solver...")
            x_hover = self._build_state_vector()
            # Build 9D reference [p, v, euler]. u_ref is built internally
            # by solve_mpc_control from quad.m * quad.g (see generate_nmpc.py).
            x_ref = self._build_reference_trajectory(start_elapsed=0.0)

            for i in range(SOLVER_WARMUP_ITERATIONS):
                _, _, _ = self._solve(x_hover, x_ref)
                if (i + 1) % 5 == 0:
                    rospy.loginfo(f"[NMPC] Warmup iteration {i + 1}/{SOLVER_WARMUP_ITERATIONS}")

            self.solver_warmup_done = True
            rospy.loginfo("[NMPC] Solver warm-up complete")

        rospy.loginfo("[NMPC] WAIT_FOR_FCU: moving to STREAM_SETPOINTS")
        self.setpoint_stream_start_time = time.time()
        return FlightPhase.STREAM_SETPOINTS

    def _phase_stream_setpoints(self):
        """STREAM_SETPOINTS: publish position keepalive before operator command."""
        self._publish_current_position_keepalive()

        elapsed = time.time() - self.setpoint_stream_start_time
        if elapsed < SETPOINT_STREAM_DURATION_S:
            rospy.loginfo_throttle(
                2.0,
                f"[NMPC] STREAM: {elapsed:.1f}/{SETPOINT_STREAM_DURATION_S} s"
            )
            return FlightPhase.STREAM_SETPOINTS

        rospy.loginfo("[NMPC] STREAM: proof-of-life complete, waiting for takeoff command")
        return FlightPhase.IDLE

    def _is_airborne(self):
        """Return True when local pose suggests vehicle is already above the ground."""
        if self.pose is None:
            return False
        return self.pose.pose.position.z > 0.30

    def _phase_idle(self):
        """IDLE: stream position keepalive and wait for keyboard takeoff command."""
        self._publish_current_position_keepalive()

        command = self._requested_command
        if command != 'takeoff':
            if command == 'circle':
                self._requested_command = None
                rospy.logwarn("[NMPC] Circle command ignored: not in HOVER")
                return FlightPhase.IDLE
            if command == 'eightloop':
                self._requested_command = None
                rospy.logwarn("[NMPC] Eightloop command ignored: not in HOVER")
                return FlightPhase.IDLE
            rospy.loginfo_throttle(2.0, "[NMPC] IDLE: waiting for takeoff command")
            return FlightPhase.IDLE

        self._requested_command = None
        if self.mavros_state is None or not self.mavros_state.connected:
            rospy.logwarn("[NMPC] Takeoff command rejected: FCU is not connected")
            return FlightPhase.IDLE

        if self.mavros_state.armed or self._is_airborne():
            rospy.logwarn("[NMPC] Takeoff command rejected: vehicle already armed/airborne")
            return FlightPhase.IDLE

        rospy.loginfo("[NMPC] Takeoff command accepted: arming then requesting AUTO.TAKEOFF")
        self._takeoff_phase = "ARMING"
        self._arm_requested = False
        self._offboard_requested = False
        self._takeoff_mode_requested = False
        self._takeoff_start_time = None
        self._takeoff_settle_since = None
        self._takeoff_timeout_logged = False
        self.phase_enter_time = time.time()
        return FlightPhase.TAKEOFF_PX4

    def _phase_takeoff_px4(self):
        """TAKEOFF_PX4: let PX4 take off, then hand off to OFFBOARD after settle."""
        self._publish_current_position_keepalive()
        now = time.time()

        if self._requested_command == 'circle':
            self._requested_command = None
            rospy.logwarn("[NMPC] Circle command ignored: not in HOVER")

        if self._requested_command == 'eightloop':
            self._requested_command = None
            rospy.logwarn("[NMPC] Eightloop command ignored: not in HOVER")

        if self.mavros_state is None or not self.mavros_state.connected:
            rospy.logwarn_throttle(2.0, "[NMPC] TAKEOFF: waiting for FCU connection")
            return FlightPhase.TAKEOFF_PX4

        if self._takeoff_phase == "ARMING":
            if not self.mavros_state.armed:
                if not self._arm_requested:
                    rospy.loginfo("[NMPC] TAKEOFF: arming")
                    try:
                        self.arm_service(value=True)
                        self._arm_requested = True
                        self.phase_enter_time = now
                    except Exception as e:
                        rospy.logerr(f"[NMPC] TAKEOFF: arming failed: {e}")
                    return FlightPhase.TAKEOFF_PX4

                if now - self.phase_enter_time > ARM_WAIT_TIMEOUT_S:
                    rospy.logerr("[NMPC] TAKEOFF: arming timed out, retrying")
                    self._arm_requested = False
                    self.phase_enter_time = now
                return FlightPhase.TAKEOFF_PX4

            self._takeoff_phase = "AUTO_TAKEOFF"

        if self._takeoff_phase == "AUTO_TAKEOFF":
            if not self._takeoff_mode_requested:
                rospy.loginfo("[NMPC] TAKEOFF: requesting AUTO.TAKEOFF")
                try:
                    self.set_mode_service(custom_mode="AUTO.TAKEOFF")
                    self._takeoff_mode_requested = True
                    self._takeoff_start_time = now
                    self.phase_enter_time = now
                except Exception as e:
                    rospy.logerr(f"[NMPC] TAKEOFF: AUTO.TAKEOFF request failed: {e}")
                return FlightPhase.TAKEOFF_PX4

            if self.pose is None or self.velocity is None:
                rospy.logwarn_throttle(2.0, "[NMPC] TAKEOFF: waiting for pose/velocity")
                return FlightPhase.TAKEOFF_PX4

            altitude = self.pose.pose.position.z
            vz = self.velocity.twist.linear.z
            settled = (
                altitude >= 0.9 * self.takeoff_alt
                and abs(vz) < TAKEOFF_VERTICAL_VELOCITY_TOLERANCE_MPS
            )
            if settled:
                if self._takeoff_settle_since is None:
                    self._takeoff_settle_since = now
                elif now - self._takeoff_settle_since >= TAKEOFF_SETTLE_HOLD_S:
                    rospy.loginfo("[NMPC] TAKEOFF: settle gate met, requesting OFFBOARD")
                    self._takeoff_phase = "OFFBOARD_REQUEST"
                    self._offboard_requested = False
                    self.phase_enter_time = now
            else:
                self._takeoff_settle_since = None

            rospy.loginfo_throttle(
                1.0,
                f"[NMPC] TAKEOFF: alt={altitude:.2f}/{self.takeoff_alt:.2f} m "
                f"vz={vz:+.2f} m/s settled={settled}"
            )

            if (
                self._takeoff_start_time is not None
                and now - self._takeoff_start_time > TAKEOFF_SETTLE_TIMEOUT_S
                and not self._takeoff_timeout_logged
            ):
                rospy.logerr(
                    "[NMPC] TAKEOFF: settle timeout; staying in AUTO.TAKEOFF and not handing off"
                )
                self._takeoff_timeout_logged = True

            return FlightPhase.TAKEOFF_PX4

        if self._takeoff_phase == "OFFBOARD_REQUEST":
            if self.mavros_state.mode == "OFFBOARD":
                if self.pose is None:
                    rospy.logwarn_throttle(2.0, "[NMPC] TAKEOFF: OFFBOARD active but pose unavailable")
                    return FlightPhase.TAKEOFF_PX4

                self._takeoff_handoff_pose_enu = np.array([
                    self.pose.pose.position.x,
                    self.pose.pose.position.y,
                    self.pose.pose.position.z,
                ], dtype=float)
                target_pos = np.array(
                    enu_to_ned_position(0.0, 0.0, self.takeoff_alt),
                    dtype=float,
                )
                target_vel = np.zeros(3)
                target_yaw = self.hover_yaw_ned
                self.hover_position_ned = target_pos
                if not self._begin_ramp(target_pos, target_vel, target_yaw):
                    return FlightPhase.TAKEOFF_PX4

                self.trajectory_mode = TrajectoryMode.RAMP_TO_HOVER
                self.trajectory_type = 'hover'
                self.run_start_time = None
                self.phase_enter_time = now
                rospy.loginfo(
                    f"[NMPC] TAKEOFF: OFFBOARD active; ramping to hover target NED={self.hover_position_ned}"
                )
                return FlightPhase.RUN

            if not self._offboard_requested:
                rospy.loginfo("[NMPC] TAKEOFF: setting OFFBOARD mode")
                try:
                    self.set_mode_service(custom_mode="OFFBOARD")
                    self._offboard_requested = True
                    self.phase_enter_time = now
                except Exception as e:
                    rospy.logerr(f"[NMPC] TAKEOFF: OFFBOARD request failed: {e}")
                return FlightPhase.TAKEOFF_PX4

            if now - self.phase_enter_time > OFFBOARD_WAIT_TIMEOUT_S:
                rospy.logerr_throttle(
                    2.0,
                    f"[NMPC] TAKEOFF: OFFBOARD not accepted after {now - self.phase_enter_time:.1f}s, retrying"
                )
                self._offboard_requested = False

        return FlightPhase.TAKEOFF_PX4

    def _phase_request_offboard(self):
        """REQUEST_OFFBOARD: arm and switch to OFFBOARD mode."""
        # BUG 2: Publish hover setpoint continuously
        hover_u = np.array([
            self.platform_cfg.mass_kg * 9.81,
            0.0, 0.0, 0.0
        ])
        self._publish_attitude_target(hover_u)

        # BUG 2: Arm only once
        if not self._arm_requested and not self.mavros_state.armed:
            rospy.loginfo("[NMPC] REQUEST_OFFBOARD: arming...")
            try:
                self.arm_service(value=True)
                self._arm_requested = True
                self.phase_enter_time = time.time()
            except Exception as e:
                rospy.logerr(f"[NMPC] Arming failed: {e}")
            return FlightPhase.REQUEST_OFFBOARD

        # Check arming timeout
        elapsed = time.time() - self.phase_enter_time
        if self._arm_requested and not self.mavros_state.armed and elapsed > ARM_WAIT_TIMEOUT_S:
            rospy.logerr("[NMPC] Arming timed out, staying in REQUEST_OFFBOARD")
            self._arm_requested = False  # Reset for retry
            return FlightPhase.REQUEST_OFFBOARD

        # Once armed, request OFFBOARD mode (only once)
        if self.mavros_state.armed:
            if not self._offboard_requested:
                rospy.loginfo("[NMPC] REQUEST_OFFBOARD: armed, setting OFFBOARD mode")
                try:
                    self.set_mode_service(custom_mode="OFFBOARD")
                    self._offboard_requested = True
                    self.phase_enter_time = time.time()
                except Exception as e:
                    rospy.logerr(f"[NMPC] Set mode failed: {e}")
                return FlightPhase.REQUEST_OFFBOARD

            # Check if OFFBOARD has been achieved
            if self.mavros_state.mode == "OFFBOARD":
                rospy.loginfo("[NMPC] REQUEST_OFFBOARD: OFFBOARD active, moving to RUN")
                self.phase_enter_time = time.time()
                return FlightPhase.RUN

            # Check timeout on OFFBOARD acceptance
            elapsed = time.time() - self.phase_enter_time
            if elapsed > OFFBOARD_WAIT_TIMEOUT_S:
                rospy.logerr_throttle(
                    2.0,
                    f"[NMPC] OFFBOARD not accepted after {elapsed:.1f}s, retrying"
                )
                self._offboard_requested = False  # Allow retry

        return FlightPhase.REQUEST_OFFBOARD

    def _phase_run(self):
        """RUN: execute NMPC control loop."""
        if self.trajectory_mode == TrajectoryMode.LAND_DESCEND:
            if not self._land_descend_requested:
                try:
                    self.set_mode_service(custom_mode="AUTO.LAND")
                    self._land_descend_requested = True
                    self.trajectory_mode = TrajectoryMode.DONE
                except Exception as e:
                    rospy.logerr(f"[NMPC] AUTO.LAND request failed: {e}")
            return FlightPhase.RUN

        if self.trajectory_mode == TrajectoryMode.DONE:
            landed = (
                self.mavros_state is not None
                and not self.mavros_state.armed
                and (self.pose is None or self.pose.pose.position.z <= 0.30)
            )
            if landed:
                rospy.loginfo("[NMPC] Landing complete; ready for takeoff command")
                self._requested_command = None
                self._arm_requested = False
                self._offboard_requested = False
                self._takeoff_phase = None
                self._takeoff_mode_requested = False
                self._takeoff_start_time = None
                self._takeoff_settle_since = None
                self._takeoff_timeout_logged = False
                self._land_hold_start = None
                self._land_descend_requested = False
                self.trajectory_mode = TrajectoryMode.HOVER
                self.phase_enter_time = time.time()
                return FlightPhase.IDLE

            rospy.loginfo_throttle(2.0, "[NMPC] Landing handed to PX4 AUTO.LAND")
            return FlightPhase.RUN

        if self._requested_command == 'takeoff':
            self._requested_command = None
            rospy.logwarn("[NMPC] Takeoff command rejected: vehicle already armed/airborne")

        # Check if we've been waiting for OFFBOARD and it happened
        if self.mavros_state.mode != "OFFBOARD":
            if self.enable_offboard_on_start:
                elapsed = time.time() - self.phase_enter_time
                if elapsed < OFFBOARD_WAIT_TIMEOUT_S:
                    rospy.loginfo_throttle(
                        2.0,
                        f"[NMPC] RUN: waiting for OFFBOARD ({elapsed:.1f}s)"
                    )
                    return FlightPhase.RUN
                else:
                    rospy.logerr("[NMPC] OFFBOARD mode not achieved, staying in RUN")
            else:
                # BUG 3: Publish hover setpoints while waiting for RC OFFBOARD
                hover_u = np.array([
                    self.platform_cfg.mass_kg * 9.81,
                    0.0, 0.0, 0.0
                ])
                self._publish_attitude_target(hover_u)
                rospy.loginfo_throttle(2.0, "[NMPC] RUN: waiting for RC OFFBOARD switch")
                return FlightPhase.RUN

        land_modes = (
            TrajectoryMode.LAND_APPROACH,
            TrajectoryMode.LAND_HOLD,
            TrajectoryMode.LAND_DESCEND,
            TrajectoryMode.DONE,
        )
        reference_switch_modes = (
            TrajectoryMode.HOVER,
            TrajectoryMode.RAMP_TO_CIRCLE,
            TrajectoryMode.CIRCLE,
            TrajectoryMode.RAMP_TO_EIGHTLOOP,
            TrajectoryMode.EIGHTLOOP,
            TrajectoryMode.WAYPOINT_RAMP,
            TrajectoryMode.WAYPOINT_HOLD,
        )

        if self._requested_command == 'land':
            self._requested_command = None
            if self.trajectory_mode in land_modes:
                rospy.logwarn("[NMPC] Land command ignored: already landing")
                return FlightPhase.RUN

            target_pos = np.array(enu_to_ned_position(*LAND_TARGET_ENU), dtype=float)
            target_vel = np.zeros(3)
            target_yaw = enu_yaw_to_ned_yaw(LAND_TARGET_YAW_ENU)
            if not self._begin_ramp(target_pos, target_vel, target_yaw):
                return FlightPhase.RUN

            self.trajectory_mode = TrajectoryMode.LAND_APPROACH
            self._land_hold_start = None
            self._land_descend_requested = False
            rospy.loginfo("[NMPC] Land commanded: ramping to (0,0,1)")

        if self._requested_command == 'go to points':
            self._requested_command = None
            if self.trajectory_mode in land_modes:
                rospy.logwarn("[NMPC] Waypoint command ignored: already landing")
                return FlightPhase.RUN
            if self.trajectory_mode not in reference_switch_modes:
                rospy.logwarn("[NMPC] Waypoint command ignored: current mode cannot switch")
                return FlightPhase.RUN
            if not self._begin_waypoint_ramp(0):
                return FlightPhase.RUN
            self.trajectory_mode = TrajectoryMode.WAYPOINT_RAMP
            self._waypoint_hold_start = None
            rospy.loginfo("[NMPC] Waypoint command accepted")

        if self._requested_command == 'circle':
            self._requested_command = None
            if self.trajectory_mode in land_modes:
                rospy.logwarn("[NMPC] Circle command ignored: already landing")
                return FlightPhase.RUN
            if self.trajectory_mode not in reference_switch_modes:
                rospy.logwarn("[NMPC] Circle command ignored: current mode cannot switch")
                return FlightPhase.RUN
            if not self._begin_ramp_to_circle():
                return FlightPhase.RUN
            self.trajectory_mode = TrajectoryMode.RAMP_TO_CIRCLE
            self._waypoint_hold_start = None
            rospy.loginfo("[NMPC] Circle command accepted")

        if self._requested_command == 'eightloop':
            self._requested_command = None
            if self.trajectory_mode in land_modes:
                rospy.logwarn("[NMPC] Eightloop command ignored: already landing")
                return FlightPhase.RUN
            if self.trajectory_mode not in reference_switch_modes:
                rospy.logwarn("[NMPC] Eightloop command ignored: current mode cannot switch")
                return FlightPhase.RUN
            if not self._begin_ramp_to_eightloop():
                return FlightPhase.RUN
            self.trajectory_mode = TrajectoryMode.RAMP_TO_EIGHTLOOP
            self._waypoint_hold_start = None
            rospy.loginfo("[NMPC] Eightloop command accepted")

        now = time.time()
        if (
            self.trajectory_mode in (
                TrajectoryMode.RAMP_TO_CIRCLE,
                TrajectoryMode.RAMP_TO_EIGHTLOOP,
                TrajectoryMode.RAMP_TO_HOVER,
                TrajectoryMode.LAND_APPROACH,
                TrajectoryMode.WAYPOINT_RAMP,
            )
            and now - self._ramp_t_start >= self._ramp_duration
        ):
            if not self._ramp_target_reached():
                pass
            elif self.trajectory_mode == TrajectoryMode.LAND_APPROACH:
                self.trajectory_mode = TrajectoryMode.LAND_HOLD
                self._land_hold_start = now
                rospy.loginfo("[NMPC] Land approach complete; holding 2 s")
            elif self.trajectory_mode == TrajectoryMode.WAYPOINT_RAMP:
                self.trajectory_mode = TrajectoryMode.WAYPOINT_HOLD
                self._waypoint_hold_start = now
                rospy.loginfo(
                    f"[NMPC] Waypoint {self._waypoint_index + 1} reached; holding "
                    f"{WAYPOINT_HOLD_SEC:.1f} s"
                )
            elif self.trajectory_mode == TrajectoryMode.RAMP_TO_HOVER:
                self.trajectory_mode = TrajectoryMode.HOVER
                rospy.loginfo("[NMPC] Hover ramp complete; entering HOVER")
            elif self.trajectory_mode == TrajectoryMode.RAMP_TO_EIGHTLOOP:
                self._eightloop_start_time = now
                self.trajectory_mode = TrajectoryMode.EIGHTLOOP
                rospy.loginfo("[NMPC] Eightloop ramp complete; entering EIGHTLOOP")
            else:
                self._circle_start_time = now
                self.trajectory_mode = TrajectoryMode.CIRCLE
                rospy.loginfo("[NMPC] Circle ramp complete; entering CIRCLE")

        if (
            self.trajectory_mode == TrajectoryMode.WAYPOINT_HOLD
            and self._waypoint_hold_start is not None
            and now - self._waypoint_hold_start >= WAYPOINT_HOLD_SEC
        ):
            next_index = self._waypoint_index + 1
            if next_index < len(WAYPOINTS_ENU_RPY):
                if self._begin_waypoint_ramp(next_index):
                    self.trajectory_mode = TrajectoryMode.WAYPOINT_RAMP
                    rospy.loginfo(f"[NMPC] Advancing to waypoint {next_index + 1}")
            else:
                self._waypoint_hold_start = None
                rospy.loginfo("[NMPC] Waypoint sequence complete; holding final point")

        if (
            self.trajectory_mode == TrajectoryMode.LAND_HOLD
            and self._land_hold_start is not None
            and now - self._land_hold_start >= LAND_HOLD_SEC
        ):
            self.trajectory_mode = TrajectoryMode.LAND_DESCEND
            rospy.loginfo("[NMPC] Hold complete; commanding AUTO.LAND")
            if not self._land_descend_requested:
                try:
                    self.set_mode_service(custom_mode="AUTO.LAND")
                    self._land_descend_requested = True
                    self.trajectory_mode = TrajectoryMode.DONE
                except Exception as e:
                    rospy.logerr(f"[NMPC] AUTO.LAND request failed: {e}")
            return FlightPhase.RUN

        # ===== Main NMPC Control Loop =====
        # Build current state
        x_now = self._build_state_vector()
        if x_now is None:
            rospy.logwarn_throttle(5.0, "[NMPC] RUN: state not ready, publishing safe hover")
            self._publish_safe_hover()
            return FlightPhase.RUN

        # Build 9D reference [p, v, euler]. u_ref is built internally
        # by solve_mpc_control from quad.m * quad.g (see generate_nmpc.py).
        x_ref = self._build_reference_trajectory()

        self._solve_and_publish_reference(x_now, x_ref, "RUN")
        return FlightPhase.RUN

    def _control_loop(self, timer_event):
        """Main control loop (called at control_rate_hz)."""
        try:
            # State machine
            if self.phase == FlightPhase.INIT:
                self.phase = self._phase_init()
            elif self.phase == FlightPhase.WAIT_FOR_FCU:
                self.phase = self._phase_wait_for_fcu()
            elif self.phase == FlightPhase.STREAM_SETPOINTS:
                self.phase = self._phase_stream_setpoints()
            elif self.phase == FlightPhase.IDLE:
                self.phase = self._phase_idle()
            elif self.phase == FlightPhase.TAKEOFF_PX4:
                self.phase = self._phase_takeoff_px4()
            elif self.phase == FlightPhase.RUN:
                self.phase = self._phase_run()
        except Exception as e:
            rospy.logerr(f"[NMPC] Control loop exception: {e}")
            self._publish_safe_hover()

    def _on_shutdown(self):
        """Stop the control timer when ROS shuts down."""
        rospy.loginfo("[NMPC] Shutdown received")
        try:
            self.control_timer.shutdown()
        except Exception as e:
            rospy.logwarn(f"[NMPC] Failed to stop control timer on shutdown: {e}")


def main():
    """ROS node entry point."""
    try:
        node = NmpcNode()
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("[NMPC] Shutting down...")
    except Exception as e:
        rospy.logerr(f"[NMPC] Fatal error: {e}")
        raise


if __name__ == '__main__':
    main()
