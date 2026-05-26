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
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.srv import CommandBool, SetMode

from nmpc_s500.platform_config import load_platform_config
from nmpc_s500.solver_setup import create_solver
# TODO: rename to flip_body_rates_frd_flu - name says world frame, function operates on body frame.
from nmpc_s500.frames import (
    enu_body_to_ned_body_rates,
    enu_to_ned_position,
    enu_to_ned_velocity,
    enu_yaw_to_ned_yaw,
    flu_to_frd_quaternion,
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
        shutdown_approach_param = rospy.get_param('~shutdown_approach_position', [0.0, 0.0, 1.0])
        self.shutdown_approach_position_enu = np.array(shutdown_approach_param, dtype=float)
        shutdown_land_param = rospy.get_param('~shutdown_land_position', [0.0, 0.0, 0.0])
        self.shutdown_land_position_enu = np.array(shutdown_land_param, dtype=float)
        self.shutdown_yaw_enu = float(rospy.get_param('~shutdown_yaw', 0.0))
        self.shutdown_land_on_ctrl_c = bool(rospy.get_param('~shutdown_land_on_ctrl_c', True))
        self.shutdown_approach_duration_sec = float(rospy.get_param('~shutdown_approach_duration_sec', 8.0))
        self.shutdown_approach_timeout_sec = float(rospy.get_param('~shutdown_approach_timeout_sec', 20.0))
        self.shutdown_land_timeout_sec = float(rospy.get_param('~shutdown_land_timeout_sec', 20.0))
        self.shutdown_land_rate_hz = float(rospy.get_param('~shutdown_land_rate_hz', 20.0))
        self.shutdown_land_tolerance_m = float(rospy.get_param('~shutdown_land_tolerance_m', 0.20))
        self.shutdown_land_hold_sec = float(rospy.get_param('~shutdown_land_hold_sec', 1.0))
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
        rospy.loginfo(
            f"[NMPC] Shutdown land: enabled={self.shutdown_land_on_ctrl_c}, "
            f"approach ENU={self.shutdown_approach_position_enu}, "
            f"yaw ENU={self.shutdown_yaw_enu:+.3f} rad, "
            f"land ENU={self.shutdown_land_position_enu}, "
            f"tolerance={self.shutdown_land_tolerance_m:.2f} m"
        )
        if self.trajectory_type == 'circle':
            rospy.loginfo(
                f"[NMPC] Circle trajectory: center ENU={circle_center_enu}, "
                f"center NED={self.circle_center_ned}, radius={self.circle_radius_m:.2f} m, "
                f"period={self.circle_period_sec:.1f} s, hover={self.circle_hover_sec:.1f} s, "
                f"ramp={self.circle_ramp_sec:.1f} s"
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

        # NICE TO HAVE: Register shutdown handler
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

    def _build_reference_trajectory(self, start_elapsed=None):
        """Build the NMPC reference trajectory for the full prediction horizon."""
        if self.trajectory_type == 'circle':
            if start_elapsed is None:
                start_elapsed = 0.0
            dt = DEFAULT_HORIZON / DEFAULT_NUM_STEPS
            return np.vstack([
                self._circle_reference_at(start_elapsed + i * dt)
                for i in range(DEFAULT_NUM_STEPS)
            ])

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
        """STREAM_SETPOINTS: publish hover setpoint for PX4 to accept OFFBOARD mode."""
        # Publish hover command at every cycle
        hover_u = np.array([
            self.platform_cfg.mass_kg * 9.81,  # hover thrust
            0.0, 0.0, 0.0  # zero body rates
        ])
        self._publish_attitude_target(hover_u)

        elapsed = time.time() - self.setpoint_stream_start_time
        if elapsed < SETPOINT_STREAM_DURATION_S:
            rospy.loginfo_throttle(
                2.0,
                f"[NMPC] STREAM: {elapsed:.1f}/{SETPOINT_STREAM_DURATION_S} s"
            )
            return FlightPhase.STREAM_SETPOINTS

        # Time to request OFFBOARD
        if self.enable_offboard_on_start:
            rospy.loginfo("[NMPC] STREAM: requesting OFFBOARD mode")
            return FlightPhase.REQUEST_OFFBOARD
        else:
            rospy.loginfo("[NMPC] STREAM: waiting for RC to engage OFFBOARD")
            return FlightPhase.RUN  # Skip OFFBOARD request, wait for manual RC

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

        # ===== Main NMPC Control Loop =====
        # Build current state
        x_now = self._build_state_vector()
        if x_now is None:
            rospy.logwarn_throttle(5.0, "[NMPC] RUN: state not ready, publishing safe hover")
            self._publish_safe_hover()
            return FlightPhase.RUN

        if self.run_start_time is None:
            self.run_start_time = time.time()
        run_elapsed = time.time() - self.run_start_time

        # Build 9D reference [p, v, euler]. u_ref is built internally
        # by solve_mpc_control from quad.m * quad.g (see generate_nmpc.py).
        x_ref = self._build_reference_trajectory(start_elapsed=run_elapsed)

        u_optimal, solve_time, status = self._solve(x_now, x_ref)

        # BUG 5: Use cycle counter for logging
        self._run_cycle_count += 1
        if self._run_cycle_count % 250 == 0 and len(self.solver_times) > 0:
            mean_time = np.mean(self.solver_times)
            max_time = np.max(self.solver_times)
            rospy.loginfo(
                f"[NMPC] Solver: mean {mean_time*1000:.1f} ms, "
                f"max {max_time*1000:.1f} ms (last 100 cycles)"
            )

        # Handle solver status
        use_solution = self._handle_solver_status(status)
        publish_solution = use_solution and u_optimal is not None
        u_to_publish = u_optimal if publish_solution else self.last_control_input

        if publish_solution:
            # Update last control for warm-start
            self.last_control_input = u_optimal
            self.consecutive_solver_failures = 0
        else:
            # Hold last good command
            self.consecutive_solver_failures += 1
            rospy.logwarn_throttle(
                5.0,
                f"[NMPC] Using last control (failure #{self.consecutive_solver_failures})"
            )

        # Diagnostic status for live SITL debugging.
        # State is solver-frame: position/velocity NED, Euler ZYX [roll, pitch, yaw].
        # u_to_publish is solver-frame FRD: [thrust_N, p, q, r in rad/s].
        # Must match the conversion in _publish_attitude_target; update both if changed.
        p_flu, q_flu, r_flu = enu_body_to_ned_body_rates(
            u_to_publish[1], u_to_publish[2], u_to_publish[3]
        )
        status_str = ACADOS_STATUS_LABELS.get(status, f"UNKNOWN({status})")
        rospy.loginfo_throttle(
            self.diag_log_period_sec,
            "[NMPC] RUN diag: "
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

        # Switch to safe hover after too many failures
        if (
            not publish_solution
            and self.consecutive_solver_failures >= MAX_CONSECUTIVE_SOLVER_FAILURES
        ):
            rospy.logerr("[NMPC] Too many solver failures, switching to safe hover")
            self._publish_safe_hover()
            # Can add a recovery strategy here if needed

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
            elif self.phase == FlightPhase.REQUEST_OFFBOARD:
                self.phase = self._phase_request_offboard()
            elif self.phase == FlightPhase.RUN:
                self.phase = self._phase_run()
        except Exception as e:
            rospy.logerr(f"[NMPC] Control loop exception: {e}")
            self._publish_safe_hover()

    @staticmethod
    def _enu_yaw_to_quaternion(yaw_enu):
        """Return a FLU/ENU quaternion for a world-Z yaw setpoint."""
        half_yaw = 0.5 * yaw_enu
        return 0.0, 0.0, float(np.sin(half_yaw)), float(np.cos(half_yaw))

    @staticmethod
    def _quaternion_to_enu_yaw(q):
        """Extract ENU yaw from a ROS quaternion."""
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    def _build_shutdown_land_setpoint(self, position_enu=None, yaw_enu=None):
        """Build a MAVROS local ENU position setpoint for shutdown landing."""
        if position_enu is None:
            position_enu = self.shutdown_land_position_enu

        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.pose.position.x = float(position_enu[0])
        msg.pose.position.y = float(position_enu[1])
        msg.pose.position.z = float(position_enu[2])

        if yaw_enu is None and self.pose is not None:
            msg.pose.orientation = self.pose.pose.orientation
        else:
            qx, qy, qz, qw = self._enu_yaw_to_quaternion(0.0 if yaw_enu is None else yaw_enu)
            msg.pose.orientation.x = qx
            msg.pose.orientation.y = qy
            msg.pose.orientation.z = qz
            msg.pose.orientation.w = qw

        return msg

    def _shutdown_land_position_error(self, target_enu):
        """Return current ENU distance to a shutdown target, or None."""
        if self.pose is None:
            return None

        dx = self.pose.pose.position.x - target_enu[0]
        dy = self.pose.pose.position.y - target_enu[1]
        dz = self.pose.pose.position.z - target_enu[2]
        return float(np.sqrt(dx * dx + dy * dy + dz * dz))

    def _publish_shutdown_position_until_reached(self, target_enu, yaw_enu, timeout_sec, label):
        """Publish one shutdown position target until it is held or times out."""
        rospy.loginfo(
            f"[NMPC] Shutdown {label}: target ENU "
            f"[{target_enu[0]:+.2f},{target_enu[1]:+.2f},{target_enu[2]:+.2f}], "
            f"yaw={'current' if yaw_enu is None else f'{yaw_enu:+.3f} rad'}"
        )

        target = self._build_shutdown_land_setpoint(target_enu, yaw_enu=yaw_enu)
        period = 1.0 / max(self.shutdown_land_rate_hz, 1.0)
        deadline = time.time() + timeout_sec
        reached_since = None
        last_offboard_request = 0.0

        while time.time() < deadline:
            target.header.stamp = rospy.Time.now()
            self.position_pub.publish(target)

            now = time.time()
            if now - last_offboard_request > 1.0:
                try:
                    self.set_mode_service(custom_mode="OFFBOARD")
                except Exception as e:
                    rospy.logwarn(f"[NMPC] Failed to keep OFFBOARD during shutdown {label}: {e}")
                last_offboard_request = now

            err = self._shutdown_land_position_error(target_enu)
            if err is not None:
                rospy.loginfo_throttle(
                    1.0,
                    f"[NMPC] Shutdown {label} error: {err:.2f} m"
                )
                if err <= self.shutdown_land_tolerance_m:
                    if reached_since is None:
                        reached_since = now
                    elif now - reached_since >= self.shutdown_land_hold_sec:
                        rospy.loginfo(f"[NMPC] Shutdown {label} target reached")
                        return True
                else:
                    reached_since = None

            time.sleep(period)

        rospy.logwarn(f"[NMPC] Shutdown {label} timed out")
        return False

    def _publish_shutdown_interpolated_approach(self):
        """Smoothly move from the current pose to the approach pose and yaw."""
        if self.pose is None:
            return False

        start_position = np.array([
            self.pose.pose.position.x,
            self.pose.pose.position.y,
            self.pose.pose.position.z,
        ], dtype=float)
        start_yaw = self._quaternion_to_enu_yaw(self.pose.pose.orientation)
        yaw_delta = wrap_to_pi(self.shutdown_yaw_enu - start_yaw)
        period = 1.0 / max(self.shutdown_land_rate_hz, 1.0)
        duration = max(self.shutdown_approach_duration_sec, period)
        steps = max(1, int(duration * self.shutdown_land_rate_hz))
        last_offboard_request = 0.0

        rospy.loginfo(
            f"[NMPC] Shutdown approach ramp: ENU "
            f"[{start_position[0]:+.2f},{start_position[1]:+.2f},{start_position[2]:+.2f}] -> "
            f"[{self.shutdown_approach_position_enu[0]:+.2f},"
            f"{self.shutdown_approach_position_enu[1]:+.2f},"
            f"{self.shutdown_approach_position_enu[2]:+.2f}] "
            f"and yaw {start_yaw:+.3f} -> {self.shutdown_yaw_enu:+.3f} rad"
        )

        for i in range(steps + 1):
            s = float(i) / float(steps)
            ramp, _ = self._smoothstep(s)
            position = start_position + ramp * (self.shutdown_approach_position_enu - start_position)
            yaw = wrap_to_pi(start_yaw + ramp * yaw_delta)
            target = self._build_shutdown_land_setpoint(position, yaw_enu=yaw)
            self.position_pub.publish(target)

            now = time.time()
            if now - last_offboard_request > 1.0:
                try:
                    self.set_mode_service(custom_mode="OFFBOARD")
                except Exception as e:
                    rospy.logwarn(f"[NMPC] Failed to keep OFFBOARD during shutdown approach ramp: {e}")
                last_offboard_request = now

            time.sleep(period)

        return True

    def _land_to_shutdown_position(self):
        """Best-effort Ctrl-C landing by publishing local position setpoints."""
        if not self.shutdown_land_on_ctrl_c:
            return False

        if self.pose is None:
            rospy.logwarn("[NMPC] Shutdown landing skipped: no local pose available")
            return False

        if self.mavros_state is not None and not self.mavros_state.connected:
            rospy.logwarn("[NMPC] Shutdown landing skipped: FCU is not connected")
            return False

        reached_approach = self._publish_shutdown_interpolated_approach()
        if not reached_approach:
            return False

        return self._publish_shutdown_position_until_reached(
            self.shutdown_land_position_enu,
            yaw_enu=self.shutdown_yaw_enu,
            timeout_sec=self.shutdown_land_timeout_sec,
            label="landing",
        )

    def _on_shutdown(self):
        """Shutdown handler: land to the configured origin, then leave OFFBOARD."""
        rospy.loginfo("[NMPC] Shutdown signal received")
        try:
            self.control_timer.shutdown()
        except Exception as e:
            rospy.logwarn(f"[NMPC] Failed to stop control timer on shutdown: {e}")

        landed = False
        try:
            landed = self._land_to_shutdown_position()
        except Exception as e:
            rospy.logwarn(f"[NMPC] Shutdown landing failed: {e}")

        try:
            self.set_mode_service(custom_mode="POSCTL")
        except Exception as e:
            rospy.logwarn(f"[NMPC] Failed to set POSCTL on shutdown: {e}")

        if landed:
            rospy.loginfo("[NMPC] Shutdown complete after landing target reached")


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
