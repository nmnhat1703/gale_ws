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
        rospy.loginfo(f"[NMPC] Enable OFFBOARD on start: {self.enable_offboard_on_start}")

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

        # Running statistics
        self.solver_times = deque(maxlen=100)
        self._run_cycle_count = 0  # BUG 5: track cycles for logging

        # ===== Publishers =====
        self.attitude_pub = rospy.Publisher(
            '/mavros/setpoint_raw/attitude',
            AttitudeTarget,
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
            ref_per_stage = np.hstack((
                self.hover_position_ned,                # 3: position (NED)
                [0.0, 0.0, 0.0],                        # 3: velocity
                [0.0, 0.0, self.hover_yaw_ned],         # 3: euler (yaw only, NED)
            ))                                          # = 9D
            x_ref = np.tile(ref_per_stage, (DEFAULT_NUM_STEPS, 1))

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

        # Build 9D reference [p, v, euler]. u_ref is built internally
        # by solve_mpc_control from quad.m * quad.g (see generate_nmpc.py).
        ref_per_stage = np.hstack((
            self.hover_position_ned,                # 3: position (NED)
            [0.0, 0.0, 0.0],                        # 3: velocity
            [0.0, 0.0, self.hover_yaw_ned],         # 3: euler (yaw only, NED)
        ))                                          # = 9D
        x_ref = np.tile(ref_per_stage, (DEFAULT_NUM_STEPS, 1))

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
            1.0,
            "[NMPC] RUN diag: "
            f"status={status_str} use_solution={publish_solution} "
            f"failures={self.consecutive_solver_failures} "
            f"solve_ms={solve_time * 1000.0:.1f} "
            f"pos_ned_m=[{x_now[0]:+.2f},{x_now[1]:+.2f},{x_now[2]:+.2f}] "
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

    def _on_shutdown(self):
        """Shutdown handler: gracefully transition to POSCTL and stop publishing."""
        rospy.loginfo("[NMPC] Shutdown signal received")
        try:
            self.set_mode_service(custom_mode="POSCTL")
        except Exception as e:
            rospy.logwarn(f"[NMPC] Failed to set POSCTL on shutdown: {e}")


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
