#!/usr/bin/env python3

import math
import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State


current_state = State()
have_pose = False
current_pose = PoseStamped()


def state_cb(msg):
    global current_state
    current_state = msg


def pose_cb(msg):
    global have_pose, current_pose
    have_pose = True
    current_pose = msg


def make_pose(x, y, z, orientation):
    """
    Create a MAVROS local ENU position setpoint.
    Keeps the provided orientation, so yaw is not changed.
    """
    pose = PoseStamped()
    pose.header.stamp = rospy.Time.now()
    pose.header.frame_id = "odom"

    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = z
    pose.pose.orientation = orientation

    return pose


def publish_pose(pub, pose):
    pose.header.stamp = rospy.Time.now()
    pub.publish(pose)


def position_error(pose, target_pose):
    dx = pose.pose.position.x - target_pose.pose.position.x
    dy = pose.pose.position.y - target_pose.pose.position.y
    dz = pose.pose.position.z - target_pose.pose.position.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def smoothstep(a):
    """
    Smooth interpolation from 0 to 1.
    It starts slow, moves faster in the middle, then slows near the target.
    """
    return a * a * (3.0 - 2.0 * a)


def move_interpolated_to_waypoint(
    pub,
    target_pose,
    duration_s=8.0,
    final_tolerance_m=0.18,
    final_stable_time_s=0.8,
    final_timeout_s=8.0,
    rate_hz=20
):
    """
    Move from current position to target by publishing many interpolated setpoints.

    Phase 1:
    - Generate smooth intermediate setpoints from current pose to target_pose.
    - This prevents a sudden position jump and reduces overshoot.

    Phase 2:
    - Hold the final target until the drone is actually close enough and stable.
    """
    rate = rospy.Rate(rate_hz)

    start_pose = make_pose(
        current_pose.pose.position.x,
        current_pose.pose.position.y,
        current_pose.pose.position.z,
        target_pose.pose.orientation
    )

    sx = start_pose.pose.position.x
    sy = start_pose.pose.position.y
    sz = start_pose.pose.position.z

    tx = target_pose.pose.position.x
    ty = target_pose.pose.position.y
    tz = target_pose.pose.position.z

    rospy.loginfo(
        "Interpolated move: start (%.2f, %.2f, %.2f) -> target (%.2f, %.2f, %.2f) over %.1f s",
        sx, sy, sz, tx, ty, tz, duration_s
    )

    steps = max(1, int(duration_s * rate_hz))

    # Phase 1: publish interpolated points
    for i in range(steps):
        if rospy.is_shutdown():
            return False

        if current_state.mode != "OFFBOARD":
            rospy.logwarn("Exited OFFBOARD during interpolated move.")
            return False

        raw_a = float(i + 1) / float(steps)
        a = smoothstep(raw_a)

        x = sx + a * (tx - sx)
        y = sy + a * (ty - sy)
        z = sz + a * (tz - sz)

        cmd = make_pose(x, y, z, target_pose.pose.orientation)
        publish_pose(pub, cmd)

        rospy.loginfo_throttle(
            1.0,
            "Interpolating: setpoint x=%.2f y=%.2f z=%.2f | current x=%.2f y=%.2f z=%.2f",
            x, y, z,
            current_pose.pose.position.x,
            current_pose.pose.position.y,
            current_pose.pose.position.z
        )

        rate.sleep()

    # Phase 2: final precise/stable hold near corner
    rospy.loginfo(
        "Interpolation complete. Holding final corner until error < %.2f m for %.1f s",
        final_tolerance_m,
        final_stable_time_s
    )

    start_wait = rospy.Time.now()
    inside_since = None

    while not rospy.is_shutdown():
        if current_state.mode != "OFFBOARD":
            rospy.logwarn("Exited OFFBOARD while waiting at final corner.")
            return False

        publish_pose(pub, target_pose)

        err = position_error(current_pose, target_pose)

        rospy.loginfo_throttle(
            1.0,
            "Final corner hold: target x=%.2f y=%.2f z=%.2f | current x=%.2f y=%.2f z=%.2f | error=%.2f m",
            tx, ty, tz,
            current_pose.pose.position.x,
            current_pose.pose.position.y,
            current_pose.pose.position.z,
            err
        )

        if err < final_tolerance_m:
            if inside_since is None:
                inside_since = rospy.Time.now()

            if (rospy.Time.now() - inside_since).to_sec() >= final_stable_time_s:
                rospy.loginfo("Corner reached smoothly. Final error = %.2f m", err)
                return True
        else:
            inside_since = None

        if (rospy.Time.now() - start_wait).to_sec() > final_timeout_s:
            rospy.logwarn(
                "Final corner wait timeout. Error still %.2f m. Continue to next corner.",
                err
            )
            return True

        rate.sleep()


def hold_waypoint(pub, target_pose, hold_time_s=2.0, rate_hz=20):
    """
    Hold current waypoint for a short time before sending next waypoint.
    """
    rate = rospy.Rate(rate_hz)
    end_time = rospy.Time.now() + rospy.Duration(hold_time_s)

    rospy.loginfo("Holding waypoint for %.1f s", hold_time_s)

    while not rospy.is_shutdown() and rospy.Time.now() < end_time:
        if current_state.mode != "OFFBOARD":
            rospy.logwarn("Exited OFFBOARD during hold.")
            return False

        publish_pose(pub, target_pose)
        rate.sleep()

    return True


def main():
    rospy.init_node("square_offboard_waypoints_interpolated")

    rospy.Subscriber("/mavros/state", State, state_cb)
    rospy.Subscriber("/mavros/local_position/pose", PoseStamped, pose_cb)

    setpoint_pub = rospy.Publisher(
        "/mavros/setpoint_position/local",
        PoseStamped,
        queue_size=10
    )

    rate_hz = 20
    rate = rospy.Rate(rate_hz)

    rospy.loginfo("Waiting for FCU connection...")
    while not rospy.is_shutdown() and not current_state.connected:
        rate.sleep()

    rospy.loginfo("Waiting for local position...")
    while not rospy.is_shutdown() and not have_pose:
        rate.sleep()

    rospy.loginfo("Connected and local position received.")

    # Square settings
    square_edge_m = 2.0
    half_edge = square_edge_m / 2.0
    flight_z = 1.0

    # Takeoff position/local origin is the centre
    centre_x = 0.0
    centre_y = 0.0

    # Square corners around centre.
    # 2 m x 2 m square centred at (0, 0).
    square_points = [
        (centre_x + half_edge, centre_y + half_edge, flight_z),
        (centre_x + half_edge, centre_y - half_edge, flight_z),
        (centre_x - half_edge, centre_y - half_edge, flight_z),
        (centre_x - half_edge, centre_y + half_edge, flight_z),
        (centre_x + half_edge, centre_y + half_edge, flight_z),  # close square
        (centre_x,             centre_y,             flight_z),  # return to centre
    ]

    rospy.loginfo("Publishing current-position hold setpoint.")
    rospy.loginfo("Take off manually in Position mode.")
    rospy.loginfo("When stable at around z=1 m, flip RC switch 5 to OFFBOARD.")
    rospy.loginfo("Mission: interpolated 2 m x 2 m square, centre=(0,0), z=1 m, yaw unchanged.")

    mission_started = False
    final_pose = None

    while not rospy.is_shutdown():

        # Before Offboard starts, keep updating hold pose to current position.
        # This lets you take off manually first.
        if current_state.mode != "OFFBOARD" and not mission_started:
            hold_pose = make_pose(
                current_pose.pose.position.x,
                current_pose.pose.position.y,
                current_pose.pose.position.z,
                current_pose.pose.orientation
            )

            publish_pose(setpoint_pub, hold_pose)
            rospy.loginfo_throttle(
                2.0,
                "Waiting for OFFBOARD. Publishing current-position hold setpoint."
            )
            rate.sleep()
            continue

        # Start square mission once when OFFBOARD becomes active.
        if current_state.mode == "OFFBOARD" and not mission_started:
            mission_started = True

            # Keep yaw fixed using current orientation when Offboard starts.
            fixed_orientation = current_pose.pose.orientation

            rospy.loginfo("OFFBOARD detected. Starting interpolated square mission.")
            rospy.loginfo(
                "Current pose at OFFBOARD: x=%.2f y=%.2f z=%.2f",
                current_pose.pose.position.x,
                current_pose.pose.position.y,
                current_pose.pose.position.z
            )

            for i, (x, y, z) in enumerate(square_points):
                if rospy.is_shutdown():
                    break

                target_pose = make_pose(x, y, z, fixed_orientation)
                final_pose = target_pose

                rospy.loginfo(
                    "Moving to corner %d/%d: x=%.2f y=%.2f z=%.2f",
                    i + 1,
                    len(square_points),
                    x,
                    y,
                    z
                )

                # For a 2 m side on S500, 8 s per side is gentle.
                # Increase to 10-12 s if it still overshoots.
                reached = move_interpolated_to_waypoint(
                    setpoint_pub,
                    target_pose,
                    duration_s=8.0,
                    final_tolerance_m=0.18,
                    final_stable_time_s=0.8,
                    final_timeout_s=8.0,
                    rate_hz=rate_hz
                )

                if not reached:
                    rospy.logwarn("Mission interrupted before reaching corner.")
                    break

                held = hold_waypoint(
                    setpoint_pub,
                    target_pose,
                    hold_time_s=2.0,
                    rate_hz=rate_hz
                )

                if not held:
                    rospy.logwarn("Mission interrupted during corner hold.")
                    break

            rospy.loginfo("Interpolated square mission complete. Holding final setpoint.")

        # After mission, keep publishing final pose if still in Offboard.
        if mission_started and final_pose is not None:
            if current_state.mode == "OFFBOARD":
                publish_pose(setpoint_pub, final_pose)
                rospy.loginfo_throttle(
                    2.0,
                    "OFFBOARD active: holding final point x=%.2f y=%.2f z=%.2f",
                    final_pose.pose.position.x,
                    final_pose.pose.position.y,
                    final_pose.pose.position.z
                )
            else:
                rospy.logwarn_throttle(
                    2.0,
                    "Not in OFFBOARD. Script still running but vehicle is not following Offboard setpoint."
                )

        rate.sleep()


if __name__ == "__main__":
    main()