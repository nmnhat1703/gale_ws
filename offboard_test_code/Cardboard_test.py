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


def smoothstep(a):
    """
    Smooth interpolation factor from 0 to 1.
    This avoids sudden acceleration at start/end.
    """
    return a * a * (3.0 - 2.0 * a)


def position_error(pose, target_pose):
    dx = pose.pose.position.x - target_pose.pose.position.x
    dy = pose.pose.position.y - target_pose.pose.position.y
    dz = pose.pose.position.z - target_pose.pose.position.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def move_smoothly(pub, start_pose, target_pose, duration_s=5.0, rate_hz=20):
    """
    Smoothly move from start_pose to target_pose over duration_s.
    If the drone exits OFFBOARD, stop the motion.
    """
    rate = rospy.Rate(rate_hz)
    steps = max(1, int(duration_s * rate_hz))

    sx = start_pose.pose.position.x
    sy = start_pose.pose.position.y
    sz = start_pose.pose.position.z

    tx = target_pose.pose.position.x
    ty = target_pose.pose.position.y
    tz = target_pose.pose.position.z

    orientation = target_pose.pose.orientation

    rospy.loginfo(
        "Smooth motion: start (%.2f, %.2f, %.2f) -> target (%.2f, %.2f, %.2f) over %.1f s",
        sx, sy, sz, tx, ty, tz, duration_s
    )

    for i in range(steps):
        if rospy.is_shutdown():
            return False

        if current_state.mode != "OFFBOARD":
            rospy.logwarn("Exited OFFBOARD during smooth motion. Stopping mission.")
            return False

        a = float(i + 1) / float(steps)
        a = smoothstep(a)

        x = sx + a * (tx - sx)
        y = sy + a * (ty - sy)
        z = sz + a * (tz - sz)

        cmd = make_pose(x, y, z, orientation)
        publish_pose(pub, cmd)

        rospy.loginfo_throttle(
            1.0,
            "Moving: setpoint x=%.2f y=%.2f z=%.2f | current x=%.2f y=%.2f z=%.2f",
            x, y, z,
            current_pose.pose.position.x,
            current_pose.pose.position.y,
            current_pose.pose.position.z
        )

        rate.sleep()

    rospy.loginfo("Finished smooth motion to target setpoint.")
    return True


def wait_until_reached(
    pub,
    target_pose,
    tolerance_m=0.18,
    stable_time_s=0.8,
    timeout_s=8.0,
    rate_hz=20
):
    """
    After interpolation, hold the final point until the drone is close enough.
    """
    rate = rospy.Rate(rate_hz)
    start_time = rospy.Time.now()
    inside_since = None

    while not rospy.is_shutdown():
        if current_state.mode != "OFFBOARD":
            rospy.logwarn("Exited OFFBOARD while waiting at target.")
            return False

        publish_pose(pub, target_pose)

        err = position_error(current_pose, target_pose)

        rospy.loginfo_throttle(
            1.0,
            "Waiting target: x=%.2f y=%.2f z=%.2f | current x=%.2f y=%.2f z=%.2f | error=%.2f m",
            target_pose.pose.position.x,
            target_pose.pose.position.y,
            target_pose.pose.position.z,
            current_pose.pose.position.x,
            current_pose.pose.position.y,
            current_pose.pose.position.z,
            err
        )

        if err < tolerance_m:
            if inside_since is None:
                inside_since = rospy.Time.now()

            if (rospy.Time.now() - inside_since).to_sec() >= stable_time_s:
                rospy.loginfo("Target reached and stable. Error = %.2f m", err)
                return True
        else:
            inside_since = None

        if (rospy.Time.now() - start_time).to_sec() > timeout_s:
            rospy.logwarn(
                "Target wait timeout after %.1f s. Error still %.2f m. Continue anyway.",
                timeout_s,
                err
            )
            return True

        rate.sleep()


def hold_target(pub, target_pose, hold_time_s=2.0, rate_hz=20):
    """
    Hold a point briefly before going to the next point.
    """
    rate = rospy.Rate(rate_hz)
    end_time = rospy.Time.now() + rospy.Duration(hold_time_s)

    rospy.loginfo("Holding target for %.1f s", hold_time_s)

    while not rospy.is_shutdown() and rospy.Time.now() < end_time:
        if current_state.mode != "OFFBOARD":
            rospy.logwarn("Exited OFFBOARD during hold.")
            return False

        publish_pose(pub, target_pose)
        rate.sleep()

    return True


def main():
    rospy.init_node("two_point_offboard_smooth")

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

    # Waypoints in MAVROS local ENU frame.
    # These are absolute local positions.
    waypoints = [
        (1.5, 0.0, 0.3),
        (3.4, 0.0, 0.3),
        (2.0, 0.0, 0.3),
        (0.0, 0.0, 1.0),
    ]

    rospy.loginfo("Publishing hold setpoint at current position.")
    rospy.loginfo("Take off manually in Position mode.")
    rospy.loginfo("When stable, flip RC switch 5 to OFFBOARD.")
    rospy.loginfo("Mission: go to (1.5, 0, 0.3), then go to (3.4, 0, 0.3).")

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

        # Once Offboard is detected, start the mission once.
        if current_state.mode == "OFFBOARD" and not mission_started:
            mission_started = True

            rospy.loginfo("OFFBOARD detected. Starting two-point mission.")

            # Keep current yaw/orientation. Only command position.
            fixed_orientation = current_pose.pose.orientation

            for i, (x, y, z) in enumerate(waypoints):
                if rospy.is_shutdown():
                    break

                rospy.loginfo(
                    "Going to waypoint %d/%d: x=%.2f, y=%.2f, z=%.2f",
                    i + 1,
                    len(waypoints),
                    x,
                    y,
                    z
                )

                start_pose = make_pose(
                    current_pose.pose.position.x,
                    current_pose.pose.position.y,
                    current_pose.pose.position.z,
                    fixed_orientation
                )

                target_pose = make_pose(
                    x,
                    y,
                    z,
                    fixed_orientation
                )

                final_pose = target_pose

                # First point is 1.5 m from origin, second segment is 1.9 m.
                # 5 seconds is gentle. Use 4 seconds if you want faster.
                success = move_smoothly(
                    setpoint_pub,
                    start_pose,
                    target_pose,
                    duration_s=5.0,
                    rate_hz=rate_hz
                )

                if not success:
                    rospy.logwarn("Mission interrupted during movement.")
                    break

                reached = wait_until_reached(
                    setpoint_pub,
                    target_pose,
                    tolerance_m=0.18,
                    stable_time_s=0.8,
                    timeout_s=8.0,
                    rate_hz=rate_hz
                )

                if not reached:
                    rospy.logwarn("Mission interrupted while waiting for target.")
                    break
                
                if i == 1:
                    rospy.loginfo("Waypoint 2 reached. Skipping hold and going immediately to waypoint 3.")
                    continue

                held = hold_target(
                    setpoint_pub,
                    target_pose,
                    hold_time_s=2.0,
                    rate_hz=rate_hz
                )

                if not held:
                    rospy.logwarn("Mission interrupted during hold.")
                    break

            rospy.loginfo("Mission complete. Holding final target.")

        # After mission, keep publishing final target if still in Offboard.
        if mission_started and final_pose is not None:
            if current_state.mode == "OFFBOARD":
                publish_pose(setpoint_pub, final_pose)
                rospy.loginfo_throttle(
                    2.0,
                    "OFFBOARD active: holding final x=%.2f, y=%.2f, z=%.2f",
                    final_pose.pose.position.x,
                    final_pose.pose.position.y,
                    final_pose.pose.position.z
                )
            else:
                rospy.logwarn_throttle(
                    2.0,
                    "Not in OFFBOARD. Script still running, but vehicle is not following Offboard setpoint."
                )

        rate.sleep()


if __name__ == "__main__":
    main()