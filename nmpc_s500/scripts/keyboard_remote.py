#!/usr/bin/env python3
"""Terminal menu publisher for NMPC operator commands."""

import rospy
from std_msgs.msg import String


MENU = """========================================
       NMPC Keyboard Remote
========================================
1: takeoff
2: go to points
3: circle
4: eightloop
5: land
q: quit
========================================
Input your remote mode: """

COMMANDS = {
    '1': 'takeoff',
    '2': 'go to points',
    '3': 'circle',
    '4': 'eightloop',
    '5': 'land',
}


def main():
    rospy.init_node('nmpc_keyboard_remote', anonymous=True)
    pub = rospy.Publisher('/nmpc/command', String, queue_size=1)

    while not rospy.is_shutdown():
        try:
            choice = input(MENU).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if choice == 'q':
            break

        command = COMMANDS.get(choice)
        if command is None:
            rospy.logwarn(f"[NMPC] Unknown remote mode: {choice}")
            continue

        pub.publish(String(data=command))
        rospy.loginfo(f"[NMPC] Published command: {command}")


if __name__ == '__main__':
    main()
