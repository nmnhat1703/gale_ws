cmake_minimum_required(VERSION 3.0.2)
project(nmpc_s500)

find_package(catkin REQUIRED COMPONENTS
  rospy
  mavros_msgs
  geometry_msgs
  std_msgs
  std_srvs
  tf2_ros
  tf2_geometry_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS rospy mavros_msgs geometry_msgs std_msgs std_srvs tf2_ros tf2_geometry_msgs
)

# Install Python entrypoints with executable permissions in devel/install space.
file(GLOB NMPC_S500_PYTHON_PROGRAMS
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  scripts/*.py
  nmpc_s500/*.py
)
list(REMOVE_ITEM NMPC_S500_PYTHON_PROGRAMS nmpc_s500/__init__.py)

catkin_install_python(PROGRAMS
  ${NMPC_S500_PYTHON_PROGRAMS}
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

# Install config files
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  catkin_add_nosetests(tests/)
endif()
