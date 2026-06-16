source /opt/ros/foxy/setup.bash
source /opt/robot/scripts/setup_ros2.sh
source /home/user/sdk_deploy/install/setup.bash

ros2 bag record -o /home/user/sdk_deploy/src/M20_sdk_DF_deploy/traj_real/traj_real_1 \
  /WAYPOINT_PATH \
  /WAYPOINT_PATH/path \
  /WAYPOINT_PATH/markers \
  /WAYPOINT_PATH/poses \
  /SLAM_ODOM \
  