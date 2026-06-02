#!/bin/bash

# 使用 -t 参数分配伪终端，因为 sudo 通常需要 tty
ssh -t user@10.21.31.106 "sudo drmap mapping -s"

sleep 5


source /opt/ros/foxy/setup.bash #source ROS2 env
source /opt/robot/scripts/setup_ros2.sh
# Use the gamepad to enable SDK mode. Need authorization code, please contact technical support team.

# Run
source install/setup.bash
ros2 run m20_sdk_DF_deploy rl_deploy