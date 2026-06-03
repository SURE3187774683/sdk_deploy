#!/bin/bash

# 使用 -t 参数分配伪终端，因为 sudo 通常需要 tty
# 注意: 如果 drmap mapping -s 是持续运行的进程，需要在另一个终端单独执行
ssh -t user@10.21.31.106 "sudo drmap mapping -s"

sleep 5

cd /home/user/sdk_deploy

# 启动轨迹记录 (后台运行, --real 使用 foxy 环境)
bash src/M20_sdk_DF_deploy/scripts/start_record.sh --real &

source /opt/ros/foxy/setup.bash
source /opt/robot/scripts/setup_ros2.sh
# Use the gamepad to enable SDK mode. Need authorization code, please contact technical support team.

# Run
source install/setup.bash
ros2 run m20_sdk_DF_deploy rl_deploy