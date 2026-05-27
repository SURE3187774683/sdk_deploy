# M20 SDK Deploy

[![Discord](https://img.shields.io/badge/-Discord-5865F2?style=flat&logo=Discord&logoColor=white)](https://discord.gg/gdM9mQutC8)
## Overview
This repository uses ROS2 to implement the entire Sim-to-sim and Sim-to-real workflow. Therefore, ROS2 must first be installed on your computer, such as installing [ROS2 Humble](https://docs.ros.org/en/humble/index.html) on Ubuntu 22.04. We've also released an introduction [video](https://www.youtube.com/watch?v=FNaxsDBtD7A), please check it out! Please go through the whole process on a Ubuntu system.
```mermaid
graph LR
    A["/rl_deploy"] -->|/JOINTS_CMD| B["/mujoco_simulation"]
    B -->|/IMU_DATA| A
    B -->|/JOINTS_DATA| A
```
```bash
# ros2 topic list
/BATTERY_DATA
/IMU_DATA
/JOINTS_CMD
/JOINTS_DATA
/parameter_events
/rosout


# ros2 node info /mujoco_simulation 
/mujoco_simulation
  Subscribers:
    /JOINTS_CMD: drdds/msg/JointsDataCmd
  Publishers:
    /IMU_DATA: drdds/msg/ImuData
    /JOINTS_DATA: drdds/msg/JointsData
    /parameter_events: rcl_interfaces/msg/ParameterEvent
    /rosout: rcl_interfaces/msg/Log
  Service Servers:
    /mujoco_simulation/describe_parameters: rcl_interfaces/srv/DescribeParameters
    /mujoco_simulation/get_parameter_types: rcl_interfaces/srv/GetParameterTypes
    /mujoco_simulation/get_parameters: rcl_interfaces/srv/GetParameters
    /mujoco_simulation/list_parameters: rcl_interfaces/srv/ListParameters
    /mujoco_simulation/set_parameters: rcl_interfaces/srv/SetParameters
    /mujoco_simulation/set_parameters_atomically: rcl_interfaces/srv/SetParametersAtomically
  Service Clients:

  Action Servers:

  Action Clients:


# ros2 node info /rl_deploy 
/rl_deploy
  Subscribers:
    /BATTERY_DATA: drdds/msg/BatteryData
    /IMU_DATA: drdds/msg/ImuData
    /JOINTS_DATA: drdds/msg/JointsData
    /parameter_events: rcl_interfaces/msg/ParameterEvent
  Publishers:
    /JOINTS_CMD: drdds/msg/JointsDataCmd
    /parameter_events: rcl_interfaces/msg/ParameterEvent
    /rosout: rcl_interfaces/msg/Log
  Service Servers:
    /rl_deploy/describe_parameters: rcl_interfaces/srv/DescribeParameters
    /rl_deploy/get_parameter_types: rcl_interfaces/srv/GetParameterTypes
    /rl_deploy/get_parameters: rcl_interfaces/srv/GetParameters
    /rl_deploy/list_parameters: rcl_interfaces/srv/ListParameters
    /rl_deploy/set_parameters: rcl_interfaces/srv/SetParameters
    /rl_deploy/set_parameters_atomically: rcl_interfaces/srv/SetParametersAtomically
  Service Clients:

  Action Servers:

  Action Clients:

```
## Contribution 

Everyone is welcome to contribute to this repo. If you discover a bug or optimize our training config, just submit a pull request and we will look into it.
## Sim-to-sim

```bash
pip install "numpy < 2.0" mujoco
git clone https://github.com/DeepRoboticsLab/sdk_deploy.git

# Compile
cd sdk_deploy
source /opt/ros/humble/setup.bash
colcon build --packages-up-to m20_sdk_DF_deploy --cmake-args -DBUILD_PLATFORM=x86
```

```bash
# Run (Open 2 terminals)
# Terminal 1
export ROS_DOMAIN_ID=1
source install/setup.bash
ros2 run m20_sdk_DF_deploy rl_deploy

# Terminal 2 
export ROS_DOMAIN_ID=1
source install/setup.bash
python3 src/M20_sdk_DF_deploy/interface/robot/simulation/mujoco_simulation_ros2_waypoint12.py
```

### Control (Terminal 2)

<span style="color: red;">**Note:**</span>
> - Right click simulator window and select "always on top"
> - When the robot dog stands up, it may become stuck due to self-collision in the simulation. This is not a bug; please try again.
> - z： default position
> - c： rl control default position
> - wasd：forward/leftward/backward/rightward
> - qe：clockwise/counter clockwise


# Sim-to-Real
This process is almost identical to simulation-simulation. You only need to add the step of connecting to Wi-Fi to transfer data, and then modify the compilation instructions.Real-robot control is divided into keyboard mode and gamepad control mode. You need to modify the RemoteCommandType parameter in the main function to select the desired mode.


**Please first use the OTA upgrade function in the handle settings to upgrade the hardware to version 1.1.8. We require a sdk authentication code to activate the sdk mode. Please contact our technical support team to get this unique code for each robot.**


```bash

# computer and gamepad should both connect to WiFi
# WiFi: M20********
# Passward: 12345678 (If wrong, contact technical support)

# scp to transfer files to quadruped (open a terminal on your local computer) password is ' (a single quote)
scp -r ~/sdk_deploy/src user@10.21.31.103:~/sdk_deploy

# ssh connect for remote development, 
ssh user@10.21.31.103
cd sdk_deploy
source /opt/ros/foxy/setup.bash #source ROS2 env
colcon build --packages-select m20_sdk_DF_deploy --cmake-args -DBUILD_PLATFORM=arm


sudo su # Root
source /opt/ros/foxy/setup.bash #source ROS2 env
source /opt/robot/scripts/setup_ros2.sh
# Use the gamepad to enable SDK mode. Need authorization code, please contact technical support team.

# Run
source install/setup.bash
ros2 run m20_sdk_DF_deploy rl_deploy

# exit sdk mode：
# Use the gamepad to enable SDK mode.
```

### ⌨️ Keyboard Control
- z： default position
- s： waypoint tracking
- x： lie down
- p： 暂停回站立；站立状态下再按 P 恢复并切到下一个 waypoint
- r： 关节阻尼/急停式保护

### 🎮 Gamepad Control
*(Note: When using the gamepad control function, please ensure that the Gamepad APP version is V1.5.11 or higher.)*
![2838b054246d4700247b36207243258f](https://github.com/user-attachments/assets/ed4e8340-c1fe-4202-916a-84e80e537b7f)

- L1： default position
- L2： rl control default position
- R1： lie down
- R2： joint damping
- Left joystick：forward/leftward/backward/rightward
- Right joystick：clockwise/counter clockwise

/BATTERY_CHARGE_ENABLE
/BATTERY_DATA
/CHARGE_CMD
/CHARGE_STATUS
/CPU_103
/CPU_104
/CPU_106
/DEPTH_IMAGE
/EXCEPTION_NOTIFICATION
/FAULT_STATUS
/FIBOCOM/net_rtk/gngga
/FIBOCOM/net_rtk/heading
/GAIT
/GLOBAL_PLANNER_STATUS
/GPS
/GPS_CFGSYS
/GPS_SYS_MODE
/GRIDS_ID
/GRID_MAP
/HANDLER_POINTS_DEBUG
/HANDLE_STEER
/HEIGHT_IMAGE
/HEIGHT_MAP_STATUS
/HES_STATUS
/IMU
/IMU_YESENSE
/JOINTS_CMD
/JOINTS_DATA
/JOINTS_DATA_10HZ
/LED/STATUS
/LIDAR/IMU201
/LIDAR/IMU202
/LIDAR/STATUS
/LIO_ODOM
/LIO_ODOM_HIGH_FREQUENCY
/LOCATION_STATUS
/LOCATION_STATUS/MATCHING_ERROR
/LOC_BODY_POINTS
/MOTION_INFO
/MOTION_STATE
/MOTION_STATUS
/NAV_CMD
/NAV_STATUS
/ODOM
/OOA_STATUS
/PASSABLE_AREA_ENABLE
/PLANNER_STATUS
/REAL_STEER
/SEG_CLOUD
/STEER
/TERRAIN_CLASSIFIER_STATUS
/TRACK_PATH
/UWB_ENABLE
/UWB_ODOM_LOST
/UWB_ONLINE
/WEIGHT_ITEMS
/accumulate_cloud/cloud_base
/accumulate_cloud/cloud_gravity
/accumulate_cloud/status_code
/body_visual
/cloud_depth
/cloud_local
/cloud_local_g
/cloud_nav
/cloud_now_g
/cloud_obs
/dr/MotionInfo
/fibo_fusion_pose
/fibo_fusion_state
/free_paths
/global_path
/global_path_markers
/goal_baselink
/grid_map
/grid_map_3d
/height_map
/impassable_area
/initialpose
/local_goal
/local_goal_baselink
/local_map
/local_path
/local_scans
/parameter_events
/passable_area
/passable_status_code
/path
/path_Astar
/planner_mode
/pose_in_apriltag_corrected
/rosout
/tag_status
/target_goal
/tf
/track_path_baselink
/traversal_cost
/vis_global_points
/vis_global_points_pruned