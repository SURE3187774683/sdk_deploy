/**
 * @file dds_hardware_interface.hpp
 * @brief hardware interface for dds
 * @author DeepRobotics
 * @version 1.0
 * @date 2025-11-07
 * 
 * @copyright Copyright (c) 2025  DeepRobotics
 * 
 */

#pragma once


#include <algorithm>
#include "common_types.h"
#include "robot_interface.h"
#include "dds_types.h"

#include "drdds/msg/joints_data.hpp"
#include "drdds/msg/joints_data_cmd.hpp"
#include "drdds/msg/imu_data.hpp"
#include "drdds/msg/battery_data.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include <cstdint>

using namespace dds;

struct JointConfig {
    float dir;
    float offset;
};


class DdsInterface : public RobotInterface{
protected:
    double ri_ts_ = 0.0, imu_ts_ = 0.0;
    Vec3f omega_body_, rpy_, acc_;
    Vec3f base_pos_w_ = Vec3f::Zero();
    float base_height_ref_ = 0.55f;
    bool slam_odom_received_ = false;
    bool sensor_imu_received_ = false;
    uint64_t dds_imu_debug_cnt_ = 0;
    uint64_t sensor_imu_debug_cnt_ = 0;
    uint64_t slam_odom_debug_cnt_ = 0;
    uint64_t base_pose_debug_cnt_ = 0;
    VecXf joint_pos_, joint_vel_, joint_tau_;
    VecXf motor_temperture_, driver_temperture_;
    float *pos_offset_;
    float *joint_dir_;
    bool *data_updated_;
    uint16_t *control_word;

    JointConfig *joint_config_;
    std::vector<uint16_t> driver_status_;
    std::vector<uint16_t> joint_data_id_;
    BatteryInfo_dds battery_info_[2];
    std::vector<uint16_t> battery_data_;

    double time_stamp_;
    float remain_bat;

    rclcpp::Publisher<drdds::msg::JointsDataCmd>::SharedPtr joint_cmd_pub_;
    rclcpp::Subscription<drdds::msg::JointsData>::SharedPtr joint_data_sub_;
    rclcpp::Subscription<drdds::msg::ImuData>::SharedPtr imu_data_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_raw_sub_;
    rclcpp::Subscription<drdds::msg::BatteryData>::SharedPtr health_data_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr slam_odom_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr base_pose_sub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr waypoint_path_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr waypoint_path_viz_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr waypoint_pose_array_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr waypoint_marker_pub_;


    bool IsDataUpdatedFinished() {
        bool res = data_updated_[0];
        for (int i = 0; i < dof_num_; ++i) {
            res = res & data_updated_[i];
        }
        return res;
    }

    Vec3f QuaternionToRpy(const double x, const double y, const double z, const double w) {
        const double sinr_cosp = 2.0 * (w * x + y * z);
        const double cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
        const double roll = std::atan2(sinr_cosp, cosr_cosp);

        const double sinp = 2.0 * (w * y - z * x);
        double pitch = 0.0;
        if (std::abs(sinp) >= 1.0) {
            pitch = std::copysign(M_PI / 2.0, sinp);
        } else {
            pitch = std::asin(sinp);
        }

        const double siny_cosp = 2.0 * (w * z + x * y);
        const double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
        const double yaw = std::atan2(siny_cosp, cosy_cosp);

        return Vec3f(static_cast<float>(roll), static_cast<float>(pitch), static_cast<float>(yaw));
    }

    void ResetPositionOffset() {
        memset(data_updated_, 0, dof_num_ * sizeof(bool));
        this->SetJointCommand(MatXf::Zero(dof_num_, 5));
        usleep(100 * 1000);
        VecXf last_joint_pos = this->GetJointPosition();
        VecXf current_joint_pos = this->GetJointPosition();
        int cnt = 0;
        while (!IsDataUpdatedFinished()) {
            ++cnt;
            this->RefreshRobotData();
            rclcpp::spin_some(this->get_node());
            current_joint_pos = this->GetJointPosition();

            for (int i = 0; i < dof_num_; ++i) {
                if (!data_updated_[i] && current_joint_pos(i) != last_joint_pos(i)) {
                    data_updated_[i] = true;
                }
            }
            last_joint_pos = current_joint_pos;
            usleep(1000);
            if (cnt == 10000) {
                std::cout << data_updated_ << std::endl;
                std::cout << "joint data update is not finished\n";
            }
        }
        for (int i = 1; i < dof_num_; i += 4) {
            if (current_joint_pos(i) < -210. / 180. * M_PI) {
                pos_offset_[i] = pos_offset_[i] + 360.;
                std::cout << "joint " << i << " offset is changed to " << pos_offset_[i] << "\n";
            } else if (current_joint_pos(i) > 90. / 180. * M_PI) {
                pos_offset_[i] = pos_offset_[i] - 360.;
                std::cout << "joint " << i << " offset is changed to " << pos_offset_[i] << "\n";
            }
        }
        for (int i = 0; i < dof_num_; ++i) {
            joint_config_[i].dir = joint_dir_[i];
            joint_config_[i].offset = Deg2Rad(pos_offset_[i]);
        }
    }

public:
    int run_cnt_ = 0;

    DdsInterface(const std::string &robot_name, const int &dof_num) : RobotInterface(robot_name, dof_num) {
        std::cout << robot_name << " is using Ecan Hardware Interface" << std::endl;
        joint_pos_ = VecXf::Zero(dof_num_);
        joint_vel_ = VecXf::Zero(dof_num_);
        joint_tau_ = VecXf::Zero(dof_num_);
        motor_temperture_ = VecXf::Zero(dof_num_);
        driver_temperture_ = VecXf::Zero(dof_num_);

        pos_offset_ = new float[dof_num_];
        joint_dir_ = new float[dof_num_];
        data_updated_ = new bool[dof_num_];
        joint_config_ = new JointConfig[dof_num_];
        control_word = new uint16_t[4];

        driver_status_.resize(dof_num_);
        joint_data_id_.resize(dof_num_);
        battery_data_.resize(BATTERY_DATA_SIZE);

        joint_cmd_pub_ = node_->create_publisher<drdds::msg::JointsDataCmd>("/JOINTS_CMD", 10);
        joint_data_sub_ = node_->create_subscription<drdds::msg::JointsData>("/JOINTS_DATA", 10,
                                    std::bind(&DdsInterface::Handler, this,std::placeholders::_1));
        imu_data_sub_ = node_->create_subscription<drdds::msg::ImuData>("/IMU_DATA", 10,
                                    std::bind(&DdsInterface::HandlerIMU, this, std::placeholders::_1));
        imu_raw_sub_ = node_->create_subscription<sensor_msgs::msg::Imu>("/IMU", 10,
                                    std::bind(&DdsInterface::HandlerSensorImu, this, std::placeholders::_1));
        health_data_sub_ = node_->create_subscription<drdds::msg::BatteryData>("/BATTERY_DATA", 10,
                                    std::bind(&DdsInterface::HandlerHealth, this, std::placeholders::_1));
        slam_odom_sub_ = node_->create_subscription<nav_msgs::msg::Odometry>(
                                    "/SLAM_ODOM", 10,
                                    std::bind(&DdsInterface::HandlerSlamOdom, this, std::placeholders::_1));
        base_pose_sub_ = node_->create_subscription<std_msgs::msg::Float32MultiArray>(
                                    "/BASE_POSE2D", 10,
                                    std::bind(&DdsInterface::HandlerBasePose, this, std::placeholders::_1));
        waypoint_path_pub_ = node_->create_publisher<std_msgs::msg::Float32MultiArray>(
                                    "/WAYPOINT_PATH", 10);
        waypoint_path_viz_pub_ = node_->create_publisher<nav_msgs::msg::Path>(
                                    "/WAYPOINT_PATH/path", 10);
        waypoint_pose_array_pub_ = node_->create_publisher<geometry_msgs::msg::PoseArray>(
                                    "/WAYPOINT_PATH/poses", 10);
        waypoint_marker_pub_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>(
                                    "/WAYPOINT_PATH/markers", 10);
        
        sleep(1);

        control_word[0] = kIndexDisable;
        control_word[1] = kIndexErrorReset;
        control_word[2] = kIndexEnable;
        control_word[3] = kIndexGetStatusWord;
        ResetJointError();
    }

    virtual ~DdsInterface() {
        delete[] pos_offset_;
        delete[] joint_dir_;
        delete[] data_updated_;
        delete[] joint_config_;
    }

    virtual void Start() {
    }

    virtual void Stop() {
    }

    virtual double GetInterfaceTimeStamp() {
        return ri_ts_;
    }

    virtual VecXf GetJointPosition() {
        return joint_pos_;
    }

    virtual VecXf GetJointVelocity() {
        return joint_vel_;
    }

    virtual VecXf GetJointTorque() {
        return joint_tau_;
    }

    virtual Vec3f GetImuRpy() {
        return rpy_;
    }

    virtual Vec3f GetImuAcc() {
        return acc_;
    }

    virtual Vec3f GetImuOmega() {
        return omega_body_;
    }

    virtual Vec3f GetBasePositionWorld() {
        return base_pos_w_;
    }

    virtual VecXf GetContactForce() {
        return VecXf::Zero(4);
    }

    virtual void ResetJointError() {
        auto msg = drdds::msg::JointsDataCmd();

        for (int j = 0; j < 4; ++j) {
            for (int i = 0; i < dof_num_; ++i) {
                msg.data.joints_data[i].position = 0;
                msg.data.joints_data[i].velocity = 0;
                msg.data.joints_data[i].torque = 0;
                msg.data.joints_data[i].kp = 0;
                msg.data.joints_data[i].kd = 0;
                msg.data.joints_data[i].control_word = control_word[j];
            }
            std::cout << "ResetJointError publish!! \n" << std::endl;
            joint_cmd_pub_->publish(msg);
            usleep(10 * 1000);
        }
    }

    virtual void SetJointCommand(Eigen::Matrix<float, Eigen::Dynamic, 5> input) {
        joint_cmd_ = input;
        auto msg = drdds::msg::JointsDataCmd();

        for (int i = 0; i < dof_num_; ++i) {
            msg.data.joints_data[i].position =
                    (input(i, 1) - joint_config_[i].offset) * joint_config_[i].dir;
            msg.data.joints_data[i].velocity = input(i, 3) * joint_dir_[i];
            msg.data.joints_data[i].torque = input(i, 4) * joint_dir_[i];
            msg.data.joints_data[i].kp = input(i, 0);
            msg.data.joints_data[i].kd = input(i, 2);
            msg.data.joints_data[i].control_word = kIndexMotorControl;
        }
        std::cout << "SetJointCommand publish!! \n" << std::endl;
        joint_cmd_pub_->publish(msg);
    }

    virtual VecXf GetMotorTemperture() {
        return motor_temperture_;
    }

    virtual VecXf GetDriverTemperture() {
        return driver_temperture_;
    }

    virtual double GetImuTimestamp() {
        return imu_ts_;
    }

    virtual std::vector<uint16_t> GetBatteryData() {
        return battery_data_;
    }

    virtual std::vector<uint16_t> GetDriverStatusWord() {
        return driver_status_;
    }

    virtual std::vector<uint16_t> GetJointDataID() {
        return joint_data_id_;
    }

    virtual void RefreshRobotData() {
    }

    virtual void Handler(const drdds::msg::JointsData::SharedPtr msg) {
        ++run_cnt_;
        for (int i = 0; i < dof_num_; ++i) {
            data_updated_[i] = true;
        }
        for (int i = 0; i < dof_num_; ++i) {
            joint_pos_(i) = msg->data.joints_data[i].position * joint_dir_[i] + pos_offset_[i] / 180. * M_PI;
            joint_vel_(i) = msg->data.joints_data[i].velocity * joint_dir_[i];
            joint_tau_(i) = msg->data.joints_data[i].torque * joint_dir_[i];
            motor_temperture_(i) = float(msg->data.joints_data[i].motion_temp);
            driver_temperture_(i) = float(msg->data.joints_data[i].driver_temp);
            driver_status_[i] = msg->data.joints_data[i].status_word;
            joint_data_id_[i] = uint16_t(run_cnt_);
        }
        ri_ts_ = rclcpp::Time(msg->header.stamp).seconds();
    }

    void HandlerIMU(const drdds::msg::ImuData::SharedPtr msg) {
        if (!sensor_imu_received_) {
            const float yaw = slam_odom_received_ ? rpy_(2) : Deg2Rad(msg->data.yaw);
            rpy_ = Vec3f(Deg2Rad(msg->data.roll), Deg2Rad(msg->data.pitch), yaw);
        }
        if (!sensor_imu_received_) {
            acc_ << msg->data.acc_x, msg->data.acc_y, msg->data.acc_z;
            omega_body_ << msg->data.omega_x, msg->data.omega_y, msg->data.omega_z;
        }
        imu_ts_ = rclcpp::Time(msg->header.stamp).seconds();
        ++dds_imu_debug_cnt_;
        // if (dds_imu_debug_cnt_ <= 5 || dds_imu_debug_cnt_ % 200 == 0) {
        //     std::cout << "[DdsInterface] /IMU_DATA"
        //               << " cnt=" << dds_imu_debug_cnt_
        //               << " used_for_roll_pitch=" << (!sensor_imu_received_ ? "Y" : "N")
        //               << " used_for_yaw=" << ((!sensor_imu_received_ && !slam_odom_received_) ? "Y" : "N")
        //               << " sensor_imu_received=" << (sensor_imu_received_ ? "Y" : "N")
        //               << " slam_odom_received=" << (slam_odom_received_ ? "Y" : "N")
        //               << " rpy=(" << rpy_.transpose() << ")"
        //               << " acc=(" << acc_.transpose() << ")"
        //               << " omega=(" << omega_body_.transpose() << ")"
        //               << std::endl;
        // }
    }

    void HandlerSensorImu(const sensor_msgs::msg::Imu::SharedPtr msg) {
        sensor_imu_received_ = true;
        const auto& q = msg->orientation;
        Vec3f imu_rpy = QuaternionToRpy(q.x, q.y, q.z, q.w);
        if (slam_odom_received_) {
            imu_rpy(2) = rpy_(2);
        }
        rpy_ = imu_rpy;
        acc_ << msg->linear_acceleration.x,
                msg->linear_acceleration.y,
                msg->linear_acceleration.z;
        omega_body_ << msg->angular_velocity.x,
                       msg->angular_velocity.y,
                       msg->angular_velocity.z;
        imu_ts_ = rclcpp::Time(msg->header.stamp).seconds();
        ++sensor_imu_debug_cnt_;
        // if (sensor_imu_debug_cnt_ <= 5 || sensor_imu_debug_cnt_ % 200 == 0) {
        //     std::cout << "[DdsInterface] /IMU"
        //               << " cnt=" << sensor_imu_debug_cnt_
        //               << " used_for_roll_pitch=Y"
        //               << " used_for_yaw=" << (slam_odom_received_ ? "N" : "Y")
        //               << " slam_odom_received=" << (slam_odom_received_ ? "Y" : "N")
        //               << " rpy=(" << rpy_.transpose() << ")"
        //               << " acc=(" << acc_.transpose() << ")"
        //               << " omega=(" << omega_body_.transpose() << ")"
        //               << std::endl;
        // }
    }

    void HandlerHealth(const drdds::msg::BatteryData::SharedPtr msg) {
        battery_data_[0] = uint8_t(msg->data[0].battery_level);
        remain_bat = battery_data_[0] / 100;
        battery_info_[0].battery_level = msg->data[0].battery_level;
    }

    void HandlerSlamOdom(const nav_msgs::msg::Odometry::SharedPtr msg) {
        base_pos_w_ << msg->pose.pose.position.x,
                       msg->pose.pose.position.y,
                       base_height_ref_;
        Vec3f odom_rpy = rpy_;
        const auto& q = msg->pose.pose.orientation;
        odom_rpy = QuaternionToRpy(q.x, q.y, q.z, q.w);
        rpy_(2) = odom_rpy(2);
        slam_odom_received_ = true;
        ++slam_odom_debug_cnt_;
        if (slam_odom_debug_cnt_ <= 5 || slam_odom_debug_cnt_ % 100 == 0) {
            std::cout << "[DdsInterface] /SLAM_ODOM"
                      << " cnt=" << slam_odom_debug_cnt_
                      << " pos=(" << base_pos_w_.transpose() << ")"
                      << " raw_z=" << msg->pose.pose.position.z
                      << " z_ref=" << base_height_ref_
                      << " odom_rpy=(" << odom_rpy.transpose() << ")"
                      << " active_rpy=(" << rpy_.transpose() << ")"
                      << " yaw_source=/SLAM_ODOM"
                      << std::endl;
        }
    }

    void HandlerBasePose(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
        if (msg->data.size() < 3) {
            return;
        }
        base_pos_w_ << msg->data[0], msg->data[1], msg->data[2];
        ++base_pose_debug_cnt_;
        // if (base_pose_debug_cnt_ <= 5 || base_pose_debug_cnt_ % 100 == 0) {
        //     std::cout << "[DdsInterface] /BASE_POSE2D"
        //               << " cnt=" << base_pose_debug_cnt_
        //               << " pos=(" << base_pos_w_.transpose() << ")"
        //               << " active_rpy=(" << rpy_.transpose() << ")";
        //     if (msg->data.size() >= 6) {
        //         std::cout << " msg_rpy=(" << msg->data[3] << ","
        //                   << msg->data[4] << "," << msg->data[5] << ")"
        //                   << " msg_rpy_used=N";
        //     }
        //     std::cout << std::endl;
        // }
    }

    void PublishWaypointPath(const std::vector<float>& waypoint_data) override {
        if (waypoint_data.empty()) return;
        auto msg = std_msgs::msg::Float32MultiArray();
        msg.data = waypoint_data;
        waypoint_path_pub_->publish(msg);
        PublishWaypointPathVisualization(waypoint_data);
    }

    void PublishWaypointPathVisualization(const std::vector<float>& waypoint_data) {
        if (waypoint_data.size() < 2) return;
        const int wp_idx = static_cast<int>(waypoint_data[0]);
        const int n = static_cast<int>(waypoint_data[1]);
        if (n <= 0 || waypoint_data.size() < 2 + static_cast<size_t>(3 * n)) return;

        auto stamp = node_->now();
        auto path = nav_msgs::msg::Path();
        path.header.stamp = stamp;
        path.header.frame_id = "map";
        path.poses.reserve(static_cast<size_t>(n));
        auto poses = geometry_msgs::msg::PoseArray();
        poses.header = path.header;
        poses.poses.reserve(static_cast<size_t>(n));

        for (int i = 0; i < n; ++i) {
            geometry_msgs::msg::PoseStamped pose;
            pose.header = path.header;
            pose.pose.position.x = waypoint_data[2 + 2 * i];
            pose.pose.position.y = waypoint_data[2 + 2 * i + 1];
            pose.pose.position.z = 0.05;
            pose.pose.orientation.w = 1.0;
            poses.poses.push_back(pose.pose);
            path.poses.push_back(pose);
        }
        waypoint_path_viz_pub_->publish(path);
        waypoint_pose_array_pub_->publish(poses);

        auto markers = visualization_msgs::msg::MarkerArray();
        auto line = visualization_msgs::msg::Marker();
        line.header = path.header;
        line.ns = "waypoint_path";
        line.id = 0;
        line.type = visualization_msgs::msg::Marker::LINE_STRIP;
        line.action = visualization_msgs::msg::Marker::ADD;
        line.pose.orientation.w = 1.0;
        line.scale.x = 0.04;
        line.color.g = 0.85;
        line.color.b = 0.25;
        line.color.a = 0.9;

        auto points = visualization_msgs::msg::Marker();
        points.header = path.header;
        points.ns = "waypoint_path";
        points.id = 1;
        points.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        points.action = visualization_msgs::msg::Marker::ADD;
        points.pose.orientation.w = 1.0;
        points.scale.x = 0.12;
        points.scale.y = 0.12;
        points.scale.z = 0.12;
        points.color.r = 0.1;
        points.color.g = 0.45;
        points.color.b = 1.0;
        points.color.a = 0.85;

        for (const auto& pose : path.poses) {
            line.points.push_back(pose.pose.position);
            points.points.push_back(pose.pose.position);
        }

        auto current = visualization_msgs::msg::Marker();
        current.header = path.header;
        current.ns = "waypoint_path";
        current.id = 2;
        current.type = visualization_msgs::msg::Marker::SPHERE;
        current.action = visualization_msgs::msg::Marker::ADD;
        current.pose.orientation.w = 1.0;
        current.scale.x = 0.25;
        current.scale.y = 0.25;
        current.scale.z = 0.25;
        current.color.r = 1.0;
        current.color.g = 0.15;
        current.color.b = 0.05;
        current.color.a = 1.0;
        const int clamped_idx = std::max(0, std::min(wp_idx, n - 1));
        current.pose.position = path.poses[static_cast<size_t>(clamped_idx)].pose.position;

        markers.markers.push_back(line);
        markers.markers.push_back(points);
        markers.markers.push_back(current);
        waypoint_marker_pub_->publish(markers);
    }
};
