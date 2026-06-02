/**
 * @file m20_policy_runner.hpp
 * @brief m20_policy_runner
 * @author Bo (Percy) Peng
 * @version 1.0
 * @date 2025-11-07
 * 
 * @copyright Copyright (c) 2025  DeepRobotics
 * 
 */

#pragma once
#define PI 3.14159265358979323846

#include "policy_runner_base.hpp"
#include <ctime>
#include <cmath>
#include <utility>
#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstdlib>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>

class M20PolicyRunner : public PolicyRunnerBase {
private:
    VecXf kp_, kd_;
    VecXf dof_default_eigen_policy, dof_default_eigen_robot;
    Vec3f max_cmd_vel_, gravity_direction = Vec3f(0., 0., -1.);
    VecXf dof_pos_default_;
    timespec system_time;

    const int motor_num = 16;
    static constexpr int observation_dim_ = 79;
    const int action_dim = 16;
    float agent_timestep = 0.02;
    float current_time;
    bool is_fallen = true;

    VecXf joint_pos_rl = VecXf(action_dim);// in rl squenece
    VecXf joint_vel_rl = VecXf(action_dim);
    
    const std::string policy_path_;

    float omega_scale_ = 0.25;
    float dof_vel_scale_ = 0.05;
    VecXf imu_w_eigen, base_acc_eigen, motor_p_eigen, motor_v_eigen,
          current_action_eigen, last_action_eigen, current_observation_, projected_gravity,
          tmp_action_eigen;

    RobotAction robot_action;
    std::vector<std::string> robot_order = {
        "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint", "fl_wheel_joint",
        "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint", "fr_wheel_joint",
        "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint", "hl_wheel_joint",
        "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint", "hr_wheel_joint"};


    std::vector<std::string> policy_order = {
        "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
        "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
        "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
        "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
        "fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint",
    };


    std::vector<float> action_scale_robot = {0.125, 0.25, 0.25, 5,
                                             0.125, 0.25, 0.25, 5,
                                             0.125, 0.25, 0.25, 5,
                                             0.125, 0.25, 0.25, 5};


    Ort::SessionOptions session_options_;
    Ort::Session session_{nullptr};
    
    Ort::Env env_;
    std::vector<int> robot2policy_idx, policy2robot_idx;

    const char* input_names_[1] = {"obs"}; // must keep the same as model export
    const char* output_names_[1] = {"actions"};
    VecXf command;
    Ort::MemoryInfo memory_info{nullptr};
    const std::array<int64_t, 2> input_observationShape_ = {1, observation_dim_};
    
    float time_step = 0.;
    int stop_count = 1000;
    int waypoint_idx_ = 1;
    std::vector<Eigen::Vector2f> waypoints_xy_;
    std::vector<float> waypoints_vE_;
    Eigen::Vector2f waypoint_origin_xy_ = Eigen::Vector2f::Zero();
    float waypoint_origin_yaw_ = 0.0f;
    bool waypoint_origin_initialized_ = false;
    Vec4f last_waypoint_cmd_ = Vec4f::Zero();
    float waypoint_reach_threshold_ = 0.05f;
    static constexpr float raw_action_limit_ = 10.0f;
    static constexpr float wheel_vel_limit_ = 50.0f;

    struct WaypointPathConfig {
        int num_waypoints = 100;
        float r_min = 0.2f;
        float r_max = 0.2f;
        float theta_min = -0.05f;
        float theta_max = 0.05f;
        float point_move_vE_min = 0.5f;
        float point_move_vE_max = 2.5f;
        int point_move_vE_cycle_min_points = 50;
        int point_move_vE_cycle_max_points = 100;
        int obs_num_points = 5;
        float reach_threshold = 0.05f;
        float target_z = 0.45f;
        int seed = 42;
    };
    WaypointPathConfig waypoint_cfg_;
    float ve_cycle_t_ = 0.0f;
    int ve_cycle_period_pts_ = 60;

    static std::filesystem::path ResolveWaypointCfgPath() {
        if (const char* env_path = std::getenv("M20_WAYPOINT_CFG")) {
            if (std::filesystem::exists(env_path)) {
                return std::filesystem::path(env_path);
            }
        }
        const std::filesystem::path this_file(__FILE__);
        const auto pkg_root = this_file.parent_path().parent_path();
        return pkg_root / "config" / "waypoint_path.cfg";
    }

    bool LoadWaypointPathConfig() {
        const auto cfg_path = ResolveWaypointCfgPath();
        std::ifstream ifs(cfg_path);
        if (!ifs.is_open()) {
            std::cerr << "[M20PolicyRunner] Failed to open waypoint cfg: "
                      << cfg_path << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(ifs, line)) {
            if (line.empty() || line[0] == '#' || line.find('=') == std::string::npos) {
                continue;
            }
            const auto pos = line.find('=');
            std::string key = line.substr(0, pos);
            std::string val = line.substr(pos + 1);
            try {
                if (key == "num_waypoints") waypoint_cfg_.num_waypoints = std::stoi(val);
                else if (key == "r_min") waypoint_cfg_.r_min = std::stof(val);
                else if (key == "r_max") waypoint_cfg_.r_max = std::stof(val);
                else if (key == "theta_min") waypoint_cfg_.theta_min = std::stof(val);
                else if (key == "theta_max") waypoint_cfg_.theta_max = std::stof(val);
                else if (key == "point_move_vE_min") waypoint_cfg_.point_move_vE_min = std::stof(val);
                else if (key == "point_move_vE_max") waypoint_cfg_.point_move_vE_max = std::stof(val);
                else if (key == "point_move_vE_cycle_min_points") waypoint_cfg_.point_move_vE_cycle_min_points = std::stoi(val);
                else if (key == "point_move_vE_cycle_max_points") waypoint_cfg_.point_move_vE_cycle_max_points = std::stoi(val);
                else if (key == "obs_num_points") waypoint_cfg_.obs_num_points = std::stoi(val);
                else if (key == "reach_threshold") waypoint_cfg_.reach_threshold = std::stof(val);
                else if (key == "target_z") waypoint_cfg_.target_z = std::stof(val);
                else if (key == "seed") waypoint_cfg_.seed = std::stoi(val);
            } catch (const std::exception&) {
                std::cerr << "[M20PolicyRunner] Invalid cfg item: " << line << std::endl;
            }
        }

        if (waypoint_cfg_.num_waypoints < 2 || waypoint_cfg_.r_min <= 0.0f || waypoint_cfg_.r_max < waypoint_cfg_.r_min) {
            std::cerr << "[M20PolicyRunner] Invalid waypoint cfg values in: "
                      << cfg_path << std::endl;
            return false;
        }
        waypoint_reach_threshold_ = waypoint_cfg_.reach_threshold;
        std::cout << "[M20PolicyRunner] Waypoint cfg loaded from " << cfg_path
                  << " (num_waypoints=" << waypoint_cfg_.num_waypoints
                  << ", r=[" << waypoint_cfg_.r_min << "," << waypoint_cfg_.r_max << "]"
                  << ", theta=[" << waypoint_cfg_.theta_min << "," << waypoint_cfg_.theta_max << "]"
                  << ", obs_num_points=" << waypoint_cfg_.obs_num_points
                  << ", seed=" << waypoint_cfg_.seed << ")" << std::endl;
        return true;
    }

    static uint32_t NextLcg(uint32_t& state) {
        state = state * 1664525u + 1013904223u;
        return state;
    }
    static float Rand01(uint32_t& state) {
        return static_cast<float>(NextLcg(state)) / 4294967295.0f;
    }
    int RandInt(uint32_t& state, int lo, int hi) {
        if (hi < lo) hi = lo;
        const float u = Rand01(state);
        return lo + static_cast<int>(u * static_cast<float>(hi - lo + 1));
    }
    float SampleCyclicVEMag(uint32_t& state) {
        const float v_min = waypoint_cfg_.point_move_vE_min;
        const float v_max = waypoint_cfg_.point_move_vE_max;
        if (v_max <= v_min) return v_min;
        const float tri = (ve_cycle_t_ < 1.0f) ? ve_cycle_t_ : (2.0f - ve_cycle_t_);
        const float speed = v_min + tri * (v_max - v_min);
        ve_cycle_t_ += 2.0f / std::max(1, ve_cycle_period_pts_);
        if (ve_cycle_t_ >= 2.0f) {
            ve_cycle_t_ -= 2.0f;
            ve_cycle_period_pts_ = RandInt(state,
                waypoint_cfg_.point_move_vE_cycle_min_points,
                waypoint_cfg_.point_move_vE_cycle_max_points);
        }
        return speed;
    }

    void GenerateWaypoints(float initial_heading_w = 0.0f) {
        waypoints_xy_.clear();
        waypoints_vE_.clear();
        waypoints_xy_.reserve(static_cast<size_t>(waypoint_cfg_.num_waypoints));
        waypoints_vE_.assign(static_cast<size_t>(waypoint_cfg_.num_waypoints), 0.0f);
        waypoints_xy_.emplace_back(0.0f, 0.0f);

        uint32_t state = (waypoint_cfg_.seed >= 0)
            ? static_cast<uint32_t>(waypoint_cfg_.seed)
            : static_cast<uint32_t>(std::time(nullptr));
        ve_cycle_t_ = 2.0f * Rand01(state);
        ve_cycle_period_pts_ = RandInt(state,
            waypoint_cfg_.point_move_vE_cycle_min_points,
            waypoint_cfg_.point_move_vE_cycle_max_points);
        // Force first segment to align with chosen heading (robot yaw at RL start).
        float heading = initial_heading_w;
        for (int i = 1; i < waypoint_cfg_.num_waypoints; ++i) {
            const float r = waypoint_cfg_.r_min + Rand01(state) * (waypoint_cfg_.r_max - waypoint_cfg_.r_min);
            if (i > 1) {
                const float dtheta = (waypoint_cfg_.theta_min +
                    Rand01(state) * (waypoint_cfg_.theta_max - waypoint_cfg_.theta_min))
                    * 2.0f * static_cast<float>(M_PI);
                heading += dtheta;
            }
            const Eigen::Vector2f prev = waypoints_xy_.back();
            const Eigen::Vector2f next = prev + Eigen::Vector2f(r * std::cos(heading), r * std::sin(heading));
            waypoints_xy_.emplace_back(next);
            waypoints_vE_[static_cast<size_t>(i - 1)] = SampleCyclicVEMag(state);
        }
    }

public:
    M20PolicyRunner(const std::string &policy_name, const std::string &policy_path) :
            PolicyRunnerBase(policy_name), policy_path_(policy_path),env_(ORT_LOGGING_LEVEL_WARNING, "M20PolicyRunner"),
            session_options_{},
            session_{nullptr},
            memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {

        dof_default_eigen_policy.setZero(action_dim);
        dof_default_eigen_robot.setZero(action_dim);
        dof_default_eigen_policy << 0.0, -0.3,  0.6, 
                                    0.0, -0.3,  0.6,  
                                    0.0,  0.3, -0.6,  
                                    0.0,  0.3, -0.6, 
                                    0.0, 0.0, 0.0, 0.0;
        dof_default_eigen_robot << 0.0, -0.3,  0.6, 0.0,
                                   0.0, -0.3,  0.6, 0.0,
                                   0.0,  0.3, -0.6, 0.0,
                                   0.0,  0.3, -0.6, 0.0;
        SetDecimation(4);
        session_options_.SetIntraOpNumThreads(4);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        if (access(policy_path_.c_str(), F_OK) != 0) {
            std::cerr << "Model file not found: " << policy_path_ << std::endl;
            throw std::runtime_error("Model file missing");
            }

        session_ = Ort::Session(env_, policy_path_.c_str(), session_options_);
        std::cout << "[M20PolicyRunner] ONNX obs dim = " << observation_dim_ << std::endl;
        kp_ = Vec4f(80, 80, 80, 0.).replicate(4, 1);
        kd_ = Vec4f(2, 2, 2, 0.6).replicate(4, 1);
        
        robot2policy_idx = generate_permutation(robot_order, policy_order);
        policy2robot_idx = generate_permutation(policy_order, robot_order);
        // for (int i = 0; i < action_dim; ++i){
        //     std::cout << "robot2policy_idx[" << i << "]: " << robot2policy_idx[i] << std::endl;
        //     std::cout << "policy2robot_idx[" << i << "]: " << policy2robot_idx[i] << std::endl;
        // }

        robot_action.kp = kp_;
        robot_action.kd = kd_;
        robot_action.tau_ff = VecXf::Zero(motor_num);
        robot_action.goal_joint_pos = VecXf::Zero(motor_num);
        robot_action.goal_joint_vel = VecXf::Zero(motor_num);


        current_observation_.setZero(observation_dim_);
        last_action_eigen.setZero(action_dim);
        last_waypoint_cmd_.setZero();
        tmp_action_eigen.setZero(action_dim);
        current_action_eigen.setZero(action_dim);

        memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        if (!LoadWaypointPathConfig()) {
            throw std::runtime_error("Failed to load waypoint path config");
        }
        GenerateWaypoints();
    }

    ~M20PolicyRunner() override = default;

    std::vector<int> generate_permutation(
        const std::vector<std::string>& from, 
        const std::vector<std::string>& to, 
        int default_index = 0) 
    {
        std::unordered_map<std::string, int> idx_map;
        for (int i = 0; i < from.size(); ++i) {
            idx_map[from[i]] = i;
        }

        std::vector<int> perm;
        for (const auto& name : to) {
            auto it = idx_map.find(name);
            if (it != idx_map.end()) {
                perm.push_back(it->second);
            } else {
                perm.push_back(default_index);  // 如果找不到，就填默认值
            }
        }

        return perm;
    }

    void DisplayPolicyInfo(){}

    void OnEnter() override {
        OnEnter(UserCommand());
    }

    void OnEnter(const UserCommand& uc) {
        run_cnt_ = 0;
        if (uc.reserved_scale > 0.5f && !waypoints_xy_.empty()) {
            waypoint_idx_ = (waypoint_idx_ + 1) % static_cast<int>(waypoints_xy_.size());
            std::cout << "[M20PolicyRunner] Resume waypoint tracking from next point #"
                      << waypoint_idx_ << std::endl;
        } else {
            waypoint_idx_ = 1;
            std::cout << "[M20PolicyRunner] Start waypoint tracking from point #1" << std::endl;
        }
        waypoint_origin_initialized_ = false;
        waypoint_origin_xy_.setZero();
        waypoint_origin_yaw_ = 0.0f;
        GenerateWaypoints();
        cmd_vel_input_.setZero();
        last_action_eigen.setZero(action_dim);
        tmp_action_eigen.setZero(action_dim);
        motor_p_eigen.setZero(12);
        motor_v_eigen.setZero(motor_num);
    }

    static float WrapToPi(float angle) {
        while (angle > static_cast<float>(M_PI)) angle -= 2.0f * static_cast<float>(M_PI);
        while (angle < -static_cast<float>(M_PI)) angle += 2.0f * static_cast<float>(M_PI);
        return angle;
    }

    Eigen::Vector2f LocalWaypointToWorld(const Eigen::Vector2f& local_wp) const {
        // Training command uses env-origin translation only (no yaw0 rotation).
        return waypoint_origin_xy_ + local_wp;
    }

    Vec4f BuildWaypointCommand(const RobotBasicState &ro) {
        if (waypoints_xy_.empty()) {
            return Vec4f::Zero();
        }

        Eigen::Vector2f base_xy(ro.base_pos_w(0), ro.base_pos_w(1));
        if (!waypoint_origin_initialized_) {
            waypoint_origin_xy_ = base_xy;
            waypoint_origin_yaw_ = ro.base_rpy(2);
            waypoint_origin_initialized_ = true;
            GenerateWaypoints(waypoint_origin_yaw_);
            waypoint_idx_ = 1;
            std::cout << "[M20PolicyRunner] Waypoint origin set to ("
                      << waypoint_origin_xy_(0) << ", " << waypoint_origin_xy_(1)
                      << "), yaw0=" << waypoint_origin_yaw_ << std::endl;
        }

        const Eigen::Vector2f &local_wp = waypoints_xy_[waypoint_idx_];
        Eigen::Vector2f target_xy = LocalWaypointToWorld(local_wp);
        Eigen::Vector2f delta_w = target_xy - base_xy;
        float dist = delta_w.norm();
        if (dist < waypoint_reach_threshold_) {
            waypoint_idx_ = std::min(waypoint_idx_ + 1, static_cast<int>(waypoints_xy_.size()) - 1);
            const Eigen::Vector2f &local_wp_next = waypoints_xy_[waypoint_idx_];
            target_xy = LocalWaypointToWorld(local_wp_next);
            delta_w = target_xy - base_xy;
        }

        const float yaw = ro.base_rpy(2);
        const float cy = std::cos(yaw);
        const float sy = std::sin(yaw);
        const float dx_b = cy * delta_w(0) + sy * delta_w(1);
        const float dy_b = -sy * delta_w(0) + cy * delta_w(1);
        const float dz_b = waypoint_cfg_.target_z - ro.base_pos_w(2);
        const float desired_heading_w = std::atan2(delta_w(1), delta_w(0));
        const float dheading = WrapToPi(desired_heading_w - yaw);

        return Vec4f(dx_b, dy_b, dz_b, dheading);
    }

    VecXf BuildFutureWaypointObservation(const RobotBasicState &ro) {
        const int k = std::max(1, waypoint_cfg_.obs_num_points);
        VecXf out = VecXf::Zero(k * 5);
        if (waypoints_xy_.empty()) return out;

        Eigen::Vector2f base_xy(ro.base_pos_w(0), ro.base_pos_w(1));
        const float yaw = ro.base_rpy(2);
        const float cy = std::cos(yaw);
        const float sy = std::sin(yaw);

        for (int i = 0; i < k; ++i) {
            const int idx = std::min(waypoint_idx_ + i, static_cast<int>(waypoints_xy_.size()) - 1);
            const Eigen::Vector2f target_xy = LocalWaypointToWorld(waypoints_xy_[idx]);
            const Eigen::Vector2f delta_w = target_xy - base_xy;
            const float dx_b = cy * delta_w(0) + sy * delta_w(1);
            const float dy_b = -sy * delta_w(0) + cy * delta_w(1);
            const float dz_b = waypoint_cfg_.target_z - ro.base_pos_w(2);
            const float desired_heading_w = std::atan2(delta_w(1), delta_w(0));
            const float dheading = WrapToPi(desired_heading_w - yaw);
            const float point_move_vE = (idx >= 0 && idx < static_cast<int>(waypoints_vE_.size())) ? waypoints_vE_[idx] : 0.0f;
            const int off = i * 5;
            out(off + 0) = dx_b;
            out(off + 1) = dy_b;
            out(off + 2) = dz_b;
            out(off + 3) = dheading;
            out(off + 4) = point_move_vE;
        }
        return out;
    }

    VecXf Onnx_infer(VecXf current_observation){
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            current_observation.data(),
            current_observation.size(),
            input_observationShape_.data(),
            input_observationShape_.size()
        );

        std::vector<Ort::Value> inputs;
        inputs.emplace_back(std::move(input_tensor));  // 避免拷贝构造
        
        auto outputs = session_.Run(
            Ort::RunOptions{nullptr},
            input_names_,
            inputs.data(),
            1,
            output_names_,
            1
        );

        float* action_data = outputs[0].GetTensorMutableData<float>();
        Eigen::Map<Eigen::VectorXf> action_map(action_data, action_dim);
        return VecXf(action_map);  // 返回一个Eigen向量的副本
    }

    VecXf BuildObservation(const Vec3f& base_omgea,
                           const Vec3f& projected_gravity,
                           const RobotBasicState& ro,
                           const VecXf& joint_pos_rl,
                           const VecXf& joint_vel_rl,
                           const VecXf& last_action_eigen) {
        VecXf obs(observation_dim_);
        last_waypoint_cmd_ = BuildWaypointCommand(ro);
        VecXf waypoint_obs = BuildFutureWaypointObservation(ro);
        obs << base_omgea,
               projected_gravity,
               waypoint_obs,
               joint_pos_rl,
               joint_vel_rl,
               last_action_eigen;
        return obs;
    }

    RobotAction getRobotAction(const RobotBasicState &ro, const UserCommand &) {

        Vec3f base_omgea = ro.base_omega * omega_scale_;
        Vec3f projected_gravity = ro.base_rot_mat.inverse() * gravity_direction;
        for (int i = 0; i < action_dim; ++i){
            joint_pos_rl(i) = ro.joint_pos(robot2policy_idx[i]);
            joint_vel_rl(i) = ro.joint_vel(robot2policy_idx[i]) * dof_vel_scale_;
        }
        joint_pos_rl.segment(12, 4).setZero();

        joint_pos_rl -= dof_default_eigen_policy;
        current_observation_ = BuildObservation(
            base_omgea, projected_gravity, ro, joint_pos_rl, joint_vel_rl, last_action_eigen);
        current_action_eigen = Onnx_infer(current_observation_);
        VecXf raw_action_eigen = current_action_eigen;
        current_action_eigen = current_action_eigen
            .cwiseMax(-raw_action_limit_)
            .cwiseMin(raw_action_limit_);
        last_action_eigen = current_action_eigen;

        
        for (int i = 0; i < action_dim; ++i){
            tmp_action_eigen(i) = current_action_eigen(policy2robot_idx[i]);
            tmp_action_eigen(i) *= action_scale_robot[i];
        }
        tmp_action_eigen += dof_default_eigen_robot;
        
        for (int i = 0; i < 4; ++i){
            robot_action.goal_joint_pos.segment(i*4, 3) = tmp_action_eigen.segment(i*4, 3);
            robot_action.goal_joint_vel(i*4+3) = LimitNumber(tmp_action_eigen(i*4+3), -wheel_vel_limit_, wheel_vel_limit_);
        }

        // if (run_cnt_ < 20 || run_cnt_ % 50 == 0) {
        //     std::cout << "[M20PolicyRunner] debug"
        //               << " cnt=" << run_cnt_
        //               << " obs_dim=" << observation_dim_
        //               << " wp=" << waypoint_idx_
        //               << " base_xy=(" << ro.base_pos_w(0) << "," << ro.base_pos_w(1) << ")"
        //               << " cmd=(" << last_waypoint_cmd_.transpose() << ")"
        //               << " joint_rel_minmax=(" << joint_pos_rl.minCoeff() << "," << joint_pos_rl.maxCoeff() << ")"
        //               << " raw_action_minmax=(" << raw_action_eigen.minCoeff() << "," << raw_action_eigen.maxCoeff() << ")"
        //               << " clipped_action_minmax=(" << current_action_eigen.minCoeff() << "," << current_action_eigen.maxCoeff() << ")"
        //               << std::endl;
        // }

        
        ++run_cnt_;
        ++time_step;
        return robot_action;
    }

    void setDefaultJointPos(const VecXf& pos){
        dof_pos_default_.setZero(motor_num); 
        for(int i=0;i<motor_num;++i) {
            dof_pos_default_(i) = pos(i);
        }
    }

    double getCurrentTime() {
        clock_gettime(1, &system_time);
        return system_time.tv_sec + system_time.tv_nsec / 1e9;
    }
};
