"""
 * @file mujoco_simulation_ros2_waypoint12.py
 * @brief MuJoCo simulation with 12-waypoint tracking
 * @author DeepRobotics
 * @version 1.0
 * @date 2026-05-22
"""

import os
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from std_msgs.msg import Float32MultiArray
from drdds.msg import ImuData, JointsData, JointsDataCmd, MetaType, ImuDataValue, JointsDataValue, JointData


MODEL_NAME = "M20"
CURRENT_DIR = Path(__file__).resolve().parent
XML_PATH = CURRENT_DIR / ".." / ".." / ".." / "M20_description" / "m20_mjcf" / "mjcf" / "M20_stair.xml"
XML_PATH = str(XML_PATH.resolve())

USE_VIEWER = True
DT = 0.001
RENDER_INTERVAL = 50
CMD_TIMEOUT_SEC = 0.5

# autopilot
AUTO_TRACK_WAYPOINTS = True
MAX_VXY = 0.8
MAX_WZ = 1.2
Kp_VXY = 0.9
Kp_WZ = 1.5
FALLBACK_KP = 80.0
FALLBACK_KD = 2.0
MAX_JOINT_TORQUE_ABS = 120.0

def _resolve_waypoint_cfg_path() -> Path:
    env_path = os.getenv("M20_WAYPOINT_CFG")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.is_file():
            return p
    return (CURRENT_DIR / ".." / ".." / ".." / ".." /
            "M20_sdk_DF_deploy" / "config" / "waypoint_path.cfg").resolve()


def _load_waypoint_rect_cfg() -> dict:
    cfg_path = _resolve_waypoint_cfg_path()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Waypoint cfg not found: {cfg_path}")

    float_keys = {"r_min", "r_max", "theta_min", "theta_max",
                  "point_move_vE_min", "point_move_vE_max", "target_z",
                  "reach_threshold", "curve_scale", "point_spacing",
                  "curve_num_cycles", "ellipse_b_ratio", "spiral_growth",
                  "trochoid_d_ratio", "hypocycloid_R_ratio",
                  "hypocycloid_d_ratio"}
    int_keys = {"num_waypoints", "point_move_vE_cycle_min_points",
                "point_move_vE_cycle_max_points", "obs_num_points", "seed",
                "path_type", "rose_n", "rose_d"}
    cfg = {}
    with cfg_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key in float_keys:
                cfg[key] = float(value)
            elif key in int_keys:
                cfg[key] = int(value)

    required = {"path_type", "reach_threshold", "point_move_vE_min",
                "point_move_vE_max", "point_move_vE_cycle_min_points",
                "point_move_vE_cycle_max_points", "seed"}
    missing = sorted(required - cfg.keys())
    if missing:
        raise ValueError(f"Missing waypoint cfg keys: {missing}")
    if int(cfg["path_type"]) == 0:
        random_required = {"num_waypoints", "r_min", "r_max",
                           "theta_min", "theta_max"}
        missing = sorted(random_required - cfg.keys())
        if missing:
            raise ValueError(f"Missing random waypoint cfg keys: {missing}")
        if cfg["num_waypoints"] < 2 or cfg["r_min"] <= 0.0 or cfg["r_max"] < cfg["r_min"]:
            raise ValueError(f"Invalid random waypoint cfg: {cfg}")
    return cfg


WAYPOINT_CFG = _load_waypoint_rect_cfg()


def _equal_arc_length_sample(curve_fn, t_start, t_end, spacing, max_points=10000):
    n_dense = 20000
    ts = np.linspace(t_start, t_end, n_dense + 1, dtype=np.float64)
    dense = np.array([curve_fn(t) for t in ts], dtype=np.float32)

    diffs = np.diff(dense, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    arc = np.zeros(n_dense + 1, dtype=np.float32)
    arc[1:] = np.cumsum(seg_len)
    total_arc = float(arc[-1])
    if total_arc < 1e-6 or spacing < 1e-6:
        return dense[:1].copy()

    n_pts = min(max_points, int(total_arc / spacing) + 2)
    result = [dense[0].copy()]
    j = 0
    for k in range(1, n_pts):
        target = k * spacing
        if target >= total_arc:
            break
        while j < n_dense - 1 and arc[j + 1] < target:
            j += 1
        denom = arc[j + 1] - arc[j]
        frac = (target - arc[j]) / denom if denom > 1e-9 else 0.0
        result.append(dense[j] + frac * (dense[j + 1] - dense[j]))
    return np.array(result, dtype=np.float32)


def _generate_fixed_curve_waypoints(cfg: dict, initial_heading_w: float = 0.0):
    scale = float(cfg.get("curve_scale", 3.0))
    spacing = float(cfg.get("point_spacing", 0.2))
    ncycles = float(cfg.get("curve_num_cycles", 3.0))
    path_type = int(cfg.get("path_type", 0))
    k_pi = float(np.pi)
    t_start, t_end = 0.0, 2.0 * k_pi
    curve_fn = None

    if path_type == 1:
        curve_fn = lambda t: np.array([scale * np.sin(t),
                                       scale * 0.5 * np.sin(2.0 * t)], dtype=np.float32)
    elif path_type == 2:
        t_end = ncycles * 2.0 * k_pi
        curve_fn = lambda t, s=scale, p=k_pi: np.array(
            [s * t / (2.0 * p), s * 0.3 * np.sin(t)], dtype=np.float32)
    elif path_type == 3:
        t_end = ncycles * 2.0 * k_pi
        curve_fn = lambda t, s=scale: np.array(
            [s * np.sin(t), s * (1.0 - np.cos(t))], dtype=np.float32)
    elif path_type == 4:
        growth = float(cfg.get("spiral_growth", 0.3))
        t_end = ncycles * 2.0 * k_pi
        curve_fn = lambda t, g=growth, s=scale: np.array(
            [s * g * t * np.cos(t), s * g * t * np.sin(t)], dtype=np.float32)
    elif path_type == 5:
        b = scale * float(cfg.get("ellipse_b_ratio", 0.5))
        t_end = ncycles * 2.0 * k_pi
        curve_fn = lambda t, s=scale, b_=b: np.array(
            [s * np.sin(t), b_ * (1.0 - np.cos(t))], dtype=np.float32)
    elif path_type == 6:
        n = max(1, int(cfg.get("rose_n", 3)))
        d = max(1, int(cfg.get("rose_d", 1)))
        nd = float(n) / float(d)
        t_end = (2.0 * k_pi * float(d)) if (n * d) % 2 == 0 else (k_pi * float(d))
        curve_fn = lambda t, s=scale, nd_=nd: np.array(
            [s * np.cos(nd_ * t) * np.cos(t),
             s * np.cos(nd_ * t) * np.sin(t)], dtype=np.float32)
    elif path_type == 7:
        radius = scale * 0.5
        dist = radius * float(cfg.get("trochoid_d_ratio", 0.5))
        t_end = ncycles * 2.0 * k_pi
        curve_fn = lambda t, r=radius, d=dist: np.array(
            [r * t - d * np.sin(t), r - d * np.cos(t)], dtype=np.float32)
    elif path_type == 8:
        radius = scale
        k = max(2.0, float(cfg.get("hypocycloid_R_ratio", 4.0)))
        r_in = radius / k
        dist = r_in * float(cfg.get("hypocycloid_d_ratio", 1.0))
        ratio = (radius - r_in) / r_in
        curve_fn = lambda t, r=radius, ri=r_in, d=dist, ratio_=ratio: np.array(
            [(r - ri) * np.cos(t) + d * np.cos(ratio_ * t),
             (r - ri) * np.sin(t) - d * np.sin(ratio_ * t)], dtype=np.float32)
    else:
        return None, None

    raw = _equal_arc_length_sample(curve_fn, t_start, t_end, spacing)
    if len(raw) == 0:
        return None, None
    raw -= raw[0]

    ch, sh = np.cos(initial_heading_w), np.sin(initial_heading_w)
    rotated = np.empty_like(raw)
    rotated[:, 0] = ch * raw[:, 0] - sh * raw[:, 1]
    rotated[:, 1] = sh * raw[:, 0] + ch * raw[:, 1]

    vmag = np.zeros(len(rotated), dtype=np.float32)
    state = (int(cfg["seed"]) & 0xFFFFFFFF) if int(cfg["seed"]) >= 0 else (int(time.time()) & 0xFFFFFFFF)

    def next_rand01():
        nonlocal state
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        return float(state) / 4294967295.0

    ve_cycle_t = 2.0 * next_rand01()
    pmin = int(cfg["point_move_vE_cycle_min_points"])
    pmax = max(pmin, int(cfg["point_move_vE_cycle_max_points"]))
    period = pmin + int(next_rand01() * (pmax - pmin + 1))
    for i in range(len(rotated)):
        vmin = float(cfg["point_move_vE_min"])
        vmax = float(cfg["point_move_vE_max"])
        if vmax <= vmin:
            vmag[i] = vmin
        else:
            tri = ve_cycle_t if ve_cycle_t < 1.0 else 2.0 - ve_cycle_t
            vmag[i] = vmin + tri * (vmax - vmin)
            ve_cycle_t += 2.0 / max(1, period)
            if ve_cycle_t >= 2.0:
                ve_cycle_t -= 2.0
                period = pmin + int(next_rand01() * (pmax - pmin + 1))
    return rotated, vmag

JOINT_DIR = np.array([1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1], dtype=np.float32)
POS_OFFSET_DEG = np.array([-25, 229, 160, 0, 25, -131, -200, 0, -25, -229, -160, 0, 25, 131, 200, 0], dtype=np.float32)
POS_OFFSET_RAD = POS_OFFSET_DEG / 180.0 * np.pi

JOINT_INIT = {
    "M20": np.array([-0.438, -1.16, 2.76, 0,
                     0.438, -1.16, 2.76, 0,
                     -0.438, 1.16, -2.76, 0,
                     0.438, 1.16, -2.76, 0], dtype=np.float32),
}


class MuJoCoSimulationWaypointNode(Node):
    def __init__(self, model_key: str = MODEL_NAME, xml_path: str = XML_PATH):
        super().__init__('mujoco_simulation_waypoint12')

        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f"Cannot find MJCF: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = DT
        self.data = mujoco.MjData(self.model)

        self.actuator_ids = [a for a in range(self.model.nu)]
        self.dof_num = len(self.actuator_ids)
        assert self.dof_num == 16, "Expected 16 DOF for M20"

        self._set_initial_pose(model_key)

        self.kp_cmd = np.zeros((self.dof_num, 1), np.float32)
        self.kd_cmd = np.zeros((self.dof_num, 1), np.float32)
        self.pos_cmd = np.zeros((self.dof_num, 1), np.float32)
        self.vel_cmd = np.zeros((self.dof_num, 1), np.float32)
        self.tau_ff = np.zeros((self.dof_num, 1), np.float32)
        self.input_tq = np.zeros((self.dof_num, 1), np.float32)

        self.timestamp = 0.0
        self.last_cmd_time = -1.0

        self.wp_idx = 0
        self.waypoint_origin_xy = self.data.qpos[:2].copy().astype(np.float32)
        yaw0 = float(self.quaternion_to_euler(self.data.qpos[3:7].copy())[2])
        self.waypoints_xy, self.waypoints_vE = self._generate_world_waypoints(self.waypoint_origin_xy, yaw0)
        self.wp_total = self.waypoints_xy.shape[0]

        self.get_logger().info(f"[INFO] MuJoCo model loaded, dof={self.dof_num}, waypoints={self.wp_total}")
        self.get_logger().info(
            f"[WP] cfg={_resolve_waypoint_cfg_path()}, "
            f"origin=({self.waypoint_origin_xy[0]:.2f}, {self.waypoint_origin_xy[1]:.2f}), yaw0={yaw0:.3f}"
        )

        self.imu_pub = self.create_publisher(ImuData, '/IMU_DATA', 200)
        self.joints_pub = self.create_publisher(JointsData, '/JOINTS_DATA', 200)
        self.base_pose_pub = self.create_publisher(Float32MultiArray, '/BASE_POSE2D', 200)

        self.cmd_sub = self.create_subscription(JointsDataCmd, '/JOINTS_CMD', self._cmd_callback, 50)

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data) if USE_VIEWER else None
        if self.viewer:
            self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE

    def _set_initial_pose(self, key: str):
        qpos0 = self.data.qpos.copy()
        qpos0[7:7 + self.dof_num] = JOINT_INIT[key]
        qpos0[:3] = np.array([0.0, 0.0, 0.2])
        qpos0[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qpos[:] = qpos0
        mujoco.mj_forward(self.model, self.data)

    def _cmd_callback(self, msg: JointsDataCmd):
        if len(msg.data.joints_data) != 16:
            self.get_logger().warn("Received JointsDataCmd with incorrect number of joints")
            return

        pub_pos = np.zeros(self.dof_num, dtype=np.float32)
        pub_vel = np.zeros(self.dof_num, dtype=np.float32)
        for i in range(self.dof_num):
            joint_cmd = msg.data.joints_data[i]
            self.kp_cmd[i] = joint_cmd.kp
            self.kd_cmd[i] = joint_cmd.kd
            pub_pos[i] = joint_cmd.position
            pub_vel[i] = joint_cmd.velocity
            self.tau_ff[i] = joint_cmd.torque

        self.pos_cmd.flat = pub_pos * JOINT_DIR + POS_OFFSET_RAD
        self.vel_cmd.flat = pub_vel * JOINT_DIR
        self.last_cmd_time = self.timestamp

    def _apply_fallback_stand_cmd_if_needed(self):
        if self.last_cmd_time >= 0.0 and (self.timestamp - self.last_cmd_time) < CMD_TIMEOUT_SEC:
            return
        q = self.data.qpos[7:7 + self.dof_num]
        self.kp_cmd.fill(20.0)
        self.kd_cmd.fill(2.0)
        self.pos_cmd[:, 0] = q
        self.vel_cmd.fill(0.0)
        self.tau_ff.fill(0.0)

    @staticmethod
    def quaternion_to_euler(q):
        w, x, y, z = q
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)

        t2 = 2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch = np.arcsin(t2)

        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)

        return np.array([roll, pitch, yaw], dtype=np.float32)

    @staticmethod
    def wrap_to_pi(angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def _generate_world_waypoints(self, origin_xy: np.ndarray, yaw0: float):
        path_type = int(WAYPOINT_CFG.get("path_type", 0))
        if path_type > 0:
            local_xy, vmag = _generate_fixed_curve_waypoints(WAYPOINT_CFG, yaw0)
            if local_xy is not None and len(local_xy) > 0:
                self.get_logger().info(
                    f"[WP] fixed curve path_type={path_type}, "
                    f"n_pts={len(local_xy)}, spacing={WAYPOINT_CFG.get('point_spacing', 0.2):.2f}"
                )
                return (local_xy + origin_xy.reshape(1, 2)).astype(np.float32), vmag
            self.get_logger().warn("[WP] fixed curve generation failed, falling back to random")

        n = int(WAYPOINT_CFG["num_waypoints"])
        local = np.zeros((n, 2), dtype=np.float32)
        vmag = np.zeros((n,), dtype=np.float32)
        state = (int(WAYPOINT_CFG["seed"]) & 0xFFFFFFFF) if int(WAYPOINT_CFG["seed"]) >= 0 else (int(time.time()) & 0xFFFFFFFF)

        def next_rand01() -> float:
            nonlocal state
            state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
            return float(state) / 4294967295.0

        ve_cycle_t = 2.0 * next_rand01()
        pmin = int(WAYPOINT_CFG["point_move_vE_cycle_min_points"])
        pmax = max(pmin, int(WAYPOINT_CFG["point_move_vE_cycle_max_points"]))
        period = pmin + int(next_rand01() * (pmax - pmin + 1))
        heading = float(yaw0)

        def sample_ve() -> float:
            nonlocal ve_cycle_t, period
            vmin = float(WAYPOINT_CFG["point_move_vE_min"])
            vmax = float(WAYPOINT_CFG["point_move_vE_max"])
            if vmax <= vmin:
                return vmin
            tri = ve_cycle_t if ve_cycle_t < 1.0 else 2.0 - ve_cycle_t
            speed = vmin + tri * (vmax - vmin)
            ve_cycle_t += 2.0 / max(1, period)
            if ve_cycle_t >= 2.0:
                ve_cycle_t -= 2.0
                period = pmin + int(next_rand01() * (pmax - pmin + 1))
            return speed

        for i in range(1, n):
            r = float(WAYPOINT_CFG["r_min"]) + next_rand01() * (float(WAYPOINT_CFG["r_max"]) - float(WAYPOINT_CFG["r_min"]))
            if i > 1:
                dtheta = (float(WAYPOINT_CFG["theta_min"]) + next_rand01() * (float(WAYPOINT_CFG["theta_max"]) - float(WAYPOINT_CFG["theta_min"]))) * 2.0 * np.pi
                heading += dtheta
            local[i, 0] = local[i - 1, 0] + r * np.cos(heading)
            local[i, 1] = local[i - 1, 1] + r * np.sin(heading)
            vmag[i - 1] = sample_ve()
        return (local + origin_xy.reshape(1, 2)).astype(np.float32), vmag

    def _apply_joint_torque(self):
        q = self.data.qpos[7:7 + self.dof_num].reshape(-1, 1)
        dq = self.data.qvel[6:6 + self.dof_num].reshape(-1, 1)
        self.input_tq = self.kp_cmd * (self.pos_cmd - q) + self.kd_cmd * (self.vel_cmd - dq) + self.tau_ff
        # Guard against NaN/Inf and unrealistic spikes before writing control.
        self.input_tq = np.nan_to_num(self.input_tq, nan=0.0, posinf=0.0, neginf=0.0)
        self.input_tq = np.clip(self.input_tq, -MAX_JOINT_TORQUE_ABS, MAX_JOINT_TORQUE_ABS)
        self.data.ctrl[:] = self.input_tq.flatten()

    def _autopilot_track_waypoint(self, step: int):
        if not AUTO_TRACK_WAYPOINTS:
            return

        base_pos = self.data.qpos[:2].copy()
        q_world = self.data.qpos[3:7].copy()  # wxyz
        yaw = float(self.quaternion_to_euler(q_world)[2])

        target = self.waypoints_xy[self.wp_idx]
        delta = target - base_pos
        dist = float(np.linalg.norm(delta))

        if dist < float(WAYPOINT_CFG["reach_threshold"]):
            self.wp_idx = (self.wp_idx + 1) % self.wp_total
            target = self.waypoints_xy[self.wp_idx]
            delta = target - base_pos
            dist = float(np.linalg.norm(delta))
            self.get_logger().info(f"[WP] switch to #{self.wp_idx}: ({target[0]:.2f}, {target[1]:.2f})")

        desired_yaw = float(np.arctan2(delta[1], delta[0]))
        yaw_err = self.wrap_to_pi(desired_yaw - yaw)

        cfg_speed = float(self.waypoints_vE[self.wp_idx]) if self.wp_idx < len(self.waypoints_vE) else MAX_VXY
        v_mag = np.clip(Kp_VXY * dist, 0.0, min(MAX_VXY, cfg_speed))
        vx_w = v_mag * np.cos(desired_yaw)
        vy_w = v_mag * np.sin(desired_yaw)
        wz = np.clip(Kp_WZ * yaw_err, -MAX_WZ, MAX_WZ)

        # Soft velocity servo on free-base DOF: qvel[0]=vx, qvel[1]=vy, qvel[5]=wz
        cur_vx = float(self.data.qvel[0])
        cur_vy = float(self.data.qvel[1])
        cur_wz = float(self.data.qvel[5])

        self.data.qfrc_applied[:6] = 0.0
        self.data.qfrc_applied[0] = 40.0 * (vx_w - cur_vx)
        self.data.qfrc_applied[1] = 40.0 * (vy_w - cur_vy)
        self.data.qfrc_applied[5] = 8.0 * (wz - cur_wz)

        if step % 500 == 0:
            self.get_logger().info(
                f"[WP] idx={self.wp_idx:02d}, pos=({base_pos[0]:.2f},{base_pos[1]:.2f}), "
                f"target=({target[0]:.2f},{target[1]:.2f}), dist={dist:.2f}"
            )

    def _render_waypoints(self):
        if not self.viewer:
            return
        scn = self.viewer.user_scn
        scn.ngeom = 0
        for i, wp in enumerate(self.waypoints_xy):
            rgba = np.array([0.1, 0.8, 0.1, 0.8], dtype=np.float32)
            if i == self.wp_idx:
                rgba = np.array([0.95, 0.2, 0.2, 1.0], dtype=np.float32)
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.08, 0.0, 0.0], dtype=np.float64),
                pos=np.array([float(wp[0]), float(wp[1]), 0.06], dtype=np.float64),
                mat=np.eye(3, dtype=np.float64).reshape(-1),
                rgba=rgba,
            )
            scn.ngeom += 1

    def _publish_robot_state(self):
        q_world = self.data.sensordata[:4]
        rpy_rad = self.quaternion_to_euler(q_world)
        rpy_deg = rpy_rad * (180.0 / np.pi)

        body_acc = self.data.sensordata[4:7]
        angvel_b = self.data.sensordata[7:10]

        imu_msg = ImuData()
        imu_msg.header = MetaType()
        imu_msg.header.frame_id = 0

        stamp = Time()
        sec = int(self.timestamp)
        nanosec = int((self.timestamp - sec) * 1e9)
        stamp.sec = sec
        stamp.nanosec = nanosec
        imu_msg.header.stamp = stamp

        imu_msg.data = ImuDataValue()
        imu_msg.data.roll = float(rpy_deg[0])
        imu_msg.data.pitch = float(rpy_deg[1])
        imu_msg.data.yaw = float(rpy_deg[2])
        imu_msg.data.omega_x = float(angvel_b[0])
        imu_msg.data.omega_y = float(angvel_b[1])
        imu_msg.data.omega_z = float(angvel_b[2])
        imu_msg.data.acc_x = float(body_acc[0])
        imu_msg.data.acc_y = float(body_acc[1])
        imu_msg.data.acc_z = float(body_acc[2])
        self.imu_pub.publish(imu_msg)

        q = self.data.qpos[7:7 + self.dof_num]
        dq = self.data.qvel[6:6 + self.dof_num]
        tau = self.input_tq.flatten()

        pub_pos = np.nan_to_num((q - POS_OFFSET_RAD) * JOINT_DIR, nan=0.0, posinf=0.0, neginf=0.0)
        pub_vel = np.nan_to_num(dq * JOINT_DIR, nan=0.0, posinf=0.0, neginf=0.0)
        pub_tau = np.nan_to_num(tau * JOINT_DIR, nan=0.0, posinf=0.0, neginf=0.0)
        pub_tau = np.clip(pub_tau, -MAX_JOINT_TORQUE_ABS, MAX_JOINT_TORQUE_ABS)

        joints_msg = JointsData()
        joints_msg.header = MetaType()
        joints_msg.header.frame_id = 0
        joints_msg.header.stamp = stamp

        joints_msg.data = JointsDataValue()
        joints_msg.data.joints_data = [JointData() for _ in range(self.dof_num)]
        for i in range(self.dof_num):
            joint = joints_msg.data.joints_data[i]
            joint.name = [32, 32, 32, 32]
            joint.data_id = 0
            joint.status_word = 1
            joint.position = float(pub_pos[i])
            joint.torque = float(pub_tau[i])
            joint.velocity = float(pub_vel[i])
            joint.motion_temp = 40.0
            joint.driver_temp = 45.0
        self.joints_pub.publish(joints_msg)

        base_pose_msg = Float32MultiArray()
        base_pose_msg.data = [
            float(self.data.qpos[0]),
            float(self.data.qpos[1]),
            float(self.data.qpos[2]),
        ]
        self.base_pose_pub.publish(base_pose_msg)

    def start(self):
        step = 0
        last_time = time.time()

        self.get_logger().info(f"[WP] start from waypoint #0: ({self.waypoints_xy[0,0]:.2f}, {self.waypoints_xy[0,1]:.2f})")

        while rclpy.ok():
            if time.time() - last_time >= DT:
                last_time = time.time()
                step += 1

                self._apply_fallback_stand_cmd_if_needed()
                self._apply_joint_torque()
                self._autopilot_track_waypoint(step)

                mujoco.mj_step(self.model, self.data)
                self.timestamp = step * DT

                if step % 5 == 0:
                    self._publish_robot_state()

                if self.viewer and step % RENDER_INTERVAL == 0:
                    self._render_waypoints()
                    self.viewer.sync()

            rclpy.spin_once(self, timeout_sec=0.0)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    rclpy.init()
    sim_node = MuJoCoSimulationWaypointNode()
    try:
        sim_node.start()
    except KeyboardInterrupt:
        pass
    finally:
        sim_node.destroy_node()
        rclpy.shutdown()
