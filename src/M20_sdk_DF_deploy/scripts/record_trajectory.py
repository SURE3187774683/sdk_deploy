"""
record_trajectory.py — 记录期望轨迹与真实轨迹对比数据

订阅 ROS2 话题:
  /WAYPOINT_PATH   Float32MultiArray  C++策略发布的期望航点 [wp_idx, N, x0,y0,..., vE0,vE1,...]
  /BASE_POSE2D     Float32MultiArray  MuJoCo发布的真实位置 [x, y, z]
  /SLAM_ODOM       Odometry           真机SLAM定位 (真机模式备选)
  /IMU_DATA        ImuData            真实IMU姿态 (yaw)

输出文件 (Ctrl+C 停止时自动生成):
  trajectory_<timestamp>.csv           连续采样数据
  waypoint_summary_<timestamp>.csv     每航点到达时刻汇总
  trajectory_top_view.png              期望轨迹 vs 实际轨迹俯视图
  speed_over_time.png                  期望速度 vs 实际速度时序图

使用:
  export ROS_DOMAIN_ID=1
  source install/setup.bash
  python3.10 src/M20_sdk_DF_deploy/scripts/record_trajectory.py
  python3.10 src/M20_sdk_DF_deploy/scripts/record_trajectory.py -o /tmp/traj_data

默认输出目录: src/M20_sdk_DF_deploy/traj/
"""

from __future__ import annotations

import csv
import time
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry
from drdds.msg import ImuData


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class TrajectoryRecorderNode(Node):
    def __init__(self, output_dir: str = ".", record_hz: float = 50.0):
        super().__init__('trajectory_recorder')

        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._record_dt = 1.0 / record_hz
        self._records: list[dict] = []

        # ── Waypoint path (desired trajectory) ──────────────────────────
        self._wp_xy: np.ndarray | None = None    # (N, 2) world-frame
        self._wp_vE: np.ndarray | None = None    # (N,)   expected speeds
        self._wp_idx: int = -1
        self._wp_total: int = 0
        self._wp_stamp: float = 0.0

        self._wp_sub = self.create_subscription(
            Float32MultiArray, '/WAYPOINT_PATH', self._wp_cb, 10)

        # ── Actual position ─────────────────────────────────────────────
        self._actual_xy = np.zeros(2, dtype=np.float64)
        self._actual_yaw = 0.0
        self._pos_stamp: float = 0.0
        self._pos_source = "none"  # "BASE_POSE2D" or "SLAM_ODOM"

        self._base_pose_sub = self.create_subscription(
            Float32MultiArray, '/BASE_POSE2D', self._base_pose_cb, 200)
        self._slam_sub = self.create_subscription(
            Odometry, '/SLAM_ODOM', self._slam_cb, 10)

        # ── IMU (yaw) ──────────────────────────────────────────────────
        self._imu_yaw_rad = 0.0
        self._imu_stamp: float = 0.0
        self._imu_sub = self.create_subscription(
            ImuData, '/IMU_DATA', self._imu_cb, 200)

        # ── Velocity estimation (from position deltas) ─────────────────
        self._actual_vx = 0.0
        self._actual_vy = 0.0
        self._speed_ema_alpha = 0.35

        # ── Timer ──────────────────────────────────────────────────────
        self._timer = self.create_timer(self._record_dt, self._record_cb)
        self._start_time = time.time()

        self.get_logger().info(
            f"[REC] Trajectory recorder started, hz={record_hz}, "
            f"output_dir={self._output_dir}")

    # ── Callbacks ───────────────────────────────────────────────────────

    def _wp_cb(self, msg: Float32MultiArray):
        d = msg.data
        if len(d) < 2:
            return
        wp_idx = int(d[0])
        N = int(d[1])
        if N <= 0 or len(d) < 2 + 3 * N:
            return
        self._wp_xy = np.array(d[2:2 + 2 * N], dtype=np.float32).reshape(N, 2)
        self._wp_vE = np.array(d[2 + 2 * N:2 + 3 * N], dtype=np.float32)
        self._wp_idx = wp_idx
        self._wp_total = N
        self._wp_stamp = self.get_clock().now().nanoseconds / 1e9

    def _base_pose_cb(self, msg: Float32MultiArray):
        if len(msg.data) < 2:
            return
        stamp = self.get_clock().now().nanoseconds / 1e9
        self._update_actual_pose(float(msg.data[0]), float(msg.data[1]), stamp)
        self._pos_source = "BASE_POSE2D"

    def _slam_cb(self, msg: Odometry):
        stamp = self.get_clock().now().nanoseconds / 1e9
        self._update_actual_pose(msg.pose.pose.position.x, msg.pose.pose.position.y, stamp)
        self._pos_source = "SLAM_ODOM"
        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._actual_yaw = float(np.arctan2(siny_cosp, cosy_cosp))

    def _imu_cb(self, msg: ImuData):
        # IMU yaw is in degrees in the DDS message
        self._imu_yaw_rad = float(msg.data.yaw) * (np.pi / 180.0)
        self._imu_stamp = self.get_clock().now().nanoseconds / 1e9

    def _update_actual_pose(self, x: float, y: float, stamp: float):
        xy = np.array([x, y], dtype=np.float64)
        if self._pos_stamp > 0.0:
            dt = stamp - self._pos_stamp
            if dt > 1e-6:
                vel = (xy - self._actual_xy) / dt
                self._actual_vx = float(
                    self._speed_ema_alpha * vel[0] +
                    (1.0 - self._speed_ema_alpha) * self._actual_vx)
                self._actual_vy = float(
                    self._speed_ema_alpha * vel[1] +
                    (1.0 - self._speed_ema_alpha) * self._actual_vy)
        self._actual_xy = xy
        self._pos_stamp = stamp

    # ── Record ─────────────────────────────────────────────────────────

    def _record_cb(self):
        t_now = time.time() - self._start_time

        actual_speed = float(np.hypot(self._actual_vx, self._actual_vy))

        # Use SLAM_ODOM yaw if available, else IMU yaw
        if self._pos_source == "SLAM_ODOM":
            yaw = self._actual_yaw
        else:
            yaw = self._imu_yaw_rad

        # Expected waypoint data
        exp_x, exp_y, exp_vE = 0.0, 0.0, 0.0
        pos_error, heading_error = 0.0, 0.0
        if self._wp_xy is not None and self._wp_total > 0:
            idx = min(self._wp_idx, self._wp_total - 1)
            exp_x = float(self._wp_xy[idx, 0])
            exp_y = float(self._wp_xy[idx, 1])
            exp_vE = float(self._wp_vE[idx]) if idx < len(self._wp_vE) else 0.0

            dx = self._actual_xy[0] - exp_x
            dy = self._actual_xy[1] - exp_y
            pos_error = float(np.hypot(dx, dy))

            desired_heading = float(np.arctan2(dy, dx)) if pos_error > 1e-6 else 0.0
            # Actually heading error: desired heading from waypoint to robot
            # is the heading the robot SHOULD have to face the waypoint
            wp_to_robot_heading = float(np.arctan2(
                exp_y - self._actual_xy[1], exp_x - self._actual_xy[0]))
            heading_error = float(wrap_to_pi(wp_to_robot_heading - yaw))

        row = {
            "time": round(t_now, 4),
            "wp_idx": self._wp_idx,
            "wp_total": self._wp_total,
            "expected_x": round(exp_x, 4),
            "expected_y": round(exp_y, 4),
            "expected_vE": round(exp_vE, 4),
            "actual_x": round(float(self._actual_xy[0]), 4),
            "actual_y": round(float(self._actual_xy[1]), 4),
            "actual_vx": round(self._actual_vx, 4),
            "actual_vy": round(self._actual_vy, 4),
            "actual_yaw": round(yaw, 4),
            "actual_speed": round(actual_speed, 4),
            "position_error": round(pos_error, 4),
            "heading_error": round(heading_error, 4),
            "pos_source": self._pos_source,
        }
        self._records.append(row)

    # ── Save ────────────────────────────────────────────────────────────

    def save(self):
        if not self._records:
            self.get_logger().warn("[REC] No data recorded, skipping save.")
            return

        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        # All files for this run go into a timestamped subfolder
        run_dir = self._output_dir / ts_str
        run_dir.mkdir(parents=True, exist_ok=True)

        csv_path = run_dir / f"trajectory_{ts_str}.csv"

        fields = list(self._records[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self._records)

        self.get_logger().info(
            f"[REC] Saved {len(self._records)} rows to {csv_path}")

        # ── Waypoint-by-waypoint summary ──────────────────────────────
        summary_path = run_dir / f"waypoint_summary_{ts_str}.csv"
        summary_fields = [
            "wp_idx", "expected_x", "expected_y", "expected_vE",
            "arrival_actual_x", "arrival_actual_y", "arrival_actual_speed",
            "arrival_position_error", "arrival_heading_error",
            "arrival_time",
        ]
        summary_rows: list[dict] = []
        prev_idx = -1
        for row in self._records:
            idx = row["wp_idx"]
            if idx != prev_idx and idx >= 0 and row["wp_total"] > 0:
                # New waypoint reached (or first data with a valid index)
                summary_rows.append({
                    "wp_idx": idx,
                    "expected_x": row["expected_x"],
                    "expected_y": row["expected_y"],
                    "expected_vE": row["expected_vE"],
                    "arrival_actual_x": row["actual_x"],
                    "arrival_actual_y": row["actual_y"],
                    "arrival_actual_speed": row["actual_speed"],
                    "arrival_position_error": row["position_error"],
                    "arrival_heading_error": row["heading_error"],
                    "arrival_time": row["time"],
                })
                prev_idx = idx

        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fields)
            writer.writeheader()
            writer.writerows(summary_rows)

        self.get_logger().info(
            f"[REC] Saved waypoint summary ({len(summary_rows)} waypoints) to {summary_path}")

        # ── Generate PNG plots ──────────────────────────────────────────
        if _HAS_MPL:
            self._plot_and_save(run_dir, ts_str, summary_rows)
        else:
            self.get_logger().warn(
                "[REC] matplotlib not installed, skipping PNG generation. "
                "CSV files are saved and can be plotted offline.")


    # ── Plotting ──────────────────────────────────────────────────────

    def _records_to_arrays(self) -> dict:
        """Convert self._records list-of-dicts to numpy arrays for plotting."""
        keys = ["time", "wp_idx", "expected_x", "expected_y", "expected_vE",
                "actual_x", "actual_y", "actual_speed", "position_error"]
        result = {}
        for k in keys:
            result[k] = np.array([float(r[k]) for r in self._records])
        return result

    @staticmethod
    def _dist_to_nearest_on_polyline(pts: np.ndarray, polyline: np.ndarray) -> np.ndarray:
        """Compute distance from each point in pts (N,2) to nearest point on polyline (M,2)."""
        # pts: (N,2), polyline: (M,2)
        # For each segment polyline[i] -> polyline[i+1], compute distance from each pt
        N = pts.shape[0]
        M = polyline.shape[0]
        if M < 2 or N == 0:
            return np.zeros(N, dtype=np.float64)

        min_dists = np.full(N, np.inf)
        for i in range(M - 1):
            A = polyline[i]      # (2,)
            B = polyline[i + 1]  # (2,)
            AB = B - A           # (2,)
            AB_sq = float(np.dot(AB, AB))
            if AB_sq < 1e-12:
                # Degenerate segment
                dists = np.linalg.norm(pts - A, axis=1)
                min_dists = np.minimum(min_dists, dists)
                continue
            # Project each pt onto AB: t = dot(AP, AB) / dot(AB, AB)
            AP = pts - A  # (N,2)
            t = AP @ AB / AB_sq  # (N,)
            t = np.clip(t, 0.0, 1.0)
            # Nearest point on segment
            nearest = A + np.outer(t, AB)  # (N,2)
            dists = np.linalg.norm(pts - nearest, axis=1)  # (N,)
            min_dists = np.minimum(min_dists, dists)

        return min_dists

    def _plot_and_save(self, run_dir: Path, ts_str: str, summary_rows: list[dict]):
        """Generate trajectory_top_view, speed_over_time, pos_error_over_time PNGs."""
        traj = self._records_to_arrays()

        # Build arrivals arrays from summary_rows
        arrivals: dict[str, np.ndarray] = {}
        if summary_rows:
            for k in summary_rows[0]:
                arrivals[k] = np.array([float(r[k]) for r in summary_rows])
        else:
            arrivals = {"wp_idx": np.array([]), "expected_vE": np.array([]),
                        "arrival_time": np.array([]), "expected_x": np.array([]),
                        "expected_y": np.array([]), "arrival_actual_x": np.array([]),
                        "arrival_actual_y": np.array([])}

        # ── 1. trajectory_top_view ──────────────────────────────────────
        fig1, ax1 = plt.subplots(figsize=(10, 8))

        exp_x = traj["expected_x"]
        exp_y = traj["expected_y"]
        wp_idx = traj["wp_idx"]

        unique_wp_indices = []
        prev = -1
        for i in range(len(wp_idx)):
            idx = int(wp_idx[i])
            if idx != prev and idx >= 0:
                unique_wp_indices.append(i)
                prev = idx

        polyline = np.column_stack([exp_x[unique_wp_indices], exp_y[unique_wp_indices]]) if len(unique_wp_indices) > 1 else None

        if polyline is not None:
            ax1.plot(polyline[:, 0], polyline[:, 1], "-", color="green", linewidth=1.5,
                     label="Expected waypoints", zorder=3)

        ax1.plot(traj["actual_x"], traj["actual_y"], "-", color="steelblue",
                 linewidth=0.8, alpha=0.7, label="Actual trajectory", zorder=2)

        if len(arrivals["wp_idx"]) > 0:
            ax1.scatter(arrivals["arrival_actual_x"], arrivals["arrival_actual_y"],
                        c="red", s=40, marker="x", linewidths=1.5,
                        label="Arrival positions", zorder=4)

        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_title("Trajectory Top View")
        ax1.set_aspect("equal", adjustable="datalim")
        ax1.legend(loc="best", fontsize=8)
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()

        # ── 2. speed_over_time ──────────────────────────────────────────
        fig2, ax2 = plt.subplots(figsize=(12, 5))

        ax2.plot(traj["time"], traj["actual_speed"], "-", color="steelblue",
                 linewidth=0.8, alpha=0.8, label="Actual speed")

        if len(arrivals["wp_idx"]) > 0:
            t_step = list(arrivals["arrival_time"])
            v_step = list(arrivals["expected_vE"])
            if len(traj["time"]) > 0:
                t_step.append(traj["time"][-1])
                v_step.append(arrivals["expected_vE"][-1])
            ax2.step(t_step, v_step, where="post", color="green", linewidth=1.5,
                     alpha=0.7, label="Expected vE")

        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Speed (m/s)")
        ax2.set_title("Speed Over Time")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()

        # ── 3. pos_error_over_time ──────────────────────────────────────
        actual_pts = np.column_stack([traj["actual_x"], traj["actual_y"]])
        if polyline is not None:
            nearest_dists = self._dist_to_nearest_on_polyline(actual_pts, polyline)
        else:
            nearest_dists = traj["position_error"]  # fallback: dist to current wp

        fig3, ax3 = plt.subplots(figsize=(12, 5))
        ax3.plot(traj["time"], nearest_dists, "-", color="orangered",
                 linewidth=0.8, alpha=0.8, label="Dist to nearest trajectory point")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Position Error (m)")
        ax3.set_title("Position Error Over Time")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()

        # ── Save PNGs ──────────────────────────────────────────────────
        p1 = run_dir / f"trajectory_top_view_{ts_str}.png"
        p2 = run_dir / f"speed_over_time_{ts_str}.png"
        p3 = run_dir / f"pos_error_over_time_{ts_str}.png"
        fig1.savefig(p1, dpi=150, bbox_inches="tight")
        self.get_logger().info(f"[REC] Saved {p1}")
        fig2.savefig(p2, dpi=150, bbox_inches="tight")
        self.get_logger().info(f"[REC] Saved {p2}")
        fig3.savefig(p3, dpi=150, bbox_inches="tight")
        self.get_logger().info(f"[REC] Saved {p3}")
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)


def main():
    parser = argparse.ArgumentParser(description="Record trajectory data")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Directory to save files (default: src/M20_sdk_DF_deploy/traj/)")
    parser.add_argument("--hz", type=float, default=50.0,
                        help="Recording frequency in Hz (default: 50)")
    args = parser.parse_args()

    # Default output dir: <script_dir>/../traj/
    default_dir = str(Path(__file__).resolve().parent.parent / "traj")
    output_dir = args.output_dir if args.output_dir else default_dir

    rclpy.init()
    node = TrajectoryRecorderNode(output_dir=output_dir, record_hz=args.hz)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
