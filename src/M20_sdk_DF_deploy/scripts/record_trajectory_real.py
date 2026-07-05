"""
record_trajectory_real.py - 记录真机期望轨迹与真实轨迹对比数据

订阅 ROS2 话题:
  /WAYPOINT_PATH   Float32MultiArray  期望航点 [wp_idx, N, x0,y0,..., vE0,vE1,...]
  /SLAM_ODOM       Odometry           真机 SLAM 位姿 (仅用 pose, 速度由位置差分计算)
  /IMU_DATA        ImuData            真实 IMU 姿态 (yaw 备选)

输出文件:
  trajectory_<timestamp>.csv           连续采样数据
  waypoint_summary_<timestamp>.csv     每航点到达时刻汇总
  trajectory_top_view_<timestamp>.png
  speed_over_time_<timestamp>.png
  pos_error_over_time_<timestamp>.png

特性:
  1. 使用位置差分计算线速度 (Δposition / Δtime + EMA 滤波)
  2. 对差分速度做有限值检查、限幅、死区
  3. 连续采样 CSV 增量写盘，异常退出时也尽量保留已录数据
  4. save() 幂等并带异常保护，避免图片生成失败影响 CSV 保存
"""

from __future__ import annotations

import argparse
import atexit
import csv
import signal
import time
import threading
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
from drdds.msg import ImuData
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

try:
    from rclpy.signals import SignalHandlerOptions
except ImportError:
    SignalHandlerOptions = None


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class TrajectoryRecorderRealNode(Node):
    def __init__(self, output_dir: str = ".", record_hz: float = 50.0):
        super().__init__("trajectory_recorder_real")

        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._record_dt = 1.0 / record_hz
        self._records: list[dict] = []
        self._saved = False
        self._save_lock = threading.Lock()

        self._ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_dir = self._output_dir
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self._run_dir / f"trajectory_{self._ts_str}.csv"
        self._summary_path = self._run_dir / f"waypoint_summary_{self._ts_str}.csv"
        self._csv_fields = [
            "time",
            "wp_idx",
            "wp_total",
            "expected_x",
            "expected_y",
            "expected_vE",
            "actual_x",
            "actual_y",
            "actual_vx",
            "actual_vy",
            "actual_yaw",
            "actual_speed",
            "position_error",
            "heading_error",
            "pos_source",
            "vel_source",
        ]
        self._csv_file = self._csv_path.open("w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_fields)
        self._csv_writer.writeheader()
        self._csv_file.flush()

        self._wp_xy: np.ndarray | None = None
        self._wp_vE: np.ndarray | None = None
        self._wp_idx: int = -1
        self._wp_total: int = 0

        self._actual_xy = np.zeros(2, dtype=np.float64)
        self._actual_yaw = 0.0
        self._pos_stamp = 0.0
        self._pos_source = "none"

        self._actual_vx = 0.0
        self._actual_vy = 0.0
        self._vel_source = "none"
        self._speed_ema_alpha = 0.35
        self._speed_deadband = 0.02
        self._speed_limit = 5.0

        self._imu_yaw_rad = 0.0

        self.create_subscription(Float32MultiArray, "/WAYPOINT_PATH", self._wp_cb, 10)
        self.create_subscription(Odometry, "/SLAM_ODOM", self._slam_cb, 50)
        self.create_subscription(ImuData, "/IMU_DATA", self._imu_cb, 200)

        self._timer = self.create_timer(self._record_dt, self._record_cb)
        self._start_time = time.time()

        self.get_logger().info(
            f"[REC] Real trajectory recorder started, hz={record_hz}, "
            f"run_dir={self._run_dir}"
        )

    def _wp_cb(self, msg: Float32MultiArray):
        data = msg.data
        if len(data) < 2:
            return
        wp_idx = int(data[0])
        total = int(data[1])
        if total <= 0 or len(data) < 2 + 3 * total:
            return

        self._wp_xy = np.array(data[2 : 2 + 2 * total], dtype=np.float32).reshape(total, 2)
        self._wp_vE = np.array(data[2 + 2 * total : 2 + 3 * total], dtype=np.float32)
        self._wp_idx = wp_idx
        self._wp_total = total

    def _slam_cb(self, msg: Odometry):
        stamp = self.get_clock().now().nanoseconds / 1e9
        self._update_actual_pose(msg.pose.pose.position.x, msg.pose.pose.position.y, stamp)
        self._pos_source = "SLAM_ODOM"

        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._actual_yaw = float(np.arctan2(siny_cosp, cosy_cosp))

    def _imu_cb(self, msg: ImuData):
        self._imu_yaw_rad = float(msg.data.yaw) * (np.pi / 180.0)

    def _update_actual_pose(self, x: float, y: float, stamp: float):
        xy = np.array([x, y], dtype=np.float64)
        if self._pos_stamp > 0.0:
            dt = stamp - self._pos_stamp
            if dt > 1e-6:
                vel = (xy - self._actual_xy) / dt
                vx = float(vel[0])
                vy = float(vel[1])
                if not np.isfinite(vx) or not np.isfinite(vy):
                    pass
                else:
                    vx = float(np.clip(vx, -self._speed_limit, self._speed_limit))
                    vy = float(np.clip(vy, -self._speed_limit, self._speed_limit))
                    if abs(vx) < self._speed_deadband:
                        vx = 0.0
                    if abs(vy) < self._speed_deadband:
                        vy = 0.0
                    self._actual_vx = float(
                        self._speed_ema_alpha * vx
                        + (1.0 - self._speed_ema_alpha) * self._actual_vx
                    )
                    self._actual_vy = float(
                        self._speed_ema_alpha * vy
                        + (1.0 - self._speed_ema_alpha) * self._actual_vy
                    )
                    self._vel_source = "position_delta"
        self._actual_xy = xy
        self._pos_stamp = stamp

    def _record_cb(self):
        t_now = time.time() - self._start_time
        actual_speed = float(np.hypot(self._actual_vx, self._actual_vy))
        yaw = self._actual_yaw if self._pos_source == "SLAM_ODOM" else self._imu_yaw_rad

        exp_x, exp_y, exp_vE = 0.0, 0.0, 0.0
        pos_error, heading_error = 0.0, 0.0
        if self._wp_xy is not None and self._wp_total > 0:
            idx = min(max(self._wp_idx, 0), self._wp_total - 1)
            exp_x = float(self._wp_xy[idx, 0])
            exp_y = float(self._wp_xy[idx, 1])
            exp_vE = float(self._wp_vE[idx]) if idx < len(self._wp_vE) else 0.0
            dx = self._actual_xy[0] - exp_x
            dy = self._actual_xy[1] - exp_y
            pos_error = float(np.hypot(dx, dy))
            wp_to_robot_heading = float(
                np.arctan2(exp_y - self._actual_xy[1], exp_x - self._actual_xy[0])
            )
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
            "vel_source": self._vel_source,
        }
        self._records.append(row)
        self._write_incremental_row(row)

    def _write_incremental_row(self, row: dict):
        try:
            self._csv_writer.writerow(row)
            self._csv_file.flush()
        except Exception as exc:
            self.get_logger().error(f"[REC] Failed to append CSV row: {exc}")

    def save(self):
        with self._save_lock:
            if self._saved:
                return
            self._saved = True
        self.get_logger().info("[REC] save() started")

        try:
            if not self._csv_file.closed:
                self._csv_file.flush()
                self._csv_file.close()
        except Exception as exc:
            self.get_logger().error(f"[REC] Failed to close CSV file: {exc}")

        if not self._records:
            self.get_logger().warn("[REC] No data recorded, skipping summary and PNG save.")
            return

        try:
            self._save_summary_csv()
            self.get_logger().info(
                f"[REC] Saved {len(self._records)} rows to {self._csv_path}"
            )
        except Exception as exc:
            self.get_logger().error(f"[REC] Failed to save summary CSV: {exc}")

        if not _HAS_MPL:
            self.get_logger().warn("[REC] matplotlib not installed, skipping PNG generation.")
            return

        try:
            summary_rows = self._build_summary_rows()
            self._plot_and_save(self._run_dir, self._ts_str, summary_rows)
            self.get_logger().info("[REC] save() finished")
        except Exception as exc:
            self.get_logger().error(f"[REC] Failed to save PNG figures: {exc}")

    def _build_summary_rows(self) -> list[dict]:
        summary_rows: list[dict] = []
        prev_idx = -1
        for row in self._records:
            idx = int(row["wp_idx"])
            if idx != prev_idx and idx >= 0 and int(row["wp_total"]) > 0:
                summary_rows.append(
                    {
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
                    }
                )
                prev_idx = idx
        return summary_rows

    def _save_summary_csv(self):
        summary_rows = self._build_summary_rows()
        summary_fields = [
            "wp_idx",
            "expected_x",
            "expected_y",
            "expected_vE",
            "arrival_actual_x",
            "arrival_actual_y",
            "arrival_actual_speed",
            "arrival_position_error",
            "arrival_heading_error",
            "arrival_time",
        ]
        with self._summary_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=summary_fields)
            writer.writeheader()
            writer.writerows(summary_rows)

        self.get_logger().info(
            f"[REC] Saved waypoint summary ({len(summary_rows)} waypoints) to {self._summary_path}"
        )

    def _records_to_arrays(self) -> dict[str, np.ndarray]:
        keys = [
            "time",
            "wp_idx",
            "expected_x",
            "expected_y",
            "expected_vE",
            "actual_x",
            "actual_y",
            "actual_speed",
            "position_error",
        ]
        return {key: np.array([float(record[key]) for record in self._records]) for key in keys}

    @staticmethod
    def _dist_to_nearest_on_polyline(pts: np.ndarray, polyline: np.ndarray) -> np.ndarray:
        num_pts = pts.shape[0]
        num_poly = polyline.shape[0]
        if num_poly < 2 or num_pts == 0:
            return np.zeros(num_pts, dtype=np.float64)

        min_dists = np.full(num_pts, np.inf)
        for idx in range(num_poly - 1):
            point_a = polyline[idx]
            point_b = polyline[idx + 1]
            segment = point_b - point_a
            seg_sq = float(np.dot(segment, segment))
            if seg_sq < 1e-12:
                min_dists = np.minimum(min_dists, np.linalg.norm(pts - point_a, axis=1))
                continue
            ap = pts - point_a
            proj = np.clip(ap @ segment / seg_sq, 0.0, 1.0)
            nearest = point_a + np.outer(proj, segment)
            min_dists = np.minimum(min_dists, np.linalg.norm(pts - nearest, axis=1))
        return min_dists

    def _plot_and_save(self, run_dir: Path, ts_str: str, summary_rows: list[dict]):
        traj = self._records_to_arrays()

        arrivals: dict[str, np.ndarray]
        if summary_rows:
            arrivals = {
                key: np.array([float(row[key]) for row in summary_rows])
                for key in summary_rows[0]
            }
        else:
            arrivals = {
                "wp_idx": np.array([]),
                "expected_vE": np.array([]),
                "arrival_time": np.array([]),
                "expected_x": np.array([]),
                "expected_y": np.array([]),
                "arrival_actual_x": np.array([]),
                "arrival_actual_y": np.array([]),
            }

        fig1, ax1 = plt.subplots(figsize=(10, 8))
        exp_x = traj["expected_x"]
        exp_y = traj["expected_y"]
        wp_idx = traj["wp_idx"]

        unique_wp_indices = []
        prev_idx = -1
        for idx, wp_value in enumerate(wp_idx):
            wp_int = int(wp_value)
            if wp_int != prev_idx and wp_int >= 0:
                unique_wp_indices.append(idx)
                prev_idx = wp_int

        polyline = (
            np.column_stack([exp_x[unique_wp_indices], exp_y[unique_wp_indices]])
            if len(unique_wp_indices) > 1
            else None
        )

        if polyline is not None:
            ax1.plot(
                polyline[:, 0],
                polyline[:, 1],
                "-",
                color="green",
                linewidth=1.5,
                label="Expected waypoints",
                zorder=3,
            )

        ax1.plot(
            traj["actual_x"],
            traj["actual_y"],
            "-",
            color="steelblue",
            linewidth=0.8,
            alpha=0.7,
            label="Actual trajectory",
            zorder=2,
        )

        if len(arrivals["wp_idx"]) > 0:
            ax1.scatter(
                arrivals["arrival_actual_x"],
                arrivals["arrival_actual_y"],
                c="red",
                s=40,
                marker="x",
                linewidths=1.5,
                label="Arrival positions",
                zorder=4,
            )

        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_title("Trajectory Top View")
        ax1.set_aspect("equal", adjustable="datalim")
        ax1.legend(loc="best", fontsize=8)
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()

        fig2, ax2 = plt.subplots(figsize=(12, 5))
        ax2.plot(
            traj["time"],
            traj["actual_speed"],
            "-",
            color="steelblue",
            linewidth=0.8,
            alpha=0.8,
            label="Actual speed",
        )

        if len(arrivals["wp_idx"]) > 0:
            t_step = list(arrivals["arrival_time"])
            v_step = list(arrivals["expected_vE"])
            if len(traj["time"]) > 0:
                t_step.append(traj["time"][-1])
                v_step.append(arrivals["expected_vE"][-1])
            ax2.step(
                t_step,
                v_step,
                where="post",
                color="green",
                linewidth=1.5,
                alpha=0.7,
                label="Expected vE",
            )

        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Speed (m/s)")
        ax2.set_title("Speed Over Time")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()

        actual_pts = np.column_stack([traj["actual_x"], traj["actual_y"]])
        nearest_dists = (
            self._dist_to_nearest_on_polyline(actual_pts, polyline)
            if polyline is not None
            else traj["position_error"]
        )

        fig3, ax3 = plt.subplots(figsize=(12, 5))
        ax3.plot(
            traj["time"],
            nearest_dists,
            "-",
            color="orangered",
            linewidth=0.8,
            alpha=0.8,
            label="Dist to nearest trajectory point",
        )
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Position Error (m)")
        ax3.set_title("Position Error Over Time")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()

        plot_paths = [
            run_dir / f"trajectory_top_view_{ts_str}.png",
            run_dir / f"speed_over_time_{ts_str}.png",
            run_dir / f"pos_error_over_time_{ts_str}.png",
        ]
        for fig, path in zip((fig1, fig2, fig3), plot_paths):
            fig.savefig(path, dpi=150, bbox_inches="tight")
            self.get_logger().info(f"[REC] Saved {path}")
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Record real trajectory data")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Directory to save files (default: src/M20_sdk_DF_deploy/traj_real/)",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=50.0,
        help="Recording frequency in Hz (default: 50)",
    )
    args = parser.parse_args()

    default_dir = str(Path(__file__).resolve().parent.parent / "traj_real")
    output_dir = args.output_dir if args.output_dir else default_dir

    if SignalHandlerOptions is not None:
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    else:
        rclpy.init()
    node = TrajectoryRecorderRealNode(output_dir=output_dir, record_hz=args.hz)

    def _request_save_and_shutdown(_signum=None, _frame=None):
        node.save()
        try:
            rclpy.shutdown()
        except Exception:
            pass

    signal.signal(signal.SIGINT, _request_save_and_shutdown)
    signal.signal(signal.SIGTERM, _request_save_and_shutdown)
    atexit.register(node.save)

    try:
        rclpy.spin(node)
    finally:
        node.save()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
