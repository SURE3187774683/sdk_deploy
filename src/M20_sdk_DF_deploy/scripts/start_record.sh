#!/bin/bash
# start_record.sh — 同时启动 rosbag 录制 + 轨迹 CSV 记录
#
# 用法:
#   仿真模式 (和 MuJoCo + rl_deploy 一起跑，开第3个终端):
#     bash src/M20_sdk_DF_deploy/scripts/start_record.sh
#
#   真机模式 (在机器人上跑):
#     bash src/M20_sdk_DF_deploy/scripts/start_record.sh --real
#
#   自定义输出目录:
#     bash src/M20_sdk_DF_deploy/scripts/start_record.sh -o /tmp/traj_data
#
#   仅 rosbag 不跑 CSV:
#     bash src/M20_sdk_DF_deploy/scripts/start_record.sh --bag-only
#
# 输出:
#   <output_dir>/trajectory_<ts>.csv       连续采样 + 期望/实际对比
#   <output_dir>/waypoint_summary_<ts>.csv 每航点到达时刻汇总
#   <output_dir>/rosbag2_<ts>/             rosbag 原始录制 (可回放)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 默认
REAL_MODE=false
REAL_FLAG=""
OUTPUT_DIR=""
HZ=50
BAG_ONLY=false

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --real)
            REAL_MODE=true
            REAL_FLAG="--real"
            shift
            ;;
        --bag-only)
            BAG_ONLY=true
            shift
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --hz)
            HZ="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 选择 Python 解释器: 优先 python3.10 (仿真Humble), 否则 python3 (真机Foxy)
if command -v python3.10 &>/dev/null; then
    PYTHON="python3.10"
else
    PYTHON="python3"
fi

# Source ROS2 环境
if [ "$REAL_MODE" = true ]; then
    source /opt/ros/foxy/setup.bash
    source /opt/robot/scripts/setup_ros2.sh
    PYTHON="python3"  # 真机强制 python3
fi
if [ -f /opt/ros/humble/setup.bash ] && [ "$REAL_MODE" != true ]; then
    source /opt/ros/humble/setup.bash
fi
if [ -f "${PROJECT_ROOT}/../../install/setup.bash" ]; then
    source "${PROJECT_ROOT}/../../install/setup.bash"
fi

export ROS_DOMAIN_ID=1
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${PROJECT_ROOT}/traj"
fi
mkdir -p "${OUTPUT_DIR}"

# 要录制的话题
TOPICS="/WAYPOINT_PATH /BASE_POSE2D /IMU_DATA /JOINTS_DATA /JOINTS_CMD"
# 真机模式额外录制 SLAM_ODOM
if [ "$REAL_MODE" = true ]; then
    TOPICS="${TOPICS} /SLAM_ODOM"
fi

BAG_PID=""
CSV_PID=""
_CLEANUP_DONE=false
RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUTPUT_DIR}/${RUN_TS}"
mkdir -p "${RUN_DIR}"

LATEST_CSV=""

cleanup() {
    # Prevent double execution (INT trap + EXIT trap)
    if [ "$_CLEANUP_DONE" = true ]; then
        return
    fi
    _CLEANUP_DONE=true

    echo ""
    echo "[REC] Stopping..."
    # Python already received SIGINT from Ctrl+C (same process group).
    # Our Python signal handler will call save() and exit.
    # Do NOT send kill -INT — it would trigger a second SIGINT that
    # interrupts save() during PNG generation.
    if [ -n "$CSV_PID" ]; then
        # Wait up to 30s for Python to finish saving CSV+PNG
        for i in $(seq 1 300); do
            if ! kill -0 "$CSV_PID" 2>/dev/null; then break; fi
            sleep 0.1
        done
        # Force kill if still alive
        kill -9 "$CSV_PID" 2>/dev/null || true
    fi
    [ -n "$BAG_PID" ] && kill "$BAG_PID" 2>/dev/null || true
    wait 2>/dev/null
    echo "[REC] Done. Files saved in ${RUN_DIR}/"
}
trap cleanup EXIT INT TERM

# ── 1. rosbag record (后台) ──────────────────────────────────────────────
echo "[REC] Starting rosbag2 recording..."
ros2 bag record -o "${RUN_DIR}/rosbag2" ${TOPICS} &
BAG_PID=$!

# ── 2. CSV trajectory recorder ──────────────────────────────────────────
if [ "$BAG_ONLY" = false ]; then
    echo "[REC] Starting CSV trajectory recorder (hz=${HZ})..."
    ${PYTHON} "${SCRIPT_DIR}/record_trajectory.py" \
        -o "${RUN_DIR}" \
        --hz "${HZ}" \
        ${REAL_FLAG} &
    CSV_PID=$!
fi

echo "[REC] Recording... Press Ctrl+C to stop."
echo "[REC] Files will be saved in ${RUN_DIR}/"

# Wait for CSV recorder to finish
wait $CSV_PID 2>/dev/null || true

# Stop rosbag too
[ -n "$BAG_PID" ] && kill "$BAG_PID" 2>/dev/null || true
wait $BAG_PID 2>/dev/null || true