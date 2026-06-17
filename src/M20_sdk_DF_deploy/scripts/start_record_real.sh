#!/bin/bash
# start_record_real.sh - 真机录制: rosbag + CSV + PNG

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR=""
HZ=50
BAG_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
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

source /opt/ros/foxy/setup.bash
source /opt/robot/scripts/setup_ros2.sh
if [ -f "${PROJECT_ROOT}/../../install/setup.bash" ]; then
    source "${PROJECT_ROOT}/../../install/setup.bash"
fi

PYTHON="python3"

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${PROJECT_ROOT}/traj_real"
fi
mkdir -p "${OUTPUT_DIR}"

RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUTPUT_DIR}/${RUN_TS}"
mkdir -p "${RUN_DIR}"

TOPICS="/WAYPOINT_PATH /WAYPOINT_PATH/path /WAYPOINT_PATH/markers /WAYPOINT_PATH/poses /SLAM_ODOM /IMU_DATA /JOINTS_DATA /JOINTS_CMD"

BAG_PID=""
_CLEANUP_DONE=false

cleanup() {
    if [ "$_CLEANUP_DONE" = true ]; then
        return
    fi
    _CLEANUP_DONE=true

    echo ""
    echo "[REC] Stopping..."

    if [ -n "$BAG_PID" ]; then
        kill "$BAG_PID" 2>/dev/null || true
        wait "$BAG_PID" 2>/dev/null || true
    fi
    echo "[REC] Done. Files saved in ${RUN_DIR}/"
}
trap cleanup EXIT

echo "[REC] Starting rosbag2 recording..."
ros2 bag record -o "${RUN_DIR}/rosbag2" ${TOPICS} &
BAG_PID=$!

if [ "$BAG_ONLY" = false ]; then
    echo "[REC] Starting real trajectory recorder (hz=${HZ})..."
    ${PYTHON} "${SCRIPT_DIR}/record_trajectory_real.py" \
        -o "${RUN_DIR}" \
        --hz "${HZ}"
fi

echo "[REC] Recording... Press Ctrl+C to stop."
echo "[REC] Files will be saved in ${RUN_DIR}/"

if [ "$BAG_ONLY" = true ] && [ -n "$BAG_PID" ]; then
    wait "$BAG_PID" 2>/dev/null || true
fi
