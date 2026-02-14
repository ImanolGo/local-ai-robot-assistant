#!/bin/bash
# Install rtabmap_ros for ROS2 Humble on NVIDIA Jetson Orin Nano
#
# Usage:
#   ./scripts/install_rtabmap.sh          # Try apt first, then source build
#   ./scripts/install_rtabmap.sh --source  # Force source build
#
# This installs:
#   - rtabmap (core library)
#   - rtabmap_ros (ROS2 wrapper): rtabmap_slam, rtabmap_odom, rtabmap_sync, etc.

set -e

FORCE_SOURCE=false
ROS_DISTRO="${ROS_DISTRO:-humble}"
WORKSPACE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

while [[ $# -gt 0 ]]; do
    case $1 in
        --source) FORCE_SOURCE=true; shift ;;
        -h|--help)
            echo "Usage: $0 [--source]"
            echo "  --source  Force building rtabmap_ros from source"
            exit 0
            ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

# ------------------------------------------------------------------
# 1. Check prerequisites
# ------------------------------------------------------------------
log_info "Checking prerequisites..."

if ! command -v ros2 &>/dev/null; then
    log_error "ROS2 not found. Source your ROS2 installation first."
    exit 1
fi

log_info "ROS2 distro: $ROS_DISTRO"

# ------------------------------------------------------------------
# 2. Install system dependencies
# ------------------------------------------------------------------
log_info "Installing system dependencies..."

sudo apt-get update -qq
sudo apt-get install -y -qq \
    libsqlite3-dev \
    libpcl-dev \
    libopencv-dev \
    libproj-dev \
    libsuitesparse-dev \
    libgtsam-dev 2>/dev/null || true

# ------------------------------------------------------------------
# 3. Try apt install (fastest path)
# ------------------------------------------------------------------
install_via_apt() {
    log_info "Attempting apt install of rtabmap_ros..."

    # Check if the package is available
    if apt-cache show "ros-${ROS_DISTRO}-rtabmap-ros" &>/dev/null; then
        sudo apt-get install -y \
            "ros-${ROS_DISTRO}-rtabmap-ros" \
            "ros-${ROS_DISTRO}-rtabmap" \
            "ros-${ROS_DISTRO}-rtabmap-msgs"
        return 0
    else
        log_warn "ros-${ROS_DISTRO}-rtabmap-ros not available via apt."
        return 1
    fi
}

# ------------------------------------------------------------------
# 4. Build from source (fallback)
# ------------------------------------------------------------------
install_from_source() {
    log_info "Building rtabmap_ros from source..."

    RTABMAP_WS="${WORKSPACE_ROOT}/rtabmap_ws"
    mkdir -p "${RTABMAP_WS}/src"
    cd "${RTABMAP_WS}/src"

    # Clone rtabmap (core library)
    if [ ! -d "rtabmap" ]; then
        log_info "Cloning rtabmap core..."
        git clone --depth 1 --branch 0.21.4 \
            https://github.com/introlab/rtabmap.git
    fi

    # Clone rtabmap_ros
    if [ ! -d "rtabmap_ros" ]; then
        log_info "Cloning rtabmap_ros..."
        git clone --depth 1 --branch ros2 \
            https://github.com/introlab/rtabmap_ros.git
    fi

    # Install ROS2 dependencies via rosdep
    cd "${RTABMAP_WS}"
    log_info "Installing rosdep dependencies..."
    rosdep install --from-paths src --ignore-src -r -y 2>/dev/null || true

    # Build with limited parallelism to stay within RAM budget
    log_info "Building rtabmap_ros (this may take 30-60 minutes on Jetson)..."
    source "/opt/ros/${ROS_DISTRO}/setup.bash"
    MAKEFLAGS="-j2" colcon build \
        --symlink-install \
        --cmake-args -DCMAKE_BUILD_TYPE=Release \
        --parallel-workers 1

    log_info "Source build complete."
    log_info "Add to your .bashrc:"
    echo "  source ${RTABMAP_WS}/install/setup.bash"
}

# ------------------------------------------------------------------
# 5. Main flow
# ------------------------------------------------------------------
if [ "$FORCE_SOURCE" = true ]; then
    install_from_source
else
    if install_via_apt; then
        log_info "rtabmap_ros installed via apt."
    else
        log_warn "Falling back to source build..."
        install_from_source
    fi
fi

# ------------------------------------------------------------------
# 6. Verify installation
# ------------------------------------------------------------------
log_info "Verifying installation..."

# Source the workspace(s) so we can find the packages
source "/opt/ros/${ROS_DISTRO}/setup.bash"
if [ -f "${WORKSPACE_ROOT}/rtabmap_ws/install/setup.bash" ]; then
    source "${WORKSPACE_ROOT}/rtabmap_ws/install/setup.bash"
fi

MISSING=()
for pkg in rtabmap_slam rtabmap_odom rtabmap_sync rtabmap_msgs; do
    if ! ros2 pkg list 2>/dev/null | grep -q "$pkg"; then
        MISSING+=("$pkg")
    fi
done

if [ ${#MISSING[@]} -eq 0 ]; then
    log_info "✅ All rtabmap_ros packages verified:"
    echo "   rtabmap_slam, rtabmap_odom, rtabmap_sync, rtabmap_msgs"
else
    log_error "Missing packages: ${MISSING[*]}"
    log_error "Installation may have failed. Check logs above."
    exit 1
fi

# ------------------------------------------------------------------
# 7. Also install robot_localization if not present
# ------------------------------------------------------------------
if ! ros2 pkg list 2>/dev/null | grep -q "robot_localization"; then
    log_info "Installing robot_localization..."
    sudo apt-get install -y "ros-${ROS_DISTRO}-robot-localization" || {
        log_warn "Could not install robot_localization via apt."
        log_warn "You may need to build it from source."
    }
else
    log_info "✅ robot_localization already installed."
fi

log_info "============================================"
log_info " rtabmap_ros installation complete!"
log_info " "
log_info " Next steps:"
log_info "   1. Rebuild the project: cd $WORKSPACE_ROOT && colcon build"
log_info "   2. Test: ros2 launch localization_nodes slam_launch.py"
log_info "============================================"
