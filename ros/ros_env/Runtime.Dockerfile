ARG FROM_IMAGE=ros:humble
ARG OVERLAY_WS=/opt/ros/klask_ws

# ============================================================================
# Stage 1: Dependency Cacher
# Analyzes dependencies and creates minimal package lists for build/runtime
# ============================================================================
FROM $FROM_IMAGE AS cacher
ARG OVERLAY_WS

# overwrite defaults to persist minimal cache
RUN rosdep update --rosdistro $ROS_DISTRO && \
    cat <<EOF > /etc/apt/apt.conf.d/docker-clean && apt-get update
APT::Install-Recommends "false";
APT::Install-Suggests "false";
EOF

# Copy repository configuration files
WORKDIR $OVERLAY_WS
COPY ros_env/third_party.repos /tmp/third_party.repos

# Import third-party repositories
RUN mkdir -p $OVERLAY_WS/third_party && \
    vcs import $OVERLAY_WS/third_party < /tmp/third_party.repos

# Copy source code
COPY src $OVERLAY_WS/src

# Ignore packages not needed for runtime
RUN touch $OVERLAY_WS/third_party/ros_odrive/odrive_ros2_control/COLCON_IGNORE || true && \
    touch $OVERLAY_WS/third_party/ros_odrive/odrive_botwheel_explorer/COLCON_IGNORE || true

# Derive build and exec dependencies
RUN bash -e <<'EOF'
declare -A types=(
  [exec]="--dependency-types=exec"
  [build]="")

for type in "${!types[@]}"; do
  rosdep install -y \
    --from-paths $OVERLAY_WS/src $OVERLAY_WS/third_party \
    --ignore-src \
    --reinstall \
    --simulate \
    ${types[$type]} \
    | grep 'apt-get install' \
    | awk '{gsub(/'\''/,"",$4); print $4}' \
    | sort -u > /tmp/${type}_debs.txt
done
EOF

# ============================================================================
# Stage 2: Builder
# Installs build dependencies and compiles the workspace
# ============================================================================
FROM $FROM_IMAGE AS builder
ARG OVERLAY_WS

# Install build dependencies
COPY --from=cacher /tmp/build_debs.txt /tmp/build_debs.txt
RUN --mount=type=cache,target=/etc/apt/apt.conf.d,from=cacher,source=/etc/apt/apt.conf.d \
    --mount=type=cache,target=/var/lib/apt/lists,from=cacher,source=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    < /tmp/build_debs.txt xargs apt-get install -y

# Install additional build tools and Qt/X11 dependencies for OpenCV GUI
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    libqt5gui5 \
    libqt5widgets5 \
    libqt5core5a \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY ros_env/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Copy workspace source code
WORKDIR $OVERLAY_WS
COPY --from=cacher $OVERLAY_WS/src ./src
COPY --from=cacher $OVERLAY_WS/third_party ./third_party

# Build workspace with release optimizations
RUN . /opt/ros/$ROS_DISTRO/setup.sh && \
    colcon build \
    --cmake-args -DCMAKE_BUILD_TYPE=Release \
    --merge-install

# ============================================================================
# Stage 3: Runtime
# Minimal image with only runtime dependencies and built workspace
# ============================================================================
FROM $FROM_IMAGE-ros-core AS runtime
ARG OVERLAY_WS

# Install runtime dependencies
COPY --from=cacher /tmp/exec_debs.txt /tmp/exec_debs.txt
RUN --mount=type=cache,target=/etc/apt/apt.conf.d,from=cacher,source=/etc/apt/apt.conf.d \
    --mount=type=cache,target=/var/lib/apt/lists,from=cacher,source=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    < /tmp/exec_debs.txt xargs apt-get install -y

# Install minimal runtime tools and Qt/X11 dependencies for OpenCV GUI
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    libqt5gui5 \
    libqt5widgets5 \
    libqt5core5a \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python runtime dependencies
COPY ros_env/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Copy built workspace from builder
ENV OVERLAY_WS=$OVERLAY_WS
COPY --from=builder $OVERLAY_WS/install $OVERLAY_WS/install

# Source workspace in entrypoint
RUN sed --in-place --expression \
    '$isource "$OVERLAY_WS/install/setup.bash"' \
    /ros_entrypoint.sh

# auto source ROS setup.bash for interactive shells
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
RUN echo "source $OVERLAY_WS/install/setup.bash\n" >> ~/.bashrc

WORKDIR $OVERLAY_WS

# Default command: launch state estimator
CMD ["ros2", "launch", "klask_state_estimation_pkg", "state_estimation_launch.py"]
