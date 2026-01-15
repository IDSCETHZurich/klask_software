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
COPY ros_env/res/third_party.repos /tmp/third_party.repos

# Import third-party repositories
RUN mkdir -p $OVERLAY_WS/third_party
# TODO: Remove this entire RUN block once klask_hardware repo is public
RUN apt-get update && apt-get install -y openssh-client
RUN --mount=type=ssh \
    mkdir -p ~/.ssh && \
    echo "Host github.com-MeierTobias" > ~/.ssh/config && \
    echo "  HostName github.com" >> ~/.ssh/config && \
    echo "  User git" >> ~/.ssh/config && \
    ssh-keyscan github.com >> ~/.ssh/known_hosts && \
    vcs import $OVERLAY_WS/third_party < /tmp/third_party.repos
# TODO: Once repo is public, replace above RUN block with:
# RUN vcs import $OVERLAY_WS/third_party < /tmp/third_party.repos

# Copy source code
COPY src $OVERLAY_WS/src

# Ignore packages not needed for runtime
RUN touch $OVERLAY_WS/third_party/klask_hardware/ros/src/klask_imaging_pkg/COLCON_IGNORE || true
RUN touch $OVERLAY_WS/third_party/klask_hardware/ros/src/klask_motor_commander_pkg/COLCON_IGNORE || true

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
    libboost-python-dev \
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

# install python packages
RUN pip install --no-cache-dir --upgrade pip
# Remove conflicting distutils-installed package (sympy) to avoid installation issues
RUN rm -rf /usr/lib/python3/dist-packages/sympy* \
    /usr/local/lib/python3*/dist-packages/sympy* \
    /usr/lib/python3.*/dist-packages/sympy* || true
# Install PyTorch with CUDA support
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# Install other requirements
COPY ros_env/res/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

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
    libboost-python-dev \
    xterm \
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

# install python packages
RUN pip install --no-cache-dir --upgrade pip
# Remove conflicting distutils-installed package (sympy) to avoid installation issues
RUN rm -rf /usr/lib/python3/dist-packages/sympy* \
    /usr/local/lib/python3*/dist-packages/sympy* \
    /usr/lib/python3.*/dist-packages/sympy* || true
# Install PyTorch with CUDA support
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# Install other requirements
COPY ros_env/res/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
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

# Copy and set up runtime entrypoint
COPY ros_env/res/runtime_entrypoint.sh /runtime_entrypoint.sh
RUN chmod +x /runtime_entrypoint.sh

WORKDIR $OVERLAY_WS

# Default command: launch state estimator with two player nodes
# Usage: docker run <image> [--player left|right|both]
# Examples:
#   docker run <image>                          # launches with player:=both(defaults)
#   docker run <image> --player left            # launches with player:=left 
#   docker run <image> --player right            # launches with player:=right
#   docker run <image> --player both             # launches with player:=both
ENTRYPOINT ["/runtime_entrypoint.sh"]
