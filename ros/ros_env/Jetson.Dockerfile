# Jetson (aarch64 / L4T) variant of SDK.Dockerfile. Differs only in the L4T base image
# (ships CUDA + ROS) and the aarch64 torch wheel; the SDK's x86 cu124 wheel won't work here.
# Build on the Jetson: cd ros && ./ros_env/klask_docker_helper.sh -j build
# Run via the helper (-j) so the GPU is passed through with --runtime nvidia.

# L4T ROS Humble desktop base. The desktop variant bundles the full ROS toolset (rqt,
# rviz, all message packages, rosidl generators, ament tooling, the ros2 CLI) so we don't
# have to re-add them. r36.4.0 = CUDA 12.6 + cuDNN 9, which matches the torch wheel below
# (the smaller ros-base only goes to r36.3.0 = CUDA 12.2/cuDNN 8, which torch no longer fits).
ARG FROM_IMAGE=dustynv/ros:humble-desktop-l4t-r36.4.0
# aarch64 CUDA torch wheel index for JetPack 6 / CUDA 12.6 (use the jp5 index for JetPack 5).
ARG TORCH_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126

FROM $FROM_IMAGE
ARG OVERLAY_WS=/opt/ros/klask_ws
ARG TORCH_INDEX_URL

# Refresh the ROS 2 apt signing key — the one baked into the base image has expired.
RUN curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg

# Base / system dependencies (same non-ROS set as SDK.Dockerfile; python3-vcstool added
# for robustness). Kept in its own layer so editing the ROS list below doesn't re-run it.
RUN apt update && apt install -y \
    gdb \
    python3-pip \
    python3-vcstool \
    libboost-python-dev \
    iproute2 \
    clang-format \
    openssh-client && \
    rm -rf /var/lib/apt/lists/*

# ROS packages. The desktop base already bundles rqt, the message packages, the rosidl
# generators, ament tooling and the full ros2 CLI, so only foxglove-bridge is left to add.
# --force-overwrite guards against any apt OpenCV files clashing with the base's prebuilt
# CUDA OpenCV. (python3-opencv is intentionally not installed; cv2 comes from the base.)
RUN apt update && apt install -y -o Dpkg::Options::="--force-overwrite" \
    ros-${ROS_DISTRO}-foxglove-bridge && \
    rm -rf /var/lib/apt/lists/*

# install python packages
RUN pip install --upgrade pip
# Remove conflicting distutils-installed package (sympy) to avoid installation issues
RUN rm -rf /usr/lib/python3/dist-packages/sympy* \
    /usr/local/lib/python3*/dist-packages/sympy* \
    /usr/lib/python3.*/dist-packages/sympy* || true
# Install PyTorch from the Jetson aarch64 CUDA index (torchvision/torchaudio unused here).
RUN pip install torch --index-url ${TORCH_INDEX_URL}
# Install other requirements. The base image's default pip index is a Jetson mirror
# (only carries CUDA-specific wheels), so fetch these generic wheels from PyPI explicitly.
COPY ros_env/res/requirements.txt /tmp/requirements.txt
RUN pip install --index-url https://pypi.org/simple -r /tmp/requirements.txt

# setup colcon extensions
RUN echo "source /usr/share/colcon_cd/function/colcon_cd.sh" >> ~/.bashrc
RUN echo "export _colcon_cd_root=/opt/ros/${ROS_DISTRO}/" >> ~/.bashrc
RUN echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> ~/.bashrc

# create a workspace directory
RUN mkdir -p $OVERLAY_WS/src
RUN mkdir -p $OVERLAY_WS/.vscode
RUN mkdir -p $OVERLAY_WS/third_party

# import third party repos into workspace
COPY ros_env/res/third_party.repos /tmp/third_party.repos
RUN vcs import $OVERLAY_WS/third_party < /tmp/third_party.repos
# Ignore packages that are not needed / cause trouble in the SDK
RUN touch $OVERLAY_WS/third_party/klask_hardware/ros/src/klask_imaging_pkg/COLCON_IGNORE || true
RUN touch $OVERLAY_WS/third_party/klask_hardware/ros/src/klask_motor_commander_pkg/COLCON_IGNORE || true
RUN touch $OVERLAY_WS/third_party/klask_hardware/ros/src/klask_sprite_generator_pkg/COLCON_IGNORE || true
RUN touch $OVERLAY_WS/third_party/klask_hardware/ros/src/klask_state_estimator_pkg/COLCON_IGNORE || true
RUN touch $OVERLAY_WS/third_party/klask_hardware/ros/src/klask_system_id_pkg/COLCON_IGNORE || true

# auto source ROS setup.bash
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
# auto source workspace overlay if one exists (printf, not echo: bash's echo would write
# the \n literally on this base image, producing a broken .bashrc line)
RUN printf '\nif [ -f %s/install/setup.bash ]; then\n  source %s/install/setup.bash\nfi\n' \
    "$OVERLAY_WS" "$OVERLAY_WS" >> ~/.bashrc

WORKDIR $OVERLAY_WS
