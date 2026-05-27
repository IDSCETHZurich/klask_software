# Jetson (aarch64 / L4T) variant of SDK.Dockerfile. Differs only in the L4T base image
# (ships CUDA + ROS) and the aarch64 torch wheel; the desktop cu124 wheel won't work here.
# Build on the Jetson: cd ros && ./ros_env/klask_docker_helper.sh -j build
# Run via the helper (-j) so the GPU is passed through with --runtime nvidia.

# L4T ROS Humble base. r36.3.0 is the newest dustynv r36 tag and runs on r36.5 hosts.
ARG FROM_IMAGE=dustynv/ros:humble-ros-base-l4t-r36.3.0
# aarch64 CUDA torch wheel index for JetPack 6 / CUDA 12.6 (use the jp5 index for JetPack 5).
ARG TORCH_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126

FROM $FROM_IMAGE
ARG OVERLAY_WS=/opt/ros/klask_ws
ARG TORCH_INDEX_URL

# Refresh the ROS 2 apt signing key — the one baked into the base image has expired.
RUN curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg

# ros packages (same set as SDK.Dockerfile; python3-vcstool added for robustness).
# The L4T base already ships OpenCV (CUDA build), but rqt-common-plugins pulls cv_bridge
# -> libopencv-dev from apt, whose files clash with it — so let dpkg overwrite them.
# python3-opencv is dropped here (cv2 comes from the pip opencv-python below / the base).
RUN apt update && apt install -y -o Dpkg::Options::="--force-overwrite" \
    ros-${ROS_DISTRO}-rqt \
    ros-${ROS_DISTRO}-rqt-common-plugins \
    ros-${ROS_DISTRO}-foxglove-bridge \
    gdb \
    python3-pip \
    python3-vcstool \
    libboost-python-dev \
    iproute2 \
    clang-format \
    openssh-client && \
    rm -rf /var/lib/apt/lists/*

# install python packages
RUN pip install --upgrade pip
# Remove conflicting distutils-installed package (sympy) to avoid installation issues
RUN rm -rf /usr/lib/python3/dist-packages/sympy* \
    /usr/local/lib/python3*/dist-packages/sympy* \
    /usr/lib/python3.*/dist-packages/sympy* || true
# Install PyTorch from the Jetson aarch64 CUDA index (torchvision/torchaudio unused here).
RUN pip install torch --index-url ${TORCH_INDEX_URL}
# Install other requirements
COPY ros_env/res/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

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
# auto source workspace overlay if one exists
RUN echo "\n if [ -f $OVERLAY_WS/install/setup.bash ]; then\n source $OVERLAY_WS/install/setup.bash\n fi\n" >> ~/.bashrc

WORKDIR $OVERLAY_WS
