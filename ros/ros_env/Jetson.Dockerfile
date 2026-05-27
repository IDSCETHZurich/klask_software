# Jetson SDK image — mirrors SDK.Dockerfile but targets NVIDIA Jetson (aarch64 / L4T).
#
# Why this is a separate Dockerfile from SDK.Dockerfile:
#   1. Architecture: the regular SDK/Runtime images are built for linux/amd64 and the
#      desktop CUDA torch wheels (download.pytorch.org/whl/cu124) do not exist for
#      aarch64 and cannot drive the Jetson's integrated (Tegra) GPU.
#   2. Base image: we start from a Jetson-Linux (L4T) ROS image that ships the CUDA
#      runtime, instead of the generic ros:humble. CUDA driver libs are injected from
#      the host at *run* time by the NVIDIA container runtime (see run instructions).
#   3. PyTorch: installed from the Jetson aarch64 CUDA wheel index, not the cu124 index.
#
# Target host (from `dpkg -l | grep nvidia-l4t-core` -> 36.5.0):
#   L4T r36.5 / JetPack 6.2, Ubuntu 22.04 (jammy), CUDA 12.6, Python 3.10.>
#   ROS Humble is native to 22.04, so the rest of the SDK config carries over unchanged.
#
# Build (run ON the Jetson, with build context = the ros/ directory):
#   cd ros
#   docker build -f ros_env/Jetson.Dockerfile -t klask_ros_software_sdk:jetson .
#
# Run (the NVIDIA runtime mounts the host CUDA libs so the GPU is visible):
#   docker run --rm -it --runtime nvidia --network host \
#     -v "$PWD/src:/opt/ros/klask_ws/src" klask_ros_software_sdk:jetson
#   # On most JetPack installs the default docker runtime is already "nvidia".
#   # Set device: "cuda" in klask_player_pkg/config/player_params.yaml to use the GPU.

# Jetson-Linux ROS Humble base (ships CUDA runtime + ROS). Pick the tag matching your
# L4T: r36.4.0 is the closest published JetPack-6 tag and is compatible with r36.5 hosts.
# Alternatives: nvcr.io/nvidia/l4t-jetpack:r36.4.0 (then install ROS yourself).
ARG FROM_IMAGE=dustynv/ros:humble-ros-base-r36.4.0
# aarch64 CUDA torch wheel index for JetPack 6 / CUDA 12.6 (cp310).
# If you are actually on JetPack 5 (CUDA 11.4) use the jp5 index instead.
ARG TORCH_INDEX_URL=https://pypi.jetson-ai-lab.dev/jp6/cu126

FROM $FROM_IMAGE
ARG OVERLAY_WS=/opt/ros/klask_ws
ARG TORCH_INDEX_URL

# install ros packages (same set as SDK.Dockerfile; python3-vcstool added for robustness)
RUN apt update && apt install -y \
    ros-${ROS_DISTRO}-rqt \
    ros-${ROS_DISTRO}-rqt-common-plugins \
    ros-${ROS_DISTRO}-foxglove-bridge \
    gdb \
    python3-pip \
    python3-vcstool \
    python3-opencv \
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
# Install PyTorch with CUDA support — Jetson (aarch64) wheel, NOT the desktop cu124 wheel.
# Only `torch` is imported by the source, so torchvision/torchaudio are intentionally
# omitted (add them from the same index if you need them). The Jetson base image may
# already bundle a torch build; this line pins it to the wheel matching your JetPack.
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
