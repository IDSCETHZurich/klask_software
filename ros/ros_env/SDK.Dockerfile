FROM ros:humble-ros-base
ARG OVERLAY_WS=/opt/ros/klask_ws

# install ros package
RUN apt update && apt install -y \
    ros-${ROS_DISTRO}-rqt \
    ros-${ROS_DISTRO}-rqt-common-plugins \
    ros-${ROS_DISTRO}-foxglove-bridge \
    gdb \
    python3-pip \
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
# Install PyTorch with CUDA support
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# Install other requirements
COPY ros_env/requirements.txt /tmp/requirements.txt
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
COPY ros_env/third_party.repos /tmp/third_party.repos
# TODO: Remove this entire RUN block once klask_hardware repo is public
RUN --mount=type=ssh \
    mkdir -p ~/.ssh && \
    echo "Host github.com-MeierTobias" > ~/.ssh/config && \
    echo "  HostName github.com" >> ~/.ssh/config && \
    echo "  User git" >> ~/.ssh/config && \
    ssh-keyscan github.com >> ~/.ssh/known_hosts && \
    vcs import $OVERLAY_WS/third_party < /tmp/third_party.repos
# TODO: Once repo is public, replace above RUN block with:
# RUN vcs import $OVERLAY_WS/third_party < /tmp/third_party.repos
# Ignore packages that are not needed / cause trouble in the SDK 
RUN touch $OVERLAY_WS/third_party/klask_hardware/ros/src/klask_imaging_pkg/COLCON_IGNORE || true
RUN touch $OVERLAY_WS/third_party/klask_hardware/ros/src/klask_motor_commander_pkg/COLCON_IGNORE || true
RUN touch $OVERLAY_WS/third_party/klask_hardware/ros/src/klask_state_estimation_pkg/COLCON_IGNORE || true


# auto source ROS setup.bash
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
# auto source workspace overlay if one exists
RUN echo "\n if [ -f $OVERLAY_WS/install/setup.bash ]; then\n source $OVERLAY_WS/install/setup.bash\n fi\n" >> ~/.bashrc

WORKDIR $OVERLAY_WS