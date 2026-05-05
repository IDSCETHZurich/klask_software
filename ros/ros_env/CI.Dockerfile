FROM ros:humble-ros-base
ARG OVERLAY_WS=/opt/ros/klask_ws

# install ros package
RUN apt update && apt install -y \
    ros-${ROS_DISTRO}-foxglove-bridge \
    python3-pip \
    python3-opencv \
    libboost-python-dev \
    clang-format \
    && rm -rf /var/lib/apt/lists/*

# install python packages
RUN pip install --upgrade pip
# Remove conflicting distutils-installed package (sympy) to avoid installation issues
RUN rm -rf /usr/lib/python3/dist-packages/sympy* \
    /usr/local/lib/python3*/dist-packages/sympy* \
    /usr/lib/python3.*/dist-packages/sympy* || true
# Install PyTorch with CUDA support
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# Install other requirements
COPY ros_env/res/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# create a workspace directory
RUN mkdir -p $OVERLAY_WS/src
RUN mkdir -p $OVERLAY_WS/third_party

# import third party repos into workspace
COPY ros_env/res/third_party.repos /tmp/third_party.repos
RUN vcs import $OVERLAY_WS/third_party < /tmp/third_party.repos
# Ignore packages that are not needed / cause trouble in the CI 
RUN touch $OVERLAY_WS/third_party/klask_hardware/ros/src/klask_imaging_pkg/COLCON_IGNORE || true
RUN touch $OVERLAY_WS/third_party/klask_hardware/ros/src/klask_motor_commander_pkg/COLCON_IGNORE || true
RUN touch $OVERLAY_WS/third_party/klask_hardware/ros/src/klask_sprite_generator_pkg/COLCON_IGNORE || true
RUN touch $OVERLAY_WS/third_party/klask_hardware/ros/src/klask_state_estimator_pkg/COLCON_IGNORE || true
RUN touch $OVERLAY_WS/third_party/klask_hardware/ros/src/klask_system_id_pkg/COLCON_IGNORE || true

# auto source ROS setup.bash
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc

WORKDIR $OVERLAY_WS