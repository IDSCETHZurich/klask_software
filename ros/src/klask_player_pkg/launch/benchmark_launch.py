"""Launch file to start the inference benchmark node."""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def launch_setup(context, *args, **kwargs):
    """Setup function to launch the benchmark node with parameter overrides."""
    pkg_dir = get_package_share_directory("klask_player_pkg")
    params_file = os.path.join(pkg_dir, "config", "benchmark_params.yaml")

    # Get launch argument values
    weights_filename = LaunchConfiguration("weights_filename").perform(context)
    device = LaunchConfiguration("device").perform(context)

    # Build parameter overrides for non-empty launch arguments
    additional_params = {}
    if weights_filename:
        additional_params["weights_filename"] = weights_filename
    if device:
        additional_params["device"] = device

    benchmark_node = Node(
        package="klask_player_pkg",
        executable="benchmark_node",
        name="inference_benchmark",
        parameters=[params_file, additional_params],
        output="screen",
        emulate_tty=True,
    )

    return [benchmark_node]


def generate_launch_description():
    """Launch the inference benchmark node.

    Launch arguments:
        weights_filename: Override weights filename from config file
        device: Override device (cpu or cuda)

    Examples:
        ros2 launch klask_player_pkg benchmark_launch.py
        ros2 launch klask_player_pkg benchmark_launch.py weights_filename:=klask_ac_nn_v1.0.pth
        ros2 launch klask_player_pkg benchmark_launch.py weights_filename:=klask_ac_nn_v1.0.pth device:=cuda
    """
    weights_filename_arg = DeclareLaunchArgument(
        "weights_filename",
        default_value="",
        description="Optional: Override weights filename from config file",
    )

    device_arg = DeclareLaunchArgument(
        "device",
        default_value="",
        description="Optional: Override device (cpu or cuda)",
    )

    return LaunchDescription([weights_filename_arg, device_arg, OpaqueFunction(function=launch_setup)])
