"""Launch file to start Klask player nodes based on specified arguments."""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def launch_setup(context, *args, **kwargs):
    """Setup function to conditionally launch nodes based on player argument."""
    # Get the package directory
    pkg_dir = get_package_share_directory("klask_player_pkg")

    # Path to the parameter file
    params_file = os.path.join(pkg_dir, "config", "player_params.yaml")

    # Get the player argument value
    player = LaunchConfiguration("player").perform(context)

    # Get the weights_filename argument value (if provided)
    weights_filename = LaunchConfiguration("weights_filename").perform(context)

    nodes = []

    # Prepare additional parameters (only include weights_filename if provided)
    additional_params = {}
    if weights_filename:
        additional_params["weights_filename"] = weights_filename

    # Launch left player
    if player in ["left", "both"]:
        left_player_node = Node(
            package="klask_player_pkg",
            executable="player_node",
            name="klask_player_left",
            parameters=[
                params_file,
                {"player_side": "left", "cmd_vel_topic": "cmd_vel/left_player"},
                additional_params,
            ],
            output="screen",
            emulate_tty=True,
        )
        nodes.append(left_player_node)

    # Launch right player
    if player in ["right", "both"]:
        right_player_node = Node(
            package="klask_player_pkg",
            executable="player_node",
            name="klask_player_right",
            parameters=[
                params_file,
                {"player_side": "right", "cmd_vel_topic": "cmd_vel/right_player"},
                additional_params,
            ],
            output="screen",
            emulate_tty=True,
        )
        nodes.append(right_player_node)

    return nodes


def generate_launch_description():
    """Launch the Klask policy inference node(s) with parameters.

    Launch arguments:
        player: Which player to launch ('left', 'right', or 'both'). Default: 'left'
        weights_filename: Optional weights filename to override config file value

    Examples:
        ros2 launch klask_player_pkg player_launch.py player:=left
        ros2 launch klask_player_pkg player_launch.py player:=right
        ros2 launch klask_player_pkg player_launch.py player:=both
        ros2 launch klask_player_pkg player_launch.py player:=left weights_filename:=klask_ac_nn_v0.0.pth
    """
    # Declare launch arguments
    player_arg = DeclareLaunchArgument(
        "player",
        default_value="both",
        description="Which player to launch: left, right, or both",
    )

    weights_filename_arg = DeclareLaunchArgument(
        "weights_filename",
        default_value="",
        description="Optional: Override weights filename from config file",
    )

    return LaunchDescription([player_arg, weights_filename_arg, OpaqueFunction(function=launch_setup)])
