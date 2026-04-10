"""Unified launch file for complete Klask game system.

Launches state estimator and player node(s) with configurable player selection.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def launch_setup(context, *args, **kwargs):
    """Setup function to conditionally launch nodes based on player argument."""
    # Get package directories
    player_pkg_dir = get_package_share_directory("klask_player_pkg")
    state_estimator_pkg_dir = get_package_share_directory("klask_state_estimator_pkg")

    # Paths to parameter files
    player_params_file = os.path.join(player_pkg_dir, "config", "player_params.yaml")
    state_estimator_params_file = os.path.join(state_estimator_pkg_dir, "config", "state_estimator_params.yaml")

    # Get the player argument value
    player = LaunchConfiguration("player").perform(context)

    # Get optional argument values
    weights_filename = LaunchConfiguration("weights_filename").perform(context)
    agent_type = LaunchConfiguration("agent_type").perform(context)

    nodes = []

    # Prepare additional parameters (only include if provided)
    additional_params = {}
    if weights_filename:
        additional_params["weights_filename"] = weights_filename
    if agent_type:
        additional_params["agent_type"] = agent_type

    # Launch state estimator
    state_estimator_node = Node(
        package="klask_state_estimator_pkg",
        executable="state_estimator",
        name="state_estimator",
        parameters=[state_estimator_params_file],
        output="screen",
        emulate_tty=True,
    )
    nodes.append(state_estimator_node)

    # Launch left player
    if player in ["left", "both"]:
        left_player_node = Node(
            package="klask_player_pkg",
            executable="player_node",
            name="klask_player_left",
            parameters=[
                player_params_file,
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
                player_params_file,
                {"player_side": "right", "cmd_vel_topic": "cmd_vel/right_player"},
                additional_params,
            ],
            output="screen",
            emulate_tty=True,
        )
        nodes.append(right_player_node)

    return nodes


def generate_launch_description():
    """Launch the complete Klask game system.

    Launch arguments:
        player: Which player to launch ('left', 'right', or 'both'). Default: 'both'
        weights_filename: Optional weights filename to override config file value

    Examples:
        ros2 launch klask_player_pkg game_launch.py player:=left
        ros2 launch klask_player_pkg game_launch.py player:=right
        ros2 launch klask_player_pkg game_launch.py player:=both
        ros2 launch klask_player_pkg game_launch.py player:=both weights_filename:=klask_ac_nn_v0.0.pth
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

    agent_type_arg = DeclareLaunchArgument(
        "agent_type",
        default_value="",
        description="Optional: Override agent type (ppo or dreamer)",
    )

    return LaunchDescription([player_arg, weights_filename_arg, agent_type_arg, OpaqueFunction(function=launch_setup)])
