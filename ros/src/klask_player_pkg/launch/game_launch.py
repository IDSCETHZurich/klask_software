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

    nodes = []

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

    Examples:
        ros2 launch klask_player_pkg game_launch.py player:=left
        ros2 launch klask_player_pkg game_launch.py player:=right
        ros2 launch klask_player_pkg game_launch.py player:=both
    """
    # Declare launch arguments
    player_arg = DeclareLaunchArgument(
        "player",
        default_value="both",
        description="Which player to launch: left, right, or both",
    )

    return LaunchDescription([player_arg, OpaqueFunction(function=launch_setup)])
