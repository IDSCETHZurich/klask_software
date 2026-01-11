"""Launch file for complete state estimation system.

Launches camera node, image viewer, and state estimator with their respective
configuration files.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description for state estimation system."""

    # Declare launch arguments for config files
    state_estimator_params_arg = DeclareLaunchArgument(
        "state_estimator_params_file",
        default_value=PathJoinSubstitution(
            [
                FindPackageShare("klask_state_estimator_pkg"),
                "config",
                "state_estimator_params.yaml",
            ]
        ),
        description="Path to the state estimator parameters YAML file",
    )

    # State estimator node - launched after camera_node starts
    state_estimator_node = Node(
        package="klask_state_estimator_pkg",
        executable="state_estimator",
        name="state_estimator",
        output="screen",
        parameters=[LaunchConfiguration("state_estimator_params_file")],
        emulate_tty=True,
    )

    return LaunchDescription(
        [
            # Launch arguments
            state_estimator_params_arg,
            # Start state estimator
            state_estimator_node,
        ]
    )
