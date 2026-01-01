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
    camera_params_arg = DeclareLaunchArgument(
        "camera_params_file",
        default_value=PathJoinSubstitution(
            [FindPackageShare("klask_imaging_pkg"), "config", "camera_node_params.yaml"]
        ),
        description="Path to the camera node parameters YAML file",
    )

    image_viewer_params_arg = DeclareLaunchArgument(
        "image_viewer_params_file",
        default_value=PathJoinSubstitution(
            [
                FindPackageShare("klask_imaging_pkg"),
                "config",
                "image_viewer_params.yaml",
            ]
        ),
        description="Path to the image viewer parameters YAML file",
    )

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

    # Camera node
    camera_node = Node(
        package="klask_imaging_pkg",
        executable="camera_node",
        name="camera_node",
        output="screen",
        parameters=[LaunchConfiguration("camera_params_file")],
        emulate_tty=True,
    )

    # Image viewer node
    image_viewer_node = Node(
        package="klask_imaging_pkg",
        executable="image_viewer",
        name="image_viewer",
        output="screen",
        parameters=[LaunchConfiguration("image_viewer_params_file")],
        emulate_tty=True,
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

    # Event handler to start state estimator after camera node starts
    start_state_estimator_after_camera = RegisterEventHandler(
        OnProcessStart(
            target_action=camera_node,
            on_start=[state_estimator_node],
        )
    )

    return LaunchDescription(
        [
            # Launch arguments
            camera_params_arg,
            image_viewer_params_arg,
            state_estimator_params_arg,
            # Start camera and image viewer first
            camera_node,
            image_viewer_node,
            # Start state estimator after camera starts
            start_state_estimator_after_camera,
        ]
    )
