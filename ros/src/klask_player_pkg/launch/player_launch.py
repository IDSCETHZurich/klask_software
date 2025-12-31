import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Launch the Klask policy inference node with parameters."""
    
    # Get the package directory
    pkg_dir = get_package_share_directory('klask_player_pkg')
    
    # Path to the parameter file
    params_file = os.path.join(pkg_dir, 'config', 'player_params.yaml')
    
    # Create the player node
    player_node = Node(
        package='klask_player_pkg',
        executable='policy_inference_node',
        name='klask_policy_inference_node',
        parameters=[params_file],
        output='screen',
        emulate_tty=True,
    )
    
    return LaunchDescription([
        player_node
    ])
