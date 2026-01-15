#!/bin/bash
set -e

# Default values
PLAYER="both"

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --player)
            PLAYER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--player left|right|both]"
            exit 1
            ;;
    esac
done

# Source ROS environment after parsing arguments
source /ros_entrypoint.sh

# Launch with specified parameters
exec ros2 launch klask_player_pkg game_launch.py player:="$PLAYER"
