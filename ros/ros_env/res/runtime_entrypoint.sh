#!/bin/bash
set -e

# Default values
PLAYER="both"
WEIGHTS_FILENAME=""

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --player)
            PLAYER="$2"
            shift 2
            ;;
        --weights_filename)
            WEIGHTS_FILENAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--player left|right|both] [--weights_filename <filename>]"
            exit 1
            ;;
    esac
done

# Source ROS environment after parsing arguments
source /ros_entrypoint.sh

# Build launch command with parameters
LAUNCH_CMD="ros2 launch klask_player_pkg game_launch.py player:=\"$PLAYER\""
if [[ -n "$WEIGHTS_FILENAME" ]]; then
    LAUNCH_CMD="$LAUNCH_CMD weights_filename:=\"$WEIGHTS_FILENAME\""
fi

# Launch with specified parameters
exec bash -c "$LAUNCH_CMD"
