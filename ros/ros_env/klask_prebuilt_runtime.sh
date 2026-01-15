#!/bin/bash

# Klask Runtime Container Launcher
# Runs the Klask runtime container from GitHub Container Registry
# Usage: ./klask_prebuilt_runtime.sh [command] [options]

set -e

# Configuration
GHCR_IMAGE_BASE="ghcr.io/idscethzurich/klask_software/runtime"
DEFAULT_TAG="latest"
IMAGE_TAG="${DEFAULT_TAG}"
CONTAINER_NAME="klask_software_runtime_container"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_usage() {
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  run          Run the container (default if no command specified)"
    echo "  connect      Connect to a running container"
    echo "  stop         Stop a running container"
    echo "  pull         Pull the image from GHCR (can be combined with run)"
    echo ""
    echo "Options:"
    echo "  --tag TAG    Specify image tag (default: ${DEFAULT_TAG})"
    echo ""
    echo "Launch Arguments (for run command, passed to container):"
    echo "  --player left|right|both  Specify which player(s) to start (default: both)"
    echo ""
    echo "Examples:"
    echo "  $0                                  # Run with defaults (latest tag, both players)"
    echo "  $0 connect                          # Connect to running container"
    echo "  $0 stop                             # Stop running container"
    echo "  $0 pull                             # Pull latest image and run"
    echo "  $0 --tag v1.0.0                     # Run specific version"
    echo "  $0 pull --tag v1.0.0                # Pull and run specific version"
    echo "  $0 --tag dev --player left          # Run dev tag with left player only"
    echo ""
    echo "Current GHCR Image Base: ${GHCR_IMAGE_BASE}"
    echo ""
}

cmd_pull() {
    echo -e "${GREEN}Pulling runtime image from GHCR: ${FULL_IMAGE}${NC}"
    docker pull "${FULL_IMAGE}"
    echo -e "${GREEN}Image pull complete!${NC}"
}

cmd_run() {
    # Check if container is already running
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${YELLOW}Container '${CONTAINER_NAME}' is already running.${NC}"
        echo "Stop it first with: docker stop ${CONTAINER_NAME}"
        return 1
    fi

    echo -e "${GREEN}Starting Klask runtime container...${NC}"
    echo -e "Using image: ${FULL_IMAGE}"
    
    if [ $# -gt 0 ]; then
        echo -e "${GREEN}Passing arguments to container: $@${NC}"
    fi
    
    # Enable X11 forwarding
    xhost +local:root
    
    # Run container with GHCR image
    docker run -it -d --rm \
        --env="DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --env="ROS_DOMAIN_ID=0" \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        --volume="${CONTAINER_NAME}_nn_weights:/opt/ros/klask_ws/nn_weights:rw" \
        --gpus=all \
        --name="${CONTAINER_NAME}" \
        "${FULL_IMAGE}" "$@"
    
    echo -e "${GREEN}Runtime container started!${NC}"
    echo -e "Container name: ${CONTAINER_NAME}"
    echo -e "To view logs: docker logs -f ${CONTAINER_NAME}"
    echo -e "To connect: docker exec -it ${CONTAINER_NAME} bash"
    echo -e "To stop: docker stop ${CONTAINER_NAME}"
}

cmd_connect() {
    # Check if container is running
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${RED}Container '${CONTAINER_NAME}' is not running.${NC}"
        echo "Start it first with: $0 run"
        return 1
    fi

    echo -e "${GREEN}Connecting to runtime container...${NC}"
    
    WORKDIR="/opt/ros/klask_ws"

    docker exec -it -w "$WORKDIR" "${CONTAINER_NAME}" bash
}

cmd_stop() {
    # Check if container is running
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${YELLOW}Container '${CONTAINER_NAME}' is not running.${NC}"
        return 0
    fi

    echo -e "${GREEN}Stopping runtime container...${NC}"

    docker stop "${CONTAINER_NAME}"
    
    echo -e "${GREEN}Container stopped!${NC}"
}

# Main
if [ $# -eq 0 ]; then
    # No arguments, just run with defaults
    FULL_IMAGE="${GHCR_IMAGE_BASE}:${IMAGE_TAG}"
    cmd_run
    exit 0
fi

# Check for help
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]] || [[ "$1" == "help" ]]; then
    print_usage
    exit 0
fi

# Check for direct commands (connect, stop)
if [[ "$1" == "connect" ]]; then
    cmd_connect
    exit 0
fi

if [[ "$1" == "stop" ]]; then
    cmd_stop
    exit 0
fi

# Parse arguments for run command
PULL_IMAGE=false
RUN_CONTAINER=false
LAUNCH_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        run)
            RUN_CONTAINER=true
            shift
            ;;
        pull)
            PULL_IMAGE=true
            shift
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        *)
            # All other arguments go to launch
            LAUNCH_ARGS+=("$1")
            shift
            ;;
    esac
done

# Set the full image name after parsing arguments
FULL_IMAGE="${GHCR_IMAGE_BASE}:${IMAGE_TAG}"

# Pull image if requested
if [ "$PULL_IMAGE" = true ]; then
    cmd_pull
fi

# If only pull was requested, exit here unless run was explicitly specified or there are launch args
if [ "$PULL_IMAGE" = true ] && [ "$RUN_CONTAINER" = false ] && [ ${#LAUNCH_ARGS[@]} -eq 0 ]; then
    exit 0
fi

# Run container with launch arguments
cmd_run "${LAUNCH_ARGS[@]}"
