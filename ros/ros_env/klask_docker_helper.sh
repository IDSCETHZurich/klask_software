#!/bin/bash

# Klask Docker management script for SDK and Runtime containers
# Usage: ./klask_docker_helper.sh [--runtime|-r] <command>

# To enable tab completion, add this to your ~/.bashrc 
# or execute it in your shell for to have it for the current session:
#     source /path/to/klask_docker_helper.sh --completion

NS="software"

# Bash completion support
if [[ "$1" == "--completion" ]]; then
    _klask_docker_completions() {
        local commands="build run connect stop status --runtime -r"
        COMPREPLY=($(compgen -W "$commands" -- "${COMP_WORDS[1]}"))
    }
    # Register for various ways the script might be called
    complete -F _klask_docker_completions klask_docker_helper.sh
    complete -F _klask_docker_completions ./klask_docker_helper.sh
    complete -F _klask_docker_completions ./ros/ros_env/klask_docker_helper.sh
    complete -F _klask_docker_completions ros/ros_env/klask_docker_helper.sh
    return 0 2>/dev/null || exit 0
fi

set -e

# Get the directory of this script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Parse mode flag and configure
MODE="sdk"  # Default mode
MODE_FLAG=""
if [[ "$1" == "--runtime" ]] || [[ "$1" == "-r" ]]; then
    MODE="runtime"
    MODE_FLAG="--runtime "
    shift  # Remove the flag from arguments
fi

# Configuration based on mode
if [[ "$MODE" == "runtime" ]]; then
    IMAGE_NAME="klask_ros_${NS}_runtime"
    DOCKERFILE="Runtime.Dockerfile"
    MODE_DISPLAY="Runtime"
else
    IMAGE_NAME="klask_ros_${NS}_sdk"
    DOCKERFILE="SDK.Dockerfile"
    MODE_DISPLAY="SDK"
fi

TAG="local"
CONTAINER_NAME="${IMAGE_NAME}_container_${TAG}"
VIDEO_DEVICE="/dev/video0"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_usage() {
    echo "Usage: $0 [--runtime|-r] <command> [args...]"
    echo ""
    echo "Flags:"
    echo "  --runtime, -r  Use Runtime container instead of SDK (default: SDK)"
    echo ""
    echo "Commands:"
    echo "  build    Build the Docker image"
    echo "  run      Run the container (detached)"
    echo "           For Runtime mode, additional args are passed to the launch file"
    echo "  connect  Connect to the running container"
    echo "  stop     Stop the container"
    echo "  status   Show container status"
    echo ""
    echo "Examples:"
    echo "  $0 build                              # Build SDK image"
    echo "  $0 --runtime build                    # Build Runtime image"
    echo "  $0 -r run                             # Run Runtime container with defaults"
    echo "  $0 -r run --player left               # Run with left player only"
    echo "  $0 -r run --player right --no-viewer  # Run right player without viewer"
    echo "  $0 -r run --no-viewer                 # Run both players without viewer"
    echo ""
}

cmd_build() {
    echo -e "${GREEN}Building ${MODE_DISPLAY} Docker image...${NC}"
    
    # TODO: after the klask_hardware repo is public you can remove the --ssh flag
    # docker build -t "${IMAGE_NAME}:${TAG}" -f "$SCRIPT_DIR/$DOCKERFILE" "$SCRIPT_DIR/.."
    docker build --ssh default -t "${IMAGE_NAME}:${TAG}" -f "$SCRIPT_DIR/$DOCKERFILE" "$SCRIPT_DIR/.."

    echo -e "${GREEN}${MODE_DISPLAY} build complete!${NC}"
}

cmd_run() {
    # Check if container is already running
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${YELLOW}Container '${CONTAINER_NAME}' is already running.${NC}"
        echo "Use '$0 ${MODE_FLAG}connect' to attach to it, or '$0 ${MODE_FLAG}stop' to stop it first."
        return 1
    fi

    echo -e "${GREEN}Starting ${MODE_DISPLAY} container...${NC}"

    if [[ "$MODE" == "runtime" ]]; then
        # Runtime container: pre-built workspace, minimal volumes
        # Additional arguments are passed to the container entrypoint
        if [ $# -gt 0 ]; then
            echo -e "${GREEN}Passing arguments to container: $@${NC}"
        fi
        xhost +local:root
        docker run -it --rm \
            --env="DISPLAY" \
            --env="QT_X11_NO_MITSHM=1" \
            --env="ROS_DOMAIN_ID=0" \
            --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
            --volume="${CONTAINER_NAME}_nn_weights:/opt/ros/klask_ws/nn_weights:rw" \
            --gpus=all \
            --name="${CONTAINER_NAME}" \
            "${IMAGE_NAME}:${TAG}" "$@"
    else
        # SDK container: development mode with source mounts
        xhost +local:root
        docker run -it -d --rm \
            --env="DISPLAY" \
            --env="QT_X11_NO_MITSHM=1" \
            --env="ROS_DOMAIN_ID=0" \
            --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
            --volume="$SCRIPT_DIR/../src:/opt/ros/klask_ws/src:rw" \
            --volume="$SCRIPT_DIR/../.vscode:/opt/ros/klask_ws/.vscode:rw" \
            --volume="$SCRIPT_DIR/../.clang-format:/opt/ros/klask_ws/.clang-format:rw" \
            --volume="$SCRIPT_DIR/../.flake8:/opt/ros/klask_ws/.flake8:rw" \
            --volume="$SCRIPT_DIR/../pyproject.toml:/opt/ros/klask_ws/pyproject.toml:rw" \
            --volume="${CONTAINER_NAME}_build:/opt/ros/klask_ws/build:rw" \
            --volume="${CONTAINER_NAME}_install:/opt/ros/klask_ws/install:rw" \
            --volume="${CONTAINER_NAME}_log:/opt/ros/klask_ws/log:rw" \
            --volume="${CONTAINER_NAME}_nn_weights:/opt/ros/klask_ws/nn_weights:rw" \
            --gpus=all \
            --name="${CONTAINER_NAME}" \
            "${IMAGE_NAME}:${TAG}"
    fi

    echo -e "${GREEN}${MODE_DISPLAY} container started! Use '$0 ${MODE_FLAG}connect' to attach.${NC}"
}

cmd_connect() {
    # Check if container is running
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${RED}Container '${CONTAINER_NAME}' is not running.${NC}"
        echo "Use '$0 ${MODE_FLAG}run' to start it first."
        return 1
    fi

    echo -e "${GREEN}Connecting to ${MODE_DISPLAY} container...${NC}"
    
    WORKDIR="/opt/ros/klask_ws"

    docker exec -it -w "$WORKDIR" "${CONTAINER_NAME}" bash
}

cmd_stop() {
    # Check if container is running
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${YELLOW}Container '${CONTAINER_NAME}' is not running.${NC}"
        return 0
    fi

    echo -e "${GREEN}Stopping ${MODE_DISPLAY} container...${NC}"

    docker stop "${CONTAINER_NAME}"
    
    echo -e "${GREEN}Container stopped!${NC}"
}

cmd_status() {
    echo -e "${GREEN}${MODE_DISPLAY} Container Status:${NC}"
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "  Status: ${GREEN}Running${NC}"
        docker ps --filter "name=${CONTAINER_NAME}" --format "  ID: {{.ID}}\n  Created: {{.RunningFor}}\n  Ports: {{.Ports}}"
    else
        echo -e "  Status: ${RED}Not running${NC}"
    fi
    
    echo ""
    echo -e "${GREEN}${MODE_DISPLAY} Image:${NC}"
    if docker images "${IMAGE_NAME}:${TAG}" --format "{{.Repository}}" | grep -q "${IMAGE_NAME}"; then
        docker images "${IMAGE_NAME}:${TAG}" --format "  Repository: {{.Repository}}:{{.Tag}}\n  ID: {{.ID}}\n  Size: {{.Size}}\n  Created: {{.CreatedSince}}"
    else
        echo -e "  ${RED}Image not found. Run '$0 ${MODE_FLAG}build' first.${NC}"
    fi
}

# Main
if [ $# -eq 0 ]; then
    print_usage
    exit 1
fi

COMMAND="$1"
shift  # Remove command from arguments

case "$COMMAND" in
    build)
        cmd_build
        ;;
    run)
        cmd_run "$@"
        ;;
    connect)
        cmd_connect
        ;;
    stop)
        cmd_stop
        ;;
    status)
        cmd_status
        ;;
    -h|--help|help)
        print_usage
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        echo ""
        print_usage
        exit 1
        ;;
esac
