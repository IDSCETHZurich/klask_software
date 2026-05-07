# KLASK Robotic Software Stack

<img src="docs/res//imgs/general/klask_robotic_system.png" alt="KLASK Robotic System" width="500"/>

This repository contains the high-level ROS2 nodes that enable the KLASK robot to play autonomously. It provides a policy inference node and a state estimator node for real-time gameplay.

<!-- CI pipeline (Colcon build and test) -->
<a href="https://github.com/IDSCETHZurich/klask_software/actions/workflows/ci.yml">
  <img alt="CI pipeline" src="https://github.com/IDSCETHZurich/klask_software/actions/workflows/ci.yml/badge.svg">
</a>

<!-- Stars (social) -->
<a href="https://github.com/IDSCETHZurich/klask_software/stargazers">
  <img alt="GitHub stars" src="https://img.shields.io/github/stars/IDSCETHZurich/klask_software?style=social">
</a>

<!-- Contributors -->
<a href="https://github.com/IDSCETHZurich/klask_software/graphs/contributors">
  <img alt="Contributors" src="https://img.shields.io/github/contributors/IDSCETHZurich/klask_software">
</a>

<!-- Releases -->
<a href="https://github.com/IDSCETHZurich/klask_software/releases">
  <img alt="Release" src="https://img.shields.io/github/v/release/IDSCETHZurich/klask_software?sort=semver">
</a>

## What is the KLASK Robotic System

KLASK is a popular magnetic table game where players control magnetic pegs to hit a ball into the opponent's goal while avoiding obstacles. Researchers and students at the **Institute for Dynamic Systems and Control** (IDSC), ETH Zurich, have developed an autonomous robotic platform that can play KLASK using advanced reinforcement learning algorithms. The system provides the ability to test, benchmark and improve RL strategies in a real-world environment.

This repository contains the **software intelligence** of the KLASK robotic system: state estimation and policy inference nodes that enable autonomous gameplay. The hardware interface and low-level control are hosted in the separate [klask_hardware](https://github.com/IDSCETHZurich/klask_hardware) repository.

<img src="docs/res/diagrams/repo_overview.png" alt="KLASK Repo Overview" width="500"/>

## System Architecture

This repository contains the gameplay-side ROS2 package; the state estimator and the rest of the runtime nodes live in the `klask_hardware` repo.

### klask_player_pkg (Python)

Network policy inference for autonomous gameplay:

- Neural network policy that loads the trained model weights from Polybox
- Game state management to play repeated matches
- Configurable player behavior parameters

## Getting Started

If you want to run the full KLASK robotic system and you have already set up the mechanical gentry system you additionally only need [docker](https://www.docker.com/get-started) and [docker-compose](https://docs.docker.com/compose/install/) installed on your system. Then you only need to perform these two steps:

1. Copy the `docker-compose.yml` file from this repository to your machine.
2. Navigate to the folder containing the `docker-compose.yml` file and run:

    ```bash
    docker compose up -d
    ```

This will pull the pre-built runtime images and start both the hardware interface (motor control, camera) and the software intelligence (state estimation, policy inference) in coordinated Docker containers.

### Runtime Helper Script

If you want to run the software container only and may be want to play around with different configurations or versions, you can use our provided helper script.

Simply download the [`ros/ros_env/klask_prebuilt_runtime.sh`](https://github.com/IDSCETHZurich/klask_software/blob/main/ros/ros_env/klask_prebuilt_runtime.sh) script from this repo, which wraps some useful Docker commands to run the prebuilt container. Once you have the script, make sure it is executable:

```bash
chmod +x klask_prebuilt_runtime.sh
```

Then you can run the container with:

```bash
./klask_prebuilt_runtime.sh
```

This will pull the latest prebuilt Runtime Container from our GitHub Container Registry and start it up and launch the necessary ROS nodes to interact with the KLASK hardware.

The script also supports additional options:

```bash
# Run latest release
klask_prebuilt_runtime.sh

# Run specific version
klask_prebuilt_runtime.sh --tag v1.0.0

# Pull latest before running
klask_prebuilt_runtime.sh pull

# Connect to the running container via bash
klask_prebuilt_runtime.sh connect

# Stop the running container
klask_prebuilt_runtime.sh stop
```

You can additionally pass ROS arguments to specify if you want to launch the left, right, or both sides of the KLASK robot:

```bash
# The default is to launch both players
klask_prebuilt_runtime.sh

# Launch right player
klask_prebuilt_runtime.sh --player right
```

All the available options can be listed with:

```bash
klask_prebuilt_runtime.sh --help
```

### Local Setup

If you want to dive in further you can follow the detailed setup instructions in the [Development Setup Guide](https://idscethzurich.github.io/klask_hardware/tutorials/03-dev-setup/) to build the software from source, set up a development environment, and contribute to the codebase. This documentation section is provided by the `klask_hardware` repository but the steps are identical for both repositories.

## Roadmap

We track work in GitHub issues and milestones.

## Contributing

We welcome contributions! Whether you're fixing bugs, adding features, or enhancing documentation:

1. **Fork** the repository and create a feature branch
2. **Develop** following our [contribution guidelines](https://idscethzurich.github.io/klask_hardware/contribution/contributing/)
3. **Test** your changes thoroughly
4. **Submit a PR** with a clear description and context

## Maintainers

This project is mostly maintained by:

- [Aswin](https://github.com/akrv) - Lead Researcher
- [Tobias](https://github.com/MeierTobias) - Student

## Citing

If you use this work in an academic context, please cite the following publication:

- Aswin Karthik Ramachandran Venkatapathy, Jona Schulz, Maurus Derungs, Carlo Angelini, Tobias Meier, Raffaello D’Andrea, **"KlaskTron: An Open-Source Platform for Physical Adversarial Multi-Agent RL"**, 2026. ([PDF](https://openreview.net/pdf?id=UaLgID9r1i))

    ```bibtex
    @inproceedings{
      anonymous2026klasktron,
      title={KlaskTron: An Open-Source Platform for Physical Adversarial Multi-Agent {RL}},
      author={Anonymous},
      booktitle={Robotics: Science and Systems 2026},
      year={2026},
      url={https://openreview.net/forum?id=UaLgID9r1i}
    }
    ```

## License

- ROS2 code and software is licensed under [**AGPL-3.0**](LICENSE-AGPL-3.0)
- Documentation and tutorials are under [**CC BY 4.0**](LICENSE-CC-BY-4.0)

### Third-party Components

- Dependencies and libraries retain their original licenses

## Acknowledgements

- ETH Zurich's **Institute for Dynamic Systems and Control** for project support
- All students and researchers involved in the KLASK project
- The ROS2 and open-source robotics community
- All contributors and maintainers who make this project possible
