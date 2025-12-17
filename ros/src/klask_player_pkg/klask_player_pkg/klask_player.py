import rclpy
import yaml
import math
import time
import torch
from rclpy.node import Node

class Player(Node):
    def __init__(self):
        super().__init__('klask_player')

        checkpoint = "/home/student/klask_rl/IsaacLab/logs/rl_games/klask/demo_agents/best_one/klask.pth"
        config = "/home/student/klask_rl/IsaacLab/planned_runs/sim_to_real/training_from_checkpoint.yaml"

        device = "cpu"

        with open(config, 'r') as file:
            agent_cfg = yaml.safe_load(file)

        if device is not None:
            agent_cfg["params"]["config"]["device"] = device
            agent_cfg["params"]["config"]["device_name"] = device

        rl_device = agent_cfg["params"]["config"]["device"]
        clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
        clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

        env = KlaskPlayEnv(config_path='/home/student/ros2_ws/src/klask_state_estimation/klask_state_estimation/rl_games_custom/rl_config/player_only.yaml')
        env = SimToReal(env)
        env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

        env_configurations.register("rlgpu", {"vecenv_type": "RlgWrapper", "env_creator": lambda **kwargs: env})

        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = args_cli.checkpoint
        print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")
        agent_cfg["params"]["config"]["num_actors"] = 1

        runner = Runner()
        runner.load(agent_cfg)

        agent: BasePlayer = runner.create_player()
        agent.restore(args_cli.checkpoint)
        agent.reset()

        #env.unwrapped.search_home = True
        obs = env.reset()
        rewards = []

        loop_count = 0
        start_time = time.time()
        T1 = 0
        T2 = 0
        actions = torch.zeros(4,device=rl_device)

        self.get_logger().info('Klask Player Node has been started.')


def main(args=None):
    rclpy.init(args=args)

    player = Player()

    try:
        rclpy.spin(player)
    except KeyboardInterrupt:
        pass
    finally:
        player.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()