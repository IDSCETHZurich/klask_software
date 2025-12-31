import rclpy
import torch
import numpy as np
from rclpy.node import Node
from .policy_network import PolicyNetwork
from klask_interfaces.msg import State
from geometry_msgs.msg import Twist


class Player(Node):
    """ROS2 node for Klask policy inference."""

    def __init__(self):
        super().__init__("klask_policy_inference_node")

        # Declare ROS parameters
        self.declare_parameter(
            "checkpoint_path", "src/klask_player_pkg/klask_player_pkg/klask.pth"
        )
        self.declare_parameter("device", "cpu")
        self.declare_parameter("board_width", 0.32)
        self.declare_parameter("board_height", 0.44)
        self.declare_parameter("player_side", "left")
        self.declare_parameter("state_topic", "/board_state")
        self.declare_parameter("cmd_vel_topic", "cmd_vel/left_player_checked")
        self.declare_parameter("subscription_queue_size", 1)
        self.declare_parameter("publisher_queue_size", 1)

        # Get parameters
        checkpoint_path = self.get_parameter("checkpoint_path").value
        self.device = self.get_parameter("device").value
        board_width = self.get_parameter("board_width").value
        board_height = self.get_parameter("board_height").value
        self.player_side = self.get_parameter("player_side").value
        state_topic = self.get_parameter("state_topic").value
        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        sub_queue_size = self.get_parameter("subscription_queue_size").value
        pub_queue_size = self.get_parameter("publisher_queue_size").value

        # Board dimensions for coordinate centering
        self.board_dimensions = (board_width, board_height)

        # Initialize network
        self.policy_net = PolicyNetwork()
        self.load_checkpoint(checkpoint_path)
        self.policy_net.to(self.device)
        self.policy_net.eval()

        # ROS2 setup
        self.subscription = self.create_subscription(
            State,
            state_topic,
            self.observation_callback,
            sub_queue_size,
        )

        self.publisher = self.create_publisher(Twist, cmd_vel_topic, pub_queue_size)

        # State variables
        self.latest_obs = None

        self.get_logger().info(
            f"Policy inference node initialized with checkpoint: {checkpoint_path}"
        )
        self.get_logger().info(f"Using device: {self.device}")
        self.get_logger().info(
            f"Board dimensions: {self.board_dimensions[0]}m x {self.board_dimensions[1]}m"
        )
        self.get_logger().info(f"Player side: {self.player_side}")
        self.get_logger().info(f"Subscribed to: {state_topic}")
        self.get_logger().info(f"Publishing to: {cmd_vel_topic}")

    def load_checkpoint(self, checkpoint_path):
        """Load model weights from checkpoint file."""
        self.get_logger().info(f"Loading checkpoint from: {checkpoint_path}")

        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )  # TODO: maybe remove weights_only once everything is working and we are able to regenerate the ccheckpoint from the original training

            # RL-Games saves checkpoints with 'model' key
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

            # Map checkpoint keys to our network structure
            model_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("a2c_network."):
                    # Remove "a2c_network." prefix to match our structure
                    new_key = key.replace("a2c_network.", "")
                    model_state_dict[new_key] = value

            # Load the mapped state dict
            missing_keys, unexpected_keys = self.policy_net.load_state_dict(
                model_state_dict, strict=False
            )

            if missing_keys:
                self.get_logger().warn(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                self.get_logger().warn(f"Unexpected keys: {unexpected_keys}")

            self.get_logger().info("Checkpoint loaded successfully")

        except Exception as e:
            self.get_logger().error(f"Failed to load checkpoint: {e}")
            raise

    def observation_callback(self, msg: State):
        """Process incoming observations and publish actions."""
        # Extract observations from message (already in engineering units: meters and m/s)
        obs_with_goals = self.map_observations(msg)

        # Center coordinates to board origin
        obs_centered = self.center_coordinates(obs_with_goals)

        # Compute additional features
        obs_full = self.add_additional_features(obs_centered)

        # Get action from policy
        action = self.get_action(obs_full)

        # Publish action
        self.publish_action(action)

    def map_observations(self, msg: State):
        """Extract observations from StampedPolygon message."""

        # TODO: Map observations depending on how is playing (left/right)
        obs = np.array(
            [
                msg.right_peg.position.y,
                msg.right_peg.position.x,  # right peg pos
                msg.right_peg.velocity.y,
                msg.right_peg.velocity.x,  # right peg vel
                msg.left_peg.position.y,
                msg.left_peg.position.x,  # left peg pos
                msg.left_peg.velocity.y,
                msg.left_peg.velocity.x,  # left peg vel
                msg.ball.position.y,
                msg.ball.position.x,  # ball pos
                msg.ball.velocity.y,
                msg.ball.velocity.x,  # ball vel
                msg.right_goal_pos.y,
                msg.right_goal_pos.x,  # right goal pos
                msg.left_goal_pos.y,
                msg.left_goal_pos.x,  # left goal pos
            ],
            dtype=np.float32,
        )

        return obs

    def center_coordinates(self, obs):
        """Center all position coordinates to board origin.

        Shifts all positions by half the board dimensions so that (0,0) is at the board center.
        Velocities are not affected.
        """
        obs_centered = obs.copy()

        # Center Y positions (indices: 0, 4, 8, 12, 14)
        obs_centered[[0, 4, 8, 12, 14]] -= self.board_dimensions[0] / 2.0

        # Center X positions (indices: 1, 5, 9, 13, 15)
        obs_centered[[1, 5, 9, 13, 15]] -= self.board_dimensions[1] / 2.0

        return obs_centered

    def add_additional_features(self, obs):
        """Add geometric features (angles and distances) to observations.

        Input: 16 values [player_pos, player_vel, opp_pos, opp_vel, ball_pos, ball_vel, goal_1_pos, goal_2_pos]
        Output: 20 values [player_pos, player_vel, opp_pos, opp_vel, ball_pos, ball_vel, 8 features]

        Note: All coordinates are in engineering units (meters and m/s).
        """
        # Extract components (already in sim coordinates)
        player_pos = obs[0:2]
        opponent_pos = obs[4:6]
        ball_pos = obs[8:10]
        goal_1_pos = obs[12:14]
        goal_2_pos = obs[14:16]

        # Compute vectors
        vec_to_opp_goal = goal_2_pos - player_pos
        vec_to_ball = ball_pos - player_pos
        vec_opp_to_goal = goal_1_pos - opponent_pos
        vec_ball_to_opp = ball_pos - opponent_pos
        vec_opp_to_player = opponent_pos - player_pos
        vec_ball_to_goal = goal_1_pos - ball_pos
        vec_ball_to_oppgoal = goal_2_pos - ball_pos

        # Compute angles
        angle_pegball_pegoppgoal = self.angle_between_vectors(
            vec_to_ball, vec_to_opp_goal
        )
        angle_oppball_oppgoal = self.angle_between_vectors(
            vec_opp_to_goal, vec_ball_to_opp
        )
        angle_pegball_pegopp = self.angle_between_vectors(
            vec_to_ball, vec_opp_to_player
        )
        angle_oppball_pegopp = self.angle_between_vectors(
            vec_ball_to_opp, -vec_opp_to_player
        )

        # Compute distances
        distance_ball_goal = np.linalg.norm(vec_ball_to_goal)
        distance_ball_oppgoal = np.linalg.norm(vec_ball_to_oppgoal)
        distance_ball_player = np.linalg.norm(vec_to_ball)
        distance_ball_opp = np.linalg.norm(vec_ball_to_opp)

        # Concatenate additional features
        extra_features = np.array(
            [
                angle_pegball_pegoppgoal,
                angle_oppball_oppgoal,
                angle_pegball_pegopp,
                angle_oppball_pegopp,
                distance_ball_goal,
                distance_ball_oppgoal,
                distance_ball_player,
                distance_ball_opp,
            ],
            dtype=np.float32,
        )

        # Return first 12 values (original obs without goals) + 8 features = 20 values
        return np.concatenate([obs[:12], extra_features])

    def angle_between_vectors(self, v1, v2):
        """Compute angle between two vectors."""
        dot = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        cos_theta = dot / (norm_v1 * norm_v2 + 1e-8)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_rad = np.arccos(cos_theta)
        return angle_rad

    def get_action(self, obs):
        """Get action from policy network."""
        # Convert to tensor
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)

        # Get action from policy
        action = self.policy_net.get_action(obs_tensor, deterministic=True)

        # Convert back to numpy
        action = action.cpu().numpy().flatten()

        return action

    def publish_action(self, action):
        """Publish action as velocity command."""
        msg = Twist()
        msg.linear.x = float(action[1])
        msg.linear.y = float(action[0])
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    try:
        player = Player()
        rclpy.spin(player)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
