import rclpy
import torch
import numpy as np
import os
import urllib.request
import zipfile
import tempfile
from pathlib import Path
from rclpy.node import Node
from .policy_network import PolicyNetwork
from klask_interfaces.msg import State
from geometry_msgs.msg import Twist


class Player(Node):
    """ROS2 node for Klask policy inference."""

    def __init__(self):
        super().__init__("klask_policy_inference_node")

        # Declare ROS parameters
        self.declare_parameter("weights_filename", "klask.pth")
        self.declare_parameter(
            "weights_zip_url",
            "https://polybox.ethz.ch/index.php/s/joEoP8GgQbwmToW/download",
        )
        self.declare_parameter("device", "cpu")
        self.declare_parameter("board_width", 0.42)
        self.declare_parameter("board_height", 0.32)
        self.declare_parameter("player_side", "left")
        self.declare_parameter("state_topic", "/board_state")
        self.declare_parameter("cmd_vel_topic", "cmd_vel/left_player")
        self.declare_parameter("subscription_queue_size", 1)
        self.declare_parameter("publisher_queue_size", 1)

        # Get parameters
        weights_filename = self.get_parameter("weights_filename").value
        weights_zip_url = self.get_parameter("weights_zip_url").value

        # Set nn_weights directory path
        nn_weights_dir = (
            Path(__file__).resolve().parent.parent.parent.parent / "nn_weights"
        )

        # Ensure weights are available
        checkpoint_path = self.ensure_weights_available(
            weights_filename, weights_zip_url, nn_weights_dir
        )
        self.device = self.get_parameter("device").value
        self.player_side = self.get_parameter("player_side").value
        state_topic = self.get_parameter("state_topic").value
        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        sub_queue_size = self.get_parameter("subscription_queue_size").value
        pub_queue_size = self.get_parameter("publisher_queue_size").value

        # Board dimensions for coordinate centering
        self.board_dim_width = self.get_parameter("board_width").value
        self.board_dim_height = self.get_parameter("board_height").value

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
            f"Board dimensions: {self.board_dim_width}m x {self.board_dim_height}m"
        )
        self.get_logger().info(f"Player side: {self.player_side}")
        self.get_logger().info(f"Subscribed to: {state_topic}")
        self.get_logger().info(f"Publishing to: {cmd_vel_topic}")

    def ensure_weights_available(self, weights_filename, zip_url, nn_weights_dir):
        """Ensure weights file exists, download and extract zip if necessary."""
        # nn_weights_dir is already a Path object
        weights_dir = nn_weights_dir
        weights_path = weights_dir / weights_filename

        # Check if the specific weights file exists
        if weights_path.exists():
            self.get_logger().info(f"Weights file found: {weights_path}")
            return str(weights_path)

        # Download and extract the zip file
        self.get_logger().info(
            f"Weights file not found. Downloading weights zip from: {zip_url}"
        )
        self.download_and_extract_weights(zip_url, weights_dir)

        # Verify the weights file exists after extraction
        if weights_path.exists():
            self.get_logger().info(f"Weights extracted successfully: {weights_path}")
            return str(weights_path)
        else:
            # List available files
            available_files = [f.name for f in weights_dir.glob("*.pth")]
            error_msg = f"Weights file '{weights_filename}' not found after extraction. Available files: {available_files}"
            self.get_logger().error(error_msg)
            raise FileNotFoundError(error_msg)

    def download_and_extract_weights(self, url, destination_dir):
        """Download weights zip file and extract to destination directory."""
        try:
            # Create a temporary file for the zip download
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
                tmp_path = tmp_file.name

                self.get_logger().info(
                    f"Downloading weights zip to temporary file: {tmp_path}"
                )

                # Create a request with headers to handle redirects
                req = urllib.request.Request(url)
                req.add_header("User-Agent", "Mozilla/5.0")

                with urllib.request.urlopen(req) as response:
                    # Read and save the zip file
                    chunk_size = 8192
                    total_size = 0
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        tmp_file.write(chunk)
                        total_size += len(chunk)
                        if total_size % (chunk_size * 100) == 0:  # Log every ~800KB
                            self.get_logger().info(
                                f"Downloaded {total_size / 1024 / 1024:.2f} MB..."
                            )

                self.get_logger().info(
                    f"Download complete: {total_size / 1024 / 1024:.2f} MB"
                )

            # Extract the zip file
            self.get_logger().info(f"Extracting weights to: {destination_dir}")
            with zipfile.ZipFile(tmp_path, "r") as zip_ref:
                zip_ref.extractall(destination_dir)
                extracted_files = zip_ref.namelist()
                self.get_logger().info(f"Extracted {len(extracted_files)} files")

            # Check if everything was extracted into a single root folder and move contents up
            root_items = list(destination_dir.iterdir())
            if len(root_items) == 1 and root_items[0].is_dir():
                # Move contents up one level
                nested_dir = root_items[0]
                for item in nested_dir.iterdir():
                    item.rename(destination_dir / item.name)
                nested_dir.rmdir()

            # Clean up temporary file
            os.unlink(tmp_path)
            self.get_logger().info("Extraction complete")

        except Exception as e:
            self.get_logger().error(f"Failed to download and extract weights: {e}")
            # Clean up temporary file if it exists
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

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
        # Extract and transform observations (handles centering and player-side transformation)
        obs_base = self.map_observations(msg)

        # Compute additional features
        obs_full = self.add_additional_features(obs_base)

        # Get action from policy
        action = self.get_action(obs_full)

        # Negate action for right player (flip direction)
        if self.player_side == "right":
            action = -action

        # Publish action
        self.publish_action(action)

    def map_observations(self, msg: State):
        """Extract observations from State message, center coordinates, and transform for player side.

        Returns:
            16-dimensional centered observation array ready for feature computation.
            Format: [player_pos, player_vel, opponent_pos, opponent_vel, ball_pos, ball_vel, goal_player_pos, goal_opponent_pos]
        """
        # Extract raw observations (engineering units: meters and m/s)
        right_peg_pos = np.array(
            [msg.right_peg.position.y, msg.right_peg.position.x], dtype=np.float32
        )
        right_peg_vel = np.array(
            [msg.right_peg.velocity.y, msg.right_peg.velocity.x], dtype=np.float32
        )
        left_peg_pos = np.array(
            [msg.left_peg.position.y, msg.left_peg.position.x], dtype=np.float32
        )
        left_peg_vel = np.array(
            [msg.left_peg.velocity.y, msg.left_peg.velocity.x], dtype=np.float32
        )
        ball_pos = np.array(
            [msg.ball.position.y, msg.ball.position.x], dtype=np.float32
        )
        ball_vel = np.array(
            [msg.ball.velocity.y, msg.ball.velocity.x], dtype=np.float32
        )
        right_goal_pos = np.array(
            [msg.right_goal_pos.y, msg.right_goal_pos.x], dtype=np.float32
        )
        left_goal_pos = np.array(
            [msg.left_goal_pos.y, msg.left_goal_pos.x], dtype=np.float32
        )

        # Center all positions to board origin
        center_offset = np.array(
            [self.board_dim_height / 2.0, self.board_dim_width / 2.0], dtype=np.float32
        )
        right_peg_pos -= center_offset
        left_peg_pos -= center_offset
        ball_pos -= center_offset
        right_goal_pos -= center_offset
        left_goal_pos -= center_offset

        # Map to player/opponent based on player_side
        if self.player_side == "left":
            # Player controls left peg, opponent is right peg
            obs = np.concatenate(
                [
                    left_peg_pos,
                    left_peg_vel,  # player (left peg)
                    right_peg_pos,
                    right_peg_vel,  # opponent (right peg)
                    ball_pos,
                    ball_vel,  # ball
                    left_goal_pos,
                    right_goal_pos,  # player's goal, opponent's goal
                ]
            )
        else:  # right
            # Player controls right peg, opponent is left peg
            # Transform to opponent's perspective: swap player/opponent and negate
            obs = np.concatenate(
                [
                    -right_peg_pos,
                    -right_peg_vel,  # right peg becomes player (negated)
                    -left_peg_pos,
                    -left_peg_vel,  # left peg becomes opponent (negated)
                    -ball_pos,
                    -ball_vel,  # ball (negated)
                    -right_goal_pos,
                    -left_goal_pos,  # player's goal, opponent's goal (swapped and negated)
                ]
            )

        return obs

    def add_additional_features(self, obs):
        """Add geometric features (angles and distances) to observations.

        Input: 16 values [player_pos, player_vel, opp_pos, opp_vel, ball_pos, ball_vel, goal_player_pos, goal_opponent_pos]
        Output: 20 values [player_pos, player_vel, opp_pos, opp_vel, ball_pos, ball_vel, 8 features]

        Note: All coordinates are already centered and in engineering units (meters and m/s).
        """
        # Extract components
        player_pos = obs[0:2]
        opponent_pos = obs[4:6]
        ball_pos = obs[8:10]
        goal_player_pos = obs[12:14]
        goal_opponent_pos = obs[14:16]

        # Compute vectors
        vec_to_opp_goal = goal_opponent_pos - player_pos
        vec_to_ball = ball_pos - player_pos
        vec_opp_to_goal = goal_player_pos - opponent_pos
        vec_ball_to_opp = ball_pos - opponent_pos
        vec_opp_to_player = opponent_pos - player_pos
        vec_ball_to_goal = goal_player_pos - ball_pos
        vec_ball_to_opp_goal = goal_opponent_pos - ball_pos

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
        distance_ball_oppgoal = np.linalg.norm(vec_ball_to_opp_goal)
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
