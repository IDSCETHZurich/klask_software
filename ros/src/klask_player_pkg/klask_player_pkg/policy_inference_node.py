import rclpy
from pathlib import Path
from rclpy.node import Node
from .policy_inference import PolicyInference
from klask_interfaces.msg import State
from geometry_msgs.msg import Twist


class Player(Node):
    """ROS2 node for Klask policy inference."""

    def __init__(self):
        super().__init__("klask_policy_inference_node")

        # Declare ROS parameters
        self.declare_parameter("weights_filename", "klask_ac_nn_v0.0.pth")
        self.declare_parameter(
            "weights_zip_url",
            "https://polybox.ethz.ch/index.php/s/joEoP8GgQbwmToW/download",
        )
        self.declare_parameter("device", "cpu")
        self.declare_parameter("board_width", 0.42)
        self.declare_parameter("board_height", 0.32)
        self.declare_parameter("player_side", "left")
        self.declare_parameter("state_topic", "/board_state")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel/left_player")
        self.declare_parameter("subscription_queue_size", 1)
        self.declare_parameter("publisher_queue_size", 1)
        self.declare_parameter("clip_actions", 0.2)
        self.declare_parameter("enable_action_rescaling", True)

        # Get parameters
        state_topic = self.get_parameter("state_topic").value
        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        sub_queue_size = self.get_parameter("subscription_queue_size").value
        pub_queue_size = self.get_parameter("publisher_queue_size").value

        # Initialize policy inference class
        self.policy = PolicyInference(
            weights_filename=self.get_parameter("weights_filename").value,
            weights_zip_url=self.get_parameter("weights_zip_url").value,
            nn_weights_dir=Path(__file__).resolve().parent.parent.parent.parent
            / "nn_weights",
            device=self.get_parameter("device").value,
            clip_actions=self.get_parameter("clip_actions").value,
            enable_action_rescaling=self.get_parameter("enable_action_rescaling").value,
            player_side=self.get_parameter("player_side").value,
            board_dim_width=self.get_parameter("board_width").value,
            board_dim_height=self.get_parameter("board_height").value,
            logger=self.get_logger(),
        )

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

        self.get_logger().info(f"Subscribed to: {state_topic}")
        self.get_logger().info(f"Publishing to: {cmd_vel_topic}")

    def observation_callback(self, msg: State):
        """Process incoming observations and publish actions."""

        action = self.policy.get_action(msg)
        self.publish_action(action)

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
