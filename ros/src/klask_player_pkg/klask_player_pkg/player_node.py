import rclpy
from pathlib import Path
from rclpy.node import Node
from .policy_inference import PolicyInference
from .game_state import GameState, state_name
from klask_interfaces.msg import State
from klask_interfaces.srv import GetCalibrationStatus, HomeAndCalibrate
from klask_interfaces_py import BoardState
from geometry_msgs.msg import Twist
from std_srvs.srv import Trigger


class Player(Node):
    """ROS2 node for Klask policy inference."""

    def __init__(self):
        super().__init__("klask_player_node")

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

        # State machine parameters
        self.declare_parameter("check_calibration_on_startup", True)
        self.declare_parameter("enable_auto_reset", True)
        self.declare_parameter("goal_reset_delay_seconds", 2.0)
        self.declare_parameter("peg_goal_pause_enabled", True)

        # Motor commander service parameters
        self.declare_parameter("motor_commander_namespace", "/motor_commander")
        self.declare_parameter(
            "get_calibration_status_service", "get_calibration_status"
        )
        self.declare_parameter("home_and_calibrate_service", "home_and_calibrate")

        # Get parameters
        state_topic = self.get_parameter("state_topic").value
        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        sub_queue_size = self.get_parameter("subscription_queue_size").value
        pub_queue_size = self.get_parameter("publisher_queue_size").value

        # State machine parameters
        self.player_side = self.get_parameter("player_side").value
        self.check_calibration_on_startup = self.get_parameter(
            "check_calibration_on_startup"
        ).value
        self.enable_auto_reset = self.get_parameter("enable_auto_reset").value
        self.goal_reset_delay = self.get_parameter("goal_reset_delay_seconds").value
        self.peg_goal_pause_enabled = self.get_parameter("peg_goal_pause_enabled").value

        # Motor commander service names
        motor_namespace = self.get_parameter("motor_commander_namespace").value
        calibration_service = self.get_parameter("get_calibration_status_service").value
        home_calibrate_service = self.get_parameter("home_and_calibrate_service").value

        # Build full service names based on player side
        self.calibration_status_service_name = (
            f"{motor_namespace}/{self.player_side}/{calibration_service}"
        )
        self.home_and_calibrate_service_name = (
            f"{motor_namespace}/{self.player_side}/{home_calibrate_service}"
        )

        # Initialize policy inference class
        self.policy = PolicyInference(
            weights_filename=self.get_parameter("weights_filename").value,
            weights_zip_url=self.get_parameter("weights_zip_url").value,
            nn_weights_dir=Path(__file__).resolve().parent.parent.parent.parent
            / "nn_weights",
            device=self.get_parameter("device").value,
            clip_actions=self.get_parameter("clip_actions").value,
            enable_action_rescaling=self.get_parameter("enable_action_rescaling").value,
            player_side=self.player_side,
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

        # State machine variables
        self.game_state = GameState.INITIALIZING
        self.reset_timer = None

        # Service clients for motor commander
        self.calibration_client = self.create_client(
            GetCalibrationStatus, self.calibration_status_service_name
        )
        self.home_calibrate_client = self.create_client(
            HomeAndCalibrate, self.home_and_calibrate_service_name
        )

        self.get_logger().info(f"Player side: {self.player_side}")
        self.get_logger().info(f"Subscribed to: {state_topic}")
        self.get_logger().info(f"Publishing to: {cmd_vel_topic}")
        self.get_logger().info(
            f"Calibration service: {self.calibration_status_service_name}"
        )
        self.get_logger().info(
            f"Home/Calibrate service: {self.home_and_calibrate_service_name}"
        )
        self.get_logger().info(f"Auto-reset enabled: {self.enable_auto_reset}")
        self.get_logger().info(f"Goal reset delay: {self.goal_reset_delay}s")

        # Start state machine by checking calibration
        if self.check_calibration_on_startup:
            self.get_logger().info("Checking calibration status...")
            # Schedule one-time calibration check
            self.create_timer(0.5, self._check_calibration_status)
        else:
            # Skip calibration check and go straight to waiting for board ready
            self.get_logger().info("Skipping calibration check")
            self.game_state = GameState.WAITING_FOR_READY

    def observation_callback(self, msg: State):
        """Process incoming observations and publish actions."""

        # State machine logic
        if self.game_state == GameState.INITIALIZING:
            # Waiting for calibration check or skipping it
            return
        elif self.game_state == GameState.HOMING:
            # Waiting for homing to complete
            return
        elif self.game_state == GameState.WAITING_FOR_READY:
            # Check if board is ready
            if msg.status & BoardState.READY:
                self.get_logger().info("Board ready - transitioning to PLAYING")
                self.game_state = GameState.PLAYING
        elif self.game_state == GameState.PLAYING:
            # Check for goal conditions
            if self.enable_auto_reset:
                if msg.status & (
                    BoardState.BALL_IN_LEFT_GOAL | BoardState.BALL_IN_RIGHT_GOAL
                ):
                    self._handle_goal_detected()
                    return
                elif self.peg_goal_pause_enabled and msg.status & (
                    BoardState.PEG_IN_LEFT_GOAL | BoardState.PEG_IN_RIGHT_GOAL
                ):
                    self._handle_peg_in_goal()
                    return

            # Normal gameplay - compute and publish action
            action = self.policy.get_action(msg)
            self.publish_action(action)
        elif self.game_state == GameState.GOAL_DETECTED:
            # Waiting for reset timer
            pass
        elif self.game_state == GameState.PEG_IN_GOAL:
            # Paused until peg removed
            if not (
                msg.status
                & (BoardState.PEG_IN_LEFT_GOAL | BoardState.PEG_IN_RIGHT_GOAL)
            ):
                self.get_logger().info("Peg removed - resuming play")
                self.game_state = GameState.PLAYING
        elif self.game_state == GameState.RESETTING:
            # Waiting for reset to complete
            pass

    def publish_action(self, action):
        """Publish action as velocity command."""
        msg = Twist()
        msg.linear.x = float(action[1])
        msg.linear.y = float(action[0])
        self.publisher.publish(msg)

    def _check_calibration_status(self):
        """Check if motor controllers are calibrated."""
        if not self.calibration_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Calibration service not available, skipping check")
            self.game_state = GameState.WAITING_FOR_READY
            return

        request = GetCalibrationStatus.Request()
        future = self.calibration_client.call_async(request)
        future.add_done_callback(self._calibration_status_callback)

    def _calibration_status_callback(self, future):
        """Handle calibration status response."""
        try:
            response = future.result()
            if response.calibrated:
                self.get_logger().info("Motors already calibrated")
                self.game_state = GameState.WAITING_FOR_READY
            else:
                self.get_logger().info("Motors not calibrated - starting homing")
                self._start_homing()
        except Exception as e:
            self.get_logger().error(f"Calibration check failed: {e}")
            self.game_state = GameState.WAITING_FOR_READY

    def _start_homing(self):
        """Start homing and calibration sequence."""
        self.game_state = GameState.HOMING

        if not self.home_calibrate_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("Home/Calibrate service not available")
            self.game_state = GameState.WAITING_FOR_READY
            return

        request = HomeAndCalibrate.Request()
        future = self.home_calibrate_client.call_async(request)
        future.add_done_callback(self._homing_complete_callback)
        self.get_logger().info("Homing in progress...")

    def _homing_complete_callback(self, future):
        """Handle homing completion."""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info("Homing completed successfully")
            else:
                self.get_logger().warn(f"Homing failed: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Homing service call failed: {e}")

        self.game_state = GameState.WAITING_FOR_READY

    def _handle_goal_detected(self):
        """Handle goal detection - start reset delay timer."""
        self.get_logger().info(f"Goal detected! Resetting in {self.goal_reset_delay}s")
        self.game_state = GameState.GOAL_DETECTED

        # Stop motion immediately
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = 0.0
        cmd_vel_msg.linear.y = 0.0
        self.publisher.publish(cmd_vel_msg)

        # Start reset timer
        if self.reset_timer is not None:
            self.reset_timer.cancel()
        self.reset_timer = self.create_timer(self.goal_reset_delay, self._reset_game)

    def _handle_peg_in_goal(self):
        """Handle peg in goal - pause gameplay."""
        self.get_logger().info("Peg in goal! Pausing gameplay")
        self.game_state = GameState.PEG_IN_GOAL

        # Stop motion immediately
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = 0.0
        cmd_vel_msg.linear.y = 0.0
        self.publisher.publish(cmd_vel_msg)

    def _reset_game(self):
        """Reset the game by homing the motors."""
        if self.reset_timer is not None:
            self.reset_timer.cancel()
            self.reset_timer = None

        self.get_logger().info("Starting game reset...")
        self.game_state = GameState.RESETTING

        if not self.home_calibrate_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("Home/Calibrate service not available for reset")
            self.game_state = GameState.WAITING_FOR_READY
            return

        request = HomeAndCalibrate.Request()
        future = self.home_calibrate_client.call_async(request)
        future.add_done_callback(self._reset_complete_callback)

    def _reset_complete_callback(self, future):
        """Handle reset completion."""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info("Reset completed successfully")
            else:
                self.get_logger().warn(f"Reset failed: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Reset service call failed: {e}")

        self.game_state = GameState.WAITING_FOR_READY


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
