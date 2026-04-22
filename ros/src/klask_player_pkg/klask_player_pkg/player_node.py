"""ROS2 node for Klask policy inference."""

import rclpy
import numpy as np
from pathlib import Path
from rclpy.node import Node
from rclpy.action import ActionClient
from .ppo.ppo_inference import PolicyInference
from .game_state import GameState
from klask_interfaces.msg import State
from klask_interfaces.srv import GetCalibrationStatus, IsPlayerHomed
from klask_interfaces.action import HomeAndCalibrate
from klask_interfaces_py import BoardState
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy


class Player(Node):
    """ROS2 node for Klask policy inference."""

    def __init__(self):
        """Initialize the player node."""
        super().__init__("klask_player_node")

        # State machine variables
        self.game_state = self.last_game_state = GameState.INITIALIZING

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

        # Agent type selection
        self.declare_parameter("agent_type", "ppo")
        self.declare_parameter("image_topic", "board_image/compressed")
        self.declare_parameter("dreamer_config_filename", "")
        self.declare_parameter("debug_view", False)

        # State machine parameters
        self.declare_parameter("interaction_delay", 2.0)

        # Heartbeat watchdog parameters
        self.declare_parameter("heartbeat_topic", "heartbeat")
        self.declare_parameter("heartbeat_timeout", 1.5)

        # Motor commander service parameters (these are global, not per-player)
        self.declare_parameter("get_calibration_status_service_name", "/get_calibration_status")
        self.declare_parameter("is_player_homed_service_name", "/is_player_homed")
        # Home and calibrate action is per-player: home_and_calibrate_{left,right}_player
        self.declare_parameter("home_and_calibrate_action_prefix", "/home_and_calibrate")

        # Get parameters
        state_topic = self.get_parameter("state_topic").value
        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        sub_queue_size = self.get_parameter("subscription_queue_size").value
        pub_queue_size = self.get_parameter("publisher_queue_size").value

        # State machine parameters
        self.player_side = self.get_parameter("player_side").value
        self.interaction_delay = self.get_parameter("interaction_delay").value

        if self.player_side == "left":
            self.PEG_IN_OWN_GOAL_FLAG = BoardState.PEG_IN_LEFT_GOAL
            self.center_direction = 1
        else:
            self.PEG_IN_OWN_GOAL_FLAG = BoardState.PEG_IN_RIGHT_GOAL
            self.center_direction = -1

        # Motor commander service names (calibration status and homed are global services)
        self.calibration_status_service_name = self.get_parameter("get_calibration_status_service_name").value
        self.is_player_homed_service_name = self.get_parameter("is_player_homed_service_name").value
        # Home and calibrate action is per-player
        home_calibrate_action_prefix = self.get_parameter("home_and_calibrate_action_prefix").value
        self.home_and_calibrate_action_name = f"{home_calibrate_action_prefix}_{self.player_side}_player"

        # Action handles
        self.homing_goal_handle = None

        # Timers
        self.interaction_delay_timer = None
        self.magnet_recover_timer = None

        # Heartbeat watchdog
        self.heartbeat_timeout = self.get_parameter("heartbeat_timeout").value
        heartbeat_qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.heartbeat_sub = self.create_subscription(
            Empty,
            self.get_parameter("heartbeat_topic").value,
            self._heartbeat_callback,
            heartbeat_qos,
        )
        self.heartbeat_received = False
        self.heartbeat_watchdog = self.create_timer(self.heartbeat_timeout, self._heartbeat_lost_callback)

        # Initialize policy inference class based on agent type
        self.agent_type = self.get_parameter("agent_type").value
        nn_weights_dir = Path(__file__).resolve().parent.parent.parent.parent / "nn_weights"

        if self.agent_type == "dreamer":
            from .dreamer.dreamer_inference import DreamerInference

            self.policy = DreamerInference(
                weights_filename=self.get_parameter("weights_filename").value,
                weights_zip_url=self.get_parameter("weights_zip_url").value,
                nn_weights_dir=nn_weights_dir,
                device=self.get_parameter("device").value,
                player_side=self.player_side,
                board_dim_width=self.get_parameter("board_width").value,
                board_dim_height=self.get_parameter("board_height").value,
                config_filename=self.get_parameter("dreamer_config_filename").value,
                debug_view=self.get_parameter("debug_view").value,
                logger=self.get_logger(),
            )

            # Subscribe to camera images (same method as state estimator)
            image_topic = self.get_parameter("image_topic").value
            self.bridge = CvBridge()
            self.latest_image = None
            self.image_subscription = self.create_subscription(CompressedImage, image_topic, self._image_callback, 10)
            self.get_logger().info(f"Subscribed to camera: {image_topic}")
        else:
            self.policy = PolicyInference(
                weights_filename=self.get_parameter("weights_filename").value,
                weights_zip_url=self.get_parameter("weights_zip_url").value,
                nn_weights_dir=nn_weights_dir,
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

        # Service clients for motor commander
        self.calibration_status_client = self.create_client(GetCalibrationStatus, self.calibration_status_service_name)
        self.homed_client = self.create_client(IsPlayerHomed, self.is_player_homed_service_name)
        self.home_calibrate_client = ActionClient(self, HomeAndCalibrate, self.home_and_calibrate_action_name)

        self.get_logger().info(f"Agent type: {self.agent_type}")
        self.get_logger().info(f"Player side: {self.player_side}")
        self.get_logger().info(f"Subscribed to: {state_topic}")
        self.get_logger().info(f"Publishing to: {cmd_vel_topic}")
        self.get_logger().info(f"Calibration service: {self.calibration_status_service_name}")
        self.get_logger().info(f"Home/Calibrate service: {self.home_and_calibrate_action_name}")

    def observation_callback(self, msg: State):
        """Process incoming observations and publish actions."""
        if self.last_game_state != self.game_state:
            self.get_logger().info(f"State transition: {self.last_game_state.name} -> {self.game_state.name}")

        # State machine logic
        if self.game_state == GameState.INITIALIZING:
            if msg.status.data & BoardState.READY:
                self.game_state = GameState.STATE_ESTIMATOR_READY

        elif self.game_state == GameState.STATE_ESTIMATOR_READY:
            self._check_calibration_status()

        elif self.game_state == GameState.HW_CALIBRATED:
            self._check_at_home()

        elif self.game_state == GameState.HW_UNCALIBRATED:
            self._check_homing_in_progress()

        elif self.game_state == GameState.REQUESTING_HOME_CAL:
            self._start_homing()

        elif self.game_state == GameState.HOMING:
            # Waiting for homing to complete
            pass

        elif self.game_state == GameState.HW_READY:
            if not (msg.status.data & BoardState.READY):
                self.game_state = GameState.UNKNOWN_BOARD_STATE

            elif msg.status.data & (
                BoardState.BALL_IN_LEFT_GOAL
                | BoardState.BALL_IN_RIGHT_GOAL
                | BoardState.PEG_IN_LEFT_GOAL
                | BoardState.PEG_IN_RIGHT_GOAL
            ):
                # Wait until board is ready again
                pass
            else:
                self.game_state = GameState.INTERACTION_DELAY

        elif self.game_state == GameState.INTERACTION_DELAY:
            # Wait for interaction delay to pass
            if self.interaction_delay_timer is None:
                # Reset RSSM state for new episode (no-op for PPO)
                self.policy.reset_state()
                self.get_logger().info(f"Starting interaction delay timer ({self.interaction_delay}s)")
                self.interaction_delay_timer = self.create_timer(
                    self.interaction_delay, self._interaction_delay_complete
                )

        elif self.game_state == GameState.PLAYING:
            action = np.array([0.0, 0.0])

            if not (msg.status.data & BoardState.READY):
                self.game_state = GameState.UNKNOWN_BOARD_STATE

            elif msg.status.data & (
                BoardState.BALL_IN_LEFT_GOAL
                | BoardState.BALL_IN_RIGHT_GOAL
                | BoardState.PEG_IN_LEFT_GOAL
                | BoardState.PEG_IN_RIGHT_GOAL
            ):
                self.game_state = GameState.GAME_OVER
            else:
                # Normal gameplay - compute and publish action
                image = getattr(self, "latest_image", None)
                action = self.policy.get_action(msg, image)

            self.publish_action(action)

        elif self.game_state == GameState.GAME_OVER:
            # stop motors
            self.publish_action(np.array([0.0, 0.0]))

            if msg.status.data & self.PEG_IN_OWN_GOAL_FLAG:
                self.game_state = GameState.MOVE_MAGNET
            else:
                self.game_state = GameState.REQUESTING_HOME_CAL

        elif self.game_state == GameState.UNKNOWN_BOARD_STATE:
            self.publish_action(np.array([0.0, 0.0]))
            if msg.status.data & BoardState.READY:
                self.game_state = GameState.STATE_ESTIMATOR_READY

        elif self.game_state == GameState.MOVE_MAGNET:
            # Move magnet towards center
            recovery_time = 1.0  # seconds
            recovery_speed = 0.01  # m/s
            self.publish_action(np.array([0.0, recovery_speed * self.center_direction]))

            if self.magnet_recover_timer is None:
                self.get_logger().info(f"Starting moving the magnet away from goal for {recovery_time}s")
                self.magnet_recover_timer = self.create_timer(recovery_time, self._magnet_recover_complete)

        elif self.game_state == GameState.WAIT_FOR_PEG_RESET:
            # Wait for the peg to be removed from goal
            self.publish_action(np.array([0.0, 0.0]))
            if msg.status.data & self.PEG_IN_OWN_GOAL_FLAG:
                # still in goal, wait
                pass
            else:
                self.game_state = GameState.REQUESTING_HOME_CAL

        # remember last state for logging
        self.last_game_state = self.game_state

    def publish_action(self, action):
        """Publish action as velocity command."""
        msg = Twist()
        msg.linear.x = float(action[1])
        msg.linear.y = float(action[0])
        self.publisher.publish(msg)

    def _check_calibration_status(self):
        """Check if motor controllers are calibrated."""
        if not self.calibration_status_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Calibration service not available!")
            return

        request = GetCalibrationStatus.Request()
        request.player = f"{self.player_side}_player"
        future = self.calibration_status_client.call_async(request)
        future.add_done_callback(self._calibration_status_callback)

    def _calibration_status_callback(self, future):
        """Handle calibration status response."""
        try:
            response = future.result()
            if response.is_calibrated:
                self.get_logger().info("Motors are calibrated")
                self.game_state = GameState.HW_CALIBRATED
            else:
                self.get_logger().info("Motors not calibrated")
                self.game_state = GameState.HW_UNCALIBRATED
        except Exception as e:
            self.get_logger().error(f"Calibration check failed: {e}")

    def _check_at_home(self):
        """Check if the peg is at the home position."""
        if not self.homed_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Homed service not available!")
            return

        request = IsPlayerHomed.Request()
        request.player = f"{self.player_side}_player"
        future = self.homed_client.call_async(request)
        future.add_done_callback(self._at_home_status_callback)

    def _at_home_status_callback(self, future):
        """Handle at-home status response."""
        try:
            response = future.result()
            if response.is_homed:
                self.get_logger().info("Player is at home position")
                self.game_state = GameState.HW_READY
            else:
                self.get_logger().info("Player not at home position")
                self.game_state = GameState.REQUESTING_HOME_CAL
        except Exception as e:
            self.get_logger().error(f"Homed check failed: {e}")

    def _interaction_delay_complete(self):
        """Handle interaction delay completion."""
        if self.interaction_delay_timer is not None:
            self.interaction_delay_timer.cancel()
            self.interaction_delay_timer = None

        self.game_state = GameState.PLAYING

    def _check_homing_in_progress(self):
        """Check if the homing action is in progress."""
        if self.homing_goal_handle is not None and self.homing_goal_handle.status in [1, 2]:  # ACCEPTED or EXECUTING
            self.get_logger().info("Homing action already in progress")
            self.game_state = GameState.HOMING
        else:
            self.get_logger().info("No active homing action, requesting homing")
            self.game_state = GameState.REQUESTING_HOME_CAL

    def _start_homing(self):
        """Start homing and calibration sequence."""
        self.game_state = GameState.HOMING

        if not self.home_calibrate_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error("Home/Calibrate action not available")
            return

        goal_msg = HomeAndCalibrate.Goal()
        send_goal_future = self.home_calibrate_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self._homing_goal_response_callback)
        self.get_logger().info("Homing request sent...")

    def _homing_goal_response_callback(self, future):
        """Handle homing and calibration goal acceptance."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Homing goal rejected")
            self.homing_goal_handle = None
            self.game_state = GameState.STATE_ESTIMATOR_READY
            return

        self.get_logger().info("Homing goal accepted - now in progress")
        self.homing_goal_handle = goal_handle
        self.game_state = GameState.HOMING
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._homing_complete_callback)

    def _homing_complete_callback(self, future):
        """Handle homing and calibration completion."""
        try:
            result = future.result().result
            if result.success:
                self.get_logger().info("Homing completed successfully")
            else:
                self.get_logger().warn(f"Homing failed: {result.message}")
        except Exception as e:
            self.get_logger().error(f"Homing action call failed: {e}")

        self.homing_goal_handle = None
        self.game_state = GameState.STATE_ESTIMATOR_READY

    def _heartbeat_callback(self, msg):
        """Reset watchdog on heartbeat received."""
        self.heartbeat_received = True
        self.heartbeat_watchdog.cancel()
        self.heartbeat_watchdog = self.create_timer(self.heartbeat_timeout, self._heartbeat_lost_callback)

    def _heartbeat_lost_callback(self):
        """Handle motor commander heartbeat loss."""
        self.heartbeat_watchdog.cancel()

        if not self.heartbeat_received:
            self.get_logger().warn("No heartbeat received from motor commander yet.")
            self.heartbeat_watchdog = self.create_timer(self.heartbeat_timeout, self._heartbeat_lost_callback)
            return

        self.get_logger().error("Motor commander heartbeat lost! Resetting to INITIALIZING.")

        # Stop motors immediately
        self.publish_action(np.array([0.0, 0.0]))

        # Clean up any in-progress timers/actions
        if self.interaction_delay_timer is not None:
            self.interaction_delay_timer.cancel()
            self.interaction_delay_timer = None
        if self.magnet_recover_timer is not None:
            self.magnet_recover_timer.cancel()
            self.magnet_recover_timer = None
        self.homing_goal_handle = None

        # Reset state machine
        self.game_state = GameState.INITIALIZING

    def _image_callback(self, msg: CompressedImage):
        """Callback for receiving compressed camera images (Dreamer agent)."""
        try:
            self.latest_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to decompress image: {e}")

    def _magnet_recover_complete(self):
        """Handle magnet recovery completion."""
        if self.magnet_recover_timer is not None:
            self.magnet_recover_timer.cancel()
            self.magnet_recover_timer = None

        self.game_state = GameState.WAIT_FOR_PEG_RESET


def main(args=None):
    """Main entry point for the player node."""
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
