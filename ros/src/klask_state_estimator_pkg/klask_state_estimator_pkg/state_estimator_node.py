"""ROS2 node for ball and peg state estimation from processed camera images."""

import cv2
import time
import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import UInt64
from klask_interfaces.msg import StampedPolygon, State
from klask_interfaces_py import BoardState
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

from .kalman_filter import KalmanFilter
from .utils import create_point_from_list
from .debug import draw_object_with_velocity, plot_image


class StateEstimatorNode(Node):
    """ROS2 node for estimating ball and peg positions from camera images."""

    def __init__(self):
        super().__init__("state_estimator")

        # =============================
        # Parameters
        # =============================

        # Enable visualization windows (default: false)
        self.declare_parameter("show_image", False)
        self.show_image = bool(self.get_parameter("show_image").value)

        # =============================
        # Pixel to Engineering Units (EU) Conversion
        # =============================
        # Board dimensions in meters (physical board size)
        self.declare_parameter("board_width_meters", 0.42)
        self.board_width_meters = float(self.get_parameter("board_width_meters").value)

        self.declare_parameter("board_height_meters", 0.32)
        self.board_height_meters = float(
            self.get_parameter("board_height_meters").value
        )

        # Current image dimensions (updated each frame)
        self.current_image_width: float = 0.0
        self.current_image_height: float = 0.0

        # Radius around goal center to count as "in goal" (pixels) (default: 22)
        self.declare_parameter("goal_radius", 22)
        self.goal_radius = int(self.get_parameter("goal_radius").value)

        # Frames required before confirming goal (default: 30)
        self.declare_parameter("goal_hyst_counter", 30)
        self.goal_hyst_counter = int(self.get_parameter("goal_hyst_counter").value)

        # Distance threshold for edge/peg collision logic (pixels) (default: 35.0)
        self.declare_parameter("collision_distance", 35.0)
        self.collision_distance = float(self.get_parameter("collision_distance").value)

        # Orange ball lower bound
        self.declare_parameter("ball_hsv_lower", [10, 60, 200])
        self.ball_hsv_lower = self._hsv_param("ball_hsv_lower", (10, 60, 200))

        # Orange ball upper bound
        self.declare_parameter("ball_hsv_upper", [45, 160, 255])
        self.ball_hsv_upper = self._hsv_param("ball_hsv_upper", (45, 160, 255))

        # Black peg lower bound
        self.declare_parameter("peg_hsv_lower", [90, 150, 0])
        self.peg_hsv_lower = self._hsv_param("peg_hsv_lower", (90, 150, 0))

        # Black peg upper bound
        self.declare_parameter("peg_hsv_upper", [130, 255, 45])
        self.peg_hsv_upper = self._hsv_param("peg_hsv_upper", (130, 255, 45))

        # Visualization canvas width in pixels (default: 1280)
        self.declare_parameter("canvas_width", 1280)
        self.canvas_width = int(self.get_parameter("canvas_width").value)

        # Visualization canvas height in pixels (default: 720)
        self.declare_parameter("canvas_height", 720)
        self.canvas_height = int(self.get_parameter("canvas_height").value)

        # State publishing frequency in Hz (default: 80.0)
        self.declare_parameter("publish_frequency", 80.0)
        self.publish_frequency = float(self.get_parameter("publish_frequency").value)

        # Seconds between display updates (default: 0.1)
        self.declare_parameter("display_update_interval", 0.1)
        self.display_update_interval = float(
            self.get_parameter("display_update_interval").value
        )

        # Topic names
        self.declare_parameter("board_state_topic", "board_state")
        self.board_state_topic = str(self.get_parameter("board_state_topic").value)

        self.declare_parameter("board_image_topic", "board_image/compressed")
        self.board_image_topic = str(self.get_parameter("board_image_topic").value)

        self.declare_parameter("goal_positions_topic", "goal_positions")
        self.goal_positions_topic = str(
            self.get_parameter("goal_positions_topic").value
        )

        # Ball KF
        self.declare_parameter("ball_kf_process_noise_position", 2.0)
        self.declare_parameter("ball_kf_process_noise_velocity", 30.0)
        self.declare_parameter("ball_kf_measurement_noise_position", 1.0)

        self.ball_kf_process_noise_position = float(
            self.get_parameter("ball_kf_process_noise_position").value
        )
        self.ball_kf_process_noise_velocity = float(
            self.get_parameter("ball_kf_process_noise_velocity").value
        )
        self.ball_kf_measurement_noise_position = float(
            self.get_parameter("ball_kf_measurement_noise_position").value
        )

        # Left peg KF
        self.declare_parameter("left_peg_kf_process_noise_position", 2.0)
        self.declare_parameter("left_peg_kf_process_noise_velocity", 800.0)
        self.declare_parameter("left_peg_kf_measurement_noise_position", 12.0)
        self.declare_parameter("left_peg_kf_stop_threshold", 0.3)

        self.left_peg_kf_process_noise_position = float(
            self.get_parameter("left_peg_kf_process_noise_position").value
        )
        self.left_peg_kf_process_noise_velocity = float(
            self.get_parameter("left_peg_kf_process_noise_velocity").value
        )
        self.left_peg_kf_measurement_noise_position = float(
            self.get_parameter("left_peg_kf_measurement_noise_position").value
        )
        self.left_peg_kf_stop_threshold = float(
            self.get_parameter("left_peg_kf_stop_threshold").value
        )

        # Right peg KF
        self.declare_parameter("right_peg_kf_process_noise_position", 2.0)
        self.declare_parameter("right_peg_kf_process_noise_velocity", 800.0)
        self.declare_parameter("right_peg_kf_measurement_noise_position", 12.0)
        self.declare_parameter("right_peg_kf_stop_threshold", 0.3)

        self.right_peg_kf_process_noise_position = float(
            self.get_parameter("right_peg_kf_process_noise_position").value
        )
        self.right_peg_kf_process_noise_velocity = float(
            self.get_parameter("right_peg_kf_process_noise_velocity").value
        )
        self.right_peg_kf_measurement_noise_position = float(
            self.get_parameter("right_peg_kf_measurement_noise_position").value
        )
        self.right_peg_kf_stop_threshold = float(
            self.get_parameter("right_peg_kf_stop_threshold").value
        )

        # ============================================
        # Playing Field Boundaries
        # ============================================

        self.declare_parameter("edge_x_min", 0.0)
        self.declare_parameter("edge_x_max", 530.0)
        self.declare_parameter("edge_y_min", 0.0)
        self.declare_parameter("edge_y_max", 370.0)

        self.edge_x_min = float(self.get_parameter("edge_x_min").value)
        self.edge_x_max = float(self.get_parameter("edge_x_max").value)
        self.edge_y_min = float(self.get_parameter("edge_y_min").value)
        self.edge_y_max = float(self.get_parameter("edge_y_max").value)

        # Publishers
        self.state_publisher = self.create_publisher(State, self.board_state_topic, 10)

        # Subscribers
        self.image_subscription = self.create_subscription(
            CompressedImage, self.board_image_topic, self._image_callback, 10
        )
        self.goal_subscription = self.create_subscription(
            StampedPolygon, self.goal_positions_topic, self._goal_callback, 10
        )

        # Timer for state publishing
        self.state_timer = self.create_timer(
            1.0 / self.publish_frequency, self._publish_timer_callback
        )

        # CV Bridge for image conversion
        self.bridge = CvBridge()

        # Kalman Filters
        self.ball_kf = KalmanFilter(
            process_noise_position=self.ball_kf_process_noise_position,
            process_noise_velocity=self.ball_kf_process_noise_velocity,
            measurement_noise_position=self.ball_kf_measurement_noise_position,
        )
        self.left_peg_kf = KalmanFilter(
            process_noise_position=self.left_peg_kf_process_noise_position,
            process_noise_velocity=self.left_peg_kf_process_noise_velocity,
            measurement_noise_position=self.left_peg_kf_measurement_noise_position,
            stop_threshold=self.left_peg_kf_stop_threshold,
        )
        self.right_peg_kf = KalmanFilter(
            process_noise_position=self.right_peg_kf_process_noise_position,
            process_noise_velocity=self.right_peg_kf_process_noise_velocity,
            measurement_noise_position=self.right_peg_kf_measurement_noise_position,
            stop_threshold=self.right_peg_kf_stop_threshold,
        )

        # Timing
        self.previous_time: float | None = None
        self.dt: float = 0.01

        # FPS tracking
        self.frame_count = 0
        self.current_fps = 0.0
        self.fps_timer = self.create_timer(1.0, self._update_fps)

        # Display throttling
        self.last_display_time = 0.0

        # Object positions
        self.left_peg_position: tuple[float, float] | None = None
        self.right_peg_position: tuple[float, float] | None = None
        self.ball_position: tuple[float, float] | None = None

        # Goal positions (from camera node)
        self.left_goal: list[float] | None = None
        self.right_goal: list[float] | None = None

        # Playing field boundaries [x_min, x_max, y_min, y_max]
        self.edge = np.array(
            [self.edge_x_min, self.edge_x_max, self.edge_y_min, self.edge_y_max]
        )

        # Goal detection counters
        self.ball_in_left_goal_counter = 0
        self.ball_in_right_goal_counter = 0
        self.peg_in_left_goal_counter = 0
        self.peg_in_right_goal_counter = 0

        # Board Status
        self.board_state: BoardState = BoardState.UNKNOWN

        self.get_logger().info("State estimator node started")

    def _hsv_param(
        self, name: str, default_value: tuple[int, int, int]
    ) -> tuple[int, int, int]:
        value = self.get_parameter(name).value
        try:
            values = [int(v) for v in value]
        except Exception:
            self.get_logger().warn(
                f"Parameter '{name}' must be a list of 3 ints; using default {list(default_value)}"
            )
            return default_value

        if len(values) != 3:
            self.get_logger().warn(
                f"Parameter '{name}' must have length 3; got {len(values)}. Using default {list(default_value)}"
            )
            return default_value

        return (values[0], values[1], values[2])

    def _goal_callback(self, msg: StampedPolygon) -> None:
        """Callback for receiving goal positions."""
        if len(msg.polygon.points) >= 2:
            self.left_goal = [msg.polygon.points[0].x, msg.polygon.points[0].y]
            self.right_goal = [msg.polygon.points[1].x, msg.polygon.points[1].y]

    def _image_callback(self, msg: CompressedImage) -> None:
        """Callback for receiving compressed board images."""
        try:
            # Convert ROS CompressedImage message to OpenCV image
            cv_image = self.bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )

            # Store current image dimensions for EU conversion
            # (image size can vary each frame due to rotation/cropping)
            height, width = cv_image.shape[:2]
            self.current_image_width = float(width)
            self.current_image_height = float(height)

            # Update delta time for Kalman filters
            current_time = time.time()
            self.dt = (
                current_time - self.previous_time
                if self.previous_time is not None
                else 0.01
            )
            self.previous_time = current_time

            # Detect ball and pegs
            (
                self.left_peg_position,
                self.right_peg_position,
                self.ball_position,
            ) = self._detect_ball_peg(
                cv_image,
                left_goal=self.left_goal,
                right_goal=self.right_goal,
            )

            # Check for goals if all objects are detected
            if all(
                [
                    self.ball_position,
                    self.left_peg_position,
                    self.right_peg_position,
                    self.left_goal,
                    self.right_goal,
                ]
            ):
                self.board_state |= BoardState.READY
                self._check_goal()

            # Increment frame counter
            self.frame_count += 1

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

    def _detect_ball_peg(
        self,
        frame: np.ndarray,
        left_goal: list[float] | None = None,
        right_goal: list[float] | None = None,
    ) -> tuple[
        np.ndarray,
        tuple[float, float] | None,
        tuple[float, float] | None,
        tuple[float, float] | None,
    ]:
        """
        Detect ball and pegs using HSV color filtering and Kalman filtering.

        Returns:
            Tuple of (canvas, left_peg_position, right_peg_position, ball_position)
        """
        # Convert to HSV and apply blur for noise reduction
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_hsv_blur = cv2.GaussianBlur(frame_hsv, (7, 7), 0)

        # Create masks for ball (orange) and pegs (black)
        masked_ball = cv2.inRange(
            frame_hsv_blur, self.ball_hsv_lower, self.ball_hsv_upper
        )
        masked_peg = cv2.inRange(frame_hsv_blur, self.peg_hsv_lower, self.peg_hsv_upper)

        # Create visualization overlay
        overlaid_frame = self._create_overlay(frame, masked_peg, masked_ball)

        # Detect pegs in left and right halves
        height, width = masked_peg.shape
        left_half = masked_peg[:, : width // 2]
        right_half = masked_peg[:, width // 2 :]

        left_peg_position = self._detect_peg(
            left_half, self.left_peg_kf, 0, overlaid_frame, (255, 0, 0), (248, 193, 110)
        )
        right_peg_position = self._detect_peg(
            right_half,
            self.right_peg_kf,
            width // 2,
            overlaid_frame,
            (0, 100, 0),
            (144, 238, 144),
        )

        # Detect ball with collision detection
        ball_position = self._detect_ball(
            masked_ball, left_peg_position, right_peg_position, overlaid_frame
        )

        # Display image at set rate
        if self.show_image:
            current_time = time.time()
            if current_time - self.last_display_time >= self.display_update_interval:
                plot_image(
                    overlaid_frame,
                    left_goal,
                    right_goal,
                    self.goal_radius,
                    self.canvas_width,
                    self.canvas_height,
                    self.current_fps,
                )
                self.last_display_time = current_time

        return left_peg_position, right_peg_position, ball_position

    def _create_overlay(
        self, frame: np.ndarray, masked_peg: np.ndarray, masked_ball: np.ndarray
    ) -> np.ndarray:
        """Create visualization overlay with masks."""
        mask_3_channel_peg = cv2.cvtColor(masked_peg, cv2.COLOR_GRAY2BGR)
        mask_3_channel_ball = cv2.cvtColor(masked_ball, cv2.COLOR_GRAY2BGR)

        overlaid = cv2.addWeighted(frame, 1.0, mask_3_channel_peg, 0.3, 0)
        overlaid = cv2.addWeighted(overlaid, 1.0, mask_3_channel_ball, 0.3, 0)
        return overlaid

    def _detect_peg(
        self,
        half_mask: np.ndarray,
        kf: KalmanFilter,
        x_offset: int,
        overlaid_frame: np.ndarray,
        color: tuple[int, int, int],
        velocity_color: tuple[int, int, int],
    ) -> tuple[float, float] | None:
        """Detect peg in half of the frame."""
        contours, _ = cv2.findContours(
            half_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None

        # Calculate centroid
        cX = int(M["m10"] / M["m00"]) + x_offset
        cY = int(M["m01"] / M["m00"])

        # Apply Kalman filter
        kf.predict(self.dt)
        kf.update([cX, cY])

        position = kf.get_position()

        if self.show_image:
            velocity = kf.get_velocity()
            draw_object_with_velocity(
                overlaid_frame, position, velocity, color, velocity_color, 0.5
            )

        return (float(position[0]), float(position[1]))

    def _detect_ball(
        self,
        masked_ball: np.ndarray,
        left_peg_position: tuple[float, float] | None,
        right_peg_position: tuple[float, float] | None,
        overlaid_frame: np.ndarray,
    ) -> tuple[float, float] | None:
        """Detect ball with collision-aware Kalman filtering."""
        contours, _ = cv2.findContours(
            masked_ball, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)

        if M["m00"] == 0:
            return None

        # Calculate centroid
        cX_ball = int(M["m10"] / M["m00"])
        cY_ball = int(M["m01"] / M["m00"])

        # Detect potential collisions
        x_collision, y_collision = self._detect_collisions(
            cX_ball, cY_ball, left_peg_position, right_peg_position
        )

        # Apply Kalman filter with collision awareness
        self.ball_kf.predict(self.dt, x_collision=x_collision, y_collision=y_collision)
        self.ball_kf.update([cX_ball, cY_ball])

        position = self.ball_kf.get_position()

        if self.show_image:
            velocity = self.ball_kf.get_velocity()
            draw_object_with_velocity(
                overlaid_frame, position, velocity, (0, 0, 255), (0, 165, 255), 0.2
            )

        return (float(position[0]), float(position[1]))

    def _detect_collisions(
        self,
        ball_x: float,
        ball_y: float,
        left_peg_position: tuple[float, float] | None,
        right_peg_position: tuple[float, float] | None,
    ) -> tuple[bool, bool]:
        """Detect if ball is near edges or pegs (potential collision)."""
        at_x_edge = (
            ball_x < self.edge[0] + self.collision_distance
            or ball_x + self.collision_distance > self.edge[1]
        )
        at_y_edge = (
            ball_y < self.edge[2] + self.collision_distance
            or ball_y + self.collision_distance > self.edge[3]
        )

        close_to_peg = False
        if left_peg_position and right_peg_position:
            dist_left = np.linalg.norm(
                np.array([ball_x, ball_y]) - np.array(left_peg_position)
            )
            dist_right = np.linalg.norm(
                np.array([ball_x, ball_y]) - np.array(right_peg_position)
            )
            close_to_peg = (
                dist_left < self.collision_distance
                or dist_right < self.collision_distance
            )

        # Determine collision type
        if close_to_peg or (at_x_edge and at_y_edge):
            return True, True
        elif at_y_edge:
            return False, True
        elif at_x_edge:
            return True, True
        else:
            return False, False

    def _check_goal(self) -> None:
        """Check if ball or peg has scored and publish outcome."""

        data = [
            (self.ball_position, self.left_goal, self.ball_in_left_goal_counter),
            (self.ball_position, self.right_goal, self.ball_in_right_goal_counter),
            (self.left_peg_position, self.left_goal, self.peg_in_left_goal_counter),
            (self.right_peg_position, self.right_goal, self.peg_in_right_goal_counter),
        ]
        (
            self.ball_in_left_goal_counter,
            self.ball_in_right_goal_counter,
            self.peg_in_left_goal_counter,
            self.peg_in_right_goal_counter,
        ) = [
            self._update_goal_counter(
                np.array(obj_pos),
                np.array(goal_pos),
                counter,
            )
            for obj_pos, goal_pos, counter in data
        ]

        for counter, flag in [
            (self.ball_in_left_goal_counter, BoardState.BALL_IN_LEFT_GOAL),
            (self.ball_in_right_goal_counter, BoardState.BALL_IN_RIGHT_GOAL),
            (self.peg_in_left_goal_counter, BoardState.PEG_IN_LEFT_GOAL),
            (self.peg_in_right_goal_counter, BoardState.PEG_IN_RIGHT_GOAL),
        ]:
            if counter == self.goal_hyst_counter:
                self.board_state |= flag
            elif counter == 0:
                self.board_state &= ~flag

    def _update_goal_counter(
        self, object_pos: np.ndarray, goal_pos: np.ndarray, counter: int
    ) -> int:
        """Update goal counter based on distance."""

        distance = np.linalg.norm(object_pos - goal_pos)
        if distance < self.goal_radius:
            return min(counter + 1, self.goal_hyst_counter)
        elif counter > 0:
            return counter - 1
        return counter

    def _update_fps(self) -> None:
        """Update FPS calculation (called every second by timer)."""
        self.current_fps = float(self.frame_count)
        self.frame_count = 0

    def _get_pixel_to_meter_x(self) -> float:
        """Get current pixel-to-meter conversion factor for X axis."""
        if self.current_image_width > 0:
            return self.board_width_meters / self.current_image_width
        return 0.0

    def _get_pixel_to_meter_y(self) -> float:
        """Get current pixel-to-meter conversion factor for Y axis."""
        if self.current_image_height > 0:
            return self.board_height_meters / self.current_image_height
        return 0.0

    def _convert_position_to_eu(
        self, position: list[float] | np.ndarray
    ) -> list[float]:
        """Convert position from pixels to engineering units (meters)."""
        return [
            position[0] * self._get_pixel_to_meter_x(),
            position[1] * self._get_pixel_to_meter_y(),
        ]

    def _convert_velocity_to_eu(
        self, velocity: list[float] | np.ndarray
    ) -> list[float]:
        """Convert velocity from pixels/s to engineering units (m/s)."""
        return [
            velocity[0] * self._get_pixel_to_meter_x(),
            velocity[1] * self._get_pixel_to_meter_y(),
        ]

    def _publish_timer_callback(self) -> None:
        """Publish estimated positions/velocities in EU (meters, m/s)."""

        if not (self.board_state & BoardState.READY):
            return

        # Wait until we have valid image dimensions for conversion
        if self.current_image_width <= 0 or self.current_image_height <= 0:
            return

        # Get Kalman filter states (still in pixel space)
        ball_pos_px = self.ball_kf.get_position()
        ball_vel_px = self.ball_kf.get_velocity()
        left_peg_pos_px = self.left_peg_kf.get_position()
        left_peg_vel_px = self.left_peg_kf.get_velocity()
        right_peg_pos_px = self.right_peg_kf.get_position()
        right_peg_vel_px = self.right_peg_kf.get_velocity()

        # Convert to engineering units (meters, m/s)
        ball_pos_eu = self._convert_position_to_eu(ball_pos_px)
        ball_vel_eu = self._convert_velocity_to_eu(ball_vel_px)
        left_peg_pos_eu = self._convert_position_to_eu(left_peg_pos_px)
        left_peg_vel_eu = self._convert_velocity_to_eu(left_peg_vel_px)
        right_peg_pos_eu = self._convert_position_to_eu(right_peg_pos_px)
        right_peg_vel_eu = self._convert_velocity_to_eu(right_peg_vel_px)

        # Convert goal positions to EU
        left_goal_eu = self._convert_position_to_eu(self.left_goal)
        right_goal_eu = self._convert_position_to_eu(self.right_goal)

        # Build message with EU values
        msg = State()
        msg.ball.position = create_point_from_list(ball_pos_eu)
        msg.ball.velocity = create_point_from_list(ball_vel_eu)
        msg.left_peg.position = create_point_from_list(left_peg_pos_eu)
        msg.left_peg.velocity = create_point_from_list(left_peg_vel_eu)
        msg.right_peg.position = create_point_from_list(right_peg_pos_eu)
        msg.right_peg.velocity = create_point_from_list(right_peg_vel_eu)
        msg.left_goal_pos = create_point_from_list(left_goal_eu)
        msg.right_goal_pos = create_point_from_list(right_goal_eu)

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.status = UInt64(data=int(self.board_state))

        self.state_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    state_estimator = StateEstimatorNode()

    try:
        rclpy.spin(state_estimator)
    except KeyboardInterrupt:
        pass
    finally:
        state_estimator.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
