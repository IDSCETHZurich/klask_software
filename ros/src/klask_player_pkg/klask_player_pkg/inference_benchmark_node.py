"""ROS2 node for benchmarking policy inference speed."""

import time
import rclpy
import numpy as np
from pathlib import Path
from rclpy.node import Node
from .policy_inference import PolicyInference
from klask_interfaces.msg import State, ObjectState
from geometry_msgs.msg import Point
from std_msgs.msg import UInt64


class InferenceBenchmark(Node):
    """ROS2 node that benchmarks policy network inference time."""

    def __init__(self):
        """Initialize the inference benchmark node."""
        super().__init__("inference_benchmark_node")

        # Benchmark parameters
        self.declare_parameter("num_steps", 1000)
        self.declare_parameter("warmup_steps", 100)

        # Model parameters (same as player_node)
        self.declare_parameter("weights_filename", "klask_ac_nn_v0.0.pth")
        self.declare_parameter(
            "weights_zip_url",
            "https://polybox.ethz.ch/index.php/s/joEoP8GgQbwmToW/download",
        )
        self.declare_parameter("device", "cpu")
        self.declare_parameter("board_width", 0.42)
        self.declare_parameter("board_height", 0.32)
        self.declare_parameter("player_side", "left")
        self.declare_parameter("clip_actions", 0.2)
        self.declare_parameter("enable_action_rescaling", True)

        # Get parameters
        self.num_steps = self.get_parameter("num_steps").value
        self.warmup_steps = self.get_parameter("warmup_steps").value
        self.board_width = self.get_parameter("board_width").value
        self.board_height = self.get_parameter("board_height").value

        # Initialize policy inference
        self.policy = PolicyInference(
            weights_filename=self.get_parameter("weights_filename").value,
            weights_zip_url=self.get_parameter("weights_zip_url").value,
            nn_weights_dir=Path(__file__).resolve().parent.parent.parent.parent / "nn_weights",
            device=self.get_parameter("device").value,
            clip_actions=self.get_parameter("clip_actions").value,
            enable_action_rescaling=self.get_parameter("enable_action_rescaling").value,
            player_side=self.get_parameter("player_side").value,
            board_dim_width=self.board_width,
            board_dim_height=self.board_height,
            logger=self.get_logger(),
        )

        # Schedule benchmark to run after node is fully initialized
        self._timer = self.create_timer(0.0, self._run_benchmark)

    def _create_random_state(self):
        """Create a State message with random but physically plausible values."""
        msg = State()

        # Random positions within board dimensions, velocities in [-0.5, 0.5] m/s
        for obj_name in ["ball", "left_peg", "right_peg"]:
            obj = ObjectState()
            obj.position = Point(
                x=float(np.random.uniform(0.0, self.board_width)),
                y=float(np.random.uniform(0.0, self.board_height)),
                z=0.0,
            )
            obj.velocity = Point(
                x=float(np.random.uniform(-0.5, 0.5)),
                y=float(np.random.uniform(-0.5, 0.5)),
                z=0.0,
            )
            setattr(msg, obj_name, obj)

        # Fixed goal positions (board center-line at each end)
        msg.left_goal_pos = Point(
            x=0.0,
            y=float(self.board_height / 2.0),
            z=0.0,
        )
        msg.right_goal_pos = Point(
            x=float(self.board_width),
            y=float(self.board_height / 2.0),
            z=0.0,
        )

        msg.status = UInt64(data=1)  # READY flag

        return msg

    def _run_benchmark(self):
        """Run the inference benchmark and log results."""
        self._timer.cancel()

        total_steps = self.warmup_steps + self.num_steps

        self.get_logger().info("=" * 60)
        self.get_logger().info("Inference Benchmark")
        self.get_logger().info(f"  Weights: {self.get_parameter('weights_filename').value}")
        self.get_logger().info(f"  Device: {self.get_parameter('device').value}")
        self.get_logger().info(f"  Warmup steps: {self.warmup_steps}")
        self.get_logger().info(f"  Measured steps: {self.num_steps}")
        self.get_logger().info("=" * 60)

        # Pre-generate all random messages to avoid measuring allocation time
        messages = [self._create_random_state() for _ in range(total_steps)]

        # Run warmup phase
        warmup_start = time.perf_counter()
        timings = []
        for i in range(self.warmup_steps):
            start = time.perf_counter()
            self.policy.get_action(messages[i])
            elapsed = time.perf_counter() - start
            timings.append(elapsed)
        warmup_duration = time.perf_counter() - warmup_start

        # Run measured phase
        measurement_start = time.perf_counter()
        for i in range(self.warmup_steps, total_steps):
            start = time.perf_counter()
            self.policy.get_action(messages[i])
            elapsed = time.perf_counter() - start
            timings.append(elapsed)
        measurement_duration = time.perf_counter() - measurement_start

        # Split warmup and measured timings
        measured = np.array(timings[self.warmup_steps :])

        # Convert to milliseconds for display
        measured_ms = measured * 1000.0

        mean_ms = np.mean(measured_ms)
        std_ms = np.std(measured_ms)
        min_ms = np.min(measured_ms)
        max_ms = np.max(measured_ms)
        median_ms = np.median(measured_ms)
        freq_hz = 1000.0 / mean_ms

        self.get_logger().info("")
        self.get_logger().info("Results:")
        self.get_logger().info(f"  Mean:   {mean_ms:.4f} ms")
        self.get_logger().info(f"  Std:    {std_ms:.4f} ms")
        self.get_logger().info(f"  Median: {median_ms:.4f} ms")
        self.get_logger().info(f"  Min:    {min_ms:.4f} ms")
        self.get_logger().info(f"  Max:    {max_ms:.4f} ms")
        self.get_logger().info(f"  Freq:   {freq_hz:.1f} Hz")
        self.get_logger().info("")
        self.get_logger().info("Duration:")
        self.get_logger().info(f"  Warmup:      {warmup_duration:.3f} s")
        self.get_logger().info(f"  Measurement: {measurement_duration:.3f} s")
        self.get_logger().info("=" * 60)

        raise SystemExit


def main(args=None):
    """Main entry point for the inference benchmark node."""
    rclpy.init(args=args)

    try:
        node = InferenceBenchmark()
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
