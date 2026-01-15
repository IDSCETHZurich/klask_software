"""Kalman Filter implementation for tracking position and velocity in 2D space."""

import numpy as np


class KalmanFilter:
    """Kalman Filter for tracking position and velocity in 2D space."""

    # Collision noise constants
    COLLISION_POSITION_NOISE = 4.0
    COLLISION_VELOCITY_NOISE = 10000.0
    DEFAULT_VELOCITY_THRESHOLD = 1.0

    def __init__(
        self,
        process_noise_position: float = 1.0,
        process_noise_velocity: float = 1.0,
        measurement_noise_position: float = 1.0,
        stop_threshold: float = -1.0,
        velocity_threshold: float = DEFAULT_VELOCITY_THRESHOLD,
    ):
        """Initialize Kalman Filter.

        Args:
            process_noise_position: Process noise for position
            process_noise_velocity: Process noise for velocity
            measurement_noise_position: Measurement noise for position
            stop_threshold: Threshold for detecting stopped motion (-1 to disable)
            velocity_threshold: Minimum velocity magnitude to consider non-zero
        """
        # State vector: [pos_x, pos_y, vel_x, vel_y]
        self.state = np.zeros(4)
        self.stop_threshold = stop_threshold
        self.velocity_threshold = velocity_threshold
        self.previous_position = None

        # Process noise covariance matrix
        self.Q = np.diag(
            [
                process_noise_position,
                process_noise_position,
                process_noise_velocity,
                process_noise_velocity,
            ]
        )

        # Measurement noise covariance matrix
        self.R = measurement_noise_position * np.eye(2)

        # Measurement matrix (observe positions only)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        # Initial estimation error covariance
        self.P = np.eye(4)

        # State transition matrix (will be updated with dt)
        self.F = np.eye(4)

    def predict(self, dt: float, x_collision: bool = False, y_collision: bool = False) -> None:
        """Predict the next state based on the motion model.

        Args:
            dt: Time step since last prediction
            x_collision: Whether collision occurred in x-direction
            y_collision: Whether collision occurred in y-direction
        """
        # Update state transition matrix with time step
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

        # Adjust process noise for collisions
        Q = self.Q.copy()
        if x_collision:
            Q += np.diag([self.COLLISION_POSITION_NOISE, 0.0, self.COLLISION_VELOCITY_NOISE, 0.0])
        if y_collision:
            Q += np.diag([0.0, self.COLLISION_POSITION_NOISE, 0.0, self.COLLISION_VELOCITY_NOISE])

        # Prediction step
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + Q

    def update(self, measurement: tuple[float, float]) -> None:
        """Update state with new measurement.

        Args:
            measurement: Measured position (x, y)
        """
        z = np.array(measurement)
        y = z - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        if self.stop_threshold > 0:
            self._check_stop_threshold()

    def get_position(self) -> np.ndarray:
        """Get current estimated position."""
        return self.state[0:2]

    def get_velocity(self) -> list[float]:
        """Get current estimated velocity with threshold filtering."""
        v_x = self.state[2] if abs(self.state[2]) >= self.velocity_threshold else 0.0
        v_y = self.state[3] if abs(self.state[3]) >= self.velocity_threshold else 0.0
        return [v_x, v_y]

    def _check_stop_threshold(self) -> None:
        """Check if object has stopped moving based on position change."""
        current_position = self.get_position()

        if self.previous_position is None:
            self.previous_position = current_position
            return

        displacement = np.linalg.norm(current_position - self.previous_position)

        if displacement < self.stop_threshold:
            self.state[2:4] = 0.0  # Set velocities to zero

        self.previous_position = current_position
