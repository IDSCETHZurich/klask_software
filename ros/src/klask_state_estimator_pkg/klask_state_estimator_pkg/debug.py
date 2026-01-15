"""Debugging utilities for Klask state estimator visualization."""

import cv2
import numpy as np


def draw_object_with_velocity(
    frame: np.ndarray,
    position: np.ndarray,
    velocity: list[float],
    dot_color: tuple[int, int, int],
    arrow_color: tuple[int, int, int],
    velocity_scale: float,
) -> None:
    """Draw object position and velocity vector."""
    pos_int = (int(position[0]), int(position[1]))

    # Draw position
    cv2.circle(frame, pos_int, 5, dot_color, -1)

    # Draw velocity arrow
    end_point = (
        int(position[0] + velocity[0] * velocity_scale),
        int(position[1] + velocity[1] * velocity_scale),
    )
    cv2.arrowedLine(frame, pos_int, end_point, arrow_color, 2, tipLength=0.3)


def draw_goals(
    frame: np.ndarray,
    left_goal: list[float] | None,
    right_goal: list[float] | None,
    goal_radius: int,
) -> None:
    """Draw goal circles on the frame."""
    goal_color = (140, 255, 0)

    if left_goal is not None:
        center = (int(left_goal[0]), int(left_goal[1]))
        cv2.circle(frame, center, goal_radius, goal_color, 2)
        cv2.circle(frame, center, 5, goal_color, -1)

    if right_goal is not None:
        center = (int(right_goal[0]), int(right_goal[1]))
        cv2.circle(frame, center, goal_radius, goal_color, 2)
        cv2.circle(frame, center, 5, goal_color, -1)


def plot_image(
    overlaid_frame: np.ndarray,
    left_goal: list[float],
    right_goal: list[float],
    goal_radius: int,
    canvas_width: int,
    canvas_height: int,
    current_fps: float,
) -> None:
    """Plot the overlaid frame with goals on a canvas and display it."""
    # Draw goals
    draw_goals(overlaid_frame, left_goal, right_goal, goal_radius)

    # Resize and center on canvas
    canvas = create_canvas(overlaid_frame, canvas_width, canvas_height, current_fps)

    cv2.imshow("State Estimator", canvas)
    cv2.waitKey(1)


def create_canvas(
    overlaid_frame: np.ndarray,
    canvas_width: int,
    canvas_height: int,
    current_fps: float,
) -> np.ndarray:
    """Resize frame and center it on a canvas."""
    resized_image, new_w, new_h = resize_with_aspect_ratio(overlaid_frame, canvas_width, canvas_height)

    # Ensure dimensions are within bounds
    if new_w > canvas_width or new_h > canvas_height:
        new_w = min(new_w, canvas_width)
        new_h = min(new_h, canvas_height)
        resized_image = cv2.resize(resized_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center on black canvas
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    x_offset = (canvas_width - new_w) // 2
    y_offset = (canvas_height - new_h) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized_image

    # Draw FPS counter on canvas
    if current_fps > 0:
        fps_text = f"FPS: {current_fps:.1f}"
        cv2.putText(
            canvas,
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return canvas


def resize_with_aspect_ratio(image: np.ndarray, target_width: int, target_height: int) -> tuple[np.ndarray, int, int]:
    """Resize image while maintaining aspect ratio to fit within target dimensions.

    Args:
        image: Input image
        target_width: Maximum width
        target_height: Maximum height

    Returns:
        Tuple of (resized_image, actual_width, actual_height)
    """
    h, w = image.shape[:2]
    aspect_ratio = w / h
    target_aspect_ratio = target_width / target_height

    # Determine which dimension is the limiting factor
    if aspect_ratio > target_aspect_ratio:
        # Width is limiting
        new_w = target_width
        new_h = int(target_width / aspect_ratio)
    else:
        # Height is limiting
        new_h = target_height
        new_w = int(target_height * aspect_ratio)

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_image, new_w, new_h
