"""Utility functions for Klask player package."""

import numpy as np
import os
import urllib.request
import zipfile
import tempfile
from klask_interfaces.msg import State


def map_state_observations(msg: State, player_side: str, board_dim_width: float, board_dim_height: float) -> np.ndarray:
    """Extract observations from State message, center coordinates, and transform for player side.
    
    It transforms form image frame (center top-left, y-down) to ego-centric frame (center-origin, y-forward) and maps player/opponent based on player_side.

    Returns:
        16-dimensional centered observation array ready for feature computation.
        Format: [player_pos, player_vel, opponent_pos, opponent_vel,
                 ball_pos, ball_vel, goal_player_pos, goal_opponent_pos]
    """
    # Extract raw observations (engineering units: meters and m/s)
    right_peg_pos = np.array([msg.right_peg.position.y, msg.right_peg.position.x], dtype=np.float32)
    right_peg_vel = np.array([msg.right_peg.velocity.y, msg.right_peg.velocity.x], dtype=np.float32)
    left_peg_pos = np.array([msg.left_peg.position.y, msg.left_peg.position.x], dtype=np.float32)
    left_peg_vel = np.array([msg.left_peg.velocity.y, msg.left_peg.velocity.x], dtype=np.float32)
    ball_pos = np.array([msg.ball.position.y, msg.ball.position.x], dtype=np.float32)
    ball_vel = np.array([msg.ball.velocity.y, msg.ball.velocity.x], dtype=np.float32)
    right_goal_pos = np.array([msg.right_goal_pos.y, msg.right_goal_pos.x], dtype=np.float32)
    left_goal_pos = np.array([msg.left_goal_pos.y, msg.left_goal_pos.x], dtype=np.float32)

    # Center all positions to board origin
    center_offset = np.array([board_dim_height / 2.0, board_dim_width / 2.0], dtype=np.float32)
    right_peg_pos -= center_offset
    left_peg_pos -= center_offset
    ball_pos -= center_offset
    right_goal_pos -= center_offset
    left_goal_pos -= center_offset

    # Map to player/opponent based on player_side
    if player_side == "left":
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


def add_additional_state_features(obs: np.ndarray) -> np.ndarray:
    """Add geometric features (angles and distances) to observations.

    Input: 16 values [player_pos, player_vel, opp_pos, opp_vel,
                      ball_pos, ball_vel, goal_player_pos, goal_opponent_pos]
    Returns: 8 additional features:
             In total 20 values [player_pos, player_vel, opp_pos, opp_vel, ball_pos, ball_vel, 8 features]

    Note: All coordinates are already in ego frame and in engineering units (meters and m/s).
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
    angle_pegball_pegoppgoal = _angle_between_vectors(vec_to_ball, vec_to_opp_goal)
    angle_oppball_oppgoal = _angle_between_vectors(vec_opp_to_goal, vec_ball_to_opp)
    angle_pegball_pegopp = _angle_between_vectors(vec_to_ball, vec_opp_to_player)
    angle_oppball_pegopp = _angle_between_vectors(vec_ball_to_opp, -vec_opp_to_player)

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
    full_obs = np.concatenate([obs[:12], extra_features])

    return full_obs


def _angle_between_vectors(v1, v2):
    """Compute angle between two vectors."""
    dot = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot / (norm_v1 * norm_v2 + 1e-8)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    return angle_rad


def ensure_weights_available(weights_filename, zip_url, nn_weights_dir, logger):
    """Ensure weights file exists, download and extract zip if necessary."""
    # nn_weights_dir is already a Path object
    weights_dir = nn_weights_dir
    weights_path = weights_dir / weights_filename

    # Check if the specific weights file exists
    if weights_path.exists():
        logger.info(f"Weights file found: {weights_path}")
        return weights_path

    # Download and extract the zip file
    logger.info(f"Weights file not found. Downloading weights zip from: {zip_url}")

    # Clean existing contents if directory exists
    if weights_dir.exists():
        import shutil

        logger.info(f"Cleaning existing weights directory contents: {weights_dir}")
        for item in weights_dir.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception as e:
                logger.warn(f"Could not remove {item}: {e}")
    else:
        weights_dir.mkdir(parents=True, exist_ok=True)

    download_and_extract_weights(zip_url, weights_dir, logger)

    # Verify the weights file exists after extraction
    if weights_path.exists():
        logger.info(f"Weights extracted successfully: {weights_path}")
        return weights_path
    else:
        # List available files
        available_files = [f.name for f in weights_dir.glob("*.pth")]
        error_msg = f"Weights file '{weights_filename}' not found after extraction. Available files: {available_files}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)


def download_and_extract_weights(url, destination_dir, logger):
    """Download weights zip file and extract to destination directory."""
    try:
        # Create a temporary file for the zip download
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
            tmp_path = tmp_file.name

            logger.info(f"Downloading weights zip to temporary file: {tmp_path}")

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
                        logger.info(f"Downloaded {total_size / 1024 / 1024:.2f} MB...")

            logger.info(f"Download complete: {total_size / 1024 / 1024:.2f} MB")

        # Extract the zip file
        logger.info(f"Extracting weights to: {destination_dir}")
        with zipfile.ZipFile(tmp_path, "r") as zip_ref:
            zip_ref.extractall(destination_dir)
            extracted_files = zip_ref.namelist()
            logger.info(f"Extracted {len(extracted_files)} files")
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
        logger.info("Extraction complete")

    except Exception as e:
        logger.error(f"Failed to download and extract weights: {e}")
        # Clean up temporary file if it exists
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
