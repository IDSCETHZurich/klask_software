"""Dreamer policy inference class — handles network loading and stateful inference."""

import json
import os
import shutil
import tempfile
import urllib.request
import zipfile

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Any

from .dreamer_config import DreamerModelConfig
from .dreamer_network import MultiEncoder, RSSM, MLPHead, to_f32
from klask_interfaces.msg import State


class DreamerInference:
    """Handle Dreamer network loading and stateful RSSM inference."""

    def __init__(
        self,
        weights_filename: str,
        weights_zip_url: str,
        nn_weights_dir: Path,
        device: str = "cpu",
        player_side: str = "left",
        board_dim_width: float = 0.42,
        board_dim_height: float = 0.32,
        config_filename: str = "",
        logger: Optional[Any] = None,
    ):
        """Initialize Dreamer inference.

        Args:
            weights_filename: Name of the weights file to load (.pt).
            weights_zip_url: URL to download weights if not found locally.
            nn_weights_dir: Directory to store/load neural network weights.
            device: Device to run inference on ('cpu' or 'cuda').
            player_side: 'left' or 'right' player side.
            board_dim_width: Board width in meters.
            board_dim_height: Board height in meters.
            config_filename: Name of the config JSON file in nn_weights_dir.
                If empty, looks for '<checkpoint_stem>_config.json' next to the checkpoint.
            logger: Optional logger object with info(), warn(), error() methods.
        """
        self.logger = logger
        self.device = device
        self.player_side = player_side
        self.board_dim_width = board_dim_width
        self.board_dim_height = board_dim_height
        self.act_dim = 2  # 2D velocity actions

        # Ensure weights are available
        checkpoint_path = self._ensure_weights_available(weights_filename, weights_zip_url, nn_weights_dir)

        # Load model config (also sets self.image_size, self.obs_mode, self.max_velocity)
        config = self._load_config(checkpoint_path, config_filename)

        # Build model components (encoder, RSSM, actor only — no decoder/value/reward)
        self._build_model(config)

        # Load checkpoint weights
        self._load_checkpoint(checkpoint_path)

        # Move to device and set eval mode
        self.encoder.to(self.device)
        self.rssm.to(self.device)
        self.actor.to(self.device)
        self.encoder.eval()
        self.rssm.eval()
        self.actor.eval()

        # Initialize RSSM state
        self.reset_state()

        self.logger.info(f"Dreamer inference initialized with checkpoint: {checkpoint_path}")
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Obs mode: {self.obs_mode}, image size: {self.image_size}x{self.image_size}")
        self.logger.info(f"Max velocity: {self.max_velocity} m/s")
        self.logger.info(f"Player side: {self.player_side}")

    def _load_config(self, checkpoint_path, config_filename):
        """Load model config from JSON file.

        Also reads image_size, obs_mode, and max_velocity from the config and
        sets them as instance attributes so they don't need to be passed separately.

        Resolution order:
        1. If config_filename is provided, look for it in the same directory as the checkpoint.
        2. Otherwise, look for <checkpoint_stem>_config.json next to the checkpoint.
           e.g. dreamer_inference.pt -> dreamer_inference_config.json
        """
        checkpoint_dir = checkpoint_path.parent

        if config_filename:
            config_path = checkpoint_dir / config_filename
        else:
            config_path = checkpoint_dir / f"{checkpoint_path.stem}_config.json"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Dreamer config not found: {config_path}. "
                f"Place the config JSON next to the checkpoint or set 'dreamer_config_filename'."
            )

        self.logger.info(f"Loading Dreamer config from: {config_path}")

        # Read raw JSON to extract env/training params
        with open(config_path, "r") as f:
            raw = json.load(f)

        self.image_size = raw.get("image_size", 64)
        self.obs_mode = raw.get("obs_mode", "image")
        self.max_velocity = raw.get("max_velocity", 0.6)

        self.logger.info(
            f"Config env params — image_size: {self.image_size}, "
            f"obs_mode: {self.obs_mode}, max_velocity: {self.max_velocity}"
        )

        return DreamerModelConfig.from_json(str(config_path), device=self.device)

    def _build_model(self, config):
        """Build encoder, RSSM, and actor from config."""
        # Define observation shapes based on obs_mode
        obs_shapes = {}
        obs_shapes["image"] = (self.image_size, self.image_size, 3)
        if self.obs_mode == "image_and_state":
            obs_shapes["policy"] = (20,)

        # Build encoder
        self.encoder = MultiEncoder(config.encoder, obs_shapes)
        embed_size = self.encoder.out_dim

        # Build RSSM
        self.rssm = RSSM(config.rssm, embed_size, self.act_dim)

        # Build actor
        config.actor.shape = [self.act_dim]
        self.actor = MLPHead(config.actor, self.rssm.feat_size)

        self.logger.info(
            f"Model built: encoder out_dim={embed_size}, "
            f"RSSM feat_size={self.rssm.feat_size}, act_dim={self.act_dim}"
        )

    def reset_state(self):
        """Reset RSSM hidden state for a new episode."""
        stoch, deter = self.rssm.initial(1)
        self._stoch = stoch.to(self.device)
        self._deter = deter.to(self.device)
        self._prev_action = torch.zeros(1, self.act_dim, dtype=torch.float32, device=self.device)
        self._is_first = True

    def get_action(self, msg: State, image: np.ndarray = None) -> np.ndarray:
        """Get action from State message and optional camera image.

        Args:
            msg: ROS State message with board state information.
            image: BGR uint8 image from camera (via CvBridge). Required.

        Returns:
            2D velocity action array [action_y, action_x].
        """
        if image is None:
            self.logger.warn("No image available for Dreamer inference, returning zero action")
            return np.array([0.0, 0.0])

        # Build observation dict
        obs_dict = self._build_obs_dict(msg, image)

        # Run inference
        action = self._predict(obs_dict)

        # Negate action for right player (flip direction)
        if self.player_side == "right":
            action = -action

        return action

    def _build_obs_dict(self, msg: State, image: np.ndarray) -> dict:
        """Build observation dictionary for the Dreamer encoder.

        Args:
            msg: ROS State message.
            image: BGR uint8 image from camera.

        Returns:
            Dict with "image" tensor and optionally "policy" tensor.
        """
        obs = {}

        # Process image: BGR → RGB, resize, package as tensor
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        # (H, W, 3) uint8 → (1, H, W, 3) float32 [0, 255]
        image_tensor = torch.from_numpy(image_resized).unsqueeze(0).to(dtype=torch.float32, device=self.device)
        obs["image"] = image_tensor

        # Optionally add state observations
        if self.obs_mode == "image_and_state":
            obs_base = self._map_observations(msg)
            obs_full = self._add_additional_features(obs_base)
            policy_tensor = torch.from_numpy(obs_full).unsqueeze(0).to(dtype=torch.float32, device=self.device)
            obs["policy"] = policy_tensor

        return obs

    def _predict(self, obs_dict: dict) -> np.ndarray:
        """Run Dreamer inference: encode → RSSM step → actor."""
        with torch.no_grad():
            # Preprocess: images to [0, 1]
            if "image" in obs_dict:
                obs_dict["image"] = obs_dict["image"] / 255.0

            # Encode observations
            embed = self.encoder(obs_dict)

            # is_first flag as tensor
            is_first = torch.tensor([[self._is_first]], dtype=torch.bool, device=self.device)

            # RSSM posterior step
            stoch, deter, _ = self.rssm.obs_step(
                self._stoch, self._deter, self._prev_action, embed, is_first
            )

            # Get features and action distribution
            feat = self.rssm.get_feat(stoch, deter)
            action_dist = self.actor(feat)
            action = action_dist.mode  # deterministic: tanh(mean) ∈ [-1, 1]

            # Update persistent state
            self._stoch = stoch
            self._deter = deter
            self._prev_action = action
            self._is_first = False

        # Scale action by max_velocity
        scaled_action = (action * self.max_velocity).cpu().numpy().flatten()
        return scaled_action

    # ---- Observation mapping (shared with PPO, duplicated for simplicity) ----

    def _map_observations(self, msg: State):
        """Extract observations from State message, center coordinates, and transform for player side.

        Returns:
            16-dimensional centered observation array.
            Format: [player_pos, player_vel, opponent_pos, opponent_vel,
                     ball_pos, ball_vel, goal_player_pos, goal_opponent_pos]
        """
        right_peg_pos = np.array([msg.right_peg.position.y, msg.right_peg.position.x], dtype=np.float32)
        right_peg_vel = np.array([msg.right_peg.velocity.y, msg.right_peg.velocity.x], dtype=np.float32)
        left_peg_pos = np.array([msg.left_peg.position.y, msg.left_peg.position.x], dtype=np.float32)
        left_peg_vel = np.array([msg.left_peg.velocity.y, msg.left_peg.velocity.x], dtype=np.float32)
        ball_pos = np.array([msg.ball.position.y, msg.ball.position.x], dtype=np.float32)
        ball_vel = np.array([msg.ball.velocity.y, msg.ball.velocity.x], dtype=np.float32)
        right_goal_pos = np.array([msg.right_goal_pos.y, msg.right_goal_pos.x], dtype=np.float32)
        left_goal_pos = np.array([msg.left_goal_pos.y, msg.left_goal_pos.x], dtype=np.float32)

        center_offset = np.array([self.board_dim_height / 2.0, self.board_dim_width / 2.0], dtype=np.float32)
        right_peg_pos -= center_offset
        left_peg_pos -= center_offset
        ball_pos -= center_offset
        right_goal_pos -= center_offset
        left_goal_pos -= center_offset

        if self.player_side == "left":
            obs = np.concatenate([
                left_peg_pos, left_peg_vel,
                right_peg_pos, right_peg_vel,
                ball_pos, ball_vel,
                left_goal_pos, right_goal_pos,
            ])
        else:
            obs = np.concatenate([
                -right_peg_pos, -right_peg_vel,
                -left_peg_pos, -left_peg_vel,
                -ball_pos, -ball_vel,
                -right_goal_pos, -left_goal_pos,
            ])

        return obs

    def _add_additional_features(self, obs):
        """Add geometric features (angles and distances) to observations.

        Input: 16 values [player_pos, player_vel, opp_pos, opp_vel,
                          ball_pos, ball_vel, goal_player_pos, goal_opponent_pos]
        Output: 20 values [player_pos, player_vel, opp_pos, opp_vel, ball_pos, ball_vel, 8 features]
        """
        player_pos = obs[0:2]
        opponent_pos = obs[4:6]
        ball_pos = obs[8:10]
        goal_player_pos = obs[12:14]
        goal_opponent_pos = obs[14:16]

        vec_to_opp_goal = goal_opponent_pos - player_pos
        vec_to_ball = ball_pos - player_pos
        vec_opp_to_goal = goal_player_pos - opponent_pos
        vec_ball_to_opp = ball_pos - opponent_pos
        vec_opp_to_player = opponent_pos - player_pos
        vec_ball_to_goal = goal_player_pos - player_pos  # Match the original bug!
        vec_ball_to_opp_goal = goal_opponent_pos - ball_pos

        angle_pegball_pegoppgoal = self._angle_between_vectors(vec_to_ball, vec_to_opp_goal)
        angle_oppball_oppgoal = self._angle_between_vectors(vec_opp_to_goal, vec_ball_to_opp)
        angle_pegball_pegopp = self._angle_between_vectors(vec_to_ball, vec_opp_to_player)
        angle_oppball_pegopp = self._angle_between_vectors(vec_ball_to_opp, -vec_opp_to_player)

        distance_ball_goal = np.linalg.norm(vec_ball_to_goal)
        distance_ball_oppgoal = np.linalg.norm(vec_ball_to_opp_goal)
        distance_ball_player = np.linalg.norm(vec_to_ball)
        distance_ball_opp = np.linalg.norm(vec_ball_to_opp)

        extra_features = np.array([
            angle_pegball_pegoppgoal, angle_oppball_oppgoal,
            angle_pegball_pegopp, angle_oppball_pegopp,
            distance_ball_goal, distance_ball_oppgoal,
            distance_ball_player, distance_ball_opp,
        ], dtype=np.float32)

        return np.concatenate([obs[:12], extra_features])

    def _angle_between_vectors(self, v1, v2):
        """Compute angle between two vectors."""
        dot = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        cos_theta = dot / (norm_v1 * norm_v2 + 1e-8)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.arccos(cos_theta)

    # ---- Checkpoint management ----

    def _ensure_weights_available(self, weights_filename, zip_url, nn_weights_dir):
        """Ensure weights file exists, download and extract zip if necessary."""
        weights_dir = nn_weights_dir
        weights_path = weights_dir / weights_filename

        if weights_path.exists():
            self.logger.info(f"Weights file found: {weights_path}")
            return weights_path

        self.logger.info(f"Weights file not found. Downloading weights zip from: {zip_url}")

        if weights_dir.exists():
            self.logger.info(f"Cleaning existing weights directory contents: {weights_dir}")
            for item in weights_dir.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                except Exception as e:
                    self.logger.warn(f"Could not remove {item}: {e}")
        else:
            weights_dir.mkdir(parents=True, exist_ok=True)

        self._download_and_extract_weights(zip_url, weights_dir)

        if weights_path.exists():
            self.logger.info(f"Weights extracted successfully: {weights_path}")
            return weights_path
        else:
            available_files = [f.name for f in weights_dir.glob("*.pt")] + [f.name for f in weights_dir.glob("*.pth")]
            error_msg = (
                f"Weights file '{weights_filename}' not found after extraction. Available files: {available_files}"
            )
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

    def _download_and_extract_weights(self, url, destination_dir):
        """Download weights zip file and extract to destination directory."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
                tmp_path = tmp_file.name

                self.logger.info(f"Downloading weights zip to temporary file: {tmp_path}")

                req = urllib.request.Request(url)
                req.add_header("User-Agent", "Mozilla/5.0")

                with urllib.request.urlopen(req) as response:
                    chunk_size = 8192
                    total_size = 0
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        tmp_file.write(chunk)
                        total_size += len(chunk)
                        if total_size % (chunk_size * 100) == 0:
                            self.logger.info(f"Downloaded {total_size / 1024 / 1024:.2f} MB...")

                self.logger.info(f"Download complete: {total_size / 1024 / 1024:.2f} MB")

            self.logger.info(f"Extracting weights to: {destination_dir}")
            with zipfile.ZipFile(tmp_path, "r") as zip_ref:
                zip_ref.extractall(destination_dir)
                extracted_files = zip_ref.namelist()
                self.logger.info(f"Extracted {len(extracted_files)} files")

            # Flatten nested directory if zip contained a single root folder
            root_items = list(destination_dir.iterdir())
            if len(root_items) == 1 and root_items[0].is_dir():
                nested_dir = root_items[0]
                for item in nested_dir.iterdir():
                    item.rename(destination_dir / item.name)
                nested_dir.rmdir()

            os.unlink(tmp_path)
            self.logger.info("Extraction complete")

        except Exception as e:
            self.logger.error(f"Failed to download and extract weights: {e}")
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def _load_checkpoint(self, checkpoint_path):
        """Load model weights from checkpoint file."""
        self.logger.info(f"Loading Dreamer checkpoint from: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            # Auto-detect format: full training checkpoint vs inference-only export
            if "agent_state_dict" in checkpoint:
                self.logger.info("Detected full training checkpoint format")
                state_dict = checkpoint["agent_state_dict"]
            else:
                self.logger.info("Detected inference-only checkpoint format")
                state_dict = checkpoint

            # Extract only encoder, rssm, and actor weights
            encoder_dict = {k.removeprefix("encoder."): v for k, v in state_dict.items() if k.startswith("encoder.")}
            rssm_dict = {k.removeprefix("rssm."): v for k, v in state_dict.items() if k.startswith("rssm.")}
            actor_dict = {k.removeprefix("actor."): v for k, v in state_dict.items() if k.startswith("actor.")}

            # Load weights
            missing, unexpected = self.encoder.load_state_dict(encoder_dict, strict=False)
            if missing:
                self.logger.warn(f"Encoder missing keys: {missing}")
            if unexpected:
                self.logger.warn(f"Encoder unexpected keys: {unexpected}")

            missing, unexpected = self.rssm.load_state_dict(rssm_dict, strict=False)
            if missing:
                self.logger.warn(f"RSSM missing keys: {missing}")
            if unexpected:
                self.logger.warn(f"RSSM unexpected keys: {unexpected}")

            missing, unexpected = self.actor.load_state_dict(actor_dict, strict=False)
            if missing:
                self.logger.warn(f"Actor missing keys: {missing}")
            if unexpected:
                self.logger.warn(f"Actor unexpected keys: {unexpected}")

            self.logger.info("Dreamer checkpoint loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load Dreamer checkpoint: {e}")
            raise
