"""Dreamer policy inference class — handles network loading and stateful inference."""

import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Any

from klask_player_pkg.utils import (
    add_additional_state_features,
    map_state_observations,
    ensure_weights_available,
)

from .dreamer_config import DreamerModelConfig
from .dreamer_network import MultiEncoder, RSSM, MLPHead
from .debug import show_image_tensor
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
        debug_view: bool = False,
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
            debug_view: If True, show the rotated ego-view image in an OpenCV window each step.
            logger: Optional logger object with info(), warn(), error() methods.
        """
        self.logger = logger
        self.device = device
        self.player_side = player_side
        self.debug_view = debug_view
        self.board_dim_width = board_dim_width
        self.board_dim_height = board_dim_height
        self.act_dim = 2  # 2D velocity actions

        # Ensure weights are available
        checkpoint_path = ensure_weights_available(weights_filename, weights_zip_url, nn_weights_dir, self.logger)

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

        # (H, W, 3) uint8 → (1, H, W, 3) float32 [0, 255]
        image_tensor = torch.from_numpy(image_rgb).unsqueeze(0).to(dtype=torch.float32, device=self.device)

        # rotate image to left or right player's perspective
        if self.player_side == "left":
            k = 1  # rotate 90 degrees counterclockwise for left player
        else:
            k = -1  # rotate 90 degrees clockwise for right player
        image_tensor_ego = torch.rot90(image_tensor, k=k, dims=[1, 2])

        # Pad image to expected size
        image_tensor_ego = self._pad_image_to_target(image_tensor_ego, self.image_size, self.image_size)

        if self.debug_view:
            show_image_tensor(image_tensor_ego, window_name=f"dreamer_ego_{self.player_side}")

        obs["image"] = image_tensor_ego

        # Optionally add state observations
        if self.obs_mode == "image_and_state":
            obs_base = map_state_observations(msg, self.player_side, self.board_dim_width, self.board_dim_height)
            obs_full = add_additional_state_features(obs_base)
            policy_tensor = torch.from_numpy(obs_full).unsqueeze(0).to(dtype=torch.float32, device=self.device)
            obs["policy"] = policy_tensor

        return obs

    def _pad_image_to_target(images: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """Zero-pad image tensor (N, H, W, C) to (N, target_h, target_w, C)."""
        _, h, w, _ = images.shape
        pad_bottom = target_h - h
        pad_right = target_w - w
        if pad_bottom > 0 or pad_right > 0:
            # For (N, H, W, C): pad W on the right and H on the bottom.
            images = F.pad(images, (0, 0, 0, pad_right, 0, pad_bottom), value=0)
        return images

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
            stoch, deter, _ = self.rssm.obs_step(self._stoch, self._deter, self._prev_action, embed, is_first)

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

    # ---- Checkpoint management ----

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
