"""Policy inference class - handles network loading and inference."""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Any
from .ppo_network import PolicyNetwork
from klask_interfaces.msg import State
from klask_player_pkg.utils import (
    map_state_observations,
    add_additional_state_features,
    ensure_weights_available,
)


class PolicyInference:
    """Handle policy network loading and inference."""

    def __init__(
        self,
        weights_filename: str,
        weights_zip_url: str,
        nn_weights_dir: Path,
        device: str = "cpu",
        clip_actions: float = 0.2,
        enable_action_rescaling: bool = True,
        player_side: str = "left",
        board_dim_width: float = 0.42,
        board_dim_height: float = 0.32,
        logger: Optional[Any] = None,
    ):
        """Initialize policy inference.

        Args:
            weights_filename: Name of the weights file to load
            weights_zip_url: URL to download weights if not found locally
            nn_weights_dir: Directory to store/load neural network weights
            device: Device to run inference on ('cpu' or 'cuda')
            clip_actions: Action clipping bound
            enable_action_rescaling: Enable action rescaling
            player_side: 'left' or 'right' player side
            board_dim_width: Board width in meters
            board_dim_height: Board height in meters
            logger: Optional logger object with info(), warn(), error() methods
        """
        self.logger = logger
        self.device = device
        self.player_side = player_side
        self.board_dim_width = board_dim_width
        self.board_dim_height = board_dim_height

        # Ensure weights are available
        checkpoint_path = ensure_weights_available(weights_filename, weights_zip_url, nn_weights_dir, self.logger)

        # Initialize network
        self.policy_net = PolicyNetwork(
            actions_low=-clip_actions,
            actions_high=clip_actions,
            clip_actions=enable_action_rescaling,
        )
        self._load_checkpoint(checkpoint_path)
        self.policy_net.to(self.device)
        self.policy_net.eval()

        self.logger.info(f"Policy inference node initialized with checkpoint: {checkpoint_path}")
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Board dimensions: {self.board_dim_width}m x {self.board_dim_height}m")
        self.logger.info(f"Player side: {self.player_side}")
        self.logger.info(f"Action rescaling: {enable_action_rescaling} (clip_actions: {clip_actions})")

    def _load_checkpoint(self, checkpoint_path):
        """Load model weights from checkpoint file."""
        self.logger.info(f"Loading checkpoint from: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            # TODO: maybe remove weights_only once everything is working and we are
            # able to regenerate the checkpoint from the original training

            # RL-Games saves checkpoints with 'model' key
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

            # Map checkpoint keys to our network structure
            model_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("a2c_network."):
                    # Remove "a2c_network." prefix to match our structure
                    new_key = key.replace("a2c_network.", "")
                    model_state_dict[new_key] = value
                elif key.startswith("running_mean_std."):
                    # Extract observation normalization parameters
                    new_key = key.replace("running_mean_std.", "")
                    model_state_dict[new_key] = value

            # Load the mapped state dict
            missing_keys, unexpected_keys = self.policy_net.load_state_dict(model_state_dict, strict=False)

            if missing_keys:
                self.logger.warn(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                self.logger.warn(f"Unexpected keys: {unexpected_keys}")

            # Log normalization statistics
            if hasattr(self.policy_net, "running_mean") and hasattr(self.policy_net, "running_var"):
                self.logger.info(
                    f"Loaded observation normalization: mean range [{self.policy_net.running_mean.min():.4f}, "
                    f"{self.policy_net.running_mean.max():.4f}], "
                    f"var range [{self.policy_net.running_var.min():.4f}, "
                    f"{self.policy_net.running_var.max():.4f}]"
                )

            self.logger.info("Checkpoint loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise

    def reset_state(self):
        """No-op for PPO (stateless policy). Exists for interface compatibility."""
        pass

    def get_action(self, msg: State, image: np.ndarray = None) -> np.ndarray:
        """Get action from State message using the policy network."""
        # Extract and transform observations (handles centering and player-side transformation)
        obs_base = map_state_observations(msg, self.player_side, self.board_dim_width, self.board_dim_height)

        # Compute additional features
        obs_full = add_additional_state_features(obs_base)

        # Get action from policy
        action = self._predict(obs_full)

        # Negate action for right player (flip direction)
        if self.player_side == "right":
            action = -action

        return action

    def _predict(self, obs):
        """Get action from policy network."""
        # Convert to tensor
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)

        # Get action from policy
        action = self.policy_net.get_action(obs_tensor, deterministic=True)

        # Convert back to numpy
        action = action.cpu().numpy().flatten()

        return action
