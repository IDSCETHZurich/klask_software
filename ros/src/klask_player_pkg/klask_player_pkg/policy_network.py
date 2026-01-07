import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """
    Actor-Critic network for Klask player. As it was used in the RL-Games training (checkpoint).
    """

    def __init__(self, actions_low=-0.2, actions_high=0.2, clip_actions=True):
        super(PolicyNetwork, self).__init__()

        # Shared feature extractor
        self.actor_mlp = nn.Sequential(
            nn.Linear(20, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )

        # Actor head
        self.mu = nn.Linear(64, 2)

        # Critic head
        self.value = nn.Linear(64, 1)

        # Sigma (not used with fixed_sigma=true, but in checkpoint)
        self.sigma = nn.Parameter(torch.zeros(2))

        # Action rescaling parameters
        self.clip_actions = clip_actions
        if isinstance(actions_low, (int, float)):
            self.actions_low = torch.tensor([actions_low, actions_low], dtype=torch.float32)
            self.actions_high = torch.tensor([actions_high, actions_high], dtype=torch.float32)
        else:
            self.actions_low = torch.tensor(actions_low, dtype=torch.float32)
            self.actions_high = torch.tensor(actions_high, dtype=torch.float32)

        # Observation normalization parameters (loaded from checkpoint)
        # These will be set when loading the checkpoint
        self.register_buffer('running_mean', torch.zeros(20, dtype=torch.float64))
        self.register_buffer('running_var', torch.ones(20, dtype=torch.float64))
        self.normalize_input = True  # Matches rl-games config
        self.norm_epsilon = 1e-05  # Default epsilon from rl-games

    def forward(self, obs):
        """Forward pass through shared trunk and actor head."""
        features = self.actor_mlp(obs)
        return self.mu(features)

    def norm_obs(self, obs):
        """Normalize observations using running mean and variance.
        
        This matches the rl-games RunningMeanStd normalization:
        https://github.com/Denys88/rl_games/blob/master/rl_games/algos_torch/running_mean_std.py
        
        Formula: (obs - running_mean) / sqrt(running_var + epsilon)
        Then clamp to [-5, 5]
        """
        if not self.normalize_input:
            return obs
        
        with torch.no_grad():
            # Convert to float32 for computation (running stats are float64)
            current_mean = self.running_mean.float()
            current_var = self.running_var.float()
            
            # Normalize
            normalized = (obs - current_mean) / torch.sqrt(current_var + self.norm_epsilon)
            
            # Clamp to prevent extreme values
            normalized = torch.clamp(normalized, min=-5.0, max=5.0)
            
        return normalized

    def rescale_actions(self, actions):
        """Rescale actions from [-1, 1] to [actions_low, actions_high].
        """
        # Ensure actions_low and actions_high are on the same device as actions
        actions_low = self.actions_low.to(actions.device)
        actions_high = self.actions_high.to(actions.device)
        
        d = (actions_high - actions_low) / 2.0
        m = (actions_high + actions_low) / 2.0
        scaled_action = actions * d + m
        return scaled_action

    def get_action(self, obs, deterministic=True):
        """Get action from observation.
        
        Mimics rl-games PpoPlayerContinuous.get_action behavior:
        1. Normalize observations using running mean/var
        2. Get mu from network (in [-1, 1] range due to no activation)
        3. Clamp to [-1, 1]
        4. Rescale to [actions_low, actions_high]
        """
        with torch.no_grad():
            # Normalize observations first (critical step!)
            normalized_obs = self.norm_obs(obs)
            
            # Forward pass with normalized observations
            mu = self.forward(normalized_obs)
            
            if self.clip_actions:
                # Clamp to [-1, 1] then rescale to action bounds
                clamped_action = torch.clamp(mu, -1.0, 1.0)
                action = self.rescale_actions(clamped_action)
            else:
                action = mu
                
        return action
