import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """
    Actor-Critic network for Klask player.
    """

    def __init__(self, obs_dim=20, action_dim=2, hidden_dim=32):
        super(PolicyNetwork, self).__init__()

        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Critic network (value function) - not needed for inference but part of checkpoint
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        """Forward pass through actor network."""
        return self.actor(obs)

    def get_action(self, obs, deterministic=True):
        """Get action from observation."""
        with torch.no_grad():
            action = self.forward(obs)
        return action
