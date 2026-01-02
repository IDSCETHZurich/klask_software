import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """
    Actor-Critic network for Klask player. As it was used in the RL-Games training (checkpoint).
    """

    def __init__(self):
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

    def forward(self, obs):
        """Forward pass through shared trunk and actor head."""
        features = self.actor_mlp(obs)
        return self.mu(features)

    def get_action(self, obs, deterministic=True):
        """Get action from observation."""
        with torch.no_grad():
            action = self.forward(obs)
        return action
