"""Networks for RLPD (the RL algorithm underneath HIL-SERL).

RLPD = SAC + three things that make off-policy RL work from scratch on a real
robot in an hour:
  (a) a large critic ENSEMBLE with a random subset taken for the target min,
  (b) LAYER NORM in the critic, which stops value blow-up on out-of-distribution
      actions - the single most important trick when half your batch is human
      data the policy would never have produced,
  (c) a high UPDATE-TO-DATA ratio, so each real robot step is squeezed hard.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -10.0, 2.0


def mlp(sizes, layer_norm=True, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            if layer_norm:
                layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(nn.ReLU())
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


class EnsembleLinear(nn.Module):
    """N independent Linear layers evaluated as one batched matmul."""

    def __init__(self, n, fan_in, fan_out):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n, fan_in, fan_out))
        self.b = nn.Parameter(torch.zeros(n, 1, fan_out))
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.W, -bound, bound)
        nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x):                     # x: (N, B, fan_in)
        return torch.baddbmm(self.b, x, self.W)


class EnsembleCritic(nn.Module):
    """N Q-functions, each LayerNorm-MLP, all updated against the same target."""

    def __init__(self, obs_dim, act_dim, n=10, hidden=256):
        super().__init__()
        self.n = n
        d = obs_dim + act_dim
        self.l1, self.l2, self.l3 = (EnsembleLinear(n, d, hidden),
                                     EnsembleLinear(n, hidden, hidden),
                                     EnsembleLinear(n, hidden, 1))
        self.ln1, self.ln2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)

    def forward(self, obs, act):              # (B,obs),(B,act) -> (N,B)
        x = torch.cat([obs, act], -1).unsqueeze(0).expand(self.n, -1, -1)
        x = F.relu(self.ln1(self.l1(x)))
        x = F.relu(self.ln2(self.l2(x)))
        return self.l3(x).squeeze(-1)


class TanhGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.trunk = mlp([obs_dim, hidden, hidden, 2 * act_dim], layer_norm=True)
        self.act_dim = act_dim

    def forward(self, obs, deterministic=False, with_logp=True):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        if deterministic:
            u = mu
        else:
            u = mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        if not with_logp:
            return a, None
        # tanh change-of-variables correction
        logp = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        logp = logp - (2 * (math.log(2) - u - F.softplus(-2 * u))).sum(-1)
        return a, logp


class GraspCritic(nn.Module):
    """A separate DQN over {stay, open, close}.

    HIL-SERL splits the problem into two MDPs that share the same state and
    reward: a continuous one solved with SAC, and a discrete gripper one solved
    with DQN. Mixing a discrete dimension into a tanh-Gaussian is awkward and
    the gripper needs very few, very decisive actions - a Q-table over 3 actions
    learns that far faster than a squashed Gaussian ever will.
    """

    def __init__(self, obs_dim, n_actions=3, hidden=256):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden, n_actions], layer_norm=True)

    def forward(self, obs):
        return self.net(obs)
