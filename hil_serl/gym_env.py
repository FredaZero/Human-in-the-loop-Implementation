"""Gymnasium wrapper, so off-the-shelf RL libraries can run on this task.

Used by the stable-baselines3 baseline in scripts/03b_train_sb3.py. It exists to
answer one question: how much of HIL-SERL's result comes from the RLPD machinery
and the human, and how much would you get from standard SAC plus the same
demonstrations?

TWO THINGS ARE FORCED BY THE WRAPPER, AND BOTH ARE THE POINT.

1. THE GRIPPER BECOMES A THIRD CONTINUOUS DIMENSION.
   SAC's action space is a Box. It cannot emit {stay, open, close}. So the
   gripper is a continuous value squashed through tanh and thresholded here.
   That is exactly the design HIL-SERL rejects by training a separate DQN over
   three actions (the "grasp critic"), on the grounds that a tanh-Gaussian over
   a dimension that wants three decisive values learns slowly and chatters. This
   baseline is what that rejection costs you, measured.

2. THERE IS ONE REPLAY BUFFER.
   SB3 samples `self.replay_buffer.sample(batch_size)` once per gradient step,
   so demonstrations can be pre-loaded but they cannot be held at a fixed
   fraction of every batch. With 20 demos (~660 transitions) against a 30k-step
   run, they fall to ~2% of the buffer and get sampled into irrelevance - which
   is the precise problem symmetric sampling exists to solve.

Reward and termination come from the same learned classifier the main pipeline
uses, so the comparison is like-for-like.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .env import PegInsertEnv, OBS_DIM, G_STAY, G_OPEN, G_CLOSE

# Thresholds carving the third action dimension into three commands. The dead
# zone in the middle is deliberate: without it, noise around 0 makes the policy
# open and close the gripper on alternate steps.
GRASP_DEADZONE = 0.33

# Value written into the buffer when replaying a demonstration's discrete
# gripper command back as a continuous action.
_GRASP_TO_CONT = {G_STAY: 0.0, G_OPEN: -0.66, G_CLOSE: 0.66}


def cont_to_grasp(a: float) -> int:
    if a > GRASP_DEADZONE:
        return G_CLOSE
    if a < -GRASP_DEADZONE:
        return G_OPEN
    return G_STAY


def grasp_to_cont(g: int) -> float:
    return _GRASP_TO_CONT[int(g)]


class PegInsertGym(gym.Env):
    """PegInsertEnv behind the Gymnasium API, with a learned reward."""

    metadata = {"render_modes": []}

    def __init__(self, reward_fn, seed: int = 0, max_steps: int = 120):
        super().__init__()
        self.core = PegInsertEnv(seed=seed, max_steps=max_steps)
        self.reward_fn = reward_fn
        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.core.rng = np.random.default_rng(seed)
        return self.core.reset().astype(np.float32), {}

    def step(self, action):
        a = np.asarray(action, np.float32)
        obs, true_succ, truncated, info = self.core.step(a[:2], cont_to_grasp(float(a[2])))
        reward, success, _p = self.reward_fn(self.core)
        return (obs.astype(np.float32), float(reward), bool(success), bool(truncated),
                {"true_success": bool(true_succ), **info})
