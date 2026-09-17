"""Two buffers and symmetric sampling - the mechanical heart of HIL-SERL.

    demo buffer : human demonstrations  +  every human intervention transition
    rl buffer   : everything the robot did, interventions included

Every gradient batch is drawn HALF from each. That fixed 50/50 ratio is what
makes human data keep mattering: if you threw everything into one buffer, 20
demos would be 0.2% of a 10k-step buffer and would be sampled into oblivion.

Routing rule from the paper (this exact asymmetry matters):
  * intervention transitions      -> BOTH buffers
  * the policy's own transitions  -> RL buffer only
"""

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim, seed=0):
        self.cap = capacity
        self.rng = np.random.default_rng(seed)
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.act = np.zeros((capacity, act_dim), np.float32)
        self.grasp = np.zeros(capacity, np.int64)
        self.rew = np.zeros(capacity, np.float32)
        self.mask = np.zeros(capacity, np.float32)       # 0 iff terminal success
        self.is_human = np.zeros(capacity, np.float32)
        self.ptr, self.size = 0, 0

    def add(self, obs, act, grasp, rew, next_obs, mask, is_human=0.0):
        i = self.ptr
        self.obs[i], self.act[i], self.grasp[i] = obs, act, grasp
        self.rew[i], self.next_obs[i], self.mask[i] = rew, next_obs, mask
        self.is_human[i] = is_human
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def add_traj(self, traj, is_human=0.0):
        for k in range(len(traj["obs"])):
            succ = traj["success"][k]
            self.add(traj["obs"][k], traj["act"][k], traj["grasp"][k],
                     1.0 if succ else 0.0, traj["next_obs"][k],
                     0.0 if succ else 1.0, is_human)

    def sample(self, n):
        i = self.rng.integers(0, self.size, n)
        return dict(obs=self.obs[i], act=self.act[i], grasp=self.grasp[i],
                    rew=self.rew[i], next_obs=self.next_obs[i],
                    mask=self.mask[i], is_human=self.is_human[i])


def symmetric_sample(demo_buf, rl_buf, batch):
    """50% human data, 50% robot data, every single gradient step."""
    if demo_buf.size == 0:
        return rl_buf.sample(batch)
    if rl_buf.size == 0:
        return demo_buf.sample(batch)
    h = batch // 2
    a, b = demo_buf.sample(h), rl_buf.sample(batch - h)
    return {k: np.concatenate([a[k], b[k]], 0) for k in a}
