"""RLPD agent + grasp critic.

WHERE HUMAN INTERVENTIONS ENTER THE LEARNING
--------------------------------------------
Nowhere special - and that is the whole point.

There is no imitation loss in HIL-SERL. An intervention is stored as an ordinary
off-policy transition (s, a_human, r, s'), and off-policy RL is allowed to learn
from actions any policy took. Two mechanisms then do the work:

  1. VALUE PROPAGATION. The human's corrections are the only trajectories that
     ever reach reward=1 early on. The critic backs that 1 up through them, so
     the states along a correction acquire high Q. This is what turns a sparse
     reward into a dense gradient.

  2. POLICY IMPROVEMENT TOWARD THE CRITIC. The actor maximises Q. Because
     symmetric sampling guarantees half of every batch is human states, the
     actor is repeatedly asked "what is the best action *here*, in the states
     the human rescued you into?" and the critic already knows the answer.

So the policy imitates the human only through the value function. That is
strictly better than behaviour cloning the corrections: if the human's action
was mediocre, the critic can score it low and the actor will do something
better, whereas BC would copy it. HIL-SERL routinely ends up faster and more
reliable than the human who taught it - impossible under pure BC.

(An optional BC term is provided below, off by default, for comparison against
HG-DAgger / IWR style methods. It is NOT part of the paper's method.)
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import EnsembleCritic, TanhGaussianActor, GraspCritic


class RLPDAgent:
    def __init__(self, obs_dim, act_dim, n_grasp=3, *, hidden=256,
                 ensemble=10, subsample=2, discount=0.97, tau=0.005,
                 lr=3e-4, utd=2, target_entropy=None, grasp_penalty=0.02, alpha_init=0.05,
                 bc_weight=0.0, device="cpu", seed=0):
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.discount, self.tau, self.utd = discount, tau, utd
        self.subsample, self.ensemble = subsample, ensemble
        self.grasp_penalty, self.bc_weight = grasp_penalty, bc_weight

        self.actor = TanhGaussianActor(obs_dim, act_dim, hidden).to(self.device)
        self.critic = EnsembleCritic(obs_dim, act_dim, ensemble, hidden).to(self.device)
        self.critic_targ = copy.deepcopy(self.critic).requires_grad_(False)
        self.grasp = GraspCritic(obs_dim, n_grasp, hidden).to(self.device)
        self.grasp_targ = copy.deepcopy(self.grasp).requires_grad_(False)

        # Temperature init matters more here than in benchmark SAC. With a sparse
        # 0/1 reward and gamma=0.97 every Q lives in [0,1], so alpha=1 (the usual
        # default) makes `alpha*logp` an order of magnitude larger than the Q term:
        # the actor maximises entropy, stays uniform-random, the critic bootstraps
        # off random actions, and nothing ever propagates. Start alpha small and
        # let the Lagrange update raise it if the policy collapses.
        self.log_alpha = torch.tensor([np.log(alpha_init)], requires_grad=True,
                                      device=self.device)
        self.target_entropy = target_entropy if target_entropy is not None else -act_dim / 2

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.opt_grasp = torch.optim.Adam(self.grasp.parameters(), lr=lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=lr)
        self.n_grasp = n_grasp

    # -- acting -------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs, deterministic=False, eps=0.0):
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, _ = self.actor(o, deterministic=deterministic, with_logp=False)
        q = self.grasp(o).squeeze(0)
        if (not deterministic) and np.random.rand() < eps:
            g = int(np.random.randint(self.n_grasp))
        else:
            g = int(q.argmax().item())
        return a.squeeze(0).cpu().numpy(), g

    # -- learning -----------------------------------------------------------
    def _to_torch(self, batch):
        return {k: torch.as_tensor(v, device=self.device) for k, v in batch.items()}

    def update(self, sample_fn, batch_size=128):
        """One RLPD step: `utd` critic updates, then one actor/alpha/grasp update."""
        info = {}
        for _ in range(self.utd):
            info.update(self._update_critic(self._to_torch(sample_fn(batch_size))))
        b = self._to_torch(sample_fn(batch_size))
        info.update(self._update_actor_and_alpha(b))
        info.update(self._update_grasp(b))
        self._soft_update(self.critic, self.critic_targ)
        self._soft_update(self.grasp, self.grasp_targ)
        return info

    def _update_critic(self, b):
        with torch.no_grad():
            next_a, _ = self.actor(b["next_obs"])
            tq = self.critic_targ(b["next_obs"], next_a)                 # (N,B)
            # RLPD: min over a RANDOM SUBSET of the ensemble, not the whole thing.
            # Full min over 10 critics is crushingly pessimistic; 2-of-10 keeps
            # the anti-overestimation benefit while staying learnable.
            idx = torch.randperm(self.ensemble, device=self.device)[:self.subsample]
            target_q = tq[idx].min(0).values
            y = b["rew"] + self.discount * b["mask"] * target_q           # mask=0 on success

        q = self.critic(b["obs"], b["act"])                              # (N,B)
        loss = F.mse_loss(q, y.unsqueeze(0).expand_as(q))
        self.opt_critic.zero_grad(); loss.backward(); self.opt_critic.step()
        return {"critic_loss": loss.item(), "q_mean": q.mean().item()}

    def _update_actor_and_alpha(self, b):
        a, logp = self.actor(b["obs"])
        q = self.critic(b["obs"], a).mean(0)          # mean over ensemble for the actor
        alpha = self.log_alpha.exp().detach()
        actor_loss = (alpha * logp - q).mean()

        if self.bc_weight > 0:                        # optional, not in the paper
            mu, _ = self.actor(b["obs"], deterministic=True, with_logp=False)
            bc = ((mu - b["act"]) ** 2).sum(-1) * b["is_human"]
            actor_loss = actor_loss + self.bc_weight * bc.mean()

        self.opt_actor.zero_grad(); actor_loss.backward(); self.opt_actor.step()

        alpha_loss = -(self.log_alpha.exp() * (logp.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad(); alpha_loss.backward(); self.opt_alpha.step()
        return {"actor_loss": actor_loss.item(), "alpha": self.log_alpha.exp().item(),
                "entropy": -logp.mean().item()}

    def _update_grasp(self, b):
        """Double DQN on the gripper MDP, same reward minus an action penalty.

        The penalty is why the trained policy does not chatter the gripper: any
        open/close that is not needed costs a little return."""
        g = b["grasp"]
        r = b["rew"] - self.grasp_penalty * (g != 0).float()
        with torch.no_grad():
            next_a = self.grasp(b["next_obs"]).argmax(-1, keepdim=True)    # online selects
            next_q = self.grasp_targ(b["next_obs"]).gather(-1, next_a).squeeze(-1)  # target evaluates
            y = r + self.discount * b["mask"] * next_q
        q = self.grasp(b["obs"]).gather(-1, g.unsqueeze(-1)).squeeze(-1)
        loss = F.mse_loss(q, y)
        self.opt_grasp.zero_grad(); loss.backward(); self.opt_grasp.step()
        return {"grasp_loss": loss.item()}

    def _soft_update(self, net, targ):
        with torch.no_grad():
            for p, pt in zip(net.parameters(), targ.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)
