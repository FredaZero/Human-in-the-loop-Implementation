"""Baseline: stable-baselines3 SAC on the same task, for the ablation study.

    python scripts/03b_train_sb3.py --steps 30000 --tag abl_sb3-sac

This is "standard off-policy RL with demonstrations" - the thing you would
reach for before writing any of HIL-SERL. It gets the SAME environment, the
SAME learned reward classifier, the SAME demonstrations and the SAME evaluation
protocol, so the difference in the numbers is attributable to the algorithm.

What it does NOT get, because SB3's SAC has no hook for it:

  * min over a random 2-of-10 critic subset  (SB3 takes min over all critics)
  * no-entropy backup                        (SB3 always subtracts ent_coef*logp)
  * an actor that maximises the ensemble MEAN (SB3's actor uses the min)
  * 50/50 symmetric sampling                 (SB3 has one buffer)
  * a discrete grasp critic                  (SAC's action space is a Box)
  * human interventions                      (needs collect_rollouts overridden)

It DOES get LayerNorm critics, a 10-critic ensemble and ent_coef initialised at
0.05 - so this is not a strawman. LayerNorm is the one my own README calls "the
single most important trick" in RLPD, so leaving it out would confound it with
everything else on the list above.

Getting it in needed a subclass: `create_mlp` accepts `post_linear_modules`, but
`ContinuousCritic` calls it without that argument and `SACPolicy` has no kwarg
to forward, so ~15 lines below rebuild the Q networks. That is itself a small
illustration of the argument - the cheap parts are reachable, but rarely through
the front door.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.sac.policies import SACPolicy

from hil_serl.env import PegInsertEnv, OBS_DIM, ACT_DIM
from hil_serl.gym_env import PegInsertGym, cont_to_grasp, grasp_to_cont
from hil_serl.teleop import ScriptedHuman, rollout_demo
from hil_serl.reward_classifier import RewardClassifier, LearnedReward, OracleReward

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class LayerNormCritic(ContinuousCritic):
    """ContinuousCritic with LayerNorm after each hidden Linear.

    Produces exactly the structure of hil_serl/networks.py's `mlp()`:
    Linear -> LayerNorm -> ReLU per hidden layer, bare Linear on the output.
    """

    def __init__(self, observation_space, action_space, net_arch, features_extractor,
                 features_dim, activation_fn=nn.ReLU, normalize_images=True,
                 n_critics=2, share_features_extractor=True):
        super().__init__(observation_space, action_space, net_arch, features_extractor,
                         features_dim, activation_fn, normalize_images, n_critics,
                         share_features_extractor)
        action_dim = int(np.prod(action_space.shape))
        self.q_networks = []
        for idx in range(n_critics):
            q_net = nn.Sequential(*create_mlp(
                features_dim + action_dim, 1, net_arch, activation_fn,
                post_linear_modules=[nn.LayerNorm]))
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)


class LayerNormSACPolicy(SACPolicy):
    def make_critic(self, features_extractor=None) -> ContinuousCritic:
        kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return LayerNormCritic(**kwargs).to(self.device)


def evaluate(model, reward_fn, n=30, seed=5000, max_steps=120):
    """Identical protocol to recon of the main pipeline: deterministic policy,
    no human, report both ground truth and what the classifier believed."""
    env = PegInsertEnv(seed=seed, max_steps=max_steps)
    true_n = cls_n = 0
    for _ in range(n):
        obs = env.reset()
        for _ in range(max_steps):
            a, _ = model.predict(obs.astype(np.float32), deterministic=True)
            obs, _true, trunc, _ = env.step(np.asarray(a)[:2], cont_to_grasp(float(a[2])))
            _r, cls_succ, _p = reward_fn(env)
            if cls_succ or trunc:
                true_n += int(env.true_success())
                cls_n += int(cls_succ)
                break
    return {"true": true_n / n, "classifier": cls_n / n}


class EvalCallback(BaseCallback):
    def __init__(self, reward_fn, every, n_eval, log):
        super().__init__()
        self.reward_fn, self.every, self.n_eval, self.log_fn = reward_fn, every, n_eval, log
        self.evals = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.every == 0:
            ev = evaluate(self.model, self.reward_fn, self.n_eval)
            self.evals.append({"step": self.num_timesteps, **ev})
            self.log_fn(f"step {self.num_timesteps:6d} | eval true success {ev['true']:.2f} "
                        f"| classifier says {ev['classifier']:.2f} "
                        f"| false-pos gap {ev['classifier']-ev['true']:+.2f} "
                        f"| buffer {self.model.replay_buffer.size()}")
        return True


def preload_demos(model, n_demos, seed, reward_fn, max_steps=120):
    """Push demonstrations into SB3's single replay buffer.

    This is the closest SB3 equivalent of the demo buffer. Note what it cannot
    do: keep those transitions at a fixed share of every gradient batch. They
    are simply early entries that get progressively diluted.
    """
    env = PegInsertEnv(seed=seed + 100, max_steps=max_steps)
    human = ScriptedHuman(seed=seed + 101)
    n = 0
    for _ in range(n_demos):
        traj = rollout_demo(env, human)
        for k in range(len(traj["obs"])):
            succ = bool(traj["success"][k])
            act = np.array([traj["act"][k][0], traj["act"][k][1],
                            grasp_to_cont(traj["grasp"][k])], np.float32)
            model.replay_buffer.add(
                np.array([traj["obs"][k]], np.float32),
                np.array([traj["next_obs"][k]], np.float32),
                np.array([act], np.float32),
                np.array([1.0 if succ else 0.0], np.float32),
                np.array([succ], bool),
                [{}])
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--demos", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--utd", type=int, default=4, help="gradient steps per env step")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--oracle-reward", action="store_true")
    ap.add_argument("--small-labels", action="store_true")
    ap.add_argument("--no-side-check", action="store_true")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--n-eval", type=int, default=30)
    ap.add_argument("--tag", default="sb3")
    args = ap.parse_args()

    if args.oracle_reward:
        reward_fn = OracleReward()
        print("reward: GROUND TRUTH")
    else:
        name = "reward_classifier_small.pt" if args.small_labels else "reward_classifier.pt"
        ckpt = torch.load(os.path.join(ROOT, "data", name))
        m = RewardClassifier(); m.load_state_dict(ckpt["state_dict"])
        tau = ckpt["tau"] if args.tau is None else args.tau
        print(f"reward: learned classifier, tau={tau:.2f}, "
              f"side_check={not args.no_side_check}")
        reward_fn = LearnedReward(m, tau=tau, side_check=not args.no_side_check)

    env = PegInsertGym(reward_fn, seed=args.seed)

    # Give SB3 every advantage it can actually express: LayerNorm critics, a
    # 10-critic ensemble, ent_coef starting at 0.05 (the value that unblocked
    # the hand-written agent), and the same update-to-data ratio.
    model = SAC(
        LayerNormSACPolicy, env, seed=args.seed, verbose=0,
        learning_rate=3e-4, batch_size=128, gamma=0.97, tau=0.005,
        ent_coef="auto_0.05", learning_starts=500,
        train_freq=1, gradient_steps=args.utd,
        buffer_size=200_000,
        policy_kwargs=dict(net_arch=[256, 256], n_critics=10),
    )

    n = preload_demos(model, args.demos, args.seed, reward_fn)
    print(f"demo buffer: {n} transitions from {args.demos} demonstrations "
          f"(single buffer - they dilute as it fills)")
    print("interventions: False (SB3 has no hook for them)\n")

    cb = EvalCallback(reward_fn, args.eval_every, args.n_eval, print)
    model.learn(total_timesteps=args.steps, callback=cb, log_interval=None)

    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    out = os.path.join(ROOT, "runs", f"{args.tag}_eval.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "true", "classifier"])
        w.writeheader(); w.writerows(cb.evals)
    final = cb.evals[-3:]
    print(f"\nfinal eval (mean of last {len(final)} checkpoints, no human): "
          f"true success {np.mean([e['true'] for e in final]):.2f}, "
          f"classifier says {np.mean([e['classifier'] for e in final]):.2f}")
    print(f"log -> {out}")


if __name__ == "__main__":
    main()
