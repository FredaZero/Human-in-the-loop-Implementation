"""The HIL-SERL loop.

The real system runs an actor process and a learner process asynchronously
across two machines (the robot must keep moving at 10 Hz while gradients are
computed on a 4090). Here they are interleaved in one process - same algorithm,
same data flow, no distributed plumbing to read through.

Per environment step:
    1. policy proposes (a_cont, a_grasp)
    2. human may take over  ->  the stored action is the HUMAN's action
    3. env steps
    4. the LEARNED CLASSIFIER, not the simulator, decides reward and termination
    5. route the transition:  human -> both buffers,  policy -> rl buffer only
    6. `utd` critic updates + 1 actor/alpha/grasp update
"""

import time
import numpy as np

from .env import PegInsertEnv, OBS_DIM, ACT_DIM, N_GRASP_ACTIONS
from .teleop import ScriptedHuman, InterventionPolicy, rollout_demo
from .buffers import ReplayBuffer, symmetric_sample
from .agent import RLPDAgent


def train(reward_fn, *, steps=12000, n_demos=20, seed=0, utd=2, batch=128,
          start_after=500, eval_every=1000, n_eval=30, use_interventions=True,
          intervention_kwargs=None, agent_kwargs=None, log=print, max_ep=120):

    env = PegInsertEnv(seed=seed, max_steps=max_ep)
    eval_env = PegInsertEnv(seed=seed + 5000, max_steps=max_ep)
    human = ScriptedHuman(seed=seed + 1)
    interv = InterventionPolicy(seed=seed + 2, **(intervention_kwargs or {}))

    demo_buf = ReplayBuffer(200_000, OBS_DIM, ACT_DIM, seed=seed + 3)
    rl_buf = ReplayBuffer(200_000, OBS_DIM, ACT_DIM, seed=seed + 4)
    agent = RLPDAgent(OBS_DIM, ACT_DIM, N_GRASP_ACTIONS, utd=utd, seed=seed,
                      **(agent_kwargs or {}))

    # --- offline demonstrations seed the demo buffer (paper: 20-30 trajectories)
    demo_env = PegInsertEnv(seed=seed + 100, max_steps=max_ep)
    for _ in range(n_demos):
        demo_buf.add_traj(rollout_demo(demo_env, human), is_human=1.0)
    log(f"demo buffer: {demo_buf.size} transitions from {n_demos} demonstrations")

    obs = env.reset()
    interv.reset_episode()
    episode, ep_ret, ep_len, ep_interv = 0, 0.0, 0, 0
    hist, evals = [], []
    t0 = time.time()

    for step in range(1, steps + 1):
        # 1-2. policy proposes; human may override
        a_pi, g_pi = agent.act(obs, eps=0.05 if step < start_after else 0.0)
        human_now = use_interventions and interv(env, episode, {"contact": env.contact})
        if human_now:
            a, g = human.act(env)
            ep_interv += 1
        else:
            a, g = a_pi, g_pi

        # 3-4. step, then ASK THE CLASSIFIER what happened
        next_obs, true_succ, truncated, info = env.step(a, g)
        r, success, _p = reward_fn(env)
        done = success                      # termination is the classifier's call

        # 5. route the transition
        tr = dict(obs=obs, act=a, grasp=g, rew=r, next_obs=next_obs,
                  mask=0.0 if done else 1.0)
        rl_buf.add(**tr, is_human=1.0 if human_now else 0.0)
        if human_now:
            demo_buf.add(**tr, is_human=1.0)

        obs = next_obs
        ep_ret += r
        ep_len += 1

        if done or truncated:
            hist.append(dict(step=step, episode=episode, ret=ep_ret, len=ep_len,
                             true_success=float(true_succ),
                             classifier_success=float(success),
                             interv_frac=ep_interv / max(ep_len, 1)))
            episode += 1
            obs = env.reset()
            interv.reset_episode()
            ep_ret, ep_len, ep_interv = 0.0, 0, 0

        # 6. learn
        if step >= start_after:
            agent.update(lambda n: symmetric_sample(demo_buf, rl_buf, n), batch)

        if step % eval_every == 0:
            ev = evaluate_policy(agent, eval_env, reward_fn, n_eval)
            evals.append(dict(step=step, true=ev["true"], classifier=ev["classifier"]))
            recent = hist[-20:] or [{"interv_frac": 0.0}]
            log(f"step {step:6d} | eval true success {ev['true']:.2f} "
                f"| classifier says {ev['classifier']:.2f} "
                f"| false-pos gap {ev['classifier']-ev['true']:+.2f} "
                f"| intervention rate {np.mean([h['interv_frac'] for h in recent]):.2f} "
                f"| rl_buf {rl_buf.size} | {time.time()-t0:.0f}s")

    return agent, hist, evals


def evaluate_policy(agent, env, reward_fn, n=30):
    """Deterministic rollouts with NO human. Reports both the truth and what the
    classifier believed - the gap between them is reward hacking, measured."""
    true_n = cls_n = 0
    for _ in range(n):
        obs = env.reset()
        for _ in range(env.max_steps):
            a, g = agent.act(obs, deterministic=True)
            obs, true_succ, trunc, _ = env.step(a, g)
            _r, cls_succ, _p = reward_fn(env)
            if cls_succ or trunc:
                true_n += int(env.true_success())
                cls_n += int(cls_succ)
                break
    return {"true": true_n / n, "classifier": cls_n / n}
