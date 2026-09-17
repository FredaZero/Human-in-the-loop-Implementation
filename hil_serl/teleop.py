"""The "human" in human-in-the-loop.

Two operators are provided:

  ScriptedHuman  - an imperfect expert. Stands in for a person with a SpaceMouse
                   so the whole pipeline runs unattended. It is deliberately
                   noisy and slightly sloppy: HIL-SERL does not assume the
                   human is optimal, only that they are corrective.

  KeyboardHuman  - real teleoperation from the terminal, for when you want to
                   feel what the loop is actually like. Optional.

InterventionPolicy encodes the *protocol* the paper recommends, which matters
more than people expect:
  - intervene often early, rarely later (rate decays as the policy improves),
    but NOT constantly - see the measurements in the README. Over-intervening
    starves the critic of the policy's own failures and does more harm than not
    intervening at all. p0 here is the trigger probability, not the resulting
    intervention fraction: `burst` holds the takeover for several steps, so
    p0=0.3 lands around a 0.2 intervention fraction averaged over training.
  - intervene where the policy is actually failing (contact / jamming /
    misalignment), not uniformly at random
  - keep interventions SHORT. The paper explicitly warns that long interventions
    that drive the episode all the way to success inflate the value function:
    the critic learns "states near the human takeover are great" when really it
    was the human, not the state, that was great.
"""

import numpy as np
from .env import G_STAY, G_OPEN, G_CLOSE, WALL_Y1, BLOCK_HALF, GOAL_Y, STEP

SAFE_Y = WALL_Y1 + BLOCK_HALF + 0.06     # hover height above the wall


class ScriptedHuman:
    """A 4-phase controller: approach -> grasp -> align -> insert -> release."""

    def __init__(self, seed=0, noise=0.12, sloppiness=0.05):
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self.sloppiness = sloppiness      # prob. of a dithering / idle step

    def act(self, env):
        ee, blk, sx = env.ee, env.block, env.slot_x
        grasp = G_STAY

        if not env.holding:
            # phase 1: go to the block, close when close enough
            err = blk - ee
            if np.linalg.norm(err) < 0.022:
                grasp = G_CLOSE
                a = np.zeros(2)
            else:
                a = err / STEP
        elif blk[1] > SAFE_Y - 0.02 and abs(blk[0] - sx) > 0.007:
            # phase 2: hold above the wall and line up with the slot
            a = np.array([(sx - blk[0]) / STEP, (SAFE_Y - blk[1]) / STEP])
        elif blk[1] > GOAL_Y:
            # phase 3: descend, still servoing x (this is the precise part)
            a = np.array([(sx - blk[0]) / STEP * 1.5, -1.0])
        else:
            # phase 4: let go
            grasp = G_OPEN
            a = np.zeros(2)

        a = np.clip(a, -1, 1) + self.rng.normal(0, self.noise, 2)
        if self.rng.random() < self.sloppiness:
            a *= 0.0
        return np.clip(a, -1, 1), grasp


class InterventionPolicy:
    """Decides *when* the human grabs the controller."""

    def __init__(self, seed=0, p0=0.3, half_life_eps=20,
                 burst=8, max_len=25, min_rate=0.02):
        self.rng = np.random.default_rng(seed)
        self.p0, self.half_life = p0, half_life_eps
        self.burst, self.max_len, self.min_rate = burst, max_len, min_rate
        self.remaining = 0
        self.length = 0

    def rate(self, episode):
        return max(self.min_rate, self.p0 * 0.5 ** (episode / self.half_life))

    def reset_episode(self):
        self.remaining, self.length = 0, 0

    def __call__(self, env, episode, info):
        if self.remaining > 0 and self.length < self.max_len:
            self.remaining -= 1
            self.length += 1
            return True
        self.remaining, self.length = 0, 0

        p = self.rate(episode)
        # trouble signals: jammed on the wall, or descending while misaligned
        jammed = info.get("contact", False)
        misaligned = (env.holding and env.block[1] < SAFE_Y
                      and abs(env.block[0] - env.slot_x) > 0.02)
        trigger = p * (3.0 if (jammed or misaligned) else 1.0)
        if self.rng.random() < min(trigger, 0.95):
            self.remaining = self.burst - 1
            self.length = 1
            return True
        return False


class KeyboardHuman:
    """Real teleop: WASD moves, J/K close/open, SPACE toggles takeover, Q quits.

    Raw-mode non-blocking stdin. Only used when you pass --human keyboard.
    """

    KEYS = {"w": (0, 1), "s": (0, -1), "a": (-1, 0), "d": (1, 0)}

    def __init__(self):
        import sys, termios, tty
        self._sys, self._termios, self._tty = sys, termios, tty
        self.fd = sys.stdin.fileno()
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.engaged = False

    def poll(self):
        import select
        keys = []
        while select.select([self._sys.stdin], [], [], 0)[0]:
            keys.append(self._sys.stdin.read(1).lower())
        return keys

    def act(self, env):
        a, grasp = np.zeros(2), G_STAY
        for k in self.poll():
            if k == " ":
                self.engaged = not self.engaged
            elif k == "q":
                raise KeyboardInterrupt
            elif k in self.KEYS:
                a += np.array(self.KEYS[k], dtype=float)
            elif k == "j":
                grasp = G_CLOSE
            elif k == "k":
                grasp = G_OPEN
        return np.clip(a, -1, 1), grasp

    def close(self):
        self._termios.tcsetattr(self.fd, self._termios.TCSADRAIN, self.saved)


def rollout_demo(env, human, record_images=False):
    """One expert trajectory. Used for the demo buffer and the reward dataset."""
    obs = env.reset()
    traj = {"obs": [], "act": [], "grasp": [], "next_obs": [],
            "success": [], "truncated": [], "images": [], "true_success": []}
    while True:
        a, g = human.act(env)
        img = env.render() if record_images else None
        nobs, succ, trunc, _ = env.step(a, g)
        traj["obs"].append(obs); traj["act"].append(a); traj["grasp"].append(g)
        traj["next_obs"].append(nobs); traj["success"].append(succ)
        traj["truncated"].append(trunc); traj["true_success"].append(succ)
        if record_images:
            traj["images"].append(env.render())   # image of the NEXT state
        obs = nobs
        if succ or trunc:
            break
    return traj
