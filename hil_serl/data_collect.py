"""Building the labelled dataset for the reward classifier.

On a real robot this is ~5 minutes of work: you teleoperate, and you press one
key for "success" and another for "not success". Two sources of frames:

  A. WHOLE TRAJECTORIES. Almost every frame is a negative. Cheap and it covers
     the on-distribution states the policy will actually visit.

  B. DELIBERATE NEAR-MISSES. You drive the robot to poses that *look* like
     success and label them 0: peg in the hole but still gripped, peg beside the
     hole, gripper open at the right height with nothing in it. This is the part
     everyone skips and it is the part that decides whether your policy learns
     the task or learns to pose for the camera.

Here (B) is produced by sampling configurations near the decision boundary
directly. `label_noise` flips a fraction of labels, because humans misclick.
"""

import numpy as np
from .env import PegInsertEnv, WALL_Y1, INSERT_Y
from .teleop import ScriptedHuman


def _sample_near_boundary(env, rng, hard_negatives=True):
    """Poses a human would deliberately record as hard negatives/positives."""
    env.slot_x = float(rng.uniform(0.35, 0.65))
    # Weighted, not uniform. A classifier's errors concentrate immediately past
    # the decision boundary, which is exactly where a reward-maximising policy
    # will go looking. So most of the negative budget is spent there.
    if hard_negatives:
        kind = rng.choice(6, p=[0.20, 0.13, 0.32, 0.08, 0.13, 0.14])
    else:
        # The naive dataset a first-time user collects: successes, plus negatives
        # sampled from wherever the robot happens to be. No deliberate near-misses.
        kind = rng.choice([0, 5], p=[0.25, 0.75])
    if kind == 0:        # through and released  -> positive
        env.block = np.array([env.slot_x + rng.normal(0, 0.02),
                              rng.uniform(0.11, INSERT_Y)])
        env.gripper_open = True
    elif kind == 1:      # through but STILL GRIPPED -> key hard negative
        env.block = np.array([env.slot_x + rng.normal(0, 0.02),
                              rng.uniform(0.11, INSERT_Y)])
        env.gripper_open = False
    elif kind == 2:      # NOT QUITE SEATED - through the slot but 1-5 px short.
        # The single most important slice of the dataset. Without it the policy
        # learns to hover just above the seated pose and collect reward forever.
        env.block = np.array([env.slot_x + rng.normal(0, 0.015),
                              rng.uniform(INSERT_Y + 0.005, INSERT_Y + 0.09)])
        env.gripper_open = bool(rng.random() < 0.5)
    elif kind == 3:      # hovering above the wall, about to go in
        env.block = np.array([env.slot_x + rng.normal(0, 0.03), rng.uniform(0.48, 0.62)])
        env.gripper_open = bool(rng.random() < 0.5)
    elif kind == 4:      # released beside the slot, wrong x
        env.block = np.array([env.slot_x + rng.choice([-1, 1]) * rng.uniform(0.08, 0.3),
                              rng.uniform(0.11, INSERT_Y)])
        env.gripper_open = True
    else:                # anywhere
        env.block = np.array([rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)])
        env.gripper_open = bool(rng.random() < 0.5)

    env.block[0] = np.clip(env.block[0], 0.08, 0.92)
    env.ee = env.block + rng.normal(0, 0.015, 2) if rng.random() < 0.8 \
        else np.array([rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)])
    env.holding = (not env.gripper_open) and np.linalg.norm(env.ee - env.block) < 0.05
    env.t = 0


def _push(env, pos, neg, pos_open, neg_open):
    img = env.render()
    if env.true_success():
        pos.append(img); pos_open.append(float(env.gripper_open))
    else:
        neg.append(img); neg_open.append(float(env.gripper_open))


def collect_reward_dataset(n_pos=200, n_neg=1000, seed=0, label_noise=0.02,
                           n_traj=10, img_size=64, verbose=True, hard_negatives=True):
    rng = np.random.default_rng(seed)
    env = PegInsertEnv(seed=seed, img_size=img_size)
    human = ScriptedHuman(seed=seed + 1)
    pos, neg = [], []
    pos_open, neg_open = [], []   # gripper state: the cheap "force/torque" side channel

    # --- source A: teleoperated trajectories (mostly negatives) --------------
    for _ in range(n_traj):
        env.reset()
        for _ in range(env.max_steps):
            a, g = human.act(env)
            env.step(a, g)
            _push(env, pos, neg, pos_open, neg_open)
            if env.true_success():
                # linger in the success state so we get more than one positive
                for _ in range(12):
                    env.step(rng.normal(0, 0.4, 2), 0)
                    _push(env, pos, neg, pos_open, neg_open)
                break

    # --- source B: deliberate near-boundary poses ----------------------------
    guard = 0
    while (len(pos) < n_pos or len(neg) < n_neg) and guard < 200_000:
        guard += 1
        _sample_near_boundary(env, rng, hard_negatives)
        if env.true_success():
            if len(pos) < n_pos:
                _push(env, pos, neg, pos_open, neg_open)
        elif len(neg) < n_neg:
            _push(env, pos, neg, pos_open, neg_open)

    images = np.array(pos[:n_pos] + neg[:n_neg], dtype=np.uint8)
    grip_open = np.array(pos_open[:n_pos] + neg_open[:n_neg], dtype=np.float32)
    labels = np.concatenate([np.ones(len(pos[:n_pos])), np.zeros(len(neg[:n_neg]))]).astype(np.float32)

    # humans misclick
    if label_noise > 0:
        flip = rng.random(len(labels)) < label_noise
        labels[flip] = 1 - labels[flip]
        if verbose:
            print(f"  simulated human label noise: flipped {flip.sum()} / {len(labels)} labels")

    if verbose:
        print(f"  dataset: {int((labels==1).sum())} positives, "
              f"{int((labels==0).sum())} negatives, images {images.shape}")
    return images, labels, grip_open
