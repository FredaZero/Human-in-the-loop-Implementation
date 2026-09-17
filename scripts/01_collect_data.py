"""STEP 1 - collect the two datasets a human has to provide.

  (a) ~1200 labelled frames for the reward classifier  (~5 min of teleop)
  (b) 20 demonstrations for the demo buffer            (~5 min of teleop)

Real system: a SpaceMouse, a wrist camera, and two keys for the labels.
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hil_serl.data_collect import collect_reward_dataset
from hil_serl.env import PegInsertEnv
from hil_serl.teleop import ScriptedHuman, rollout_demo

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    noise = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02

    print("Collecting reward-classifier labels...")
    imgs, labels, grip = collect_reward_dataset(n_pos=200, n_neg=1000, label_noise=noise)
    np.savez_compressed(os.path.join(OUT, "reward_data.npz"),
                        images=imgs, labels=labels, gripper_open=grip)

    print("Collecting demonstrations...")
    env, human = PegInsertEnv(seed=100), ScriptedHuman(seed=101)
    lens = [len(rollout_demo(env, human)["obs"]) for _ in range(20)]
    print(f"  20 demos, mean length {np.mean(lens):.1f} steps "
          f"({np.sum(lens)} transitions total)")
    print(f"\nSaved to {OUT}/reward_data.npz")
