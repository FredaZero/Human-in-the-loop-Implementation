"""STEP 2 - fit the reward function to the human's labels.

Watch the FP column. That is the number that decides whether step 3 works.
"""
import argparse, os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hil_serl.reward_classifier import train_classifier, evaluate, select_threshold, to_tensor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true",
                    help="train the skimped-budget classifier used by the ablation: "
                         "20 positives / 100 negatives and no deliberate near-misses")
    args = ap.parse_args()

    if args.small:
        from hil_serl.data_collect import collect_reward_dataset
        print("SKIMPED LABEL BUDGET: 20 positives / 100 negatives, no near-misses\n")
        images, labels, grip = collect_reward_dataset(n_pos=20, n_neg=100, n_traj=2,
                                                      label_noise=0.02, hard_negatives=False)
        d = {"gripper_open": grip}
        out_name = "reward_classifier_small.pt"
    else:
        d = np.load(os.path.join(ROOT, "data", "reward_data.npz"))
        images, labels = d["images"], d["labels"]
        out_name = "reward_classifier.pt"

    print(f"{len(labels)} labelled frames "
          f"({int(labels.sum())} positive / {int((1-labels).sum())} negative)\n")

    model, metrics, (Xva, yva, val_i) = train_classifier(images, labels, epochs=50, seed=0)

    print("\nHeld-out performance vs. the classification threshold tau:")
    print(f"  {'tau':>6} {'accuracy':>9} {'false pos':>10} {'false neg':>10}")
    for k, m in metrics.items():
        print(f"  {k.split('=')[1]:>6} {m['acc']:>9.3f} {m['fp']:>10.3f} {m['fn']:>10.3f}")
    print("\n  false pos = a NON-success the classifier calls success -> reward hacking")
    print("  false neg = a real success it misses      -> a wasted episode, recoverable")

    # what the cheap side-channel veto buys you, on top of the threshold
    print("\nWith the gripper-open side check ANDed in (the paper ANDs a")
    print("force/torque threshold), false positives at each tau:")
    X, y, g = to_tensor(images), torch.as_tensor(labels), torch.as_tensor(d["gripper_open"])
    with torch.no_grad():
        p = torch.sigmoid(model(X))
    for tau in (0.5, 0.9, 0.95):
        raw = (p > tau).float()
        gated = raw * g                      # must also have an open gripper
        neg = y == 0
        print(f"    tau={tau:<5} FP {raw[neg].mean():.3f} -> {gated[neg].mean():.3f}"
              f"   (FN {(1-raw[y==1]).mean():.3f} -> {(1-gated[y==1]).mean():.3f})")

    gate_va = torch.as_tensor(d["gripper_open"])[val_i]
    tau, fn = select_threshold(model, Xva, yva, gate=gate_va, fp_budget=0.01)
    print(f"\nSelected tau = {tau:.2f}  (largest tau within 5 points of the best")
    print(f"false-negative rate among those meeting FP <= 0.01 with the side check")
    print(f"on. Costs {fn:.1%} false negatives, which is recoverable; the extra")
    print(f"margin protects against the policy's state distribution not matching")
    print(f"the label distribution.)")

    torch.save({"state_dict": model.state_dict(), "tau": tau},
               os.path.join(ROOT, "data", out_name))
    print(f"Saved {ROOT}/data/{out_name}")
