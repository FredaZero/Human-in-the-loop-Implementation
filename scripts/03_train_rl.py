"""STEP 3 - RL fine-tuning with the learned reward and a human in the loop.

  python scripts/03_train_rl.py                    # full HIL-SERL
  python scripts/03_train_rl.py --no-interventions # RLPD from demos only
  python scripts/03_train_rl.py --oracle-reward    # perfect reward, upper bound
  python scripts/03_train_rl.py --tau 0.5 --no-side-check   # watch it hack
"""
import argparse, os, sys, csv, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hil_serl.reward_classifier import RewardClassifier, LearnedReward, OracleReward
from hil_serl.train import train

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_reward(args):
    if args.oracle_reward:
        print("reward: GROUND TRUTH (ablation upper bound)")
        return OracleReward()
    name = "reward_classifier_small.pt" if args.small_labels else "reward_classifier.pt"
    ckpt = torch.load(os.path.join(ROOT, "data", name))
    m = RewardClassifier(); m.load_state_dict(ckpt["state_dict"])
    tau = ckpt["tau"] if args.tau is None else args.tau      # step 2 picked one
    print(f"reward: learned classifier, tau={tau:.2f}, side_check={not args.no_side_check}")
    return LearnedReward(m, tau=tau, side_check=not args.no_side_check)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--demos", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--utd", type=int, default=4)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--bc-weight", type=float, default=0.0)
    ap.add_argument("--no-side-check", action="store_true")
    ap.add_argument("--no-interventions", action="store_true")
    ap.add_argument("--intervene-p0", type=float, default=0.3,
                    help="initial probability the human takes over")
    ap.add_argument("--intervene-halflife", type=float, default=20,
                    help="episodes for that probability to halve")
    ap.add_argument("--oracle-reward", action="store_true")
    ap.add_argument("--small-labels", action="store_true")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    reward_fn = build_reward(args)
    print(f"interventions: {not args.no_interventions}\n")

    agent, hist, evals = train(reward_fn, steps=args.steps, n_demos=args.demos, seed=args.seed,
                        utd=args.utd, use_interventions=not args.no_interventions,
                        intervention_kwargs={"p0": args.intervene_p0,
                                             "half_life_eps": args.intervene_halflife},
                        agent_kwargs={"bc_weight": args.bc_weight})

    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    out = os.path.join(ROOT, "runs", f"{args.tag}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0].keys())); w.writeheader(); w.writerows(hist)

    ev_out = os.path.join(ROOT, "runs", f"{args.tag}_eval.csv")
    with open(ev_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(evals[0].keys())); w.writeheader(); w.writerows(evals)

    final = evals[-3:]
    print(f"\nfinal eval (mean of last {len(final)} checkpoints, {30*len(final)} episodes, "
          f"no human): true success {np.mean([e['true'] for e in final]):.2f}, "
          f"classifier says {np.mean([e['classifier'] for e in final]):.2f}")
    last = hist[-50:]
    print(f"\nfinal 50 training episodes: true success "
          f"{np.mean([h['true_success'] for h in last]):.2f}, "
          f"intervention rate {np.mean([h['interv_frac'] for h in last]):.2f}")
    print(f"log -> {out}")
