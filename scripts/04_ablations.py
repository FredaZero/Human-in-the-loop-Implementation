"""STEP 4 - ablations. Each one removes a single piece of HIL-SERL.

  full        everything
  no-interv   demos only, no human in the loop      -> is the human necessary?
  oracle      perfect reward, human in the loop     -> upper bound
  few-labels  reward classifier from 20/100 labels
              instead of 200/1000, side check off   -> watch it get hacked

Run them in parallel; each takes a few minutes.
"""
import argparse, os, subprocess, sys, csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

ABLATIONS = {
    "full":      [],
    "no-interv": ["--no-interventions"],
    "oracle":    ["--oracle-reward"],
    "few-labels": ["--small-labels", "--no-side-check"],
}


def bar(v, width=28):
    n = int(round(v * width))
    return "#" * n + "." * (width - n)


def summarise(steps):
    """Scored on EVALUATION rollouts - deterministic policy, no human - not on
    training episodes. Training success is inflated by whatever the human did."""
    print(f"\n{'ablation':<12} {'true success':>13} {'classifier says':>16} {'gap':>6}   eval curve")
    for name in ABLATIONS:
        path = os.path.join(ROOT, "runs", f"abl_{name}_eval.csv")
        if not os.path.exists(path):
            print(f"{name:<12} {'(missing)':>13}")
            continue
        rows = list(csv.DictReader(open(path)))
        last = rows[-3:]
        t = sum(float(r["true"]) for r in last) / len(last)
        c = sum(float(r["classifier"]) for r in last) / len(last)
        curve = "".join("#" if float(r["true"]) > 0.6 else
                        ("+" if float(r["true"]) > 0.2 else ".") for r in rows)
        print(f"{name:<12} {t:>13.2f} {c:>16.2f} {c-t:>+6.2f}   {curve}")
    print("\n  'true success' is the honest metric. 'classifier says' is what the")
    print("  agent was optimising. A gap between them is reward hacking.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--summarise-only", action="store_true")
    args = ap.parse_args()

    if not args.summarise_only:
        procs = []
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        for name, flags in ABLATIONS.items():
            cmd = [PY, "-u", os.path.join(ROOT, "scripts", "03_train_rl.py"),
                   "--steps", str(args.steps), "--tag", f"abl_{name}"] + flags
            log = open(os.path.join(ROOT, "runs", f"abl_{name}.log"), "w")
            procs.append((name, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)))
            print(f"launched {name}")
        for name, p in procs:
            p.wait(); print(f"{name} finished (exit {p.returncode})")

    summarise(args.steps)
