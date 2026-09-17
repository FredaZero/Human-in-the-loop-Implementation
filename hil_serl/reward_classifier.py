"""The learned binary reward function.

This is the answer to "how do you get a reward on a real robot?". You cannot
instrument success, and you cannot hand-write it from images, so HIL-SERL asks
a human to *label* a few hundred frames and fits a binary classifier:

    r(s) = 1[ sigmoid(f_psi(image)) > tau ]      and optionally  AND  side_check(s)

Three details carry almost all of the practical weight:

1. LABEL BUDGET AND BALANCE. The paper collects ~200 positives and ~1000
   negatives (~10 teleoped trajectories, ~5 minutes of human time). Negatives
   are cheap because every non-final frame of a demo is a negative; positives
   are the scarce class.

2. THE NEGATIVES MUST BE HARD. A classifier trained only on "far from goal"
   negatives will fire the moment the robot is near the goal. The frames you
   must include are the near-misses: block in the slot but still gripped, block
   beside the slot, gripper closed at the right height. We generate exactly
   these below.

3. THE THRESHOLD IS ASYMMETRIC. A false negative costs you one wasted episode.
   A false positive is a reward-hacking hole: the policy will find the pose that
   fools the classifier and sit there forever, and your true success rate will
   flatline while the plotted return looks great. So tau is pushed up (0.9-0.95)
   and, where a cheap reliable signal exists (force/torque in the paper, gripper
   state here), it is ANDed in as a veto.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardClassifier(nn.Module):
    """Small CNN. The paper uses a frozen pretrained ResNet-10 + MLP head;
    at 64x64 with ~1200 labels a 4-layer conv net is the right size."""

    def __init__(self, img_size=64, ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, 16, 3, 2, 1), nn.GroupNorm(4, 16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.ReLU(),
        )
        feat = 64 * (img_size // 16) ** 2
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(feat, 128),
                                  nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):                       # x: (B,3,H,W) float in [0,1]
        return self.head(self.net(x)).squeeze(-1)


def to_tensor(imgs):
    """uint8 NHWC -> float NCHW in [0,1]."""
    x = torch.as_tensor(np.asarray(imgs), dtype=torch.float32) / 255.0
    if x.ndim == 3:
        x = x.unsqueeze(0)
    return x.permute(0, 3, 1, 2).contiguous()


def augment(x, rng):
    """Shift/brightness/noise. With ~1200 labels this is what keeps the
    classifier from memorising camera position instead of task state."""
    B = x.shape[0]
    pad = F.pad(x, (3, 3, 3, 3), mode="replicate")
    dx, dy = rng.integers(0, 7, 2)
    x = pad[:, :, dy:dy + x.shape[2], dx:dx + x.shape[3]]
    x = x * (1 + 0.15 * torch.randn(B, 1, 1, 1))
    x = x + 0.02 * torch.randn_like(x)
    return x.clamp(0, 1)


def train_classifier(images, labels, epochs=25, batch=64, lr=3e-4,
                     val_frac=0.2, seed=0, verbose=True):
    """Plain BCE on human labels. Returns (model, metrics)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    idx = rng.permutation(len(images))
    n_val = int(len(idx) * val_frac)
    val_i, tr_i = idx[:n_val], idx[n_val:]
    Xtr, ytr = to_tensor(images[tr_i]), torch.as_tensor(labels[tr_i], dtype=torch.float32)
    Xva, yva = to_tensor(images[val_i]), torch.as_tensor(labels[val_i], dtype=torch.float32)

    model = RewardClassifier(img_size=images.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            logits = model(augment(Xtr[b], rng))
            loss = F.binary_cross_entropy_with_logits(logits, ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        if verbose and (ep + 1) % 5 == 0:
            m = evaluate(model, Xva, yva, 0.5)
            print(f"  epoch {ep+1:3d}  loss {tot/len(perm):.4f}  "
                  f"val acc {m['acc']:.3f}  FP {m['fp']:.3f}  FN {m['fn']:.3f}")

    metrics = {f"tau={t}": evaluate(model, Xva, yva, t) for t in (0.5, 0.9, 0.95, 0.99)}
    return model, metrics, (Xva, yva, val_i)


@torch.no_grad()
def select_threshold(model, X, y, gate=None, fp_budget=0.01, slack=0.05, grid=None):
    """Pick tau the way you should: minimise false NEGATIVES subject to a hard
    budget on false POSITIVES.

    The two errors are not symmetric. A false negative costs you one episode -
    the policy tried, succeeded, got nothing, and moves on. A false positive is
    a hole in the objective: the policy will find the pose that fools the
    classifier and optimise into it, and your true success rate flatlines while
    everything you are logging looks healthy. So fix the FP budget first and
    spend whatever FN you must to meet it.

    `gate` is the cheap side-channel (gripper open / force-torque threshold);
    ANDing it in lets you run a LOWER tau at the same FP rate, which buys back
    false negatives for free.
    """
    model.eval()
    p = torch.sigmoid(model(X))
    grid = grid if grid is not None else np.arange(0.05, 1.0, 0.05)
    rows = []
    for tau in grid:
        pred = (p > float(tau)).float()
        if gate is not None:
            pred = pred * gate
        fp = pred[y == 0].mean().item() if (y == 0).any() else 0.0
        fn = (1 - pred[y == 1]).mean().item() if (y == 1).any() else 0.0
        rows.append((float(tau), fp, fn))

    ok = [r for r in rows if r[1] <= fp_budget + 1e-9]
    if ok:
        # Among feasible thresholds, take the LARGEST tau whose FN is within
        # `slack` of the best. Picking the smallest feasible tau minimises FN on
        # the validation set but parks you exactly on the budget edge, and that
        # edge is estimated from only a few hundred negatives. Under the policy's
        # own state distribution - which is not the label distribution - the true
        # FP rate is higher, and it is the FP rate the policy attacks. Buying
        # margin costs a couple of points of FN and is worth it every time.
        best_fn = min(r[2] for r in ok)
        near = [r for r in ok if r[2] <= best_fn + slack]
        tau, _fp, fn = max(near, key=lambda r: r[0])
    else:
        # Budget unreachable on this validation set - take the lowest FP we can
        # get. Never silently fall back to a hardcoded tau.
        tau, _fp, fn = min(rows, key=lambda r: (r[1], r[2]))
    return tau, fn


@torch.no_grad()
def evaluate(model, X, y, tau):
    model.eval()
    p = torch.sigmoid(model(X))
    pred = (p > tau).float()
    pos, neg = y == 1, y == 0
    return {
        "acc": (pred == y).float().mean().item(),
        "fp": (pred[neg] == 1).float().mean().item() if neg.any() else 0.0,   # negatives called success
        "fn": (pred[pos] == 0).float().mean().item() if pos.any() else 0.0,   # missed successes
    }


class LearnedReward:
    """What the RL loop actually queries, once per environment step.

    `side_check` is the analogue of the force/torque gate in the paper: a cheap,
    trustworthy predicate ANDed with the classifier to veto false positives.
    Here: "the gripper must be open" - you cannot have released the peg while
    still holding it, no matter what the pixels say.
    """

    def __init__(self, model, tau=0.9, side_check=True):
        self.model = model.eval()
        self.tau = tau
        self.side_check = side_check

    @torch.no_grad()
    def __call__(self, env):
        p = torch.sigmoid(self.model(to_tensor(env.render()))).item()
        fires = p > self.tau
        if self.side_check and not env.gripper_open:
            fires = False
        return (1.0 if fires else 0.0), fires, p


class OracleReward:
    """Ground-truth reward, for the ablation that isolates classifier error."""
    def __call__(self, env):
        s = env.true_success()
        return (1.0 if s else 0.0), s, float(s)
