# HIL-SERL from scratch

A working, readable implementation of **Human-in-the-Loop Sample-Efficient Robotic
Reinforcement Learning** ([Luo et al., 2024](https://hil-serl.github.io/static/hil-serl-paper.pdf)),
on a 2-D insertion task that runs on a laptop CPU in about five minutes.

Everything the paper describes as load-bearing is here: the binary reward
classifier trained on human labels, the two-buffer symmetric sampling scheme,
human interventions routed asymmetrically into those buffers, RLPD's LayerNorm
ensemble critic, and a separate DQN "grasp critic" for the gripper.

```
python scripts/01_collect_data.py              # human labels + demonstrations
python scripts/02_train_reward_classifier.py   # fit the reward function
python scripts/03_train_rl.py                  # RL with a human in the loop
python scripts/04_ablations.py                 # remove one piece at a time
```

---

## 1. The problem HIL-SERL is solving

On a real robot, two things are missing that every RL benchmark hands you for free:

**There is no reward function.** Nothing in the world tells the robot that the
RAM stick is seated. You cannot instrument it, and you cannot write it down from
pixels.

**There is no useful exploration.** Insertion tasks have millimetre clearances.
A randomly-initialised policy will never, not once in a week, stumble into
success. So there is nothing for RL to reinforce.

HIL-SERL's answer to the first is *learn the reward from a few hundred human
labels*. Its answer to the second is *let a human grab the controller whenever
the robot is about to fail*. The rest of the system exists to make those two
things work together.

---

## 2. How the reward function is trained

Code: [hil_serl/reward_classifier.py](hil_serl/reward_classifier.py),
[hil_serl/data_collect.py](hil_serl/data_collect.py)

The reward is a **binary image classifier**. Not a preference model, not a
learned dense potential — a plain sigmoid that answers "is this frame a success?"

```
r(s) = 1[ sigmoid(f_psi(image)) > tau ]   AND   side_check(s)
```

and zero otherwise. A successful episode gets exactly one reward of 1, on its
final transition, and terminates there.

### The data

The paper's budget is **~200 positives and ~1000 negatives**, about ten
teleoperated trajectories, roughly five minutes of a person's time. Two sources:

- **Whole trajectories.** Nearly every frame is a negative, so negatives are
  almost free. This also guarantees the classifier sees the states the policy
  will actually visit.
- **Deliberate near-misses.** You drive the robot into poses that *look* like
  success and label them 0. This is the part that matters, and the part people
  skip.

`_sample_near_boundary` in [data_collect.py](hil_serl/data_collect.py) spends
**32% of the negative budget** on one slice: block through the slot but not yet
seated. Not because that slice is common, but because it is where the classifier
is wrong, and a reward-maximising policy searches exactly there.

> A classifier's errors concentrate immediately past its decision boundary.
> That is also precisely where an optimiser goes. Uniformly-sampled negatives
> spend your label budget where the policy will never be.

### The threshold

The two error types are not symmetric, so accuracy is the wrong summary
statistic:

| error | what it costs |
|---|---|
| **false negative** — missed a real success | one wasted episode. The policy did the task, got nothing, moves on. Recoverable. |
| **false positive** — called a non-success a success | a hole in the objective. The policy finds the pose that fools the classifier and optimises into it. Your true success rate flatlines while every number you are logging looks healthy. |

So `select_threshold` fixes an FP budget and minimises FN subject to it, rather
than maximising accuracy. It then takes the *largest* tau within 5 points of that
best FN rather than the smallest feasible one — see below for why that margin
matters. On this run it picks `tau = 0.20`, at 7.7% false negatives.

### The side check

The paper ANDs a **force/torque threshold** into the classifier output. Here the
analogue is "the gripper must be open" — you cannot have released the peg while
still holding it, whatever the pixels say.

This is nearly free and it buys a lot. It cut false positives by 3x at zero cost
in false negatives, which in turn lets you run a *lower* tau at the same FP rate,
which buys back false negatives:

```
tau=0.5   FP 0.006 -> 0.002   (FN 0.145 -> 0.145)
tau=0.9   FP 0.002 -> 0.001   (FN 0.584 -> 0.584)
tau=0.95  FP 0.001 -> 0.000   (FN 0.813 -> 0.813)
```

Whenever a cheap, trustworthy predicate exists, AND it in. It is strictly better
than pushing the threshold up.

### I watched this fail

This is not hypothetical. An earlier version of this repo defined success as
"block has just cleared the wall". At 64x64 one pixel is 0.016 world units, so
that criterion sat *inside the classifier's own resolution limit* and left a
1-2 pixel band it could never get right. The policy found the band within 12k
steps:

```
step  11000 | eval true success 0.00 | classifier says 0.20 | false-pos gap +0.20
step  12000 | eval true success 0.05 | classifier says 0.70 | false-pos gap +0.65
```

It learned to hover one pixel short of seated, forever. The fix was to make the
success criterion resolvable in the image the classifier actually sees (require
the block to be *seated*, a ~4 pixel margin) and to spend the negative budget on
that band. **Your task definition has to be representable at your camera
resolution.** If it isn't, you have not written a hard reward function, you have
written an exploitable one.

This is why `train.py` logs true success and classifier success side by side.
The gap between them is reward hacking, measured.

---

## 3. How the policy learns from human interventions

Code: [hil_serl/agent.py](hil_serl/agent.py), [hil_serl/buffers.py](hil_serl/buffers.py),
[hil_serl/teleop.py](hil_serl/teleop.py)

### There is no imitation loss

The thing that surprises people: **HIL-SERL has no behaviour-cloning term.** An
intervention is stored as an ordinary off-policy transition `(s, a_human, r, s')`
and off-policy RL is free to learn from actions taken by any policy. The action
written into the buffer is the *human's*, not the one the policy proposed.

Two mechanisms do the work:

1. **Value propagation.** Early in training, human corrections are the only
   trajectories that ever reach reward 1. The critic backs that 1 up through
   them, so states along a correction acquire high Q. This is what converts a
   sparse terminal reward into a dense gradient.

2. **Policy improvement against the critic.** The actor maximises Q. Because
   symmetric sampling guarantees half of every batch comes from human states,
   the actor is repeatedly asked "what is the best action *here*, in the states
   the human rescued you into?" — and the critic already knows.

So the policy imitates the human only *through the value function*. That is
strictly better than cloning: if a correction was mediocre, the critic scores it
low and the actor does something better. BC would copy it. This is why HIL-SERL
policies routinely end up faster and more reliable than the operator who taught
them — impossible under pure imitation.

(An optional `bc_weight` is in the agent, off by default, so you can compare
against HG-DAgger/IWR-style methods. It is not part of the paper's method.)

### The buffer routing rule

Two buffers, and **every gradient batch is drawn half from each**:

```
demo buffer  <- offline demonstrations  +  human intervention transitions
rl buffer    <- everything the robot did, interventions included
```

The asymmetry is deliberate and is stated in the paper:

- **intervention transitions go into BOTH buffers**
- **the policy's own transitions go into the RL buffer only**

Why 50/50 rather than one pot? Because 20 demos is ~660 transitions. In a single
buffer that is 2.6% of a 25k-step run and falling — the human data would be
sampled into irrelevance exactly as the buffer fills with the failures you are
trying to correct. Symmetric sampling pins the ratio at 50% forever.

### When the human should intervene

`InterventionPolicy` encodes the protocol, which is more opinionated than it
looks:

- **Intervene often early, rarely later.** The rate decays as the policy
  improves. In the run below it goes 0.93 → 0.09.
- **Intervene where the policy is actually failing** — jammed against the wall,
  descending misaligned — not uniformly at random.
- **Keep interventions short.** The paper explicitly warns against long
  interventions that drive the episode all the way to success. Those inflate the
  value function: the critic learns "states near the human takeover are
  excellent" when in fact it was the *human*, not the state, that was excellent.
  `max_len` caps this.

### The gripper is a separate MDP

HIL-SERL splits the problem in two, sharing one state and one reward:

- continuous end-effector deltas → SAC
- discrete gripper `{stay, open, close}` → **DQN with a target network** (double
  DQN in `_update_grasp`)

A tanh-Gaussian over a dimension that wants three decisive values learns slowly
and chatters. A 3-way Q-table learns it almost immediately. A small negative
penalty on any non-`stay` action is what stops the gripper oscillating.

---

## 4. How the reward function helps the policy learn

This is the part that is easy to state and easy to get wrong.

The reward is 1 on exactly one transition per successful episode, 0 everywhere
else, and the episode terminates on success with the bootstrap masked out. So
for a policy that reaches success in `k` steps:

```
Q(s,a) ~ gamma^k
```

**The critic is not learning "how much reward is here". It is learning a
discounted distance-to-success.** That is the whole mechanism. A sparse binary
classifier, backed up through the Bellman equation, becomes a dense, smooth
scalar field over states that is highest next to the goal and decays smoothly
away from it. The actor then just climbs it.

This is why the pieces fit together the way they do:

- **Why human data is needed at all**: `gamma^k` is only informative if *some*
  trajectory actually reaches the 1. Demonstrations and interventions are both
  ways of manufacturing those trajectories; with neither, the field is flat at
  zero and the actor has nothing to climb. Which of the two you need depends on
  the task — see the ablations in §5, where 20 demonstrations were enough on
  their own and interventions became decisive only once demos were scarce.
- **Why false positives are catastrophic**: they put a 1 somewhere that isn't the
  goal, and the field grows a second peak. The actor cannot tell the difference,
  and the fake peak is usually easier to reach.
- **Why the discount matters**: it sets how fast the field decays, i.e. how far
  from the goal the gradient is still usable.

### The trap I hit, which is worth knowing about

The per-step *action advantage* under this scheme is roughly `(1 - gamma)` times
the value. With `gamma = 0.97` and ~33 steps to go, one action changes the return
by about 3%. Measured on demo data:

```
Q(demo action) 0.610   Q(random action) 0.594   advantage +0.016
```

The critic barely distinguishes a good action from a random one, because a single
2.5mm delta genuinely *doesn't* matter much. That signal has to survive the
critic's own approximation error, and with **offline demo data alone it does not**
— all demos succeed, so the critic never sees a failure, and it collapses to a
near-constant 0.6 everywhere. Pure offline RL on 100 perfect demonstrations
scored **0.00**.

Online failures are what anchor the value function down. This is a real argument
for the human-in-the-loop setup over "just collect more demos": you need the
policy's own mistakes in the buffer, and you need enough successes among them for
the contrast to mean something. HIL-SERL gets both.

### Temperature initialisation, a practical gotcha

SAC's default `alpha = 1.0` assumes rewards of order 1-100. Here every Q lives in
`[0, 1]`, so `alpha * log_pi` is an order of magnitude larger than the Q term: the
actor maximises entropy, stays uniform-random, the critic bootstraps off random
actions, and nothing propagates. Q sat at 0.03 and the policy emitted a constant
action.

Starting at `alpha = 0.05` fixed it — Q went 0.03 → 0.42 and the policy started
tracking the demos. The Lagrange update still raises alpha if the policy
collapses. **With sparse 0/1 rewards, check your entropy term against your Q
scale before you debug anything else.**

---

## 5. Results

`python scripts/03_train_rl.py` — 30k environment steps, ~6 minutes on a CPU.
Evaluation is 30 deterministic rollouts with **no human**, so it measures the
policy alone:

```
step   1000 | eval true success 0.00 | intervention rate 0.94
step   6000 | eval true success 0.00 | intervention rate 0.63
step  11000 | eval true success 0.00 | intervention rate 0.22
step  16000 | eval true success 0.00 | intervention rate 0.21
step  21000 | eval true success 0.33 | intervention rate 0.15
step  26000 | eval true success 0.97 | intervention rate 0.10
step  30000 | eval true success 0.93 | intervention rate 0.11
```


Two things to read off this. The success rate climbs from zero, and the
**intervention rate falls as it does** — the human is needed less and less. That
decay is the signature of the method working; it is what makes a training
session a person can actually sit through.

### Ablations

`python scripts/04_ablations.py` removes one piece at a time (20k-30k steps each,
run in parallel):

```
ablation      true success  classifier says    gap   eval curve (30 checkpoints)
full                  0.98             1.00  +0.02   ............+#+.+###+..+##.###
no-interv             0.93             0.99  +0.06   .........+####################
oracle                0.98             0.98  +0.00   ..........+.#+#####.##+#######
few-labels            0.21             0.90  +0.69   ...............+.+..........+.
```

- **oracle vs full** — with the classifier tuned as in §2, the learned reward
  costs essentially nothing against a perfect one. That is the headline claim of
  the reward-classifier approach, reproduced.
- **few-labels** — 20 positives / 100 negatives instead of 200 / 1000, side check
  off. It satisfies its own reward function 90% of the time and does the actual
  task 21% of the time. This is what reward hacking looks like in a log, and it
  is the failure mode you cannot detect without ground truth you would not have
  on a real robot. Note it is *not* a subtle degradation — the classifier score
  looks like a healthy training curve.
- **no-interv** — see below. This one did not go the way I expected.


### When interventions help, and when they hurt

`no-interv` reaching 0.93 against the full method's 0.98 is not what I expected,
and it is worth taking seriously rather than explaining away. With 20
demonstrations this task simply does not need a human: the demo buffer already
contains plenty of successes for the critic to back up.

So I cut the demonstrations to 3, where the human should matter most, and swept
the intervention rate (`--demos 3 --steps 20000`):

First, a single-seed sweep of the intervention rate:

```
config (3 demos, 20k)   mean interv  final true  first>0.2   eval curve
no interventions               0.00        0.86       9000   ........++++#+####+#
p0=0.15                        0.18        0.66      12000   ...........+#.#+##.#
p0=0.30                        0.21        0.98      10000   .........+##########
p0=0.90 (old default)          0.37        0.64      12000   ...........+.+###.##
```

Then 3 seeds each for the two configurations worth comparing:

```
config                  interv   final true success   mean   steps to 0.5
no interventions          0.00     0.58 0.81 0.87     0.75   11k  9k 12k
interventions p0=0.30     0.22     0.99 0.92 0.63     0.85   13k  9k 13k
```

**Read these carefully.** A moderate intervention rate gives a better mean (0.85
vs 0.75), but the per-seed distributions overlap heavily and three seeds does not
establish the difference. What *is* reproducible is the negative result: the
original high rate was worse than not intervening at all, on every comparison I
ran.


The original default took over on ~37% of steps, and it was **worse than not
intervening at all**. This reproduces the warning in the paper almost exactly:

> we should avoid persistently providing long sparse interventions that lead to
> task successes. Such an intervention strategy will cause the overestimation of
> the value function.

The mechanism is the one from §4. A critic trained only on human-generated
successes has no failures to anchor it and collapses to a near-constant — the
same degenerate solution that made pure offline RL score 0.00 here. **The
policy's own mistakes are load-bearing training data.** Take the controller away
too often and you have quietly converted online RL back into offline RL, using a
human as an expensive data-collection script.

The paper's own phrasing of the fix is the right one: issue *specific
corrections* and let the robot explore on its own otherwise. The defaults in
[teleop.py](hil_serl/teleop.py) are now `p0=0.3`, which lands near a 0.2
intervention fraction.

### A caveat on reading these numbers

Scoring on *training* episodes instead of evaluation rollouts inflates every
configuration that has a human in it — the human's own successes get counted.
The first version of `04_ablations.py` did exactly that and reported the full
method at 0.90 where held-out evaluation says otherwise. `summarise` now reads
eval rollouts only. If you extend this, keep that distinction: **the number you
report has to come from the policy acting alone.**

Single evaluation checkpoints are also noisy at n=30 — individual points swing
by 0.2 or more. The table averages the last three.

## 6. Where this departs from the paper

Honest list, so you know what you are reading:

| | paper | here |
|---|---|---|
| policy observation | wrist + side cameras, frozen ImageNet ResNet-10 encoder | 11-D state vector |
| reward classifier input | same camera images | 64x64 rendered image (real CNN, real labels) |
| actor / learner | two processes, async, over the network | interleaved in one process |
| environment | real Franka arms, 10 Hz impedance control | 2-D kinematic sim, no contact forces |
| human | SpaceMouse operator | scripted imperfect expert (`--human keyboard` for real teleop) |
| training | 1-6 hours on an RTX 4090 | ~5 minutes on a laptop CPU |

The **policy observing state instead of pixels** is the significant one. It is
what makes this run in minutes rather than hours, and it means this repo does not
exercise the visual-representation half of the problem. Everything about the
reward function, the interventions, and the buffer mechanics is unchanged — those
are the parts you asked about, and they are the parts that transfer.

Hyperparameters: 10-critic ensemble, min over a random 2, LayerNorm throughout,
UTD 4, batch 128, gamma 0.97, lr 3e-4, tau 0.005, alpha init 0.05. RLPD's paper
uses a 10-critic ensemble; the `serl` reference implementation defaults to 2, so
treat the ensemble size as a knob rather than a constant.

---

## 7. Porting this to a real robot

In rough order of how much trouble each will give you:

1. **Reward classifier first, and check it before you train anything.** Sweep it
   over held-out near-miss poses the way [step 2](scripts/02_train_reward_classifier.py)
   does. If it has a false-positive hole, RL *will* find it. Budget the labels
   asymmetrically toward near-misses.
2. **Find your side check.** Force/torque, gripper width, joint limits — anything
   cheap and trustworthy to AND in.
3. **Make success representable at your camera resolution.** See §2.
4. **Swap in a real teleop device** and keep `InterventionPolicy`'s protocol:
   frequent early, short, targeted at failures.
5. **Split the actor and learner into two processes.** On a real robot the actor
   must hold its control rate while the learner does gradient steps. Everything
   in [train.py](hil_serl/train.py) is already separated along that seam.
6. **Add an image encoder** to the policy if you want the full method — that is
   the one piece here that is genuinely simplified.

## Files

| file | what's in it |
|---|---|
| [hil_serl/env.py](hil_serl/env.py) | 2-D insertion task, renderer, ground-truth success (used only for scoring) |
| [hil_serl/data_collect.py](hil_serl/data_collect.py) | building the labelled reward dataset, near-miss sampling |
| [hil_serl/reward_classifier.py](hil_serl/reward_classifier.py) | the CNN, training, threshold selection, the side check |
| [hil_serl/teleop.py](hil_serl/teleop.py) | scripted + keyboard operators, intervention protocol |
| [hil_serl/buffers.py](hil_serl/buffers.py) | two buffers, symmetric sampling, routing rule |
| [hil_serl/networks.py](hil_serl/networks.py) | LayerNorm ensemble critic, tanh-Gaussian actor, grasp critic |
| [hil_serl/agent.py](hil_serl/agent.py) | RLPD updates + double-DQN grasp critic |
| [hil_serl/train.py](hil_serl/train.py) | the loop |
