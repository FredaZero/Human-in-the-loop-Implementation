"""A tiny 2-D stand-in for the precise-insertion tasks in HIL-SERL.

The point of this environment is to reproduce the *properties* that make the
real tasks hard, not to be a physics simulator:

  * sparse success            - no shaped reward exists
  * tight clearance           - +/- 0.01 on a 0.08 block, so random exploration
                                essentially never solves it
  * contact / jamming         - the block cannot pass through the wall, it gets
                                stuck exactly like a peg on a chamfer
  * a discrete gripper        - which is why HIL-SERL trains a separate
                                "grasp critic" with DQN
  * success is visually       - the reward classifier has to look at the image
    obvious but hard to         to tell "released inside the slot" from
    write down                  "still held above the slot"

Frame: x,y in [0,1], y points up. The wall is a horizontal band with a notch.
"""

import numpy as np

# --- geometry ---------------------------------------------------------------
WALL_Y0, WALL_Y1 = 0.30, 0.42        # wall band
SLOT_HALF = 0.050                    # half width of the notch
BLOCK_HALF = 0.040                   # half width of the block  -> 0.010 clearance
GRASP_RADIUS = 0.050                 # gripper must be this close to grab
GOAL_Y = 0.18                        # target depth for the block centre
INSERT_Y = 0.20                      # "seated": pushed a further 0.06 down.
# The success criterion has to be resolvable in the image the classifier sees.
# At 64x64 one pixel is 0.016, so a criterion of "just barely clear of the wall"
# sits inside the classifier's own resolution limit and leaves a 1-2 pixel band
# it can never get right - which a policy will find and exploit. Requiring the
# block to be seated leaves a ~4 pixel margin between success and near-miss.
STEP = 0.025                         # metres per unit action
XLO, XHI, YLO, YHI = 0.05, 0.95, 0.08, 0.95

# discrete gripper actions
G_STAY, G_OPEN, G_CLOSE = 0, 1, 2
N_GRASP_ACTIONS = 3

OBS_DIM = 11
ACT_DIM = 2


class PegInsertEnv:
    """Pick the block, thread it through the slot, release it.

    `step` returns the *ground-truth* success flag only so that we can score the
    learned reward classifier honestly. The RL agent in train.py never reads it.
    """

    def __init__(self, seed=0, max_steps=120, img_size=64):
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.img_size = img_size
        self._grid_cache = {}

    # -- core ---------------------------------------------------------------
    def reset(self):
        self.slot_x = float(self.rng.uniform(0.35, 0.65))
        self.block = np.array([self.rng.uniform(0.20, 0.80), 0.70])
        self.ee = self.block + np.array([self.rng.uniform(-0.06, 0.06), 0.12])
        self.gripper_open = True
        self.holding = False
        self.t = 0
        self.contact = False
        return self._obs()

    def step(self, action, grasp_action=G_STAY):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0) * STEP
        self.contact = False

        # 1) gripper command resolves first (as on a real robot)
        if grasp_action == G_CLOSE:
            self.gripper_open = False
        elif grasp_action == G_OPEN:
            self.gripper_open = True
        # Attachment is re-evaluated every step rather than only at the instant
        # the gripper closes. Latching it to that instant makes "closed on empty
        # air" an absorbing failure state, which no demonstration ever visits and
        # which the grasp critic therefore cannot learn to escape - it closes at
        # t=0 and the episode is unrecoverable before it starts.
        self.holding = (not self.gripper_open
                        and np.linalg.norm(self.ee - self.block) < GRASP_RADIUS)

        # 2) Cartesian delta move, rejected on collision (the "contact" event)
        new_ee = np.clip(self.ee + a, [XLO, YLO], [XHI, YHI])
        new_block = self.block + (new_ee - self.ee) if self.holding else self.block
        if self._collides(new_ee, point=True) or self._collides(new_block, point=False):
            self.contact = True                      # move rejected, robot jams
        else:
            self.ee, self.block = new_ee, new_block

        self.t += 1
        truncated = self.t >= self.max_steps
        return self._obs(), self.true_success(), truncated, {"contact": self.contact}

    # -- predicates ---------------------------------------------------------
    def _collides(self, p, point):
        half = 0.0 if point else BLOCK_HALF
        if p[1] + half <= WALL_Y0 or p[1] - half >= WALL_Y1:
            return False                             # not level with the wall
        return not (p[0] - half >= self.slot_x - SLOT_HALF
                    and p[0] + half <= self.slot_x + SLOT_HALF)

    def true_success(self):
        """Ground truth. Used for evaluation and to simulate a human labeller."""
        return bool(
            self.block[1] <= INSERT_Y                # fully below the wall
            and self.block[1] >= 0.10
            and abs(self.block[0] - self.slot_x) < 0.05
            and self.gripper_open                    # and actually let go
        )

    def inserted_but_held(self):
        """Through the slot but still gripped - the classifier's hardest negative."""
        return bool(self.block[1] <= INSERT_Y and not self.gripper_open)

    # -- observations -------------------------------------------------------
    def _obs(self):
        ex, ey = self.ee
        bx, by = self.block
        return np.array([
            ex, ey, bx, by, self.slot_x,
            1.0 if self.gripper_open else 0.0,
            1.0 if self.holding else 0.0,            # readable from gripper width
            ex - bx, ey - by,
            bx - self.slot_x, by - GOAL_Y,
        ], dtype=np.float32)

    def _grid(self, size):
        if size not in self._grid_cache:
            xs = (np.arange(size) + 0.5) / size
            ys = 1.0 - (np.arange(size) + 0.5) / size
            self._grid_cache[size] = np.meshgrid(xs, ys)
        return self._grid_cache[size]

    def render(self, size=None):
        """What the (simulated) side camera sees. uint8 HxWx3."""
        size = size or self.img_size
        X, Y = self._grid(size)
        img = np.full((size, size, 3), 0.94, dtype=np.float32)

        wall = ((Y >= WALL_Y0) & (Y <= WALL_Y1)
                & ~((X >= self.slot_x - SLOT_HALF) & (X <= self.slot_x + SLOT_HALF)))
        img[wall] = (0.24, 0.25, 0.29)

        blk = (np.abs(X - self.block[0]) <= BLOCK_HALF) & (np.abs(Y - self.block[1]) <= BLOCK_HALF)
        img[blk] = (0.91, 0.42, 0.13)

        # two gripper prongs; their spacing is the only cue for open vs closed
        off = 0.075 if self.gripper_open else 0.042
        prong = ((np.abs(np.abs(X - self.ee[0]) - off) <= 0.012)
                 & (np.abs(Y - self.ee[1]) <= 0.045))
        img[prong] = (0.15, 0.35, 0.80)
        return (img * 255).astype(np.uint8)
