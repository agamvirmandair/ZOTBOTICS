"""
SpiderEnv — Gymnasium environment for the SpdrBot 4-arm / 12-servo robot.

URDF: SpdrBot_description copy/urdf/SpdrBot.urdf
All 12 controlled joints are type="continuous" (no hard limits in URDF).
Software limits of ±JOINT_LIMIT rad are enforced by the env.

Elbow/wrist joint axes are diagonal (±0.707, ±0.707, 0) in the XY plane,
arising from the physical servo mounting angle in the CAD model.

  Leg │ Servo    │ URDF joint   │ Axis    │ PyBullet idx │ Motion
  ────┼──────────┼──────────────┼─────────┼──────────────┼──────────────────
  1   │ shoulder │ Revolute 77  │ Z       │ 0            │ hip swing (yaw)
  1   │ elbow    │ Revolute 90  │ XY-diag │ 1            │ arm raise/lower
  1   │ wrist    │ Revolute 104 │ XY-diag │ 2            │ forearm curl
  2   │ shoulder │ Revolute 78  │ Z       │ 3            │ hip swing
  2   │ elbow    │ Revolute 93  │ XY-diag │ 4            │ arm raise/lower
  2   │ wrist    │ Revolute 103 │ XY-diag │ 5            │ forearm curl
  3   │ shoulder │ Revolute 79  │ Z       │ 6            │ hip swing
  3   │ elbow    │ Revolute 92  │ XY-diag │ 7            │ arm raise/lower
  3   │ wrist    │ Revolute 102 │ XY-diag │ 8            │ forearm curl
  4   │ shoulder │ Revolute 80  │ Z       │ 9            │ hip swing
  4   │ elbow    │ Revolute 91  │ XY-diag │ 10           │ arm raise/lower
  4   │ wrist    │ Revolute 105 │ XY-diag │ 11           │ forearm curl

Physical leg positions (shoulder servo XY in base frame):
  Leg 1 (Rev 77): rear  +Y  →  (-0.0752,  0.0301)
  Leg 2 (Rev 78): rear  −Y  →  (-0.0752, -0.0797)
  Leg 3 (Rev 79): front −Y  →  ( 0.0752, -0.0301)
  Leg 4 (Rev 80): front +Y  →  ( 0.0752,  0.0301)

Standing pose:
  STAND_ANGLES are currently set to zero (neutral / flat pose).

CPG gait (mirrored trot):
  Diagonal pair A (leg0=rear+Y, leg2=front-Y):  phase offset = 0
  Diagonal pair B (leg1=rear-Y, leg3=front+Y):  phase offset = π
  Explicit stance/swing separation (duty cycle = 0.65):
    Stance (65%): shoulder sweeps dir*(+AMP→-AMP), foot backward, elbow down.
    Swing  (35%): shoulder sweeps dir*(-AMP→+AMP), foot forward, elbow lifted.
  GAIT_SHOULDER_DIRS = [+1, -1, -1, +1]:
    +Y-side legs (0, 3): positive sweep = foot forward.
    -Y-side legs (1, 2): multiplied by -1 so foot also sweeps forward→backward.
    All four legs produce symmetric +X thrust; lateral forces cancel → no spin.
  RL action is treated as a residual correction on top of CPG targets.

Action  (12,) float32 — residual joint-angle offsets (rad), clipped to ±1.2
Obs    (375,) float32 — 360 LiDAR distances (m)  +  12 joint positions (rad)  +  3 IMU (roll, pitch, yaw_rate)

Usage
-----
    env = SpiderEnv(render_mode="human")
    obs, _ = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

import math
import os
import random

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces


# ── Controlled joint names, in action-vector order ────────────────────────────
# Derived by tracing each kinematic chain in the new URDF:
#   shoulder → servo_arm → arm-a → elbow servo → servo_arm → arm-b → wrist servo → foot
JOINT_NAMES = [
    "Revolute 77",   # leg 1 – shoulder  (Z-axis,    PyBullet idx 0)
    "Revolute 90",   # leg 1 – elbow     (XY-diag,   PyBullet idx 1)
    "Revolute 104",  # leg 1 – wrist     (XY-diag,   PyBullet idx 2)
    "Revolute 78",   # leg 2 – shoulder  (Z-axis,    PyBullet idx 3)
    "Revolute 93",   # leg 2 – elbow     (XY-diag,   PyBullet idx 4)
    "Revolute 103",  # leg 2 – wrist     (XY-diag,   PyBullet idx 5)
    "Revolute 79",   # leg 3 – shoulder  (Z-axis,    PyBullet idx 6)
    "Revolute 92",   # leg 3 – elbow     (XY-diag,   PyBullet idx 7)
    "Revolute 102",  # leg 3 – wrist     (XY-diag,   PyBullet idx 8)
    "Revolute 80",   # leg 4 – shoulder  (Z-axis,    PyBullet idx 9)
    "Revolute 91",   # leg 4 – elbow     (XY-diag,   PyBullet idx 10)
    "Revolute 105",  # leg 4 – wrist     (XY-diag,   PyBullet idx 11)
]

_DEFAULT_URDF = os.path.join("SpdrBot_description copy", "urdf", "SpdrBot.urdf")
URDF_PATH = os.environ.get("SPDRBOT_URDF", _DEFAULT_URDF)

# ── Standing-pose target angles (radians) ─────────────────────────────────────
# Layout: [shoulder, elbow, wrist] × 4 legs  (same order as JOINT_NAMES).
#
# TODO: these are placeholder zeros for the new CAD-accurate URDF.
# The elbow/wrist joints now have diagonal XY axes, so the old angle values
# no longer correspond to the same physical poses.  Calibrate by:
#   1. Run test_env(render=True, n_steps=0) to see the neutral pose.
#   2. Incrementally adjust elbow and wrist values until the feet just
#      contact the ground and the robot is stable.
#   3. Update SPAWN_Z in _load_robot() to match the new standing height.
#
STAND_ANGLES = np.array([
    0.0, 0.0, 0.0,   # leg 1 – rear  +Y
    0.0, 0.0, 0.0,   # leg 2 – rear  −Y
    0.0, 0.0, 0.0,   # leg 3 – front −Y
    0.0, 0.0, 0.0,   # leg 4 – front +Y
], dtype=np.float64)

# ── CPG gait parameters ───────────────────────────────────────────────────────
GAIT_FREQUENCY  = 2.0     # Hz — stride cycles per second (higher = faster walk)
GAIT_SIM_HZ     = 240.0   # PyBullet default physics rate
GAIT_PHASE_STEP = 2.0 * math.pi * GAIT_FREQUENCY / GAIT_SIM_HZ

GAIT_AMP_SHOULDER = 0.35  # rad — shoulder sweep amplitude
GAIT_AMP_LIFT     = 0.45  # rad — how high the elbow pulls up during swing

# Phase offsets — perfectly symmetric trot (no deliberate asymmetry at this stage):
#   Pair A (leg0=rear+Y, leg2=front-Y) — true left diagonal, phase 0.0
#   Pair B (leg1=rear-Y, leg3=front+Y) — true right diagonal, phase π
# Both legs in a pair are in swing simultaneously; the two pairs alternate.
# No asymmetric offset: any residual spin is now caused by geometry, not phase.
GAIT_PHASE_OFFSETS = np.array([0.0, math.pi, 0.0, math.pi])

# Shoulder direction multiplier per leg.
#
# The shoulder servo rotates around the Z-axis.  Positive rotation sweeps the
# arm tip counterclockwise when viewed from above.  The foot sweep direction in
# X depends on which side the arm extends from the body:
#
#   +Y side (legs 0 and 3): arm points outward in +Y direction.
#          Positive shoulder → foot sweeps toward +X (forward).
#          So during stance (power stroke), sweep is +AMP → -AMP: correct.
#
#   -Y side (legs 1 and 2): arm points outward in -Y direction.
#          Positive shoulder → foot sweeps toward -X (backward — WRONG!).
#          Without correction, the stance stroke pushes the foot forward
#          instead of backward, producing a backward ground force → net torque.
#
# Multiplying by -1 for -Y legs inverts their shoulder sweep so every leg
# performs the same physical motion: foot starts forward, sweeps backward
# during stance, then lifts and recovers during swing.  Left/right forces
# now cancel laterally and sum in +X → forward motion without rotation.
# Signs are negated from the original geometric prediction because the URDF
# arm geometry means positive Z-rotation on +Y-side legs actually sweeps the
# foot toward -X (confirmed by running the sim).  Negating corrects this.
GAIT_SHOULDER_DIRS = np.array([-1.0, +1.0, +1.0, -1.0])

# ── Substeps and episode length ───────────────────────────────────────────────
# Each call to env.step() runs PHYSICS_SUBSTEPS physics steps before returning.
# The agent makes one decision, then the robot acts on it for that many ticks.
#
#   PHYSICS_SUBSTEPS = 8  →  each agent step = 8/240 ≈ 33 ms of simulation
#   MAX_STEPS        = 1000 →  episode = 1000 × 33 ms ≈ 33 seconds
#                           = ~66 full gait cycles at 2.0 Hz
#
# Without substeps a 500-step episode was only ~2 seconds — far too short for
# the robot to move any meaningful distance before the episode ended.
PHYSICS_SUBSTEPS = 8
MAX_STEPS        = 2000

# ── Stability threshold ───────────────────────────────────────────────────────
FALL_ANGLE_RAD = 1.05    # ~60° in pitch or roll — robot considered fallen

# ── Reward weights ────────────────────────────────────────────────────────────
# Two priorities only:
#   1. Obstacle avoidance  (proximity + forward-danger + collision + avoidance)
#   2. Maximise +X         (forward displacement + progress shaping)
FORWARD_WEIGHT       = 8.0    # X-displacement reward per metre
SAFE_RADIUS          = 2.0    # m — reduced from 3.5; 3.5 put all 3 obstacles in range simultaneously, causing conflicting repulsion vectors
OBSTACLE_WEIGHT      = 5.0    # mild background proximity signal — approach+danger carry the main weight
# Exponential steepness: penalty = (exp(EXP_K * closeness) - 1) / (exp(EXP_K) - 1)
# EXP_K=3.5 → more linear early gradient so the robot gets a clear "start evading" signal
# from the outer edge of the safe zone, not just the final approach.
EXP_K                = 3.5
# COLLISION_PENALTY must dominate the maximum possible accumulated episode reward.
COLLISION_PENALTY    = -10000.0
FALL_PENALTY         = -5.0   # terminal penalty for tipping over

# Lateral evasion reward — rewards velocity *away* from each hazard.
# Set high so active evasion (away_speed ≈ 0.2–0.3 m/s) is clearly net-positive
# compared to the proximity penalty alone, making "evade" visibly better than "freeze".
LATERAL_EVASION_WEIGHT = 10.0
# Extra penalty when the robot is heading directly into a hazard (heading-based).
FORWARD_DANGER_WEIGHT  = 8.0
# Approach velocity penalty — penalises actual velocity *toward* an obstacle.
# This is the primary signal against head-on charges; kept higher than FORWARD_DANGER.
APPROACH_WEIGHT        = 12.0
# Clear-path bonus — extra reward for +X velocity when nothing is directly ahead.
# forward_threat = max(closeness × forward_alignment) across all hazards in safe zone.
# bonus = max(0, vx) × (1 − forward_threat) × weight
# Full strength when the forward arc is empty; fades to 0 as an obstacle enters it.
CLEAR_PATH_WEIGHT      = 12.0  # raised from 4.0 — must make forward speed clearly profitable vs penalty drain
ALIGNMENT_WEIGHT       = 2.0   # reward for facing +X when forward path is clear

# Direct per-step reward for actively spinning toward +X while misaligned.
# reward = max(0, -sin(yaw) * yaw_rate) * (1-max_closeness) * weight
#   yaw > 0 (facing left)  → reward negative yaw_rate (turn clockwise = toward +X)
#   yaw < 0 (facing right) → reward positive yaw_rate (turn counter-clockwise = toward +X)
# Gated by (1-max_closeness) so it doesn't fight active obstacle evasion.
YAW_REALIGN_WEIGHT     = 8.0   # raised from 4.0 — must make spinning back to +X clearly profitable
# Penalise spinning *away* from +X (wrong spin direction) when path is clear.
# Complement of yaw_realign_reward: positive when sin(yaw)*yaw_rate > 0, i.e.
#   yaw > 0 (facing left)  and yaw_rate > 0 → spinning further left  (bad)
#   yaw < 0 (facing right) and yaw_rate < 0 → spinning further right (bad)
# This directly discourages the "always turn left" policy collapse.
# Gate is now halved (0.5×) so wrong spin is still penalised even during obstacle
# evasion — prevents the robot committing to a full 90° dodge spin.
WRONG_SPIN_WEIGHT      = 12.0  # raised from 6.0

# Re-alignment penalties.
LATERAL_DRIFT_WEIGHT   = 4.0   # penalise |vy| — ungated so diagonal wall-drift is always costly
# Always-active heading error penalty — NO clearance gate.
# A gate would suppress this near walls, which is exactly when the robot must
# correct its heading fastest to avoid crashing into them.
YAW_CORRECTION_WEIGHT  = 8.0   # raised from 4.0 — robot at 90° yaw must pay 8/step to make sustained misalignment expensive
BACKWARD_WEIGHT        = 15.0  # penalise negative world-X velocity — must dominate obstacle avoidance savings from U-turns

# Observation normalisation constants (not reward weights — kept for obs building).
YAW_RATE_CLIP      = 5.0   # rad/s ceiling for obs normalisation
YAW_RATE_DEADBAND  = 0.40  # rad/s deadband (obs only)

# Arena wall proximity zone.
WALL_SAFE_RADIUS   = 2.5   # m — raised from 2.0; gives 2.5m warning before side walls instead of 1m, earlier correction signal

# Front-arc LiDAR penalty — penalises short readings in the robot-relative forward sector.
# This is the primary learnable replacement for yaw-based forward_danger:
# the agent observes short front-LiDAR → gets penalised → learns not to face walls/obstacles.
# FRONT_ARC_RAYS: number of rays on each side of heading (ray 0) → total arc = 2×FRONT_ARC_RAYS°
# FRONT_ARC_SAFE: distance (m) beyond which the penalty is zero
FRONT_ARC_RAYS   = 40     # ±40° arc = 80° total forward sector
# Raised from 2.5 → 3.5: side walls sit at y=±2.95 m.  At 2.5 m the wall beyond
# the centre-line was INVISIBLE to this penalty (1−2.95/2.5 < 0 → clamped to 0),
# so a 90° spin in open space cost nothing observable.  At 3.5 m the same wall
# gives closeness ≈ 0.16 and the agent can see and learn from it.
FRONT_ARC_SAFE   = 3.5    # m — penalty starts when anything is closer than this ahead
FRONT_ARC_WEIGHT = 10.0   # raised from 4.0 — front-arc LiDAR is the only heading signal the agent can observe; must dominate

# Potential-based progress shaping toward the goal.
# Grounded in Ng et al. (1999): Φ(s) = GOAL_X − x, so the per-step shaping reward
# equals (GOAL_X − x_prev) − (GOAL_X − x_curr) = dx (only when x < GOAL_X).
# Equivalent to adding a second forward-weight that stops contributing once the
# goal is reached, giving a strong dense signal for every centimetre of progress.
GOAL_X           = 5.0     # m — target X coordinate (must match step() check)
PROGRESS_WEIGHT  = 25.0    # raised from 15.0 — stronger pull toward goal to overcome per-step penalty drain
# Small per-step cost so the agent cannot avoid the goal-penalty of wandering.
# An efficient path (few steps, reaches goal) pays less total than a slow/wandering path.
STEP_PENALTY     = 0.4     # subtracted every step regardless of what else happens

# Minimum forward velocity penalty — fires when the front arc is clear but the
# robot is barely moving forward.  Breaks the "drift sideways = safe" local optimum
# by making low vx explicitly costly when there is nothing ahead to justify it.
# penalty = max(0, MIN_VX_TARGET − vx) × (1 − max_forward_threat) × weight
# Gated by (1 − max_forward_threat) so it stays silent when an obstacle is directly
# ahead — the robot must evade first, resume speed after.
MIN_VX_TARGET    = 0.10    # m/s — target minimum forward speed when path is clear
MIN_VX_WEIGHT    = 6.0     # weight — enough to make idling clearly worse than walking

# ── Residual action scale and smoothing ───────────────────────────────────────
# RESIDUAL_SCALE multiplies the agent's raw action before it is added to the
# CPG target.  It must be small enough that the CPG dominates.
#
#   CPG shoulder amplitude : ±0.55 rad  (range = 1.10 rad)
#   Max residual (raw)     : ±(1.2 × 0.15) = ±0.18 rad   ≈ 33% of CPG swing
#
# This ensures the CPG provides ≥35% of the control signal at all times, so
# random RL exploration cannot flip joint directions or override the gait.
# Raised from 0.15 to 0.30 to give the agent more lateral steering authority.
RESIDUAL_SCALE = 0.30

# EMA weight applied to the residual each step before it reaches the joints.
# Acts as a first-order IIR low-pass filter: only ACTION_SMOOTH_ALPHA of the
# new action bleeds through per step; the rest carries over from the previous.
#   α = 0.70  →  ~1-step time constant  →  ~33 ms lag at 33 ms/step
# Raised from 0.55 to 0.70 so evasive lateral actions respond faster.
ACTION_SMOOTH_ALPHA = 0.70

# ── Wall planes for direct proximity computation ───────────────────────────────
# Replaces getClosestPoints() for walls to fix two bugs:
#   1. approach_speed and forward_alignment are zero for side walls when the
#      robot is walking straight — the closest-point normal is perpendicular to
#      velocity/heading so all active signals vanish, leaving only proximity.
#   2. getClosestPoints() returns unreliable normals at wall end-caps (corners),
#      corrupting approach/avoidance/danger signals near arena corners.
#
# Each entry: (perp_axis, inner_face_coord, repulsion_dir_2d)
#   perp_axis        — 0=X, 1=Y (the axis perpendicular to the wall face)
#   inner_face_coord — coordinate of the wall's inner face (robot-side surface)
#   repulsion_dir    — unit vector pointing FROM wall INTO arena interior
#
# Wall geometry from reset():
#   top wall    [2.5,  3.0, 0.5] half_extent Y=0.05 → inner face y= 2.95
#   bottom wall [2.5, -3.0, 0.5] half_extent Y=0.05 → inner face y=-2.95
#   back wall   [-1.0, 0.0, 0.5] half_extent X=0.05 → inner face x=-0.95
#   front wall  [ 6.0, 0.0, 0.5] half_extent X=0.05 → inner face x= 5.95
_WALL_PLANES = [
    (1,  2.95, np.array([ 0.0, -1.0])),  # top wall    — push in -Y
    (1, -2.95, np.array([ 0.0,  1.0])),  # bottom wall — push in +Y
    (0, -0.95, np.array([ 1.0,  0.0])),  # back wall   — push in +X
    (0,  5.95, np.array([-1.0,  0.0])),  # front wall  — push in -X
]


class SpiderEnv(gym.Env):
    """
    Gymnasium env: SpdrBot 4-leg spider walks forward (+X) and avoids obstacles.

    Episode structure
    -----------------
    Each episode lasts at most MAX_STEPS steps (truncated=True at the limit).
    It ends early (terminated=True) only if the robot falls over.
    Reaching x >= 5.0 m ends the episode with a large bonus.
    Hitting an obstacle is penalised but does NOT end the episode — the robot
    must learn to walk around things, not just avoid the first one.
    """
    metadata = {"render_modes": ["human"]}

    # Joints are type="continuous" in the URDF (no hard limits).
    # JOINT_LIMIT is enforced purely in software via np.clip.
    JOINT_LIMIT  = 1.2     # ±68.7° software limit
    MAX_FORCE    = 20.0    # N·m  — tune to match physical servo stall torque
    MAX_VELOCITY = 5.0     # rad/s — tune to match physical servo speed

    MAX_LIDAR = 5.0
    NUM_RAYS  = 360

    # Mirror module-level reward constants as class attributes so they can be
    # overridden per-instance if needed.
    BACKWARD_WEIGHT        = BACKWARD_WEIGHT
    FORWARD_WEIGHT         = FORWARD_WEIGHT
    SAFE_RADIUS            = SAFE_RADIUS
    OBSTACLE_WEIGHT        = OBSTACLE_WEIGHT
    EXP_K                  = EXP_K
    COLLISION_PENALTY      = COLLISION_PENALTY
    FALL_PENALTY           = FALL_PENALTY
    LATERAL_EVASION_WEIGHT = LATERAL_EVASION_WEIGHT
    FORWARD_DANGER_WEIGHT  = FORWARD_DANGER_WEIGHT
    APPROACH_WEIGHT        = APPROACH_WEIGHT
    CLEAR_PATH_WEIGHT      = CLEAR_PATH_WEIGHT
    ALIGNMENT_WEIGHT       = ALIGNMENT_WEIGHT
    YAW_REALIGN_WEIGHT     = YAW_REALIGN_WEIGHT
    WRONG_SPIN_WEIGHT      = WRONG_SPIN_WEIGHT
    LATERAL_DRIFT_WEIGHT   = LATERAL_DRIFT_WEIGHT
    YAW_CORRECTION_WEIGHT  = YAW_CORRECTION_WEIGHT
    YAW_RATE_CLIP          = YAW_RATE_CLIP
    YAW_RATE_DEADBAND      = YAW_RATE_DEADBAND
    WALL_SAFE_RADIUS       = WALL_SAFE_RADIUS
    FRONT_ARC_RAYS         = FRONT_ARC_RAYS
    FRONT_ARC_SAFE         = FRONT_ARC_SAFE
    FRONT_ARC_WEIGHT       = FRONT_ARC_WEIGHT
    GOAL_X                 = GOAL_X
    PROGRESS_WEIGHT        = PROGRESS_WEIGHT
    STEP_PENALTY           = STEP_PENALTY
    MIN_VX_TARGET          = MIN_VX_TARGET
    MIN_VX_WEIGHT          = MIN_VX_WEIGHT
    RESIDUAL_SCALE         = RESIDUAL_SCALE
    ACTION_SMOOTH_ALPHA    = ACTION_SMOOTH_ALPHA

    def __init__(self, render_mode: str = "human", urdf_path: str = URDF_PATH):
        super().__init__()
        self.render_mode = render_mode
        self.urdf_path   = urdf_path

        # Observation: 360 LiDAR + 12 joint angles + 3 IMU → 375-dim vector
        #   [0:360]   LiDAR distances normalised to [0, 1] (divided by MAX_LIDAR)
        #   [360:372] joint positions (rad), range ≈ [-π, π]
        #   [372]     roll_n     — normalised to [-1, 1] via /π
        #   [373]     pitch_n    — normalised to [-1, 1] via /π
        #   [374]     yaw_rate_n — normalised to [-1, 1] via clip/YAW_RATE_CLIP
        # cos(yaw)/sin(yaw) are NOT included — not available on the physical robot.
        # The agent infers heading context from LiDAR (robot-relative) and yaw_rate_n.
        self.observation_space = spaces.Box(
            low=np.concatenate([
                np.zeros(self.NUM_RAYS, dtype=np.float32),
                np.full(12,  -np.pi, dtype=np.float32),
                np.full(3,   -1.0,   dtype=np.float32),  # roll_n, pitch_n, yaw_rate_n
            ]),
            high=np.concatenate([
                np.full(self.NUM_RAYS, self.MAX_LIDAR, dtype=np.float32),
                np.full(12,   np.pi, dtype=np.float32),
                np.full(3,    1.0,   dtype=np.float32),  # roll_n, pitch_n, yaw_rate_n
            ]),
            dtype=np.float32,
        )
        # Action: residual joint-angle offsets added on top of CPG targets
        self.action_space = spaces.Box(
            low=-self.JOINT_LIMIT, high=self.JOINT_LIMIT,
            shape=(12,), dtype=np.float32,
        )

        # PyBullet
        self.client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.robot_id      = None
        self.joint_ids     = []
        self.wall_ids      = []   # arena boundary walls — proximity + collision penalties, terminate on contact
        self.obstacle_ids  = []   # red box obstacles — penalise and track proximity
        self.position      = [0.0, 0.0, 0.0]
        self._prev_x       = 0.0
        self.gait_phase    = 0.0
        self.joint_torques = np.zeros(12, dtype=np.float64)
        self.step_count         = 0    # counts steps within the current episode
        self._prev_action       = np.zeros(12, dtype=np.float32)  # for smoothness penalty
        self._smoothed_residual = np.zeros(12, dtype=np.float32)  # EMA state for action filter
        self._last_lidar        = np.full(self.NUM_RAYS, self.MAX_LIDAR, dtype=np.float32)  # cached for reward
        # Running reward normalisation — persists across episodes so the estimate
        # Reward normalisation removed — rewards are clipped to [-200, 50] in step().

        self.reset()

    # ── Robot loading ─────────────────────────────────────────────────────────

    def _load_robot(self):
        # SPAWN_Z: height so feet just contact z=0 in the standing pose.
        # TODO: calibrate once STAND_ANGLES are set.
        SPAWN_Z = 0.20
        robot_id = p.loadURDF(
            self.urdf_path,
            basePosition=[0, 0, SPAWN_Z],
            useFixedBase=False,
            flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL,
        )
        p.changeDynamics(robot_id, -1, linearDamping=0.3, angularDamping=0.3)

        # Build a name→index map for all joints in the URDF
        name_to_idx = {
            p.getJointInfo(robot_id, i)[1].decode(): i
            for i in range(p.getNumJoints(robot_id))
        }

        missing = [n for n in JOINT_NAMES if n not in name_to_idx]
        if missing:
            raise RuntimeError(
                f"URDF missing joints: {missing}\n"
                f"Available: {sorted(name_to_idx)}"
            )

        joint_ids = [name_to_idx[n] for n in JOINT_NAMES]

        # Enable torque sensors so we can measure energy usage in the reward
        for jid in joint_ids:
            p.enableJointForceTorqueSensor(robot_id, jid, enableSensor=True)

        # Move joints to the standing pose and hold them there
        for i, jid in enumerate(joint_ids):
            angle = float(np.clip(STAND_ANGLES[i], -self.JOINT_LIMIT, self.JOINT_LIMIT))
            p.resetJointState(robot_id, jid, targetValue=angle, targetVelocity=0.0)
            p.setJointMotorControl2(
                robot_id, jid,
                controlMode=p.POSITION_CONTROL,
                targetPosition=angle,
                force=self.MAX_FORCE,
                maxVelocity=self.MAX_VELOCITY,
            )

        return robot_id, joint_ids

    # ── Arena construction ────────────────────────────────────────────────────

    def _make_box(self, half_extents, position, orientation=None, color=None):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half_extents,
            rgbaColor=color or [0.6, 0.6, 0.6, 1],
        )
        kw = dict(baseMass=0,
                  baseCollisionShapeIndex=col,
                  baseVisualShapeIndex=vis,
                  basePosition=position)
        if orientation is not None:
            kw["baseOrientation"] = orientation
        return p.createMultiBody(**kw)

    def _create_wall(self, pos, length=5.0):
        # Wall running along the X axis
        return self._make_box([length / 2, 0.05, 0.5], pos)

    def _create_wall90(self, pos, length=5.0):
        # Wall running along the Y axis (rotated 90°)
        orn = p.getQuaternionFromEuler([0, 0, np.pi / 2])
        return self._make_box([length / 2, 0.05, 0.5], pos, orn)

    def _create_obstacle(self, pos):
        # Red box — the things the robot should steer around
        return self._make_box([0.2, 0.2, 0.5], pos, color=[1, 0, 0, 1])

    # ── Sensors ───────────────────────────────────────────────────────────────

    def _get_lidar(self) -> np.ndarray:
        """Cast 360 rays horizontally from the robot centre; return hit distances.
        Rays are robot-relative: ray 0 points straight ahead (robot +X), ray 90
        points to the left, matching the physical spinning lidar on the robot.
        """
        base_pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        _, _, yaw     = p.getEulerFromQuaternion(orn)
        # Offset all ray angles by current yaw so the pattern rotates with the robot
        angles    = np.linspace(0, 2 * np.pi, self.NUM_RAYS, endpoint=False) + yaw
        rays_from = [base_pos] * self.NUM_RAYS
        rays_to   = [
            [base_pos[0] + self.MAX_LIDAR * np.cos(a),
             base_pos[1] + self.MAX_LIDAR * np.sin(a),
             base_pos[2]]
            for a in angles
        ]
        results = p.rayTestBatch(rays_from, rays_to)
        # r[2] is the fraction of the ray that hit something (0–1); scale to metres
        return np.array([r[2] * self.MAX_LIDAR for r in results], dtype=np.float32)

    def _get_joint_positions(self) -> np.ndarray:
        return np.array(
            [p.getJointState(self.robot_id, jid)[0] for jid in self.joint_ids],
            dtype=np.float32,
        )

    def _get_imu(self) -> tuple[float, float, float]:
        """
        Return normalised IMU signals (roll_n, pitch_n, yaw_rate_n).

        Normalisation:
          roll_n      = roll    / π          — full-circle normalisation
          pitch_n     = pitch   / π          — full-circle normalisation
          yaw_rate_n  = clip(ω_z, ±YAW_RATE_CLIP) / YAW_RATE_CLIP

        cos(yaw)/sin(yaw) are NOT returned — unavailable on the physical robot.
        """
        _, orn = p.getBasePositionAndOrientation(self.robot_id)
        roll, pitch, _ = p.getEulerFromQuaternion(orn)
        _, ang_vel = p.getBaseVelocity(self.robot_id)
        yaw_rate = float(ang_vel[2])   # world-Z angular velocity ≈ yaw rate

        roll_n     = float(roll)   / math.pi
        pitch_n    = float(pitch)  / math.pi
        yaw_rate_n = float(np.clip(yaw_rate, -self.YAW_RATE_CLIP, self.YAW_RATE_CLIP)) / self.YAW_RATE_CLIP
        return roll_n, pitch_n, yaw_rate_n

    def _get_observation(self) -> np.ndarray:
        # 360 LiDAR + 12 joint positions + 3 IMU → 375-dim obs vector
        self._last_lidar = self._get_lidar()   # cache for _compute_reward
        roll_n, pitch_n, yaw_rate_n = self._get_imu()
        return np.concatenate([
            self._last_lidar,
            self._get_joint_positions(),
            np.array([roll_n, pitch_n, yaw_rate_n], dtype=np.float32),
        ])

    # ── Stability check ───────────────────────────────────────────────────────

    def _has_fallen(self) -> bool:
        """True if roll or pitch exceeds FALL_ANGLE_RAD (~60°)."""
        _, orn = p.getBasePositionAndOrientation(self.robot_id)
        roll, pitch, _ = p.getEulerFromQuaternion(orn)
        return abs(roll) > FALL_ANGLE_RAD or abs(pitch) > FALL_ANGLE_RAD

    # ── CPG gait ──────────────────────────────────────────────────────────────

    def _compute_gait_targets(self) -> np.ndarray:
        """
        Compute 12 joint targets using an asymmetric stance/swing CPG with
        per-side shoulder mirroring.

        Each leg cycle has two phases:

        STANCE (DUTY_CYCLE = 65% of cycle):
          - Foot on ground.  Shoulder sweeps dir * (+AMP → -AMP).
          - The direction multiplier (GAIT_SHOULDER_DIRS) ensures the foot
            always moves backward relative to the body regardless of which
            side the leg is on.  Friction converts this into forward thrust.
          - Elbow held at stand angle to keep the foot in ground contact.

        SWING (remaining 35%):
          - Foot lifted.  Shoulder sweeps dir * (-AMP → +AMP) to recover.
          - Bell-curve elbow lift (sin profile) — smooth liftoff and touchdown.

        Symmetry guarantee:
          GAIT_SHOULDER_DIRS = [+1, -1, -1, +1]
          Legs 0 and 3 are on the +Y side:  positive shoulder = foot forward.
          Legs 1 and 2 are on the -Y side:  positive shoulder = foot backward.
          Negating -Y legs' sweep inverts this, so all four feet sweep forward
          then backward identically.  Left/right lateral forces cancel and only
          the +X thrust remains → no net yaw torque from the CPG.
        """
        DUTY_CYCLE = 0.65
        TWO_PI = 2.0 * math.pi

        targets = np.empty(12, dtype=np.float64)

        for leg in range(4):
            phi_norm = ((self.gait_phase + GAIT_PHASE_OFFSETS[leg]) % TWO_PI) / TWO_PI
            direction = GAIT_SHOULDER_DIRS[leg]   # +1 for +Y side, -1 for -Y side

            s_idx = leg * 3
            e_idx = leg * 3 + 1
            w_idx = leg * 3 + 2

            if phi_norm < DUTY_CYCLE:
                # ── STANCE: power stroke ──────────────────────────────────────
                t = phi_norm / DUTY_CYCLE          # 0 → 1 during stance
                # dir*(+AMP → -AMP): foot starts forward, sweeps backward.
                # For -Y legs, direction=-1 flips this to (-AMP → +AMP) in raw
                # joint space, which is still foot-forward → foot-backward in
                # Cartesian space because the arm extends in -Y.
                shoulder = STAND_ANGLES[s_idx] + direction * GAIT_AMP_SHOULDER * (1.0 - 2.0 * t)
                elbow    = STAND_ANGLES[e_idx]   # foot stays low, maintains ground contact
                wrist    = STAND_ANGLES[w_idx]
            else:
                # ── SWING: recovery stroke ────────────────────────────────────
                t = (phi_norm - DUTY_CYCLE) / (1.0 - DUTY_CYCLE)  # 0 → 1 during swing
                # dir*(-AMP → +AMP): foot returns from backward to forward.
                shoulder = STAND_ANGLES[s_idx] + direction * GAIT_AMP_SHOULDER * (2.0 * t - 1.0)
                lift     = math.sin(t * math.pi)   # bell curve: 0 at liftoff/touchdown, 1 at peak
                elbow    = STAND_ANGLES[e_idx] - GAIT_AMP_LIFT * lift
                wrist    = STAND_ANGLES[w_idx]

            targets[s_idx] = shoulder
            targets[e_idx] = elbow
            targets[w_idx] = wrist

        np.clip(targets, -self.JOINT_LIMIT, self.JOINT_LIMIT, out=targets)
        self.gait_phase = (self.gait_phase + GAIT_PHASE_STEP * PHYSICS_SUBSTEPS) % TWO_PI
        return targets.astype(np.float32)

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self, robot_pos) -> tuple[float, dict]:
        """
        Compute reward and return (total, component_dict).

        Two priorities:
          1. Obstacle avoidance  (proximity + forward-danger + avoidance bonus)
          2. Maximise +X         (forward displacement + progress shaping)

        Proximity penalty uses a steep exponential: nearly flat until the robot
        is ~halfway into the safe zone, then spikes sharply as it closes in.
        Forward-danger gives a large additional penalty when the robot is heading
        directly toward an obstacle, making head-on crashes very costly.
        All positive terms are suppressed by (1 − max_closeness) so charging into
        a hazard is never profitable regardless of forward speed.
        """
        # ── X-direction progress ──────────────────────────────────────────────
        dx = robot_pos[0] - self._prev_x
        forward_reward = dx * self.FORWARD_WEIGHT

        # Potential-based shaping: dense reward for every metre closer to GOAL_X.
        prev_dist_to_goal = max(0.0, self.GOAL_X - self._prev_x)
        curr_dist_to_goal = max(0.0, self.GOAL_X - robot_pos[0])
        progress_reward   = (prev_dist_to_goal - curr_dist_to_goal) * self.PROGRESS_WEIGHT

        # ── IMU (heading vector for reward shaping — not in obs) ─────────────
        # yaw is available in simulation for reward computation even though
        # cos(yaw)/sin(yaw) are excluded from the observation.
        _, orn = p.getBasePositionAndOrientation(self.robot_id)
        _, _, yaw = p.getEulerFromQuaternion(orn)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)
        yaw_rate  = float(ang_vel[2])   # world-Z angular velocity ≈ yaw rate
        vel_2d    = np.array([float(lin_vel[0]), float(lin_vel[1])], dtype=np.float64)
        robot_forward = np.array([math.cos(yaw), math.sin(yaw)])

        # ── Obstacle / wall proximity ─────────────────────────────────────────
        # Exponential penalty: (exp(EXP_K * c) - 1) / (exp(EXP_K) - 1)
        # Nearly zero until c ≈ 0.5, then spikes to 1.0 at contact.
        _EXP_NORM       = math.exp(self.EXP_K) - 1.0
        _PER_HAZARD_CAP = 8.0   # scaled down to match reduced OBSTACLE_WEIGHT

        proximity_total      = 0.0
        avoidance_total      = 0.0
        forward_danger_total = 0.0
        approach_total       = 0.0
        max_closeness        = 0.0
        max_forward_threat   = 0.0   # max(closeness × forward_alignment) — 0 = clear ahead

        # Obstacle-only variants of the above — used to gate penalties that walls
        # should NOT suppress.  Gates driven by wall threats create exploits:
        #   • facing a side wall → wall forward_threat gates off min_vx_penalty
        #   • near a side wall  → wall closeness gates off lateral_drift_penalty
        # Separating these ensures only box-obstacle evasion unlocks those gates.
        obs_max_closeness      = 0.0
        obs_max_forward_threat = 0.0

        for obs_id in self.obstacle_ids:
            obs_pos, _ = p.getBasePositionAndOrientation(obs_id)
            vec_away = np.array(robot_pos[:2]) - np.array(obs_pos[:2])
            dist = float(np.linalg.norm(vec_away))
            if dist < self.SAFE_RADIUS:
                closeness = (self.SAFE_RADIUS - dist) / self.SAFE_RADIUS  # [0, 1]
                # Linear base gives an early-warning gradient from the safe-radius edge;
                # exponential term spikes sharply in the final approach.
                exp_term  = (math.exp(self.EXP_K * closeness) - 1.0) / _EXP_NORM
                proximity_total += min((0.4 * closeness + 0.6 * exp_term) * self.OBSTACLE_WEIGHT,
                                       _PER_HAZARD_CAP)
                repulsion_dir = vec_away / dist if dist > 1e-6 else np.array([1.0, 0.0])
                away_speed    = float(np.clip(np.dot(vel_2d, repulsion_dir), 0.0, 1.0))
                lateral_speed = abs(float(np.cross(repulsion_dir, vel_2d)))  # |vy| when obstacle is ahead
                avoidance_total += closeness * (away_speed + lateral_speed) * self.LATERAL_EVASION_WEIGHT
                max_closeness      = max(max_closeness, closeness)
                obs_max_closeness  = max(obs_max_closeness, closeness)
                # Heading-based: penalise facing toward the obstacle.
                # Linear in closeness — only generates meaningful turning force when
                # the obstacle is genuinely close, not from 3m away.
                forward_alignment = max(0.0, float(np.dot(robot_forward, -repulsion_dir)))
                forward_danger_total += closeness * forward_alignment * self.FORWARD_DANGER_WEIGHT
                max_forward_threat     = max(max_forward_threat, closeness * forward_alignment)
                obs_max_forward_threat = max(obs_max_forward_threat, closeness * forward_alignment)
                # Velocity-based: penalise actually moving toward the obstacle.
                # Linear in closeness — distant obstacles don't generate strong
                # turning forces that sweep the robot into other obstacle zones.
                approach_speed = max(0.0, float(np.dot(vel_2d, -repulsion_dir)))
                approach_total += closeness * approach_speed * self.APPROACH_WEIGHT

        robot_xy = np.array(robot_pos[:2], dtype=np.float64)
        for axis, face_pos, repulsion_dir in _WALL_PLANES:
            # Perpendicular distance from robot to wall inner face.
            # repulsion_dir[axis] < 0 → wall is on the high side (top/front wall)
            # repulsion_dir[axis] > 0 → wall is on the low side (bottom/back wall)
            if repulsion_dir[axis] < 0:
                dist = float(face_pos - robot_xy[axis])
            else:
                dist = float(robot_xy[axis] - face_pos)
            dist = max(0.0, dist)

            if dist < self.WALL_SAFE_RADIUS:
                closeness = (self.WALL_SAFE_RADIUS - dist) / self.WALL_SAFE_RADIUS
                exp_term  = (math.exp(self.EXP_K * closeness) - 1.0) / _EXP_NORM
                proximity_total += min((0.4 * closeness + 0.6 * exp_term) * self.OBSTACLE_WEIGHT,
                                       _PER_HAZARD_CAP)
                max_closeness = max(max_closeness, closeness)
                # repulsion_dir is exact and axis-aligned — no normal uncertainty.
                away_speed    = float(np.clip(np.dot(vel_2d, repulsion_dir), 0.0, 1.0))
                avoidance_total += closeness * away_speed * self.LATERAL_EVASION_WEIGHT
                forward_alignment = max(0.0, float(np.dot(robot_forward, -repulsion_dir)))
                forward_danger_total += closeness * forward_alignment * self.FORWARD_DANGER_WEIGHT
                max_forward_threat   = max(max_forward_threat, closeness * forward_alignment)
                approach_speed = max(0.0, float(np.dot(vel_2d, -repulsion_dir)))
                approach_total += closeness * approach_speed * self.APPROACH_WEIGHT

        proximity_penalty      = min(proximity_total,       20.0)
        avoidance_bonus        = min(avoidance_total,        12.0)
        forward_danger_penalty = min(forward_danger_total,  10.0)
        approach_penalty       = min(approach_total,        15.0)
        # Clear-path bonus: reward +X velocity scaled by how unobstructed ahead is
        # AND how well the robot is facing +X.  heading_alignment = cos(yaw): 1.0
        # when facing dead ahead, 0.0 at 90°, negative behind (clamped to 0).
        # This means sprinting in world +X while pointing sideways earns no bonus,
        # creating a reward signal to re-align heading without needing yaw in obs.
        vx = float(lin_vel[0])
        # Clear-path bonus: reward +X velocity when nothing is directly ahead AND
        # the robot is facing forward.  heading_alignment = cos(yaw) is 1.0 when
        # facing +X, 0 at 90°, so driving in world-X while pointing sideways earns
        # nothing — creates a gradient toward maintaining yaw ≈ 0.
        heading_alignment = max(0.0, math.cos(yaw))
        clear_path_bonus = max(0.0, vx) * heading_alignment * (1.0 - max_forward_threat) * self.CLEAR_PATH_WEIGHT

        vy = float(lin_vel[1])
        # Gate lateral drift only by *obstacle* closeness — not wall closeness.
        # Gating by wall closeness was an exploit: near side wall → lateral motion
        # free → robot could slide into wall unchecked.  Box-obstacle evasion still
        # legitimately silences the penalty (obs_max_closeness high when dodging).
        lateral_drift_penalty  = abs(vy) * self.LATERAL_DRIFT_WEIGHT * (1.0 - obs_max_closeness)

        # Penalise heading error using yaw (available in sim for reward shaping).
        # Agent can't observe yaw directly, but this still guides CPG-based learning
        # because the gait CPG tends to produce consistent heading; the penalty
        # discourages drift without requiring the policy to explicitly represent yaw.
        yaw_error              = 1.0 - math.cos(yaw)
        yaw_correction_penalty = yaw_error * self.YAW_CORRECTION_WEIGHT

        # Front-arc LiDAR penalty — the primary learnable replacement for yaw-based
        # forward_danger.  Uses robot-relative LiDAR (ray 0 = straight ahead) so the
        # agent can directly observe and respond to this signal without knowing yaw.
        # Arc: rays [0 : FRONT_ARC_RAYS] and [NUM_RAYS-FRONT_ARC_RAYS : NUM_RAYS]
        # (wrapping around ray 0 to cover ±FRONT_ARC_RAYS° of the forward sector).
        # Each ray contributes a closeness in [0, 1]: 0 at FRONT_ARC_SAFE distance,
        # 1 at contact.  The sum is normalised by arc size so the weight is stable
        # regardless of FRONT_ARC_RAYS.
        n_arc = self.FRONT_ARC_RAYS
        front_rays = np.concatenate([
            self._last_lidar[:n_arc],
            self._last_lidar[-n_arc:],
        ])
        front_closeness = np.maximum(0.0, 1.0 - front_rays / self.FRONT_ARC_SAFE)
        front_arc_penalty = float(np.mean(front_closeness)) * self.FRONT_ARC_WEIGHT

        # Penalise moving backward in world-X — always active, no clearance gate.
        # This makes the U-turn strategy (turn 180° to dodge, then reverse) clearly
        # worse than sidestepping, since every backward step has a direct cost.
        backward_penalty = max(0.0, -vx) * self.BACKWARD_WEIGHT

        # Minimum forward velocity penalty — breaks "drift sideways = safe" local optimum.
        # Gated by (1 − max_forward_threat): silent when an obstacle is directly ahead
        # so the robot is free to evade first and resume speed after.
        # Gate min_vx only by *obstacle* forward threat — not wall forward threat.
        # Previously gated by combined max_forward_threat, which let the robot face
        # a side wall to suppress the penalty with zero risk of actually colliding
        # (backward_penalty fires if it moves toward the wall in world-X).
        min_vx_penalty = (max(0.0, self.MIN_VX_TARGET - vx)
                          * (1.0 - obs_max_forward_threat)
                          * self.MIN_VX_WEIGHT)

        # Yaw realign reward — reward spinning *toward* +X when misaligned.
        # sin(yaw) > 0 → facing left  → correct move is negative yaw_rate (clockwise)
        # sin(yaw) < 0 → facing right → correct move is positive yaw_rate (counter-clockwise)
        # In both cases, the correct rotation has the opposite sign to sin(yaw), so the
        # product -sin(yaw)*yaw_rate is positive when the robot is actively correcting.
        # Gated by (1 − obs_max_closeness) so it doesn't fight active obstacle evasion.
        yaw_realign_reward = (max(0.0, -math.sin(yaw) * yaw_rate)
                              * (1.0 - obs_max_closeness)
                              * self.YAW_REALIGN_WEIGHT)

        # Wrong spin penalty — penalise spinning *away* from +X.
        # sin(yaw)*yaw_rate > 0 means the robot is rotating further from +X heading.
        # Gate is halved: (1 − 0.5×obs_max_forward_threat) instead of full suppression.
        # Full suppression let the robot spin 90° freely during obstacle evasion and
        # then coast at the wrong heading once the obstacle cleared.  A 50% floor means
        # even acute evasion still carries a wrong-spin cost, capping dodge angle.
        wrong_spin_penalty = (max(0.0, math.sin(yaw) * yaw_rate)
                              * (1.0 - 0.5 * obs_max_forward_threat)
                              * self.WRONG_SPIN_WEIGHT)

        # Suppress positive terms only when the robot is heading INTO a hazard.
        # Using max_forward_threat (closeness × forward_alignment) instead of
        # max_closeness means the robot still earns forward/progress reward while
        # travelling alongside an obstacle — only head-on charges are penalised.
        suppression     = 1.0 - max_forward_threat
        forward_reward  *= suppression
        progress_reward *= suppression

        # Terminal penalties (applied here for logging; collision applied in step()).
        collision_penalty = 0.0
        fall_penalty = self.FALL_PENALTY if self._has_fallen() else 0.0

        step_penalty = self.STEP_PENALTY

        total = (
            forward_reward
            + progress_reward
            + clear_path_bonus
            + yaw_realign_reward
            - proximity_penalty
            - forward_danger_penalty
            - approach_penalty
            - front_arc_penalty
            + avoidance_bonus
            + fall_penalty            # already negative
            - lateral_drift_penalty
            - yaw_correction_penalty
            - wrong_spin_penalty
            - backward_penalty
            - min_vx_penalty
            - step_penalty
        )

        components = {
            "forward":       round(forward_reward,            4),
            "progress":      round(progress_reward,           4),
            "clear_path":    round(clear_path_bonus,          4),
            "yaw_realign":   round(yaw_realign_reward,        4),
            "proximity":     round(-proximity_penalty,        4),
            "fwd_danger":    round(-forward_danger_penalty,   4),
            "approach":      round(-approach_penalty,         4),
            "front_arc":     round(-front_arc_penalty,        4),
            "avoidance":     round(avoidance_bonus,           4),
            "lat_drift":     round(-lateral_drift_penalty,    4),
            "yaw_correct":   round(-yaw_correction_penalty,   4),
            "wrong_spin":    round(-wrong_spin_penalty,       4),
            "backward":      round(-backward_penalty,         4),
            "min_vx":        round(-min_vx_penalty,           4),
            "step":          round(-step_penalty,             4),
            "collision":     round(collision_penalty,         4),
            "fall":          round(fall_penalty,              4),
            "total":         round(total,                     4),
        }
        return float(total), components

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")

        self.robot_id, self.joint_ids = self._load_robot()

        # Let gravity settle the robot into its standing pose (~0.5 s at 240 Hz)
        for _ in range(120):
            p.stepSimulation()

        self.gait_phase         = 0.0
        self.joint_torques      = np.zeros(12, dtype=np.float64)
        self.step_count         = 0
        self._prev_action       = np.zeros(12, dtype=np.float32)
        self._smoothed_residual = np.zeros(12, dtype=np.float32)  # reset filter memory each episode

        # Arena walls — stored separately so they don't affect the reward signal
        self.wall_ids = [
            self._create_wall   ([ 2.5,  3.0, 0.5], length=7),  # top wall
            self._create_wall90 ([-1.0,  0.0, 0.5], length=6),  # back wall
            self._create_wall   ([ 2.5, -3.0, 0.5], length=7),  # bottom wall
            self._create_wall90 ([ 6.0,  0.0, 0.5], length=6),  # front wall (goal end)
        ]

        # Red obstacles — placed along the robot's forward path.
        # The path (X = X_OBS_START → X_OBS_END) is divided into equal segments;
        # one obstacle is placed per segment at a random X within the segment and
        # a random Y within ±Y_OBS_SPREAD of the centre line.  This guarantees
        # obstacles are spread ahead of the agent (not hiding at the arena edges)
        # while still requiring navigation in the lateral direction.
        N_OBS        = 3
        X_OBS_START  = 0.8   # first obstacle at least 0.8 m ahead of spawn
        X_OBS_END    = 4.5   # last obstacle well before the goal wall
        Y_OBS_SPREAD = 1.3   # half-width of obstacle corridor (arena half-width ≈ 2.7 m)
        seg_len = (X_OBS_END - X_OBS_START) / N_OBS

        self.obstacle_ids = []
        for i in range(N_OBS):
            seg_x = X_OBS_START + i * seg_len
            ox = random.uniform(seg_x + 0.1, seg_x + seg_len - 0.1)
            oy = random.uniform(-Y_OBS_SPREAD, Y_OBS_SPREAD)
            self.obstacle_ids.append(self._create_obstacle([ox, oy, 0.5]))

        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        self.position = list(pos)
        self._prev_x  = pos[0]
        self._passed_obstacles = set()  # tracks which obstacles the robot has navigated past

        return self._get_observation(), {}

    def step(self, action: np.ndarray):
        # 1. Get CPG baseline joint targets for this phase
        gait_targets = self._compute_gait_targets()

        # 2. Combine RL residual with CPG baseline.
        #
        #    Two-stage protection against RL destabilising the gait:
        #
        #    a) EMA low-pass filter — update the running smoothed residual with
        #       only ACTION_SMOOTH_ALPHA of the new action each step.  This prevents
        #       PPO from flipping joint targets in a single 33 ms step; sustained
        #       corrections bleed through fully over ~3 steps, but high-frequency
        #       impulses are heavily attenuated.
        #
        #    b) RESIDUAL_SCALE = 0.15 — caps the smoothed residual at ±0.18 rad,
        #       keeping it ≤33% of the CPG shoulder swing amplitude (±0.55 rad).
        #       CPG remains the dominant signal at all times.
        #
        #    The smoothness *penalty* (SMOOTH_WEIGHT) is computed on the raw action
        #    delta — this trains the policy itself to prefer gradual changes, not just
        #    filters them away silently.
        action = np.asarray(action, dtype=np.float32)
        self._smoothed_residual = (
            self.ACTION_SMOOTH_ALPHA * action
            + (1.0 - self.ACTION_SMOOTH_ALPHA) * self._smoothed_residual
        )
        residual = self._smoothed_residual * self.RESIDUAL_SCALE  # max ±0.18 rad
        joint_targets = np.clip(gait_targets + residual,
                                -self.JOINT_LIMIT, self.JOINT_LIMIT).astype(np.float32)

        # 3. Send targets to all 12 actuators
        #    PyBullet holds these targets across all substeps automatically.
        for i, jid in enumerate(self.joint_ids):
            p.setJointMotorControl2(
                self.robot_id, jid,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(joint_targets[i]),
                force=self.MAX_FORCE,
                maxVelocity=self.MAX_VELOCITY,
            )

        # 4. Advance physics PHYSICS_SUBSTEPS times.
        #    This makes each agent decision cover ~33 ms of real simulation
        #    instead of just 4 ms, so the robot moves meaningful distance
        #    per step and MAX_STEPS episodes last ~33 seconds.
        for _ in range(PHYSICS_SUBSTEPS):
            p.stepSimulation()
        self.step_count += 1

        # 5. Read motor torques after the last substep (used in energy penalty)
        self.joint_torques = np.array(
            [p.getJointState(self.robot_id, jid)[3] for jid in self.joint_ids],
            dtype=np.float64,
        )

        # 6. Gather new position and observation
        new_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        obs = self._get_observation()

        # 7. Compute reward
        reward, components = self._compute_reward(new_pos)

        # 8. Check termination / truncation
        terminated = False
        truncated  = False

        collided_obstacle = any(
            p.getContactPoints(self.robot_id, obs_id)
            for obs_id in self.obstacle_ids
        )
        collided_wall = any(
            p.getContactPoints(self.robot_id, wall_id)
            for wall_id in self.wall_ids
        )

        end_reason    = "running"
        terminal_bonus = 0.0   # applied AFTER clip so large bonuses are not squashed

        if new_pos[0] >= self.GOAL_X:
            # +500 goal bonus added after clip — if added before, np.clip would
            # squash it to the same +50 ceiling as any ordinary good step, giving
            # the agent almost no incentive to actually reach the goal.
            terminal_bonus = 500.0
            terminated     = True
            end_reason     = "goal"
        elif collided_obstacle or collided_wall:
            if collided_wall:
                # Wall collisions are always full penalty — walls are hard
                # boundaries and any contact means the U-turn strategy worked,
                # which must never be a viable option.
                scaled_penalty = self.COLLISION_PENALTY
            else:
                # Obstacle collisions scale by how head-on the crash was.
                # Glancing clips (approach≈0) cost 30%; head-on (approach≈15)
                # cost 100%.  Floor raised from 10% to 30% so grazing an obstacle
                # is never cheap enough to be a viable strategy.
                approach_factor = min(abs(components["approach"]) / 15.0, 1.0)
                scaled_penalty  = self.COLLISION_PENALTY * (0.3 + 0.7 * approach_factor)
            terminal_bonus          = scaled_penalty
            components["collision"] = scaled_penalty
            terminated              = True
            end_reason              = "collision"
        elif self._has_fallen():
            # Robot tipped over — end episode
            terminated = True
            end_reason = "fall"
        elif self.step_count >= MAX_STEPS:
            # Ran out of time — truncate (not a failure, just a timeout)
            truncated  = True
            end_reason = "timeout"

        # 9. Update bookkeeping
        self.position    = list(new_pos)
        self._prev_x     = new_pos[0]
        self._prev_action = action.copy()

        # One-time bonus for navigating past each obstacle.
        # Fires the first time the robot's X exceeds the obstacle's X by 0.5 m,
        # giving an explicit "you went around it" signal without ending the episode.
        passing_bonus = 0.0
        for i, obs_id in enumerate(self.obstacle_ids):
            if i not in self._passed_obstacles:
                obs_pos, _ = p.getBasePositionAndOrientation(obs_id)
                if new_pos[0] > obs_pos[0] + 0.5:
                    self._passed_obstacles.add(i)
                    passing_bonus += 30.0

        # Clip the continuous per-step reward, then add terminal bonuses.
        # Terminal bonuses must survive the clip: +500 goal and -10000 collision
        # are intentionally larger than any per-step reward and must remain
        # distinguishable from ordinary good/bad steps.
        reward = float(np.clip(reward, -200.0, 50.0)) + terminal_bonus + passing_bonus

        return obs, float(reward), terminated, truncated, {
            "reward_components": components,
            "end_reason": end_reason,
        }

    def render(self):
        pass  # GUI is opened in __init__ when render_mode == "human"

    def close(self):
        p.disconnect(self.client)
