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
Obs    (372,) float32 — 360 LiDAR distances (m)  +  12 joint positions (rad)

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
SAFE_RADIUS          = 2.5    # m — warning zone for avoidance signal
OBSTACLE_WEIGHT      = 5.0    # mild background proximity signal — approach+danger carry the main weight
# Exponential steepness: penalty = (exp(EXP_K * closeness) - 1) / (exp(EXP_K) - 1)
# EXP_K=6 → nearly flat until ~0.5, then spikes sharply in the final approach.
EXP_K                = 6.0
# COLLISION_PENALTY must dominate the maximum possible accumulated episode reward.
COLLISION_PENALTY    = -5000.0
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
CLEAR_PATH_WEIGHT      = 4.0

# Observation normalisation constants (not reward weights — kept for obs building).
YAW_RATE_CLIP      = 5.0   # rad/s ceiling for obs normalisation
YAW_RATE_DEADBAND  = 0.40  # rad/s deadband (obs only)

# Arena wall proximity zone.
WALL_SAFE_RADIUS   = 2.0   # m — wall proximity warning zone

# Potential-based progress shaping toward the goal.
# Grounded in Ng et al. (1999): Φ(s) = GOAL_X − x, so the per-step shaping reward
# equals (GOAL_X − x_prev) − (GOAL_X − x_curr) = dx (only when x < GOAL_X).
# Equivalent to adding a second forward-weight that stops contributing once the
# goal is reached, giving a strong dense signal for every centimetre of progress.
GOAL_X           = 5.0     # m — target X coordinate (must match step() check)
PROGRESS_WEIGHT  = 10.0    # shaping reward per metre closer to goal

# ── Residual action scale and smoothing ───────────────────────────────────────
# RESIDUAL_SCALE multiplies the agent's raw action before it is added to the
# CPG target.  It must be small enough that the CPG dominates.
#
#   CPG shoulder amplitude : ±0.55 rad  (range = 1.10 rad)
#   Max residual (raw)     : ±(1.2 × 0.15) = ±0.18 rad   ≈ 33% of CPG swing
#
# This ensures the CPG provides ≥67% of the control signal at all times, so
# random RL exploration cannot flip joint directions or override the gait.
RESIDUAL_SCALE = 0.15

# EMA weight applied to the residual each step before it reaches the joints.
# Acts as a first-order IIR low-pass filter: only ACTION_SMOOTH_ALPHA of the
# new action bleeds through per step; the rest carries over from the previous.
#   α = 0.3  →  ~3-step time constant  →  ~100 ms lag at 33 ms/step
# Effect: PPO cannot inject a full-magnitude impulse in a single step.
# The policy can still learn sustained corrections — just not high-frequency noise.
ACTION_SMOOTH_ALPHA = 0.3


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
    YAW_RATE_CLIP          = YAW_RATE_CLIP
    YAW_RATE_DEADBAND      = YAW_RATE_DEADBAND
    WALL_SAFE_RADIUS       = WALL_SAFE_RADIUS
    GOAL_X                 = GOAL_X
    PROGRESS_WEIGHT        = PROGRESS_WEIGHT
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
        #
        # Absolute yaw is excluded: the real MPU-6050 IMU has no magnetometer
        # so yaw drifts over time and cannot be trusted.  The policy is designed
        # to work without it — roll/pitch provide tilt feedback, yaw_rate provides
        # spinning feedback, and the CPG already biases motion in the local
        # forward direction.
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
        # Running reward normalisation — persists across episodes so the estimate
        # stabilises over time.  Divides by running std to keep gradients stable.
        self._rn_count = 0
        self._rn_mean  = 0.0
        self._rn_var   = 1.0

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
        """Cast 360 rays horizontally from the robot centre; return hit distances."""
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        angles    = np.linspace(0, 2 * np.pi, self.NUM_RAYS, endpoint=False)
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
        Return normalised IMU signals (roll_n, pitch_n, yaw_rate_n), each in [-1, 1].

        Normalisation:
          roll_n      = roll    / π          — full-circle normalisation
          pitch_n     = pitch   / π          — full-circle normalisation
          yaw_rate_n  = clip(ω_z, ±YAW_RATE_CLIP) / YAW_RATE_CLIP

        Why yaw_rate and not just yaw:
          Yaw (absolute heading) tells the agent *where* it is pointing.
          Yaw_rate tells the agent *how fast it is spinning right now*.
          A robot that has already spun 90° and is still accelerating needs
          both signals to correct itself; rate alone enables reactive damping.
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
        roll_n, pitch_n, yaw_rate_n = self._get_imu()
        return np.concatenate([
            self._get_lidar(),
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

        # ── IMU (needed for heading vector and obs) ───────────────────────────
        _, orn = p.getBasePositionAndOrientation(self.robot_id)
        _, _, yaw = p.getEulerFromQuaternion(orn)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)
        vel_2d   = np.array([float(lin_vel[0]), float(lin_vel[1])], dtype=np.float64)
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
                avoidance_total += closeness * away_speed * self.LATERAL_EVASION_WEIGHT
                max_closeness    = max(max_closeness, closeness)
                # Heading-based: penalise facing toward the obstacle.
                forward_alignment = max(0.0, float(np.dot(robot_forward, -repulsion_dir)))
                forward_danger_total += closeness * forward_alignment * self.FORWARD_DANGER_WEIGHT
                max_forward_threat   = max(max_forward_threat, closeness * forward_alignment)
                # Velocity-based: penalise actually moving toward the obstacle.
                approach_speed = max(0.0, float(np.dot(vel_2d, -repulsion_dir)))
                approach_total += closeness * approach_speed * self.APPROACH_WEIGHT

        for wall_id in self.wall_ids:
            pts = p.getClosestPoints(self.robot_id, wall_id, self.WALL_SAFE_RADIUS)
            if pts:
                best = min(pts, key=lambda pt: pt[8])
                dist = max(0.0, float(best[8]))
                closeness = float(np.clip(
                    (self.WALL_SAFE_RADIUS - dist) / self.WALL_SAFE_RADIUS, 0.0, 1.0))
                exp_term  = (math.exp(self.EXP_K * closeness) - 1.0) / _EXP_NORM
                proximity_total += min((0.4 * closeness + 0.6 * exp_term) * self.OBSTACLE_WEIGHT,
                                       _PER_HAZARD_CAP)
                max_closeness = max(max_closeness, closeness)
                normal_xy = np.array(best[7][:2], dtype=np.float64)
                norm_len  = np.linalg.norm(normal_xy)
                if norm_len > 1e-6:
                    repulsion_dir = normal_xy / norm_len
                    away_speed    = float(np.clip(np.dot(vel_2d, repulsion_dir), 0.0, 1.0))
                    avoidance_total += closeness * away_speed * self.LATERAL_EVASION_WEIGHT
                    forward_alignment = max(0.0, float(np.dot(robot_forward, -repulsion_dir)))
                    forward_danger_total += closeness * forward_alignment * self.FORWARD_DANGER_WEIGHT
                    max_forward_threat   = max(max_forward_threat, closeness * forward_alignment)
                    approach_speed = max(0.0, float(np.dot(vel_2d, -repulsion_dir)))
                    approach_total += closeness * approach_speed * self.APPROACH_WEIGHT

        proximity_penalty      = min(proximity_total,       20.0)
        avoidance_bonus        = min(avoidance_total,        6.0)
        forward_danger_penalty = min(forward_danger_total,  10.0)
        approach_penalty       = min(approach_total,        15.0)
        # Clear-path bonus: reward +X velocity scaled by how unobstructed ahead is.
        # max_forward_threat=0 → full bonus; =1 (dead ahead) → zero bonus.
        vx = float(lin_vel[0])
        clear_path_bonus = max(0.0, vx) * (1.0 - max_forward_threat) * self.CLEAR_PATH_WEIGHT

        # Suppress all positive terms as the robot closes in on a hazard.
        suppression     = 1.0 - max_closeness
        forward_reward  *= suppression
        progress_reward *= suppression

        # Terminal penalties (applied here for logging; collision applied in step()).
        collision_penalty = 0.0
        fall_penalty = self.FALL_PENALTY if self._has_fallen() else 0.0

        total = (
            forward_reward
            + progress_reward
            + clear_path_bonus
            - proximity_penalty
            - forward_danger_penalty
            - approach_penalty
            + avoidance_bonus
            + fall_penalty            # already negative
        )

        components = {
            "forward":    round(forward_reward,           4),
            "progress":   round(progress_reward,          4),
            "clear_path": round(clear_path_bonus,         4),
            "proximity":  round(-proximity_penalty,       4),
            "fwd_danger": round(-forward_danger_penalty,  4),
            "approach":   round(-approach_penalty,        4),
            "avoidance":  round(avoidance_bonus,          4),
            "collision":  round(collision_penalty,        4),
            "fall":       round(fall_penalty,             4),
            "total":      round(total,                    4),
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
        N_OBS        = 4
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

        end_reason = "running"
        if new_pos[0] >= self.GOAL_X:
            # Robot reached the goal.  Bonus must be large enough to clearly
            # distinguish "reached goal" from "timed out without collision".
            # A 2000-step clean episode accumulates ~+2000–4000, so +5 (old value)
            # was < 0.2% of that — no RL incentive to reach the goal at all.
            # +500 makes success unambiguously the best outcome.
            reward    += 500.0
            terminated = True
            end_reason = "goal"
        elif collided_obstacle or collided_wall:
            # Hit an obstacle or wall — apply the terminal collision penalty once
            # and end the episode.  COLLISION_PENALTY is NOT applied per-step
            # inside _compute_reward; this single application is the full signal.
            reward    += self.COLLISION_PENALTY
            # Reflect the actual terminal penalty in the components dict so that
            # the breakdown is accurate when debugging.  Without this fix,
            # components["collision"] always showed 0.0 even on crash steps.
            components["collision"] = self.COLLISION_PENALTY
            terminated = True
            end_reason = "collision"
        elif self._has_fallen():
            # Robot tipped over — end episode
            terminated = True
            end_reason = "fall"
        elif self.step_count >= MAX_STEPS:
            # Ran out of time — truncate (not a failure, just a timeout)
            truncated = True
            end_reason = "timeout"

        # 9. Update bookkeeping
        self.position    = list(new_pos)
        self._prev_x     = new_pos[0]
        self._prev_action = action.copy()

        # Normalise reward by running std (Welford online update).
        self._rn_count += 1
        delta          = reward - self._rn_mean
        self._rn_mean += delta / self._rn_count
        self._rn_var  += (delta * (reward - self._rn_mean) - self._rn_var) / self._rn_count
        reward         = reward / (self._rn_var ** 0.5 + 1e-8)

        return obs, float(reward), terminated, truncated, {
            "reward_components": components,
            "end_reason": end_reason,
        }

    def render(self):
        pass  # GUI is opened in __init__ when render_mode == "human"

    def close(self):
        p.disconnect(self.client)
